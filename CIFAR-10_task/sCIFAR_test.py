import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from mamba_ssm import Mamba

class TCNLayer(nn.Module):
    """A single Dilated 1D Convolutional block with residual connection."""
    def __init__(self, d_model, kernel_size, dilation):
        super().__init__()
        # Calculate padding to ensure input and output sequence lengths match perfectly
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 1) # Permute for Conv1d to (Batch, d_model, Length)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x + residual


class PureMamba(nn.Module):
    """Pure Mamba with N layers."""
    def __init__(self, d_model=128, num_layers=4, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            }) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.layers:
            residual = x
            x = block['norm'](x)
            x = block['mamba'](x)
            x = x + residual
            
        #x = x.mean(dim=1) (Earlier method)
        x = x[:, -1, :]
        return self.classifier(x)


class PureTCN(nn.Module):
    """Pure TCN. M must be >= 8 to cover a 1024 sequence length."""
    def __init__(self, d_model=128, num_layers=8, kernel_size=3, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        
        # Exponential dilation: 1, 2, 4, 8, 16...
        self.layers = nn.ModuleList([
            TCNLayer(d_model=d_model, kernel_size=kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            
        x = x[:, -1, :]
        return self.classifier(x)


class HybridParallelBlock(nn.Module):
    """A single block containing parallel Mamba and TCN branches."""
    def __init__(self, d_model, kernel_size, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        padding = (kernel_size - 1) * dilation // 2
        self.tcn = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # Mamba branch (Global)
        mamba_out = self.mamba(x_norm)
        
        # TCN branch (Local/Regional based on dilation)
        tcn_in = x_norm.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        tcn_out = tcn_out.permute(0, 2, 1)
        tcn_out = self.activation(tcn_out)
        
        # Element-wise addition fusion
        return mamba_out + tcn_out + residual


class HybridParallelMambaTCN(nn.Module):
    """Stacked Parallel Hybrid blocks."""
    def __init__(self, d_model=128, num_layers=4, kernel_size=3, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        
        self.layers = nn.ModuleList([
            HybridParallelBlock(d_model=d_model, kernel_size=kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            
        x = x[:, -1, :]
        return self.classifier(x)


def count_parameters(model):
    """Calculates the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_matching_d_model(TargetModelClass, target_params, base_d_model, tolerance=0.01, **kwargs):
    """
    Iteratively adjusts d_model to match a target parameter count.
    Uses a simple linear search radiating outward from the base_d_model.
    """
    current_d_model = base_d_model
    step = 1
    direction = 0 # 1 for increasing, -1 for decreasing

    while True:
        # Instantiate the model with the current test width
        test_model = TargetModelClass(d_model=current_d_model, **kwargs)
        current_params = count_parameters(test_model)
        
        error = (current_params - target_params) / target_params
        
        # Check if within 1% tolerance
        if abs(error) <= tolerance:
            return current_d_model, current_params
        
        # Determine search direction on the first pass
        if direction == 0:
            direction = -1 if current_params > target_params else 1
            
        # If we overshoot and change direction, we've found the closest integer match
        if (direction == -1 and current_params < target_params) or \
           (direction == 1 and current_params > target_params):
            return current_d_model, current_params
            
        current_d_model += direction
        
        # Failsafe to prevent zero or negative dimensions
        if current_d_model <= 0:
            raise ValueError("Cannot match parameters: d_model reached 0.")

def get_scifar_dataloaders(batch_size=128, num_workers=4):
    """
    Loads CIFAR-10 and flattens the 32x32 spatial grid into a 1D sequence.
    Input shape changes from (3, 32, 32) to (1024, 3).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(3, -1).permute(1, 0))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def get_noisy_scifar_dataloaders(batch_size=128, num_workers=4, noise_length=3000):
    """
    Loads CIFAR-10, flattens the 32x32 grid, and appends Gaussian noise.
    Input shape changes from (3, 32, 32) to (1024 + noise_length, 3).
    """
    def append_noise(x):
        # Flatten (3, 32, 32) -> (3, 1024), then transpose to (1024, 3)
        seq = x.view(3, -1).permute(1, 0)
        
        # Generate Gaussian noise matching the feature dimension
        # Shape: (3000, 3)
        noise = torch.randn(noise_length, 3)
        
        # Concatenate along the sequence length dimension
        # Final Shape: (4024, 3)
        return torch.cat([seq, noise], dim=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Standard CIFAR-10 normalization
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(append_noise)
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Note: You may need to reduce batch_size if 4024-length sequences cause OOM errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def setup_ssm_optimizer(model, base_lr=1e-3, ssm_lr_ratio=0.1):
    """
    Isolates Mamba's continuous-time state parameters to apply a lower 
    learning rate and zero weight decay, preventing numerical explosion.
    """
    standard_params = []
    ssm_state_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mamba" in name and any(key in name for key in ['dt_proj', 'A_log', 'D']):
            ssm_state_params.append(param)
        else:
            standard_params.append(param)

    optimizer = optim.AdamW([
        {'params': standard_params, 'lr': base_lr, 'weight_decay': 0.05},
        {'params': ssm_state_params, 'lr': base_lr * ssm_lr_ratio, 'weight_decay': 0.0}
    ])
    return optimizer

def train_and_evaluate(model, name, train_loader, test_loader, epochs=50, device='cuda'):
    """
    Standard training loop with dynamic tqdm progress bars and 
    conditional saving based on peak validation accuracy.
    """
    print(f"\n{'='*60}\nInitiating Training: {name}\n{'='*60}")
    model.to(device)
    optimizer = setup_ssm_optimizer(model, base_lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    best_val_acc = 0.0
    # Sanitize the model name for the file system
    save_filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_best.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Initialize the training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} [Train]", leave=False)
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            # Update the progress bar dynamically with the current batch loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_acc = 100. * correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            # Initialize the validation progress bar
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1:03d}/{epochs} [Valid]", leave=False)
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        val_acc = 100. * correct_val / total_val
        
        # Checkpointing Logic: Only save if the model improved
        save_status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_filename)
            save_status = f"--> Saved Checkpoint ({best_val_acc:.2f}%)"
            
        # Print the final summary for the epoch, replacing the progress bars
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.2f}% {save_status}")

    print(f"\nTraining Complete for {name}. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {save_filename}\n")

if __name__ == "__main__":
    # 1. Establish the Baseline Budget
    # Using 4 layers of Pure Mamba at a standard 128 width
    baseline_d_model = 128
    baseline_layers = 4
    
    baseline_model = PureMamba(d_model=baseline_d_model, num_layers=baseline_layers)
    target_budget = count_parameters(baseline_model)
    
    print("-" * 50)
    print(f"BASELINE: Pure Mamba ({baseline_layers} Layers, d_model={baseline_d_model})")
    print(f"Target Parameter Budget: {target_budget:,}")
    print("-" * 50)

    # 2. Balance the Pure TCN
    # TCN strictly requires 8 layers to achieve a >1024 receptive field
    tcn_layers = 10
    
    # Calculate unadjusted parameters first to show the discrepancy
    unadjusted_tcn = PureTCN(d_model=baseline_d_model, num_layers=tcn_layers)
    print(f"\nUnadjusted TCN (10 Layers, d_model={baseline_d_model}): {count_parameters(unadjusted_tcn):,}")
    
    # Find the matching width
    matched_tcn_d, matched_tcn_params = find_matching_d_model(
        TargetModelClass=PureTCN, 
        target_params=target_budget, 
        base_d_model=baseline_d_model, 
        num_layers=tcn_layers
    )
    print(f"BALANCED TCN: d_model adjusted to {matched_tcn_d}")
    print(f"Balanced TCN Parameters: {matched_tcn_params:,} (Error: {abs(matched_tcn_params - target_budget) / target_budget * 100:.2f}%)")

    # 3. Balance the Hybrid Parallel Model
    # Keeping depth at 4 layers to match the Mamba baseline depth
    hybrid_layers = 4
    
    unadjusted_hybrid = HybridParallelMambaTCN(d_model=baseline_d_model, num_layers=hybrid_layers)
    print(f"\nUnadjusted Hybrid (4 Layers, d_model={baseline_d_model}): {count_parameters(unadjusted_hybrid):,}")
    
    matched_hybrid_d, matched_hybrid_params = find_matching_d_model(
        TargetModelClass=HybridParallelMambaTCN, 
        target_params=target_budget, 
        base_d_model=baseline_d_model, 
        num_layers=hybrid_layers
    )
    print(f"BALANCED HYBRID: d_model adjusted to {matched_hybrid_d}")
    print(f"Balanced Hybrid Parameters: {matched_hybrid_params:,} (Error: {abs(matched_hybrid_params - target_budget) / target_budget * 100:.2f}%)")
    print("-" * 50)

    ############################################################################################################################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Executing State Space Models on CPU is exceptionally slow.")
        
    train_loader, test_loader = get_noisy_scifar_dataloaders(batch_size=64)  # Reduced batch size for longer sequences
    
    # Instantiate the balanced models derived from the previous parameter search
    models = {
        f"Pure Mamba ({baseline_d_model} width)": PureMamba(d_model=baseline_d_model, num_layers=baseline_layers),
        f"Pure TCN ({matched_tcn_d} width)": PureTCN(d_model=matched_tcn_d, num_layers=tcn_layers, kernel_size=3),
        f"Hybrid Parallel ({matched_hybrid_d} width)": HybridParallelMambaTCN(d_model=matched_hybrid_d, num_layers=hybrid_layers, kernel_size=3)
    }
    
    for name, model in models.items():
        train_and_evaluate(model, name, train_loader, test_loader, epochs=50, device=device)