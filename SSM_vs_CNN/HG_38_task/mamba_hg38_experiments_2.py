import os
import gzip
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from mamba_ssm import Mamba
from tqdm import tqdm
import math

# --- 1. Data Preparation ---
class GenomicDataset(Dataset):
    def __init__(self, data_tensor, seq_len):
        self.seq_len = seq_len
        # Calculate exactly how many non-overlapping chunks fit
        self.num_samples = (len(data_tensor) - 1) // seq_len
        self.data = data_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y

def get_data():
    file_name = "chr22.fa.gz"
    if not os.path.exists(file_name):
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
        urllib.request.urlretrieve(url, file_name)
    
    vocab = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'a': 1, 'c': 2, 'g': 3, 't': 4}
    seq_tensor = []
    
    with gzip.open(file_name, 'rt') as f:
        f.readline()
        for line in f:
            for char in line.strip():
                if char in vocab:
                    seq_tensor.append(vocab[char])
                    
    full_data = torch.tensor(seq_tensor, dtype=torch.long)
    
    # 90% Train, 10% Validation Split
    split_idx = int(len(full_data) * 0.9)
    return full_data[:split_idx], full_data[split_idx:]

# --- 2. Deep Mamba Architecture ---
class DeepGenomicMamba(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Stack multiple Mamba layers for actual feature learning
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            # Standard sequential pass through the S6 blocks
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

def print_parameter_breakdown(model):
    """
    Iterates through the PyTorch model and prints a formatted table 
    of all trainable parameter layers, their shapes, and exact counts.
    """
    total_params = 0
    
    print(f"\n{'Layer Name':<60} | {'Shape':<20} | {'Param Count':>12}")
    print("-" * 100)
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
            
        param_count = parameter.numel()
        shape = str(list(parameter.shape))
        
        print(f"{name:<60} | {shape:<20} | {param_count:>12,}")
        total_params += param_count
        
    print("-" * 100)
    print(f"{'Total Trainable Parameters:':<83} {total_params:>12,}\n")
    
    return total_params

def run_experiment(train_data, val_data, seq_len, n_layers, d_model=128, batch_size=16, epochs=10):
    """Runs a complete training loop for a specific hyperparameter configuration."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: seq_len={seq_len} | n_layers={n_layers} | d_model={d_model}")
    print(f"{'='*60}")
    
    # Rebuild datasets based on the specific seq_len
    train_loader = DataLoader(GenomicDataset(train_data, seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenomicDataset(val_data, seq_len), batch_size=batch_size, shuffle=False, drop_last=True)
    
    device = torch.device("cuda")
    model = DeepGenomicMamba(d_model=d_model, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Updated GradScaler syntax
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_pp = float('inf')
    
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} [Train]", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = criterion(logits.view(-1, 5), y.view(-1))
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(x)
                        loss = criterion(logits.view(-1, 5), y.view(-1))
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            val_pp = math.exp(avg_val_loss)
            
            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val PP: {val_pp:.4f}")
            
            if val_pp < best_val_pp:
                best_val_pp = val_pp
                
        return best_val_pp

    except RuntimeError as e:
        if "Out of memory" in str(e):
            print(f"\n[CRASH] Out of Memory for seq_len={seq_len}, n_layers={n_layers}. Skipping.")
            torch.cuda.empty_cache()
            return float('inf')
        else:
            raise e

if __name__ == "__main__":
    train_data, val_data = get_data()
    
    # Hyperparameter Grid
    sequence_lengths = [1024, 2048, 4096]
    layer_counts = [2, 4, 6]
    
    # Store results for analysis
    results = {}
    
    for sl in sequence_lengths:
        for nl in layer_counts:
            # Dynamically reduce batch size for aggressive memory configurations
            current_batch_size = 16 if (sl <= 2048 and nl <= 4) else 4
            
            final_pp = run_experiment(
                train_data=train_data, 
                val_data=val_data, 
                seq_len=sl, 
                n_layers=nl, 
                batch_size=current_batch_size
            )
            results[(sl, nl)] = final_pp

    print("\n" + "="*40)
    print("FINAL GRID SEARCH RESULTS (Best Val PP)")
    print("="*40)
    for (sl, nl), pp in results.items():
        pp_str = f"{pp:.4f}" if pp != float('inf') else "OOM Failure"
        print(f"SeqLen: {sl:<5} | Layers: {nl:<2} | PP: {pp_str}")