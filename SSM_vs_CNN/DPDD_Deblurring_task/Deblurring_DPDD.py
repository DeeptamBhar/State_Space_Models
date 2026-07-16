import os
import sys
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from mamba_ssm import Mamba
from einops import rearrange
from pytorch_msssim import ssim
import lpips

# ==========================================
# 1. Logging & Checkpoint Utilities
# ==========================================
def setup_logger(log_file):
    logger = logging.getLogger("DPDD_Training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Log to console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, filename.replace('checkpoint', 'model_best'))

def load_checkpoint(filename, model, optimizer, scaler):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        return checkpoint['epoch'], checkpoint['best_psnr']
    return 0, 0.0

# ==========================================
# 2. Dataset & Loss (Optimized for SLURM I/O)
# ==========================================
class BlurDataset(Dataset):
    def __init__(self, root_dir, split='train_c', crop_size=256):
        self.blur_dir = os.path.join(root_dir, split, 'source')
        self.sharp_dir = os.path.join(root_dir, split, 'target')
        
        # 1. Grab everything and sort them alphanumerically
        blur_files = sorted([f for f in os.listdir(self.blur_dir) if f.endswith(('.png', '.jpg'))])
        sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if f.endswith(('.png', '.jpg'))])
        
        # 2. Safety Check: If lengths don't match, the n-th mapping is broken
        if len(blur_files) != len(sharp_files):
            print(f"CRITICAL WARNING: Dataset size mismatch in {split}! "
                  f"Source has {len(blur_files)} images, Target has {len(sharp_files)} images.")
        
        # 3. Zip them together sequentially: (blur_0, sharp_0), (blur_1, sharp_1), etc.
        self.image_pairs = list(zip(blur_files, sharp_files))
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size) if 'train' in split else transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # Unpack the strictly ordered pair
        blur_name, sharp_name = self.image_pairs[idx]
        
        blur_img = Image.open(os.path.join(self.blur_dir, blur_name)).convert('RGB')
        sharp_img = Image.open(os.path.join(self.sharp_dir, sharp_name)).convert('RGB')
        
        # Synchronized transforms
        seed = torch.random.seed()
        torch.manual_seed(seed)
        blur_tensor = self.transform(blur_img)
        torch.manual_seed(seed)
        sharp_tensor = self.transform(sharp_img)
        
        return blur_tensor, sharp_tensor
    
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        return (self.alpha * ssim_loss) + ((1 - self.alpha) * self.l1(pred, target))

def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==========================================
# 3. Architecture (Modified for B=2 Stability)
# ==========================================
class CNNMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        self.norm_global = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # GroupNorm explicitly replaces LayerNorm in CNN branch for B <= 2
        self.norm_local = nn.GroupNorm(num_groups=8, num_channels=d_model)
        self.cnn = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model))

    def forward(self, x):
        B, H, W, C = x.shape
        
        x_flat = rearrange(x, 'b h w c -> b (h w) c')
        x_global = self.mamba(self.norm_global(x_flat))
        
        x_c = rearrange(x, 'b h w c -> b c h w')
        x_local = self.norm_local(x_c)
        x_local = self.cnn(x_local)
        x_local = rearrange(x_local, 'b c h w -> b (h w) c')
        
        fused = self.fusion(torch.cat([x_global, x_local], dim=-1))
        fused = rearrange(fused, 'b (h w) c -> b h w c', h=H, w=W)
        return fused + (x * self.scale)

class PureMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model))

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = rearrange(x, 'b h w c -> b (h w) c')
        out = self.mamba(self.norm(x_flat))
        out = rearrange(out, 'b (h w) c -> b h w c', h=H, w=W)
        return out + (x * self.scale)

class PureCNNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=d_model)
        self.cnn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model))

    def forward(self, x):
        x_c = rearrange(x, 'b h w c -> b c h w')
        out = self.cnn(self.norm(x_c))
        out = rearrange(out, 'b c h w -> b h w c')
        return out + (x * self.scale)

class RestorationUNet(nn.Module):
    def __init__(self, block_type='hybrid', in_ch=3, out_ch=3, base_c=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, base_c, kernel_size=3, padding=1)
        
        BlockMap = {'hybrid': CNNMambaBlock, 'mamba': PureMambaBlock, 'cnn': PureCNNBlock}
        BlockClass = BlockMap[block_type]
        
        self.encoder = BlockClass(base_c)
        self.down = nn.Conv2d(base_c, base_c * 2, kernel_size=2, stride=2)
        self.latent = BlockClass(base_c * 2)
        self.up = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.decoder = BlockClass(base_c)
        self.refinement = BlockClass(base_c)
        self.out_layer = nn.Conv2d(base_c, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        x = rearrange(self.patch_embed(x), 'b c h w -> b h w c')
        
        x_enc = self.encoder(x)
        x_down = rearrange(self.down(rearrange(x_enc, 'b h w c -> b c h w')), 'b c h w -> b h w c')
        
        x_lat = self.latent(x_down)
        x_up = rearrange(self.up(rearrange(x_lat, 'b h w c -> b c h w')), 'b c h w -> b h w c')
        
        x_dec = self.decoder(x_up + x_enc) 
        x_ref = self.refinement(x_dec)
        
        out = self.out_layer(rearrange(x_ref, 'b h w c -> b c h w'))
        return out + identity

# ==========================================
# 4. Training Loop with Timeout Logic
# ==========================================
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(f"training_{args.model}.log")
    
    # Setup Dataloaders
    train_set = BlurDataset(args.data_dir, split='train_c', crop_size=256)
    val_set = BlurDataset(args.data_dir, split='val_c', crop_size=256)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    model = RestorationUNet(block_type=args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    criterion = HybridLoss().to(device)
    
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_metric.parameters():
        param.requires_grad = False

    checkpoint_file = f'checkpoint_{args.model}.pth.tar'
    start_epoch, best_psnr = load_checkpoint(checkpoint_file, model, optimizer, scaler)
    
    # 46.5 hours in seconds to safely exit before SLURM kills the job at 48 hours
    MAX_RUNTIME_SEC = 46.5 * 3600 
    script_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (blur, sharp) in enumerate(train_loader):
            # SLURM Timeout Check
            if time.time() - script_start_time > MAX_RUNTIME_SEC:
                logger.info("Time limit approaching. Saving checkpoint and exiting for requeue.")
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_psnr': best_psnr,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }, is_best=False, filename=checkpoint_file)
                sys.exit(99) # Triggers SLURM bash script to resubmit

            blur, sharp = blur.to(device, non_blocking=True), sharp.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # AMP Forward Pass
            with autocast():
                pred = model(blur)
                loss = criterion(pred, sharp)
            
            # AMP Backward Pass & Clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Validation Phase
        model.eval()
        val_psnr = 0.0
        val_lpips = 0.0
        
        with torch.no_grad():
            for blur, sharp in val_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                
                with autocast():
                    pred = model(blur)
                
                # Metrics
                val_psnr += calculate_psnr(pred, sharp).item()
                val_lpips += lpips_metric(pred * 2 - 1, sharp * 2 - 1).mean().item()

        avg_psnr = val_psnr / len(val_loader)
        avg_lpips = val_lpips / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | Val PSNR: {avg_psnr:.2f} | Val LPIPS: {avg_lpips:.4f}")

        # Checkpoint
        is_best = avg_psnr > best_psnr
        best_psnr = max(avg_psnr, best_psnr)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, is_best, filename=checkpoint_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPDD Training")
    parser.add_argument('--model', type=str, choices=['cnn', 'mamba', 'hybrid'], required=True)
    parser.add_argument('--data_dir', type=str, required=True, help="Path to extracted DPDD dataset")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    
    train_model(args)