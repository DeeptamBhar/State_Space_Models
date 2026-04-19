import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import wandb 

from models.hybrid_model import CNNMambaEdge
from utils.dataset import UCF101Dataset

def train_model(handoff_idx, epochs=5, batch_size=4, lr=1e-4):
    
    print(f"Start training: Handoff Index {handoff_idx}")

    # Initialise W&B
    wandb.init(
        project="cnn-mamba-edge",
        name=f"handoff_{handoff_idx}",
        config={
            "handoff_index": handoff_idx,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "MobileNetV3 + Mamba 1D"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_dir = "data/raw/UCF-101"
    train_split = "data/raw/ucfTrainTestlist/trainlist01.txt" 
    val_split = "data/raw/ucfTrainTestlist/testlist01.txt" 

    # Initialize Datasets and DataLoaders
    print("Loading datasets...")
    train_dataset = UCF101Dataset(data_dir, train_split, num_frames=16, is_training=True)
    val_dataset = UCF101Dataset(data_dir, val_split, num_frames=16, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Batches per epoch: {len(train_loader)}")

    # Initialize Model
    model = CNNMambaEdge(num_classes=101, handoff_index=handoff_idx, cnn_size='small').to(device)

    # CSV Logging Setup
    os.makedirs("checkpoints", exist_ok=True)
    csv_filename = f"checkpoints/training_log_handoff_{handoff_idx}.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)

    # The Training Loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100 * correct_train / total_train:.2f}%"})

        train_acc = 100 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        print(f"Evaluating Epoch {epoch+1}")
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                videos, labels = videos.to(device), labels.to(device)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        if total_val > 0:
            val_acc = 100 * correct_val / total_val
            train_loss_val = running_loss / len(train_loader)
            val_loss_val = val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}] Summary:")
            print(f"Train Loss: {train_loss_val:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss_val:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Append data to CSV
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{train_loss_val:.4f}", f"{train_acc:.2f}", f"{val_loss_val:.4f}", f"{val_acc:.2f}"])
            
            # 3. Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss_val,
                "train_acc": train_acc,
                "val_loss": val_loss_val,
                "val_acc": val_acc,
                "best_val_acc": max(best_val_acc, val_acc)
            })

            # Save best model 
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f"checkpoints/best_model_handoff_{handoff_idx}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Saved new best model to {save_path}")
        else:
            train_loss_val = running_loss/len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss_val:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss_val,
                "train_acc": train_acc
            })

            save_path = f"checkpoints/latest_model_handoff_{handoff_idx}.pth"
            torch.save(model.state_dict(), save_path)

    # Finalize W&B
    wandb.finish()
    print(f"✅ Finished logging Handoff {handoff_idx} to Weights & Biases.")

if __name__ == "__main__":
    for handoff in [4,6,8,10,12]:
        train_model(handoff_idx=handoff, epochs=100, batch_size=4, lr=1e-4)