import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.hybrid_model import CNNMambaEdge
from utils.dataset import UCF101Dataset

def evaluate_model(handoff_idx, batch_size=4):
    print(f"Evaluating Model: Handoff Index {handoff_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Check if weights exist
    weights_path = f"checkpoints/best_model_handoff_{handoff_idx}.pth"
    if not os.path.exists(weights_path):
        print(f"Could not find weights at {weights_path}")
        return

    # Initialize Dataset and DataLoader
    data_dir = "data/raw/UCF-101"
    val_split = "data/raw/ucfTrainTestlist/testlist01.txt"   
    val_dataset = UCF101Dataset(data_dir, val_split, num_frames=16, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize Model and Load Weights
    model = CNNMambaEdge(num_classes=101, handoff_index=handoff_idx, cnn_size='small').to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # The Evaluation Loop
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(videos)

            # Get predictions (Top-1)
            _, predicted = torch.max(outputs, 1)
            
            # Store everything for scikit-learn to process
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 6. Calculate Paper-Ready Metrics
    print("📊 FINAL RESULTS")
    
    # Top-1 Accuracy
    top1_acc = accuracy_score(all_true_labels, all_predictions)
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    
    # Macro F1-Score (Treats all 101 classes equally, good for imbalanced data)
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Weighted F1-Score (Factors in if some classes have more videos)
    weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

if __name__ == "__main__":
    for handoff_idx in [4, 6, 8, 10, 12]:
        evaluate_model(handoff_idx=handoff_idx, batch_size=4)