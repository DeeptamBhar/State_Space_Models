import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

from models.hybrid_model import CNNMambaEdge
from utils.dataset import UCF101Dataset

def calculate_topk_counts(output, target, topk=(1, 3, 5)):
    """Helper function to calculate top-k counts."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            res.append(correct_k)
        return res

def evaluate_real_edge(handoff_idx, batch_size=1):
    print(f"Evaluate on videos: Handoff Index {handoff_idx}")

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Locate Weights
    weights_path = f"checkpoints/best_model_handoff_{handoff_idx}.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find weights at {weights_path}")

    # Initialize Dataset
    data_dir = "data/raw/UCF-101"
    val_split = "data/raw/ucfTrainTestlist/testlist01.txt" 
    
    val_dataset = UCF101Dataset(data_dir, val_split, num_frames=16, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    model = CNNMambaEdge(num_classes=101, handoff_index=handoff_idx, cnn_size='small').to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_predictions = []
    all_true_labels = []
    
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_videos = len(val_dataset)

    # GPU Warmup
    dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    print("Starting End-to-End Inference")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Inference Loop
    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Processing Videos"):
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

            c1, c3, c5 = calculate_topk_counts(outputs, labels, topk=(1, 3, 5))
            top1_correct += c1
            top3_correct += c3
            top5_correct += c5

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 5. Hardware Metrics
    total_time = end_time - start_time
    fps = total_videos / total_time
    latency_ms = (total_time / total_videos) * 1000

    top1_acc = (top1_correct / total_videos) * 100.0
    top3_acc = (top3_correct / total_videos) * 100.0
    top5_acc = (top5_correct / total_videos) * 100.0
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_df = pd.DataFrame(cm)
    cm_csv_path = f"results/confusion_matrix_handoff_{handoff_idx}.csv"
    cm_df.to_csv(cm_csv_path, index=False)

    # Generate Per-Class Metrics (Precision, Recall, F1, Support)
    report_dict = classification_report(all_true_labels, all_predictions, output_dict=True, zero_division=0)
    
    # Filter out the "accuracy" and "macro avg" rows so we only get the 101 specific classes
    class_metrics = {k: v for k, v in report_dict.items() if k.isdigit()}
    report_df = pd.DataFrame(class_metrics).transpose()
    report_df.index.name = 'Class_Index'
    
    # Sort by F1-Score to easily find best and worst
    report_df = report_df.sort_values(by='f1-score', ascending=False)
    
    per_class_csv_path = f"results/per_class_metrics_handoff_{handoff_idx}.csv"
    report_df.to_csv(per_class_csv_path)

    # Extract Top 5 Best and Worst for terminal printout
    best_5 = report_df.head(5)
    worst_5 = report_df.tail(5)

    print(f"Final metrics for handoff: {handoff_idx}")
    print("Predictive Power")
    print(f"Top-1 Accuracy : {top1_acc:.2f}%")
    print(f"Top-3 Accuracy : {top3_acc:.2f}%")
    print(f"Top-5 Accuracy : {top5_acc:.2f}%")
    print(f"Macro F1-Score : {macro_f1:.4f}")
    print("Hardware Performance")
    print(f"Total Videos   : {total_videos}")
    print(f"Throughput     : {fps:.2f} FPS")
    print(f"Latency/Video  : {latency_ms:.2f} ms")
    print(f"Saved Confusion Matrix to: {cm_csv_path}")
    print(f"Saved Per-Class Data to  : {per_class_csv_path}")
    print("TOP 5 BEST PERFORMING CLASSES (By F1-Score):")
    for idx, row in best_5.iterrows():
        print(f"   Class {idx.zfill(3)} | Recall: {row['recall']*100:.1f}% | Precision: {row['precision']*100:.1f}% | F1: {row['f1-score']:.3f}")

    print("TOP 5 WORST PERFORMING CLASSES (By F1-Score):")
    for idx, row in worst_5.iterrows():
        print(f"   Class {idx.zfill(3)} | Recall: {row['recall']*100:.1f}% | Precision: {row['precision']*100:.1f}% | F1: {row['f1-score']:.3f}")

if __name__ == "__main__":
    for h in [4, 6, 8, 10, 12]:
        evaluate_real_edge(handoff_idx=h, batch_size=1)