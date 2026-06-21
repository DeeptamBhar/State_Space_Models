import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report

# Import structures from your original script
# Ensure the original script is named 'train.py'
from torchaudio_test import (
    PureMamba, 
    PureTCN, 
    HybridParallelMambaTCN, 
    get_audio_dataloaders, 
    find_matching_d_model, 
    count_parameters
)

def evaluate_model(model, dataloader, device, model_name):
    model.to(device)
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    # CUDA timing requires specific synchronization
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    
    # Warmup to prevent initial CUDA context creation from skewing results
    print(f"\nWarming up {model_name}...")
    dummy_input = torch.randn(16, 4000, 1).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Inference", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            starter.record()
            outputs = model(inputs)
            ender.record()
            
            # Synchronize to get accurate timing
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time / inputs.size(0)) # Time per sample in ms
            
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    avg_inference_ms = sum(timings) / len(timings)
    
    return all_targets, all_predictions, avg_inference_ms

def generate_graphs(results_data, num_classes):
    print("\nGenerating evaluation graphs...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Professional color palette for colorblind accessibility
    colors = {
        "Pure Mamba": "#003f5c",        # Navy
        "Hybrid Parallel": "#bc5090",   # Magenta/Teal alternative
        "Pure TCN": "#ffa600"           # Orange
    }
    
    # --- Graph 1: Scatter Plot (Efficiency vs. Performance) ---
    for name, data in results_data.items():
        base_name = name.split(" (")[0]
        color = colors.get(base_name, "black")
        
        ax1.scatter(data['latency'], data['accuracy'] * 100, 
                    s=200, label=base_name, color=color, zorder=5, edgecolor='black')
        
        # Annotate exact percentages
        ax1.annotate(f"{data['accuracy']*100:.2f}%", 
                     (data['latency'], data['accuracy'] * 100), 
                     xytext=(8, -5), textcoords='offset points', fontsize=10)
        
    ax1.set_xlabel("Inference Time per sample (ms)", fontsize=12)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Efficiency vs. Performance", fontsize=14, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Graph 2: Line Graph (Per-Class Anomalies) ---
    classes = np.arange(num_classes)
    for name, data in results_data.items():
        base_name = name.split(" (")[0]
        color = colors.get(base_name, "black")
        
        ax2.plot(classes, data['f1_scores'], marker='o', markersize=4, 
                 linewidth=2, label=base_name, color=color, alpha=0.85)
        
    # Highlight specific anomalous classes with red bands
    anomalies = [8, 9, 14, 28]
    for a in anomalies:
        ax2.axvspan(a - 0.5, a + 0.5, color='red', alpha=0.15)
        
    ax2.set_xlabel("Audio Class ID", fontsize=12)
    ax2.set_ylabel("F1-Score", fontsize=12)
    ax2.set_title("Per-Class F1-Scores & Dataset Anomalies", fontsize=14, pad=15)
    ax2.set_ylim(0.65, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(np.arange(0, num_classes, 2))
    
    # Global Legend placement
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("torchaudio_results_graph.png", bbox_inches='tight', dpi=300)
    print("Saved graphs to 'torchaudio_results_graph.png'")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CRITICAL: CPU evaluation will not yield accurate latency metrics for these architectures.")

    _, test_loader = get_audio_dataloaders(batch_size=64, num_workers=4)

    # 1. Recalculate dimensions to match the saved state dictionaries exactly
    baseline_d_model = 64
    baseline_layers = 8
    tcn_layers = 11
    hybrid_layers = 8
    num_classes = 35

    dummy_baseline = PureMamba(d_model=baseline_d_model, num_layers=baseline_layers, num_classes=num_classes)
    target_budget = count_parameters(dummy_baseline)

    matched_tcn_d, _ = find_matching_d_model(
        TargetModelClass=PureTCN, target_params=target_budget, 
        base_d_model=baseline_d_model, num_layers=tcn_layers, num_classes=num_classes
    )
    
    matched_hybrid_d, _ = find_matching_d_model(
        TargetModelClass=HybridParallelMambaTCN, target_params=target_budget, 
        base_d_model=baseline_d_model, num_layers=hybrid_layers, num_classes=num_classes
    )

    # 2. Reconstruct models
    models = {
        f"Pure Mamba ({baseline_d_model} width)": PureMamba(d_model=baseline_d_model, num_layers=baseline_layers, num_classes=num_classes),
        f"Pure TCN ({matched_tcn_d} width)": PureTCN(d_model=matched_tcn_d, num_layers=tcn_layers, kernel_size=3, num_classes=num_classes),
        f"Hybrid Parallel ({matched_hybrid_d} width)": HybridParallelMambaTCN(d_model=matched_hybrid_d, num_layers=hybrid_layers, kernel_size=3, num_classes=num_classes)
    }

    # Dictionary to store data for final plotting
    results_data = {}

    # 3. Load weights and evaluate
    for name, model in models.items():
        weight_file = f"torchaudio_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_best.pth"
        
        if not os.path.exists(weight_file):
            print(f"ERROR: Weight file {weight_file} not found. Skipping {name}.")
            continue
            
        try:
            model.load_state_dict(torch.load(weight_file, map_location=device))
        except Exception as e:
            print(f"ERROR loading {weight_file}: {e}")
            continue

        targets, predictions, latency = evaluate_model(model, test_loader, device, name)
        
        # Extract detailed metrics programmatically
        report_dict = classification_report(targets, predictions, zero_division=0, output_dict=True)
        accuracy = report_dict['accuracy']
        f1_scores = [report_dict[str(i)]['f1-score'] for i in range(num_classes)]
        
        # Save to plotting dictionary
        results_data[name] = {
            'accuracy': accuracy,
            'latency': latency,
            'f1_scores': f1_scores
        }
        
        print(f"\n--- Results: {name} ---")
        print(f"Average Inference Time per sample: {latency:.4f} ms")
        
        # Print report with 4 decimal places instead of 2
        print(classification_report(targets, predictions, zero_division=0, digits=4))
        print("-" * 50)

    # 4. Generate the final combined graph for LaTeX
    if results_data:
        generate_graphs(results_data, num_classes)