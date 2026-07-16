import os
import pandas as pd
import numpy as np

# The classes we specifically want to track for the "Macro vs Micro" slide
MACRO_CLASSES = {11: "Biking", 43: "HulaHoop", 70: "PullUps", 15: "Squats"}
MICRO_CLASSES = {20: "BrushingTeeth", 62: "PlayingFlute", 100: "WritingOnBoard"}

handoffs = [4, 6, 8, 10, 12]


for h in handoffs:
    csv_path = f"results/per_class_metrics_handoff_{h}.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Handoff {h} CSV not found at {csv_path}.")
        
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Ensure Class_Index is an integer for safe matching
    df['Class_Index'] = df['Class_Index'].astype(int)
    
    # Sort the dataframe by f1-score descending to get the "Top N" rankings
    df = df.sort_values(by='f1-score', ascending=False).reset_index(drop=True)

    print(f"Handoff Deptch {h} Analysis")
    
    # Extract specific macro and micro classes
    print("Macro Motion Classes:")
    for class_id, class_name in MACRO_CLASSES.items():
        row = df[df['Class_Index'] == class_id]
        if not row.empty:
            f1 = row.iloc[0]['f1-score']
            recall = row.iloc[0]['recall'] * 100
            print(f"   - {class_name.ljust(15)} | Class Acc (Recall): {recall:5.1f}% | F1: {f1:.3f}")

    print("Micro Motion Classes:")
    for class_id, class_name in MICRO_CLASSES.items():
        row = df[df['Class_Index'] == class_id]
        if not row.empty:
            f1 = row.iloc[0]['f1-score']
            recall = row.iloc[0]['recall'] * 100
            print(f"   - {class_name.ljust(15)} | Class Acc (Recall): {recall:5.1f}% | F1: {f1:.3f}")

    print("Subset Performance:")

    subsets = [10, 30, 50]
    for n in subsets:
        if len(df) >= n:
            top_n_df = df.head(n)
            avg_acc = top_n_df['recall'].mean() * 100
            avg_f1 = top_n_df['f1-score'].mean()
            print(f"   - Top {n} Classes Average | Accuracy: {avg_acc:5.1f}% | F1: {avg_f1:.3f}")