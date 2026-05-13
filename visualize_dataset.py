import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
RAW_DIR = "cv_data"
OUTPUT_PLOT = "dataset_distribution.png"

def get_stats(data_dir):
    stats = {}
    path = Path(data_dir)
    if not path.exists():
        return None
    
    for class_dir in path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            if count > 0:
                stats[class_dir.name] = count
    return stats

def create_visualization():
    raw_stats = get_stats(RAW_DIR)
    if not raw_stats:
        print(f"Error: {RAW_DIR} not found.")
        return

    # Sort classes by count for better visualization
    classes = sorted(raw_stats.keys(), key=lambda x: raw_stats[x], reverse=True)
    counts_before = [raw_stats[cls] for cls in classes]

    # Calculate "After" (Simulated Oversampling)
    # Target: Oversample minority classes to at least 1000 or the max class count
    max_count = max(counts_before)
    target_count = max_count # Balancing to the majority class
    counts_after = [max(count, target_count) if count < target_count else count for count in counts_before]

    # Set up the plot
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Before Plot
    sns.barplot(x=counts_before, y=classes, ax=ax1, hue=classes, palette="viridis", legend=False)
    ax1.set_title("Before: Raw Dataset Distribution (Heavily Imbalanced)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Number of Images")
    for i, v in enumerate(counts_before):
        ax1.text(v + 10, i, str(v), color='black', va='center')

    # After Plot
    sns.barplot(x=counts_after, y=classes, ax=ax2, hue=classes, palette="magma", legend=False)
    ax2.set_title("After: Balanced Distribution (Using Oversampling)", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Number of Images (Augmented/Duplicated)")
    for i, v in enumerate(counts_after):
        ax2.text(v + 10, i, str(v), color='black', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Visualization saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    create_visualization()
