import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from train_cv import build_model, create_dataset, RAW_DIR, class_names, num_classes, all_file_paths, all_labels

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
EPOCHS_COMPARE = 5
ARCHITECTURES = ["mobilenet", "resnet", "densenet"]
RESULTS = {}

def run_comparison():
    print(f"Starting architecture comparison: {ARCHITECTURES}")
    
    # One clean stratified split for all models to be fair
    X_train, X_val, y_train, y_val = train_test_split(
        all_file_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    # Oversample the training set once
    unique, counts = np.unique(y_train, return_counts=True)
    max_count = max(counts)
    X_resampled, y_resampled = [], []
    for cls_idx in unique:
        cls_paths = X_train[y_train == cls_idx]
        multiplier = max_count // len(cls_paths)
        X_resampled.extend(list(cls_paths) * multiplier)
        y_resampled.extend([cls_idx] * (len(cls_paths) * multiplier))
        
    X_resampled, y_resampled = np.array(X_resampled), np.array(y_resampled)
    
    train_ds = create_dataset(X_resampled, y_resampled, augment=True)
    val_ds = create_dataset(X_val, y_val, shuffle=False)

    for arch in ARCHITECTURES:
        print(f"\n--- Training {arch.upper()} ---")
        model = build_model(model_type=arch)
        
        start_time = time.time()
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=EPOCHS_COMPARE,
            verbose=1
        )
        end_time = time.time()
        
        RESULTS[arch] = {
            "val_accuracy": max(history.history["val_accuracy"]),
            "val_loss": min(history.history["val_loss"]),
            "train_time": end_time - start_time
        }

    # ─────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    names = [n.upper() for n in ARCHITECTURES]
    accs = [RESULTS[a]["val_accuracy"] for a in ARCHITECTURES]
    times = [RESULTS[a]["train_time"] for a in ARCHITECTURES]

    # Accuracy Plot
    sns.barplot(x=names, y=accs, ax=ax1, palette="viridis")
    ax1.set_title("Validation Accuracy Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

    # Time Plot
    sns.barplot(x=names, y=times, ax=ax2, palette="magma")
    ax2.set_title("Training Time Comparison (seconds)", fontsize=14, fontweight='bold')
    for i, v in enumerate(times):
        ax2.text(i, v + 1, f"{int(v)}s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nComparison complete! Plot saved as model_comparison.png")

if __name__ == "__main__":
    run_comparison()
