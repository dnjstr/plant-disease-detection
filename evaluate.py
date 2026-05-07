import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH      = "cv_models/model_fold_1.keras" # Default to Fold 1
CLASS_NAMES_PATH = "class_names.json"
TEST_DIR        = os.path.join("dataset", "test")
RAW_DATA_DIR    = "raw_dataset"
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32

# ── Load class names ──
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
else:
    class_names = None

# ── Load model ──
if not os.path.exists(MODEL_PATH):
    # Try the old name just in case
    if os.path.exists("plant_disease_model.keras"):
        MODEL_PATH = "plant_disease_model.keras"
    else:
        print(f" Model not found. Please ensure cv_models/ contains your trained models.")
        exit()

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded!\n")

# ── Test dataset ──
if os.path.exists(TEST_DIR):
    print(f"Using test set from: {TEST_DIR}")
    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
else:
    print(f"'{TEST_DIR}' not found. Creating a test split from '{RAW_DATA_DIR}'...")
    test_ds = image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

if class_names is None:
    class_names = test_ds.class_names

print(f"Classes: {class_names}\n")

# Prefetch (Rescaling moved to model)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ── Predict ──
print("Running predictions on test set...")
y_true_batches = []
y_pred_batches = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred_batches.append(np.argmax(preds, axis=1))
    y_true_batches.append(np.argmax(labels.numpy(), axis=1))

y_pred = np.concatenate(y_pred_batches)
y_true = np.concatenate(y_true_batches)

# ── Metrics ──
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Confusion matrix ──
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=class_names, yticklabels=class_names,
)
plt.title("Confusion Matrix — Plant Disease Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")