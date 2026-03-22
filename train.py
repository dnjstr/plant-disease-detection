"""
Plant Disease Detection - Training Script
Compatible with: TensorFlow 2.20.0, Python 3.11 / 3.12
Uses MobileNetV2 (pre-trained on ImageNet) with transfer learning
Dataset: Onion Diseases (Kaggle - Tejas Barguje Patil)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# TF 2.20+ uses keras 3 — import from keras directly
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory

print(f"✅ TensorFlow version : {tf.__version__}")
print(f"✅ Keras version      : {keras.__version__}\n")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.0001
DATA_DIR      = "dataset"           # folder with train/ val/ test/ subfolders
MODEL_SAVE    = "plant_disease_model.keras"   # .keras format (recommended in TF 2.20+)

# ─────────────────────────────────────────────
# DATA LOADING  (using modern image_dataset_from_directory)
# ─────────────────────────────────────────────
train_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42,
)

val_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ Classes found ({num_classes}): {class_names}\n")

# ─────────────────────────────────────────────
# PREPROCESSING + AUGMENTATION  (as Keras layers)
# ─────────────────────────────────────────────
rescale = keras.layers.Rescaling(1.0 / 255)

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.1),
], name="augmentation")

# Apply to datasets
train_ds = train_ds.map(lambda x, y: (augment(rescale(x), training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (rescale(x), y),
                      num_parallel_calls=tf.data.AUTOTUNE)

# Cache + prefetch for speed
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ─────────────────────────────────────────────
# BUILD MODEL  (MobileNetV2 + custom head)
# ─────────────────────────────────────────────
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False   # freeze pre-trained layers first

inputs      = keras.Input(shape=(224, 224, 3))
x           = base_model(inputs, training=False)
x           = GlobalAveragePooling2D()(x)
x           = BatchNormalization()(x)
x           = Dense(256, activation="relu")(x)
x           = Dropout(0.5)(x)
x           = Dense(128, activation="relu")(x)
x           = Dropout(0.3)(x)
outputs     = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
]

# ─────────────────────────────────────────────
# PHASE 1 — Train only the custom head (base frozen)
# ─────────────────────────────────────────────
print("\n📌 Phase 1: Training custom head (base frozen)...\n")
history1 = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# PHASE 2 — Fine-tune last 30 layers of base
# ─────────────────────────────────────────────
print("\n📌 Phase 2: Fine-tuning last 30 layers...\n")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val, label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,  label="Train Loss")
    plt.plot(epochs_range, vloss, label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("📊 Training plot saved as training_history.png")

plot_history(history1, history2)

# Save class names for use in predict.py
import json
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print(f"\n✅ Model saved to  : {MODEL_SAVE}")
print(f"✅ Class names saved: class_names.json")