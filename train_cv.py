import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2, ResNet50V2, DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 0.0001
RAW_DIR       = "raw_dataset"
MODEL_DIR     = "cv_models"
NUM_FOLDS     = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD ALL FILE PATHS AND LABELS
# ─────────────────────────────────────────────
all_file_paths = []
all_labels = []
class_names = sorted([d.name for d in Path(RAW_DIR).iterdir() if d.is_dir()])
num_classes = len(class_names)
class_to_idx = {name: i for i, name in enumerate(class_names)}

print(f"Found {num_classes} classes: {class_names}")

for class_name in class_names:
    class_dir = Path(RAW_DIR) / class_name
    files = list(class_dir.glob("*"))
    all_file_paths.extend([str(f) for f in files])
    all_labels.extend([class_to_idx[class_name]] * len(files))

all_file_paths = np.array(all_file_paths)
all_labels = np.array(all_labels)

# ─────────────────────────────────────────────
# 2. DATASET CREATION FUNCTION
# ─────────────────────────────────────────────
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    label = tf.one_hot(label, num_classes)
    return img, label

def create_dataset(paths, labels, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ─────────────────────────────────────────────
# 3. MODEL BUILDING FUNCTION
# ─────────────────────────────────────────────
def build_model(model_type="mobilenet"):
    if model_type == "mobilenet":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif model_type == "resnet":
        base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif model_type == "densenet":
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError("Invalid model_type")

    base_model.trainable = False
    
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # ─────────────────────────────────────────────
    # 4. STRATIFIED K-FOLD TRAINING
    # ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    print(f"\nStarting {NUM_FOLDS}-Fold Stratified Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_file_paths, all_labels)):
        print(f"\n{'='*40}")
        print(f" FOLD {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*40}")
        
        X_train_fold, y_train_fold = all_file_paths[train_idx], all_labels[train_idx]
        X_val_fold, y_val_fold = all_file_paths[val_idx], all_labels[val_idx]
        
        # --- OVERSAMPLING LOGIC ---
        unique, counts = np.unique(y_train_fold, return_counts=True)
        max_count = max(counts)
        
        X_resampled = []
        y_resampled = []
        
        for i, cls_idx in enumerate(unique):
            cls_paths = X_train_fold[y_train_fold == cls_idx]
            current_count = len(cls_paths)
            multiplier = max_count // current_count
            remainder = max_count % current_count
            
            X_resampled.extend(list(cls_paths) * multiplier)
            y_resampled.extend([cls_idx] * (current_count * multiplier))
            
            if remainder > 0:
                extra_indices = np.random.choice(len(cls_paths), remainder, replace=False)
                X_resampled.extend(list(cls_paths[extra_indices]))
                y_resampled.extend([cls_idx] * remainder)
                
        X_resampled = np.array(X_resampled)
        y_resampled = np.array(y_resampled)
        
        print(f"Original train size: {len(X_train_fold)} -> Resampled train size: {len(X_resampled)}")
        
        train_ds = create_dataset(X_resampled, y_resampled, augment=True)
        val_ds = create_dataset(X_val_fold, y_val_fold, shuffle=False)
        
        model = build_model(model_type="mobilenet")
        checkpoint_path = os.path.join(MODEL_DIR, f"model_fold_{fold+1}.keras")
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        fold_results.append(best_val_acc)
        print(f"Fold {fold+1} Validation Accuracy: {best_val_acc:.4f}")

    print(f"\n{'='*40}")
    print(f"CV RESULTS: {fold_results}")
    print(f"Average Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"{'='*40}")

    with open("cv_results.json", "w") as f:
        json.dump({"fold_accuracies": fold_results, "mean": np.mean(fold_results), "std": np.std(fold_results)}, f)
