import argparse
import os
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array

MODEL_PATH      = "cv_models/model_fold_2.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE        = (224, 224)

# ── Load class names saved by train.py ──
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)
else:
    # Fallback — Full list of classes from the Onion Diseases dataset
    CLASS_NAMES = [
        "Alternaria_D", "Botrytis Leaf Blight", "Bulb Rot", "Bulb_blight-D",
        "Caterpillar-P", "Downy mildew", "Fusarium-D", "Healthy leaves",
        "Iris yellow virus_augment", "Purple blotch", "Rust", "Virosis-D",
        "Xanthomonas Leaf Blight", "onion1", "stemphylium Leaf Blight"
    ]
    print(f"'{CLASS_NAMES_PATH}' not found, using default class names.\n")

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    return np.expand_dims(arr, axis=0) # Rescaling is now inside the model

def predict_image(model, img_path):
    arr   = preprocess_image(img_path)
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    conf  = float(preds[idx]) * 100

    print(f"\nImage      : {os.path.basename(img_path)}")
    print(f"   Result     : {label}")
    print(f"   Confidence : {conf:.1f}%")
    print("   All scores :")
    for name, score in zip(CLASS_NAMES, preds):
        bar = "█" * int(score * 30)
        print(f"     {name:<22} {bar} {score*100:.1f}%")
    return label, conf

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("--image",  type=str, help="Path to a single leaf image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    # Check for the model
    if not os.path.exists(MODEL_PATH):
        if os.path.exists("plant_disease_model.keras"):
            MODEL_PATH = "plant_disease_model.keras"
        else:
            print(f"Model not found. Please run 'train_cv.py' first.")
            return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded!\n")

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        predict_image(model, args.image)

    elif args.folder:
        files = [f for f in os.listdir(args.folder)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if not files:
            print(f"No image files found in '{args.folder}'")
            return
        print(f"Found {len(files)} image(s) in '{args.folder}'\n")
        for fname in sorted(files):
            predict_image(model, os.path.join(args.folder, fname))

    else:
        print("Provide --image or --folder.")
        print("   Example: python predict.py --image leaf.jpg")

if __name__ == "__main__":
    main()