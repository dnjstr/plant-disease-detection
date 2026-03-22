import argparse
import os
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array

MODEL_PATH      = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE        = (224, 224)

# ── Load class names saved by train.py ──
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)
else:
    # Fallback — update these if your dataset differs
    CLASS_NAMES = ["Downy Mildew", "Healthy", "Iris Yellow Spot", "Purple Blotch"]
    print(f"'{CLASS_NAMES_PATH}' not found, using default class names.\n")

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

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

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at '{MODEL_PATH}'.")
        print("   Run train.py first to create the model.")
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