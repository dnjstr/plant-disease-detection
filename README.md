# Plant Disease Detection from Leaf Images

A machine learning project that detects onion plant diseases from leaf photos using **MobileNetV2** transfer learning with **Stratified K-Fold Cross-Validation** and **Oversampling** for imbalanced datasets.

> **Course Project** — Machine Learning
> **Dataset:** [Onion Diseases – Kaggle (Tejas Barguje Patil)](https://www.kaggle.com/datasets/tejasbargujepatil/onion-diseases/data)

---

## Table of Contents
- [About the Project](#about-the-project)
- [Disease Classes](#disease-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## About the Project

This project uses a pre-trained **MobileNetV2** convolutional neural network (CNN) with transfer learning to classify onion leaf images into 15 disease categories.

**Key features:**
- **Stratified K-Fold Cross-Validation** (5 folds) for fair, robust evaluation
- **Oversampling** to handle severe class imbalance (e.g., 3,440 Healthy vs. 7 Bulb Rot images)
- **Multi-architecture comparison** — MobileNetV2, ResNet50V2, and DenseNet121

**Training is done in two phases per fold:**
1. **Phase 1** — Train only the custom classification head (base model frozen)
2. **Phase 2** — Fine-tune the top layers of the chosen backbone

**Tech Stack:**
- Python 3.11 / 3.12
- TensorFlow 2.20.0 / Keras 3
- MobileNetV2, ResNet50V2, DenseNet121 (pre-trained on ImageNet)

---

## Disease Classes

| # | Class | Image Count |
|---|-------|-------------|
| 1 | Alternaria_D | 830 |
| 2 | Botrytis Leaf Blight | 289 |
| 3 | Bulb Rot | 7 |
| 4 | Bulb_blight-D | 394 |
| 5 | Caterpillar-P | 1,558 |
| 6 | Downy Mildew | 37 |
| 7 | Fusarium-D | 1,276 |
| 8 | Healthy Leaves | 3,440 |
| 9 | Iris Yellow Virus Augment | 1,899 |
| 10 | onion1 | 132 |
| 11 | Purple Blotch | 847 |
| 12 | Rust | 213 |
| 13 | stemphylium Leaf Blight | 1,606 |
| 14 | Virosis-D | 512 |
| 15 | Xanthomonas Leaf Blight | 189 |

---

## Requirements

- Python **3.11** or **3.12**
- Windows 10 / 11
- At least **4GB RAM**
- Internet connection (for downloading pre-trained weights)

### Python Packages

```
tensorflow==2.20.0
numpy
matplotlib
scikit-learn
seaborn
Pillow
kaggle
```

---

## Installation

**Step 1 — Clone this repository**
```cmd
git clone https://github.com/dnjstr/plant-disease-detection.git
cd plant-disease-detection
```

**Step 2 — Create and activate virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```cmd
pip install -r requirements.txt
```

**Step 4 — Download the dataset**
1. Go to: https://www.kaggle.com/datasets/tejasbargujepatil/onion-diseases/data
2. Click **Download** (free Kaggle account required)
3. Extract the ZIP and rename the folder to `raw_dataset`
4. Place `raw_dataset/` inside the project folder

---

## How to Run

Run the scripts **in this order:**

**1. Visualize the dataset** *(optional but recommended)* — shows class imbalance before and after oversampling
```cmd
python visualize_dataset.py
```

**2. Train the model with Cross-Validation** — runs 5-fold stratified CV with oversampling, saves best model per fold
```cmd
python train_cv.py
```

**3. Compare architectures** *(optional)* — trains MobileNetV2, ResNet50V2, and DenseNet121 for comparison
```cmd
python compare_architectures.py
```

**4. Evaluate the model** — prints accuracy and saves confusion matrix
```cmd
python evaluate.py
```

**5. Predict a leaf image** — test with your own photo
```cmd
python predict.py --image path/to/leaf.jpg
```

Or predict a whole folder of images:
```cmd
python predict.py --folder path/to/images/
```

---

## Project Structure

```
plant-disease-detection/
│
├── raw_dataset/                    ← download from Kaggle (not in repo)
│   ├── Alternaria_D/
│   ├── Healthy leaves/
│   └── ... (15 classes)
│
├── cv_models/                      ← auto-created by train_cv.py
│   ├── model_fold_1.keras
│   ├── model_fold_2.keras
│   └── ... (one model per fold)
│
├── train_cv.py                     ← main training script (Stratified K-Fold + Oversampling)
├── compare_architectures.py        ← compares MobileNetV2, ResNet50V2, DenseNet121
├── visualize_dataset.py            ← dataset distribution plots
├── predict.py                      ← predict on new images
├── evaluate.py                     ← model evaluation + confusion matrix
├── get_stats.py                    ← utility to count images per class
├── requirements.txt                ← all dependencies
├── class_names.json                ← saved class labels
├── dataset_stats.json              ← per-class image counts
│
├── dataset_distribution.png        ← class imbalance plot (after visualize_dataset.py)
├── model_comparison.png            ← architecture comparison chart (after compare_architectures.py)
└── confusion_matrix.png            ← confusion matrix (after evaluate.py)
```

---

## Results

After training, the following output files are generated:

| File | Description |
|------|-------------|
| `cv_models/model_fold_N.keras` | Best model saved for each CV fold |
| `cv_results.json` | Fold accuracies, mean, and std deviation |
| `dataset_distribution.png` | Before/after class balance visualization |
| `model_comparison.png` | Accuracy and speed comparison across architectures |
| `confusion_matrix.png` | Per-class prediction performance |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named tensorflow` | Run `venv\Scripts\activate` first |
| `FileNotFoundError: raw_dataset` | Download and place the dataset folder as described in Installation |
| `Could not resolve host` during pip | Add `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| Out of memory during training | Change `BATCH_SIZE = 32` to `BATCH_SIZE = 16` in `train_cv.py` |
| Slow training | Normal on CPU (~30–60 min per fold). Set `EPOCHS = 5` in `train_cv.py` for a quick test |
| Wrong model loaded in evaluate.py | Edit `MODEL_PATH` in `evaluate.py` to point to the fold you want |

---

## License

This project is for educational purposes only.
Dataset credit: [Tejas Barguje Patil](https://www.kaggle.com/tejasbargujechef) on Kaggle.