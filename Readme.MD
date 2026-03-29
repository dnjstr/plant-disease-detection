#  Plant Disease Detection from Leaf Images

A machine learning project that detects onion plant diseases from leaf photos using **MobileNetV2** transfer learning.

> **Course Project** — Machine Learning  
> **Dataset:** [Onion Diseases – Kaggle (Tejas Barguje Patil)](https://www.kaggle.com/datasets/tejasbargujechef/onion-diseases)

---

##  Table of Contents
- [About the Project](#about-the-project)
- [Disease Classes](#disease-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Results](#results)

---

##  About the Project

This project uses a pre-trained **MobileNetV2** convolutional neural network (CNN) with transfer learning to classify onion leaf images into 15 disease categories. The model is trained in two phases:

1. **Phase 1** — Train only the custom classification head (base frozen)
2. **Phase 2** — Fine-tune the last 30 layers of MobileNetV2

**Tech Stack:**
- Python 3.11 / 3.12
- TensorFlow 2.20.0 / Keras 3
- MobileNetV2 (pre-trained on ImageNet)

---

## Disease Classes

| # | Class |
|---|-------|
| 1 | Alternaria_D |
| 2 | Botrytis Leaf Blight |
| 3 | Bulb Rot |
| 4 | Bulb_blight-D |
| 5 | Caterpillar-P |
| 6 | Downy Mildew |
| 7 | Fusarium-D |
| 8 | Healthy Leaves |
| 9 | Iris Yellow Virus Augment |
| 10 | Onion1 |
| 11 | Purple Blotch |
| 12 | Rust |
| 13 | Stemphylium Leaf Blight |
| 14 | Virosis-D |
| 15 | Xanthomonas Leaf Blight |

---

## Requirements

- Python **3.11** or **3.12**
- Windows 10 / 11
- At least **4GB RAM**
- Internet connection (for downloading pre-trained MobileNetV2 weights)

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
1. Go to: https://www.kaggle.com/datasets/tejasbargujechef/onion-diseases
2. Click **Download** (free Kaggle account required)
3. Extract the ZIP and rename the folder to `raw_dataset`
4. Place `raw_dataset/` inside the project folder

---

##  How to Run

Run the scripts **in this order:**

**1. Prepare the dataset** — splits images into train / val / test
```cmd
python prepare_dataset.py
```

**2. Train the model** — trains MobileNetV2, saves model + class names
```cmd
python train.py
```

**3. Evaluate the model** — prints accuracy and saves confusion matrix
```cmd
python evaluate.py
```

**4. Predict a leaf image** — test with your own photo
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
├── raw_dataset/               ← download from Kaggle (not in repo)
│   ├── Alternaria_D/
│   ├── Healthy leaves/
│   └── ... (15 classes)
│
├── dataset/                   ← auto-created by prepare_dataset.py
│   ├── train/
│   ├── val/
│   └── test/
│
├── train.py                   ← model training script
├── predict.py                 ← predict on new images
├── prepare_dataset.py         ← dataset splitting script
├── evaluate.py                ← model evaluation + confusion matrix
├── requirements.txt           ← all dependencies
├── README.md                  ← this file
│
├── plant_disease_model.keras  ← saved model (after training)
├── class_names.json           ← saved class labels (after training)
├── training_history.png       ← accuracy/loss plot (after training)
└── confusion_matrix.png       ← confusion matrix (after evaluation)
```

---

## Results

After training, the following output files are generated:

| File | Description |
|------|-------------|
| `plant_disease_model.keras` | Trained model file |
| `training_history.png` | Accuracy and loss curves |
| `confusion_matrix.png` | Per-class prediction performance |
| `class_names.json` | List of disease class names |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named tensorflow` | Run `venv\Scripts\activate` first |
| `FileNotFoundError: dataset\train` | Run `prepare_dataset.py` before `train.py` |
| `Could not resolve host` during pip | Add `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| Out of memory during training | Change `BATCH_SIZE = 32` to `BATCH_SIZE = 16` in `train.py` |
| Slow training | Normal on CPU (~30–60 min). Set `EPOCHS = 5` for a quick test. |

---

## License

This project is for educational purposes only.  
Dataset credit: [Tejas Barguje Patil](https://www.kaggle.com/tejasbargujechef) on Kaggle.