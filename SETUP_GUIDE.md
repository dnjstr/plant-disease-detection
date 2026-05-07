# 🌿 Plant Disease Detection Project Guide
*TensorFlow 2.20.0 | Python 3.11 / 3.12 | Stratified K-Fold CV*

This project uses Deep Learning (MobileNetV2) to detect diseases in onion plants. It is optimized for imbalanced datasets using **Stratified K-Fold Cross-Validation** and **Oversampling**.

---

## 🛠️ 1. Setup & Installation

### Create Virtual Environment
Open your terminal in the project folder and run:
```powershell
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 2. Dataset Visualization
Before training, you can visualize the class imbalance and how the balancing strategy (oversampling) works.
```powershell
python visualize_dataset.py
```
- **What it does**: Generates `dataset_distribution.png`.
- **Why?**: Shows the original imbalance (e.g., 3,000 Healthy vs 7 Bulb Rot) and the balanced state used for training.

---

## 🧠 3. Training (Stratified Cross-Validation)
We use **Stratified K-Fold** to ensure the model is evaluated fairly across different subsets of data.

```powershell
python train_cv.py
```
- **Inputs**: Reads directly from the `raw_dataset/` folder.
- **Balancing**: Automatically oversamples minority classes (duplicates rare images) so the model learns them properly.
- **Outputs**: Saves the best model for each fold in the `cv_models/` directory.

> [!TIP]
> If your computer is slow, you can open `train_cv.py` and reduce `EPOCHS = 10` to `EPOCHS = 5`.

---

## 🚀 4. Multi-Model Comparison (Instructor's Request)
Requirement of comparing at least 3 models, we can run the following script to compare different architectures.

```powershell
python compare_architectures.py
```
- **Architectures Compared**: MobileNetV2, ResNet50V2, and DenseNet121.
- **What it does**: Trains all three models for a few epochs and compares their accuracy and speed.
- **Output**: Generates `model_comparison.png` (a grouped bar chart for the report).

---

## 🔍 5. Evaluation & Prediction

### Evaluate the Model
To see the accuracy and a confusion matrix:
```powershell
python evaluate.py
```

### Predict a Single Image
Test the model on any new photo of a leaf:
```powershell
python predict.py --image "path/to/your/image.jpg"
```

---

## 📂 Project Structure
- `raw_dataset/`: Original images organized by folder (Class Name).
- `cv_models/`: Contains the trained models for each cross-validation fold.
- `train_cv.py`: Main training script with Stratified K-Fold and Oversampling.
- `compare_architectures.py`: Compares MobileNet, ResNet, and DenseNet.
- `visualize_dataset.py`: Generates the distribution plots.
- `predict.py`: Script for real-world testing.

---

## ⚠️ Common Troubleshooting

| Error | Solution |
| :--- | :--- |
| **"No module named tensorflow"** | Ensure you activated the environment: `venv\Scripts\activate`. |
| **Out of Memory (OOM)** | In `train_cv.py`, change `BATCH_SIZE = 32` to `16`. |
| **Slow Training** | This is normal on CPUs. It can take 30-60 mins. Use a GPU (NVIDIA) if available. |
| **Network Error during Pip** | Use: `pip install tensorflow==2.20.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org` |