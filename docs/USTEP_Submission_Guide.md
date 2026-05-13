# 📝 USTEP Submission & Presentation Guide

Follow this structure for your documentation and slides. 

---

## 1. Overview of your Project
This project presents an automated machine learning system specifically designed to assist farmers in the early detection and diagnosis of onion leaf diseases. Leveraging a robust technology stack including Python, TensorFlow, and Keras, the system utilizes Deep Transfer Learning with the MobileNetV2 architecture. A key innovation of this work is the implementation of Stratified K-Fold Cross-Validation combined with Dynamic Oversampling, which effectively addresses the severe class imbalance present in agricultural datasets, ensuring high diagnostic accuracy even for rare disease variants.

## 2. Objective(s)
### General Objective:
The general objective of this project is to develop a robust and efficient machine learning system for the automated classification of onion leaf diseases, providing farmers with a reliable tool for precision agriculture and early pathogen diagnosis.

### Specific Objectives:
1. **Systematic Model Comparison**: To implement and compare multiple Deep Transfer Learning architectures (MobileNetV2, ResNet50V2, and DenseNet121) to determine the most effective model for agricultural disease classification.
2. **Mitigating Data Imbalance**: To solve the severe dataset imbalance problem through **Dynamic Oversampling** of minority classes, ensuring the system can accurately detect all 15 types of onion diseases.
3. **Scientific Validation**: To establish a rigorous evaluation framework using **5-Fold Stratified Cross-Validation**, ensuring the reported accuracy is reliable and consistent across different data subsets.
4. **Optimization for Deployment**: To optimize the training process using **EarlyStopping** and **ModelCheckpoint** to ensure the best model weights are saved while preventing overfitting.
5. **Efficiency Analysis**: To analyze the trade-off between diagnostic accuracy and training/inference time, prioritizing models that are efficient enough for real-world mobile deployment.

## 3. Data Collection
- **Source**: Kaggle Dataset (Onion Diseases by Tejas Barguje Patil).
- **Setup**: The dataset was organized into a **Training Pool** (merging train/val sets) and a separate **Hold-out Test Set** (10-20% of data) to ensure unbiased evaluation.
- **📸 Screenshot to include**: A screenshot of your `cv_data/` folder and your separate `test_data/` folder.

## 4. Data Preprocessing
- **Hold-out Isolation**: Removing test data from the training pool to prevent data leakage.
- **Stratified Splitting**: Ensuring every fold maintains the correct disease ratios.
- **Dynamic Oversampling**: Duplicating minority class images during training to solve the imbalance.
- **📸 Screenshot to include**: 
    1. `dataset_distribution.png` (The "Before vs After" bar chart).
    2. Code from `train_cv.py` (Lines 125-146) — the Oversampling logic.

## 5. Modeling
The modeling phase of this project utilizes Deep Transfer Learning to leverage pre-trained visual features from the ImageNet dataset. To identify the optimal solution for onion disease detection, three distinct architectures were implemented for comparative analysis: MobileNetV2, ResNet50V2, and DenseNet121. MobileNetV2 was selected for its efficiency and "Depthwise Separable Convolutions," making it suitable for resource-constrained mobile deployment. ResNet50V2 was chosen for its "Residual Learning" capabilities, which use skip connections to maintain accuracy in deeper networks. Finally, DenseNet121 was integrated due to its "Dense Connectivity," which promotes feature reuse and high parameter efficiency across layers. This multi-model approach allows for a rigorous evaluation of the trade-offs between diagnostic precision and computational latency. 

**Optimization Strategy**: The training process incorporates layer freezing for feature extraction and is managed by core callbacks, including **EarlyStopping** to prevent overfitting and **ModelCheckpoint** to ensure the preservation of the highest-performing weights from each cross-validation fold.

- **📸 Screenshot to include**: 
    1. Code from `train_cv.py` (Lines 79-87) — the `build_model` architecture logic.
    2. Code from `train_cv.py` (Lines 156-160) — the `callbacks` configuration.

## 6. Evaluation
The evaluation phase involved a rigorous comparison of the three trained models across two primary dimensions: diagnostic accuracy and computational efficiency. According to the results, **ResNet50V2** achieved the highest validation accuracy at **86.66%**, followed by **DenseNet121** at **85.71%**, and **MobileNetV2** at **84.77%**. 

However, the training time analysis revealed significant disparities in resource consumption. **MobileNetV2** was the most efficient, completing training in just **3,392 seconds**, which is approximately **2x faster than ResNet** and **3.2x faster than DenseNet**. While ResNet offered a marginal 1.89% increase in accuracy over MobileNet, the computational cost (training time and energy usage) was doubled. Consequently, MobileNetV2 was selected as the final model for this project, as it provides the most balanced trade-off between high accuracy and the low-latency requirements necessary for real-world agricultural deployment.

- **📸 Screenshot to include**: 
    1. `model_comparison.png` (The Validation Accuracy and Training Time bar charts).
    2. `confusion_matrix.png` (Showing the True vs Predicted labels for each disease).

## 7. Results
- **Sample Output**: Show a successful prediction on a new image.
- **Discussion**: The model achieves high confidence (>90%) on most common diseases and correctly identifies rare diseases thanks to the oversampling strategy.
- **📸 Screenshot to include**: A screenshot of your terminal after running `python predict.py --image test.jpg`.
