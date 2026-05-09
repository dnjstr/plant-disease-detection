# 🔄 Stratified K-Fold Cross Validation

This explains why we didn't just do a normal 80/20 split.

## 1. Fulfilling Course Requirements
Our instructor explicitly requested **NOT** to use a static 70/30 or 80/20 split. 

**Why we use the `raw_dataset` directly:**
- By using the raw dataset, we avoid "locking" images into fixed folders.
- This allows our script to perform **Dynamic Splitting**, where the validation set is different for every fold. 
- This satisfies the requirement for **Cross-Validation** because every single image in the dataset eventually gets tested, rather than just a fixed 20% or 30%.

### 1.1 The "Full Data Utilization" Strategy (Presentation Defense)
If the instructor asks why we used the entire `raw_dataset` for training:
- **The Reason**: Our dataset had a severe imbalance (some diseases only had 7 images). 
- **The Benefit**: By using 5-Fold Cross-Validation on the *entire* dataset, we ensured the model was exposed to every rare example available while still being scientifically validated.
- **The Metric**: The final accuracy reported is the **Average Validation Accuracy** across all 5 folds. This is a standard and respected approach in academic research when dealing with limited or imbalanced data.

## 2. Why Cross-Validation?
- **Concept**: A single 80/20 split can be "lucky" or "unlucky." You might accidentally put all the "easy" images in the test set. 
- **K-Fold**: We split the data into 5 parts (Folds). We train 5 times. Each time, a different part is used for validation.
- **Result**: We take the average accuracy of all 5 runs. This is the **scientific** way to prove a model works.

## 3. Why "Stratified"?
- **Location**: `train_cv.py` (Line 110)
- **Concept**: `StratifiedKFold`.
- **Why?**: Because the dataset is imbalanced. Stratification ensures that each of the 5 folds has the exact same percentage of Healthy vs Sick images as the whole dataset. 
- **Example**: If 50% of the data is Healthy, stratification makes sure every fold is also 50% Healthy.

## 4. The Loop
- **Location**: `train_cv.py` (Lines 116-179)
- **What happens**:
    1. Split the data (Line 116).
    2. Balance the training fold (Oversampling) (Lines 125-146).
    3. Build a fresh model (Line 153).
    4. Train and save the best version of that fold (Lines 161-170).
