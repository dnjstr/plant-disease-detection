# 🔄 Stratified K-Fold Cross Validation

This explains why we didn't just do a normal 80/20 split.

## 1. Why Cross-Validation?
- **Concept**: A single 80/20 split can be "lucky" or "unlucky." You might accidentally put all the "easy" images in the test set. 
- **K-Fold**: We split the data into 5 parts (Folds). We train 5 times. Each time, a different part is used for validation.
- **Result**: We take the average accuracy of all 5 runs. This is the **scientific** way to prove a model works.

## 2. Why "Stratified"?
- **Location**: `train_cv.py` (Line 110)
- **Concept**: `StratifiedKFold`.
- **Why?**: Because our dataset is imbalanced. Stratification ensures that each of the 5 folds has the exact same percentage of Healthy vs Sick images as the whole dataset. 
- **Example**: If 50% of your data is Healthy, stratification makes sure every fold is also 50% Healthy.

## 3. The Loop
- **Location**: `train_cv.py` (Lines 116-179)
- **What happens**:
    1. Split the data (Line 116).
    2. Balance the training fold (Oversampling) (Lines 125-146).
    3. Build a fresh model (Line 153).
    4. Train and save the best version of that fold (Lines 161-170).
