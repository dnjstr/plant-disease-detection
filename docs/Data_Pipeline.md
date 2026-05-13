# 📊 Data Pipeline & Balancing

This section explains how the images are prepared for the model.

## 1. Finding the Data
- **Location**: `train_cv.py` (Lines 30-46)
- **What happens**: The script walks through the `raw_dataset` folder, maps every image path to a category name, and stores them in a massive list. 
- **Key Line**: `class_names = sorted(...)` (Line 33) ensures the classes are always in the same alphabetical order.

## 2. Dataset Balancing (Oversampling)
- **Location**: `train_cv.py` (Lines 122-146)
- **The Problem**: Some classes had 3,000 images, others had only 7.
- **The Solution**: **Dynamic Oversampling**. 
    - We calculate `max_count = max(counts)` (Line 123) which is the size of the biggest class (Healthy).
    - We then duplicate the paths of the smaller classes (Lines 131-142) until they match the biggest class.
- **Why?**: This prevents the model from being biased toward the "Healthy" class.

## 3. Data Augmentation
- **Location**: `train_cv.py` (Lines 64-72)
- **What it does**: Randomly flips, rotates, and zooms the images during training.
- **Why?**: It makes the model more robust. It's like teaching the model to recognize a leaf even if it's seen from a different angle or lighting.
