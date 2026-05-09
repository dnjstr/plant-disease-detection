# 🧠 Model Architecture (Transfer Learning)

This explains the "Brain" of the project.

## 1. The Base Model (The "Experienced" Brain)
- **Location**: `train_cv.py` (Lines 79-87)
- **Concept**: We use **Transfer Learning**. Instead of starting from scratch, we use models (MobileNetV2, ResNet, DenseNet) that already know how to "see" shapes and colors from the ImageNet dataset.
- **Key Line**: `base_model.trainable = False` (Line 89). We "freeze" the base so we don't destroy its existing knowledge while training our new layer.

## 2. Preprocessing
- **Location**: `train_cv.py` (Line 92)
- **Concept**: `Rescaling(1.0 / 255)`. 
- **Why?**: Images are made of pixels from 0 to 255. Neural networks learn much faster when numbers are between 0 and 1.

## 3. The Custom "Head"
- **Location**: `train_cv.py` (Lines 94-98)
- **GlobalAveragePooling2D**: Simplifies the complex 3D data from the base model into a simple 1D list of numbers.
- **BatchNormalization**: Stabilizes the learning process (makes training faster).
- **Dense(256, activation="relu")**: A fully connected layer that learns the specific features of onion diseases.
- **Dropout(0.5)**: Randomly ignores 50% of the neurons. 
    - **Why?**: This prevents **Overfitting** (memorizing the specific training photos instead of learning general patterns).
- **Dense(num_classes, activation="softmax")**: The final output. Softmax gives us a probability (e.g., "98% sure this is Purple Blotch").
