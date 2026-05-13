# 🎤 Presentation Q&A (Cheat Sheet)

Be prepared! These are the questions instructors love to ask.

---

### Q1: "Why did you use Transfer Learning instead of building your own CNN?"
**A:** "Transfer learning allows us to leverage high-level features (like edges, textures, and shapes) already learned by models trained on millions of images. Building from scratch would require a much larger dataset and significantly more training time to reach the same accuracy."

### Q2: "What is 'Dropout' and why is it 0.5?"
**A:** "Dropout is a regularization technique. By 'dropping' 50% of the neurons during each training step, we force the model to find multiple paths to the correct answer. This prevents the model from relying too heavily on specific pixels (Overfitting)."

### Q3: "What happens if you don't oversample?"
**A:** "The model will become biased. Since 'Healthy' images are the majority, the model will learn that it can get a high accuracy just by guessing 'Healthy' every time. Oversampling forces the model to treat every disease as equally important."

### Q4: "Why use Softmax for the last layer?"
**A:** "The model's internal layers produce raw numbers (logits) that are hard to interpret. Softmax acts as a 'probability converter.' It squashes these raw numbers into a range between 0 and 1, ensuring that the sum of all 15 classes equals 100%. This allows us to provide a human-readable 'Confidence Score' (e.g., 95% certainty), which is essential for a diagnostic tool."

### Q5: "What is 'EarlyStopping'?"
**A:** "It's a safety feature. If the model's performance on the validation set stops improving for 3 rounds (patience=3), we stop training automatically. This saves time and prevents the model from starting to 'memorize' the training data."

### Q6: "What does 'Stratified' mean in your K-Fold?"
**A:** "Stratification ensures that each fold is a 'miniature version' of the whole dataset. Since our data is imbalanced (lots of healthy, few sick), stratification ensures every fold has the same percentage of diseases. Without it, a fold might accidentally contain zero examples of a rare disease."

### Q7: "Why did you choose 5 Folds instead of 10?"
**A:** "5-Fold is the standard benchmark for academic research. It provides a good balance between statistical reliability and computational time. Training 10 folds would take twice as long without significantly changing the average accuracy."

### Q8: "Why MobileNetV2 if ResNet was more accurate?"
**A:** "We prioritized **Efficiency and Deployment**. MobileNetV2 is designed for mobile/edge devices. While ResNet was 1.9% more accurate, it was 2x slower and significantly heavier. For a real-world application in a farmer's field, the speed and low battery usage of MobileNet are more important than a tiny gain in accuracy."
