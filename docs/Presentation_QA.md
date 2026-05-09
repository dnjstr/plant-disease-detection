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
**A:** "Softmax turns the raw output numbers into probabilities that sum up to 1 (100%). It's the standard for multi-class classification because it tells us the confidence level for each disease."

### Q5: "What is 'EarlyStopping'?"
**A:** "It's a safety feature. If the model's performance on the validation set stops improving for 3 rounds (patience=3), we stop training automatically. This saves time and prevents the model from starting to 'memorize' the training data."
