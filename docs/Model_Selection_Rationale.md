# 🏆 Model Selection Rationale

This document explains why we chose **MobileNetV2** as our final model despite ResNet having a slightly higher accuracy.

## 📊 Comparison Results (at 5 Epochs)

| Architecture | Val Accuracy | Training Time | Efficiency (Acc/Time) |
|--------------|--------------|---------------|-----------------------|
| **MobileNet**| **84.77%**   | **3,392s**    | ⭐ **Highest**        |
| ResNet       | 86.66%       | 6,891s        | Medium                |
| DenseNet     | 85.71%       | 10,731s       | Low                   |

---

## 🧐 Why we chose MobileNetV2

### 1. Efficiency vs. Performance
- **The Gap**: ResNet is only **1.89%** more accurate than MobileNet.
- **The Cost**: ResNet takes **2x longer** to train, and DenseNet takes **3x longer**. 
- **Conclusion**: MobileNet provides the best "Bang for your Buck." It achieves nearly the same results in a fraction of the time.

### 2. Real-World Use Case (Edge Computing)
- **Concept**: Plant disease detection is meant to be used by farmers in the field.
- **Rationale**: MobileNetV2 is an "Edge-optimized" architecture. It has fewer parameters and a smaller file size, making it perfect for **Mobile Apps** or **IoT devices**. ResNet and DenseNet are "heavyweight" models that would lag or crash on a standard smartphone.

### 3. Inference Latency (User Experience)
- **Training vs. Prediction**: Training is like "studying" (it takes hours), but prediction is like "recalling" (it take seconds).
- **The Mobile Factor**: Even though prediction is fast, heavy models like ResNet require more battery power and CPU memory. MobileNet uses **Depthwise Separable Convolutions** to reduce the number of mathematical operations per image.
- **Result**: Using MobileNet ensures the app feels "snappy" and responsive, providing an instant diagnosis without draining the farmer's phone battery.

### 4. Sustainability
- **Rationale**: Faster training means less electricity and fewer computational resources. By choosing the more efficient model, we are following **Green AI** principles—optimizing for performance without wasting energy.

---

## 🎤 Presentation "Mic Drop" Defense
> *"While ResNet50V2 achieved the highest raw accuracy, we selected **MobileNetV2** as our primary architecture. It delivers a highly competitive 84.7% accuracy while being 2-3 times more efficient. In a real-world agricultural setting, speed and low resource usage are more critical than a 2% accuracy gain."*
