# 🖐️ Sign Language Detection using Deep Learning  

This project implements **American Sign Language (ASL) recognition** using **TensorFlow/Keras** with **MobileNetV2** as the backbone.  
It can be trained on custom-collected images (via webcam or dataset) and then used for real-time predictions.  

---

## 📌 Features  
- Deep learning-based gesture recognition.  
- Transfer learning with **MobileNetV2**.  
- Data augmentation with Keras `ImageDataGenerator`.  
- Handles class imbalance using `class_weight`.  
- Three-phase training:  
  1. Train classifier head.  
  2. Fine-tune last layers.  
  3. Fine-tune full backbone.  
- Saves the **best model** automatically (`best_asl_model.h5`).  

---


---

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/sign-language-detection-1.git
   cd sign-language-detection-1


