Here is a well-structured and formal **README.md** based on your comprehensive document:

---

# 🧠 Real-Time Human Activity Recognition (HAR)

A comprehensive deep learning-based system for real-time **Human Activity Recognition** (HAR) using image classification, video analysis, object detection, and spatiotemporal models.

---

## 📁 Project Overview

This project explores four distinct approaches to HAR, combining classical CNNs, temporal sequence models, and YOLOv8-based pipelines. The system is capable of real-time activity recognition from webcam feeds and supports deployment-ready architectures for smart monitoring, safety, and surveillance applications.

---

## ✅ Approaches

### **🔹 Approach 1: Video-to-Image Frame Based Pipeline**

* Extracted frames from activity videos (e.g., walking, sitting, falling) at 1 FPS.
* Applied CNN-based transfer learning models (EfficientNetB0, MobileNetV2, ResNet50).
* Trained with one-hot encoded labels and image inputs.
* Real-time deployment using OpenCV with confidence thresholds and logging.

### **🔹 Approach 2: Image-Only HAR with CSV Label Mapping**

* Used `Training_set.csv` and `Testing_set.csv` for mapping image filenames to labels.
* Trained various models (EfficientNetB0, MobileNetV2, VGG16, custom CNNs).
* Webcam integration with `.h5`/`.keras` models for real-time inference.
* Supported bounding box overlays, alert generation, and modular deployment.

### **🔹 Approach 3: Spatiotemporal HAR Using 3D CNN, CNN+LSTM, and I3D**

* Segmented videos into 16/64-frame clips to capture temporal motion patterns.
* Trained three architectures:

  * **3D CNN**
  * **CNN + LSTM**
  * **I3D (Inflated 3D ConvNet)**
* Performed clip-wise prediction with ensemble support for higher robustness.

### **🔹 Approach 4: YOLOv8-Based HAR with Object Detection Integration**

* Used Ultralytics YOLOv8 (`yolov8n-cls.pt`) for activity classification.
* Extracted frames from fall/normal videos into `train/`, `val/`, and `test/` subfolders.
* Used `yolov8n.pt` for detecting human presence to improve prediction reliability.
* Implemented real-time webcam inference with bounding boxes and fall detection alerts.

---

## 🧪 Features

* ✅ Real-time activity detection using webcam input.
* 🧠 Multi-model architecture supporting both 2D and 3D convolutional approaches.
* 📦 Dataset frame extraction, labeling, and augmentation support.
* 📈 Evaluation with confusion matrix, precision, recall, F1-score.
* 📊 Live prediction visualization and object tracking.
* 🔔 Extensible alert/notification system for fall detection use-cases.

---

## 🚀 Getting Started

### 📦 Installation

Install all dependencies via:

```bash
pip install -r requirements.txt
```

### 📁 Dataset Structure (for Approach 4)

```
dataset/
├── train/
│   ├── Fall/
│   └── Normal/
├── val/
│   ├── Fall/
│   └── Normal/
└── test/
    ├── Fall/
    └── Normal/
```

---

## 🎯 Usage

1. Train models via respective notebooks/scripts per approach.
2. Ensure webcam access is available for real-time predictions.
3. Run the live inference script to visualize and test predictions.

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Precision / Recall**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

---

## 🛠️ Technologies Used

* Python, OpenCV, TensorFlow/Keras, PyTorch
* Ultralytics YOLOv8
* Scikit-learn, NumPy, Matplotlib
* EfficientNet, MobileNetV2, VGG16, ResNet50
* I3D, 3D CNN, LSTM (Keras)

---

## 🔧 Future Work

* 🚨 Add multi-class fall detection (slow, backward, sideways)
* 🧊 Deploy on edge devices (Jetson, Raspberry Pi)
* ☁️ Integrate alert system (email/SMS/IoT)
* 🧪 Ensemble prediction logic for robust multi-model performance

---

