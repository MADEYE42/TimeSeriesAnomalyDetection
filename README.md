# 🧠 Real-Time Human Activity Detection Using EfficientNet

This project performs real-time human activity recognition from webcam video using a fine-tuned deep learning model (EfficientNetB0 or MobileNetV2). Activities are classified frame-by-frame with predictions shown on-screen and logged in the terminal with timestamps.

---

## 📂 Project Structure

```

📁/
│
├── X.npy                  # Preprocessed image data (shape: N x 224 x 224 x 3)
├── y.npy                  # Corresponding activity labels
├── EfficientNetB0\_best\_model.h5   # Trained Keras model (EfficientNetB0)
├── activity\_detection.py  # Real-time webcam inference script
├── training\_script.py     # Model training script (EfficientNet / MobileNetV2)
├── activity\_model.tflite  # Exported TensorFlow Lite model (for mobile/edge)
├── activity\_model.onnx    # Exported ONNX model (for ONNX runtime)
└── README.md

````

---

## 📸 Activities Covered

- Falling  
- Jumping  
- Lying  
- Running  
- Shaking  
- Sitting  
- Standing  
- Turning In Place  
- Walking  
- Walking Downstairs  
- Walking Upstairs  

---

## 🚀 Getting Started

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/MADEYE42/activity-detection-project.git
````

### ✅ 2. Install Dependencies

```bash
pip install tensorflow opencv-python numpy
```

> Make sure you’re using **Python 3.10** for TensorFlow compatibility.

### ✅ 3. Run Real-Time Detection

```bash
python activity_detection.py
```

> This will open your webcam, display detected activity labels on screen, and log them in the terminal.

---

## 🏋️‍♂️ Model Training

To train your own model using extracted and labeled frames:

```bash
python training_script.py
```

* Uses transfer learning with `EfficientNetB0` (default) or `MobileNetV2`
* Automatically splits data into train/validation
* Outputs best model as `EfficientNetB0_best_model.h5`

---

## 🧠 Example Terminal Output

```
📹 Webcam started. Press 'q' to quit.
[2025-07-06 20:08:43] 🧠 Detected: Walking (94.2%)
[2025-07-06 20:08:48] 🧠 Detected: Running (96.8%)
[2025-07-06 20:08:52] 🧠 Detected: Falling (91.5%)
✅ Webcam closed. Session ended.
```

---

## 🛡️ Future Improvements

* Add alert for dangerous activities like Falling
* Upload detection logs to cloud or dashboard
* Convert to mobile app using TensorFlow Lite
* Add tracking and sequence analysis (LSTM/CNN-LSTM)

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Gouresh Madye**
📧 [goureshmadye@example.com](mailto:goureshmadye@example.com)
📌 [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/your-username)


