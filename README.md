# 🧠 Real-Time Human Activity Detection Using EfficientNet

This project performs real-time human activity recognition from webcam video using a fine-tuned deep learning model (EfficientNetB0 or MobileNetV2). Activities are classified frame-by-frame with predictions shown on-screen and logged in the terminal with timestamps.

---

## 📂 Project Structure

```

📁/
│
├── X.npy                  # Preprocessed image data (shape: N x 224 x 224 x 3)
├── y.npy                  # Corresponding activity labels
├── EfficientNetB0_best_model.h5   # Trained Keras model (EfficientNetB0)
├── activity_detection.py  # Real-time webcam inference script
├── CNN.ipynb     # Model training script (EfficientNet / MobileNetV2)
├── VideoToFrames.ipynb  # Exported Frames from videos
├── requirements.txt  # Libraries
├── preprocess_frame.ipynb  # To preprocess frames and make all of them in same size
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
git clone https://github.com/MADEYE42/TimeSeriesAnomalyDetection.git
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
run CNN.ipynb
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
