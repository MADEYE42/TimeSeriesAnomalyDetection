import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import time
import sys

# -----------------------------
# Configuration
# -----------------------------
MODEL_TYPE = "cnn_lstm"  # Options: "3dcnn", "cnn_lstm", "i3d"
MODEL_PATHS = {
    "3dcnn": "3d_cnn_model.h5",
    "cnn_lstm": "cnn_lstm_model.h5",
    "i3d": "final_i3d_model.h5"
}
SEQUENCE_LENGTH = 100
FRAME_HEIGHT, FRAME_WIDTH = 128, 128

# -----------------------------
# Activity Labels
# -----------------------------
LABELS = {
    0: 'Falling', 1: 'Jumping', 2: 'Lying', 3: 'Running',
    4: 'Sitting', 5: 'Standing', 6: 'Walking',
    7: 'Walking Downstairs', 8: 'Walking Upstairs'
}

# -----------------------------
# Load Model
# -----------------------------
if MODEL_TYPE not in MODEL_PATHS:
    sys.exit(f"[ERROR] Invalid MODEL_TYPE '{MODEL_TYPE}'. Choose from {list(MODEL_PATHS.keys())}")

print(f"[INFO] Loading model: {MODEL_PATHS[MODEL_TYPE]}")
model = load_model(MODEL_PATHS[MODEL_TYPE])

# -----------------------------
# Webcam Setup
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Unable to access webcam.")

print("[INFO] Webcam feed opened. Press 'q' to exit.")
buffer = deque(maxlen=SEQUENCE_LENGTH)
font = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------
# Inference Loop
# -----------------------------
fps_times = deque(maxlen=10)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame not captured.")
        break

    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    buffer.append(normalized)

    display_frame = frame.copy()

    # Perform prediction when buffer is full
    if len(buffer) == SEQUENCE_LENGTH:
        input_seq = np.array(buffer)

        if MODEL_TYPE == "cnn_lstm":
            input_seq = input_seq.reshape(1, SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3)
        else:  # 3D CNN or I3D
            input_seq = input_seq.reshape(1, SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3)

        preds = model.predict(input_seq, verbose=0)
        pred_label = LABELS[np.argmax(preds)]

        # Display prediction
        cv2.putText(display_frame, f'Activity: {pred_label}', (10, 30), font, 1, (0, 0, 255), 2)

    # Calculate and display FPS
    fps_times.append(time.time() - start_time)
    avg_fps = len(fps_times) / sum(fps_times) if fps_times else 0
    cv2.putText(display_frame, f'FPS: {avg_fps:.2f}', (10, 70), font, 0.7, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Activity Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting webcam feed.")
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
