# realtime_activity_pipeline.py
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque, Counter

# Load models
yolo = YOLO('yolov8n.pt')
activity_model = load_model('activity_classifier_resnet50.h5')

# Get input size from the model
input_shape = activity_model.input_shape
IMG_SIZE = (input_shape[1], input_shape[2])
print(f"[INFO] Model expects input shape: {IMG_SIZE}")

# Define label classes
ACTIVITY_CLASSES = ['fall', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking', 'walking_downstairs', 'walking_upstairs']

# Define queue to smooth predictions
prediction_queue = deque(maxlen=5)


# 🎥 Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Optional: mirror effect
    results = yolo(frame, classes=[0], conf=0.5)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Margin padding to capture more context
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Preprocess
            resized = cv2.resize(person_crop, IMG_SIZE)
            norm = resized.astype("float32") / 255.0
            input_tensor = np.expand_dims(norm, axis=0)

            # Predict activity
            activity_probs = activity_model.predict(input_tensor, verbose=0)
            activity_idx = int(np.argmax(activity_probs))
            prediction_queue.append(activity_idx)

            # Smoothing
            most_common_idx = Counter(prediction_queue).most_common(1)[0][0]
            label = ACTIVITY_CLASSES[most_common_idx]
            confidence = activity_probs[0][activity_idx]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 + Activity Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
