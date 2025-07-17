import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

# ====== Configuration ======
MODEL_PATH = 'cnn_model.h5'  # Change to 'vgg16_model.h5' if using VGG16
SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = 'output.avi'

# Define class labels (must match training)
class_labels = ['hugging', 'sitting', 'sleeping', 'using_laptop']

# ====== Load Model ======
try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Set input shape
if 'vgg16' in MODEL_PATH.lower():
    INPUT_SIZE = (224, 224)
    use_vgg = True
else:
    INPUT_SIZE = (128, 128)
    use_vgg = False

# ====== Preprocessing Function ======
def preprocess_frame(frame):
    """Resize and normalize frame based on model type"""
    resized = cv2.resize(frame, INPUT_SIZE)
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)

    return vgg_preprocess(img_array) if use_vgg else img_array / 255.0

# ====== Initialize Webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not accessible.")
    exit()

# ====== Initialize Video Writer if saving ======
out = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (640, 480))

print("[INFO] Webcam started. Press 'q' to quit.")
time.sleep(2.0)
fps_start = time.time()
frame_count = 0

# ====== Main Inference Loop ======
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    preprocessed = preprocess_frame(frame)

    try:
        predictions = model.predict(preprocessed)
        if predictions.shape[1] != len(class_labels):
            raise ValueError(f"Mismatch in predictions ({predictions.shape[1]}) and class labels ({len(class_labels)})")

        pred_index = np.argmax(predictions)
        pred_label = class_labels[pred_index]
        confidence = np.max(predictions)

        # Display prediction
        label_text = f"{pred_label} ({confidence*100:.2f}%)"
    except Exception as e:
        label_text = f"Prediction error: {e}"
        print(f"[ERROR] {e}")

    # Show label on frame
    cv2.putText(frame, label_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show FPS
    frame_count += 1
    elapsed_time = time.time() - fps_start
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Real-Time HAR", frame)

    # Save to output video
    if SAVE_VIDEO:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# ====== Cleanup ======
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()
