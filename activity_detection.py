import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================================
# ✅ Configuration
# ================================
MODEL_PATH = 'EfficientNetB0_best_model.h5'  # Make sure this is correct
CONFIDENCE_THRESHOLD = 0.60                  # Only log if confidence is above 60%
FRAME_WIDTH = 224
FRAME_HEIGHT = 224

class_names = [
    'Falling', 'Jumping', 'Lying', 'Running', 'Shaking',
    'Sitting', 'Standing', 'Turning In Place',
    'Walking', 'Walking Downstairs', 'Walking Upstairs'
]

# ================================
# ✅ Load Model
# ================================
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ================================
# ✅ Start Webcam
# ================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to access webcam.")
    exit()

print("📹 Webcam started. Press 'q' to quit.")

# ================================
# ✅ Activity Detection Loop
# ================================
prev_activity = ""
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Resize and preprocess
    img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]
    activity = class_names[class_id]

    # Display if confident
    if confidence >= CONFIDENCE_THRESHOLD:
        display_text = f"{activity} ({confidence * 100:.1f}%)"
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        # Log if activity changes
        if activity != prev_activity:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 🧠 Detected: {activity} ({confidence * 100:.1f}%)")
            prev_activity = activity
    else:
        cv2.putText(frame, "Uncertain...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("🔍 Real-Time Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. Session ended.")
