import cv2
from ultralytics import YOLO
import torch

# Load YOLOv8 object detection model (pretrained) for human detection
detector = YOLO("yolov8n.pt")  # or yolov8s.pt

# Load your trained classification model
classifier = YOLO("ActivityRecogYOLOv8/classification_v1/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("✅ Webcam running. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect humans in the frame
    detections = detector.predict(frame, conf=0.3, imgsz=640, classes=[0], device=device, verbose=False)
    
    # Convert frame for drawing
    output_frame = frame.copy()

    for box in detections[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop the detected person
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # Classify the cropped region (fall/normal)
        result = classifier.predict(person_crop, imgsz=224, device=device, verbose=False)
        label = result[0].names[result[0].probs.top1]
        confidence = result[0].probs.top1conf

        # Determine color for label
        color = (0, 0, 255) if label.lower() == "falling" else (0, 255, 0)
        label_text = f"{label}: {confidence*100:.1f}%"

        # Draw box and label
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show output
    cv2.imshow("Human Activity Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
