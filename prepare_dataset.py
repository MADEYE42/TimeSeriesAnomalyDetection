import os
import cv2
import shutil
import random
from pathlib import Path

# --- Configuration ---
VIDEO_DIRS = {
    'fall': 'Activities/Falling',
    'walk': 'Activities/Walking',
    'run': 'Activities/Running',
    'sit': 'Activities/Sitting',
    'stand': 'Activities/Standing',
    'exercise': 'Activities/Exercise',
    'lying': 'Activities/Lying',
    'walking_downstairs': 'Activities/Walking Downstairs',
    'walking_upstairs': 'Activities/Walking Upstairs',
}
FRAME_RATE = 1  # frames per second
OUTPUT_DIR = 'dataset'
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
RANDOM_SEED = 42
IMG_EXT = ".jpg"

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def extract_frames_and_annotate(video_path, label, output_folder, rate=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // rate) if fps >= rate else 1

    count = 0
    saved = 0
    label_folder = output_folder / label / "images"
    annot_folder = output_folder / label / "labels"
    label_folder.mkdir(parents=True, exist_ok=True)
    annot_folder.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = f"{video_path.stem}_{saved}"
            image_path = label_folder / f"{filename}{IMG_EXT}"
            label_path = annot_folder / f"{filename}.txt"

            # Detect humans using HOG
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))

            height, width = frame.shape[:2]
            annotations = []

            for (x, y, w, h) in boxes:
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                bw = w / width
                bh = h / height
                # YOLO format: class_id cx cy bw bh
                annotations.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if annotations:
                cv2.imwrite(str(image_path), frame)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
                saved += 1

        count += 1

    cap.release()

def prepare_dirs():
    for split in SPLIT_RATIOS:
        for cls in VIDEO_DIRS:
            Path(OUTPUT_DIR, split, cls, "images").mkdir(parents=True, exist_ok=True)
            Path(OUTPUT_DIR, split, cls, "labels").mkdir(parents=True, exist_ok=True)
    for cls in VIDEO_DIRS:
        Path('temp_frames', cls, "images").mkdir(parents=True, exist_ok=True)
        Path('temp_frames', cls, "labels").mkdir(parents=True, exist_ok=True)

def extract_all_frames():
    print("[INFO] Extracting and annotating frames...")
    for label, folder in VIDEO_DIRS.items():
        video_files = list(Path(folder).glob("*.mp4")) + list(Path(folder).glob("*.avi"))
        for video_file in video_files:
            extract_frames_and_annotate(video_file, label, Path('temp_frames'), FRAME_RATE)
    print("[INFO] Frame extraction and annotation completed.")

def split_dataset():
    print("[INFO] Splitting dataset...")
    random.seed(RANDOM_SEED)
    for label in VIDEO_DIRS:
        images = list(Path('temp_frames', label, "images").glob("*" + IMG_EXT))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS['train'])
        n_val = int(n * SPLIT_RATIOS['val'])

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, img_list in splits.items():
            for img_path in img_list:
                txt_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
                if not txt_path.exists():
                    continue  # Skip images with no annotations
                dest_img = Path(OUTPUT_DIR, split, label, "images", img_path.name)
                dest_lbl = Path(OUTPUT_DIR, split, label, "labels", txt_path.name)
                shutil.move(str(img_path), dest_img)
                shutil.move(str(txt_path), dest_lbl)

    shutil.rmtree('temp_frames')
    print("[INFO] Dataset split and saved in YOLO format under 'dataset/'")

def main():
    prepare_dirs()
    extract_all_frames()
    split_dataset()
    print("[SUCCESS] YOLO object detection dataset is ready!")

if __name__ == "__main__":
    main()
