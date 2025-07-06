
---

### 📄 Annotated Frame Extraction Process

This project extracts **1 frame per second** from activity videos (e.g., Walking, Running, Falling) and saves each frame with the activity name annotated both **on the image** and in the **file name**, organized in folders for each activity.

---

#### 📁 Folder Structure

```
/Activities/
├── Falling/
│   ├── fall1.mp4
│   ├── fall2.mp4
│   └── ...
├── Walking/
│   ├── walk1.mp4
│   └── ...
├── Running/
│   └── run1.mp4
...
```

---

#### ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```txt
opencv-python
```

---

#### ▶️ How It Works

* Reads each video from every activity folder.
* Extracts frames at **1-second intervals** using OpenCV.
* Annotates each frame with the **activity label** (e.g., `WALKING`) on the image.
* Saves each annotated frame as a `.jpg` inside:

  ```
  /annotated_frames/<ActivityName>/<activity_name>_<video_name>_<timestamp>.jpg
  ```

---

#### 📂 Output Example

```
/annotated_frames/
├── Falling/
│   ├── falling_fall1_0001.jpg
│   ├── falling_fall1_0002.jpg
│   └── ...
├── Walking/
│   ├── walking_walk1_0001.jpg
│   └── ...
...
```

Each image contains the label rendered in red text on the top-left corner.

---

#### 🧠 Activities Supported

* Falling
* Jumping
* Lying
* Running
* Shaking
* Sitting
* Standing
* Turning In Place
* Walking
* Walking Downstairs
* Walking Upstairs

---

#### 🧾 Code Snippet

```python
cv2.putText(annotated_frame, activity.upper(), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
```

---
