# Time Series Anomaly Detection using Video Frames

This project focuses on detecting human activity anomalies using time series classification models trained on video frame sequences. The objective was to classify actions from video data in two formats:

- **Binary Classification**: Fall vs Normal
- **Multi-Class Classification**: Exercise, Falling, Lying, Running, Sitting, Standing, Walking, Walking Downstairs, Walking Upstairs

## Data Collection and Processing

- Original data was in video format.
- Converted to image frames at:
  - **1 FPS** for binary classification
  - **2 FPS** for multi-class classification

## Models Used

1. **CNN + LSTM**
2. **I3D (Inflated 3D ConvNet)**
3. **YOLOv8 + ResNet50**
4. **YOLOv8 + ResNet101**

These models were chosen to capture both spatial and temporal features effectively.

---

## Model Choices, Working, and Comparisons

### 1. CNN + LSTM
**Why This Model**  
Perfect for modeling spatiotemporal sequences in video-based tasks. Using frames per second, CNN extracts spatial features, LSTM captures temporal dependencies.

**How It Works**  
1. CNN extracts spatial features (e.g., postures, objects).  
2. LSTM models sequence changes over time.  

**Pros**  
- Good generalization on small–medium datasets.  
- Captures both appearance and time dynamics.  

**Cons**  
- Slower training due to sequential LSTM.  
- Struggles with very long sequences.  

**Comparison**  
- Better than YOLO+ResNet when temporal structure is key.  
- Not as optimized as I3D for spatiotemporal features.

---

### 2. I3D (Inflated 3D ConvNet)
**Why This Model**  
Built for video classification, inflates 2D ConvNets to 3D to capture spatial and temporal features together.

**How It Works**  
- Extends 2D CNN (e.g., Inception) into 3D with filters over height, width, and time.  
- Processes stacked frames as a single video volume.

**Pros**  
- Strong temporal modeling.  
- No separate LSTM needed.  

**Cons**  
- Requires high GPU memory and training time.  
- Needs large datasets.

**Comparison**  
- Outperforms CNN+LSTM for short, dense action clips.  
- Not ideal for low compute setups.

---

### 3. YOLOv8 + ResNet50
**Why This Model**  
Combines fast object detection (YOLO) with feature extraction (ResNet50). Good for identifying action-relevant regions before classification.

**How It Works**  
1. YOLOv8 detects human/action regions in each frame.  
2. Cropped regions passed to ResNet50 for feature encoding.  
3. Features classified.

**Pros**  
- Fast, lightweight.  
- Excellent for localized motion detection.  

**Cons**  
- ResNet50 may miss deep spatial cues.  
- No inherent temporal modeling.

**Comparison**  
- Highly interpretable (detections visible).  
- Less temporal awareness than I3D or CNN+LSTM.

---

### 4. YOLOv8 + ResNet101
**Why This Model**  
Same as above, but deeper ResNet101 improves complex feature learning.

**How It Works**  
- YOLOv8 for detection.  
- ResNet101 for deeper feature encoding.

**Pros**  
- Higher accuracy than ResNet50.  
- Retains YOLO's speed for detection.

**Cons**  
- Heavier, slower than ResNet50.  
- Still no temporal modeling.

**Comparison**  
- More accurate than ResNet50.  
- Less unified than I3D/CNN+LSTM for spatial-temporal learning.

---

### Model Comparison Summary

| Model                 | Spatial | Temporal | Speed     | Accuracy | Data Requirement | Best For |
|-----------------------|---------|----------|-----------|----------|------------------|----------|
| CNN + LSTM            | ✅      | ✅        | ⚠️ Medium | ✅ Good  | Low–Medium       | Balanced Tasks |
| I3D                   | ✅✅     | ✅✅       | ❌ Slow    | ✅✅ High | High             | Dense Video Patterns |
| YOLOv8 + ResNet50     | ✅✅     | ❌        | ✅ Fast    | ⚠️ Avg   | Low–Medium       | Real-time Detection |
| YOLOv8 + ResNet101    | ✅✅✅   | ❌        | ⚠️ Medium | ✅ Better| Medium–High      | Complex Scenes |

---

### Alternatives to Consider

| Model                  | Description                                    | Pros                              | Cons                        |
|------------------------|------------------------------------------------|------------------------------------|-----------------------------|
| TSM (Temporal Shift)   | Efficient temporal modeling in CNNs             | Low compute, good accuracy        | Less explored than I3D      |
| SlowFast Networks      | Two-stream CNN for slow and fast motions        | Strong for activity recognition   | High GPU need               |
| Transformer-based      | Attention across space & time (TimeSFormer, ViViT) | State-of-the-art performance      | Large data + compute        |
| ST-GCN                 | Graph model using skeleton/keypoints           | Lightweight, accurate             | Needs pose estimation step  |

---

### Recommendations

- **Binary Classification (Fall vs Normal)**: CNN+LSTM is sufficient and interpretable.  
- **Multi-Class**: I3D or YOLOv8 + ResNet101 if you have enough data and compute.  
- **Real-Time**: YOLOv8 + ResNet50 or fast temporal networks.

---

## Confusion Matrices

### Multi-Class Activity Classification

| Model | Confusion Matrix |
|-------|------------------|
| CNN + LSTM | ![Multi CNN+LSTM](https://github.com/user-attachments/assets/ce680256-5c26-4e35-b954-d6b9e54170cb) |
| I3D | ![Multi I3D](https://github.com/user-attachments/assets/b29fb07c-0404-4b83-8d95-2142445b6fbb) |
| YOLOv8 + ResNet50 | ![Multi YOLO+ResNet50](https://github.com/user-attachments/assets/e826b6d4-5236-4722-a5bd-fd8afaafeb7e) |
| YOLOv8 + ResNet101 | ![Multi YOLO+ResNet101](https://github.com/user-attachments/assets/2c422e16-f036-43a7-9e02-93e75f8e9f09) |

### Binary Fall Detection (Fall vs Normal)

| Model | Confusion Matrix |
|-------|------------------|
| CNN + LSTM | ![Binary CNN+LSTM](https://github.com/user-attachments/assets/68ac7819-fa90-4dc7-bfcc-c911418af71c) |
| I3D | ![Binary I3D](https://github.com/user-attachments/assets/f8befb7c-c860-4966-8407-c742abb6d930) |
| YOLOv8 + ResNet50 | ![Binary YOLO+ResNet50](https://github.com/user-attachments/assets/f5b7208f-4e0f-4efa-8e78-ac7e881d481c) |
| YOLOv8 + ResNet101 | ![Binary YOLO+ResNet101](https://github.com/user-attachments/assets/d358b114-4a0b-4411-b00c-16d0b9e164aa) |

---

## Dataset Links

- **Multi-Class Dataset**: [Google Drive](https://drive.google.com/drive/folders/1Cf3F3h30XrZixpv8k8O-clh-vf5y0wXt?usp=share_link)
- **Binary Dataset (Fall vs Normal)**: [Google Drive](https://drive.google.com/drive/folders/1q6smqhNk5hbGhx1LNVXcH54K_L8ZCuu_?usp=share_link)

## Project Highlights

- End-to-end anomaly detection pipeline using video frame-based time series data.
- Comparison of multiple deep learning architectures for binary and multi-class classification.
- Evaluation through confusion matrices for each model and classification type.


