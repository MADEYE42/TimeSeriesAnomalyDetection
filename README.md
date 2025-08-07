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

## Dataset Links

- **Multi-Class Dataset**: [Google Drive](https://drive.google.com/drive/folders/1Cf3F3h30XrZixpv8k8O-clh-vf5y0wXt?usp=share_link)
- **Binary Dataset (Fall vs Normal)**: [Google Drive](https://drive.google.com/drive/folders/1q6smqhNk5hbGhx1LNVXcH54K_L8ZCuu_?usp=share_link)

## Project Highlights

- End-to-end anomaly detection pipeline using video frame-based time series data.
- Compared multiple deep learning architectures on both binary and multi-class classification tasks.
- Evaluation included confusion matrices for each model and class type.

---

