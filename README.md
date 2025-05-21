# AI Vision Suite

A GPU-accelerated app for Object Detection, Image Segmentation, Interactive Segmentation, Gesture Recognition, and Pose Landmark Detection using your webcam.

## Features
- **Object Detection**: YOLOv8 (Ultralytics)
- **Image Segmentation**: DeepLabV3 or YOLOv8-seg
- **Interactive Segmentation**: Segment Anything (Meta AI)
- **Gesture Recognition**: (To be implemented without MediaPipe)
- **Pose Landmark Detection**: (To be implemented without MediaPipe)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Requires a CUDA-capable GPU for best performance.
- All processing is done locally.
- Webcam access is required.
- MediaPipe is NOT used in this project (as per user request).
