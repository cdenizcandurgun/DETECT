import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="AI Vision Suite", layout="wide")

st.title("AI Vision Suite")

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Select Task",
    [
        "Object Detection",
        "Image Segmentation",
        "Hand Detection",
        "Interactive Segmentation",
        "Gesture Recognition (No MediaPipe)",
        "Pose Landmark Detection (No MediaPipe)"
    ]
)

# Load YOLOv8 models for detection and segmentation
def load_yolo_model(task):
    if task == "Object Detection":
        model = YOLO('yolov8n.pt')
    elif task == "Image Segmentation":
        model = YOLO('yolov8n-seg.pt')
    
    elif task == "Pose Landmark Detection (No MediaPipe)":
        model = YOLO('yolov8n-pose.pt')
    else:
        model = None
    return model

# Webcam capture
def get_webcam_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to access webcam.")
        return None
    return frame

# Object Detection
import time
def object_detection():
    st.subheader("Object Detection (YOLOv8)")
    model = load_yolo_model("Object Detection")
    run = st.checkbox("Start Live Object Detection")
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop")
        while run and (not stop):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            results = model(frame)
            img = results[0].plot()
            FRAME_WINDOW.image(img, channels="BGR", caption="Detection Result")
            time.sleep(0.03)
        if cap:
            cap.release()

# Image Segmentation
def image_segmentation():
    st.subheader("Image Segmentation (YOLOv8-seg)")
    model = load_yolo_model("Image Segmentation")
    run = st.checkbox("Start Live Image Segmentation")
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop")
        while run and (not stop):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            results = model(frame)
            img = results[0].plot()
            FRAME_WINDOW.image(img, channels="BGR", caption="Segmentation Result")
            time.sleep(0.03)
        if cap:
            cap.release()

# Hand Detection (MediaPipe)
def count_fingers_and_draw(frame, contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return frame, 0, []
    defects = cv2.convexityDefects(contour, hull)
    finger_tips = []
    count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            # Use angle and distance to filter real finger gaps
            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(end) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))
            angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b+1e-5))
            if angle <= np.pi/2 and d > 10000:
                count += 1
                finger_tips.append(start)
                finger_tips.append(end)
                cv2.circle(frame, start, 8, (0,0,255), -1)
                cv2.circle(frame, end, 8, (0,0,255), -1)
        finger_tips = list(set(finger_tips))
    # Draw hull
    hull_points = cv2.convexHull(contour)
    cv2.drawContours(frame, [hull_points], -1, (255,255,0), 2)
    # Draw contour
    cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
    # Gesture label
    label = str(min(count+1, 5)) if count > 0 else 'fist'
    cv2.putText(frame, f'Fingers: {min(count+1,5)}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f'Gesture: {label}', (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame, min(count+1,5), finger_tips

def yolo_hand_landmark_detection():
    st.subheader("Hand Detection (YOLO11n-pose, finger-by-finger)")
    st.info("This uses the YOLO11n-pose hand keypoint model for live finger and joint detection. No MediaPipe.")
    run = st.checkbox("Start Live Hand Detection (YOLO)")
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        try:
            from ultralytics import YOLO
        except ImportError:
            st.error("Ultralytics YOLO is not installed. Please add 'ultralytics' to requirements.txt and install it.")
            return
        model = YOLO('best.pt')
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop")
        while run and (not stop):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            results = model.predict(source=frame, conf=0.3, verbose=False)
            annotated = frame.copy()
            for r in results:
                if hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints.xy) > 0:
                    kps = r.keypoints.xy[0].cpu().numpy().astype(int)
                    if kps.shape[0] >= 21:
                        # Draw keypoints
                        for idx, (x, y) in enumerate(kps):
                            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(annotated, str(idx), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        # Draw skeleton (connect keypoints as in MediaPipe)
                        skeleton = [
                            (0,1),(1,2),(2,3),(3,4),      # Thumb
                            (0,5),(5,6),(6,7),(7,8),      # Index
                            (0,9),(9,10),(10,11),(11,12), # Middle
                            (0,13),(13,14),(14,15),(15,16), # Ring
                            (0,17),(17,18),(18,19),(19,20) # Pinky
                        ]
                        for (i,j) in skeleton:
                            if i < kps.shape[0] and j < kps.shape[0]:
                                cv2.line(annotated, tuple(kps[i]), tuple(kps[j]), (255,0,0), 2)
            FRAME_WINDOW.image(annotated, channels="BGR", caption="YOLO Hand Landmark Detection")
            import time
            time.sleep(0.03)
        if cap:
            cap.release()

def hand_detection():
    yolo_hand_landmark_detection()

# Interactive Segmentation (using Segment Anything)
def interactive_segmentation():
    st.subheader("Interactive Segmentation (SAM)")
    st.info("Please upload an image for interactive segmentation.")
    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image")
        st.warning("Interactive segmentation UI coming soon (requires user clicks)")
        # Placeholder for SAM integration

# Gesture Recognition (No MediaPipe)
def gesture_recognition():
    st.subheader("Gesture Recognition (OpenCV, finger-by-finger)")
    st.info("Classic CV: skin color segmentation, convex hull, and finger counting. No MediaPipe, no YOLO.")
    run = st.checkbox("Start Live Gesture Recognition")
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop")
        while run and (not stop):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            mask = cv2.inRange(ycrcb, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 2000:
                    frame, finger_count, finger_tips = count_fingers_and_draw(frame, max_contour)
            FRAME_WINDOW.image(frame, channels="BGR", caption="Gesture Recognition Result")
            import time
            time.sleep(0.03)
        if cap:
            cap.release()

# Pose Landmark Detection (No MediaPipe)
def pose_landmark_detection():
    st.subheader("Pose Landmark Detection (No MediaPipe)")
    st.info("This uses YOLOv8-pose model for pose landmark detection.")
    model = YOLO('yolov8n-pose.pt')
    run = st.checkbox("Start Live Pose Landmark Detection")
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop")
        while run and (not stop):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            results = model(frame)
            img = results[0].plot()
            FRAME_WINDOW.image(img, channels="BGR", caption="Pose Landmark Detection Result")
            time.sleep(0.03)
        if cap:
            cap.release()

# Main logic
if task == "Object Detection":
    object_detection()
elif task == "Image Segmentation":
    image_segmentation()
elif task == "Hand Detection":
    hand_detection()
elif task == "Interactive Segmentation":
    interactive_segmentation()
elif task == "Gesture Recognition (No MediaPipe)":
    gesture_recognition()
elif task == "Pose Landmark Detection (No MediaPipe)":
    pose_landmark_detection()
