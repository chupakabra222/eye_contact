import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as IMG
import time

@st.cache_resource
def init_models():
    model = YOLO('best.pt') 
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker, model

landmarker, yolo_model = init_models()

if 'video_cap' not in st.session_state:
    st.session_state.video_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if 'freeze_data' not in st.session_state:
    st.session_state.freeze_data = {"image": None, "conf": 0.0, "label": "None"}

if 'needs_capture' not in st.session_state:
    st.session_state.needs_capture = False

st.set_page_config(page_title="YOLO Eye Monitor", layout="wide")
st.title("👁 YOLO Eye Contact: Hard Cache Mode")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Live Поток (Зум)")
    FRAME_WINDOW = st.empty()

with col_right:
    st.subheader("Управление")
    conf_threshold = st.slider("Порог уверенности", 0.0, 1.0, 0.3, key="conf_slider")
    
    if st.button("📸 ЗАФИКСИРОВАТЬ КАДР", use_container_width=True):
        st.session_state.needs_capture = True
    
    st.write("---")
    st.subheader("Зафиксированный результат")
    FIXED_WINDOW = st.empty()
    METRIC_WINDOW = st.empty()

def get_zoom_crop(image, target_size=(640, 640)):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = IMG(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return None
    
    land = res.face_landmarks[0]
    all_x = [l.x * w for l in land]
    all_y = [l.y * h for l in land]
    
    xmin, xmax, ymin, ymax = min(all_x), max(all_x), min(all_y), max(all_y)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    side = max(xmax - xmin, ymax - ymin) * 1.6
    
    x1, y1 = int(max(0, cx - side/2)), int(max(0, cy - side/2))
    x2, y2 = int(min(w, cx + side/2)), int(min(h, cy + side/2))
    
    crop = image[y1:y2, x1:x2]
    if crop.size > 0:
        return cv2.resize(crop, target_size)
    return None

cap = st.session_state.video_cap

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    face_zoom = get_zoom_crop(frame)
    
    if face_zoom is not None:
        results = yolo_model(face_zoom, conf=conf_threshold, augment=True, verbose=False)
        display_frame = results[0].plot()
        
        c_conf = 0.0
        c_label = "None"
        if len(results[0].boxes) > 0:
            c_conf = results[0].boxes.conf[0].item()
            c_label = yolo_model.names[int(results[0].boxes.cls[0].item())]
        
        if st.session_state.needs_capture:
            st.session_state.freeze_data["image"] = np.copy(display_frame)
            st.session_state.freeze_data["conf"] = c_conf
            st.session_state.freeze_data["label"] = c_label
            st.session_state.needs_capture = False
    else:
        display_frame = frame.copy()
        cv2.putText(display_frame, "FACE NOT FOUND", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    if st.session_state.freeze_data["image"] is not None:
        FIXED_WINDOW.image(
            cv2.cvtColor(st.session_state.freeze_data["image"], cv2.COLOR_BGR2RGB), 
            width=400
        )
        METRIC_WINDOW.metric(
            f"Зафиксировано: {st.session_state.freeze_data['label']}", 
            f"{st.session_state.freeze_data['conf']:.4f}"
        )

    time.sleep(0.01)