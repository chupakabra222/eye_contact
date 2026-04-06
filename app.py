import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from torchvision import models
import torchvision.transforms.v2 as v2
import time

# --- 1. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
@st.cache_resource
def init_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = FaceLandmarker.create_from_options(options)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    model.load_state_dict(torch.load('resnet18_stage2.pth', map_location=device))
    model.to(device).eval()
    return detector, model, device

detector, model, device = init_models()

val_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. SESSION STATE ---
if 'freeze_frame' not in st.session_state: st.session_state.freeze_frame = None
if 'freeze_crop' not in st.session_state: st.session_state.freeze_crop = None
if 'freeze_prob' not in st.session_state: st.session_state.freeze_prob = 0.0

st.set_page_config(page_title="Eye Contact AI", layout="wide")
st.title("👁 Eye Contact AI: Финальное исправление цветов")

col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("Live Поток")
    FRAME_WINDOW = st.empty()
with col_right:
    st.subheader("Вход модели (Live)")
    LIVE_CROP_WINDOW = st.empty()
    btn_press = st.button("📸 ЗАФИКСИРОВАТЬ (Усреднить 10 кадров)")
    PROGRESS_BAR = st.empty()
    
    st.write("---")
    FIXED_WINDOW = st.empty()
    FIXED_CROP_WINDOW = st.empty()
    METRIC_WINDOW = st.empty()
    threshold = st.slider("Порог детекции", 0.0, 1.0, 0.4387)

# --- 3. ФУНКЦИЯ ОБРАБОТКИ (Работает только с RGB) ---
def process_rgb(frame_rgb, threshold_val):
    h, w, _ = frame_rgb.shape
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    res = detector.detect(mp_img)
    
    if not res.face_landmarks:
        return frame_rgb, None, 0.0
    
    land = res.face_landmarks[0]
    l_e = (np.array([land[33].x * w, land[33].y * h]) + np.array([land[133].x * w, land[133].y * h])) / 2
    r_e = (np.array([land[362].x * w, land[362].y * h]) + np.array([land[263].x * w, land[263].y * h])) / 2
    
    dist = np.linalg.norm(r_e - l_e)
    side = int(dist * 1.8)
    cx, cy = int((l_e[0] + r_e[0]) / 2), int((l_e[1] + r_e[1]) / 2)
    x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    
    crop_rgb = frame_rgb[y1:y2, x1:x2].copy() # Важно сделать .copy()
    if crop_rgb.size == 0: return frame_rgb, None, 0.0
    
    tensor = val_transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0,1].item()
    
    # Рисуем на копии кадра, чтобы не портить оригинал для кэша
    draw_frame = frame_rgb.copy()
    color = (0, 255, 0) if prob > threshold_val else (255, 0, 0) # RGB: Зеленый / Красный
    cv2.putText(draw_frame, f"Prob: {prob:.4f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    return draw_frame, crop_rgb, prob

# --- 4. ЦИКЛ ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        
        # 1. ЕДИНСТВЕННАЯ КОНВЕРТАЦИЯ В RGB
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. ОБРАБОТКА (рисуем текст на копии, возвращаем вероятности)
        live_show, live_crop, live_prob = process_rgb(full_rgb, threshold)
        
        # 3. ЛОГИКА УСРЕДНЕНИЯ ПРИ НАЖАТИИ
        if btn_press:
            temp_probs = []
            # Собираем пачку
            for i in range(10):
                r, fb = cap.read()
                if r:
                    fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                    _, _, p = process_rgb(fr, threshold)
                    temp_probs.append(p)
                PROGRESS_BAR.progress((i + 1) * 10)
            
            if temp_probs:
                # Сохраняем в кэш результаты ПОСЛЕДНЕГО кадра из пачки
                # Мы сохраняем именно RGB вариант
                st.session_state.freeze_frame = live_show.copy()
                st.session_state.freeze_crop = live_crop.copy() if live_crop is not None else None
                st.session_state.freeze_prob = sum(temp_probs) / len(temp_probs)
            
            PROGRESS_BAR.empty()
            btn_press = False 

        # 4. ВЫВОД (уже в RGB, конвертация не нужна)
        FRAME_WINDOW.image(live_show)
        if live_crop is not None:
            LIVE_CROP_WINDOW.image(live_crop, width=200)

        # ВЫВОД КЭША
        if st.session_state.freeze_frame is not None:
            FIXED_WINDOW.image(st.session_state.freeze_frame)
            if st.session_state.freeze_crop is not None:
                FIXED_CROP_WINDOW.image(st.session_state.freeze_crop, width=150)
            
            p_final = st.session_state.freeze_prob
            res_txt = "CONTACT" if p_final > threshold else "NO CONTACT"
            METRIC_WINDOW.metric(f"Результат усреднения: {res_txt}", f"{p_final:.4f}")

        time.sleep(0.01)
finally:
    cap.release()