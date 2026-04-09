# Eye Contact Detection AI (YOLO Edition) 👁

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.1.0-007bff.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-ff4b4b.svg)](https://streamlit.io/)

Интеллектуальная система детекции зрительного контакта в реальном времени. В основе решения лежит нейросеть **YOLO**, оптимизированная для работы с мелкими деталями лица за счет динамического кропа.

## 📊 Характеристики системы

* **Архитектура:** YOLO (Object Detection + Classification)
* **Детектор лиц:** MediaPipe Face Landmarker (478 точек)
* **Метод обработки:** **Smart Zoom** — автоматическое вырезание и масштабирование области глаз перед инференсом.

## 📈 Результаты обучения

| Метрика | Значение |
| :--- | :--- |
| **F1-Score** | `0.87` |
| **mAP50** | `0.92` |
| **Status** | Fine-tuned & Ready |

---

### Визуализация обучения и примеры:

<p align="center">
  <img width="800" alt="results" src="https://github.com/user-attachments/assets/41d57d2f-0319-425e-9787-fdcdf8622a99" />
</p>

<p align="center">
  <img width="48%" alt="BoxP_curve" src="https://github.com/user-attachments/assets/f5a56989-d9b9-45c8-86f8-00474569b53a" />
  <img width="48%" alt="BoxR_curve" src="https://github.com/user-attachments/assets/acae1328-004b-4358-99a5-bea01a65e490" />
</p>

<p align="center">
  <img width="48%" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/d2da72b8-5049-439e-b252-aa7033f4da57" />
  <img width="48%" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/d027d69a-b873-4ae7-b1ed-2dd519563b55" />
</p>

<p align="center">
  <img width="70%" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e76d0d2d-698c-45d4-9375-b15fed86dd46" />
</p>

<p align="center">
  <img width="800" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/221a32d5-e4b9-47b7-a5da-f4de6dbbc392" />
  <br><em>Пример работы модели на валидационной выборке</em>
</p>

## ⚙️ Установка

1.  **Клонируйте репозиторий и установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Разместите веса моделей в корневой папке:**
    * `best.pt` — обученные веса YOLO.
    * `face_landmarker.task` — модель MediaPipe для трекинга лиц.

## 🚀 Использование

1.  **Запустите веб-интерфейс:**
    ```bash
    streamlit run app.py
    ```

## 🔥 Ключевые фичи:

1.  **Smart Zoom Preprocessing:** Система находит лицо через 478 точек Landmarker и подает в YOLO только область глаз. Это увеличивает детализацию зрачка в 5-10 раз.
2.  **Real-time TTA:** Включен режим `augment=True`, который заменяет ручное усреднение кадров. Модель "смотрит" на каждый кадр под разными углами за миллисекунды.
3.  **Live Confidence Metric:** Мгновенный вывод уверенности модели и визуализация боксов прямо в браузере.
