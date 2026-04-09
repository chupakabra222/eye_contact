# Eye Contact Detection AI

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-ff4b4b.svg)](https://streamlit.io/)

Решение задачи классификации зрительного контакта (Eye Contact) с использованием дообученной нейросети **ResNet18**. Проект включает в себя интерактивное веб-приложение для тестирования модели в реальном времени.

## 📊 Характеристики модели

* **Архитектура:** ResNet18 (Fine-tuned)
* **Датасет:** [Eye Contact Dataset (Kaggle)](https://www.kaggle.com/datasets/pratikyuvrajchougule/eye-contact)
* **Метрики:**
    * **F1.5 Score:** `0.80`
    * **AUC:** `0.96`
* **Препроцессинг:** Динамический зум области лица с помощью **MediaPipe Face Landmarker**.
---
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/41d57d2f-0319-425e-9787-fdcdf8622a99" />
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/d2da72b8-5049-439e-b252-aa7033f4da57" />
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e76d0d2d-698c-45d4-9375-b15fed86dd46" />
![val_batch0_labels](https://github.com/user-attachments/assets/221a32d5-e4b9-47b7-a5da-f4de6dbbc392)

## ⚙️ Установка

1. **Установите зависимости:**
   ```pip install -r requirements.txt```

2. **Установите в корень приложения веса модели (check releases)**

## 🚀 Использование
1. **Запустите приложение:**
 ```streamlit run app.py```

## Основной функционал:
1. **Real-time Inference: Мгновенное предсказание вероятности контакта.**

2. **Temporal Averaging: Кнопка «ЗАФИКСИРОВАТЬ» запускает сбор 10 последовательных кадров и вычисляет среднюю вероятность для устранения шумов модели.**

3. **Custom Threshold: Слайдер для ручной настройки порога чувствительности (по умолчанию 0.4387).**
