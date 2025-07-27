# Drowsiness-Detection-Yolov8
Real-time driver drowsiness detection using YOLOv8 — classifies drowsy vs. alert states based on facial features with 98%+ precision and recall
# Real-Time Drowsiness Detection with YOLOv8

This project implements a **deep learning-based driver drowsiness detection system** using [YOLOv8](https://docs.ultralytics.com/). By analyzing facial features such as prolonged eye closure, yawning, and head tilt, the model detects whether a driver is **drowsy or alert**. The goal is to provide a **non-intrusive, real-time safety system** to help prevent fatigue-related accidents.

---

## Project Goals

- Detect signs of driver fatigue from images using facial cues.
- Provide a lightweight, real-time detection pipeline using YOLOv8.
- Build a scalable, accurate model trained on realistic driving scenarios.
- Enable future extension to time-based models (e.g., LSTM) for tracking blink patterns over time.

---

## Model Performance

| Metric     | Value  |
|------------|--------|
| Precision  | 99.3%  |
| Recall     | 98.7%  |
| mAP@0.5    | 98.7%  |

Trained on a custom YOLOv8-compatible dataset of annotated images of **drowsy** and **non-drowsy** individuals.

---

## Dataset

The project uses the **[Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)**, which includes labeled images under varying lighting, demographics, and head positions.

**Dataset Download (via `kagglehub`):**
```python
import kagglehub

path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
Images are annotated in YOLOv8 format with:

Class 0: Drowsy
Class 1: Non-Drowsy
