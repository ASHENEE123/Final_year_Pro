# 🧠 Brain Stroke Prediction Using CNN

Welcome to the **Brain Stroke Prediction** project!  
This repository contains a powerful deep learning solution for automated brain stroke detection and classification using Convolutional Neural Networks (CNN). The model achieves an impressive **97% accuracy** and offers a web API for real-time MRI scan analysis.

---

## 🚩 Project Purpose

Brain stroke is a life-threatening medical emergency. Rapid and accurate diagnosis is crucial for effective treatment. This project leverages deep learning to automatically classify MRI brain scans into:
- **Normal**
- **Hemorrhagic Stroke**
- **Ischemic Stroke**

The goal is to assist medical professionals and researchers in quick and reliable stroke assessment, thus improving patient outcomes.

---

## ✨ Key Features

- 🔥 **Custom CNN Model**: Built from scratch for high accuracy in brain stroke classification.
- 📈 **97% Accuracy**: Robust performance validated on real-world dataset.
- 🖼️ **MRI Scan Upload**: Users can upload MRI images for instant prediction.
- 🎨 **Stroke Highlighting**: Visual overlay to mark stroke regions on scans.
- 🗨️ **RESTful API**: FastAPI-powered backend for seamless integration.
- 🌐 **CORS Enabled**: Ready for web deployment and secure cross-origin requests.

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** (Deep Learning)
- **OpenCV & PIL** (Image Processing)
- **FastAPI** (Web API)
- **NumPy** (Numerical Operations)

---

## 🚀 Usage Instructions

### 1. Clone the repository
```bash
git clone https://github.com/ASHENEE123/Final_year_Pro.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API server
```bash
python App.py
```

### 4. Make a prediction (example using `curl`)
```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@your_mri_scan.png"
```
- Response includes classification (`Normal`, `Hemorrhagic`, `Ischemic`) and confidence score.
- Visualization overlay returned as image.

---

## 🧩 Insightful Highlights

- **Custom Preprocessing**: MRI scans are normalized, resized (256x256), and converted to grayscale for optimal CNN performance.
- **Stroke Region Highlighting**: Detected stroke areas are visually marked using color overlays and edge detection.
- **Model Deployment**: The system is ready for cloud deployment (compatible with Render, Heroku, etc.).
- **Medical Impact**: Enables rapid, automated stroke screening, potentially saving lives and aiding in early intervention.
- **API Design**: Includes CORS setup for integration with web-based frontends or hospital systems.

---

## 🖼️ Screenshots & Icons

| Prediction | Highlighted Scan | API Response |
|:----------:|:----------------:|:------------:|
| ![🧠](https://img.icons8.com/color/96/000000/brain.png) | ![🩺](https://img.icons8.com/color/96/000000/mri.png) | ![⚡](https://img.icons8.com/color/96/000000/artificial-intelligence.png) |

---

> **Empowering healthcare with AI for a better tomorrow!**
