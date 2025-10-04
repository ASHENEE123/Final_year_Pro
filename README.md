# 🧠 Brain Stroke Prediction Using CNN
This Is deployed Version 

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

#SCREEENSHOTS
![imagealt](https://github.com/ASHENEE123/Final_year_Pro/blob/a9246c0ed7b21e11aebe0cc303946815322020c3/Screenshot%202025-08-29%20133517.png)
![imagealt](https://github.com/ASHENEE123/Final_year_Pro/blob/a9246c0ed7b21e11aebe0cc303946815322020c3/Screenshot%202025-08-29%20133542.png)
![imagealt](https://github.com/ASHENEE123/Final_year_Pro/blob/a9246c0ed7b21e11aebe0cc303946815322020c3/Screenshot%202025-08-29%20133634.png)
![imagealt](https://github.com/ASHENEE123/Final_year_Pro/blob/ae81cb82f0f5db59cd77c1e38833160aefe5767a/Screenshot%202025-04-18%20182401.png)

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
[curl -X POST "http://localhost:8000/predict/" -F "file=@your_mri_scan.png"] LOCALHost
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




