from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse, JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware  


model = tf.keras.models.load_model("latest_model.h5")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://brainstrokedetection.onrender.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
     expose_headers=["X-Predicted-Class", "X-Confidence-Score"],
)

def preprocess_image(image_data):
    try:
        img = Image.open(BytesIO(image_data)).convert("L")  
        img = img.resize((256, 256))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = np.expand_dims(img_array, axis=-1)  
        return img_array
    except Exception:
        return None

def highlight_stroke(mri_scan):
    if len(mri_scan.shape) == 3:
        mri_scan = cv2.cvtColor(mri_scan, cv2.COLOR_BGR2GRAY)

    equalized_img = cv2.equalizeHist(mri_scan)

    mean_intensity = np.mean(equalized_img)
    threshold_value = mean_intensity + 20  
    _, mask = cv2.threshold(equalized_img, threshold_value, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(equalized_img, 100, 200)
    combined_mask = cv2.bitwise_or(mask, edges)

    colored_scan = cv2.applyColorMap(mri_scan, cv2.COLORMAP_JET)  
    overlay = cv2.addWeighted(colored_scan, 0.7, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

    return overlay

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    
    img_array = preprocess_image(image_data)
    if img_array is None:
        return JSONResponse(content={"error": "Invalid image format. Please upload a valid MRI scan."}, status_code=400)

    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions, axis=1)[0])  
    confidence_score = float(np.max(predictions))  

    nparr = np.frombuffer(image_data, np.uint8)
    mri_scan = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if mri_scan is None:
        return JSONResponse(content={"error": "Invalid image format. Please upload a valid MRI scan."}, status_code=400)

    processed_image = highlight_stroke(mri_scan) if predicted_class in [0, 1] else cv2.cvtColor(mri_scan, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode(".png", processed_image)
    img_bytes = io.BytesIO(buffer)

    return StreamingResponse(img_bytes, media_type="image/png", headers={
        "X-Predicted-Class": str(predicted_class),
        "X-Confidence-Score": str(confidence_score)
    })
