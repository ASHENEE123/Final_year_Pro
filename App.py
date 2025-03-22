from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("latest_model.h5")  

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://brainstrokedetection.onrender.com",
                   "http://localhost:3000/"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Function to preprocess the image
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((256, 256)).convert("L") 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=(0, -1))  
    return img_array

# Prediction API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img_array = preprocess_image(image_data)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    confidence_score = float(np.max(predictions))  

    return {
        "predicted_class": int(predicted_class),
        "confidence_score": confidence_score
    }
