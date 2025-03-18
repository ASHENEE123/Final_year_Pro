from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("final_model.h5")  # Ensure this file exists

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://brainstrokedetection.onrender.com",
                  "http://localhost:3000/"],  # Replace with your React frontend URL e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Function to preprocess the image
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((256, 256))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img_array = preprocess_image(image_data)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class with highest probability
    confidence_score = float(np.max(predictions))  # Get the highest probability score

    return {
        "predicted_class": int(predicted_class),
        "confidence_score": confidence_score
    }
