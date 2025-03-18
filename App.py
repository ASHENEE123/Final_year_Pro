from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("final_model.h5")  # Ensure this file exists

app = FastAPI()

# Function to preprocess the image
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((256, 256))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# API endpoint to classify an image
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
