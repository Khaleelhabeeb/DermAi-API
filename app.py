from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = FastAPI()

# Load your pre-trained model
model = load_model("my_model_weights.h5")

# Define the disease classes
disease_classes = ["BA-cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus", "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"]

def preprocess_image(image):
    # Resize and preprocess the image as needed
    resized_image = cv2.resize(image, (150, 150))
    normalized_image = resized_image.astype('float32') / 255
    return np.expand_dims(normalized_image, axis=0)

def predict_disease(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    if prediction[0][predicted_class_index] > 0.5:
        return disease_classes[predicted_class_index]
    else:
        return "No disease detected"

@app.post("/predict/")
async def predict_skin_disease(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    disease = predict_disease(img)
    return {"disease_detected": disease}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
