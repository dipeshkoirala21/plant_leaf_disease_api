from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(tf.__version__)

# Define model paths and class names for different plants
MODELS = {
    "potato": {
        "model_path": "./models/potato-model",
        "class_names": ["Early Blight", "Late Blight", "Healthy"],
        "input_shape": (256, 256, 3)
    },
    "tomato": {
        "model_path": "./models/tomato_model",
        "class_names": ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
                        "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"],
        "input_shape": (256, 256, 3)
    },
    "bell_pepper": {
        "model_path": "./models/bell_pepper_model",
        "class_names": ["Bacterial Spot", "Healthy"],
        "input_shape": (256, 256, 3)
    }
}


def load_model(model_path: str):
    # Use TFSMLayer to load the model
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

    # Create a functional model with the loaded TFSMLayer
    inputs = keras.Input(shape=(None, None, 3))  # Adjust the input shape as per your model
    outputs = model_layer(inputs)
    model = keras.Model(inputs, outputs)
    return model


# Load models
MODELS_LOADED = {plant: load_model(info["model_path"]) for plant, info in MODELS.items()}


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data, input_shape) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((input_shape[1], input_shape[0]))  # Resize to the model's input shape
    return np.array(image)


@app.post("/predict/{plant_type}")
async def predict(plant_type: str, file: UploadFile = File(...)):
    if plant_type not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid plant type")

    model_info = MODELS[plant_type]
    model = MODELS_LOADED[plant_type]
    class_names = model_info["class_names"]
    input_shape = model_info["input_shape"]

    image = read_file_as_image(await file.read(), input_shape)
    img_batch = np.expand_dims(image, 0)

    predictions = model.predict(img_batch)

    # Debugging: Print the predictions
    print("Predictions:", predictions)

    # Extract the prediction array from the dictionary
    prediction_array = predictions['output_0']

    if prediction_array.size == 0:
        return {"error": "No predictions made"}

    predicted_class = class_names[np.argmax(prediction_array[0])]
    confidence = np.max(prediction_array[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
