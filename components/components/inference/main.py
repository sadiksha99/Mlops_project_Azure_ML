# -------- main.py (fixed) --------
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from keras.layers import Input, TFSMLayer
from keras.models import Model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

DIGITS = ["2", "7", "8"]

# Use the folder structure that actually exists inside the image
MODEL_DIR = (
    Path(__file__).parent           # → /app
    / "mnist-classifier"
    / "INPUT_model_path"
    / "mnist-cnn"
)

input_layer = Input(shape=(28, 28, 1))
output_layer = TFSMLayer(str(MODEL_DIR), call_endpoint="serving_default")(input_layer)
model = Model(inputs=input_layer, outputs=output_layer)

@app.post("/upload/image")
async def upload_image(img: UploadFile = File(...)):
    image = Image.open(img.file).convert("L").resize((28, 28))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    preds = model.predict(arr)
    pred_idx = int(np.argmax(preds))
    return {"prediction": DIGITS[pred_idx]}
