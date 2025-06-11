from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.layers import Input, TFSMLayer
from keras.models import Model
from PIL import Image

app = FastAPI()

# ---------------------------------------------------------------------------
# CORS (open for quick front‑end testing; tighten for production)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model + label map
# ---------------------------------------------------------------------------
DIGITS = ["2", "7", "8"]  # classes in the order your model outputs them

MODEL_DIR = (
    Path(__file__).parent
    / "mnist-classifier"
    / "INPUT_model_path"  # ⬅️ replace with the real folder
    / "mnist-cnn"
)

# If your SavedModel is exported with flexible spatial dims (None, None, C)
# you can declare the Keras input correspondingly. Otherwise use the exact
# shape your model requires.
input_layer = Input(shape=(None, None, 1), name="input_image")

# Wrap the SavedModel in a TFSMLayer so it can be called as part of a Keras graph
output_layer = TFSMLayer(str(MODEL_DIR), call_endpoint="serving_default")(
    input_layer
)
model = Model(inputs=input_layer, outputs=output_layer)

# ---------------------------------------------------------------------------
# Inference endpoint – **no preprocessing**
# ---------------------------------------------------------------------------
@app.post("/upload/image")
async def upload_image(img: UploadFile = File(...)):
    # Read the image and convert to grayscale explicitly
    image = Image.open(img.file).convert("L")  # "L" ensures 1 channel (grayscale)

    # Resize to 28x28 and convert to numpy array
    image = image.resize((28, 28))
    arr = np.array(image, dtype=np.float32) / 255.0

    # Add batch and channel dimensions: (1, 28, 28, 1)
    arr = arr.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(arr)
    pred_idx = int(np.argmax(preds))

    return {"prediction": DIGITS[pred_idx]}
