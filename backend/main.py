"""
FastAPI backend for the Cat vs Dog classifier.

Serves a single endpoint, POST /predict, that accepts an uploaded image and
returns whether it's a cat or a dog, using the exact model architecture and
preprocessing from the training notebook (VGG16 transfer-learning model).

Run with:
    uvicorn main:app --reload --port 8000

The model weights file (default: model/cat_dog_vgg16.h5) must exist first.
Train it by running train.py (see README.md), or point MODEL_PATH at your
own weights file trained with the same architecture.
"""

import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "cat_dog_vgg16.h5"))
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]  # index 0 -> cat, index 1 -> dog (matches notebook's sigmoid convention)

app = FastAPI(title="Cat vs Dog Classifier", version="1.0.0")

# Allow the React frontend (served from anywhere, e.g. a local file or a dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # lazy-loaded so the server can boot even before a model exists


def get_model():
    """Load the Keras model once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model file not found at '{MODEL_PATH}'. "
                    "Train it first by running train.py (see README.md), "
                    "or set the MODEL_PATH environment variable to point at your .h5 file."
                ),
            )
        # Imported lazily: TensorFlow is slow to import and not needed until we actually predict.
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH)
    return _model


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Turn uploaded image bytes into the (1, 224, 224, 3) float32 array the model expects.

    Note: the model itself contains a Lambda layer that runs
    tf.keras.applications.vgg16.preprocess_input, so here we only resize and
    convert to RGB float32 pixel values (0-255) -- we do NOT normalize again.
    """
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image file: {exc}")

    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "ok", "model_ready": model_ready, "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg, png, etc).")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model = get_model()
    batch = preprocess_image(raw_bytes)

    prediction = model.predict(batch, verbose=0)
    dog_probability = float(prediction[0][0])
    cat_probability = 1.0 - dog_probability

    label = CLASS_NAMES[1] if dog_probability > 0.5 else CLASS_NAMES[0]
    confidence = dog_probability if dog_probability > 0.5 else cat_probability

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(cat_probability, 4),
            "dog": round(dog_probability, 4),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
