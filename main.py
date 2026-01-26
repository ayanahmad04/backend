import os
import io
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf

# =======================
# CONFIG
# =======================
APP_TITLE = "Image Model API"
MODEL_PATH = "model/best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Tuberculosis"]

FILE_ID = "1yPQhpal3_QiVWe5rs2JZSUFFX1K0_9Wv"

app = FastAPI(title=APP_TITLE)
model = None


# =======================
# GOOGLE DRIVE DOWNLOADER
# =======================
def download_model():
    os.makedirs("model", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists")
        return

    print("📥 Downloading model from Google Drive...")

    try:
        session = requests.Session()
        # Use the usercontent endpoint directly for large files
        url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
        
        response = session.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Verify we got binary data (model file) not HTML
        if response.headers.get('content-type', '').startswith('text/html'):
            raise Exception("Received HTML instead of model file")
        
        with open(MODEL_PATH, "wb") as f:
            total_size = 0
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"✅ Model downloaded successfully ({total_size / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise


# =======================
# APP STARTUP
# =======================
@app.on_event("startup")
def load_model():
    global model
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🚀 Model loaded")


# =======================
# IMAGE PREPROCESSING
# =======================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# =======================
# ROUTES
# =======================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    preds = model.predict(x)[0]
    best_idx = int(np.argmax(preds))

    return {
        "scores": preds.tolist(),
        "predicted_index": best_idx,
        "predicted_label": CLASS_NAMES[best_idx],
        "confidence": float(preds[best_idx]),
    }
