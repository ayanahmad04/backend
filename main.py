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

IMG_SIZE = (224, 224)

# ---- TB MODEL ----
MODEL_PATH = "model/best_model.keras"
CLASS_NAMES = ["Normal", "Tuberculosis"]
FILE_ID = "1yPQhpal3_QiVWe5rs2JZSUFFX1K0_9Wv"

# ---- PNEUMONIA MODEL ----
PNO_MODEL_PATH = "model/pno3_model.keras"
PNO_CLASS_NAMES = ["Normal", "Pneumonia"]
PNO_FILE_ID = "1CV1InkqHp4uEg9jSgByBFhJ_jmt24CTV"

app = FastAPI(title=APP_TITLE)

model = None
pno_model = None


# =======================
# GOOGLE DRIVE DOWNLOADER
# =======================
def download_model(file_id: str, model_path: str):
    os.makedirs("model", exist_ok=True)

    if os.path.exists(model_path):
        print(f"✅ {model_path} already exists")
        return

    print(f"📥 Downloading {model_path} from Google Drive...")

    try:
        session = requests.Session()
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

        response = session.get(url, stream=True, timeout=300)
        response.raise_for_status()

        if response.headers.get("content-type", "").startswith("text/html"):
            raise Exception("Received HTML instead of model file")

        with open(model_path, "wb") as f:
            total_size = 0
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        print(f"✅ Downloaded {model_path} ({total_size / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise


# =======================
# APP STARTUP
# =======================
@app.on_event("startup")
def load_models():
    global model, pno_model

    # Load TB model
    download_model(FILE_ID, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🚀 TB model loaded")

    # Load Pneumonia model
    download_model(PNO_FILE_ID, PNO_MODEL_PATH)
    pno_model = tf.keras.models.load_model(PNO_MODEL_PATH)
    print("🚀 Pneumonia model loaded")


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


# ---- TB PREDICTION ----
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


# ---- PNEUMONIA PREDICTION ----
@app.post("/predict-pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    preds = pno_model.predict(x)[0]
    best_idx = int(np.argmax(preds))

    return {
        "scores": preds.tolist(),
        "predicted_index": best_idx,
        "predicted_label": PNO_CLASS_NAMES[best_idx],
        "confidence": float(preds[best_idx]),
    }
