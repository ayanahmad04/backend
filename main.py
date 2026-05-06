import os
import io
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

import tensorflow as tf

# =======================
# CONFIG
# =======================
APP_TITLE = "X-Ray Classification API (TFLite)"
MODEL_PATH = "model/model.tflite"
IMG_SIZE = (224, 224)

CLASS_NAMES = ["Normal", "Pneumonia", "Pneumothorax", "TB"]

# ✅ Correct File ID from your Google Drive link
FILE_ID = "15l9QCRs0OdlNAeZix6agNXgyH3qdWHnw"

app = FastAPI(title=APP_TITLE)

interpreter = None
input_details = None
output_details = None


# =======================
# GOOGLE DRIVE DOWNLOADER
# =======================
def download_model():
    os.makedirs("model", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✅ TFLite Model already exists")
        return

    print("📥 Downloading TFLite model from Google Drive...")

    try:
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        print("✅ TFLite Model downloaded successfully")

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
    global interpreter, input_details, output_details

    download_model()

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("🚀 TFLite Interpreter loaded and ready")


# =======================
# ROUTES
# =======================
@app.get("/health")
def health():
    return {"status": "ok", "engine": "tflite_runtime"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image")

    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    best_idx = int(np.argmax(preds))

    return {
        "scores": preds.tolist(),
        "predicted_index": best_idx,
        "predicted_label": CLASS_NAMES[best_idx],
        "confidence": float(preds[best_idx]),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)