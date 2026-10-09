import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import MODEL_REPO, MODEL_REVISION, MAX_IMAGE_MB

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("imager-api")
processor = None
model = None
load_error = None
model_lock = threading.Lock()


def normalize_model_repo(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("MODEL_REPO URL must point to huggingface.co")
        value = parsed.path.strip("/")
        if value.startswith("models/"):
            value = value[len("models/"):]
    if not value or "/" not in value:
        raise ValueError("Use a Hugging Face model ID such as Imagers/Imager-1")
    return value


MODEL_ID = normalize_model_repo(MODEL_REPO)


def load_classifier():
    global processor, model, load_error
    try:
        kwargs = {"trust_remote_code": False}
        if MODEL_REVISION:
            kwargs["revision"] = MODEL_REVISION
        log.info("Loading Hugging Face model %s", MODEL_ID)
        p = AutoImageProcessor.from_pretrained(MODEL_ID, **kwargs)
        m = AutoModelForImageClassification.from_pretrained(MODEL_ID, **kwargs)
        m.eval()
        processor, model, load_error = p, m, None
        log.info("Model loaded")
    except Exception as exc:
        load_error = str(exc)
        log.exception("Model loading failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_classifier, daemon=True).start()
    yield


app = FastAPI(title="Imager Model API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"]
)


@app.get("/")
def root():
    status = "ready" if model is not None else ("error" if load_error else "loading")
    return {"service": "Imager Model API", "model_repo": MODEL_ID, "status": status,
            "health": "/health", "prediction_endpoint": "POST /predict"}


@app.get("/health")
def health():
    if model is not None:
        return {"status": "ready", "model_repo": MODEL_ID}
    if load_error:
        return {"status": "error", "model_repo": MODEL_ID, "detail": load_error[:1200]}
    return {"status": "loading", "model_repo": MODEL_ID}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or processor is None:
        detail = "Model is still loading; retry shortly."
        if load_error:
            detail = "Model failed to load: " + load_error[:800]
        raise HTTPException(status_code=503, detail=detail)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_MB} MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    try:
        with model_lock, torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()

        labels = getattr(model.config, "id2label", {}) or {}
        results = []
        for i, score in enumerate(probabilities):
            label = str(labels.get(i, labels.get(str(i), f"LABEL_{i}")))
            results.append({"label": label, "score": float(score)})
        results.sort(key=lambda x: x["score"], reverse=True)

        def kind(item):
            label = item["label"].lower().replace("_", "-")
            if any(w in label for w in ("ai", "artificial", "synthetic", "fake", "generated")):
                return "ai"
            if any(w in label for w in ("real", "human", "authentic", "natural")):
                return "real"
            # Imager-1's model card maps class 0 to AI-generated and class 1 to Real.
            if label in {"label-0", "0", "class-0"}:
                return "ai"
            if label in {"label-1", "1", "class-1"}:
                return "real"
            return "unknown"

        ai = next((x["score"] for x in results if kind(x) == "ai"), None)
        real = next((x["score"] for x in results if kind(x) == "real"), None)
        if ai is None or real is None:
            return {"model_repo": MODEL_ID, "label": results[0]["label"],
                    "confidence": round(results[0]["score"] * 100, 2),
                    "scores": [{"label": x["label"], "score": round(x["score"] * 100, 2)}
                               for x in results],
                    "note": "Could not identify AI/Real labels; verify this model's label mapping."}
        is_ai = ai >= real
        return {"model_repo": MODEL_ID, "label": "AI-Generated" if is_ai else "Real Image",
                "isAI": is_ai, "confidence": round(max(ai, real) * 100, 2),
                "aiScore": round(ai * 100, 2), "realScore": round(real * 100, 2)}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)[:500]}")
