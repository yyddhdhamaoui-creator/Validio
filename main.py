import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ExifTags, UnidentifiedImageError
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

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
        raise ValueError("Use a Hugging Face model ID such as Imagers/imager-1")
    return value


MODEL_ID = normalize_model_repo(MODEL_REPO)


# ============================================================
# MINI HERRAMIENTA DE ANÁLISIS TÉCNICO
# ============================================================
def analyze_technical_signals(image: Image.Image) -> dict:
    """Analiza metadatos, FFT y ruido de la imagen. Devuelve un dict con las señales."""

    signals = {
        "has_exif": False,
        "exif_fields": [],
        "format": image.format or "unknown",
        "fft_ratio": None,
        "noise_std": None,
        "ai_score": 0,
        "max_score": 7,
        "verdict": "unknown",
    }

    # 1. EXIF
    try:
        exif = image.getexif()
        if exif:
            signals["has_exif"] = True
            for tag_id, value in list(exif.items())[:5]:
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                signals["exif_fields"].append(f"{tag_name}: {str(value)[:60]}")
    except Exception:
        pass

    # 2. FFT (frecuencia)
    try:
        gray = image.convert("L").resize((256, 256))
        arr = np.array(gray, dtype=np.float32)
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        h, w = magnitude.shape
        center = magnitude[h//4:3*h//4, w//4:3*w//4]
        edges = np.concatenate([
            magnitude[:h//4, :].flatten(),
            magnitude[3*h//4:, :].flatten(),
            magnitude[:, :w//4].flatten(),
            magnitude[:, 3*w//4:].flatten(),
        ])
        ratio = float(edges.mean() / (center.mean() + 1e-8))
        signals["fft_ratio"] = round(ratio, 4)
    except Exception:
        pass

    # 3. Ruido
    try:
        gray = image.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
        noise = np.array(gray, dtype=np.float32) - np.array(blurred, dtype=np.float32)
        signals["noise_std"] = round(float(noise.std()), 2)
    except Exception:
        pass

    # 4. Puntuación (más alto = más probable IA)
    score = 0
    if not signals["has_exif"]:
        score += 2
    if signals["format"] == "PNG":
        score += 1
    if signals["fft_ratio"] is not None and signals["fft_ratio"] > 0.5:
        score += 2
    if signals["noise_std"] is not None and signals["noise_std"] < 2:
        score += 2

    signals["ai_score"] = score
    if score >= 5:
        signals["verdict"] = "likely_ai"
    elif score >= 3:
        signals["verdict"] = "possibly_ai"
    else:
        signals["verdict"] = "likely_real"

    return signals


# ============================================================
# CARGA DEL MODELO
# ============================================================
def detect_architecture(state_dict):
    """Detecta ConvNeXt o ResNet por las claves del state_dict."""
    keys = set(state_dict.keys())

    if "layer1.0.conv1.weight" in keys or "fc.weight" in keys:
        return "resnet50"

    if "features.0.0.weight" in keys or "classifier.2.weight" in keys:
        return "convnext_small"

    return "convnext_small"


def load_classifier():
    global processor, model, load_error
    try:
        log.info("Loading model %s", MODEL_ID)
        weights_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="pytorch_model.bin",
            revision=MODEL_REVISION or None,
        )

        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        arch = detect_architecture(state_dict)
        log.info("Detected architecture: %s", arch)

        if arch == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
        else:
            m = models.convnext_small(weights=None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)

        m.load_state_dict(state_dict, strict=True)
        m.eval()

        p = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        processor, model, load_error = p, m, None
        log.info("Model loaded successfully (%s)", arch)

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
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    status = "ready" if model is not None else ("error" if load_error else "loading")
    return {
        "service": "Imager Model API",
        "model_repo": MODEL_ID,
        "status": status,
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }


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
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    # === Análisis técnico (mini herramienta) ===
    try:
        technical = analyze_technical_signals(image)
    except Exception as exc:
        log.warning("Technical analysis failed: %s", exc)
        technical = {"error": str(exc)[:200]}

    # === Predicción del modelo ===
    try:
        with model_lock, torch.inference_mode():
            tensor = processor(image).unsqueeze(0)
            logits = model(tensor)[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        # Clase 0 = AI, Clase 1 = Real
        ai_score = probs[0]
        real_score = probs[1]
        is_ai = ai_score >= real_score

        return {
            "model_repo": MODEL_ID,
            "label": "AI-Generated" if is_ai else "Real Image",
            "isAI": is_ai,
            "confidence": round(max(ai_score, real_score) * 100, 2),
            "aiScore": round(ai_score * 100, 2),
            "realScore": round(real_score * 100, 2),
            "technical_signals": technical,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)[:500]}")
