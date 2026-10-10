import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ExifTags, UnidentifiedImageError
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from config import MODEL_REPO, MAX_IMAGE_MB

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("imager-api")

# Caché de modelos: {repo_id: (model, transform)}
MODELS_CACHE = {}
model_lock = threading.Lock()

# Lista blanca de modelos permitidos
ALLOWED_MODELS = {
    "imagers/imager-1",
    "imagers/imager-1-tiny",
    "imagers/imager-1-pro",
    "imagers/imager-2",
}


def normalize_repo(value: str) -> str:
    value = value.strip().rstrip("/").lower()
    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("URL must point to huggingface.co")
        value = parsed.path.strip("/")
        if value.startswith("models/"):
            value = value[len("models/"):]
    if "/" not in value:
        raise ValueError("Invalid model ID")
    return value


# Normalizar los permitidos
ALLOWED_NORMALIZED = {normalize_repo(m) for m in ALLOWED_MODELS}


def detect_architecture(state_dict):
    keys = set(state_dict.keys())
    if "layer1.0.conv1.weight" in keys or "fc.weight" in keys:
        return "resnet50"
    if "features.0.0.weight" in keys or "classifier.2.weight" in keys:
        return "convnext_small"
    return "convnext_small"


def load_model_by_repo(repo_id: str):
    """Carga un modelo y lo cachea. Devuelve (model, transform)."""
    repo_id = normalize_repo(repo_id)

    if repo_id not in ALLOWED_NORMALIZED:
        raise ValueError(f"Model '{repo_id}' is not allowed.")

    if repo_id in MODELS_CACHE:
        return MODELS_CACHE[repo_id]

    log.info("Loading model %s", repo_id)
    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    MODELS_CACHE[repo_id] = (m, p)
    log.info("Loaded %s (%s)", repo_id, arch)
    return m, p


def analyze_technical_signals(image: Image.Image) -> dict:
    signals = {
        "has_exif": False, "exif_fields": [], "format": image.format or "unknown",
        "fft_ratio": None, "noise_std": None, "ai_score": 0, "max_score": 7,
        "verdict": "unknown",
    }
    try:
        exif = image.getexif()
        if exif:
            signals["has_exif"] = True
            for tag_id, value in list(exif.items())[:5]:
                signals["exif_fields"].append(f"{ExifTags.TAGS.get(tag_id, tag_id)}: {str(value)[:60]}")
    except Exception:
        pass

    try:
        gray = image.convert("L").resize((256, 256))
        arr = np.array(gray, dtype=np.float32)
        mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(arr))) + 1)
        h, w = mag.shape
        center = mag[h//4:3*h//4, w//4:3*w//4]
        edges = np.concatenate([mag[:h//4, :].flatten(), mag[3*h//4:, :].flatten(),
                                mag[:, :w//4].flatten(), mag[:, 3*w//4:].flatten()])
        signals["fft_ratio"] = round(float(edges.mean() / (center.mean() + 1e-8)), 4)
    except Exception:
        pass

    try:
        gray = image.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
        noise = np.array(gray, dtype=np.float32) - np.array(blurred, dtype=np.float32)
        signals["noise_std"] = round(float(noise.std()), 2)
    except Exception:
        pass

    score = 0
    if not signals["has_exif"]: score += 2
    if signals["format"] == "PNG": score += 1
    if signals["fft_ratio"] and signals["fft_ratio"] > 0.5: score += 2
    if signals["noise_std"] and signals["noise_std"] < 2: score += 2
    signals["ai_score"] = score
    signals["verdict"] = "likely_ai" if score >= 5 else ("possibly_ai" if score >= 3 else "likely_real")
    return signals


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precargar el modelo por defecto en background
    def preload():
        try:
            load_model_by_repo(MODEL_REPO)
        except Exception as e:
            log.exception("Failed to preload default model: %s", e)
    threading.Thread(target=preload, daemon=True).start()
    yield


app = FastAPI(title="Imager Model API", version="2.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.get("/")
def root():
    loaded = list(MODELS_CACHE.keys())
    return {"service": "Imager Model API", "default_model": MODEL_REPO,
            "loaded_models": loaded, "allowed_models": sorted(ALLOWED_NORMALIZED)}


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS_CACHE.keys())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    visitor_id: str = Form("anonymous"),
    model: str = Form(MODEL_REPO),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Max {MAX_IMAGE_MB} MB.")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        technical = analyze_technical_signals(image)
    except Exception as e:
        technical = {"error": str(e)[:200]}

    try:
        m, p = load_model_by_repo(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Model load failed")
        raise HTTPException(status_code=503, detail=f"Model '{model}' unavailable: {str(e)[:200]}")

    try:
        with model_lock, torch.inference_mode():
            tensor = p(image).unsqueeze(0)
            logits = m(tensor)[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        ai_score = probs[0]
        real_score = probs[1]
        is_ai = ai_score >= real_score

        return {
            "model_repo": model,
            "label": "AI-Generated" if is_ai else "Real Image",
            "isAI": is_ai,
            "confidence": round(max(ai_score, real_score) * 100, 2),
            "aiScore": round(ai_score * 100, 2),
            "realScore": round(real_score * 100, 2),
            "technical_signals": technical,
        }
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)[:200]}")
