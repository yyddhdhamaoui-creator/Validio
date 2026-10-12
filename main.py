import io
import logging
import os
import threading
import time
import gc
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

# ============================================================
# CONFIGURACIÓN DEL CACHÉ DE MODELOS
# ============================================================
# Tiempo (segundos) que un modelo se queda cargado tras la última petición
MODEL_TTL_SECONDS = int(os.getenv("MODEL_TTL_SECONDS", "300"))  # 5 minutos

# Caché: {repo_id: {"model": m, "transform": t, "last_used": timestamp}}
MODELS_CACHE = {}
cache_lock = threading.Lock()
LOAD_LOCK = threading.Lock()

# Lista blanca de modelos permitidos
ALLOWED_MODELS = {
    "imagers/imager-1",
    "imagers/imager-1-tiny",
    "imagers/imager-1-pro",
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


ALLOWED_NORMALIZED = {normalize_repo(m) for m in ALLOWED_MODELS}


def detect_architecture(state_dict):
    keys = set(state_dict.keys())
    if "layer1.0.conv1.weight" in keys or "fc.weight" in keys:
        return "resnet50"
    if "features.0.0.weight" in keys or "classifier.2.weight" in keys:
        return "convnext_small"
    return "convnext_small"


def unload_model(repo_id: str):
    """Libera la RAM del modelo."""
    with cache_lock:
        if repo_id in MODELS_CACHE:
            del MODELS_CACHE[repo_id]
            log.info("🗑️ Modelo descargado de RAM: %s", repo_id)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models():
    """Libera TODOS los modelos de RAM."""
    with cache_lock:
        loaded = list(MODELS_CACHE.keys())
        MODELS_CACHE.clear()
    for m in loaded:
        log.info("🗑️ Modelo descargado: %s", m)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_worker():
    """Hilo que cada 60s libera modelos inactivos."""
    while True:
        try:
            time.sleep(60)
            now = time.time()
            to_unload = []
            with cache_lock:
                for repo_id, entry in MODELS_CACHE.items():
                    if now - entry["last_used"] > MODEL_TTL_SECONDS:
                        to_unload.append(repo_id)
            for repo_id in to_unload:
                unload_model(repo_id)
        except Exception as e:
            log.error("Cleanup worker error: %s", e)


def _torch_load(path):
    """Carga los pesos con mmap para no duplicar la RAM al leer el archivo."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_by_repo(repo_id: str):
    """
    Carga un modelo y lo cachea. Solo 1 modelo en RAM a la vez.
    Devuelve (model, transform).
    """
    repo_id = normalize_repo(repo_id)

    if repo_id not in ALLOWED_NORMALIZED:
        raise ValueError(f"Model '{repo_id}' not allowed.")

    with LOAD_LOCK:
        with cache_lock:
            if repo_id in MODELS_CACHE:
                MODELS_CACHE[repo_id]["last_used"] = time.time()
                entry = MODELS_CACHE[repo_id]
                return entry["model"], entry["transform"]

            if MODELS_CACHE:
                log.info("🔄 Cambiando de modelo. Descargando anteriores...")
                MODELS_CACHE.clear()

        gc.collect()

        log.info("📥 Cargando modelo en RAM: %s", repo_id)
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = _torch_load(weights_path)

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
        del state_dict
        gc.collect()
        m.eval()

        p = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        with cache_lock:
            MODELS_CACHE[repo_id] = {
                "model": m,
                "transform": p,
                "last_used": time.time(),
            }

        log.info("✅ Modelo cargado en RAM: %s (%s)", repo_id, arch)
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
        edges = np.concatenate([
            mag[:h//4, :].flatten(), mag[3*h//4:, :].flatten(),
            mag[:, :w//4].flatten(), mag[:, 3*w//4:].flatten(),
        ])
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


# ============================================================
# APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=cleanup_worker, daemon=True).start()
    log.info("🚀 Backend iniciado (lazy loading activado, TTL=%ds)", MODEL_TTL_SECONDS)
    yield
    unload_all_models()


app = FastAPI(title="Imager Model API", version="3.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
)


@app.get("/")
def root():
    with cache_lock:
        loaded = list(MODELS_CACHE.keys())
    return {
        "service": "Imager Model API",
        "loaded_models": loaded,
        "allowed_models": sorted(ALLOWED_NORMALIZED),
        "ttl_seconds": MODEL_TTL_SECONDS,
    }


@app.get("/health")
def health():
    with cache_lock:
        loaded = list(MODELS_CACHE.keys())
    return {"status": "ok", "loaded_models": loaded}


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
        with torch.inference_mode():
            tensor = p(image).unsqueeze(0)
            logits = m(tensor)[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        with cache_lock:
            if normalize_repo(model) in MODELS_CACHE:
                MODELS_CACHE[normalize_repo(model)]["last_used"] = time.time()

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
