import io
import logging
import os
import threading
import time
import gc
import math
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ExifTags, UnidentifiedImageError
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from config import MODEL_REPO, MAX_IMAGE_MB

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("imager-api")


# ============================================================
# HILOS DE CPU
# ============================================================
# En Railway torch ve todos los núcleos del servidor físico, pero el contenedor
# solo tiene unos pocos vCPU. Eso provoca sobresuscripción y la inferencia pasa
# de ~1 s a ~60 s. Limitamos los hilos a la cuota real del contenedor.
def detect_cpu_limit() -> int:
    env = os.getenv("TORCH_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except Exception:
        try:  # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if quota > 0:
                n = min(n, max(1, math.ceil(quota / period)))
        except Exception:
            pass
    return max(1, n)


CPU_THREADS = detect_cpu_limit()
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
log.info("🧵 Hilos de torch: %d", CPU_THREADS)

# ============================================================
# CONFIGURACIÓN DEL CACHÉ DE MODELOS
# ============================================================
# Tiempo (segundos) que un modelo se queda cargado tras la última petición
MODEL_TTL_SECONDS = int(os.getenv("MODEL_TTL_SECONDS", "900"))  # 15 min: evita recargas de ~1 min
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "30"))
# Modelo que se carga en segundo plano al arrancar ("" para desactivar)
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", MODEL_REPO)

# Caché: {repo_id: {"model": m, "transform": t, "last_used": ts, "busy": n}}
MODELS_CACHE = {}
cache_lock = threading.Lock()
LOAD_LOCK = threading.Lock()
# Solo una inferencia a la vez: evita pelear por la poca CPU disponible.
INFER_LOCK = threading.Lock()

# Para usar bfloat16 (mitad de RAM): MODEL_DTYPE=bfloat16
MODEL_DTYPE = (
    torch.float32
    if os.getenv("MODEL_DTYPE", "float32").lower() == "float32"
    else torch.bfloat16
)

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


def finite_or_none(value):
    """Devuelve el número solo si es finito; JSON no admite NaN ni infinito."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def release_memory():
    """Devuelve al sistema la RAM liberada tras descargar un modelo (glibc)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_first_nonfinite_layer(m, tensor):
    """Diagnóstico: repite la inferencia con hooks y devuelve la primera capa que produce NaN/Inf."""
    found = []
    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, out):
            if not found and torch.is_tensor(out) and out.is_floating_point() and not bool(torch.isfinite(out).all()):
                found.append(name)
        return hook

    for name, module in m.named_modules():
        if not list(module.children()):  # solo capas hoja
            hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            m(tensor)
    finally:
        for h in hooks:
            h.remove()
    return found[0] if found else None


def clean_json(obj):
    """Recorre la respuesta completa y reemplaza NaN/infinito por None."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def detect_architecture(state_dict):
    keys = set(state_dict.keys())
    if "layer1.0.conv1.weight" in keys or "fc.weight" in keys:
        return "resnet50"
    if "features.0.0.weight" in keys:
        stem_channels = state_dict["features.0.0.weight"].shape[0]
        if stem_channels == 128:
            return "convnext_base"
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


def set_busy(repo_id: str, delta: int):
    """Marca un modelo como en uso para que el limpiador no lo descargue a mitad de petición."""
    with cache_lock:
        entry = MODELS_CACHE.get(repo_id)
        if entry is not None:
            entry["busy"] = max(0, entry.get("busy", 0) + delta)
            entry["last_used"] = time.time()


def cleanup_worker():
    """Hilo que libera periódicamente los modelos inactivos."""
    while True:
        try:
            time.sleep(CLEANUP_INTERVAL)
            now = time.time()
            to_unload = []
            with cache_lock:
                for repo_id, entry in MODELS_CACHE.items():
                    if entry.get("busy", 0) > 0:
                        continue
                    if now - entry["last_used"] > MODEL_TTL_SECONDS:
                        to_unload.append(repo_id)
            for repo_id in to_unload:
                unload_model(repo_id)
            if to_unload:
                release_memory()
        except Exception as e:
            log.error("Cleanup worker error: %s", e)


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
        # mmap=True: el archivo se lee de disco bajo demanda y NO se duplica en RAM.
        # Sin assign=True: los pesos se COPIAN a un modelo float32 propio, así el modelo
        # no depende de páginas mapeadas (esto evita los NaN). Resultado idéntico, menos pico de RAM.
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Diagnóstico: ¿el checkpoint trae NaN/Inf en sus pesos?
        ckpt_dtypes = sorted({str(v.dtype).replace("torch.", "") for v in state_dict.values() if torch.is_tensor(v)})
        bad_tensors = [
            k for k, v in state_dict.items()
            if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())
        ]
        log.info("🔎 %s: dtypes del checkpoint = %s", repo_id, ckpt_dtypes)
        if bad_tensors:
            log.error("⚠️ %d tensores con NaN/Inf en el checkpoint de %s. Primeros: %s",
                      len(bad_tensors), repo_id, bad_tensors[:8])

        arch = detect_architecture(state_dict)

        if arch == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
        elif arch == "convnext_base":
            m = models.convnext_base(weights=None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        else:
            m = models.convnext_small(weights=None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)

        m = m.float()
        m.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        m.eval()
        if MODEL_DTYPE != torch.float32:
            m = m.to(MODEL_DTYPE)

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
                "busy": 0,
            }

        release_memory()  # devuelve al sistema la RAM temporal usada durante la carga
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

    signals["fft_ratio"] = finite_or_none(signals["fft_ratio"])
    signals["noise_std"] = finite_or_none(signals["noise_std"])

    score = 0
    if not signals["has_exif"]: score += 2
    if signals["format"] == "PNG": score += 1
    if signals["fft_ratio"] and signals["fft_ratio"] > 0.5: score += 2
    if signals["noise_std"] and signals["noise_std"] < 2: score += 2
    signals["ai_score"] = score
    signals["verdict"] = "likely_ai" if score >= 5 else ("possibly_ai" if score >= 3 else "likely_real")
    return signals


def run_prediction(model: str, image: Image.Image, fmt: str) -> dict:
    """Parte pesada (carga + inferencia). Se ejecuta en un hilo, NO en el event loop."""
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

    repo = normalize_repo(model)
    set_busy(repo, +1)
    try:
        with INFER_LOCK:
            t_inf = time.time()
            with torch.inference_mode():
                tensor = p(image).unsqueeze(0).to(next(m.parameters()).dtype)
                logits = m(tensor)[0].float()
                probs = torch.softmax(logits, dim=-1).tolist()
            log.info("⏱️ %s: inferencia %.1fs", model, time.time() - t_inf)

        if len(probs) < 2 or not all(math.isfinite(x) for x in probs):
            input_ok = bool(torch.isfinite(tensor).all())
            try:
                bad_layer = find_first_nonfinite_layer(m, tensor)
            except Exception as diag_err:
                bad_layer = f"(no se pudo diagnosticar: {diag_err})"
            log.error(
                "Salida no válida del modelo %s: %s | entrada finita: %s | primera capa con NaN/Inf: %s",
                model, probs, input_ok, bad_layer,
            )
            raise HTTPException(
                status_code=422,
                detail="This model returned an invalid result. Try another model.",
            )

        ai_score = probs[0]
        real_score = probs[1]
        is_ai = ai_score >= real_score

        return clean_json({
            "model_repo": model,
            "label": "AI-Generated" if is_ai else "Real Image",
            "isAI": is_ai,
            "confidence": round(max(ai_score, real_score) * 100, 2),
            "aiScore": round(ai_score * 100, 2),
            "realScore": round(real_score * 100, 2),
            "technical_signals": technical,
        })
    finally:
        set_busy(repo, -1)


# ============================================================
# APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=cleanup_worker, daemon=True).start()

    def _preload():
        try:
            if PRELOAD_MODEL:
                load_model_by_repo(PRELOAD_MODEL)
        except Exception as e:
            log.error("Preload failed: %s", e)

    threading.Thread(target=_preload, daemon=True).start()
    log.info("🚀 Backend iniciado (lazy loading activado, TTL=%ds)", MODEL_TTL_SECONDS)
    yield
    unload_all_models()


app = FastAPI(title="Imager Model API", version="3.1.0", lifespan=lifespan)
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
        "torch_threads": CPU_THREADS,
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
        image = Image.open(io.BytesIO(data))
        fmt = image.format or "unknown"
        image = image.convert("RGB")
        image.load()
        image.format = fmt  # convert() pierde el formato original
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        # En un hilo aparte: así el servidor sigue respondiendo (/health, CORS, otras
        # peticiones) mientras se carga el modelo o se hace la inferencia.
        return await run_in_threadpool(run_prediction, model, image, fmt)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)[:200]}")
