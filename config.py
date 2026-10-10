import os

# Hugging Face model ID or full URL. Change this to use another compatible model.
MODEL_REPO = os.getenv("MODEL_REPO", "Imagers/imager-1-tiny").strip()
MODEL_REVISION = os.getenv("MODEL_REVISION", "").strip()
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
