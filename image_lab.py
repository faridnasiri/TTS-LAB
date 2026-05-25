"""
image_lab.py — FastAPI entry point for the Arthur Image & Video Generation Lab.
Runs on port 8002. Managed by arthur-imglab.service (systemd).

Usage:
    python image_lab.py
    uvicorn image_lab:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ---- Env file support (.env next to this script) ----
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

# ---- Set HF cache dirs BEFORE importing diffusers/transformers ----
hf_home = os.environ.get("HF_HOME", "/opt/arthur-img-models/huggingface")
os.environ.setdefault("HF_HOME",            hf_home)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_home)

# ---- Logging ----
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("image_lab")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from image_lab_config import HOST, PORT
from image_lab_dispatch import router
from image_lab_engines import probe_availability
from image_lab_ui import get_ui_html
from image_lab_utils import ensure_dirs

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Arthur Image & Video Lab starting ===")
    ensure_dirs()
    probe_availability()
    log.info("=== Image Lab ready on port %d ===", PORT)
    yield
    log.info("=== Image Lab shutting down ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Arthur Image & Video Lab",
    version     = "1.0.0",
    description = "FLUX.2 [dev] · SD 3.5 Large · Wan2.2  — port 8002",
    lifespan    = lifespan,
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(get_ui_html())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "image_lab:app",
        host        = HOST,
        port        = PORT,
        log_level   = "info",
        access_log  = True,
        reload      = False,
    )
