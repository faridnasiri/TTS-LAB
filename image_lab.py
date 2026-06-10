"""
image_lab.py — FastAPI entry point for the Arthur Image & Video Generation Lab.
Runs on port 8002. Managed by arthur-imglab.service (systemd).

Usage:
    python image_lab.py
    uvicorn image_lab:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations
import asyncio
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
import collections

class _RingLogHandler(logging.Handler):
    """Captures last N log records in memory for the /logs API."""
    def __init__(self, maxlen=200):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        from datetime import datetime
        self.buffer.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        })

    def snapshot(self) -> list[dict]:
        return list(self.buffer)

_ring_handler = _RingLogHandler(maxlen=200)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("image_lab")
# Also attach ring handler so /logs endpoint sees our messages
log.addHandler(_ring_handler)

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
    # Pre-load ideogram4 model in background so first request is instant
    asyncio.create_task(_preload_ideogram4())
    log.info("=== Image Lab ready on port %d ===", PORT)
    yield
    log.info("=== Image Lab shutting down ===")


async def _preload_ideogram4():
    """Pre-load Ideogram 4 in a background thread so first API call is instant."""
    import image_lab_engines as engines
    from image_lab_config import ENGINES, STATE
    if not ENGINES.get("ideogram4") or not ENGINES["ideogram4"].available:
        return
    if ENGINES["ideogram4"].loaded or STATE.loading:
        return  # Already loaded or loading in progress

    # Free GPU memory from TTS service (port 8000) before loading
    import subprocess
    log.info("Restarting TTS service to free GPU memory for Ideogram 4...")
    subprocess.run(["sudo", "systemctl", "restart", "arthur.service"],
                   capture_output=True, timeout=30)
    import time; time.sleep(3)

    log.info("Pre-loading Ideogram 4 model (background thread, ~6-8 min)...")
    try:
        await asyncio.to_thread(engines._load_ideogram4, "nf4")
        log.info("Ideogram 4 pre-loaded — %.0f MiB VRAM used, ready for requests",
                 __import__('torch').cuda.memory_allocated() / 1024**2)
    except Exception as e:
        log.warning("Ideogram 4 pre-load failed (will load on first request): %s", e)


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


@app.get("/logs")
async def get_logs(n: int = 100):
    """Return the most recent N log entries from the ring buffer."""
    entries = _ring_handler.snapshot()
    return {"count": len(entries), "logs": entries[-n:]}


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
