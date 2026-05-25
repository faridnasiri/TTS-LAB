"""
image_lab_utils.py — Shared utilities: VRAM stats, image/video saving,
gallery index (JSON), and slug generation.
"""

from __future__ import annotations
import gc
import io
import json
import os
import time
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("image_lab")

# ---------------------------------------------------------------------------
# Lazy imports so the module loads even without torch installed
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch

# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def ensure_dirs():
    from image_lab_config import IMAGES_DIR, VIDEOS_DIR, OUTPUT_ROOT, GALLERY_DB
    for d in (IMAGES_DIR, VIDEOS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    if not Path(GALLERY_DB).exists():
        Path(GALLERY_DB).write_text("[]", encoding="utf-8")

# ---------------------------------------------------------------------------
# VRAM stats
# ---------------------------------------------------------------------------

def vram_stats() -> dict:
    """Return current VRAM usage in GB (allocated and total)."""
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return {"available": False, "allocated_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0)  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free      = total - reserved
        return {
            "available":    True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb":  round(reserved, 2),
            "total_gb":     round(total, 2),
            "free_gb":      round(free, 2),
            "device_name":  torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def free_vram():
    """Release all cached VRAM."""
    try:
        torch = _torch()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_image(pil_image, engine_key: str, params: dict) -> dict:
    """
    Save a PIL image to disk and return metadata dict with url + base64.
    """
    from image_lab_config import IMAGES_DIR

    gen_id   = str(uuid.uuid4())
    filename = f"{engine_key}_{gen_id}.png"
    filepath = os.path.join(IMAGES_DIR, filename)

    pil_image.save(filepath, format="PNG", optimize=False)

    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    entry = {
        "id":         gen_id,
        "engine":     engine_key,
        "filename":   filename,
        "url":        f"/files/images/{filename}",
        "base64":     b64,
        "type":       "image",
        "width":      pil_image.width,
        "height":     pil_image.height,
        "params":     _strip_file_params(params),
        "created_at": time.time(),
    }
    _append_gallery(entry)
    log.info("Saved image %s (%dx%d)", filename, pil_image.width, pil_image.height)
    return entry


def save_images(pil_images: list, engine_key: str, params: dict) -> list[dict]:
    """Save multiple PIL images and return a list of metadata dicts."""
    return [save_image(img, engine_key, params) for img in pil_images]

# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_video(frames, fps: int, engine_key: str, params: dict) -> dict:
    """
    Save a list of PIL/numpy frames as an MP4 to disk.
    Returns metadata dict with url (no base64 — videos are too large).
    """
    from image_lab_config import VIDEOS_DIR

    gen_id   = str(uuid.uuid4())
    filename = f"{engine_key}_{gen_id}.mp4"
    filepath = os.path.join(VIDEOS_DIR, filename)

    _write_mp4(frames, fps, filepath)

    entry = {
        "id":         gen_id,
        "engine":     engine_key,
        "filename":   filename,
        "url":        f"/files/videos/{filename}",
        "base64":     None,   # videos not base64-encoded
        "type":       "video",
        "fps":        fps,
        "num_frames": len(frames),
        "params":     _strip_file_params(params),
        "created_at": time.time(),
    }
    _append_gallery(entry)
    log.info("Saved video %s (%d frames @ %d fps)", filename, len(frames), fps)
    return entry


def _write_mp4(frames, fps: int, filepath: str):
    """Write frames to MP4 using imageio (ffmpeg backend)."""
    import imageio
    import numpy as np

    np_frames = []
    for f in frames:
        if hasattr(f, "numpy"):          # torch tensor
            arr = f.numpy()
        elif hasattr(f, "__array__"):    # PIL Image or numpy array
            arr = np.array(f)
        else:
            arr = f
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        np_frames.append(arr)

    writer = imageio.get_writer(filepath, fps=fps, codec="libx264",
                                 output_params=["-crf", "23", "-preset", "fast"])
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()

# ---------------------------------------------------------------------------
# Gallery index (flat JSON list, append-only, last 500 entries)
# ---------------------------------------------------------------------------

def _append_gallery(entry: dict):
    from image_lab_config import GALLERY_DB
    try:
        path = Path(GALLERY_DB)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.append(entry)
        data = data[-500:]   # keep last 500 entries
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Gallery write failed: %s", exc)


def read_gallery(limit: int = 50, offset: int = 0,
                 engine_filter: Optional[str] = None) -> list[dict]:
    from image_lab_config import GALLERY_DB
    try:
        data = json.loads(Path(GALLERY_DB).read_text(encoding="utf-8"))
        data = list(reversed(data))   # newest first
        if engine_filter:
            data = [e for e in data if e.get("engine") == engine_filter]
        # Strip base64 from gallery listings to keep payload small
        for e in data:
            e.pop("base64", None)
        return data[offset: offset + limit]
    except Exception:
        return []


def delete_gallery_entry(gen_id: str) -> bool:
    from image_lab_config import GALLERY_DB, IMAGES_DIR, VIDEOS_DIR
    try:
        path = Path(GALLERY_DB)
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = next((e for e in data if e["id"] == gen_id), None)
        if entry is None:
            return False
        data = [e for e in data if e["id"] != gen_id]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Delete file too
        sub = IMAGES_DIR if entry["type"] == "image" else VIDEOS_DIR
        fp  = os.path.join(sub, entry["filename"])
        if os.path.exists(fp):
            os.remove(fp)
        return True
    except Exception as exc:
        log.warning("Gallery delete failed: %s", exc)
        return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_file_params(params: dict) -> dict:
    """Remove any binary/file values from params before storing in gallery."""
    return {k: v for k, v in params.items()
            if not isinstance(v, (bytes, bytearray))}


def image_to_base64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def random_seed() -> int:
    import random
    return random.randint(0, 2**31 - 1)
