"""
image_lab_utils.py — Shared utilities: VRAM stats, image/video saving,
gallery index (JSON), and slug generation.
"""

from __future__ import annotations
import gc
import io
import json
import os
import re
import subprocess
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
# Device-wide GPU stats — nvidia-smi based, TTL-cached. Unlike vram_stats()
# (this process's torch-allocator view), gpu_stats() reports the whole card as
# the driver sees it — including the TTS engine containers and any other CUDA
# processes sharing the GPU — plus a per-process attribution list.
# ---------------------------------------------------------------------------

_GPU_STATS_CACHE: dict = {"ts": 0.0, "data": None}
_DOCKER_NAMES_CACHE: dict = {"ts": 0.0, "map": {}}


def gpu_stats(ttl: float = 2.0) -> dict:
    """Device-wide GPU stats for the VRAM report (MB ints).

    Runs nvidia-smi at most once per `ttl` seconds. Falls back to the torch
    view from vram_stats() when nvidia-smi is unavailable (non-GPU host).
    """
    now = time.time()
    if _GPU_STATS_CACHE["data"] and (now - _GPU_STATS_CACHE["ts"]) < ttl:
        return _GPU_STATS_CACHE["data"]
    stats = _gpu_stats_impl()
    _GPU_STATS_CACHE["ts"], _GPU_STATS_CACHE["data"] = now, stats
    return stats


def _gpu_stats_impl() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            raise RuntimeError("nvidia-smi query failed")
        head = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        parts = [s.strip() for s in head.split(",")] if head else ["", "0", "0", "0"]
        name, total_mb, used_mb, free_mb = parts[0], int(parts[1]), int(parts[2]), int(parts[3])

        processes: list[dict] = []
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if apps.returncode == 0:
            for line in apps.stdout.splitlines():
                cols = [c.strip() for c in line.split(",", 2)]
                if len(cols) < 3:
                    continue
                try:
                    pid = int(cols[0])
                    mb  = int(cols[1].split()[0])
                except ValueError:
                    continue
                container = _proc_container_name(pid) if mb > 0 else ""
                processes.append({
                    "pid": pid, "mb": mb, "process": cols[2],
                    "container": container,   # tts-lab-* name or "" for host procs
                })
        processes.sort(key=lambda p: -p["mb"])
        return {
            "available":    True,
            "name":         name,
            "vram_total_mb": total_mb,
            "vram_used_mb": used_mb,
            "vram_free_mb": free_mb,
            "source":       "nvidia-smi",
            "processes":    processes,
            "ts":           time.time(),
        }
    except Exception as exc:
        # Fallback — this process's torch view (GB floats → MB ints)
        v = vram_stats()
        if v.get("available"):
            return {
                "available":     True,
                "name":          v.get("device_name", ""),
                "vram_total_mb": round(v["total_gb"] * 1024),
                "vram_used_mb":  round(v["reserved_gb"] * 1024),
                "vram_free_mb":  round(v["free_gb"] * 1024),
                "source":        "torch",
                "processes":     [],
                "ts":            time.time(),
            }
        return {"available": False, "processes": [], "error": str(exc), "ts": time.time()}


def _docker_name_map(max_age: float = 5.0) -> dict:
    """Map container id-prefix → container name via `docker ps` (TTL-cached)."""
    now = time.time()
    cache = _DOCKER_NAMES_CACHE
    if cache["map"] and (now - cache["ts"]) < max_age:
        return cache["map"]
    names: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                cid, _, cname = line.partition("\t")
                if cid:
                    names[cid[:12].lower()] = cname
    except Exception:
        pass
    cache["map"], cache["ts"] = names, time.time()
    return names


def _proc_container_name(pid: int) -> str:
    """Container name owning host pid `pid` ('' = host process)."""
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return ""
    for m in re.finditer(r"docker[-/]([0-9a-f]{12,64})", text):
        return _docker_name_map().get(m.group(1)[:12].lower(), "")
    return ""


def system_stats() -> dict:
    """Host RAM {total, used, free} in MB — psutil, else /proc/meminfo."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": vm.total >> 20,
            "used":  (vm.total - vm.available) >> 20,
            "free":  vm.available >> 20,
        }
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    info[k] = int(v.strip().split()[0])   # kB
        if "MemTotal" in info:
            total = info["MemTotal"] >> 10
            free  = info.get("MemAvailable", info["MemTotal"]) >> 10
            return {"total": total, "used": total - free, "free": free}
    except Exception:
        pass
    return {"total": 0, "used": 0, "free": 0}


# ---------------------------------------------------------------------------
# Run-stats stamping — save_image/save_video record when the generation that
# produced the file started/finished (STATE.run_started is stamped by
# engines.generate() around load + inference).
# ---------------------------------------------------------------------------

def _stamp_run_stats(entry: dict) -> None:
    """Merge run-timing stats into a saved entry (no-op outside a generate())."""
    try:
        from image_lab_config import STATE
        if not STATE.run_started:
            return
        now = time.time()
        load_s = None
        if STATE.run_loaded_at:
            load_s = round(max(STATE.run_loaded_at - STATE.run_started, 0.0), 2)
        entry["stats"] = {
            "started_at":  round(STATE.run_started, 3),
            "finished_at": round(now, 3),
            "load_s":      load_s,
            "total_s":     round(max(now - STATE.run_started, 0.0), 2),
        }
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
    _stamp_run_stats(entry)
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
    _stamp_run_stats(entry)
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
        # The gallery DB must never persist base64 blobs: it is rewritten in
        # full on every save, so storing each entry's ~1.5 MB base64 grew the
        # file unboundedly (measured 1.26 GB at ~750 entries) and every
        # generation paid a synchronous full-file read+parse+rewrite — a
        # ~15 s stall between the stats stamp and the HTTP response. The
        # live response copy keeps its base64; gallery media loads by /files
        # URL and read_gallery() already strips base64 from listings. Old
        # entries are stripped too — self-healing migration for DBs written
        # before this fix.
        for old in data:
            old.pop("base64", None)
        store = dict(entry)
        store.pop("base64", None)
        data.append(store)
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
