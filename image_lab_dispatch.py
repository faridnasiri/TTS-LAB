"""
image_lab_dispatch.py — FastAPI route handlers for the Image & Video Lab.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from image_lab_config import (
    ENGINES, STATE, IMAGES_DIR, VIDEOS_DIR,
)
from image_lab_utils import read_gallery, delete_gallery_entry, vram_stats
import image_lab_engines as engines

log     = logging.getLogger("image_lab")
router  = APIRouter()

# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@router.get("/status")
async def status():
    vram = vram_stats()
    engine_list = []
    for key, eng in ENGINES.items():
        engine_list.append({
            "key":         key,
            "label":       eng.label,
            "description": eng.description,
            "output_type": eng.output_type,
            "vram_gb":     eng.vram_gb,
            "available":   eng.available,
            "loaded":      eng.loaded,
            "error":       eng.error,
            "params":      eng.params,
        })
    return {
        "engines":        engine_list,
        "active_engine":  STATE.active_engine,
        "active_quant":   STATE.active_quant,
        "generating":     STATE.generating,
        "loading":        STATE.loading,
        "vram":           vram,
    }

# ---------------------------------------------------------------------------
# /generate/{engine}   (multipart — supports optional file upload)
# ---------------------------------------------------------------------------

@router.post("/generate/{engine_key}")
async def generate(
    engine_key:           str,
    prompt:               str           = Form(...),
    negative_prompt:      str           = Form(""),
    width:                int           = Form(1024),
    height:               int           = Form(1024),
    num_inference_steps:  int           = Form(28),
    guidance_scale:       float         = Form(4.0),
    num_images:           int           = Form(1),
    seed:                 int           = Form(-1),
    # Wan-specific
    mode:                 str           = Form("t2v"),
    num_frames:           int           = Form(49),
    fps:                  int           = Form(16),
    resolution:           str           = Form("720p"),
    # Ideogram 4-specific
    preset:               str           = Form("V4_DEFAULT_20"),
    mu:                   float         = Form(0.0),
    std:                  float         = Form(1.75),
    use_magic_prompt:     bool          = Form(False),
    magic_prompt_input:   str           = Form(""),
    magic_prompt_aspect_ratio: str      = Form("1:1"),
    # Optional reference image (FLUX.2 I2I / Wan I2V)
    reference_image:      Optional[UploadFile] = File(None),
    # Quantization format (engine-specific; empty = use engine default)
    quant:                str           = Form(""),
):
    if engine_key not in ENGINES:
        raise HTTPException(404, f"Unknown engine: {engine_key}")

    ref_bytes: Optional[bytes] = None
    if reference_image is not None:
        ref_bytes = await reference_image.read()

    params = {
        "prompt":               prompt,
        "negative_prompt":      negative_prompt,
        "width":                width,
        "height":               height,
        "num_inference_steps":  num_inference_steps,
        "guidance_scale":       guidance_scale,
        "num_images":           num_images,
        "seed":                 seed,
        "mode":                 mode,
        "num_frames":           num_frames,
        "fps":                  fps,
        "resolution":           resolution,
        "reference_image":      ref_bytes,
        "quant":                quant,
        # Ideogram 4-specific
        "preset":               preset,
        "mu":                   mu,
        "std":                  std,
        "use_magic_prompt":     use_magic_prompt,
        "magic_prompt_input":   magic_prompt_input,
        "magic_prompt_aspect_ratio": magic_prompt_aspect_ratio,
    }

    try:
        results = engines.generate(engine_key, params)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Generation error for engine %s", engine_key)
        raise HTTPException(500, f"Generation failed: {exc}")

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# /files/{subdir}/{filename}  — serve saved images and videos
# ---------------------------------------------------------------------------

@router.get("/files/{subdir}/{filename}")
async def serve_file(subdir: str, filename: str):
    if subdir not in ("images", "videos"):
        raise HTTPException(404, "Not found")
    # Prevent path traversal
    safe_name = Path(filename).name
    if subdir == "images":
        filepath = os.path.join(IMAGES_DIR, safe_name)
    else:
        filepath = os.path.join(VIDEOS_DIR, safe_name)

    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")

    media_type = "image/png" if subdir == "images" else "video/mp4"
    return FileResponse(filepath, media_type=media_type)


# ---------------------------------------------------------------------------
# /gallery  — list recent generations
# ---------------------------------------------------------------------------

@router.get("/gallery")
async def gallery(
    limit:  int           = 50,
    offset: int           = 0,
    engine: Optional[str] = None,
):
    entries = read_gallery(limit=limit, offset=offset, engine_filter=engine)
    return {"entries": entries, "limit": limit, "offset": offset}


@router.delete("/gallery/{gen_id}")
async def delete_generation(gen_id: str):
    ok = delete_gallery_entry(gen_id)
    if not ok:
        raise HTTPException(404, "Generation not found")
    return {"deleted": gen_id}


# ---------------------------------------------------------------------------
# /engines/{engine}/load  — preload into VRAM
# ---------------------------------------------------------------------------

@router.post("/engines/{engine_key}/load")
async def load_engine(engine_key: str):
    if engine_key not in ENGINES:
        raise HTTPException(404, f"Unknown engine: {engine_key}")
    if STATE.generating or STATE.loading:
        raise HTTPException(503, "Server is busy")
    try:
        engines.load_engine(engine_key)
    except Exception as exc:
        log.exception("Load error for engine %s", engine_key)
        raise HTTPException(500, str(exc))
    return {"loaded": engine_key}


@router.post("/engines/unload")
async def unload_engine():
    engines.unload_engine()
    return {"unloaded": True}


# ---------------------------------------------------------------------------
# /engines  — list engine metadata (no state)
# ---------------------------------------------------------------------------

@router.get("/engines")
async def list_engines():
    return {
        key: {
            "label":       e.label,
            "description": e.description,
            "output_type": e.output_type,
            "vram_gb":     e.vram_gb,
            "hf_repo":     e.hf_repo,
            "params":      e.params,
        }
        for key, e in ENGINES.items()
    }
