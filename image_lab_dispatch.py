"""
image_lab_dispatch.py — FastAPI route handlers for the Image & Video Lab.
"""

from __future__ import annotations
import asyncio
import importlib
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
        "magic_prompt_aspect_ratio": magic_prompt_aspect_ratio,
    }

    try:
        if engine_key == "ideogram4":
            log.info(
                "IDEogram4 REQUEST | magic=%s | ratio=%s | %dx%d | preset=%s | steps=%s | guidance=%.1f | seed=%s | quant=%s | prompt_head=%s...",
                use_magic_prompt, magic_prompt_aspect_ratio,
                width, height, preset, num_inference_steps,
                guidance_scale, seed, quant,
                prompt[:120].replace("\n", " "),
            )
        # Run generation in a thread so the event loop stays free for /status and /logs
        results = await asyncio.to_thread(engines.generate, engine_key, params)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Generation error for engine %s", engine_key)
        raise HTTPException(500, f"Generation failed: {exc}")

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# /generate/ideogram4/caption — expand plain text → JSON caption (fast, no generation)
# ---------------------------------------------------------------------------

@router.post("/generate/ideogram4/caption")
async def ideogram4_caption(
    prompt:               str   = Form(...),
    use_magic_prompt:     bool  = Form(True),
    magic_prompt_aspect_ratio: str = Form("16:9"),
):
    """
    Expand a plain-text prompt into a structured Ideogram 4 JSON caption.
    No image generation — returns in ~2-5s. Never times out (>120s).

    Priority chain: Ideogram hosted (free) → DeepSeek → OpenRouter.

    Response includes per-provider error details so callers can debug
    missing API keys or network issues.
    """
    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")

    try:
        mod = importlib.import_module("ideogram4_lab_engine")
    except ImportError:
        raise HTTPException(503, "Ideogram 4 engine not available")

    caption = prompt
    provider = "none"
    errors: list[dict] = []

    if use_magic_prompt and prompt.strip():
        providers = [
            ("ideogram_hosted", mod._expand_via_ideogram),
            ("deepseek",        mod._expand_via_deepseek),
            ("openrouter",      mod._expand_via_openrouter),
        ]
        for prov_name, expand_fn in providers:
            try:
                expanded = expand_fn(prompt, magic_prompt_aspect_ratio)
                if expanded:
                    caption = expanded
                    provider = prov_name
                    break
                else:
                    errors.append({
                        "provider": prov_name,
                        "error": "expansion returned empty — API key missing or request failed",
                    })
            except Exception as exc:
                errors.append({
                    "provider": prov_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    if provider != "none":
        log.info("Caption expanded via %s (%d chars)", provider, len(caption))
    else:
        log.warning("Caption expansion failed for all providers: %s",
                    "; ".join(e["provider"] + ": " + e["error"] for e in errors))

    return JSONResponse({
        "success":      provider != "none",
        "caption":      caption,
        "provider":     provider,
        "aspect_ratio": magic_prompt_aspect_ratio,
        "errors":       errors,
        "user_message": (
            f"Caption expanded via {provider}" if provider != "none"
            else "All caption providers failed — see 'errors' for details. "
                 "Ensure IDEOGRAM_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY is set. "
                 "Returning plain-text prompt as-is (will produce poor results)."
        ),
    })


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
