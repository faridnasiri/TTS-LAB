#!/usr/bin/env python3
"""
Arthur TTS Lab -- 21-Engine Edition
Port: 8001  |  Open: http://192.168.0.87:8001

Entry point. All logic lives in:
  tts_lab_shims.py    -- startup-time env vars + compatibility patches
  tts_lab_config.py   -- catalogues, MODEL_INFO, shared _state
  tts_lab_utils.py    -- small utility functions
  tts_lab_engines.py  -- 21 _load_* / _synth_* pairs + LOADERS/SYNTHERS
  tts_lab_dispatch.py -- availability probing, _ensure_loaded, _do_synth
  tts_lab_ui.py       -- CSS, JS, param widgets, build_page()
"""
from __future__ import annotations

# shims MUST be first -- patches transformers/torchaudio before any ML import
import tts_lab_shims  # noqa: F401

import asyncio, shutil, threading, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tts_lab_config import (
    MODEL_ORDER, MODEL_INFO, _state, UPLOAD_DIR,
    SYNTH_TIMEOUT, DEFAULT_SYNTH_TIMEOUT,
    ALL_KOKORO_VOICES, ALL_XTTS_SPEAKERS, BARK_PRESETS, OUTETTS_SPEAKERS,
    MATCHA_VOICES,
    _server_log, _server_log_seq, slog,
)
from tts_lab_shims import DEVICE, DEVICE_NAME, VRAM_TOTAL_MB
from tts_lab_utils import _ram_mb, _piper_voices, _safe_del
from tts_lab_dispatch import (
    _available, _do_synth, _ensure_loaded, _sweep_availability,
    _import_cache, _import_cache_lock, _sweep_done,
)
from tts_lab_ui import build_page
from tts_lab_engines import _process_persian_text
from voice_library import (
    list_voices, get_voice, get_voice_path, get_stats,
    add_voice, remove_voice, get_embedding,
    download_common_voice_persian, import_from_uploads,
    VOICE_LIBRARY_DIR, VOICES_DIR,
)

app = FastAPI(title="Arthur TTS Lab")


class SynthReq(BaseModel):
    text:   str
    params: dict = {}


@app.on_event("startup")
async def _startup():
    t = threading.Thread(target=_sweep_availability, name="avail-sweep", daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(build_page())


@app.get("/status")
async def status():
    models = {}
    sweep_running = not _sweep_done.is_set()
    for n in MODEL_ORDER:
        ok, reason = _available(n)
        st = _state[n]
        models[n] = {
            **MODEL_INFO[n],
            "available":    ok,
            "reason":       reason,
            "status":       st["status"],
            "load_time_s":  st["load_time_s"],
            "error":        st["error"],
            "loaded_model": st.get("loaded_model") or st.get("loaded_voice"),
        }
        if sweep_running and n not in _import_cache:
            models[n]["available"] = False
            models[n]["reason"]    = "checking..."
    tot, used, free = _ram_mb()
    gpu_info = {}
    if DEVICE == "cuda":
        try:
            import torch
            gpu_info = {
                "name":       DEVICE_NAME,
                "vram_total": VRAM_TOTAL_MB,
                "vram_used":  int(torch.cuda.memory_allocated(0) / 1048576),
                "vram_free":  int((torch.cuda.get_device_properties(0).total_memory
                                   - torch.cuda.memory_allocated(0)) / 1048576),
            }
        except Exception:
            gpu_info = {"name": DEVICE_NAME, "vram_total": VRAM_TOTAL_MB}
    return JSONResponse({
        "models": models,
        "system": {"total": tot, "used": used, "free": free},
        "gpu":    gpu_info,
        "device": DEVICE,
    })


@app.get("/logs")
async def get_logs(since: int = 0):
    """Return server-side log entries with seq > since."""
    entries = [e for e in _server_log if e["seq"] > since]
    return JSONResponse({"entries": entries, "seq": _server_log_seq})


@app.get("/voices/{model}")
async def voices(model: str):
    vmap = {
        "piper":     _piper_voices() or ["en_US-ryan-high"],
        "kokoro":    ALL_KOKORO_VOICES,
        "melo":      ["EN-Default", "EN-US", "EN-BR", "EN-AU", "EN_INDIA"],
        "outetts":   [v for v, _ in OUTETTS_SPEAKERS],
        "bark":      [v for v, _ in BARK_PRESETS],
        "xtts":      ALL_XTTS_SPEAKERS,
        "cosyvoice": ["English Female", "English Male"],
        "matcha":    [v for v, _ in MATCHA_VOICES],
        "manatts":   ["Persian Female (built-in)"],
    }
    return JSONResponse({"voices": vmap.get(model, [])})


@app.post("/synthesize/{model}")
async def synthesize(model: str, req: SynthReq):
    if model not in MODEL_ORDER:
        return JSONResponse({"error": f"Unknown engine: {model}"}, status_code=400)
    if not req.text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)
    timeout = SYNTH_TIMEOUT.get(model, DEFAULT_SYNTH_TIMEOUT)
    try:
        loop   = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _do_synth, model, req.text, req.params),
            timeout=float(timeout),
        )
        return JSONResponse(result)
    except asyncio.TimeoutError:
        return JSONResponse({
            "error": f"Synthesis timeout after {timeout}s -- {model!r} requires a GPU."
        }, status_code=408)
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc(limit=4)},
            status_code=500,
        )


@app.delete("/models/{model}")
async def unload_model(model: str):
    st = _state.get(model)
    if st and st["instance"] is not None:
        _safe_del(st["instance"])
        st["instance"] = None
        st["status"]   = "unloaded"
    return {"unloaded": model}


@app.post("/models/{model}/load")
async def preload_model(model: str, request: Request):
    st = _state.get(model)
    if st is None:
        raise HTTPException(404, f"Unknown engine: {model}")
    try:
        body = await request.json()
    except Exception:
        body = {}
    params = body.get("params", {})
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: _ensure_loaded(model, params))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"status": "loaded", "model": model, "load_time_s": st["load_time_s"]}


@app.post("/refresh")
async def refresh_availability():
    with _import_cache_lock:
        _import_cache.clear()
    _sweep_done.clear()
    t = threading.Thread(target=_sweep_availability, name="avail-resweep", daemon=True)
    t.start()
    return JSONResponse({
        "refreshed": True,
        "models":    list(MODEL_ORDER),
        "note":      "sweep running in background -- poll /status in ~60 s",
    })


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    uid  = str(uuid.uuid4())[:8]
    dest = UPLOAD_DIR / f"{uid}.wav"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"id": uid, "filename": file.filename, "size": dest.stat().st_size})


# Permanent reference voices directory (survives reboots, shipped with deploy)
REFERENCE_VOICES_DIR = Path("/opt/arthur/reference_voices")


@app.get("/refs")
async def list_refs():
    """List available reference WAV files for dropdown selection.

    Scans two locations:
      1. Permanent reference voices (shipped, survive reboots) — listed first
      2. User-uploaded voices in UPLOAD_DIR — listed after
    """
    refs = []
    seen = set()

    # 1. Permanent reference voices (shipped with lab)
    if REFERENCE_VOICES_DIR.exists():
        for p in sorted(REFERENCE_VOICES_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
            seen.add(p.stem)
            refs.append({
                "id": p.stem,
                "name": p.name,
                "size": p.stat().st_size,
            })

    # 2. User-uploaded voices (may not survive reboot)
    for p in sorted(UPLOAD_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.stem not in seen:
            refs.append({
                "id": p.stem,
                "name": f"{p.name}  (uploaded)",
                "size": p.stat().st_size,
            })

    return JSONResponse({"refs": refs})


@app.get("/preview-text")
async def preview_text(text: str = "", provider: str = "none"):
    """Preview what the Persian text processor will produce for a given provider.

    Providers: persian_phonemizer, hazm, parsivar, none
    Returns the processed text for display in the UI.
    """
    if not text:
        return JSONResponse({"processed_text": "", "provider": provider})
    result = _process_persian_text(text, provider)
    return JSONResponse({"processed_text": result, "provider": provider})


# ── Voice Library endpoints ──────────────────────────────────────────────────

@app.get("/voice-library")
async def voice_library_list(
    gender: str = "",
    min_duration: float = 0,
    max_duration: float = 999,
    min_quality: float = 0,
    limit: int = 200,
):
    """List voices in the library with optional filters."""
    voices = list_voices(
        gender=gender,
        min_duration=min_duration,
        max_duration=max_duration,
        min_quality=min_quality,
        limit=limit,
    )
    return JSONResponse({"voices": voices, "count": len(voices)})


@app.get("/voice-library/stats")
async def voice_library_stats():
    """Get voice library statistics."""
    return JSONResponse(get_stats())


@app.get("/voice-library/{voice_id}")
async def voice_library_get(voice_id: str):
    """Get metadata for a specific voice."""
    v = get_voice(voice_id)
    if not v:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    return JSONResponse(v)


@app.get("/voice-library/{voice_id}/audio")
async def voice_library_audio(voice_id: str):
    """Stream the WAV audio for a voice (for <audio> preview)."""
    from fastapi.responses import Response
    path = get_voice_path(voice_id)
    if not path:
        raise HTTPException(404, f"Voice audio not found: {voice_id}")
    return Response(content=path.read_bytes(), media_type="audio/wav")


@app.post("/voice-library/{voice_id}/use-ref")
async def voice_library_use_ref(voice_id: str, engine: str = ""):
    """Copy a voice library clip to the uploads directory so it appears in
    the reference WAV dropdown for the specified engine."""
    path = get_voice_path(voice_id)
    if not path:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    import shutil
    dest = UPLOAD_DIR / f"{voice_id}.wav"
    shutil.copy2(path, dest)
    v = get_voice(voice_id)
    return JSONResponse({
        "ok": True,
        "audio_prompt_id": voice_id,
        "voice": v,
        "url": f"/voice-library/{voice_id}/audio",
    })


@app.post("/voice-library/import-uploads")
async def voice_library_import():
    """Import existing WAVs from the TTS uploads directory into the library."""
    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(None, import_from_uploads, UPLOAD_DIR)
    return JSONResponse({"ok": True, "imported": count})


@app.post("/voice-library/download")
async def voice_library_download(
    count: int = 40,
    min_duration: float = 3.0,
    max_duration: float = 12.0,
    female_ratio: float = 0.5,
):
    """Download Persian voices from Common Voice."""
    loop = asyncio.get_running_loop()
    n = await loop.run_in_executor(
        None,
        download_common_voice_persian,
        count, min_duration, max_duration, 1, female_ratio,
    )
    return JSONResponse({"ok": True, "downloaded": n})


@app.delete("/voice-library/{voice_id}")
async def voice_library_delete(voice_id: str):
    """Remove a voice from the library."""
    remove_voice(voice_id)
    return JSONResponse({"ok": True, "deleted": voice_id})


@app.get("/voice-library/{voice_id}/embedding/{emb_type}")
async def voice_library_embedding(voice_id: str, emb_type: str = "ge2e"):
    """Get pre-computed speaker embedding info."""
    emb = get_embedding(voice_id, emb_type)
    if emb is None:
        raise HTTPException(404, f"Embedding not available for {voice_id}/{emb_type}")
    return JSONResponse({
        "voice_id": voice_id,
        "emb_type": emb_type,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_lab:app", host="0.0.0.0", port=8001, reload=False, workers=1)
