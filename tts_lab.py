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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tts_lab_config import (
    MODEL_ORDER, MODEL_INFO, _state, UPLOAD_DIR,
    SYNTH_TIMEOUT, DEFAULT_SYNTH_TIMEOUT,
    ALL_KOKORO_VOICES, ALL_XTTS_SPEAKERS, BARK_PRESETS, OUTETTS_SPEAKERS,
    _server_log, _server_log_seq, slog,
)
from tts_lab_shims import DEVICE, DEVICE_NAME, VRAM_TOTAL_MB
from tts_lab_utils import _ram_mb, _piper_voices, _safe_del
from tts_lab_dispatch import (
    _available, _do_synth, _ensure_loaded, _sweep_availability,
    _import_cache, _import_cache_lock, _sweep_done,
)
from tts_lab_ui import build_page

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_lab:app", host="0.0.0.0", port=8001, reload=False, workers=1)
