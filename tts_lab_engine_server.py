#!/usr/bin/env python3
"""
tts_lab_engine_server.py — FastAPI server for engine containers.

Each engine container (engine-current, engine-legacy) runs this server.
It loads all available engines at startup and exposes:
  GET  /health      → engine status + loaded models
  POST /synthesize  → {engine, text, params} → audio_b64 + metadata

Usage:
  python tts_lab_engine_server.py --port 8101 --stack current
  python tts_lab_engine_server.py --port 8102 --stack legacy
"""
from __future__ import annotations

import argparse
import base64
import os
import time
import traceback

# ── Shims MUST be imported before any ML library ────────────────
# The --stack arg controls which shims module to use.
# We parse it from sys.argv before imports.
import sys as _sys

_parser = argparse.ArgumentParser()
_parser.add_argument("--port", type=int, default=8101)
_parser.add_argument("--stack", type=str, default="current")
_args, _unknown = _parser.parse_known_args()

_STACK = _args.stack
_PORT = _args.port

if _STACK == "legacy":
    import tts_lab_shims_legacy as _shims  # noqa: F401 — side effects
else:
    import tts_lab_shims as _shims  # noqa: F401 — side effects

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tts_lab_config import MODEL_ORDER, MODEL_INFO, _state, slog
from tts_lab_dispatch import _available, LOADERS, SYNTHERS
from tts_lab_utils import _wav_dur, _safe_del

app = FastAPI(title=f"TTS Lab — Engine Server ({_STACK} stack)")

# ── Engine loading state ────────────────────────────────────────
_loaded: dict[str, object] = {}      # engine_name → model instance
_load_times: dict[str, float] = {}   # engine_name → load time in seconds
_available_engines: list[str] = []   # engines that passed availability check
_unavailable: dict[str, str] = {}    # engine_name → reason not available
_startup_done = False


class SynthRequest(BaseModel):
    engine: str
    text: str
    params: dict = {}


class HealthResponse(BaseModel):
    status: str
    stack: str
    engines_available: int
    engines_loaded: int
    engines: dict[str, dict]


@app.on_event("startup")
async def load_all_engines():
    global _available_engines, _unavailable, _startup_done

    print(f"[engine-server:{_STACK}] Starting up on port {_PORT} ...")
    print(f"[engine-server:{_STACK}] Probing engine availability ...")

    # Discover which engines in MODEL_ORDER are available in this container
    for name in MODEL_ORDER:
        # Skip engines that belong to other stacks
        # (The SGLang engines and orpheus won't have their packages here,
        # so _available() will naturally return False for them.)
        ok, reason = _available(name)
        if ok:
            _available_engines.append(name)
        else:
            _unavailable[name] = reason

    print(f"[engine-server:{_STACK}] Available: {len(_available_engines)} engines")
    print(f"[engine-server:{_STACK}] Unavailable: {len(_unavailable)} engines")

    # Load available engines
    for name in _available_engines:
        try:
            print(f"[engine-server:{_STACK}] Loading {name} ...")
            t0 = time.perf_counter()
            instance = LOADERS[name]()
            elapsed = round(time.perf_counter() - t0, 2)
            _loaded[name] = instance
            _load_times[name] = elapsed
            _state[name]["instance"] = instance
            _state[name]["status"] = "loaded"
            _state[name]["load_time_s"] = elapsed
            print(f"[engine-server:{_STACK}]   {name} loaded in {elapsed}s")
        except Exception as e:
            print(f"[engine-server:{_STACK}]   {name} FAILED: {e}")
            traceback.print_exc()
            _unavailable[name] = str(e)

    _startup_done = True
    print(f"[engine-server:{_STACK}] Startup complete. "
          f"Loaded: {len(_loaded)}/{len(_available_engines)}. "
          f"Listening on port {_PORT}")


@app.get("/health", response_model=HealthResponse)
async def health():
    engines_status = {}
    for name in _available_engines:
        engines_status[name] = {
            "loaded": name in _loaded,
            "load_time_s": _load_times.get(name, 0),
        }
    for name, reason in _unavailable.items():
        engines_status[name] = {
            "loaded": False,
            "reason": reason,
        }

    return HealthResponse(
        status="ok" if _startup_done else "starting",
        stack=_STACK,
        engines_available=len(_available_engines),
        engines_loaded=len(_loaded),
        engines=engines_status,
    )


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    if req.engine not in _loaded:
        if req.engine in _unavailable:
            raise HTTPException(
                status_code=503,
                detail=f"Engine '{req.engine}' not available: {_unavailable[req.engine]}"
            )
        raise HTTPException(
            status_code=503,
            detail=f"Engine '{req.engine}' not loaded"
        )

    instance = _loaded[req.engine]
    try:
        t0 = time.perf_counter()
        wav, sr = SYNTHERS[req.engine](instance, req.text, req.params)
        synth_ms = int((time.perf_counter() - t0) * 1000)
        dur_ms = int(_wav_dur(wav) * 1000)

        return {
            "audio_b64": base64.b64encode(wav).decode(),
            "sample_rate": sr,
            "synth_time_ms": synth_ms,
            "audio_dur_ms": dur_ms,
            "rtf": round(synth_ms / dur_ms, 4) if dur_ms > 0 else 0,
            "load_time_s": _load_times.get(req.engine, 0),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT, log_level="info")
