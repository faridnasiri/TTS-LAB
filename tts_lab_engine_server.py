#!/usr/bin/env python3
"""
tts_lab_engine_server.py — FastAPI server for engine containers.

LAZY-LOADING: Engines are only loaded on first synthesis request.
Only ONE engine is kept in VRAM at a time. Before loading a new
engine, the previous one is evicted and GPU memory is cleared.

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
import threading

# ── Shims MUST be imported before any ML library ────────────────
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
from tts_lab_dispatch import _available
from tts_lab_engines import LOADERS, SYNTHERS
from tts_lab_utils import _wav_dur, _safe_del

app = FastAPI(title=f"TTS Lab — Engine Server ({_STACK} stack)")

# ── Engine state (LAZY — nothing loaded at startup) ─────────────
_available_engines: list[str] = []   # passed availability probe
_unavailable: dict[str, str] = {}    # engine_name → reason
_load_times: dict[str, float] = {}   # engine_name → last load time (cached)

# Current loaded engine — AT MOST ONE
_current_engine: str | None = None
_current_instance: object | None = None
_lock = threading.Lock()


class SynthRequest(BaseModel):
    engine: str
    text: str
    params: dict = {}


class HealthResponse(BaseModel):
    status: str
    stack: str
    engines_available: int
    engines_loaded: int
    current_engine: str | None
    engines: dict[str, dict]


# ── VRAM management ──────────────────────────────────────────────

def _evict_current() -> None:
    """Unload the currently loaded engine and free GPU memory."""
    global _current_engine, _current_instance
    if _current_instance is not None:
        name = _current_engine
        print(f"[engine-server:{_STACK}] Evicting {name} ...")
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        _safe_del(_current_instance)
        _current_instance = None
        _current_engine = None
        try:
            import torch
            torch.cuda.empty_cache()
            _free, _total = torch.cuda.mem_get_info()
            print(f"[engine-server:{_STACK}] VRAM after evict: {_free//1048576} / {_total//1048576} MB free")
        except Exception:
            pass


def _load_engine(name: str) -> object:
    """Evict current engine (if any), load `name`, return instance.
    Thread-safe: only one load at a time."""
    global _current_engine, _current_instance

    with _lock:
        # Already loaded? Return it.
        if _current_engine == name and _current_instance is not None:
            print(f"[engine-server:{_STACK}] {name} already loaded — reusing")
            return _current_instance

        # Evict whatever is currently loaded
        _evict_current()

        # Load the requested engine
        print(f"[engine-server:{_STACK}] Loading {name} ...")
        t0 = time.perf_counter()
        try:
            instance = LOADERS[name]()
        except Exception as e:
            print(f"[engine-server:{_STACK}] {name} LOAD FAILED: {e}")
            traceback.print_exc()
            # Try to clear whatever partial state may exist
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load engine '{name}': {e}"
            )

        elapsed = round(time.perf_counter() - t0, 2)
        _current_instance = instance
        _current_engine = name
        _load_times[name] = elapsed
        _state[name]["instance"] = instance
        _state[name]["status"] = "loaded"
        _state[name]["load_time_s"] = elapsed

        print(f"[engine-server:{_STACK}] {name} loaded in {elapsed}s")
        return instance


# ── Startup: probe availability only, do NOT load engines ────────

@app.on_event("startup")
async def probe_availability():
    print(f"[engine-server:{_STACK}] Starting on port {_PORT} (lazy-load mode)")
    print(f"[engine-server:{_STACK}] Probing engine availability ...")

    for name in MODEL_ORDER:
        ok, reason = _available(name)
        if ok:
            _available_engines.append(name)
        else:
            _unavailable[name] = reason

    print(f"[engine-server:{_STACK}] Available: {len(_available_engines)} engines "
          f"(none loaded — lazy mode)")
    print(f"[engine-server:{_STACK}] Unavailable: {len(_unavailable)} engines")
    print(f"[engine-server:{_STACK}] Listening on port {_PORT}")


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    engines_status = {}
    for name in _available_engines:
        engines_status[name] = {
            "loaded": name == _current_engine,
            "load_time_s": _load_times.get(name, 0),
        }
    for name, reason in _unavailable.items():
        engines_status[name] = {
            "loaded": False,
            "reason": reason,
        }

    return HealthResponse(
        status="ok",
        stack=_STACK,
        engines_available=len(_available_engines),
        engines_loaded=1 if _current_instance is not None else 0,
        current_engine=_current_engine,
        engines=engines_status,
    )


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    # Validate engine is available
    if req.engine not in _available_engines:
        if req.engine in _unavailable:
            raise HTTPException(
                status_code=503,
                detail=f"Engine '{req.engine}' not available: {_unavailable[req.engine]}"
            )
        raise HTTPException(
            status_code=503,
            detail=f"Engine '{req.engine}' not available in this container"
        )

    # Lazy-load (evicts previous engine, loads this one)
    instance = _load_engine(req.engine)

    try:
        t0 = time.perf_counter()
        wav, sr = SYNTHERS[req.engine](instance, req.text, req.params)
        synth_ms = int((time.perf_counter() - t0) * 1000)
        dur_ms = int(_wav_dur(wav) * 1000)

        slog("SYNTH", req.engine,
             f"synth {synth_ms}ms  dur {dur_ms}ms  "
             f"RTF {round(synth_ms/dur_ms,4) if dur_ms>0 else 0}×  {sr} Hz")

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


@app.post("/unload")
async def unload():
    """Manually evict the current engine. For testing/debugging."""
    _evict_current()
    return {"unloaded": True, "current_engine": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT, log_level="info")
