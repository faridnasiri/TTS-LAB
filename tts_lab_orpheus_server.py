#!/usr/bin/env python3
"""
tts_lab_orpheus_server.py — FastAPI server for the Orpheus 3B container.

Runs in the CUDA container (Dockerfile.orpheus).
Loads Orpheus at startup. Exposes /health and /synthesize.

Usage:
  python tts_lab_orpheus_server.py --port 8002
"""
from __future__ import annotations

import argparse
import base64
import os
import time
import traceback
import sys as _sys

_parser = argparse.ArgumentParser()
_parser.add_argument("--port", type=int, default=8002)
_args, _unknown = _parser.parse_known_args()
_PORT = _args.port

# ── Import shims first ──────────────────────────────────────────
# Orpheus runs on its own CUDA 12.1 + vllm stack.
# Most of the shims aren't needed here (no transformers, no tf 5.x).
# But we import them for the thread-pool env vars and device detection.
import tts_lab_shims  # noqa: F401 — side effects

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tts_lab_engines import _load_orpheus, _synth_orpheus
from tts_lab_utils import _wav_dur

app = FastAPI(title="TTS Lab — Orpheus 3B")

_model = None
_load_time_s: float = 0.0


class SynthRequest(BaseModel):
    engine: str = "orpheus"       # for API compatibility with engine server
    text: str
    params: dict = {}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    load_time_s: float
    model_name: str


@app.on_event("startup")
async def load_model():
    global _model, _load_time_s
    print("[orpheus-server] Loading Orpheus 3B model ...")
    try:
        t0 = time.perf_counter()
        _model = _load_orpheus()
        _load_time_s = round(time.perf_counter() - t0, 2)
        print(f"[orpheus-server] Loaded in {_load_time_s}s")
    except Exception as e:
        print(f"[orpheus-server] Load FAILED: {e}")
        traceback.print_exc()
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _model is not None else "error",
        model_loaded=_model is not None,
        load_time_s=_load_time_s,
        model_name="canopylabs/orpheus-3b-0.1-ft",
    )


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.perf_counter()
        wav, sr = _synth_orpheus(_model, req.text, req.params)
        synth_ms = int((time.perf_counter() - t0) * 1000)
        dur_ms = int(_wav_dur(wav) * 1000)

        return {
            "audio_b64": base64.b64encode(wav).decode(),
            "sample_rate": sr,
            "synth_time_ms": synth_ms,
            "audio_dur_ms": dur_ms,
            "rtf": round(synth_ms / dur_ms, 4) if dur_ms > 0 else 0,
            "load_time_s": _load_time_s,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT, log_level="info")
