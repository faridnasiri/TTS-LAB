"""
tts_lab_utils.py — shared utility functions.
"""
from __future__ import annotations
import gc, io, wave
import numpy as np
from pathlib import Path
from tts_lab_config import MODELS_DIR, HEAVY, _state, DEVICE


def _ram_mb() -> tuple[int, int, int]:
    try:
        import psutil
        v = psutil.virtual_memory()
        return v.total // 1048576, v.used // 1048576, v.available // 1048576
    except Exception:
        return 16384, 0, 16384


def _to_wav(audio, sr: int) -> bytes:
    arr = np.array(audio, dtype=np.float64).flatten()
    if arr.dtype != np.int16:
        arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def _wav_dur(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _safe_del(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()


def _evict_heavy(keep: str) -> None:
    for n in HEAVY:
        if n != keep and _state[n]["instance"] is not None:
            _safe_del(_state[n]["instance"])
            _state[n]["instance"] = None
            _state[n]["status"]   = "unloaded"


def _piper_voices() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.onnx") if "kokoro" not in p.name)


def _read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        fr         = wf.getframerate()
        frames     = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise RuntimeError("Reference WAV must be 16-bit PCM.")
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, fr


def _require_gpu(engine: str) -> None:
    """Raise immediately if no CUDA GPU — fast fail instead of hanging for minutes."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{engine} requires a CUDA GPU and will not run on CPU.\n"
                "Add a GPU and restart the server, then try again."
            )
    except ImportError:
        pass
