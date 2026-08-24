"""
tts_lab_history.py — Generation history store. Stdlib-only (orchestrator-safe).

Every saved synthesis is stored as a WAV file + one metadata entry in a flat
JSON index, mirroring the image-lab gallery and /upload sidecar patterns:

  GENERATION_DIR/index.json   -- flat list of entries, oldest -> newest
  GENERATION_DIR/{id}.wav     -- audio for each entry

The orchestrator container has no ML libraries and its writable layer is
ephemeral, so this module imports ONLY stdlib + tts_lab_config (pure Python)
and everything lives under GENERATION_DIR (a bind mount in Docker mode).
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import uuid
import wave
from pathlib import Path

from tts_lab_config import MODEL_INFO, slog

GENERATION_DIR = Path(os.environ.get("GENERATION_DIR", "/opt/arthur/generation_history"))
HISTORY_CAP = int(os.environ.get("TTS_HISTORY_CAP", "200"))
_INDEX_PATH = GENERATION_DIR / "index.json"

# One lock guards every index read-modify-write; synth runs in a thread
# executor, so concurrent saves are the norm.
_lock = threading.Lock()

# Params whose values the dedicated "voice" filter matches against.
_VOICE_PARAM_KEYS = frozenset((
    "voice", "speaker", "speaker_name", "prompt", "audio_prompt_id", "ref_id",
    "ref_text", "voice_characteristics", "preset", "lang", "language",
))

# Sort orderings: key -> (field, reverse)
_SORTS = {
    "newest":   ("created_at", True),
    "oldest":   ("created_at", False),
    "duration": ("audio_dur_ms", True),
    "rtf":      ("rtf", True),
}


def _init_dir() -> None:
    """Create GENERATION_DIR once at import so the first save never races a mkdir."""
    try:
        GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # read-only env: first save will surface the real error via slog


_init_dir()


def _read_index() -> list:
    """Read + parse index.json. On any failure rename it aside and return [] —
    a corrupt index must never 500 the API."""
    try:
        if _INDEX_PATH.exists():
            data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        return []
    except Exception:
        try:
            _INDEX_PATH.replace(_INDEX_PATH.with_suffix(".json.corrupt"))
        except Exception:
            pass
        return []


def _write_index(data: list) -> None:
    """Atomic write: tmp file + os.replace so a crash never leaves torn JSON."""
    tmp = _INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_INDEX_PATH)


def save_generation(*, engine: str, engine_label: str, text: str, params: dict,
                    audio_b64: str, sample_rate: int, synth_time_ms: int,
                    audio_dur_ms: int, rtf: float, load_time_s: float) -> str | None:
    """Persist one generation. Returns the entry id, or None on any failure —
    callers must treat a failed save as non-fatal (synth already succeeded)."""
    try:
        wav = base64.b64decode(audio_b64)
        if not wav:
            return None
        with _lock:
            data = _read_index()
            existing = {e.get("id") for e in data}
            entry_id = None
            for _ in range(5):
                cand = uuid.uuid4().hex[:8]
                if cand not in existing:
                    entry_id = cand
                    break
            if entry_id is None:
                return None

            # Sniff: engines return full WAV (RIFF); SGLang returns raw PCM.
            if wav.startswith(b"RIFF"):
                raw = wav
            else:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(sample_rate) if sample_rate else 22050)
                    wf.writeframes(wav)
                raw = buf.getvalue()
            (GENERATION_DIR / f"{entry_id}.wav").write_bytes(raw)

            entry = {
                "id":            entry_id,
                "engine":        engine,
                "engine_label":  engine_label or MODEL_INFO.get(engine, {}).get("label", engine),
                "text":          text,
                "params":        params or {},
                "sample_rate":   sample_rate,
                "synth_time_ms": synth_time_ms,
                "audio_dur_ms":  audio_dur_ms,
                "rtf":           rtf,
                "load_time_s":   load_time_s,
                "created_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            data.append(entry)

            # Cap eviction — oldest entries sit at the front.
            while len(data) > HISTORY_CAP:
                dropped = data.pop(0)
                try:
                    (GENERATION_DIR / f"{dropped['id']}.wav").unlink(missing_ok=True)
                except Exception:
                    pass
                slog("HISTORY", engine,
                     f"cap {HISTORY_CAP} reached — evicted {dropped['id']}")

            _write_index(data)
        return entry_id
    except Exception as e:
        slog("HISTORY", engine, f"save_generation failed: {e}")
        return None


def list_history(*, engine: str = "", q: str = "", voice: str = "",
                 min_dur_s: float = 0, max_dur_s: float = 0,
                 date_from: str = "", date_to: str = "",
                 sort: str = "newest", limit: int = 50, offset: int = 0) -> dict:
    """List entries with filters. Entries never carry audio bytes. All text
    matching is case-insensitive substring — no regex, nothing to escape."""
    with _lock:
        data = _read_index()

    q = q.strip().lower()
    voice = voice.strip().lower()
    engine = engine.strip()

    out = []
    for e in data:
        if engine and e.get("engine") != engine:
            continue
        if q:
            hay = (e.get("text") or "").lower()
            for v in (e.get("params") or {}).values():
                hay += "\n" + str(v).lower()
            if q not in hay:
                continue
        if voice:
            vals = [str(v).lower() for k, v in (e.get("params") or {}).items()
                    if k in _VOICE_PARAM_KEYS]
            if not any(voice in v for v in vals):
                continue
        dur_s = (e.get("audio_dur_ms") or 0) / 1000.0
        if min_dur_s and dur_s < min_dur_s:
            continue
        if max_dur_s and dur_s > max_dur_s:
            continue
        day = (e.get("created_at") or "")[:10]
        if date_from and day < date_from:
            continue
        if date_to and day > date_to:
            continue
        out.append(e)

    total = len(out)
    field, reverse = _SORTS.get(sort, _SORTS["newest"])
    out.sort(key=lambda e: e.get(field, 0 if field != "created_at" else ""),
             reverse=reverse)
    return {
        "entries": out[offset:offset + limit],
        "total":   total,
        "limit":   limit,
        "offset":  offset,
    }


def get_history_path(entry_id: str) -> Path | None:
    if not _entry_exists(entry_id):
        return None
    path = GENERATION_DIR / f"{entry_id}.wav"
    return path if path.exists() else None


def get_history_entry(entry_id: str) -> dict | None:
    with _lock:
        for e in _read_index():
            if e.get("id") == entry_id:
                return e
    return None


def delete_history_entry(entry_id: str) -> bool:
    with _lock:
        data = _read_index()
        for i, e in enumerate(data):
            if e.get("id") == entry_id:
                del data[i]
                _write_index(data)
                try:
                    (GENERATION_DIR / f"{entry_id}.wav").unlink(missing_ok=True)
                except Exception:
                    pass
                return True
    return False


def history_stats() -> dict:
    with _lock:
        data = _read_index()
    by_engine: dict = {}
    for e in data:
        by_engine[e.get("engine", "?")] = by_engine.get(e.get("engine", "?"), 0) + 1
    return {"total": len(data), "by_engine": by_engine}


def _entry_exists(entry_id: str) -> bool:
    with _lock:
        return any(e.get("id") == entry_id for e in _read_index())
