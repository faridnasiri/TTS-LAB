#!/usr/bin/env python3
"""
Arthur TTS Lab -- 28-Engine Edition
Port: 8001  |  Open: http://192.168.0.87:8001

Entry point. All logic lives in:
  tts_lab_shims.py    -- startup-time env vars + compatibility patches
  tts_lab_config.py   -- catalogues, MODEL_INFO, shared _state
  tts_lab_utils.py    -- small utility functions
  tts_lab_engines.py  -- 21 _load_* / _synth_* pairs + LOADERS/SYNTHERS
  tts_lab_dispatch.py -- availability probing, _ensure_loaded, _do_synth
  tts_lab_ui.py       -- CSS, JS, param widgets, build_page()

Two modes:
  ORCHESTRATOR (remote): set ORCHESTRATOR_MODE=1 env var.
    No ML libraries needed. All engines via HTTP.

  LOCAL (bare metal): default when ORCHESTRATOR_MODE is not set.
    Full in-process engine loading.
"""
from __future__ import annotations

import os as _os

_ORCHESTRATOR_MODE = _os.environ.get("ORCHESTRATOR_MODE", "") == "1"

# shims MUST be first -- patches transformers/torchaudio before any ML import
if _ORCHESTRATOR_MODE:
    # In orchestrator mode, we don't have torch. Import a minimal shim.
    # Set defaults for what tts_lab_shims would normally export.
    import types as _types
    import os as _os2
    _N_CORES = _os2.cpu_count() or 6
    _os2.environ.setdefault("OMP_NUM_THREADS", str(_N_CORES))
    _os2.environ.setdefault("MKL_NUM_THREADS", str(_N_CORES))
    DEVICE = "remote"
    DEVICE_NAME = "orchestrator"
    VRAM_TOTAL_MB = 0
else:
    import tts_lab_shims  # noqa: F401
    from tts_lab_shims import DEVICE, DEVICE_NAME, VRAM_TOTAL_MB

import asyncio, json, shutil, threading, time, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tts_lab_config import (
    MODEL_ORDER, MODEL_INFO, _state, UPLOAD_DIR, REFERENCE_VOICES_DIR,
    SYNTH_TIMEOUT, DEFAULT_SYNTH_TIMEOUT,
    ALL_KOKORO_VOICES, ALL_XTTS_SPEAKERS, BARK_PRESETS, OUTETTS_SPEAKERS,
    MATCHA_VOICES,
    _server_log, _server_log_seq, slog,
)
from tts_lab_dispatch import (
    _available, _do_synth, _ensure_loaded, _sweep_availability,
    _import_cache, _import_cache_lock, _sweep_done,
)
from tts_lab_ui import build_page
from tts_lab_history import (
    save_generation, list_history, get_history_path,
    delete_history_entry, history_stats,
)
# Shared container/infrastructure dashboard (GET /infra) — same router is
# mounted in image_lab.py so both labs serve one identical page.
from lab_infra import router as infra_router

# ── Conditional imports (not available in orchestrator mode) ────
if _ORCHESTRATOR_MODE:
    _process_persian_text = None
    _piper_voices_fn = None

    def _ram_mb():
        """Host RAM via /proc/meminfo — no psutil/torch in the orchestrator.
        /proc/meminfo inside a container reports HOST totals, which is what
        the 'RAM' bar in the UI means here. Returns (total, used, avail) MB."""
        try:
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    mem[k] = int(v.strip().split()[0]) // 1024  # kB → MB
            total = mem.get("MemTotal", 0)
            avail = mem.get("MemAvailable", mem.get("MemFree", 0))
            return total, max(0, total - avail), avail
        except Exception:
            return 0, 0, 0

    voice_library_mod = None
else:
    from tts_lab_engines import _process_persian_text
    from tts_lab_utils import _ram_mb, _piper_voices as _piper_voices_fn, _safe_del
    try:
        from voice_library import (
            list_voices, get_voice, get_voice_path, get_stats,
            add_voice, remove_voice, get_embedding,
            download_common_voice_persian, import_from_uploads,
            VOICE_LIBRARY_DIR, VOICES_DIR,
        )
        voice_library_mod = True
    except ImportError:
        voice_library_mod = None

app = FastAPI(title="Arthur TTS Lab")
app.include_router(infra_router)


class SynthReq(BaseModel):
    text:   str
    params: dict = {}
    save:   bool = True  # per-generation opt-in to the history library; UI checkbox, default checked


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
    elif DEVICE == "remote":
        # Merge live loaded-state from every engine container so the UI can
        # show which model(s) are resident in VRAM (and evict them one at a
        # time). Containers report engines.<name>.loaded + current_engine;
        # SGLang servers keep models always-resident while the server runs.
        from tts_lab_dispatch import _REMOTE_ENGINES, _SGLANG_URLS, _probe_containers_loaded
        probes = _probe_containers_loaded()
        best_gpu = None
        for n in MODEL_ORDER:
            url = _REMOTE_ENGINES.get(n)
            if not url:
                continue
            stripped = url.rstrip("/")
            if stripped in _SGLANG_URLS:
                base = stripped.split("/v1/")[0]
                probe = probes.get(base, {})
                if probe.get("sglang_up"):
                    models[n]["status"] = "loaded"  # resident while server runs
                if probe.get("gpu"):
                    best_gpu = probe["gpu"]
                continue
            probe = probes.get(stripped, {})
            eng_info = probe.get("engines", {}).get(n, {})
            if eng_info.get("loaded"):
                models[n]["status"]       = "loaded"
                models[n]["loaded_model"] = probe.get("current_engine")
                models[n]["container"]    = stripped
            if probe.get("gpu"):
                best_gpu = probe["gpu"]
        if best_gpu:
            gpu_info = best_gpu
        else:
            gpu_info = {"mode": "orchestrator — engines served by remote containers"}
        # Per-process GPU breakdown (pid → mb → container) so the UI can
        # show WHO holds the VRAM, incl. the bare-metal Image Lab service.
        procs = probes.get("__gpu_processes__")
        if procs:
            gpu_info["processes"] = procs
    return JSONResponse({
        "models": models,
        "system": {"total": tot, "used": used, "free": free},
        "gpu":    gpu_info,
        "device": DEVICE,
    })


@app.get("/logs")
async def get_logs(since: int = 0, engine: str = ""):
    """Return server-side log entries with seq > since.

    With `engine=<name>` of a remote engine container, proxies to that
    container's own /logs — includes engine-side per-chunk CHUNK lines
    (e.g. chatterboxturbo chunking) that never reach the orchestrator.
    """
    if engine:
        import httpx
        try:
            from tts_lab_dispatch import _REMOTE_ENGINES
            url = _REMOTE_ENGINES.get(engine)
            if not url:
                return JSONResponse({"error": f"Unknown remote engine: {engine}", "entries": [], "seq": 0}, status_code=404)
            r = httpx.get(f"{url}/logs", params={"since": since}, timeout=10.0)
            return JSONResponse(r.json())
        except Exception as e:
            return JSONResponse({"error": str(e), "entries": [], "seq": 0}, status_code=502)
    entries = [e for e in _server_log if e["seq"] > since]
    # Read the seq cursor live from the module — the by-value import above
    # freezes it at import time and would otherwise always report 0.
    import tts_lab_config as _cfg
    return JSONResponse({"entries": entries, "seq": _cfg._server_log_seq})


@app.get("/voices/{model}")
async def voices(model: str):
    vmap = {
        "piper":     (_piper_voices_fn() if _piper_voices_fn else []) or ["en_US-ryan-high"],
        "kokoro":    ALL_KOKORO_VOICES,
        "melo":      ["EN-Default", "EN-US", "EN-BR", "EN-AU", "EN_INDIA"],
        "outetts":   [v for v, _ in OUTETTS_SPEAKERS],
        "bark":      [v for v, _ in BARK_PRESETS],
        "xtts":      ALL_XTTS_SPEAKERS,
        "xttsfa":    ["ParsVoice — model default (clone via ref WAV)"],
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
        # ── Generation history (opt-in; must never fail the synth) ──
        # audio_b64 present in all three dispatch modes; LLM engines return
        # text only and are skipped by the guard regardless.
        if req.save and result.get("audio_b64"):
            try:
                hid = await loop.run_in_executor(None, lambda: save_generation(
                    engine=model,
                    engine_label=MODEL_INFO[model]["label"],
                    text=req.text, params=req.params,
                    audio_b64=result["audio_b64"],
                    sample_rate=result.get("sample_rate", 0),
                    synth_time_ms=result.get("synth_time_ms", 0),
                    audio_dur_ms=result.get("audio_dur_ms", 0),
                    rtf=result.get("rtf", 0),
                    load_time_s=result.get("load_time_s", 0),
                ))
                if hid:
                    result["history_id"] = hid
            except Exception as e:
                slog("HISTORY", model, f"history save failed (synth unaffected): {e}")
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
    if _ORCHESTRATOR_MODE:
        return JSONResponse({"unloaded": model, "note": "orchestrator mode — models managed by engine containers"})
    st = _state.get(model)
    if st and st["instance"] is not None:
        _safe_del(st["instance"])
        st["instance"] = None
        st["status"]   = "unloaded"
    return {"unloaded": model}


@app.post("/models/{model}/evict")
async def evict_model(model: str):
    """Evict ONE engine from VRAM — routes to the engine's container /evict
    (standard engine servers) or stops the SGLang server container
    (s2pro/vibevoice/higgs — always-resident). Local mode unloads the
    in-process instance. Mirrors /evict-all but for a single engine.
    """
    if model not in MODEL_ORDER:
        raise HTTPException(404, f"Unknown engine: {model}")
    from tts_lab_dispatch import _evict_engine
    return JSONResponse(_evict_engine(model))


@app.post("/models/{model}/load")
async def preload_model(model: str, request: Request):
    if _ORCHESTRATOR_MODE:
        return JSONResponse({"status": "loaded", "model": model, "note": "orchestrator mode — models managed by engine containers"})
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


@app.post("/evict-all")
async def evict_all_tts():
    """Evict all TTS engines from VRAM across all engine containers.

    Called by the UI 'Evict VRAM' button. Useful before loading the LLM
    or when VRAM needs to be cleared for any reason.
    """
    from tts_lab_dispatch import _evict_all_tts_engines
    results = _evict_all_tts_engines()
    evicted = sum(1 for v in results.values() if v.get("evicted"))
    errors = {k: v for k, v in results.items() if "error" in v}
    # Aggregate how much was actually freed vs still held (vLLM/SGLang
    # containers report no freed_mb — a restart/stop frees ~everything).
    freed_mb_total = sum(v.get("freed_mb", 0) or 0 for v in results.values())
    held_mb_total  = sum(v.get("held_mb", 0) or 0 for v in results.values())
    return JSONResponse({
        "evicted_count": evicted,
        "containers_checked": len(results),
        "freed_mb_total": freed_mb_total,
        "held_mb_total": held_mb_total,
        "errors": errors,
        "details": {k: v for k, v in results.items() if "error" not in v},
    })


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), lang: str = Form("")):
    uid  = str(uuid.uuid4())[:8]
    dest = UPLOAD_DIR / f"{uid}.wav"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Sidecar metadata: preserve original filename + language so the UI dropdown
    # can show a meaningful name instead of the random uid.
    try:
        (UPLOAD_DIR / f"{uid}.json").write_text(json.dumps({
            "original_name": file.filename or "",
            "lang": lang or "",
            "uploaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False))
    except Exception:
        pass
    return JSONResponse({"id": uid, "filename": file.filename, "size": dest.stat().st_size})


# Permanent reference voices directory — defined in tts_lab_config.py (shared
# with the engine containers, host bind-mounted everywhere).

# Display labels for reference voice languages (subset of OMNIVOICE_LANGUAGES)
_LANG_LABELS = {"fa": "فارسی (FA)", "en": "English (EN)", "other": "Other"}

# Voice Library — voices/{id}/sample.wav + metadata.json (host bind-mounted,
# same path in every container). Kept in sync with tts_lab_config.VOICE_LIBRARY_DIR.
_VOICE_LIB_DIR = _os.environ.get("VOICE_LIBRARY_DIR", "/opt/arthur/voice_library")


def _scan_refs():
    """Scan reference + uploaded WAVs with sidecar metadata (original_name/lang).

    Sidecars are `{stem}.json` files written next to each WAV by /upload and
    /voice-library/*/use-ref. transcription is present for voice-library
    voices; curated en-* voices carry none (clone quality degrades without it).

    Voice Library voices (voices/{id}/sample.wav) are included directly so
    they appear in the ref dropdowns; clone engines resolve the id via
    tts_lab_config._ref_wav_path (library layout branch). metadata.json
    carries the transcription like a sidecar.
    """
    refs = []
    for d, source in ((REFERENCE_VOICES_DIR, "reference"), (UPLOAD_DIR, "uploaded")):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
            meta = {}
            sc = p.with_suffix(".json")
            if sc.exists():
                try:
                    meta = json.loads(sc.read_text())
                except Exception:
                    meta = {}
            refs.append({
                "id": p.stem,
                "path": p,
                "original_name": str(meta.get("original_name", "") or ""),
                "lang": str(meta.get("lang", "") or ""),
                "transcription": str(meta.get("transcription", "") or ""),
                "source": source,
            })
    lib_voices = Path(_VOICE_LIB_DIR) / "voices"
    if lib_voices.exists():
        for p in sorted(lib_voices.glob("*/sample.wav"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            meta = {}
            md = p.with_name("metadata.json")
            if md.exists():
                try:
                    meta = json.loads(md.read_text())
                except Exception:
                    meta = {}
            refs.append({
                "id": p.parent.name,
                "path": p,
                "original_name": str(meta.get("speaker_name", "") or p.parent.name),
                "lang": str(meta.get("language", "") or ""),
                "transcription": str(meta.get("transcription", "") or ""),
                "source": "library",
            })
    return refs


@app.get("/refs")
async def list_refs():
    """List available reference WAV files for dropdown selection."""
    refs = []
    seen = set()
    for r in _scan_refs():
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        refs.append({
            "id": r["id"],
            "name": r["original_name"] or r["path"].name,
            "original_name": r["original_name"],
            "lang": r["lang"],
            "lang_label": _LANG_LABELS.get(r["lang"], ""),
            "transcription": r["transcription"],
            "size": r["path"].stat().st_size,
            "source": r["source"],
        })
    return JSONResponse({"refs": refs})


@app.get("/refs/{ref_id}/audio")
async def ref_audio(ref_id: str):
    """Serve a reference voice WAV by id (dropdown ▶ preview)."""
    from fastapi.responses import Response
    for r in _scan_refs():
        if r["id"] == ref_id:
            return Response(content=r["path"].read_bytes(), media_type="audio/wav")
    raise HTTPException(404, f"Reference voice not found: {ref_id}")


@app.get("/preview-text")
async def preview_text(text: str = "", provider: str = "none"):
    """Preview Persian text processing for a given provider."""
    if not text:
        return JSONResponse({"processed_text": "", "provider": provider})
    if _process_persian_text is None:
        return JSONResponse({"processed_text": text, "provider": provider, "note": "orchestrator mode — raw text"})
    result = _process_persian_text(text, provider)
    return JSONResponse({"processed_text": result, "provider": provider})


# ── Generation History endpoints (all modes) ────────────────────

@app.get("/history/stats")
async def generation_history_stats():
    return JSONResponse(history_stats())


@app.get("/history")
async def generation_history_list(
    engine: str = "", q: str = "", voice: str = "",
    min_dur: float = 0, max_dur: float = 0,
    date_from: str = "", date_to: str = "",
    sort: str = "newest", limit: int = 50, offset: int = 0,
):
    return JSONResponse(list_history(
        engine=engine, q=q, voice=voice,
        min_dur_s=min_dur, max_dur_s=max_dur,
        date_from=date_from, date_to=date_to,
        sort=sort, limit=limit, offset=offset,
    ))


@app.get("/history/{entry_id}/audio")
async def generation_history_audio(entry_id: str):
    from fastapi.responses import Response
    path = get_history_path(entry_id)
    if not path:
        raise HTTPException(404, f"History entry not found: {entry_id}")
    return Response(content=path.read_bytes(), media_type="audio/wav")


@app.delete("/history/{entry_id}")
async def generation_history_delete(entry_id: str):
    if not delete_history_entry(entry_id):
        raise HTTPException(404, f"History entry not found: {entry_id}")
    return JSONResponse({"ok": True, "deleted": entry_id})


# ── Voice Library endpoints (only in non-orchestrator mode) ─────

if voice_library_mod:

    @app.get("/voice-library")
    async def voice_library_list(
        gender: str = "", min_duration: float = 0, max_duration: float = 999,
        min_quality: float = 0, limit: int = 200,
    ):
        voices = list_voices(gender=gender, min_duration=min_duration,
                             max_duration=max_duration, min_quality=min_quality, limit=limit)
        return JSONResponse({"voices": voices, "count": len(voices)})

    @app.get("/voice-library/stats")
    async def voice_library_stats():
        return JSONResponse(get_stats())

    @app.get("/voice-library/{voice_id}")
    async def voice_library_get(voice_id: str):
        v = get_voice(voice_id)
        if not v:
            raise HTTPException(404, f"Voice not found: {voice_id}")
        return JSONResponse(v)

    @app.get("/voice-library/{voice_id}/audio")
    async def voice_library_audio(voice_id: str):
        from fastapi.responses import Response
        path = get_voice_path(voice_id)
        if not path:
            raise HTTPException(404, f"Voice audio not found: {voice_id}")
        return Response(content=path.read_bytes(), media_type="audio/wav")

    @app.post("/voice-library/{voice_id}/use-ref")
    async def voice_library_use_ref(voice_id: str, engine: str = ""):
        path = get_voice_path(voice_id)
        if not path:
            raise HTTPException(404, f"Voice not found: {voice_id}")
        dest = UPLOAD_DIR / f"{voice_id}.wav"
        shutil.copy2(path, dest)
        v = get_voice(voice_id)
        # Sidecar metadata — clone engines (editx, s2pro) read the
        # transcription from here when ref_text isn't typed. Without a
        # faithful prompt transcript the clone reads flat/robotic.
        try:
            (UPLOAD_DIR / f"{voice_id}.json").write_text(json.dumps({
                "original_name": v.get("speaker_name", "") or f"{voice_id}.wav",
                "lang": v.get("language", v.get("lang", "")),
                "transcription": v.get("transcription", ""),
                "source": "voice-library",
            }, ensure_ascii=False))
        except Exception:
            pass
        return JSONResponse({"ok": True, "audio_prompt_id": voice_id, "voice": v,
                             "url": f"/voice-library/{voice_id}/audio"})

    @app.post("/voice-library/import-uploads")
    async def voice_library_import():
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, import_from_uploads, UPLOAD_DIR)
        return JSONResponse({"ok": True, "imported": count})

    @app.post("/voice-library/download")
    async def voice_library_download(
        count: int = 40, min_duration: float = 3.0, max_duration: float = 12.0,
        female_ratio: float = 0.5,
    ):
        loop = asyncio.get_running_loop()
        n = await loop.run_in_executor(None, download_common_voice_persian,
                                       count, min_duration, max_duration, 1, female_ratio)
        return JSONResponse({"ok": True, "downloaded": n})

    @app.delete("/voice-library/{voice_id}")
    async def voice_library_delete(voice_id: str):
        remove_voice(voice_id)
        return JSONResponse({"ok": True, "deleted": voice_id})

    @app.get("/voice-library/{voice_id}/embedding/{emb_type}")
    async def voice_library_embedding(voice_id: str, emb_type: str = "ge2e"):
        emb = get_embedding(voice_id, emb_type)
        if emb is None:
            raise HTTPException(404, f"Embedding not available for {voice_id}/{emb_type}")
        return JSONResponse({"voice_id": voice_id, "emb_type": emb_type,
                             "shape": list(emb.shape), "dtype": str(emb.dtype)})

# ── Voice Library proxy (orchestrator mode) ────────────────────────
# The orchestrator loads ZERO ML libraries — the library lives on the
# engine-current container (numpy + f5_tts/whisper for transcription) and
# every /voice-library/* route is forwarded there. The UI fetches these
# paths from this origin, so the proxy makes the whole library page work
# without importing voice_library.py here (orchestrator-safe, convention #10).
# In bare-metal mode the in-process routes above are registered instead.
if voice_library_mod is None:
    _VOICE_LIB_URL = _os.environ.get("VOICE_LIB_URL", "").rstrip("/")

    async def _voice_library_forward(subpath: str, request: Request):
        from fastapi.responses import Response
        if not _VOICE_LIB_URL:
            raise HTTPException(
                503, "VOICE_LIB_URL not configured — voice library unavailable")
        import httpx
        # No trailing slash on the bare root: engine-current's "/voice-library"
        # route 307s "/voice-library/" → "/voice-library" (a redirect the UI's
        # fetch would not follow), so build the URL without it for subpath "".
        url = (f"{_VOICE_LIB_URL}/voice-library/{subpath}" if subpath
               else f"{_VOICE_LIB_URL}/voice-library")
        try:
            # Long timeout: import-uploads whisper-transcribes each file;
            # download pulls a full dataset. Like /synthesize forwarding.
            async with httpx.AsyncClient(timeout=1800.0) as client:
                r = await client.request(
                    request.method, url,
                    params=request.query_params,
                    content=await request.body(),
                    headers={"content-type":
                             request.headers.get("content-type", "")},
                )
        except Exception as e:
            raise HTTPException(
                502, f"Voice library (engine-current) unreachable: {e}")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type"))

    # Bare /voice-library (no trailing slash) does not match {rest:path}
    # (the converter needs the slash) — register it explicitly.
    @app.api_route("/voice-library", methods=["GET", "POST"])
    async def voice_library_proxy_root(request: Request):
        return await _voice_library_forward("", request)

    @app.api_route("/voice-library/{rest:path}",
                   methods=["GET", "POST", "PUT", "DELETE"])
    async def voice_library_proxy(rest: str, request: Request):
        return await _voice_library_forward(rest, request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_lab:app", host="0.0.0.0", port=8001, reload=False, workers=1)
