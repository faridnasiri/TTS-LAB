"""
tts_lab_dispatch.py — availability checks, model loading, synthesis dispatch.

Supports TWO modes:
  1. REMOTE (containerized): engines run in separate containers.
     Set {ENGINE}_URL env vars (e.g. PIPER_URL=http://engine-current:8101).
     All _check_available / _do_synth use HTTP.

  2. LOCAL (bare metal): no URL env vars set.
     Legacy in-process behavior using find_spec + LOADERS/SYNTHERS.
     Used when running outside Docker.

The orchestrator container uses REMOTE mode exclusively.
"""
from __future__ import annotations
import base64, json, os, threading, time
from typing import Dict, Tuple

from tts_lab_config import (
    MODEL_ORDER, MODEL_INFO, HEAVY, _state,
    slog,
)
from tts_lab_utils import _wav_dur

# ── Remote engine URL resolution ─────────────────────────────────
# Engine containers expose HTTP APIs. URLs are set via env vars.
# Format: PIPER_URL=http://engine-current:8101
#         INDEXTTS_URL=http://engine-legacy:8102
#         ORPHEUS_URL=http://orpheus:8002
#         VIBEVOICE_SGLANG_URL=http://vibevoice:8000/v1/audio/speech
#
# SGLang engines use different URL env var names (historical reasons).
# All others use {UPPER_NAME}_URL.

def _build_remote_urls() -> Dict[str, str]:
    """Build remote engine URL map from environment variables."""
    urls: Dict[str, str] = {}

    for name in MODEL_ORDER:
        # Only S2-Pro genuinely needs SGLang — special env var name
        if name == "s2pro":
            url = os.environ.get("S2PRO_SGLANG_URL", "")
        # VibeVoice and Higgs are now local models in engine-mid container
        # — use standard {NAME}_URL env vars
        else:
            url = os.environ.get(f"{name.upper()}_URL", "")

        if url:
            urls[name] = url

    return urls


_REMOTE_ENGINES: Dict[str, str] = _build_remote_urls()
_REMOTE_MODE = len(_REMOTE_ENGINES) > 0

if _REMOTE_MODE:
    slog("DISPATCH", "SYSTEM", f"Remote mode — {len(_REMOTE_ENGINES)} engine URLs configured")
else:
    slog("DISPATCH", "SYSTEM", "Local mode — no engine URLs configured, using in-process dispatch")


# ── Availability ─────────────────────────────────────────────────
_import_cache: Dict[str, Tuple[bool, str]] = {}
_import_cache_lock = threading.Lock()
_sweep_done = threading.Event()


def _available(name: str) -> Tuple[bool, str]:
    with _import_cache_lock:
        if name in _import_cache:
            return _import_cache[name]

    if name in _REMOTE_ENGINES:
        result = _check_available_remote(name)
    else:
        result = _check_available_local(name)

    with _import_cache_lock:
        _import_cache[name] = result
    return result


def _check_available_remote(name: str) -> Tuple[bool, str]:
    """HTTP health check for remote engine containers."""
    url = _REMOTE_ENGINES[name]
    try:
        import httpx
        r = httpx.get(f"{url}/health", timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "")
            if status == "ok":
                # Check if the engine is AVAILABLE (not necessarily loaded).
                # In lazy-load mode, engines are available but not loaded until
                # first synthesis — so we check for absence of a "reason" field
                # (which indicates a failed startup probe), NOT the "loaded" flag.
                engines = data.get("engines", {})
                if engines:
                    engine_info = engines.get(name, {})
                    if engine_info:
                        if "reason" in engine_info:
                            return False, engine_info["reason"]
                        # Engine is available — loaded or not (lazy-load mode)
                        return True, ""
                    return False, f"engine '{name}' not found in container"
                # Single-engine containers (orpheus)
                if data.get("model_loaded"):
                    return True, ""
                return False, "model not loaded"
            return False, data.get("detail", f"status: {status}")
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def _check_available_local(name: str) -> Tuple[bool, str]:
    """Synchronous availability probe using find_spec + fs checks — no C-ext imports."""
    import importlib.util as ilu
    pkg_map = {
        "piper":      "piper",
        "kokoro":     "kokoro_onnx",
        "melo":       "melo",
        "chattts":    "ChatTTS",
        "outetts":    "outetts",
        "bark":       "bark",
        "styletts2":  "styletts2",
        "f5tts":      "f5_tts",
        "dia":        "dia",
        "xtts":       "TTS",
        "cosyvoice":  None,
        "parler":     "parler_tts",
        "chatterbox": "chatterbox",
        "chatterboxturbo": "chatterbox",
        "fishspeech": "fish_speech",
        "csm":        None,
        "qwen3tts":   "qwen_tts",
        "orpheus":    "orpheus_tts",
        "neutts":     None,
        "indextts":   "indextts",
        "zonos":      "zonos",
        "openvoice":  "openvoice",
        "matcha":     "sherpa_onnx",
        "manatts":    None,
        "vibevoice":  "transformers",
        "higgs":      "transformers",
        "omnivoice":  "omnivoice",
        "s2pro":      None,
    }

    # 1. Quick package-present check
    pkg = pkg_map.get(name)
    if pkg and not ilu.find_spec(pkg):
        return False, f"pip install {pkg} needed"

    # 2. GPU-required engines
    if name == "orpheus":
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA GPU required — not available on this machine"
        except ImportError:
            pass

    # 3. Orpheus — gated model check
    if name == "orpheus":
        if not ilu.find_spec("orpheus_tts"):
            return False, "pip install orpheus-speech"
        try:
            from huggingface_hub import hf_hub_download as _hf_dl
            from huggingface_hub.errors import GatedRepoError as _GatedErr
            _hf_dl("canopylabs/orpheus-3b-0.1-ft", "config.json",
                   local_files_only=False, local_dir="/tmp/_orpheus_check")
        except _GatedErr:
            return False, ("canopylabs/orpheus-3b-0.1-ft is gated — "
                           "request access at https://huggingface.co/canopylabs/orpheus-3b-0.1-ft "
                           "then run: huggingface-cli login")
        except Exception:
            pass

    # 4. Engine-specific file / directory checks
    from tts_lab_config import MODELS_DIR, COSYVOICE_DIR, OPENVOICE_MODELS_DIR, MANATTS_REPO_DIR, INDEXTTS_DIR
    if name == "piper":
        if not _piper_voices():
            return False, "No .onnx voice found in models/"
    elif name == "kokoro":
        if not (MODELS_DIR / "kokoro-v1.0.onnx").exists():
            return False, "kokoro-v1.0.onnx missing"
    elif name == "cosyvoice":
        if not COSYVOICE_DIR.exists():
            return False, "git clone FunAudioLLM/CosyVoice /opt/CosyVoice"
        if not (COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B").exists():
            return False, "CosyVoice2-0.5B model not downloaded"
        _yaml = COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B" / "cosyvoice2.yaml"
        if not _yaml.exists():
            return False, "CosyVoice2-0.5B yaml missing — run: python tools/download_model.py CosyVoice2-0.5B"
        if not ilu.find_spec("hyperpyyaml"):
            return False, "pip install hyperpyyaml"
    elif name == "fishspeech":
        if not ilu.find_spec("fish_speech.models.text2semantic"):
            return False, (
                "Clone v1.5.1: git clone --branch v1.5.1 https://github.com/fishaudio/fish-speech /tmp/fish-speech\n"
                "Install: pip install /tmp/fish-speech --no-build-isolation"
            )
    elif name == "neutts":
        return False, "NeuTTS Air: not configured — edit _load_neutts() in tts_lab_engines.py"
    elif name == "openvoice":
        if not (OPENVOICE_MODELS_DIR / "converter" / "config.json").exists():
            return False, f"Checkpoints missing at {OPENVOICE_MODELS_DIR}"
    elif name == "csm":
        if not (ilu.find_spec("generator") or ilu.find_spec("csm_mlx")):
            return False, "Clone: git clone SesameAILabs/csm /opt/models/csm + add .pth"
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download("sesame/csm-1b", "config.json")
        except Exception as _e:
            _s = str(_e).lower()
            if "gated" in _s or "401" in _s or "403" in _s or "authori" in _s:
                return False, "sesame/csm-1b is gated — run: huggingface-cli login"
            # Other errors (network, etc.) — ignore, probe best-effort
    elif name == "indextts":
        if not ilu.find_spec("indextts"):
            return False, "pip install git+https://github.com/index-tts/index-tts"
    elif name == "qwen3tts":
        if not ilu.find_spec("qwen_tts"):
            return False, "pip install -U qwen-tts"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        _hdrs = {"Authorization": "Bearer " + hf_token} if hf_token else {}
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                headers=_hdrs)
            with urllib.request.urlopen(req, timeout=5): pass
        except Exception as _e:
            if "401" in str(_e) or "403" in str(_e):
                return False, "Qwen3-TTS: run huggingface-cli login"
            if "404" in str(_e):
                return False, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice not found on HuggingFace"
    elif name == "matcha":
        if not ilu.find_spec("sherpa_onnx"):
            return False, "pip install sherpa-onnx"
    elif name == "manatts":
        if not MANATTS_REPO_DIR.exists():
            return False, (
                "Clone MahtaFetrat/Persian-MultiSpeaker-Tacotron2 to "
                f"{MANATTS_REPO_DIR}"
            )
        if not ilu.find_spec("scipy"):
            return False, "pip install scipy"
        if not ilu.find_spec("librosa"):
            return False, "pip install librosa"
        if not ilu.find_spec("soundfile"):
            return False, "pip install soundfile"
        if not ilu.find_spec("parallel_wavegan"):
            return False, "pip install parallel-wavegan"

    return True, ""


def _piper_voices():
    """Discover Piper ONNX voice files. Used by _check_available_local."""
    from pathlib import Path
    from tts_lab_config import MODELS_DIR
    voices = sorted(Path(str(MODELS_DIR)).glob("*.onnx"))
    return [v.stem for v in voices if "kokoro" not in v.name.lower()]


# ── Model loading ────────────────────────────────────────────────
def _ensure_loaded(name: str, params: dict) -> None:
    """Ensure the model is available. In remote mode this is a no-op
    (engine containers manage their own loading). In local mode this
    does the full VRAM-aware loading pipeline."""
    st = _state[name]

    # Remote mode: no loading needed. Engine container handles it.
    if name in _REMOTE_ENGINES:
        ok, reason = _available(name)
        if not ok:
            slog("ERROR", name, f"Remote engine not healthy: {reason}")
            raise RuntimeError(f"Remote engine '{name}' not available: {reason}")
        return

    # Local mode: full in-process loading (existing behavior)
    with st["lock"]:
        # Detect voice / model changes that require a reload
        if name == "piper":
            wanted = params.get("voice", "en_US-ryan-high")
            if st["instance"] and st.get("loaded_voice") != wanted:
                slog("LOAD", name, f"Voice change: {st.get('loaded_voice')!r} → {wanted!r} — evicting")
                _safe_del(st["instance"]); st["instance"] = None
        if name == "matcha":
            wanted_voice = params.get("voice", "khadijah")
            wanted_temp  = str(params.get("temperature", "0.333"))
            if (st["instance"] and
                (st.get("loaded_voice") != wanted_voice or
                 st.get("loaded_temperature") != wanted_temp)):
                slog("LOAD", name,
                     f"Voice/temp change: {st.get('loaded_voice')!r}/{st.get('loaded_temperature')!r} "
                     f"→ {wanted_voice!r}/{wanted_temp!r} — evicting")
                _safe_del(st["instance"]); st["instance"] = None
        if name == "chatterbox":
            wanted_model = params.get("model", "default")
            if st["instance"] and st.get("loaded_model") != wanted_model:
                slog("LOAD", name,
                     f"Model change: {st.get('loaded_model')!r} → {wanted_model!r} — evicting")
                _safe_del(st["instance"]); st["instance"] = None
        if name in ("outetts", "parler", "zonos"):
            key = {"outetts": "model_path", "parler": "model_id", "zonos": "variant"}[name]
            defaults = {"outetts": "/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf",
                        "parler": "parler-tts/parler-tts-mini-v1",
                        "zonos": "transformer"}
            wanted = params.get(key, defaults[name])
            if st["instance"] and st.get("loaded_model") != wanted:
                slog("LOAD", name, f"Model change detected — evicting current instance")
                _safe_del(st["instance"]); st["instance"] = None

        if st["instance"] is None:
            ok, reason = _available(name)
            if not ok:
                slog("ERROR", name, f"Not available: {reason}")
                raise RuntimeError(f"Not available: {reason}")
            if MODEL_INFO[name]["heavy"]:
                try:
                    import torch
                    _free, _total = torch.cuda.mem_get_info()
                    slog("VRAM", name, f"Free before evict: {_free//1048576} / {_total//1048576} MB")
                except Exception:
                    pass
                _evict_heavy(keep=name)
                try:
                    import torch
                    torch.cuda.empty_cache()
                    _free, _total = torch.cuda.mem_get_info()
                    slog("VRAM", name, f"Free after evict: {_free//1048576} / {_total//1048576} MB")
                except Exception:
                    pass
            st["status"] = "loading"
            t0 = time.perf_counter()
            try:
                if name == "piper":
                    model_arg = params.get("voice", "en_US-ryan-high")
                elif name == "matcha":
                    model_arg = params.get("voice", "khadijah")
                elif name == "chatterbox":
                    model_arg = params.get("model", "default")
                elif name == "outetts":
                    model_arg = params.get("model_path", "/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf")
                elif name == "parler":
                    model_arg = params.get("model_id", "parler-tts/parler-tts-mini-v1")
                elif name == "zonos":
                    model_arg = params.get("variant", "transformer")
                else:
                    model_arg = None
                slog("LOAD", name, f"Loading{'  arg=' + repr(model_arg) if model_arg else ''}  …")
                from tts_lab_engines import LOADERS
                if model_arg is not None:
                    st["instance"] = LOADERS[name](model_arg)
                else:
                    st["instance"] = LOADERS[name]()
                st["load_time_s"] = round(time.perf_counter() - t0, 2)
                st["status"] = "loaded"
                st["error"]  = ""
                if name == "piper":
                    st["loaded_voice"] = params.get("voice", "en_US-ryan-high")
                if name == "matcha":
                    st["loaded_voice"] = params.get("voice", "khadijah")
                    st["loaded_temperature"] = str(params.get("temperature", "0.333"))
                if name == "chatterbox":
                    st["loaded_model"] = params.get("model", "default")
                if name in ("outetts", "parler", "zonos"):
                    key = {"outetts": "model_path", "parler": "model_id", "zonos": "variant"}[name]
                    defaults = {"outetts": "/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf",
                                "parler": "parler-tts/parler-tts-mini-v1",
                                "zonos": "transformer"}
                    st["loaded_model"] = params.get(key, defaults[name])
                slog("LOAD", name, f"✅ Loaded in {st['load_time_s']}s  loaded_model={st.get('loaded_model')!r}")
            except Exception as e:
                st["status"] = "error"
                st["error"]  = str(e)
                slog("ERROR", name, f"Load failed: {e}")
                raise
        else:
            slog("LOAD", name, f"Already loaded ({st.get('loaded_model') or st.get('loaded_voice') or 'default'}) — skipping reload")


# ── Synthesis ────────────────────────────────────────────────────
def _do_synth(name: str, text: str, params: dict) -> dict:
    slog("SYNTH", name, f"▶ text={text[:60]!r}{'…' if len(text)>60 else ''}")
    slog("PARAMS", name, f"params={params}")

    # Remote mode: HTTP POST to engine container
    if name in _REMOTE_ENGINES:
        return _do_synth_remote(name, text, params)

    # Local mode: in-process synthesis
    _ensure_loaded(name, params)
    st = _state[name]
    t0 = time.perf_counter()
    from tts_lab_engines import SYNTHERS
    wav, sr = SYNTHERS[name](st["instance"], text, params)
    synth_s = time.perf_counter() - t0
    dur = _wav_dur(wav)
    slog("RESULT", name, f"✅ synth {int(synth_s*1000)} ms  dur {int(dur*1000)} ms  RTF {round(synth_s/dur,3) if dur>0 else 0}×  {sr} Hz")
    return {
        "audio_b64":    base64.b64encode(wav).decode(),
        "sample_rate":  sr,
        "synth_time_ms": int(synth_s * 1000),
        "audio_dur_ms":  int(dur * 1000),
        "rtf":          round(synth_s / dur, 4) if dur > 0 else 0,
        "load_time_s":  st["load_time_s"],
    }


def _do_synth_remote(name: str, text: str, params: dict) -> dict:
    """Synthesize via remote engine container over HTTP."""
    import httpx

    url = _REMOTE_ENGINES[name]

    # SGLang engines use a different API (OpenAI-compatible /v1/audio/speech)
    if name in ("vibevoice", "higgs", "s2pro"):
        return _do_synth_sglang(name, text, params, url)

    # Standard engine server API
    t0 = time.perf_counter()
    r = httpx.post(
        f"{url}/synthesize",
        json={"engine": name, "text": text, "params": params},
        timeout=300.0,
    )
    r.raise_for_status()
    result = r.json()
    synth_s = time.perf_counter() - t0
    dur_ms = result.get("audio_dur_ms", 0)
    dur_s = dur_ms / 1000.0 if dur_ms > 0 else 0
    slog("RESULT", name, f"✅ synth {int(synth_s*1000)} ms (remote)  dur {dur_ms} ms  RTF {round(synth_s/dur_s,3) if dur_s>0 else 0}×  {result.get('sample_rate', 0)} Hz")
    return {
        "audio_b64":    result["audio_b64"],
        "sample_rate":  result["sample_rate"],
        "synth_time_ms": int(synth_s * 1000),
        "audio_dur_ms": dur_ms,
        "rtf":          round(synth_s / dur_s, 4) if dur_s > 0 else 0,
        "load_time_s":  result.get("load_time_s", 0),
    }


def _do_synth_sglang(name: str, text: str, params: dict, url: str) -> dict:
    """Synthesize via SGLang OpenAI-compatible API."""
    import httpx

    t0 = time.perf_counter()
    r = httpx.post(
        url,
        json={"input": text, **params},
        timeout=600.0,
    )
    r.raise_for_status()
    result = r.json()
    synth_s = time.perf_counter() - t0

    # SGLang returns audio as base64 in the response
    audio_b64 = result.get("audio", result.get("audio_b64", ""))
    if not audio_b64:
        raise RuntimeError(f"SGLang response missing audio data: {list(result.keys())}")

    raw = base64.b64decode(audio_b64)
    dur_s = len(raw) / (result.get("sample_rate", 24000) * 2)  # 16-bit mono estimate

    slog("RESULT", name, f"✅ synth {int(synth_s*1000)} ms (SGLang)  dur ~{int(dur_s*1000)} ms  {result.get('sample_rate', 24000)} Hz")
    return {
        "audio_b64":    audio_b64,
        "sample_rate":  result.get("sample_rate", 24000),
        "synth_time_ms": int(synth_s * 1000),
        "audio_dur_ms":  int(dur_s * 1000),
        "rtf":          round(synth_s / dur_s, 4) if dur_s > 0 else 0,
        "load_time_s":  0,
    }


# ── Startup sweep ────────────────────────────────────────────────
def _sweep_availability() -> None:
    """Run once at startup: probe every engine and populate _import_cache."""
    for n in MODEL_ORDER:
        try:
            _available(n)
        except Exception:
            pass
    _sweep_done.set()


# ── Heavy engine eviction (local mode only) ─────────────────────
def _evict_heavy(keep: str = "") -> None:
    """Evict heavy engines from VRAM, keeping `keep`."""
    from tts_lab_utils import _safe_del
    for n in MODEL_ORDER:
        if n == keep:
            continue
        if MODEL_INFO[n].get("heavy"):
            st = _state[n]
            with st["lock"]:
                if st["instance"] is not None:
                    slog("VRAM", n, "Evicting …")
                    _safe_del(st["instance"])
                    st["instance"] = None
                    st["status"] = "idle"
                    st["loaded_model"] = None
                    st["loaded_voice"] = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
