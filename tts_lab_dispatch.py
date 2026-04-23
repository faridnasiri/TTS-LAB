"""
tts_lab_dispatch.py — availability checks, model loading, synthesis dispatch.
"""
from __future__ import annotations
import base64, os, threading, time
from typing import Dict, Tuple

from tts_lab_config  import (
    MODEL_ORDER, MODEL_INFO, HEAVY, _state,
    OUTETTS_DEFAULT_GGUF, COSYVOICE_DIR, INDEXTTS_DIR, OPENVOICE_MODELS_DIR,
    slog,
)
from tts_lab_utils   import _safe_del, _evict_heavy, _wav_dur, _piper_voices, _require_gpu
from tts_lab_engines import LOADERS, SYNTHERS

# ── Availability cache ────────────────────────────────────────────────────────
_import_cache: Dict[str, Tuple[bool, str]] = {}
_import_cache_lock = threading.Lock()
_sweep_done = threading.Event()


def _available(name: str) -> Tuple[bool, str]:
    with _import_cache_lock:
        if name in _import_cache:
            return _import_cache[name]
    result = _check_available(name)
    with _import_cache_lock:
        _import_cache[name] = result
    return result


def _check_available(name: str) -> Tuple[bool, str]:
    """Synchronous availability probe using find_spec + fs checks only — no C-ext imports."""
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
        "fishspeech": "fish_speech",
        "csm":        None,
        "qwen3tts":   "qwen_tts",
        "orpheus":    "orpheus_tts",
        "neutts":     None,
        "indextts":   "indextts",
        "zonos":      "zonos",
        "openvoice":  "openvoice",
    }

    # 1. Quick package-present check
    pkg = pkg_map.get(name)
    if pkg and not ilu.find_spec(pkg):
        return False, f"pip install {pkg} needed"

    # 2. GPU-required engines
    _GPU_REQUIRED = {"outetts", "bark", "orpheus"}
    if name in _GPU_REQUIRED:
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
            import urllib.request, urllib.error
            req = urllib.request.Request(
                "https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/resolve/main/config.json")
            with urllib.request.urlopen(req, timeout=5): pass
        except urllib.error.HTTPError as _e:
            if _e.code in (401, 403):
                return False, "canopylabs/orpheus-3b-0.1-ft is gated — run: huggingface-cli login"
        except Exception:
            pass

    # 4. Engine-specific file / directory checks
    if name == "piper":
        if not _piper_voices():
            return False, "No .onnx voice found in models/"
    elif name == "kokoro":
        from tts_lab_config import MODELS_DIR
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
            import urllib.request, urllib.error
            req = urllib.request.Request(
                "https://huggingface.co/sesame/csm-1b/resolve/main/config.json")
            with urllib.request.urlopen(req, timeout=5): pass
        except urllib.error.HTTPError as _e:
            if _e.code in (401, 403):
                return False, "sesame/csm-1b is gated — run: huggingface-cli login"
        except Exception:
            pass
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

    return True, ""


def _ensure_loaded(name: str, params: dict) -> None:
    """Load the model for `name` into VRAM if not already loaded. Thread-safe."""
    st = _state[name]
    with st["lock"]:
        # Detect voice / model changes that require a reload
        if name == "piper":
            wanted = params.get("voice", "en_US-ryan-high")
            if st["instance"] and st.get("loaded_voice") != wanted:
                slog("LOAD", name, f"Voice change: {st.get('loaded_voice')!r} → {wanted!r} — evicting")
                _safe_del(st["instance"]); st["instance"] = None
        if name in ("outetts", "parler", "zonos"):
            key = {"outetts": "model_path", "parler": "model_id", "zonos": "variant"}[name]
            defaults = {"outetts": OUTETTS_DEFAULT_GGUF,
                        "parler": "parler-tts/parler-tts-mini-v1",
                        "zonos": "transformer"}
            wanted = params.get(key, defaults[name])
            slog("LOAD", name, f"Wanted model: {wanted!r}  |  currently loaded: {st.get('loaded_model')!r}")
            if st["instance"] and st.get("loaded_model") != wanted:
                slog("LOAD", name, f"Model change detected — evicting current instance")
                _safe_del(st["instance"]); st["instance"] = None

        if st["instance"] is None:
            ok, reason = _available(name)
            if not ok:
                slog("ERROR", name, f"Not available: {reason}")
                raise RuntimeError(f"Not available: {reason}")
            if MODEL_INFO[name]["heavy"]:
                _evict_heavy(keep=name)
            st["status"] = "loading"
            t0 = time.perf_counter()
            try:
                if name == "piper":
                    model_arg = params.get("voice", "en_US-ryan-high")
                elif name == "outetts":
                    model_arg = params.get("model_path", OUTETTS_DEFAULT_GGUF)
                elif name == "parler":
                    model_arg = params.get("model_id", "parler-tts/parler-tts-mini-v1")
                elif name == "zonos":
                    model_arg = params.get("variant", "transformer")
                else:
                    model_arg = None
                slog("LOAD", name, f"Loading{'  arg=' + repr(model_arg) if model_arg else ''}  …")
                if model_arg is not None:
                    st["instance"] = LOADERS[name](model_arg)
                else:
                    st["instance"] = LOADERS[name]()
                st["load_time_s"] = round(time.perf_counter() - t0, 2)
                st["status"] = "loaded"
                st["error"]  = ""
                if name == "piper":
                    st["loaded_voice"] = params.get("voice", "en_US-ryan-high")
                if name in ("outetts", "parler", "zonos"):
                    key = {"outetts": "model_path", "parler": "model_id", "zonos": "variant"}[name]
                    defaults = {"outetts": OUTETTS_DEFAULT_GGUF,
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


def _do_synth(name: str, text: str, params: dict) -> dict:
    slog("SYNTH", name, f"▶ text={text[:60]!r}{'…' if len(text)>60 else ''}")
    slog("PARAMS", name, f"params={params}")
    _ensure_loaded(name, params)
    st = _state[name]
    t0 = time.perf_counter()
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


def _sweep_availability() -> None:
    """Run once at startup: probe every engine and populate _import_cache."""
    for n in MODEL_ORDER:
        try:
            _available(n)
        except Exception:
            pass
    _sweep_done.set()
