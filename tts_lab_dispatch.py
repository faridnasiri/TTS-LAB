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
    MODEL_ORDER, MODEL_INFO, HEAVY, SYNTH_TIMEOUT, _state,
    _ref_wav_path, _ref_transcript, slog,
)
from tts_lab_utils import _wav_dur

# ── Container Management (Docker socket) ─────────────────────────
# The LLM (Qwen 3.6) uses ~13 GB VRAM, S2-Pro ~11 GB (always-resident).
# Heavy TTS engines need that VRAM, so we stop these containers before
# heavy synthesis and restart them when the engine is called again.
# Requires /var/run/docker.sock mounted in the orchestrator container.
_LLM_CONTAINER_NAME = "tts-lab-llm-qwen36"
_S2PRO_CONTAINER_NAME = "tts-lab-s2pro"
_DOCKER_SOCK = "/var/run/docker.sock"
_HAS_DOCKER_SOCK = os.path.exists(_DOCKER_SOCK)

# vLLM-backed engine containers: vLLM never returns its memory arena to the
# driver on an in-process evict (engine core can't re-init either) — the only
# real eviction is a container restart. editx (8105) + orpheus (8002) are
# vLLM; engine-current/qwen/mid/legacy are torch-based and /evict in-place
# works for them.
_VLLM_CONTAINERS: dict[str, str] = {
    "editx":   "tts-lab-engine-editx",
    "orpheus": "tts-lab-orpheus",
}

# Engine containers (for PID→container mapping of GPU processes).
_GPU_CONTAINERS: tuple[str, ...] = (
    "tts-lab-engine-current", "tts-lab-engine-qwen", "tts-lab-engine-mid",
    "tts-lab-engine-legacy", "tts-lab-engine-editx", "tts-lab-s2pro",
    "tts-lab-orpheus", "tts-lab-vibevoice", "tts-lab-higgs",
)


def _docker_api(method: str, path: str, content: bytes | None = None,
                timeout: float = 10.0) -> tuple[int, str]:
    """Call Docker Engine API via Unix socket. Returns (status_code, body).

    Content-bearing calls (exec create, etc.) MUST send the JSON Content-Type
    header — without it the API rejects with 400 "malformed Content-Type".
    """
    import httpx
    headers = {"Content-Type": "application/json"} if content is not None else None
    transport = httpx.HTTPTransport(uds=_DOCKER_SOCK)
    with httpx.Client(transport=transport, timeout=timeout) as client:
        r = client.request(method, f"http://localhost{path}", content=content,
                           headers=headers)
        return r.status_code, r.text


def _container_running(name: str) -> bool:
    """Check if a named container is currently running."""
    if not _HAS_DOCKER_SOCK:
        return False
    try:
        code, body = _docker_api("GET", f"/v1.49/containers/{name}/json")
        if code == 200:
            import json as _j
            state = _j.loads(body).get("State", {})
            return state.get("Running", False)
    except Exception:
        pass
    return False


def _container_stop(name: str, label: str = "container") -> bool:
    """Stop a named container to free VRAM."""
    if not _HAS_DOCKER_SOCK:
        slog("VRAM", "evict", f"No Docker socket — cannot stop {name}")
        return False
    try:
        if not _container_running(name):
            slog("VRAM", "evict", f"{label} container already stopped ({name})")
            return True
        code, _ = _docker_api("POST", f"/v1.49/containers/{name}/stop")
        ok = code in (204, 304)
        slog("VRAM", "evict", f"Stop {label} container ({name}) → HTTP {code} {'✓' if ok else '✗'}")
        return ok
    except Exception as e:
        slog("VRAM", "evict", f"Failed to stop {name}: {e}")
        return False


def _container_start(name: str, label: str = "container") -> bool:
    """Start a named container (called before its engine is used)."""
    if not _HAS_DOCKER_SOCK:
        return False
    try:
        if _container_running(name):
            return True
        code, _ = _docker_api("POST", f"/v1.49/containers/{name}/start")
        ok = code in (204, 304)
        slog("VRAM", "start", f"Start {label} container ({name}) → HTTP {code} {'✓' if ok else '✗'}")
        return ok
    except Exception as e:
        slog("VRAM", "start", f"Failed to start {name}: {e}")
        return False


def _container_restart(name: str, label: str = "container") -> bool:
    """Restart a named container — the only real eviction for vLLM-backed
    containers, whose memory arena survives any in-process unload."""
    if not _HAS_DOCKER_SOCK:
        slog("VRAM", "evict", f"No Docker socket — cannot restart {name}")
        return False
    try:
        if not _container_running(name):
            slog("VRAM", "evict", f"{label} container not running ({name}) — nothing to restart")
            return True
        code, _ = _docker_api("POST", f"/v1.49/containers/{name}/restart")
        ok = code in (204, 304)
        slog("VRAM", "evict", f"Restart {label} container ({name}) → HTTP {code} {'✓' if ok else '✗'}")
        return ok
    except Exception as e:
        slog("VRAM", "evict", f"Failed to restart {name}: {e}")
        return False


def _running_containers() -> set[str]:
    """Names of all running containers (via Docker API)."""
    if not _HAS_DOCKER_SOCK:
        return set()
    try:
        import json as _j
        code, body = _docker_api("GET", "/v1.49/containers/json")
        if code != 200:
            return set()
        return {c["Names"][0].lstrip("/") for c in _j.loads(body)}
    except Exception:
        return set()


def _docker_exec(container: str, cmd: str, timeout: float = 10.0) -> str:
    """Run a shell command inside a container via the Docker API; return
    combined stdout (Tty:true merges the streams, no multiplexing headers)."""
    import json as _j
    try:
        code, body = _docker_api(
            "POST", f"/v1.49/containers/{container}/exec", timeout=timeout,
            content=_j.dumps({"AttachStdout": True, "AttachStderr": True,
                              "Tty": True, "Cmd": ["sh", "-c", cmd]}).encode())
        if code != 201:
            return ""
        exec_id = _j.loads(body)["Id"]
        code, body = _docker_api(
            "POST", f"/v1.49/exec/{exec_id}/start", timeout=timeout,
            content=b'{"Detach": false, "Tty": true}')
        return body if code == 200 else ""
    except Exception:
        return ""


def _llm_container_running() -> bool:
    """Check if the LLM container is currently running."""
    return _container_running(_LLM_CONTAINER_NAME)


def _stop_llm_container() -> bool:
    """Stop the LLM container to free VRAM for heavy TTS engines."""
    return _container_stop(_LLM_CONTAINER_NAME, label="LLM")


def _start_llm_container() -> bool:
    """Start the LLM container (called before LLM inference)."""
    return _container_start(_LLM_CONTAINER_NAME, label="LLM")


def _start_llm_container() -> bool:
    """Start the LLM container (called before LLM inference)."""
    if not _HAS_DOCKER_SOCK:
        return False
    try:
        if _llm_container_running():
            return True
        code, _ = _docker_api("POST", f"/v1.49/containers/{_LLM_CONTAINER_NAME}/start")
        ok = code in (204, 304)
        slog("LLM", "start", f"Start LLM container → HTTP {code} {'✓' if ok else '✗'}")
        return ok
    except Exception as e:
        slog("LLM", "start", f"Failed to start LLM container: {e}")
        return False

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
        else:
            url = os.environ.get(f"{name.upper()}_URL", "")

        if url:
            urls[name] = url

    # Build deduplicated set of unique engine container URLs
    # (multiple engines share the same container URL, e.g. piper + kokoro both
    # point to http://engine-current:8101 — we only evict ONCE per container.)
    # Skip SGLang URLs — those containers don't have /evict endpoint.
    global _ENGINE_CONTAINER_URLS
    _ENGINE_CONTAINER_URLS = set()
    for url in urls.values():
        stripped = url.rstrip("/")
        if stripped in _SGLANG_URLS:
            continue
        _ENGINE_CONTAINER_URLS.add(stripped)

    return urls


_SGLANG_URLS: set[str] = {
    os.environ[k].rstrip("/") for k in os.environ if k.endswith("_SGLANG_URL")
}
_ENGINE_CONTAINER_URLS: set[str] = set()  # populated by _build_remote_urls
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
        # SGLang-style containers (s2pro via sgl-omni) return a bare ok on
        # /health with no `engines` map and no `model_loaded` — any HTTP 200
        # while serving counts as available. The configured URL points at
        # the API endpoint (…/v1/audio/speech), but /health lives at the
        # server ROOT — strip the /v1/ path before probing.
        if url.rstrip("/") in _SGLANG_URLS:
            base = url.rstrip("/").split("/v1/")[0]
            r = httpx.get(f"{base}/health", timeout=10.0)
            if r.status_code == 200:
                return True, ""
            return False, f"HTTP {r.status_code}"
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
        # A stopped/removed container surfaces as a Docker-DNS gaierror
        # ("Temporary failure in name resolution") or a connect refusal —
        # collapse those into one clean line instead of leaking the raw
        # errno into the UI status panel.
        if "name resolution" in str(e) or type(e).__name__ in (
            "gaierror", "ConnectError", "ConnectTimeout", "ConnectFail",
        ):
            return False, "offline — engine container not running"
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
        "indextts":   "indextts",
        "zonos":      "zonos",
        "openvoice":  "openvoice",
        "matcha":     "sherpa_onnx",
        "manatts":    None,
        "vibevoice":  "transformers",
        "higgs":      "transformers",
        "omnivoice":  "omnivoice",
        "s2pro":      None,
        "editx":      "vllm",
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
    from tts_lab_config import MODELS_DIR, COSYVOICE_DIR, OPENVOICE_MODELS_DIR, MANATTS_REPO_DIR, INDEXTTS_DIR, EDITX_REPO_DIR, EDITX_MODEL_DIR
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
                "https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                headers=_hdrs)
            with urllib.request.urlopen(req, timeout=5): pass
        except Exception as _e:
            if "401" in str(_e) or "403" in str(_e):
                return False, "Qwen3-TTS: run huggingface-cli login"
            if "404" in str(_e):
                return False, "Qwen/Qwen3-TTS-12Hz-1.7B-Base not found on HuggingFace"
    elif name == "matcha":
        if not ilu.find_spec("sherpa_onnx"):
            return False, "pip install sherpa-onnx"
    elif name == "editx":
        if not EDITX_REPO_DIR.exists():
            return False, f"Clone stepfun-ai/Step-Audio-EditX to {EDITX_REPO_DIR}"
        if not any((EDITX_MODEL_DIR / n).exists()
                   for n in ("Step-Audio-EditX", "Step-Audio-EditX-AWQ-4bit")):
            return False, f"EditX weights missing at {EDITX_MODEL_DIR}"
        if not (EDITX_MODEL_DIR / "Step-Audio-Tokenizer").exists():
            return False, "Step-Audio-Tokenizer missing under " + str(EDITX_MODEL_DIR)
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
            wanted_model = params.get("model", "persian")
            if st["instance"] and st.get("loaded_model") != wanted_model:
                slog("LOAD", name,
                     f"Model change: {st.get('loaded_model')!r} -> {wanted_model!r} - evicting")
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
                    model_arg = params.get("model", "persian")
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
                    st["loaded_model"] = params.get("model", "persian")
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
        # LLM engines (text→text) — evict TTS first, then route to llama.cpp
        if MODEL_INFO.get(name, {}).get("engine_type") == "llm":
            return _do_synth_llm(name, text, params)
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
    resp = {
        "audio_b64":    base64.b64encode(wav).decode(),
        "sample_rate":  sr,
        "synth_time_ms": int(synth_s * 1000),
        "audio_dur_ms":  int(dur * 1000),
        "rtf":          round(synth_s / dur, 4) if dur > 0 else 0,
        "load_time_s":  st["load_time_s"],
    }
    # Engines that chunk (chatterboxturbo) attach per-chunk offsets via _state
    chunks = _state.get(name, {}).get("last_chunks")
    if chunks:
        resp["chunks"] = chunks
    return resp


def _do_synth_remote(name: str, text: str, params: dict) -> dict:
    """Synthesize via remote engine container over HTTP."""
    import httpx

    url = _REMOTE_ENGINES[name]

    # SGLang engines use a different API (OpenAI-compatible /v1/audio/speech)
    if name in ("vibevoice", "higgs", "s2pro"):
        return _do_synth_sglang(name, text, params, url)

    # Heavy engines need significant VRAM — evict LLM first if it's running
    if name in HEAVY and _HAS_DOCKER_SOCK:
        _stop_llm_container()
        # S2-Pro is always-resident (~11 GB) — stop its container so heavy
        # engines (EditX, Bark, ...) fit in 16 GB. Restarted on next s2pro call.
        if _container_running(_S2PRO_CONTAINER_NAME):
            _container_stop(_S2PRO_CONTAINER_NAME, label="S2-Pro")

    # Standard engine server API. Timeout follows SYNTH_TIMEOUT (tts_lab.py
    # waits at least this long on its side): editx's first load pays a
    # ~9 GB HF download + vLLM init and needs the 600 s allowance.
    t0 = time.perf_counter()
    r = httpx.post(
        f"{url}/synthesize",
        json={"engine": name, "text": text, "params": params},
        timeout=SYNTH_TIMEOUT.get(name, 300.0),
    )
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Forward the engine's real error detail (FastAPI "detail" or our
        # own "error" field) so the UI shows why synthesis failed instead
        # of a generic HTTP wrapper message.
        detail = ""
        try:
            body = r.json()
            detail = body.get("detail") or body.get("error") or ""
        except Exception:
            pass
        raise RuntimeError(f"{name} failed (HTTP {r.status_code})"
                           + (f": {detail}" if detail else "")) from e
    result = r.json()
    synth_s = time.perf_counter() - t0
    dur_ms = result.get("audio_dur_ms", 0)
    dur_s = dur_ms / 1000.0 if dur_ms > 0 else 0
    slog("RESULT", name, f"✅ synth {int(synth_s*1000)} ms (remote)  dur {dur_ms} ms  RTF {round(synth_s/dur_s,3) if dur_s>0 else 0}×  {result.get('sample_rate', 0)} Hz")
    resp = {
        "audio_b64":    result["audio_b64"],
        "sample_rate":  result["sample_rate"],
        "synth_time_ms": int(synth_s * 1000),
        "audio_dur_ms": dur_ms,
        "rtf":          round(synth_s / dur_s, 4) if dur_s > 0 else 0,
        "load_time_s":  result.get("load_time_s", 0),
    }
    # Per-chunk offsets (chatterboxturbo) — forwarded from the engine server
    if result.get("chunks"):
        resp["chunks"] = result["chunks"]
    return resp


# sgl-omni /v1/audio/speech accepts ONLY this language enum — 2-letter codes
# 400 ("language must be one of: Auto, Chinese, English, French, German,
# Italian, Japanese, Korean, Portuguese, Russian, Spanish" — verified against
# the s2pro container 2026-08-23). Persian has no enum entry; "Auto" lets the
# model auto-detect it (S2-Pro's model-level language support includes fa).
_SGLANG_LANG_ENUM = {
    "auto": "Auto", "chinese": "Chinese", "english": "English",
    "french": "French", "german": "German", "italian": "Italian",
    "japanese": "Japanese", "korean": "Korean", "portuguese": "Portuguese",
    "russian": "Russian", "spanish": "Spanish",
}


def _sglang_lang(v) -> str:
    """Map a lab language code to the sgl-omni enum; unknown → Auto (auto-
    detect). Lowercased comparison so 'en'/'English' both land on the enum."""
    return _SGLANG_LANG_ENUM.get(str(v).strip().lower(), "Auto")


# sgl-omni validates numeric request fields STRICTLY — no pydantic coercion:
# a JSON string "0.8" 400s ("temperature must be a float", "max_new_tokens
# must be an integer", "seed must be an integer" — verified 2026-08-23). The
# UI sliders send strings, so the dispatcher coerces these to real JSON
# numbers before POSTing. Int vs float sets.
_SGLANG_INT_PARAMS = frozenset(("top_k", "max_new_tokens", "seed"))
_SGLANG_FLOAT_PARAMS = frozenset(("temperature", "top_p", "repetition_penalty", "speed"))


def _do_synth_sglang(name: str, text: str, params: dict, url: str) -> dict:
    """Synthesize via SGLang OpenAI-compatible API (/v1/audio/speech).

    Mirrors _do_synth_llm's eviction protocol: SGLang engines are
    always-resident (~11 GB for S2-Pro), so the LLM and all in-process
    engine containers are evicted first. If the s2pro container was
    stopped to free VRAM for another heavy engine, it is started again
    and we wait for it to serve before POSTing.
    """
    import httpx

    if name == "s2pro" and _HAS_DOCKER_SOCK:
        _stop_llm_container()
        _evict_all_tts_engines()
        if not _container_running(_S2PRO_CONTAINER_NAME):
            _container_start(_S2PRO_CONTAINER_NAME, label="S2-Pro")
            # Model load after container start takes minutes — poll health.
            # 600 s covers first boot: ~10 GB model download + the one-time
            # flashinfer sm_120 JIT kernel compile. /health lives at the
            # server ROOT, not under the configured /v1/audio/speech path.
            health_url = f"{url.rstrip('/').split('/v1/')[0]}/health"
            deadline = time.monotonic() + 600.0
            while time.monotonic() < deadline:
                try:
                    if httpx.get(health_url, timeout=5.0).status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(5.0)
            else:
                raise RuntimeError("S2-Pro container started but /health never returned 200 — check docker logs tts-lab-s2pro")

    # Voice cloning: map the lab's ref-id params to SGLang's references
    # array ({audio_path, text}). Paths must be valid inside the container —
    # the s2pro service mounts /opt/arthur/reference_voices + /tmp/tts_uploads.
    payload: dict = {"input": text}
    for k, v in params.items():
        if k in ("audio_prompt_id", "ref_audio", "ref_text"):
            continue
        # Blank params must not be forwarded — sgl-omni's pydantic schema
        # rejects "" for int/float fields (blank seed → 422), and an empty
        # string is never a meaningful value here (blank voice → server
        # "default" kicks in). A remote caller sending numbers passes them
        # through untouched.
        if v is None or v == "":
            continue
        # Coerce numeric fields to real JSON numbers (sgl-omni is strict —
        # string "0.8" → 400). Garbage values are dropped rather than 400'd.
        if k in _SGLANG_INT_PARAMS:
            try:
                v = int(v)
            except (ValueError, TypeError):
                continue
        elif k in _SGLANG_FLOAT_PARAMS:
            try:
                v = float(v)
            except (ValueError, TypeError):
                continue
        # sgl-omni's language enum rejects 2-letter codes; map them so
        # Persian (fa → Auto) and other non-enum languages stop 400ing.
        if name == "s2pro" and k == "language":
            v = _sglang_lang(v)
        payload[k] = v
    ref_id = (params.get("audio_prompt_id") or params.get("ref_audio") or "").strip()
    if ref_id:
        ref_path = _ref_wav_path(ref_id)
        if ref_path:
            ref_text = (params.get("ref_text") or "").strip()
            if not ref_text:
                # Voice-library voices carry a real transcript in their
                # sidecar (written by /use-ref) — a clone without a faithful
                # prompt transcript reads flat/robotic.
                ref_text = _ref_transcript(ref_path)
            payload["references"] = [{
                "audio_path": str(ref_path),
                "text":       ref_text,
            }]

    t0 = time.perf_counter()
    r = httpx.post(
        url,
        json=payload,
        timeout=600.0,
    )
    r.raise_for_status()
    synth_s = time.perf_counter() - t0

    # sgl-omni's /v1/audio/speech returns the audio as a BINARY body —
    # response_format ∈ {wav, mp3, flac, pcm, aac, opus}, there is NO
    # JSON/base64 mode (SUPPORTED_TTS_RESPONSE_FORMATS in serve/protocol.py;
    # verified 2026-08-23 against the s2pro container, default = wav). The
    # initial plan assumed a JSON {audio: b64} payload — wrong; keep the
    # binary handling.
    raw = r.content
    if not raw:
        raise RuntimeError("SGLang response missing audio data (empty body)")

    # Sample rate + duration from the WAV header via stdlib only — the
    # orchestrator container has no numpy, and tts_lab_utils._wav_dur()
    # sits behind ML imports. Fallbacks: 24000 Hz estimate by size.
    sample_rate, n_frames = 24000, 0
    try:
        import io as _io
        import wave as _wave
        with _wave.open(_io.BytesIO(raw)) as _w:
            sample_rate = _w.getframerate()
            n_frames = _w.getnframes()
    except Exception:
        pass
    dur_s = n_frames / sample_rate if n_frames > 0 else len(raw) / (sample_rate * 2)

    audio_b64 = base64.b64encode(raw).decode()
    slog("RESULT", name, f"✅ synth {int(synth_s*1000)} ms (SGLang)  dur {int(dur_s*1000)} ms  {sample_rate} Hz")
    return {
        "audio_b64":    audio_b64,
        "sample_rate":  sample_rate,
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


# ── Global engine eviction (for LLM VRAM clearance) ────────────────

def _evict_all_tts_engines() -> dict:
    """Evict every engine from VRAM. Returns per-target results.

    Called by the UI 'Evict VRAM' button (and historically before LLM
    synthesis). Three classes of target, because one eviction method does
    NOT fit all:
      1. SGLang engines (s2pro — always-resident, no /evict endpoint)
         → stop the container (restarted lazily on next synthesis)
      2. vLLM-backed containers (editx, orpheus) → restart the container
         (vLLM's arena can't be freed in-process)
      3. standard torch engine containers → POST /evict (in-process unload)
    """
    import httpx
    results: dict[str, dict] = {}
    running = _running_containers()
    _vllm_urls = {_REMOTE_ENGINES[e].rstrip("/")
                  for e in _VLLM_CONTAINERS if e in _REMOTE_ENGINES}

    # 1) SGLang: the model is resident while the server container runs.
    if _HAS_DOCKER_SOCK and _S2PRO_CONTAINER_NAME in running:
        results[f"container:{_S2PRO_CONTAINER_NAME}"] = {
            "evicted": _container_stop(_S2PRO_CONTAINER_NAME, label="S2-Pro"),
            "mode": "container-stop",
        }

    # 2) vLLM-backed containers: restart is the only real eviction.
    for eng, cname in sorted(_VLLM_CONTAINERS.items()):
        if cname in running:
            results[f"container:{cname}"] = {
                "evicted": _container_restart(cname, label=eng),
                "mode": "container-restart",
            }

    # 3) Standard torch engine containers: in-process /evict.
    for base_url in sorted(_ENGINE_CONTAINER_URLS):
        if base_url in _vllm_urls:
            continue  # already recycled above
        # Compose-style URL (host has no dot) whose container is down →
        # skip: avoids DNS errors from dead services (legacy, orpheus).
        host = base_url.split("//")[-1].split(":")[0]
        if _HAS_DOCKER_SOCK and "." not in host and f"tts-lab-{host}" not in running:
            results[base_url] = {"error": "container not running — skipped"}
            continue
        try:
            evict_url = f"{base_url}/evict"
            r = httpx.post(evict_url, timeout=10.0)
            if r.status_code == 200:
                results[base_url] = r.json()
            else:
                results[base_url] = {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
        except Exception as e:
            results[base_url] = {"error": str(e)}
    return results


# ── Loaded-model probing (for orchestrator /status) ────────────────
# Engine containers keep AT MOST ONE model resident (lazy-load + evict).
# /health exposes which one via `engines.<name>.loaded` + `current_engine`
# + device-wide GPU stats. Cache briefly — /status gets polled.

_loaded_probe_cache: dict = {}
_loaded_probe_ts: float = 0.0
_LOADED_PROBE_TTL = 2.0  # seconds


# nvidia-smi exec'd inside a normal container namespace-translates PIDs
# (every process shows as pid 1) and hides other containers' processes —
# useless for attribution. Instead we run it in a lazily-created --pid=host
# probe container (created via the Docker API, kept running, AutoRemove) so
# the driver reports HOST PIDs for the whole box, which then map onto the
# `docker top` table below. The probe is ~0 VRAM and recreated on demand if
# it dies (reboot, compose down, manual kill).
_GPU_PROBE_NAME = "tts-lab-gpu-probe"
_GPU_PROBE_IMAGE_CANDIDATES = (
    "tts-lab-engine-editx:latest",
    "tts-lab-engine-current:latest",
    "tts-lab-sglang-omni:latest",
)
_gpu_probe_id: str | None = None  # container ID once created


def _gpu_probe_exec(cmd: str, timeout: float = 15.0) -> str:
    """Run a shell command in the --pid=host GPU probe container.

    Returns combined stdout (Tty:true raw stream), or "" when no GPU image
    is present / the Docker API fails. The probe container is created once
    and reused; a dead probe is removed and recreated on the next call.
    """
    global _gpu_probe_id
    import json as _j
    if not _HAS_DOCKER_SOCK:
        return ""
    for attempt in range(2):  # [ensure-probe, exec] x2 with recreate between
        if _gpu_probe_id and _container_running(_gpu_probe_id):
            out = _docker_exec(_gpu_probe_id, cmd, timeout)
            if out:
                return out
        # probe missing / dead / failed — (re)create it
        code, body = _docker_api("GET", "/v1.49/images/json")
        if code != 200:
            return ""
        have = set()
        for img in _j.loads(body):
            for t in img.get("RepoTags") or []:
                have.add(t)
        img = next((t for t in _GPU_PROBE_IMAGE_CANDIDATES if t in have), None)
        if img is None:
            return ""
        if _gpu_probe_id:
            _docker_api("DELETE", f"/v1.49/containers/{_gpu_probe_id}?force=1")
            _gpu_probe_id = None
        payload = _j.dumps({
            "Image": img,
            "Cmd": ["tail", "-f", "/dev/null"],
            "HostConfig": {
                "PidMode": "host",
                "AutoRemove": True,
                "DeviceRequests": [
                    {"Driver": "nvidia", "Count": 1, "Capabilities": [["gpu"]]},
                ],
            },
        }).encode()
        code, body = _docker_api(
            "POST", f"/v1.49/containers/create?name={_GPU_PROBE_NAME}",
            content=payload)
        if code == 409:  # name taken by a stale container — force-remove, retry
            _docker_api("DELETE", f"/v1.49/containers/{_GPU_PROBE_NAME}?force=1")
            code, body = _docker_api(
                "POST", f"/v1.49/containers/create?name={_GPU_PROBE_NAME}",
                content=payload)
        if code != 201:
            return ""
        cid = _j.loads(body)["Id"]
        code, _ = _docker_api("POST", f"/v1.49/containers/{cid}/start")
        if code not in (204, 304):
            return ""
        _gpu_probe_id = cid
    return ""


def _gpu_process_breakdown() -> list[dict]:
    """Which processes actually hold the GPU, and which container each is in.

    nvidia-smi runs in a --pid=host probe container (see _gpu_probe_exec) and
    reports HOST PIDs for every GPU process on the box; `docker top` maps
    those back to containers. Processes outside the container fleet (the
    bare-metal Image Lab service, stray python) are labelled "host".
    Returns [] when no GPU container is up or the Docker API fails.
    """
    import json as _j
    import re as _re
    if not _HAS_DOCKER_SOCK:
        return []
    pid2cont: dict[str, str] = {}
    for cname in _GPU_CONTAINERS:
        try:
            code, body = _docker_api("GET", f"/v1.49/containers/{cname}/top")
            if code != 200:
                continue
            data = _j.loads(body)
            titles = data.get("Titles", [])
            pid_idx = titles.index("PID") if "PID" in titles else 1
            for row in data.get("Processes", []):
                if len(row) > pid_idx and row[pid_idx].isdigit():
                    pid2cont[row[pid_idx]] = cname
        except Exception:
            continue
    raw = _gpu_probe_exec("nvidia-smi --query-compute-apps=pid,used_memory,process_name "
                          "--format=csv,noheader,nounits")
    procs = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        # used_memory may come back as "1234" or "1234 MiB" depending on the
        # driver — never let the int() cast 500 the whole /status.
        m = _re.search(r"\d+", parts[1])
        if m is None:
            continue
        procs.append({
            "pid":      int(parts[0]),
            "mb":       int(m.group()),
            "process":  parts[2],
            "container": pid2cont.get(parts[0], "host"),
        })
    return procs


def _probe_containers_loaded() -> dict:
    """Query every engine container's /health; return per-container info.

    Result shape: {base_url: {"engines": {name: {"loaded": bool, ...}},
                              "current_engine": str | None,
                              "gpu": {...} | None}}
    SGLang servers (s2pro/vibevoice/higgs — always-resident models, no
    /evict endpoint) are probed at their server root and reported under
    `sglang_up` instead.
    """
    global _loaded_probe_cache, _loaded_probe_ts
    import httpx
    now = time.time()
    with _import_cache_lock:
        if _loaded_probe_cache and now - _loaded_probe_ts < _LOADED_PROBE_TTL:
            return _loaded_probe_cache

    out: dict = {}
    for base in sorted(_ENGINE_CONTAINER_URLS):
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                out[base] = {
                    "engines":         data.get("engines", {}),
                    "current_engine":  data.get("current_engine"),
                    "gpu":             data.get("gpu"),
                }
            else:
                out[base] = {"engines": {}, "current_engine": None, "gpu": None,
                             "error": f"HTTP {r.status_code}"}
        except Exception as e:
            out[base] = {"engines": {}, "current_engine": None, "gpu": None,
                         "error": str(e)}

    # SGLang servers: no `engines` map — model(s) resident iff server up.
    for sgl_url in sorted(_SGLANG_URLS):
        base = sgl_url.split("/v1/")[0]
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            out[base] = {
                "engines": {}, "current_engine": None, "gpu": None,
                "sglang_up": r.status_code == 200,
            }
        except Exception:
            out[base] = {"engines": {}, "current_engine": None, "gpu": None,
                         "sglang_up": False}

    # GPU process breakdown (per-container, incl. bare-metal Image Lab).
    # Stored under a sentinel key — never a container URL.
    out["__gpu_processes__"] = _gpu_process_breakdown()

    with _import_cache_lock:
        _loaded_probe_cache = out
        _loaded_probe_ts = now
    return out


def _evict_engine(name: str) -> dict:
    """Evict ONE engine from VRAM.

    Local mode: unload the in-process instance + empty CUDA cache.
    Remote mode: POST /evict on the engine's container (standard engine
    servers). SGLang servers have no /evict — their models are
    always-resident, so stop the container itself (restarted lazily on
    next synthesis).
    """
    if name not in MODEL_ORDER:
        return {"model": name, "error": f"Unknown engine: {name}"}

    url = _REMOTE_ENGINES.get(name)
    if url is None:
        # ── Local (bare-metal) — in-process unload ──
        st = _state[name]
        with st["lock"]:
            if st["instance"] is not None:
                from tts_lab_utils import _safe_del
                slog("VRAM", name, "Evicting …")
                _safe_del(st["instance"])
                st["instance"] = None
                st["status"] = "unloaded"
                st["loaded_model"] = None
                st["loaded_voice"] = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"model": name, "evicted": True, "mode": "local"}

    import httpx
    stripped = url.rstrip("/")
    if stripped in _SGLANG_URLS:
        base = stripped.split("/v1/")[0]
        if _HAS_DOCKER_SOCK and _container_stop(_S2PRO_CONTAINER_NAME, label="S2-Pro"):
            return {"model": name, "evicted": True, "mode": "remote-sglang",
                    "container": base}
        return {"model": name, "evicted": False, "mode": "remote-sglang",
                "container": base,
                "note": "SGLang server holds its model always-resident — no "
                        "/evict endpoint (and no Docker socket to stop it)"}
    # vLLM-backed engines: an in-process /evict cannot free vLLM's memory
    # arena — restart the container (weights re-load lazily on next synth).
    if name in _VLLM_CONTAINERS:
        cname = _VLLM_CONTAINERS[name]
        if not _HAS_DOCKER_SOCK:
            return {"model": name, "evicted": False, "mode": "remote-vllm",
                    "container": cname,
                    "note": "vLLM memory can't be freed in-process — no Docker "
                            "socket to restart the container"}
        if not _container_running(cname):
            return {"model": name, "evicted": True, "mode": "remote-vllm",
                    "container": cname,
                    "note": "container not running — nothing resident"}
        ok = _container_restart(cname, label=name)
        return {"model": name, "evicted": ok, "mode": "remote-vllm",
                "container": cname}
    try:
        r = httpx.post(f"{stripped}/evict", timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            return {"model": name, "evicted": bool(data.get("evicted")),
                    "container": stripped,
                    "freed_mb": data.get("freed_mb", 0),
                    "held_mb": data.get("held_mb", 0)}
        return {"model": name, "evicted": False, "container": stripped,
                "error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"model": name, "evicted": False, "container": stripped,
                "error": str(e)}


def _do_synth_llm(name: str, text: str, params: dict) -> dict:
    """LLM text-generation dispatch with global TTS eviction.

    1. Evict ALL TTS engines from VRAM across all containers
    2. Route to llama.cpp OpenAI-compatible API
    3. Return text response (NOT audio — LLMs generate text)
    """
    import httpx
    import json as _json

    # ── Phase 0: Ensure LLM container is running ──────────────────
    if _HAS_DOCKER_SOCK:
        _start_llm_container()

    # ── Phase 1: Global eviction ──────────────────────────────────
    slog("LLM", name, "Evicting all TTS engines before LLM inference ...")
    evict_results = _evict_all_tts_engines()
    evicted_count = sum(1 for v in evict_results.values() if v.get("evicted"))
    errors = {k: v for k, v in evict_results.items() if "error" in v}
    slog("LLM", name,
         f"Eviction complete — {evicted_count} evicted, "
         f"{len(evict_results) - evicted_count - len(errors)} already idle, "
         f"{len(errors)} errors")
    if errors:
        slog("LLM", name, f"Eviction errors (non-fatal): {errors}")

    # ── Phase 2: Route to LLM ────────────────────────────────────
    llm_url = _REMOTE_ENGINES.get(name, f"http://llm-qwen36:8006")
    payload = {
        "messages": [
            {"role": "system", "content": params.get("system_prompt",
                "You are a helpful AI assistant specialized in reasoning and programming.")},
            {"role": "user", "content": text},
        ],
        "temperature": float(params.get("temperature", 0.7)),
        "max_tokens": int(params.get("max_tokens", 2048)),
        "top_p": float(params.get("top_p", 0.9)),
        "presence_penalty": float(params.get("presence_penalty", 0.0)),
        "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
    }
    seed = int(params.get("seed", -1))
    if seed >= 0:
        payload["seed"] = seed

    t0 = time.perf_counter()
    r = httpx.post(
        f"{llm_url}/v1/chat/completions",
        json=payload,
        timeout=600.0,
    )
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0

    choice = data.get("choices", [{}])[0]
    response_text = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    tok_per_sec = round(total_tokens / elapsed, 1) if elapsed > 0 else 0

    slog("LLM", name,
         f"✅ {total_tokens} tokens in {elapsed:.1f}s "
         f"({tok_per_sec} tok/s)  "
         f"response={response_text[:80]!r}{'…' if len(response_text) > 80 else ''}")

    reasoning = choice.get("message", {}).get("reasoning_content", "")
    return {
        "text": response_text,
        "reasoning": reasoning,
        "tokens": total_tokens,
        "tokens_per_sec": tok_per_sec,
        "model": data.get("model", "qwen3.6"),
        "finish_reason": choice.get("finish_reason", ""),
        "synth_time_ms": int(elapsed * 1000),
    }
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
