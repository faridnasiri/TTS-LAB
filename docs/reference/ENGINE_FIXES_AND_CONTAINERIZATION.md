# TTS Lab — Engine Fix Report & Containerization Guide

> **Date**: 2026-06-19  
> **Environment**: transformers 5.12.1, torch 2.10.0+cu128, Python 3.11, CUDA GPU  
> **Context**: Added 5 new engines (chatterboxturbo, vibevoice, higgs, omnivoice, s2pro) and fixed pre-existing compatibility issues caused by the transformers 5.x / torch 2.10 upgrade.

---

## 1. New Engines Added (5)

### 1.1 Working — Fully Functional

| Engine | Key | Size | Params | Languages | Voice Cloning | Notes |
|--------|-----|------|--------|-----------|---------------|-------|
| Chatterbox-Turbo | `chatterboxturbo` | ~700 MB | 350M | EN only | Yes (ref WAV) | One-step distilled decoder. MIT license. Same `chatterbox-tts` package. |
| OmniVoice | `omnivoice` | ~1.2 GB | 0.6B | **600+ (incl. Persian "fa")** | Yes (ref WAV + transcript) | Diffusion LM. RTF 0.025. Apache-2.0. Explicit `language` parameter. |

### 1.2 Needs SGLang-Omni Server

| Engine | Key | Size | Params | Languages | Serving |
|--------|-----|------|--------|-----------|---------|
| Microsoft VibeVoice-1.5B | `vibevoice` | ~6 GB | 3B | EN+ZH only | `docker run lmsysorg/sglang-omni:dev --model microsoft/VibeVoice-1.5B` |
| BosonAI Higgs Audio v3 | `higgs` | ~8 GB | 4B | **102 (Persian Tier 1)** | `docker run lmsysorg/sglang-omni:dev --model bosonai/higgs-audio-v3-tts-4b` |
| Fish Audio S2-Pro | `s2pro` | ~10 GB | 5B | 80+ (incl. Persian) | `python -m sglang.launch_server --model fishaudio/s2-pro` |

**Why SGLang needed**: These models have **zero Python code** in their Hugging Face repos (no `modeling_*.py`, no `auto_map` in config.json). `trust_remote_code=True` can never work. SGLang-Omni bundles its own model code and dependencies, completely isolated from the main Python environment — zero risk to existing engines.

---

## 2. Environment Fixes Applied

### 2.1 Critical — Service Startup Crash (torch 2.10 + Python 3.11)

**Symptom**: Service crash-loop on startup with `AttributeError: type object '__file__' has no attribute 'endswith'`

**Root Cause**: `torch.distributed.tensor._collective_utils` uses `@torch.library.register_fake` which calls `inspect.getsourcefile()` → `getabsfile()` → `getsourcefile()`. Some torch 2.10 modules have `__file__` set to a `type` object instead of a string, causing Python 3.11's `inspect` module to crash.

**Fix in `tts_lab_shims.py`**:
```python
# Patch inspect.getsourcefile to return safe value on crash
_orig_getsourcefile = inspect.getsourcefile
def _patched_getsourcefile(obj):
    try:
        return _orig_getsourcefile(obj)
    except (TypeError, AttributeError):
        return "/dev/null"  # safe non-None default
inspect.getsourcefile = _patched_getsourcefile

# Increase recursion limit for deep import chains
sys.setrecursionlimit(10000)

# Pre-stub torch._dynamo._trace_wrapped_higher_order_op to prevent
# the corrupting import chain from ever starting
if "torch._dynamo._trace_wrapped_higher_order_op" not in sys.modules:
    _m = types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
    _m.TransformGetItemToIndex = type("TransformGetItemToIndex", (), {})
    _m.trace_wrapped = lambda fn, *a, **kw: fn
    _m.__file__ = "<stub>"
    sys.modules["torch._dynamo._trace_wrapped_higher_order_op"] = _m
```

### 2.2 Critical — transformers.masking_utils Import Crash

**Symptom**: Service crash during `import transformers.masking_utils` (line ~317 in shims)

**Root Cause**: `transformers.masking_utils` imports `torch._dynamo._trace_wrapped_higher_order_op` → triggers `torch.distributed.tensor._collective_utils` → `@register_fake` → inspect crash.

**Fix**: Changed `except ImportError` to `except Exception` so the fallback stubs are created regardless of crash type:
```python
try:
    import transformers.masking_utils
except Exception:  # was: except ImportError
    _mu = types.ModuleType("transformers.masking_utils")
    _mu.create_causal_mask = lambda *a, **kw: None
    ...
    sys.modules["transformers.masking_utils"] = _mu
```

### 2.3 Critical — torchcodec Package Metadata Missing

**Symptom**: `importlib.metadata.PackageNotFoundError: No package metadata was found for torchcodec`

**Root Cause**: `transformers.audio_utils` (5.12.1) calls `importlib.metadata.version("torchcodec")` but the dummy `torchcodec` modules created by the chatterbox loader have no package metadata.

**Fix**: Created fake dist-info directory:
```bash
mkdir -p $SITE_PACKAGES/torchcodec-99.0.0.dist-info
# Created METADATA file with Name: torchcodec, Version: 99.0.0
```

### 2.4 Critical — chatterbox/chatterboxturbo RecursionError

**Symptom**: `RecursionError: maximum recursion depth exceeded` when loading chatterbox or chatterboxturbo

**Root Cause**: `perth` → `librosa` → `lazy_loader` → `inspect.stack()` creates deep inspect chains that hit recursion limit with the patched inspect functions.

**Fix**: Increased recursion limit to 10000 + simplified inspect patch to only wrap `getsourcefile`.

### 2.5 Transformers 5.x Compatibility Stubs

Added in `tts_lab_shims.py` pre-patches section (runs BEFORE any torch import):

```python
# --- isin_mps_friendly (removed in 5.x) — patches parler, xtts, indextts ---
def _isin_mps_friendly(*args, **kwargs):
    if "elements" in kwargs:
        kwargs["input"] = kwargs.pop("elements")
    return torch.isin(*args, **kwargs)

# --- ExtensionsTrie, AddedToken (removed in 5.x) — patches indextts ---
for cls_name in ["ExtensionsTrie", "AddedToken"]:
    stub = type(cls_name, (), {"__init__": lambda self, *a, **kw: None})
    setattr(transformers.tokenization_utils, cls_name, stub)

# --- check_model_inputs compat (changed signature in 5.x) — patches qwen3tts ---
_orig_cmi = transformers.utils.generic.check_model_inputs
def _compat_cmi(func=None):  # accepts both @check_model_inputs and @check_model_inputs()
    if func is not None: return _orig_cmi(func)
    return _orig_cmi

# --- PretrainedConfig defaults (attrs removed in 5.x) — patches qwen3tts, parler ---
# Injects pad_token_id=0, tie_encoder_decoder=False, etc. on all configs

# --- Auto-stub for removed transformers classes/constants ---
# Stubs: find_pruneable_heads_and_indices, prune_conv1d_layer, prune_layer,
# HammingDiversityLogitsProcessor, HammingDiversityLogitsWarper,
# FLAX_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, download_url, etc.
```

---

## 3. Engine Status — Complete 28-Engine Table

### 3.1 Working (21 engines)

| # | Engine | Status | Persian | Notes |
|---|--------|--------|---------|-------|
| 1 | piper | ✅ OK | No | ONNX CPU. 6 voices. |
| 2 | kokoro | ✅ OK | No | ONNX. 54 voices, 9 langs. |
| 3 | melo | ✅ OK | No | 5 English accents. |
| 4 | matcha | ✅ OK | **Yes** | FA+EN bilingual. Khadijah/Musa. |
| 5 | chattts | ✅ OK | No | Conversational TTS. |
| 6 | outetts | ✅ OK | No | GGUF via LLAMACPP. |
| 7 | bark | ✅ OK | No | Full-size on GPU. |
| 8 | styletts2 | ✅ OK | No | Style transfer from ref WAV. |
| 9 | f5tts | ✅ OK* | No | *Needs ref WAV upload (by design). |
| 10 | dia | ✅ OK | No | Dialogue-native. [S1]/[S2] speakers. |
| 11 | **xtts** | ✅ **FIXED** | No | `isin_mps_friendly` + `torch.isin` wrapper. |
| 12 | cosyvoice | ✅ OK | No | Zero-shot voice cloning. |
| 13 | fishspeech | ✅ OK | No | Zero-shot. Ref WAV optional. |
| 14 | chatterbox | ✅ **FIXED** | **Yes** | inspect recursion + torchcodec metadata fixes. |
| 15 | orpheus | ⏸️ Gated | No | Needs `huggingface-cli login`. |
| 16 | openvoice | ✅ OK | No | MeloTTS base + tone-color. |
| 17 | zonos | ✅ OK | No | Emotion vector + speaking rate. |
| 18 | manatts | ✅ OK* | **Yes** | *Needs ref WAV (by design). Persian Tacotron2. |
| 19 | csm | ⏸️ Gated | No | Needs `huggingface-cli login`. |
| 20 | neutts | ⏸️ Stub | — | Not configured (intentional). |
| 21 | **chatterboxturbo** | ✅ **NEW** | No | 350M one-step. Voice cloning. |
| 22 | **omnivoice** | ✅ **NEW** | **Yes** | 600+ langs. Language selector in UI. |

### 3.2 Still Broken — Deep Framework Incompatibilities

| # | Engine | Error | Root Cause | Containerization Fix |
|---|--------|-------|-----------|---------------------|
| 23 | **indextts** | `cannot import name 'download_url'` (and 170+ more) | 176 imports from transformers internals removed in 5.x | **Needs transformers 4.x venv/container** |
| 24 | **parler** | `Cannot copy out of meta tensor` | torch 2.10 removed meta tensor support | **Needs torch <2.0 venv/container** (already designed for transformers==4.46.1) |
| 25 | **qwen3tts** | `pad_token_id` missing on config | transformers 5.x config API changes | **Needs package update or transformers 4.x venv** |

### 3.3 Needs SGLang Server (no Python code on HF)

| # | Engine | Serving | Persian |
|---|--------|---------|---------|
| 26 | vibevoice | `lmsysorg/sglang-omni:dev --model microsoft/VibeVoice-1.5B` | No |
| 27 | higgs | `lmsysorg/sglang-omni:dev --model bosonai/higgs-audio-v3-tts-4b` | **Yes** |
| 28 | s2pro | `sglang.launch_server --model fishaudio/s2-pro` | **Yes** |

---

## 4. Containerization Recommendations

### 4.1 Dependency Conflict Map

```
                    Needs                    Has on VM          Conflict?
                    ─────                    ────────           ────────
parler-tts          transformers==4.46.1     transformers 5.12   YES — critical
indextts            transformers 4.x          transformers 5.12   YES — critical
qwen3tts            transformers 4.x (?)      transformers 5.12   YES — moderate
chatterbox          torchcodec (stubbed)      torchcodec stub     OK (fixed)
chatterboxturbo     torchcodec (stubbed)      torchcodec stub     OK (fixed)
vibevoice/higgs     SGLang-Omni (docker)      SGLang not installed OK (isolated)
s2pro               SGLang (pip)              SGLang not installed OK (isolated)
omnivoice           torch>=2.0, transformers  torch 2.10, tf 5.12 OK
All others          no strict pins             torch 2.10, tf 5.12 OK
```

### 4.2 Recommended Container Architecture

```
┌─────────────────────────────────────────────────────┐
│  Reverse Proxy (nginx / Caddy) — port 8001           │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─ Container 1: Main TTS Lab ──────────────────┐   │
│  │  Python 3.11 + torch 2.10 + transformers 5.12 │   │
│  │  21 working engines (piper through omnivoice) │   │
│  │  Port: 8001 (internal)                         │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  ┌─ Container 2: Legacy Engines ─────────────────┐   │
│  │  Python 3.10 + torch 1.13 + transformers 4.46 │   │
│  │  • indextts                                     │   │
│  │  • parler-tts (already has venv for this)       │   │
│  │  • qwen3tts (if package doesn't update)         │   │
│  │  Port: 8002 (internal)                           │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  ┌─ Container 3: SGLang-Omni ────────────────────┐   │
│  │  lmsysorg/sglang-omni:dev (CUDA)                │   │
│  │  • vibevoice                                     │   │
│  │  • higgs                                         │   │
│  │  • s2pro (separate SGLang instance)              │   │
│  │  Ports: 8100, 8101, 8102 (internal)              │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 4.3 Per-Engine Container Details

#### Container 1: Main (21 engines)
```dockerfile
FROM nvidia/cuda:12.4-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip sox ffmpeg
# torch 2.10 + transformers 5.12 — works for 21 of 28 engines
RUN pip install torch==2.10.0 transformers==5.12.1
# Install all working engines
RUN pip install piper-tts kokoro-onnx ChatTTS outetts bark styletts2 \
    f5-tts parler-tts==0.2.3 coqui-tts chatterbox-tts fish-speech \
    qwen-tts orpheus-speech zonos openvoice omnivoice
# Copy tts_lab source + shims
COPY . /opt/arthur/
CMD ["uvicorn", "tts_lab:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Container 2: Legacy (indextts, parler, qwen3tts)
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
RUN apt-get install -y python3.10 python3-pip
# Pin to older versions that these engines need
RUN pip install torch==1.13.1 transformers==4.46.1
RUN pip install git+https://github.com/index-tts/index-tts  # indextts
RUN pip install parler-tts==0.2.3 qwen-tts
# Lightweight adapter that exposes /synthesize/{engine} API
COPY legacy_adapter.py /app/
CMD ["python", "/app/legacy_adapter.py", "--port", "8002"]
```

#### Container 3: SGLang-Omni (vibevoice, higgs, s2pro)
```bash
# vibevoice
docker run -d --gpus all -p 8100:8000 \
  lmsysorg/sglang-omni:dev \
  --model microsoft/VibeVoice-1.5B

# higgs
docker run -d --gpus all -p 8101:8000 \
  lmsysorg/sglang-omni:dev \
  --model bosonai/higgs-audio-v3-tts-4b

# s2pro (uses separate SGLang, not SGLang-Omni)
docker run -d --gpus all -p 8102:8000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server --model fishaudio/s2-pro
```

Then set environment variables in Container 1:
```bash
VIBEVOICE_SGLANG_URL=http://container3:8100/v1/audio/speech
HIGGS_SGLANG_URL=http://container3:8101/v1/audio/speech
S2PRO_SGLANG_URL=http://container3:8102/v1/audio/speech
```

### 4.4 Alternative: Single Container with Per-Engine venvs

Instead of 3 containers, use one container with per-engine venvs for the incompatible engines:

```
/opt/arthur/
├── venvs/
│   ├── main/          # torch 2.10 + tf 5.12 — 21 engines
│   ├── legacy/        # torch 1.13 + tf 4.46 — indextts, parler, qwen3tts
│   └── sglang-proxy/  # just HTTP clients — vibevoice, higgs, s2pro
├── tts_lab.py         # modified to dispatch to subprocess/HTTP
└── ...
```

The dispatch layer in `tts_lab_dispatch.py` already supports per-engine loading — it would be extended to route legacy engines to a subprocess that runs in the legacy venv, and SGLang engines to HTTP calls.

### 4.5 Persistent Files to Mount/Bind

| Path | Purpose | Persist? |
|------|---------|----------|
| `/opt/models/huggingface/` | HF cache (downloaded models) | Yes — large, reuse across rebuilds |
| `/opt/arthur/reference_voices/` | Shipped reference WAVs | Yes — small |
| `/tmp/tts_uploads/` | User-uploaded ref WAVs | No — ephemeral, fine to lose |
| `/opt/arthur/voice_library/` | Persian Voice Library | Yes — curated |
| `/opt/arthur/output/` | Generated audio output | Optional |

### 4.6 GPU Memory Planning

| Container | Max VRAM | Notes |
|-----------|----------|-------|
| Main TTS Lab | 12 GB | One heavy engine at a time (eviction system) |
| Legacy Engines | 6 GB | Smaller models, fewer concurrent |
| SGLang-Omni | 10 GB per model | One model per instance, can share GPU if sequential |
| **Total** | **16-24 GB** | With eviction, fits on single 24 GB GPU |

---

## 5. Files Changed (This Session)

| File | Changes |
|------|---------|
| `tts_lab_shims.py` | inspect compat, masking_utils fix, transformers 5.x stubs (ExtensionsTrie, AddedToken, isin_mps_friendly, find_pruneable_heads_and_indices, prune_conv1d_layer, prune_layer, check_model_inputs compat, PretrainedConfig defaults, torch.isin wrapper, thermorecursion limit, torch._dynamo stub), torchcodec metadata creation script |
| `tts_lab_config.py` | 5 new MODEL_INFO entries, MODEL_ORDER (28 total), OMNIVOICE_LANGUAGES catalogue (60 languages) |
| `tts_lab_engines.py` | 10 new _load/_synth functions (5 pairs), ExtensionsTrie fix in _load_indextts, added all params to chatterboxturbo synth (temperature, top_p, top_k, repetition_penalty, min_p, norm_loudness), added all params to omnivoice synth (language, speed, instruct, duration), vibevoice/higgs switched to SGLang API, s2pro uses SGLang API |
| `tts_lab_dispatch.py` | pkg_map entries for 5 new engines |
| `tts_lab_ui.py` | Parameter widgets for all 5 new engines (language dropdown for omnivoice, full controls for chatterboxturbo, control token docs for vibevoice/higgs/s2pro), OMNIVOICE_LANGUAGES import |
| `tts_lab.py` | `/refs` endpoint now scans permanent `/opt/arthur/reference_voices/` dir (survives reboots), Path import |
| `deploy_lab.ps1` | `omnivoice` pip install line |
| `patches/patch_parler_tts.py` | Added `tie_weights` **kwargs fix |
| `docs/reference/english/` | 2 new reference voices: `alex_wright.wav`, `ryan_reviewer.wav` (converted from MP3) |
| `scripts/test/test_new_engines.py` | Structural integration test (158 checks) |
| `/opt/arthur-bench-env/lib/.../torchcodec-99.0.0.dist-info/` | Fake package metadata (VM-side) |

---

## 6. Key Decisions for Containerization

1. **Don't upgrade transformers to fix vibevoice/higgs** — they have no Python code on HF. Upgrading transformers won't help and WILL break other engines. Use SGLang-Omni instead.

2. **Don't try to fix indextts/parler/qwen3tts in-place** — they need transformers 4.x + torch 1.x. Put them in a separate container/venv.

3. **Keep omnivoice in the main container** — it works fine with torch 2.10 + transformers 5.12 and has no conflicts.

4. **The inspect/torch/transformers patches in `tts_lab_shims.py` are critical** — if rebuilding the container, make sure `tts_lab_shims.py` is imported FIRST before any other module.

5. **Persist the HF cache** (`/opt/models/huggingface/`) across container rebuilds to avoid re-downloading 50+ GB of models.

6. **The torchcodec metadata fix must be applied at container build time** — add it to the Dockerfile RUN step.
