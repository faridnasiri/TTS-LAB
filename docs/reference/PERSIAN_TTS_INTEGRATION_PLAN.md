# Persian TTS Models — Container Integration Plan

> **Date:** 2026-06-23
> **Depends on:** [PERSIAN_TTS_HUGGINGFACE_CATALOG.md](PERSIAN_TTS_HUGGINGFACE_CATALOG.md) — full model catalog
> **Status:** Plan — ready for implementation

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Container Architecture — No New Containers Needed](#2-container-architecture--no-new-containers-needed)
3. [Phase 1: Trivial Integrations (3 engines, ~1 hr)](#3-phase-1-trivial-integrations-3-engines-1-hr)
4. [Phase 2: VITS Community Voices (5 engines, ~2 hrs)](#4-phase-2-vits-community-voices-5-engines-2-hrs)
5. [Phase 3: Architecture Gems (2 engines, ~3 hrs)](#5-phase-3-architecture-gems-2-engines-3-hrs)
6. [Files Changed — Complete Inventory](#6-files-changed--complete-inventory)
7. [Environment Variables](#7-environment-variables)
8. [Rollback Plan](#8-rollback-plan)

---

## 1. Strategy Overview

**Decision: All 10 models go into `engine-current`. Zero new containers.**

Rationale:
- 8 of 10 models use Coqui TTS or transformers — both already in engine-current
- MOSS-TTS-Nano's pip package needs evaluation; if it conflicts, it goes to a new container in Phase 3
- VRAM is not a concern — single-engine-at-a-time loading with eviction
- Adding a new container adds ~200 MB base overhead + build time + CI path

**Fallback (if MOSS-TTS-Nano conflicts):**
- Create `Dockerfile.engine-persian` → `engine-persian` container, port 8105
- Only if `moss-tts-nano` pip install breaks existing engine-current deps

---

## 2. Container Architecture — No New Containers Needed

```
                         TTS-LAB Orchestrator (port 8001)
                                  │
    ┌─────────────┬───────────────┼───────────────┬──────────────┐
    │             │               │               │              │
    ▼             ▼               ▼               ▼              ▼
engine-current  engine-mid   engine-qwen    engine-legacy   (orpheus/sglang)
  port 8101     port 8103     port 8104      port 8102

  ┌──────────────────────────────────────────────────────────┐
  │ engine-current: 21 existing + 10 new Persian = 31 engines │
  │                                                          │
  │ [+10] mmsfas, mosstts, kamtera_f, kamtera_m,            │
  │        saillab_az, saillab_cv, gptinf_fa,               │
  │        speecht5_fa, xtts_fa (replaces base xtts fa),    │
  │        piper_gyro (voice preset on existing piper)       │
  │                                                          │
  │ VRAM: ~200-500 MB per Persian model (one at a time)      │
  │ Stack: torch nightly + tf 5.12.1 + CUDA 12.8             │
  └──────────────────────────────────────────────────────────┘
```

**Key insight:** engine-current already has `coqui-tts`, `transformers`, `piper-tts`, and `phonemizer` installed. No new system dependencies needed for any of the 10 models.

---

## 3. Phase 1: Trivial Integrations (3 engines, ~1 hr)

### 3.1 facebook/mms-tts-fas — Engine Key: `mmsfas`

**Files to change:**

**`tts_lab_config.py`** — Add to MODEL_INFO:
```python
"mmsfas": {"label":"MMS Persian (Meta)","size":"~150 MB","rtf_est":"RTF ~0.5×",
           "ram_est_mb":200,"heavy":False,
           "notes":"Meta MMS-TTS Persian VITS. Reference baseline. Non-deterministic.","arthur_fit":5},
```

Add to MODEL_ORDER after `manatts`:
```python
"mmsfas",
```

**`tts_lab_engines.py`** — Add load/synth pair:
```python
# ── MMS Persian TTS (Meta) ──────────────────────────────────────────────────────
def _load_mmsfas():
    from transformers import VitsModel, AutoTokenizer
    model = VitsModel.from_pretrained("facebook/mms-tts-fas")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
    return (model, tokenizer)

def _synth_mmsfas(inst, text, params):
    import torch
    model, tokenizer = inst
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    arr = output.numpy().squeeze()
    return _to_wav(arr, model.config.sampling_rate), model.config.sampling_rate
```

Add to LOADERS/SYNTHERS dicts:
```python
"mmsfas": _load_mmsfas,      # in LOADERS
"mmsfas": _synth_mmsfas,      # in SYNTHERS
```

**`docker-compose.yml`** — Add to orchestrator env:
```yaml
MMSFAS_URL: http://engine-current:8101
```

**`tts_lab_dispatch.py`** — No changes needed. Auto-detected from MODEL_ORDER + env var.

**`Dockerfile.engine-current`** — No changes needed. transformers already installed.

---

### 3.2 Hamid20/speecht5_tts_persian — Engine Key: `speecht5_fa`

**Files to change:**

**`tts_lab_config.py`**:
```python
"speecht5_fa": {"label":"SpeechT5 Persian","size":"~400 MB","rtf_est":"RTF ~1.0×",
                "ram_est_mb":500,"heavy":False,
                "notes":"Fine-tuned Microsoft SpeechT5 for Persian. Encoder-decoder transformer.","arthur_fit":4},
```

**`tts_lab_engines.py`**:
```python
# ── SpeechT5 Persian ────────────────────────────────────────────────────────────
def _load_speecht5_fa():
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
    import torch
    model = SpeechT5ForTextToSpeech.from_pretrained("Hamid20/speecht5_tts_persian")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return (model, processor, vocoder)

def _synth_speecht5_fa(inst, text, params):
    import torch
    model, processor, vocoder = inst
    inputs = processor(text=text, return_tensors="pt")
    # Default speaker embedding — use first from model's embeddings
    speaker_embeddings = model.speecht5.encoder.speaker_embedding.weight[:1]
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    arr = speech.numpy().squeeze()
    return _to_wav(arr, 16000), 16000
```

**`docker-compose.yml`**:
```yaml
SPEECHT5_FA_URL: http://engine-current:8101
```

**`Dockerfile.engine-current`** — No changes needed. transforms has SpeechT5.

---

### 3.3 rhasspy/piper-voices fa_IR/gyro — Voice preset on existing `piper`

**Files to change:**

**`tts_lab_config.py`** — Add voice to PIPER_VOICES:
```python
PIPER_VOICES = [
    # ... existing ...
    ("fa_IR-gyro-medium", "Gyro — Persian Male, Medium quality, ONNX"),
]
```

**`Makefile`** — Add to model download section (download ONNX files to /opt/models/):
```makefile
# In the model download target:
	curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/fa_IR/gyro-medium/fa_IR-gyro-medium.onnx \
	     -o /opt/models/fa_IR-gyro-medium.onnx
	curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/fa_IR/gyro-medium/fa_IR-gyro-medium.onnx.json \
	     -o /opt/models/fa_IR-gyro-medium.onnx.json
```

**`tts_lab_engines.py`** — No changes. Piper loader auto-discovers ONNX files by voice name.

**`Dockerfile.engine-current`** — No changes. piper-tts already installed.

**`docker-compose.yml`** — No changes (uses existing PIPER_URL).

---

## 4. Phase 2: VITS Community Voices (5 engines, ~2 hrs)

All five use the same Coqui TTS `Synthesizer` API. Create a shared helper to avoid duplication.

### 4.1 Shared VITS Helper

**`tts_lab_config.py`** — Add model paths:
```python
# Persian VITS community models
PERSIAN_VITS_MODELS = {
    "kamtera_f": {
        "repo": "Kamtera/persian-tts-female-vits",
        "model_file": "best_model.pth",    # verify exact filename
        "config_file": "config.json",
    },
    "kamtera_m": {
        "repo": "Kamtera/persian-tts-male-vits",
        "model_file": "best_model.pth",
        "config_file": "config.json",
    },
    "saillab_az": {
        "repo": "saillab/persian-tts-azure-grapheme-60K",
        "model_file": "best_model.pth",
        "config_file": "config.json",
    },
    "saillab_cv": {
        "repo": "saillab/persian-tts-cv15-reduct-grapheme-multispeaker",
        "model_file": "best_model.pth",
        "config_file": "config.json",
        "speakers_file": "speakers.pth",
    },
    "gptinf_fa": {
        "repo": "karim23657/persian-tts-female-GPTInformal-Persian-vits",
        "model_file": "best_model.pth",
        "config_file": "config.json",
    },
}
```

### 4.2 Engine Load/Synth (pattern for all 5)

**`tts_lab_engines.py`** — One shared loader:
```python
# ── Persian VITS Community Models ───────────────────────────────────────────────
def _load_persian_vits(engine_key):
    """Shared loader for Kamtera + Saillab Persian VITS models."""
    from TTS.utils.synthesizer import Synthesizer
    from huggingface_hub import snapshot_download
    info = PERSIAN_VITS_MODELS[engine_key]
    model_dir = snapshot_download(info["repo"], ignore_patterns=["*.md", "*.txt"])
    model_path = str(Path(model_dir) / info["model_file"])
    config_path = str(Path(model_dir) / info["config_file"])
    synthesizer = Synthesizer(
        model_path=model_path,
        config_path=config_path,
        use_cuda=True,
    )
    if "speakers_file" in info:
        speakers_path = Path(model_dir) / info["speakers_file"]
        if speakers_path.exists():
            synthesizer.tts_config.speakers_file = str(speakers_path)
    return synthesizer

def _synth_persian_vits(inst, text, params):
    """Shared synth for Kamtera + Saillab Persian VITS models."""
    speaker = params.get("speaker", None)
    wavs = inst.tts(text, speaker_name=speaker) if speaker else inst.tts(text)
    arr = np.array(wavs)
    if arr.ndim > 1:
        arr = arr.squeeze()
    sr = inst.output_sample_rate
    return _to_wav(arr, sr), sr

# Per-engine loaders (thin wrappers):
def _load_kamtera_f():   return _load_persian_vits("kamtera_f")
def _load_kamtera_m():   return _load_persian_vits("kamtera_m")
def _load_saillab_az():  return _load_persian_vits("saillab_az")
def _load_saillab_cv():  return _load_persian_vits("saillab_cv")
def _load_gptinf_fa():   return _load_persian_vits("gptinf_fa")
```

### 4.3 MODEL_INFO entries

```python
"kamtera_f":  {"label":"Kamtera Persian ♀","size":"~200 MB","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Community VITS female Persian voice. Most popular (104 dl/mo).","arthur_fit":5},
"kamtera_m":  {"label":"Kamtera Persian ♂","size":"~200 MB","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Community VITS male Persian voice.","arthur_fit":5},
"saillab_az": {"label":"Saillab Azure FA","size":"~200 MB","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"VITS trained on Azure TTS Persian data. Grapheme-based.","arthur_fit":5},
"saillab_cv": {"label":"Saillab CV Multi-FA","size":"~250 MB","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Multi-speaker VITS (Common Voice 15). Select speaker ID.","arthur_fit":4},
"gptinf_fa":  {"label":"GPTInformal Persian","size":"~200 MB","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Trained on informal/colloquial Persian. Handles slang.","arthur_fit":3},
```

### 4.4 docker-compose.yml additions
```yaml
KAMTERA_F_URL: http://engine-current:8101
KAMTERA_M_URL: http://engine-current:8101
SAILLAB_AZ_URL: http://engine-current:8101
SAILLAB_CV_URL: http://engine-current:8101
GPTINF_FA_URL: http://engine-current:8101
```

### 4.5 Dockerfile.engine-current — No changes needed
Coqui TTS is already installed at line 43.

---

## 5. Phase 3: Architecture Gems (2 engines, ~3 hrs)

### 5.1 saillab/xtts_v2_fa — Persian XTTS Fine-tune

**Decision: Augment existing `xtts` engine rather than add new engine key.**

When `language="fa"` is passed, the loader uses the Persian fine-tune checkpoint. When `language` is anything else, it uses the base XTTS v2 model. This avoids engine key proliferation.

**`tts_lab_config.py`** — Add path:
```python
XTTS_FA_CHECKPOINT = "/opt/models/xtts_v2_fa/best_model_110880.pth"
XTTS_FA_CONFIG = "/opt/models/xtts_v2_fa/config.json"
```

**`tts_lab_engines.py`** — Modify `_load_xtts()`:
```python
def _load_xtts(language=None):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    # Use Persian fine-tune when language is "fa"
    if language == "fa" and Path(XTTS_FA_CHECKPOINT).exists():
        config = XttsConfig()
        config.load_json(XTTS_FA_CONFIG)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=XTTS_FA_CHECKPOINT)
        model.cuda()
        return (model, config, "fa")
    else:
        # Existing base XTTS logic
        ...
        return (model, config, language or "en")
```

**`tts_lab_dispatch.py`** — Pass language param when calling loader:
```python
# In _do_synth() or _ensure_loaded():
language = params.get("language", "en")
inst = _ensure_loaded(name, language=language)
```

**`Dockerfile.engine-current`** — No changes. coqui-tts already installed.

**Model download** — Add to Makefile model download target:
```makefile
	git clone https://huggingface.co/saillab/xtts_v2_fa /opt/models/xtts_v2_fa
```

---

### 5.2 MOSS-TTS-Nano-100M — Engine Key: `mosstts`

**⚠️ Risk: pip package may conflict with engine-current deps. Test first.**

**Step 1 — Evaluate compatibility (5 min):**
```bash
# In engine-current container or venv with matching deps:
pip install moss-tts-nano 2>&1 | tee /tmp/mosstts-install.log
# Check for torch/transformers downgrade warnings
```

**Step 2a — If compatible (engine-current):**

**`Dockerfile.engine-current`** — Add after existing pip installs:
```dockerfile
# MOSS-TTS-Nano — multilingual LLM-based TTS (April 2026)
RUN pip install --no-cache-dir moss-tts-nano
```

**`tts_lab_config.py`**:
```python
"mosstts": {"label":"MOSS-TTS Nano","size":"~200 MB","rtf_est":"RTF ~1.0×",
            "ram_est_mb":300,"heavy":False,
            "notes":"LLM-based (0.1B). 20 languages. Streaming voice clone. Apache 2.0.","arthur_fit":4},
```

**`tts_lab_engines.py`**:
```python
# ── MOSS-TTS-Nano ───────────────────────────────────────────────────────────────
def _load_mosstts():
    from moss_tts_nano import MOSSTTSNano
    model = MOSSTTSNano.from_pretrained("FenomAI/MOSS-TTS-Nano-100M")
    return model

def _synth_mosstts(inst, text, params):
    ref_audio = params.get("reference_audio")
    output = inst.generate(text=text, prompt_speech=ref_audio)
    # output is (sample_rate, audio_array) or similar — verify exact API
    sr, arr = output if isinstance(output, tuple) else (24000, output)
    return _to_wav(arr.squeeze(), sr), sr
```

**Step 2b — If NOT compatible (dedicated container):**

Create **`docker/Dockerfile.engine-persian`**:
```dockerfile
FROM tts-lab-stack-current:latest
LABEL org.opencontainers.image.title="TTS Lab — Engine Server (persian)"
LABEL org.opencontainers.image.description="Persian TTS models — MOSS-TTS-Nano + Coqui VITS"
LABEL tts-lab.tier="3-engine"
LABEL tts-lab.stack="persian"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir moss-tts-nano coqui-tts phonemizer scipy soundfile

COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_engine_server.py /opt/arthur/
COPY tts_lab_engines.py /opt/arthur/
COPY tts_lab_dispatch.py /opt/arthur/

EXPOSE 8105
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8105/health || exit 1
CMD ["python3", "/opt/arthur/tts_lab_engine_server.py", "--port", "8105", "--stack", "persian"]
```

**`docker-compose.yml`** — Add service:
```yaml
engine-persian:
  build:
    context: .
    dockerfile: docker/Dockerfile.engine-persian
  image: tts-lab-engine-persian:latest
  container_name: tts-lab-engine-persian
  volumes:
    - /opt/models:/opt/models
    - /tmp/tts_uploads:/tmp/tts_uploads
    - /opt/arthur/reference_voices:/opt/arthur/reference_voices
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    <<: *common-env
  restart: unless-stopped
  profiles:
    - persian
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8105/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 120s
  networks:
    - tts-lab-net
```

And route MOSS-TTS-Nano + any isolated Persian VITS engines to it:
```yaml
# In orchestrator:
MOSSTTS_URL: http://engine-persian:8105
```

---

## 6. Files Changed — Complete Inventory

### Always changed (all phases):

| File | Change | Risk |
|------|--------|------|
| `tts_lab_config.py` | +10 MODEL_INFO entries, +10 MODEL_ORDER entries, +PERSIAN_VITS_MODELS dict | Low |
| `tts_lab_engines.py` | +10 load/synth pairs (~200 lines), +LOADERS/SYNTHERS entries | Low |
| `docker-compose.yml` | +10 env vars in orchestrator service | Low |
| `docs/engine_compatibility.yaml` | +10 engine entries | Low |
| `Makefile` | +model download targets for XTTS FA, Piper Gyro | Low |

### Conditionally changed (Phase 3 only):

| File | Change | Risk |
|------|--------|------|
| `Dockerfile.engine-current` | +`pip install moss-tts-nano` (if compatible) | Medium |
| `docker/Dockerfile.engine-persian` | NEW FILE (if moss-tts-nano conflicts) | Medium |
| `docker-compose.yml` | +engine-persian service (if conflicts) | Medium |

### NOT changed:

| File | Reason |
|------|--------|
| `Dockerfile.stack.current` | No new system deps needed |
| `Dockerfile.base` | No new system deps needed |
| `Dockerfile.orchestrator` | Orchestrator is pure HTTP dispatch — no ML deps |
| `tts_lab_shims.py` | No new patches needed for these models |
| `tts_lab_ui.py` | UI auto-renders from MODEL_ORDER |
| `tts_lab_utils.py` | No new utility functions needed |

---

## 7. Environment Variables

### New env vars in orchestrator (docker-compose.yml):

```yaml
# Phase 1
MMSFAS_URL: http://engine-current:8101
SPEECHT5_FA_URL: http://engine-current:8101
# Piper Gyro uses existing PIPER_URL — no new env var

# Phase 2
KAMTERA_F_URL: http://engine-current:8101
KAMTERA_M_URL: http://engine-current:8101
SAILLAB_AZ_URL: http://engine-current:8101
SAILLAB_CV_URL: http://engine-current:8101
GPTINF_FA_URL: http://engine-current:8101

# Phase 3
MOSSTTS_URL: http://engine-current:8101  # or engine-persian:8105
# XTTS FA uses existing XTTS_URL — no new env var
```

---

## 8. Rollback Plan

Each phase is independently reversible:

```bash
# Phase 1 rollback:
git revert <phase-1-commit>
docker compose up -d --force-recreate engine-current orchestrator

# Phase 2 rollback:
git revert <phase-2-commit>
docker compose up -d --force-recreate engine-current orchestrator

# Phase 3 rollback:
git revert <phase-3-commit>
# If dedicated container was created:
docker compose --profile persian down
docker rmi tts-lab-engine-persian:latest
```

The engine-current container rebuilds in ~5 minutes (layers are cached). No data loss — models are on host volumes.
