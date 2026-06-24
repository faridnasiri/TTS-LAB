# Persian TTS Models — Container Integration Plan (Revised)

> **Date:** 2026-06-23 (revised after repo verification + architecture review)
> **Depends on:** [PERSIAN_TTS_HUGGINGFACE_CATALOG.md](PERSIAN_TTS_HUGGINGFACE_CATALOG.md) — full model catalog
> **Status:** Plan — ready for Phase 1 implementation
> **Review feedback incorporated:** Reordered phases, Saillab gating discovered, VITS filenames verified, SpeechT5 risk flagged, MOSS deferred, benchmark suite added

---

## Table of Contents

1. [Repo Verification Results](#1-repo-verification-results)
2. [Revised Phase Plan](#2-revised-phase-plan)
3. [Phase 0: Persian Benchmark Suite](#3-phase-0-persian-benchmark-suite)
4. [Phase 1: Zero-Risk Additions](#4-phase-1-zero-risk-additions)
5. [Phase 2: Community VITS Voices](#5-phase-2-community-vits-voices)
6. [Phase 3: XTTS Persian Fine-tune](#6-phase-3-xtts-persian-fine-tune)
7. [Phase 4: SpeechT5 (after verification)](#7-phase-4-speecht5-after-verification)
8. [Phase 5: MOSS-TTS (independent evaluation)](#8-phase-5-moss-tts-independent-evaluation)
9. [Files Changed — Complete Inventory](#9-files-changed--complete-inventory)
10. [Pre-Integration Verification Checklist](#10-pre-integration-verification-checklist)

---

## 1. Repo Verification Results

Before writing code, every repository was checked for actual file structure.

### Verified — Public, Ready to Use

| Repo | Actual Model File | Config File | Size | Notes |
|------|------------------|-------------|------|-------|
| `Kamtera/persian-tts-female-vits` | `best_model_30824.pth` | `config.json` | 998 MB | Training ckpt (incl. optimizer). Use `best_model_30824.pth` (highest step count among bests). |
| `Kamtera/persian-tts-male-vits` | `best_model_98066.pth` | `config.json` | 998 MB | Training ckpt. 7 checkpoints in repo. Use `best_model_98066.pth`. |
| `karim23657/persian-tts-female-GPTInformal-Persian-vits` | `best_model_98066.pth` | `config.json` | 998 MB | Training ckpt. 6 checkpoints. Has `tests/` dir (harmless). |
| `Hamid20/speecht5_tts_persian` | Safetensors | N/A (HF format) | 0.1B params | MIT license. Not gated. Uses standard HF `from_pretrained()`. |
| `facebook/mms-tts-fas` | Safetensors | N/A (HF format) | 36M params | CC-BY-NC 4.0. Standard `VitsModel.from_pretrained()`. |
| `rhasspy/piper-voices` fa_IR/gyro-* | `fa_IR-gyro-medium.onnx` | `.onnx.json` | ~50 MB | Public. Direct URL download. |
| `FenomAI/MOSS-TTS-Nano-100M` | Safetensors | N/A (HF format) | 0.1B params | Apache 2.0. `MOSSTTSNano.from_pretrained()`. |

### Gated — Requires HF Token

| Repo | Status | Active in ZabanZad PoC? |
|------|--------|------------------------|
| `saillab/persian-tts-azure-grapheme-60K` | 401 — gated | ❌ Commented out |
| `saillab/persian-tts-cv15-reduct-grapheme-multispeaker` | 401 — gated | ❌ Commented out |
| `saillab/persian-tts-grapheme-arm24-finetuned-on1` | 401 — gated | ❌ Commented out |
| `saillab/multi_speaker` | 401 — gated | ❌ Commented out |
| `saillab/female_cv_azure_male_azure_female` | 401 — gated | ❌ Commented out |
| `saillab/xtts_v2_fa` | 401 — gated | ✅ Active (XTTS tab) |

### Active Saillab Models (ZabanZad PoC — different repos!)

| Repo | Checkpoint | Config | Speakers |
|------|-----------|--------|----------|
| `saillab/ZabanZad_VITS_MAle` | `checkpoint_61000.pth` | `config.json` | None |
| `saillab/ZabanZad_VITS_Female` | `best_model_15397.pth` | `config.json` | `speakers1.pth` |

> **⚠️ Important:** The Saillab models in the original catalog (`azure-grapheme-60K`, `cv15-reduct`, etc.) are **commented out** in the ZabanZad PoC. The actively maintained models are under different repo names (`ZabanZad_VITS_MAle`, `ZabanZad_VITS_Female`). All Saillab repos require `HUGGING_FACE_HUB_TOKEN`.

### Critical Technical Finding: Checkpoint Format

All Kamtera and karim23657 `.pth` files are **~998 MB training checkpoints** (model + optimizer + scheduler state). A pure VITS model is ~50-80 MB. This matters because:

1. **Download:** 998 MB per model (not 200 MB as estimated). Use `hf_hub_download` for individual files, never `snapshot_download`.
2. **Loading:** Coqui TTS `Synthesizer` handles training checkpoints — it extracts the `model` key from the state dict. The API is:
   ```python
   Synthesizer(
       tts_checkpoint=model_path,    # NOT model_path
       tts_config_path=config_path,   # NOT config_path
       tts_speakers_file=...,         # optional
       use_cuda=True,
   )
   ```
3. **Disk:** 1 GB per model on `/opt/models/`. Budget: ~5 GB for all VITS models.

---

## 2. Revised Phase Plan

```
Phase 0: Persian Benchmark Suite    (30 min, no code changes)
Phase 1: Zero-Risk Additions        (~1 hr, 2 engines)
  ├── Piper Gyro (ONNX file download)
  └── MMS Persian (transformers API)
Phase 2: Community VITS Voices      (~3 hrs, 3-5 engines)
  ├── Kamtera Female
  ├── Kamtera Male
  ├── GPTInformal
  ├── ZabanZad Female (gated — needs HF token)
  └── ZabanZad Male   (gated — needs HF token)
Phase 3: XTTS Persian Fine-tune     (~2 hrs, 1 engine augmentation)
Phase 4: SpeechT5 (blocked on verification)  (~2 hrs, 1 engine)
Phase 5: MOSS-TTS (independent eval)         (timeline TBD)
```

**Rationale for reordering (per review feedback):**
- Piper Gyro is free → deploy first
- MMS is stable/official → deploy next as quality baseline
- Community VITS voices are the bulk of value → Phase 2
- XTTS-FA has cache key risk → separate phase with careful testing
- SpeechT5 speaker embedding is unverified → block until proven
- MOSS-TTS dependency risk is high → independent evaluation, do not delay other phases

---

## 3. Phase 0: Persian Benchmark Suite

**Before any engine integration**, create a standardized evaluation set. This catches quality issues before exposing engines to users and prevents maintaining clearly inferior models.

### 3.1 Benchmark Sentences

```python
# /opt/arthur/persian_benchmark.py
PERSIAN_BENCHMARK = [
    # ── Basic pronunciation ──
    ("basic_hello",       "سلام، حال شما چطور است؟"),
    ("basic_weather",     "امروز هوا در سیاتل بسیار خوب است."),
    ("basic_quality",     "این یک تست کیفیت برای سامانه تبدیل متن به گفتار است."),

    # ── Numbers ──
    ("numbers_phone",     "شماره تماس من ۱۲۳۴۵۶۷۸۹۰ است."),
    ("numbers_price",     "قیمت بیت کوین امروز چقدر است؟"),
    ("numbers_date",      "تاریخ امروز بیست و سوم ژوئن سال دو هزار و بیست و شش است."),

    # ── Punctuation & structure ──
    ("punct_question",    "آیا واقعاً فکر میکنی که این راهحل درستی است؟"),
    ("punct_exclamation", "چه روز قشنگی! واقعاً که هوا عالی است."),
    ("punct_quotes",      "او گفت: «من فردا به تهران میروم.»"),

    # ── Persian-specific characters ──
    ("chars_alef",        "آب آمد و آن آسیاب را باد برد."),
    ("chars_hamza",       "مؤمنان مؤثر در جامعه مؤاخذه نمیشوند."),
    ("chars_yeh",         "خانهای بزرگ در کنار رودخانه."),

    # ── Mixed Persian/Arabic words ──
    ("mixed_arabic",      "قرآن کریم کتاب مقدس مسلمانان است."),
    ("mixed_formal",      "اداره کل امور مالیاتی اعلام کرد که مهلت تسلیم اظهارنامه تمدید شد."),

    # ── Informal / colloquial ──
    ("informal_chat",     "سلام رفیق، چطوری؟ دلم برات تنگ شده بود."),
    ("informal_slang",    "بابا این چه وضعشه؟ آخه کی این کارو کرده؟"),

    # ── Long-form ──
    ("long_paragraph",
        "امروز میخواهم درباره یکی از مهمترین پیشرفتهای علمی قرن اخیر صحبت کنم. "
        "هوش مصنوعی در سالهای اخیر تحول عظیمی در صنعت و فناوری ایجاد کرده است. "
        "از خودروهای خودران گرفته تا دستیارهای پزشکی، هوش مصنوعی همه جا حضور دارد. "
        "اما چالشهای اخلاقی زیادی هم پیش روی ما قرار دارد که باید به آنها فکر کنیم."),
]

# Arthur-specific (for the scam-baiting use case)
ARTHUR_PERSIAN_BENCHMARK = [
    ("arthur_greeting",   "الو؟ ببخشید، کی هستید؟ من تقریباً صدای تلفن را نشنیدم."),
    ("arthur_confused",   "حالا این کاغذ را کجا گذاشتم... اوه، ببخشید عزیزم، باز گیج شدم."),
    ("arthur_numbers",    "میشه اون شماره رو یک بار دیگه برام بگید؟ آروم آروم لطفاً."),
]
```

### 3.2 Evaluation Script

```python
# scripts/benchmark/persian_bench.py
"""Generate benchmark audio from all Persian-capable engines and save to disk."""
import sys, json, time, wave, io
from pathlib import Path
import requests

OUTPUT_DIR = Path("/opt/arthur/benchmarks/persian")
ENGINE_URLS = {}  # populated from env or config

def synth(engine: str, text: str) -> tuple[bytes, float]:
    """Synthesize text and return (wav_bytes, elapsed_seconds)."""
    url = ENGINE_URLS[engine]
    t0 = time.time()
    resp = requests.post(f"{url}/synthesize/{engine}",
                         json={"text": text, "params": {}},
                         timeout=120)
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.content, elapsed

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for engine in ENGINE_URLS:
        engine_results = {}
        for name, text in PERSIAN_BENCHMARK:
            try:
                wav, elapsed = synth(engine, text)
                path = OUTPUT_DIR / f"{engine}_{name}.wav"
                path.write_bytes(wav)
                dur = wav_duration(wav)
                engine_results[name] = {"duration_s": dur, "synth_s": elapsed, "rtf": elapsed / dur if dur > 0 else 0}
            except Exception as e:
                engine_results[name] = {"error": str(e)}
        results[engine] = engine_results
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
```

### 3.3 Evaluation Criteria

For each engine, score 1–5 on:

| Dimension | What to Listen For |
|-----------|-------------------|
| **Pronunciation** | Are Persian phonemes correct? Especially ق, غ, خ, ع |
| **Diacritics** | Does short/long vowel distinction sound right? |
| **Numbers** | Are Persian digits (۱۲۳) read correctly? |
| **Punctuation** | Does it pause at periods, rise at questions? |
| **Naturalness** | Does it sound like a human or robotic? |
| **Speed** | Is RTF acceptable for interactive use? |

**Gate:** Any engine scoring < 2 on pronunciation or naturalness should not be exposed in the UI. Mark as `experimental` with a note.

---

## 4. Phase 1: Zero-Risk Additions

### 4.1 Piper Gyro (FIRST — before everything else)

**Why first:** Zero GPU, zero code risk, existing Piper infrastructure. Deploy in 5 minutes.

**Files to change:**

**`tts_lab_config.py`** — Add to existing `PIPER_VOICES` list:
```python
# In PIPER_VOICES (around line 80):
("fa_IR-gyro-medium", "Gyro — Persian Male (Medium, ONNX, ~50 MB)"),
```

**Model download** (on VM, one-time):
```bash
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa_IR/gyro-medium/fa_IR-gyro-medium.onnx" \
     -o /opt/models/fa_IR-gyro-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa_IR/gyro-medium/fa_IR-gyro-medium.onnx.json" \
     -o /opt/models/fa_IR-gyro-medium.onnx.json
```

**`tts_lab_engines.py`** — No changes. Piper's `_load_piper()` auto-discovers ONNX files by voice name.

**`docker-compose.yml`** — No changes. Uses existing `PIPER_URL`.

**`Dockerfile.engine-current`** — No changes. `piper-tts` already installed.

**Verification:**
```bash
curl -X POST "http://192.168.0.87:8001/synthesize/piper" \
     -H "Content-Type: application/json" \
     -d '{"text": "سلام، حال شما چطور است؟", "params": {"voice": "fa_IR-gyro-medium"}}' \
     --output /tmp/piper_gyro_test.wav
```

---

### 4.2 MMS Persian (SECOND — Meta baseline)

**Why second:** Official Meta model. Sets the quality baseline for all later Persian engines. Trivial `transformers` API.

**`tts_lab_config.py`:**
```python
# In MODEL_INFO:
"mmsfas": {
    "label": "MMS Persian (Meta)", "size": "~150 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 200, "heavy": False,
    "notes": "Meta MMS-TTS Persian VITS. Reference baseline. CC-BY-NC 4.0.",
    "arthur_fit": 5,
},
# In MODEL_ORDER, insert after "manatts":
"mmsfas",
```

**`tts_lab_engines.py`:**
```python
# ── MMS Persian TTS (Meta) ──────────────────────────────────────────────────────
def _load_mmsfas():
    from transformers import VitsModel, AutoTokenizer
    import torch
    model = VitsModel.from_pretrained("facebook/mms-tts-fas").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
    return (model, tokenizer)

def _synth_mmsfas(inst, text, params):
    import torch
    model, tokenizer = inst
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform
    arr = output.cpu().numpy().squeeze()
    return _to_wav(arr, model.config.sampling_rate), model.config.sampling_rate

# In LOADERS:
"mmsfas": _load_mmsfas,
# In SYNTHERS:
"mmsfas": _synth_mmsfas,
```

**`docker-compose.yml`** — orchestrator:
```yaml
MMSFAS_URL: http://engine-current:8101
```

**`Dockerfile.engine-current`** — No changes. `transformers` already installed.

**Verification:**
```bash
curl -X POST "http://192.168.0.87:8001/synthesize/mmsfas" \
     -H "Content-Type: application/json" \
     -d '{"text": "سلام، حال شما چطور است؟"}' \
     --output /tmp/mmsfas_test.wav
```

---

## 5. Phase 2: Community VITS Voices

### 5.1 Shared Infrastructure

All Persian VITS models use Coqui TTS `Synthesizer`. Key differences from the original plan:

| Original Plan | Verified Reality |
|---------------|-----------------|
| `Synthesizer(model_path=..., config_path=...)` | `Synthesizer(tts_checkpoint=..., tts_config_path=...)` |
| `snapshot_download(repo)` | `hf_hub_download(repo, filename)` — single file, not entire repo |
| Files are ~200 MB | Files are **~998 MB** (training checkpoints) |
| Saillab: `azure-grapheme-60K`, `cv15-reduct` | Saillab: `ZabanZad_VITS_MAle`, `ZabanZad_VITS_Female` (different repos!) |

### 5.2 Saillab Decision

**Saillab repos are gated** — require `HUGGING_FACE_HUB_TOKEN`. The models I originally catalogued (`azure-grapheme-60K`, `cv15-reduct-multispeaker`, etc.) are **commented out** in the ZabanZad PoC — they appear to be deprecated/abandoned.

The actively maintained Saillab models are:
- `saillab/ZabanZad_VITS_MAle` — Male, checkpoint `checkpoint_61000.pth`
- `saillab/ZabanZad_VITS_Female` — Female, checkpoint `best_model_15397.pth`, includes `speakers1.pth`

**Recommendation:** Include them IF the HF token is available (you already have one for Qwen3TTS). Mark as gated in MODEL_INFO. If token issues arise, skip — Kamtera models provide female/male coverage.

### 5.3 Config

**`tts_lab_config.py`:**
```python
# Persian VITS model registry — verified filenames from repo inspection (2026-06-23)
PERSIAN_VITS_MODELS = {
    "kamtera_f": {
        "repo": "Kamtera/persian-tts-female-vits",
        "checkpoint": "best_model_30824.pth",    # highest step among best_model_*.pth
        "config": "config.json",
        "speakers": None,
        "gated": False,
    },
    "kamtera_m": {
        "repo": "Kamtera/persian-tts-male-vits",
        "checkpoint": "best_model_98066.pth",    # verified: latest best
        "config": "config.json",
        "speakers": None,
        "gated": False,
    },
    "gptinf_fa": {
        "repo": "karim23657/persian-tts-female-GPTInformal-Persian-vits",
        "checkpoint": "best_model_98066.pth",    # verified: latest best
        "config": "config.json",
        "speakers": None,
        "gated": False,
    },
    # Gated — requires HUGGING_FACE_HUB_TOKEN
    "zabanzad_f": {
        "repo": "saillab/ZabanZad_VITS_Female",
        "checkpoint": "best_model_15397.pth",
        "config": "config.json",
        "speakers": "speakers1.pth",             # NOTE: speakers1.pth, not speakers.pth
        "gated": True,
    },
    "zabanzad_m": {
        "repo": "saillab/ZabanZad_VITS_MAle",
        "checkpoint": "checkpoint_61000.pth",
        "config": "config.json",
        "speakers": None,
        "gated": True,
    },
}
```

### 5.4 Loader (shared)

**`tts_lab_engines.py`:**
```python
# ── Persian VITS Community Models (shared loader) ───────────────────────────────
import os as _os
from huggingface_hub import hf_hub_download as _hf_dl

def _load_persian_vits(engine_key):
    """Shared loader for Persian VITS models.

    Downloads individual checkpoint + config files from HF Hub.
    Uses HUGGING_FACE_HUB_TOKEN for gated repos (Saillab/ZabanZad).
    Checkpoints are ~998 MB training checkpoints — Coqui TTS Synthesizer
    extracts the model key from the state dict automatically.
    """
    from TTS.utils.synthesizer import Synthesizer

    info = PERSIAN_VITS_MODELS[engine_key]
    token = _os.environ.get("HUGGING_FACE_HUB_TOKEN") if info["gated"] else None

    # Download individual files — NOT snapshot_download (would pull 7+ GB)
    ckpt_path = _hf_dl(
        repo_id=info["repo"],
        filename=info["checkpoint"],
        cache_dir=str(MODELS_DIR / "huggingface"),
        token=token,
    )
    cfg_path = _hf_dl(
        repo_id=info["repo"],
        filename=info["config"],
        cache_dir=str(MODELS_DIR / "huggingface"),
        token=token,
    )

    kwargs = dict(
        tts_checkpoint=ckpt_path,      # verified API: tts_checkpoint, not model_path
        tts_config_path=cfg_path,       # verified API: tts_config_path, not config_path
        use_cuda=DEVICE == "cuda",
    )

    # Multi-speaker models include a speakers.pth file
    if info.get("speakers"):
        spk_path = _hf_dl(
            repo_id=info["repo"],
            filename=info["speakers"],
            cache_dir=str(MODELS_DIR / "huggingface"),
            token=token,
        )
        kwargs["tts_speakers_file"] = spk_path

    synthesizer = Synthesizer(**kwargs)
    slog("LOAD", engine_key, f"Loaded {info['repo']} ({info['checkpoint']})")
    return synthesizer

def _synth_persian_vits(inst, text, params):
    """Shared synth for Persian VITS models.

    VITS grapheme models don't need phonemization — pass text directly.
    Some models support speaker_name for multi-speaker selection.
    """
    speaker = params.get("speaker", None)
    try:
        wavs = inst.tts(text, speaker_name=speaker) if speaker else inst.tts(text)
    except TypeError:
        wavs = inst.tts(text)  # single-speaker models don't accept speaker_name
    arr = np.array(wavs)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return _to_wav(arr, inst.output_sample_rate), inst.output_sample_rate

# Per-engine loaders (thin wrappers for LOADERS dict):
def _load_kamtera_f():  return _load_persian_vits("kamtera_f")
def _load_kamtera_m():  return _load_persian_vits("kamtera_m")
def _load_gptinf_fa():  return _load_persian_vits("gptinf_fa")
def _load_zabanzad_f(): return _load_persian_vits("zabanzad_f")
def _load_zabanzad_m(): return _load_persian_vits("zabanzad_m")
```

### 5.5 MODEL_INFO entries

```python
"kamtera_f":  {"label":"Kamtera Persian ♀","size":"~1 GB (ckpt)","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Community VITS female Persian. ~31K steps. Most popular (104 dl/mo).",
               "arthur_fit":5},
"kamtera_m":  {"label":"Kamtera Persian ♂","size":"~1 GB (ckpt)","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Community VITS male Persian. ~98K steps.",
               "arthur_fit":5},
"gptinf_fa":  {"label":"GPTInformal Persian","size":"~1 GB (ckpt)","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"Trained on informal/colloquial Persian. Handles slang and casual speech.",
               "arthur_fit":3},
"zabanzad_f": {"label":"ZabanZad VITS ♀","size":"~1 GB (ckpt)","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"⚠ Gated — needs HF token. Active Saillab model (ZabanZad PoC).",
               "arthur_fit":4},
"zabanzad_m": {"label":"ZabanZad VITS ♂","size":"~1 GB (ckpt)","rtf_est":"RTF ~0.5×",
               "ram_est_mb":500,"heavy":False,
               "notes":"⚠ Gated — needs HF token. Active Saillab model (ZabanZad PoC).",
               "arthur_fit":4},
```

### 5.6 docker-compose.yml additions
```yaml
KAMTERA_F_URL: http://engine-current:8101
KAMTERA_M_URL: http://engine-current:8101
GPTINF_FA_URL: http://engine-current:8101
ZABANZAD_F_URL: http://engine-current:8101
ZABANZAD_M_URL: http://engine-current:8101
```

### 5.7 Verification
```bash
# Test each VITS model after deployment:
for engine in kamtera_f kamtera_m gptinf_fa zabanzad_f zabanzad_m; do
    echo "=== Testing $engine ==="
    curl -s -X POST "http://192.168.0.87:8001/synthesize/$engine" \
         -H "Content-Type: application/json" \
         -d '{"text": "سلام، حال شما چطور است؟"}' \
         --output "/tmp/${engine}_test.wav"
    echo "OK: /tmp/${engine}_test.wav"
done
```

---

## 6. Phase 3: XTTS Persian Fine-tune

### 6.1 Cache Key Issue (per review feedback)

**Problem:** `_ensure_loaded("xtts")` caches by engine key. If Persian and English both use key `xtts`, switching language would reload the wrong checkpoint — or worse, silently use the wrong one.

**Solution:** Two internal cache keys, one user-facing engine:

```python
# Internal dispatch (tts_lab_dispatch.py):
# When language="fa" → cache key "xtts_fa" → loads Persian checkpoint
# When language="en" (or other) → cache key "xtts" → loads base checkpoint

# MODEL_ORDER stays unchanged: just "xtts"
# The UI sees one "xtts" engine with a language selector.
```

### 6.2 Implementation

**`tts_lab_config.py`:**
```python
XTTS_FA_REPO = "saillab/xtts_v2_fa"
XTTS_FA_CHECKPOINT = "best_model_110880.pth"
XTTS_FA_CONFIG = "config.json"
```

**`tts_lab_engines.py`** — Modify `_load_xtts()` to accept language:
```python
def _load_xtts(language=None):
    """Load XTTS v2. Uses Persian fine-tune when language='fa'.

    The Persian fine-tune is gated (needs HUGGING_FACE_HUB_TOKEN).
    Falls back to base XTTS v2 if the checkpoint is unavailable.
    """
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from huggingface_hub import hf_hub_download

    if language == "fa":
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            ckpt = hf_hub_download(
                repo_id=XTTS_FA_REPO,
                filename=XTTS_FA_CHECKPOINT,
                cache_dir=str(MODELS_DIR / "huggingface"),
                token=token,
            )
            cfg = hf_hub_download(
                repo_id=XTTS_FA_REPO,
                filename=XTTS_FA_CONFIG,
                cache_dir=str(MODELS_DIR / "huggingface"),
                token=token,
            )
            config = XttsConfig()
            config.load_json(cfg)
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_path=ckpt)
            model.cuda()
            slog("LOAD", "xtts_fa", f"Persian XTTS fine-tune loaded ({XTTS_FA_CHECKPOINT})")
            return (model, config, "fa")
        except Exception as e:
            slog("LOAD", "xtts_fa", f"Persian XTTS unavailable: {e} — falling back to base")
            # Fall through to base model

    # Base XTTS v2 (existing logic):
    from TTS.api import TTS
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    return (model, None, language or "en")
```

**`tts_lab_dispatch.py`** — Modify `_ensure_loaded()` to handle xtts cache keys:
```python
# In _ensure_loaded() or equivalent dispatch logic:
def _ensure_loaded(name, **kwargs):
    # XTTS has language-specific cache keys to avoid collisions
    if name == "xtts":
        language = kwargs.get("language", "en")
        cache_key = f"xtts_{language}" if language == "fa" else "xtts"
    else:
        cache_key = name

    with _state[cache_key]["lock"]:
        if _state[cache_key]["instance"] is None:
            loader = LOADERS[name]
            if name == "xtts":
                _state[cache_key]["instance"] = loader(language=language)
            else:
                _state[cache_key]["instance"] = loader()
    return _state[cache_key]["instance"]
```

**`tts_lab_dispatch.py`** — `_do_synth()` passes language:
```python
# In _do_synth():
language = params.get("language", "en")
inst = _ensure_loaded(name, language=language)
```

### 6.3 Verification
```bash
# Test base XTTS (English):
curl -X POST "http://192.168.0.87:8001/synthesize/xtts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a test.", "params": {"language": "en"}}' \
     --output /tmp/xtts_en.wav

# Test XTTS Persian fine-tune:
curl -X POST "http://192.168.0.87:8001/synthesize/xtts" \
     -H "Content-Type: application/json" \
     -d '{"text": "سلام، حال شما چطور است؟", "params": {"language": "fa"}}' \
     --output /tmp/xtts_fa.wav
```

---

## 7. Phase 4: SpeechT5 (after verification)

### 7.1 Pre-Integration Verification (REQUIRED before coding)

**The speaker embedding issue flagged in review must be resolved first:**

```python
# Run THIS interactively on the VM BEFORE writing any integration code:
# $ source /opt/arthur-bench-env/bin/activate
# $ python3

from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    SpeechT5HifiGan,
    AutoModelForTextToSpectrogram,
)
import torch

model_id = "Hamid20/speecht5_tts_persian"

# Step 1: Can we load the model?
model = AutoModelForTextToSpectrogram.from_pretrained(model_id)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
print("✅ Model loaded")

# Step 2: Does generate_speech() work?
# The critical question: where do speaker embeddings come from?
inputs = processor(text="سلام، حال شما چطور است؟", return_tensors="pt")

# Try 1: first row of speaker_embedding weight (common pattern)
try:
    spk = model.speecht5.encoder.speaker_embedding.weight[:1]
    speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
    print(f"✅ Try 1 worked: speaker_embedding.weight[:1], shape={speech.shape}")
except Exception as e:
    print(f"❌ Try 1 failed: {e}")

# Try 2: Check if the model has get_speaker_embeddings()
try:
    spk = model.get_speaker_embeddings()
    speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
    print(f"✅ Try 2 worked: get_speaker_embeddings()")
except Exception as e:
    print(f"❌ Try 2 failed: {e}")

# Try 3: Does it expose any speaker-related attribute?
import inspect
for attr in dir(model):
    if 'speaker' in attr.lower() or 'embed' in attr.lower():
        print(f"  Found: model.{attr}")

# Step 3: Inspect the checkpoint
# Look at what keys exist in the safetensors
from safetensors import safe_open
import glob
safetensors_files = glob.glob(f"{model_id}/*.safetensors") if hasattr(model_id, 'glob') else []
# If loaded from HF cache, check there
from transformers.utils import TRANSFORMERS_CACHE
print(f"HF cache: {TRANSFORMERS_CACHE}")
```

**Gate:** Do NOT proceed to code integration until Step 2 succeeds. If all 3 tries fail, SpeechT5 Persian is blocked until we determine where the speaker embeddings live.

### 7.2 Implementation (only after verification passes)

**`tts_lab_config.py`:**
```python
"speecht5_fa": {"label":"SpeechT5 Persian","size":"~400 MB","rtf_est":"RTF ~1.0×",
                "ram_est_mb":500,"heavy":False,
                "notes":"Fine-tuned Microsoft SpeechT5. 4000 steps, val loss 0.5369. MIT.",
                "arthur_fit":4},
```

**`tts_lab_engines.py`:**
```python
# ── SpeechT5 Persian ────────────────────────────────────────────────────────────
def _load_speecht5_fa():
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
    model = SpeechT5ForTextToSpeech.from_pretrained("Hamid20/speecht5_tts_persian").to(DEVICE)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
    # Use the approach validated in pre-integration verification
    speaker_embedding = model.speecht5.encoder.speaker_embedding.weight[:1].to(DEVICE)
    return (model, processor, vocoder, speaker_embedding)

def _synth_speecht5_fa(inst, text, params):
    import torch
    model, processor, vocoder, default_spk = inst
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)
    spk = params.get("speaker_embedding", default_spk)
    speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
    arr = speech.cpu().numpy().squeeze()
    return _to_wav(arr, 16000), 16000
```

**`docker-compose.yml`:**
```yaml
SPEECHT5_FA_URL: http://engine-current:8101
```

---

## 8. Phase 5: MOSS-TTS (independent evaluation)

### 8.1 Why Deferred

Per review feedback:

> April–June 2026 TTS projects frequently pin transformers, tokenizers, accelerate, protobuf, sentencepiece and can silently downgrade working dependencies.

MOSS-TTS-Nano was released April 10, 2026. It is the newest model in the catalog. Its `moss-tts-nano` pip package has unknown dependency constraints. A silent torch/transformers downgrade in engine-current would break 21 existing engines.

**Decision:** Evaluate MOSS-TTS completely independently. Do not let it delay Phases 1–4.

### 8.2 Evaluation Procedure

```bash
# Step 1: Create isolated venv
python3 -m venv /tmp/mosstts-test
source /tmp/mosstts-test/bin/activate

# Step 2: Install and snapshot deps
pip install moss-tts-nano
pip freeze > /tmp/mosstts-deps-before.txt

# Step 3: Check for conflicts with engine-current deps
# Compare against engine-current's installed packages:
# docker exec tts-lab-engine-current pip freeze > /tmp/engine-current-deps.txt
# diff /tmp/mosstts-deps-before.txt /tmp/engine-current-deps.txt | grep -E "^[<>]"

# Step 4: Verify with pip check
pip check

# Step 5: Tree check
pip install pipdeptree
pipdeptree -p moss-tts-nano

# Step 6: Test synthesis
python3 -c "
from moss_tts_nano import MOSSTTSNano
model = MOSSTTSNano.from_pretrained('FenomAI/MOSS-TTS-Nano-100M')
# ... test generate ...
print('✅ MOSS-TTS-Nano works in isolation')
"

# Step 7: Co-install with engine-current deps
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip install 'transformers>=5.12.0'
pip check  # ← THIS MUST PASS
```

### 8.3 Go/No-Go Criteria

| Criterion | Threshold |
|-----------|-----------|
| `pip check` in isolated venv | Must pass |
| `pip check` with engine-current deps | Must pass |
| No torch downgrade | torch version must stay ≥ 2.12.0 |
| No transformers downgrade | transformers must stay ≥ 5.12.0 |
| Synthesis produces valid audio | Non-silent, recognizable Persian speech |

**If all criteria pass:** Create `Dockerfile.engine-current` addition:
```dockerfile
RUN pip install --no-cache-dir moss-tts-nano
```

**If any criterion fails:** Create dedicated `Dockerfile.engine-persian` (see original plan) OR skip MOSS-TTS entirely. Do NOT force it into engine-current.

---

## 9. Files Changed — Complete Inventory

| File | Phase | Change | Risk |
|------|-------|--------|------|
| `tts_lab_config.py` | 1–4 | +10 MODEL_INFO, +10 MODEL_ORDER, +PERSIAN_VITS_MODELS dict, +XTTS_FA paths | Low |
| `tts_lab_engines.py` | 1–4 | +_load/_synth pairs: mmsfas, 5× persian_vits wrappers, xtts lang-aware, speecht5_fa | Low |
| `tts_lab_dispatch.py` | 3 | xtts cache key split (xtts / xtts_fa) | Medium |
| `docker-compose.yml` | 1–4 | +10 env vars in orchestrator | Low |
| `Dockerfile.engine-current` | none | No changes needed (all deps already installed) | — |
| `docs/engine_compatibility.yaml` | 1–4 | +10 engine entries | Low |
| `scripts/benchmark/persian_bench.py` | 0 | NEW — benchmark script | Low |
| `/opt/arthur/persian_benchmark.py` | 0 | NEW — benchmark sentences (on VM) | Low |

**NOT changed:**
- `Dockerfile.stack.current` — no new system deps
- `Dockerfile.base` — no new system deps
- `Dockerfile.orchestrator` — pure HTTP, no ML
- `tts_lab_shims.py` — no new patches needed
- `tts_lab_ui.py` — auto-renders from MODEL_ORDER

---

## 10. Pre-Integration Verification Checklist

Before writing a single line of integration code, verify:

### Must Verify (gate for Phase 4+5)

- [ ] **SpeechT5 speaker embeddings** — Run the interactive test from §7.1. Do not proceed if `generate_speech()` fails all 3 approaches.
- [ ] **MOSS-TTS dependency tree** — Run `pip install moss-tts-nano && pip check` in an isolated venv. Do not proceed if torch/transformers are downgraded.

### Should Verify (before Phase 2)

- [ ] **VITS checkpoint loading** — Download one Kamtera `.pth` and confirm `Synthesizer(tts_checkpoint=..., tts_config_path=...)` loads without error
- [ ] **Saillab token validity** — Confirm `HUGGING_FACE_HUB_TOKEN` can access `saillab/ZabanZad_VITS_Female`
- [ ] **HF Hub download speed** — Single-file `hf_hub_download` for a 998 MB checkpoint. If slow, pre-download to `/opt/models/` before engine registration.

### Nice to Verify

- [ ] **Piper Gyro audio quality** — Generate sample audio manually, listen for obvious artifacts
- [ ] **MMS Persian audio quality** — Same, establish baseline MOS score
- [ ] **All benchmark sentences** — Run Phase 0 benchmark against existing Persian engines (chatterbox, matcha, manatts, piper-mana) to establish pre-integration baselines

---

## Appendix A: Revised Saillab Strategy

The Saillab models I originally catalogued differ from what's actually active:

| Catalogued (my original plan) | Status | Active Alternative |
|-------------------------------|--------|-------------------|
| `saillab/persian-tts-azure-grapheme-60K` | Gated, commented out in PoC | → `saillab/ZabanZad_VITS_Female` |
| `saillab/persian-tts-cv15-reduct-grapheme-multispeaker` | Gated, commented out | → `saillab/ZabanZad_VITS_MAle` |
| `saillab/multi_speaker` | Gated, commented out | — (no active multi-speaker) |
| `saillab/female_cv_azure_male_azure_female` | Gated, commented out | — (ZabanZad Female replaces) |
| `saillab/xtts_v2_fa` | Gated, active in PoC | → Keep (XTTS tab) |

**Action:** Replace 4 catalogued Saillab VITS models with 2 ZabanZad active models. The ZabanZad PoC is the authoritative source for which Saillab models are maintained.

---

## Appendix B: Updated MODEL_ORDER (cumulative, all phases)

```python
MODEL_ORDER = [
    # ... existing engines ...
    "manatts",      # existing Persian Tacotron2
    # Phase 1
    "mmsfas",       # Meta MMS Persian — baseline
    # Phase 2
    "kamtera_f",    # Community VITS female
    "kamtera_m",    # Community VITS male
    "zabanzad_f",   # Saillab ZabanZad female (gated)
    "zabanzad_m",   # Saillab ZabanZad male (gated)
    "gptinf_fa",    # Informal Persian VITS
    # Phase 4 (blocked on verification)
    "speecht5_fa",  # SpeechT5 Persian fine-tune
    # Phase 5 (independent eval)
    # "mosstts",    # MOSS-TTS-Nano — uncomment after dependency validation
    # ... rest of existing engines ...
]
```

Piper Gyro is a voice preset on the existing `piper` engine — no new engine key needed.
