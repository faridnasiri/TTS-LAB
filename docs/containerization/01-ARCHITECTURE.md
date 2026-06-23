# TTS Lab — Container Architecture (Final)

> **Date:** 2026-06-23
> **Status:** Architecture complete — runtime validation in progress
> **Supersedes:** archive/CONTAINERIZATION_PLAN.md, archive/CONTAINERIZATION_MASTER_PLAN.md, archive/IAC_REWRITE_PLAN.md
> **Sibling docs:**
> - [05-STATE-2026-06-21.md](05-STATE-2026-06-21.md) — ad-hoc deployment snapshot (2026-06-21)
> - [04-ADHOC-LOG.md](04-ADHOC-LOG.md) — day-by-day fix log
> - [../reference/ARCHITECTURE_REFERENCE.md](../reference/ARCHITECTURE_REFERENCE.md) — deployed-state reference
> - [../engine_compatibility.yaml](../engine_compatibility.yaml) — machine-readable single source of truth

---

## Table of Contents

- [1. Architecture Principle](#1-architecture-principle)
- [2. Container Topology](#2-container-topology)
- [3. Stack Definitions](#3-stack-definitions)
- [4. Engine Distribution](#4-engine-distribution)
- [5. Engine Maturity Classification](#5-engine-maturity-classification)
- [6. Compatibility Matrix](#6-compatibility-matrix)
- [7. Validation Framework](#7-validation-framework)
- [8. Phase Progression](#8-phase-progression)
- [9. Current State → Target State](#9-current-state--target-state)
- [10. Remaining Gates](#10-remaining-gates)
- [11. Operational Contracts](#11-operational-contracts)
- [12. File Inventory](#12-file-inventory)

---

## 1. Architecture Principle

The project has evolved from an **engine-centric** architecture (one container per engine) to a **compatibility-domain** architecture (one container per dependency boundary).

```
Before (engine-centric):          After (compatibility-domain):

1 container per engine            1 container per stack
28 Dockerfiles                    7 Dockerfiles (6 custom + 1 pre-built)
28 build targets                  7 build targets
28 CI paths                       7 CI paths
```

A container exists not because an engine needs isolation, but because a **set of engines shares a dependency compatibility boundary**. Engines that agree on torch, transformers, CUDA, and Python versions co-reside in the same container. Engines that disagree get their own container scoped to exactly the versions they need.

---

## 2. Container Topology

```
                          TTS-LAB Orchestrator (port 8001)

        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼

┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  engine-current   │  │    engine-mid     │  │   engine-qwen     │
│  torch 2.12 n.    │  │  torch 2.10 stbl  │  │  torch 2.10 stbl  │
│  tf 5.12.1        │  │  tf 4.50-5.0      │  │  tf 4.51-4.54     │
│  cuda 12.8        │  │  cuda 12.1        │  │  hf-hub <1.0      │
│  python 3.10      │  │  python 3.10      │  │  cuda 12.1        │
│  port 8101        │  │  port 8103        │  │  port 8104        │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ Piper       SUPP  │  │ VibeVoice  EXPER  │  │ Qwen3TTS   EXPER  │
│ Kokoro      SUPP  │  │ Higgs      EXPER  │  │                   │
│ Melo        SUPP  │  └───────────────────┘  └───────────────────┘
│ Matcha      SUPP  │
│ ChatTTS     SUPP  │
│ OuteTTS     SUPP  │
│ Bark        SUPP  │
│ StyleTTS2   SUPP  │
│ F5-TTS      SUPP  │
│ Dia         SUPP  │
│ Chatterbox  SUPP  │
│ ChatterTurbo SUPP │
│ FishSpeech  SUPP  │
│ OmniVoice   SUPP  │
│ Zonos       SUPP  │
│ XTTS        SUPP  │
│ CosyVoice   EXPER │
│ CSM         EXPER │
│ ManaTTS     EXPER │
│ NeuTTS      EXPER │
│ OpenVoice   EXPER │
│ (+ orpheus probe) │
└───────────────────┘

        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │  engine-legacy    │  │     orpheus       │  │      s2pro        │
        │  torch 1.13       │  │  vllm + cuda 12.1 │  │  SGLang pre-built │
        │  tf 4.46          │  │  port 8002        │  │  port 8005        │
        │  cuda 11.7        │  │  profile: gpu     │  │  profile: sglang  │
        │  port 8102        │  ├───────────────────┤  ├───────────────────┤
        ├───────────────────┤  │ Orpheus    BLOCK  │  │ S2-Pro     BLOCK  │
        │ IndexTTS   BLOCK  │  └───────────────────┘  └───────────────────┘
        │ Parler     BLOCK  │
        └───────────────────┘
```

**7 containers — 6 custom + 1 pre-built (SGLang).**

---

## 3. Stack Definitions

### 3.1 Stack Layers (Tiered Inheritance)

```
Tier 1: tts-lab-base  (~1.5 GB)
  FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04
  System: espeak-ng, ffmpeg, MeCab, Python 3.11, git, wget, curl
  Python utilities: fastapi, uvicorn, httpx, soundfile, huggingface_hub
  NLTK data: punkt, punkt_tab, cmudict, averaged_perceptron_tagger
  Application code: tts_lab*.py
  Shared by ALL 6 custom containers. Stored ONCE on disk.

    ├── Tier 2: stack-current  (+3.5 GB → ~5 GB total)
    │     torch 2.12 nightly (cu128), transformers 5.12.1, onnxruntime
    │     Used by: engine-current
    │
    ├── Tier 2: stack-mid      (+3.0 GB → ~4.5 GB total)
    │     torch 2.10 stable (cu121), transformers 4.x
    │     Used by: engine-mid, engine-qwen
    │
    └── Tier 2: stack-legacy   (+2.5 GB → ~4.0 GB total)
          torch 1.13 (cu117), transformers 4.46
          Used by: engine-legacy
```

### 3.2 Stack Version Constraints

| Stack | torch | transformers | CUDA | Python | Notes |
|-------|-------|-------------|------|--------|-------|
| **current** | `>=2.12,<2.13` | `>=5.12.1,<5.13` | 12.8 | 3.10 | torch nightly (2.12.0.dev*) — **largest risk surface: 21 engines** |
| **mid** | `>=2.10,<2.11` | `>=4.50,<5.0` | 12.1 | 3.10 | Stable — no nightly regressions |
| **legacy** | `>=1.13,<1.14` | `>=4.46,<5.0` | 11.7 | 3.10 | Frozen — sealed capsule |

### 3.3 Torch Nightly Risk

The `current` stack carries 21 engines on **torch nightly** — the largest risk surface. A single nightly regression can affect the majority of supported engines simultaneously. The `engine_compatibility.yaml` stack definition includes a `validated_on` fingerprint to freeze the exact validated versions:

```yaml
stacks:
  current:
    torch: '>=2.12,<2.13'
    validated_on:
      torch: null     # set after deployment to e.g. 2.12.0.dev20260622
      transformers: null
      cuda: null
      driver: null
```

This allows answering "what exact nightly was this validated against?" after deployment, even if the range constraint remains broad.

### 3.4 engine-qwen Pins (within mid stack)

engine-qwen inherits from stack-mid but applies tighter pins:

| Dependency | Constraint | Reason |
|-----------|-----------|--------|
| transformers | `>=4.51,<4.54` | ROPE_INIT_FUNCTIONS present; TransformGetItemToIndex not yet added |
| huggingface-hub | `>=0.34,<1.0` | qwen_tts 0.1.1 conflicts with is_offline_mode removal in >=1.0 |
| qwen-tts | `==0.1.1` | Last known-compatible release |

These constraints are too narrow to share with VibeVoice/Higgs in engine-mid — hence the dedicated container.

---

## 4. Engine Distribution

| Container | Total | Supported | Experimental | Blocked |
|-----------|:-----:|:---------:|:------------:|:-------:|
| `engine-current` | **21** | 16 | 5 | 0 |
| `engine-mid` | **2** | 0 | 2 | 0 |
| `engine-qwen` | **1** | 0 | 1 | 0 |
| `engine-legacy` | **2** | 0 | 0 | 2 |
| `orpheus` | **1** | 0 | 0 | 1 |
| `s2pro` | **1** | 0 | 0 | 1 |
| **Total** | **28** | **16** | **8** | **4** |

### 4.1 Per-Container Engine List

**engine-current** (21 engines — torch 2.12 nightly + tf 5.12)

| Engine | Status | Notes |
|--------|:------:|-------|
| piper | SUPP | ONNX CPU — no torch dependency |
| kokoro | SUPP | ONNX CPU |
| melo | SUPP | Needs MeCab + unidic |
| matcha | SUPP | ONNX flow-matching, real-time (0.24× RTF) |
| chattts | SUPP | Ref voice: LZMA bug, fallback to random speaker |
| outetts | SUPP | LLM-based, auto-capped max_length, 15-26× RTF |
| bark | SUPP | Heavy VRAM (~12 GB) |
| styletts2 | SUPP | Real-time (0.22× RTF), needs langchain<0.3.0 |
| f5tts | SUPP | Voice cloning, needs hf-hub>=1.0 |
| dia | SUPP | 1.6B, use Dia-1.6B-0626 only, KV cache patch |
| chatterbox | SUPP | Needs torchcodec metadata stub |
| chatterboxturbo | SUPP | One-step distilled, near real-time (1.11× RTF) |
| fishspeech | SUPP | Voice cloning, needs lightning (not pytorch_lightning) |
| omnivoice | SUPP | 600+ languages, real-time (0.67× RTF) |
| zonos | SUPP | Voice cloning, backbone copy from git repo needed |
| xtts | SUPP | torchcodec issue with torch nightly |
| cosyvoice | EXPER | Needs git clone + model download + hyperpyyaml |
| csm | EXPER | Meta license gated, needs torchtune/torchao/moshi/silentcipher |
| manatts | EXPER | Persian multi-speaker Tacotron2, not configured |
| neutts | EXPER | NeuTTS Air, not configured |
| openvoice | EXPER | Checkpoints not downloaded |

**engine-mid** (2 engines — torch 2.10 stable + tf 4.x)

| Engine | Status | Notes |
|--------|:------:|-------|
| vibevoice | EXPER | POC pending — AutoConfig → AutoModel → inference → VRAM |
| higgs | EXPER | POC pending — AutoModelForSeq2SeqLM → inference → VRAM |

**engine-qwen** (1 engine — torch 2.10 + tf 4.51-4.54, hf-hub<1.0)

| Engine | Status | Notes |
|--------|:------:|-------|
| qwen3tts | EXPER | Gated model, needs HF_TOKEN. Build-time validation passes. |

**engine-legacy** (2 engines — torch 1.13 + tf 4.46)

| Engine | Status | Notes |
|--------|:------:|-------|
| indextts | BLOCK | Needs legacy stack build |
| parler | BLOCK | Needs legacy stack build |

**orpheus** (1 engine — vllm + CUDA 12.1)

| Engine | Status | Notes |
|--------|:------:|-------|
| orpheus | BLOCK | vllm incompatible with torch nightly. Gated model. |

**s2pro** (1 engine — SGLang pre-built)

| Engine | Status | Notes |
|--------|:------:|-------|
| s2pro | BLOCK | SGLang image tf 5.6.0 too old. Requires paged KV cache, RadixAttention, CUDA graph replay. Do not attempt local inference. |

---

## 5. Engine Maturity Classification

Every engine is assigned one of three states. The state is stored in [engine_compatibility.yaml](../engine_compatibility.yaml) and determines CI behavior.

### 5.1 States

| State | Icon | Meaning | CI: Build | CI: Smoke Test | Required |
|-------|:----:|---------|:---------:|:--------------:|:--------:|
| **SUPPORTED** | SUPP | Synthesis confirmed on target hardware | Yes | Yes | Pass |
| **DEPRECATED** | DEPR | Still works, no longer recommended | Yes | Yes | Warn |
| **EXPERIMENTAL** | EXPER | Container defined, not yet validated | Yes | Best effort | Warn |
| **BLOCKED** | BLOCK | Cannot work — upstream missing or incompatible | No | No | Skip |

### 5.2 Lifecycle

```
EXPERIMENTAL  ──(all promotion gates pass)──▶  SUPPORTED
EXPERIMENTAL  ──(blocker found)─────────────▶  BLOCKED
BLOCKED       ──(upstream releases fix)─────▶  EXPERIMENTAL
SUPPORTED     ──(superseded by better)──────▶  DEPRECATED
DEPRECATED    ──(eventually breaks)─────────▶  BLOCKED
```

DEPRECATED prevents conflating "cannot run" with "can run but should not receive future investment." A deprecated engine still builds, still passes smoke tests, and is counted alongside supported engines. It exists for models that work but have been superseded by better alternatives.

**Promotion from EXPERIMENTAL to SUPPORTED is deterministic:** every gate in `promotion_requirements` must be `passed`. No manual judgment. The `update_engine_status.py` script enforces this automatically.

### 5.3 Promotion Requirements per Engine

| Engine | Gates Required |
|--------|---------------|
| **qwen3tts** | build_import, model_load, synthesis, vram_measured |
| **vibevoice** | config_load, model_load, inference, vram_measured |
| **higgs** | config_load, model_load, inference, vram_measured |

---

## 6. Compatibility Matrix

### 6.1 Single Source of Truth

The file [docs/engine_compatibility.yaml](../engine_compatibility.yaml) is the **machine-readable single source of truth** for:

- Engine maturity status
- Dependency version constraints with documented incompatibility reasons
- Container assignment
- VRAM estimates
- Validation gate status (with timestamps, durations, VRAM peaks)
- Environment fingerprints (torch, transformers, CUDA, driver versions)
- Gated model requirements (HF_TOKEN needed)
- SGLang fallback paths
- Upstream monitoring URLs (for BLOCKED engines)
- Promotion requirements (explicit gate lists)

### 6.2 Derived Fields

The `summary` block at the bottom of the matrix is **computed from engine data on every write** — never hand-maintained. The `recompute_summary()` function in `update_engine_status.py` walks every engine entry and rebuilds the counts. Any drift caused by manual editing, engine additions, or removals is automatically corrected on the next write.

### 6.3 CI Contracts

```
SUPPORTED     → build: yes, smoke-test: yes, required: pass
EXPERIMENTAL  → build: yes, smoke-test: best-effort, required: warn
BLOCKED       → build: no,  smoke-test: no,  required: skip
```

---

## 7. Validation Framework

### 7.1 Script

[scripts/update_engine_status.py](../../scripts/update_engine_status.py) — updates the compatibility matrix from test results.

### 7.2 CLI Surface

```bash
# Record a passed gate (with metrics):
python scripts/update_engine_status.py vibevoice model_load passed \
  --duration 41 --vram-mb 6420 --container engine-mid

# Record a failed gate (with error):
python scripts/update_engine_status.py qwen3tts build_import failed \
  --error "ROPE_INIT_FUNCTIONS missing default key"

# Check promotion eligibility:
python scripts/update_engine_status.py vibevoice --check

# Promote when all gates pass:
python scripts/update_engine_status.py vibevoice --promote

# Fix summary drift without changing gates:
python scripts/update_engine_status.py --recompute

# Print environment fingerprint:
python scripts/update_engine_status.py --fingerprint --container engine-mid

# Print current VRAM:
python scripts/update_engine_status.py --vram-now
```

### 7.3 Auto-Populated Fields

| Field | Source | When |
|-------|--------|------|
| `last_tested` | `datetime.now(UTC)` | Every gate update |
| `validated_on.torch` | `torch.__version__` (from target container) | Every gate update |
| `validated_on.transformers` | `transformers.__version__` (from target container) | Every gate update |
| `validated_on.cuda` | `torch.version.cuda` (from target container) | Every gate update |
| `validated_on.driver` | `nvidia-smi` (host) | Every gate update |
| `duration_seconds` | `--duration` flag | When provided |
| `peak_vram_mb` | `--vram-mb` flag | When provided |
| `error` | `--error` flag | On failure (cleared on pass) |
| `history` | Previous gate state | Every gate update (last 5) |
| `promoted_on` | Auto-set | On `--promote` |
| `summary` | Auto-recomputed from engine data | Every `save_matrix()` call |

### 7.4 Anti-Drift Mechanisms

| Drift Type | Prevention |
|-----------|-----------|
| **Compatibility drift** | Version constraints in `engine_compatibility.yaml` with documented incompatibility reasons |
| **Documentation drift** | Validation results written directly into the matrix by the test harness |
| **Summary drift** | `recompute_summary()` on every `save_matrix()` — derived from engine data, never hand-maintained |
| **Environment fingerprint drift** | `--container` flag collects versions from the target container runtime, not the host |
| **Stale error drift** | `error` field auto-cleared when a gate flips from `failed` to `passed` |

### 7.5 History Tracking

Every gate update preserves the previous state in a `history` list (last 5 entries):

```yaml
model_load:
  status: passed
  last_tested: 2026-06-23T05:45:46Z
  duration_seconds: 41.0
  peak_vram_mb: 6420
  history:
    - status: failed
      timestamp: 2026-06-23T05:45:20Z
      error: "OOM at 6.8 GB"
      peak_vram_mb: 6800
```

Current truth and historical audit trail are separated — no mixing.

---

## 8. Phase Progression

```
PHASE 1 — Architecture           COMPLETE
  Dependency boundaries identified
  Container topology designed
  Stack inheritance defined

PHASE 2 — Container Isolation     COMPLETE
  7 Dockerfiles (6 custom + 1 pre-built)
  Tiered inheritance (base → stack → engine)
  All 12 ad-hoc fixes baked as RUN steps
  Build-time validation for qwen3tts

PHASE 3 — Compatibility Matrix    COMPLETE
  engine_compatibility.yaml — single source of truth
  Version constraints with documented incompatibility reasons
  Maturity classification (SUPPORTED/EXPERIMENTAL/BLOCKED)
  Promotion requirements per engine

PHASE 4 — Validation Framework    COMPLETE
  update_engine_status.py — automated matrix updates
  Runtime fingerprinting (with container introspection)
  Gate-level status tracking (with history)
  Automatic summary recomputation
  Promotion enforcement

PHASE 5 — Runtime Evidence        IN PROGRESS
  VibeVoice: 4 gates pending
  Qwen3TTS:  4 gates pending
  ├─ config_load
  ├─ model_load
  ├─ inference / synthesis
  └─ vram_measured
```

---

## 9. Current State → Target State

### 9.1 Current

```
Supported:    16  (engine-current only)
Experimental:  8  (engine-current: 5, engine-mid: 2, engine-qwen: 1)
Blocked:       4  (engine-legacy: 2, orpheus: 1, s2pro: 1)
```

### 9.2 Target (if all 3 experimental engines pass POC)

```
Supported:    19  (engine-current: 16, engine-mid: 2, engine-qwen: 1)
Experimental:  5  (engine-current: 5)
Blocked:       4  (engine-legacy: 2, orpheus: 1, s2pro: 1)
```

No new containers needed. Promotion changes status, not topology.

### 9.3 VRAM Budget (RTX 5060 Ti, 16 GB)

With lazy-load (one engine in VRAM per container at a time):

| Scenario | engine-current | engine-mid | engine-qwen | Total |
|----------|:-------------:|:----------:|:-----------:|:-----:|
| Idle | ~300 MB | ~300 MB | ~300 MB | ~0.9 GB |
| Light (piper + idle + idle) | ~700 MB | ~300 MB | ~300 MB | ~1.3 GB |
| Medium (f5tts + idle + idle) | ~3 GB | ~300 MB | ~300 MB | ~3.6 GB |
| Heavy (bark + idle + idle) | ~12 GB | ~300 MB | ~300 MB | ~12.6 GB |
| Mixed (matcha + VibeVoice + idle) | ~700 MB | ~6.5 GB | ~300 MB | ~7.5 GB |
| Mixed (kokoro + idle + Qwen3TTS) | ~500 MB | ~300 MB | ~3 GB | ~3.8 GB |
| Worst (bark + VibeVoice + Qwen3TTS) | ~12 GB | ~6.5 GB | ~3 GB | **~21.5 GB OOM** |

Only ONE engine-mid engine can be loaded at a time. Only ONE engine across all containers can be loaded simultaneously in practice on 16 GB if any is heavy.

---

## 10. Remaining Gates

### 10.1 VibeVoice (4 gates)

| Gate | Test | Success Criteria |
|------|------|-----------------|
| `config_load` | `AutoConfig.from_pretrained('microsoft/VibeVoice-1.5B', trust_remote_code=True)` | Config loads without error |
| `model_load` | `AutoModel.from_pretrained(...)` | Model loads, no OOM, VRAM ≤ 7 GB |
| `inference` | `model.generate('Hello world.')` | Valid audio output, not silence/noise |
| `vram_measured` | `nvidia-smi` before/after | Peak ≤ 7 GB |

**SGLang fallback:** `lmsysorg/sglang-omni:dev --model microsoft/VibeVoice-1.5B`

### 10.2 Qwen3TTS (4 gates)

| Gate | Test | Success Criteria |
|------|------|-----------------|
| `build_import` | `import qwen_tts` + ROPE check (build-time) | Already passes in Dockerfile |
| `model_load` | `Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')` | Model loads, no OOM, VRAM ≤ 3.5 GB |
| `synthesis` | `generate_custom_voice` or `generate_voice_clone` | Valid audio output |
| `vram_measured` | `nvidia-smi` before/after | Peak ≤ 3.5 GB |

**Gated model:** Requires HF_TOKEN for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

---

## 11. Operational Contracts

### 11.1 Starting Services

```bash
# Default (orchestrator + engine-current + engine-qwen + engine-legacy):
docker compose up -d

# + engine-mid (VibeVoice + Higgs):
docker compose --profile mid up -d

# + Orpheus (GPU mandatory):
docker compose --profile gpu up -d

# + S2-Pro (SGLang, blocked upstream):
docker compose --profile sglang up -d

# Everything:
docker compose --profile mid --profile gpu --profile sglang up -d
```

### 11.2 Engine URLs (Orchestrator → Containers)

| Engine Group | URL Pattern | Container |
|-------------|------------|-----------|
| 21 current engines | `http://engine-current:8101` | engine-current |
| VibeVoice, Higgs | `http://engine-mid:8103` | engine-mid |
| Qwen3TTS | `http://engine-qwen:8104` | engine-qwen |
| IndexTTS, Parler | `http://engine-legacy:8102` | engine-legacy |
| Orpheus | `http://orpheus:8002` | orpheus |
| S2-Pro | `http://s2pro:8000/v1/audio/speech` | s2pro (SGLang) |

### 11.3 Gated Models

| Engine | Model | HF_TOKEN Required |
|--------|-------|:-----------------:|
| qwen3tts | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ✅ Yes |
| orpheus | `canopylabs/orpheus-3b-0.1-ft` | ✅ Yes |
| csm | `sesame/csm-1b` | ✅ Yes (Meta license) |

Set in `.env`:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 11.4 Health Checks

| Container | Port | Start Period | Endpoint |
|-----------|:----:|:------------:|----------|
| orchestrator | 8001 | 15s | `/status` |
| engine-current | 8101 | 180s | `/health` |
| engine-mid | 8103 | 120s | `/health` |
| engine-qwen | 8104 | 120s | `/health` |
| engine-legacy | 8102 | 120s | `/health` |
| orpheus | 8002 | 120s | `/health` |

---

## 12. File Inventory

```
TTS-LAB/
├── docker/
│   ├── Dockerfile.base              ← Tier 1: universal foundation
│   ├── Dockerfile.stack.current     ← Tier 2: torch 2.12 nightly + tf 5.12
│   ├── Dockerfile.stack.mid         ← Tier 2: torch 2.10 stable + tf 4.x
│   ├── Dockerfile.stack.legacy      ← Tier 2: torch 1.13 + tf 4.46
│   ├── Dockerfile.engine-current    ← Tier 3: 21 engines (current stack)
│   ├── Dockerfile.engine-mid        ← Tier 3: 2 engines (mid stack)
│   ├── Dockerfile.engine-qwen       ← Tier 3: 1 engine (mid stack, pinned)
│   ├── Dockerfile.engine-legacy     ← Tier 3: 2 engines (legacy stack)
│   ├── Dockerfile.orpheus           ← Tier 3: 1 engine (vllm, CUDA 12.1)
│   ├── Dockerfile.orchestrator      ← Orchestrator (no ML libs)
│   ├── Dockerfile.base-py311        ← Future: CUDA 13 + Python 3.11 base
│   ├── Dockerfile.stack-py311       ← Future: CUDA 13 + Python 3.11 stack
│   ├── Dockerfile.engine-py311      ← Future: py311 engine container
│   └── Dockerfile.engine-omni       ← Future: omnivoice on latest tf
├── docker-compose.yml               ← 7 services, profiles, volumes, health checks
├── docs/
│   ├── engine_compatibility.yaml    ← Machine-readable single source of truth
│   ├── containerization/
│   │   ├── 01-ARCHITECTURE.md          ← This document (design)
│   │   ├── 02-ENGINE-FIXES.md          ← Engine fixes reference
│   │   ├── 03-METHODOLOGY.md           ← IaC methodology
│   │   ├── 04-ADHOC-LOG.md             ← Ad-hoc deployment log
│   │   ├── 05-STATE-2026-06-21.md      ← Pre-IaC deployment snapshot
│   │   └── archive/                    ← Superseded plans
│   ├── reference/
│   │   └── ARCHITECTURE_REFERENCE.md   ← Deployed-state reference
│   └── issues/
├── scripts/
│   └── update_engine_status.py      ← Validation gate updater + matrix management
├── .github/workflows/
│   ├── build-images.yml             ← CI: builds all 7 images
│   └── deploy.yml                   ← CI: SSH deploy via workflow_dispatch
└── ansible/
    ├── site.yml                     ← Main playbook
    ├── inventory.yml                ← VM connection details
    └── roles/                       ← docker, disk, deploy, monitoring
```
