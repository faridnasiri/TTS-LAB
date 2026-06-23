# TTS Lab — Architecture Reference (2026-06-23)

> **Status:** Deployed & validated — 16/16 supported engines runtime-confirmed
> **Target hardware:** NVIDIA RTX 5060 Ti, 16 GB VRAM, Ubuntu 22.04
> **Git commit:** `f20b6f9`
> **Deployed at:** 192.168.0.87:8001

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Container Topology](#2-container-topology)
3. [Stack Definitions](#3-stack-definitions)
4. [Image Layer Hierarchy](#4-image-layer-hierarchy)
5. [Network Layout](#5-network-layout)
6. [Volume Mounts](#6-volume-mounts)
7. [Engine Distribution](#7-engine-distribution)
8. [Runtime Fingerprints](#8-runtime-fingerprints)
9. [Synthesis Performance](#9-synthesis-performance)
10. [Maturity Classification](#10-maturity-classification)
11. [Docker Compose Profiles](#11-docker-compose-profiles)
12. [Environment Variables](#12-environment-variables)
13. [Health Checks](#13-health-checks)
14. [VRAM Budget](#14-vram-budget)
15. [Deployment Commands](#15-deployment-commands)
16. [File Inventory](#16-file-inventory)

---

## 1. Architecture Overview

### 1.1 Design Principle

Containers are organized by **dependency compatibility boundary**, not engine count. An engine is placed in a container because it shares a torch+transformers+CUDA compatibility domain with other engines — not because it needs isolation.

```
Before (engine-centric):               After (compatibility-domain):

28 containers (one per engine)          7 containers (one per stack)
28 Dockerfiles                          7 Dockerfiles
28 build targets                        7 build targets
Duplicated ML stacks on disk            Base shared once, stacks shared
                                        → ~24 GB total (vs 57 GB ad-hoc)
```

### 1.2 Physical Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        HOST: arthur-server                                │
│                        Ubuntu 22.04, RTX 5060 Ti 16 GB                    │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Docker Bridge: tts-lab-net (172.18.0.0/16)       │  │
│  │                                                                     │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                 │  │
│  │  │ tts-lab-orchestrator │  │ tts-lab-engine-current│                │  │
│  │  │    172.18.0.5        │  │    172.18.0.2         │                 │  │
│  │  │                      │  │                        │                 │  │
│  │  │ Port 8001 (external) │  │ Port 8101 (internal)   │                 │  │
│  │  │                      │  │                        │                 │  │
│  │  │ FastAPI Web UI       │  │ 21 engines             │                 │  │
│  │  │ HTTP dispatch ───────┼──▶ torch 2.14 nightly     │                 │  │
│  │  │ No ML libraries      │  │ tf 5.12.1, CUDA 13.0   │                 │  │
│  │  │ Size: 7.5 GB         │  │ Size: 71.5 GB          │                 │  │
│  │  └──────────────────────┘  └──────────────────────┘                 │  │
│  │                                                                     │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                 │  │
│  │  │ tts-lab-engine-mid   │  │ tts-lab-engine-qwen  │                 │  │
│  │  │    172.18.0.4        │  │    172.18.0.3         │                 │  │
│  │  │                      │  │                        │                 │  │
│  │  │ Port 8103 (internal) │  │ Port 8104 (internal)   │                 │  │
│  │  │                      │  │                        │                 │  │
│  │  │ 2 engines (EXPER.)   │  │ 1 engine (SUPPORTED)   │                 │  │
│  │  │ torch 2.12 nightly   │  │ torch 2.12 nightly     │                 │  │
│  │  │ tf 4.51.3, CUDA 12.8 │  │ tf 4.57.3, CUDA 12.8  │                 │  │
│  │  │ Size: 20.7 GB        │  │ Size: 20.3 GB          │                 │  │
│  │  └──────────────────────┘  └──────────────────────┘                 │  │
│  │                                                                     │  │
│  │                        ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                       │  │
│  │                        │ tts-lab-engine-legacy│  (not deployed)     │  │
│  │                        │   Port 8102         │                     │  │
│  │                        │   2 engines (BLOCKED)│                    │  │
│  │                        │   torch 1.13 + tf 4.46                    │  │
│  │                        └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘                       │  │
│  │                                                                     │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                 │  │
│  │  │ tts-lab-orpheus      │  │ tts-lab-s2pro        │                 │  │
│  │  │   (not deployed)     │  │   (not deployed)     │                 │  │
│  │  │                      │  │                        │                 │  │
│  │  │ Port 8002            │  │ Port 8005              │                 │  │
│  │  │ 1 engine (BLOCKED)   │  │ 1 engine (BLOCKED)     │                 │  │
│  │  │ vllm + CUDA 12.1     │  │ SGLang pre-built       │                 │  │
│  │  │ Profile: gpu         │  │ Profile: sglang        │                 │  │
│  │  └──────────────────────┘  └──────────────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  Shared volumes:                                                          │
│    /opt/models           → /opt/models          (HF cache, ONNX, GGUF)    │
│    /tmp/tts_uploads      → /tmp/tts_uploads     (Reference WAV files)     │
│    /opt/arthur/reference_voices → /opt/arthur/reference_voices            │
│                                                                           │
│  Disk: /opt/models = 137 GB used / 177 GB (33 GB free)                    │
│  GPU:  RTX 5060 Ti = 13014 MiB used / 16311 MiB                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Container Topology

### 2.1 Deployed (4 containers)

| # | Container | Image | Port | GPU | Engines | Size | IP |
|---|-----------|-------|:----:|:---:|:-------:|------|:---:|
| 1 | `tts-lab-orchestrator` | `tts-lab-orchestrator:latest` | **8001** | No | Web UI + HTTP dispatch | 7.5 GB | 172.18.0.5 |
| 2 | `tts-lab-engine-current` | `tts-lab-engine-current:latest` | 8101 | Yes | **21** (15 SUPP + 5 EXPER + 1 probe) | 71.5 GB | 172.18.0.2 |
| 3 | `tts-lab-engine-mid` | `tts-lab-engine-mid:latest` | 8103 | Yes | **2** (VibeVoice, Higgs — EXPER) | 20.7 GB | 172.18.0.4 |
| 4 | `tts-lab-engine-qwen` | `tts-lab-engine-qwen:latest` | 8104 | Yes | **1** (Qwen3TTS — SUPP) | 20.3 GB | 172.18.0.3 |

### 2.2 Not Deployed (3 containers — blocked/deferred)

| # | Container | Port | Profile | Engines | Blocker |
|---|-----------|:----:|:-------:|:-------:|---------|
| 5 | `tts-lab-engine-legacy` | 8102 | `legacy` | 2 (indextts, parler) | Image not built — torch 1.13 + tf 4.46 deferred |
| 6 | `tts-lab-orpheus` | 8002 | `gpu` | 1 (Orpheus 3B) | vllm incompatible with torch nightly |
| 7 | `tts-lab-s2pro` | 8005 | `sglang` | 1 (Fish S2-Pro 5B) | SGLang transformers 5.6.0 too old |

---

## 3. Stack Definitions

### 3.1 Stack Inheritance Tree

```
                         nvidia/cuda:12.8.2-runtime-ubuntu22.04
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              tts-lab-base (7.5 GB)              (future: base-py311)
          System + Python utils + app code            CUDA 13 + Python 3.11
                    │
        ┌───────────┼───────────┐
        │           │           │
   stack-current  stack-mid  stack-legacy
   (16.2 GB)     (19.4 GB)   (not built)
        │           │
   engine-current engine-mid   engine-qwen
   (71.5 GB)     (20.7 GB)    (20.3 GB)
```

### 3.2 Stack Version Matrix (Deployed)

| Stack | torch | torchvision | transformers | CUDA | Python | Driver |
|-------|-------|-------------|:------------:|:----:|:------:|:------:|
| **current** | `2.14.0.dev20260622+cu130` | `0.29.0.dev20260623+cu130` | `5.12.1` | 13.0 | 3.10.12 | 580.159.03 |
| **mid** | `2.12.0.dev20260408+cu128` | — | `4.51.3` | 12.8 | 3.10.12 | 580.159.03 |
| **qwen** | `2.12.0.dev20260408+cu128` | — | `4.57.3` | 12.8 | 3.10.12 | 580.159.03 |

**Note:** All stacks use torch nightly — required for Blackwell sm_120 (RTX 5060 Ti). The "mid" designation refers to transformers version (4.x vs 5.x), not torch stability. engine-mid and engine-qwen are on an older cu128 build; a future rebuild cycle will unify all three on cu130.

### 3.3 Stack Content Detail

**stack-current** (16.2 GB)
- torch 2.14 nightly + torchvision 0.29 nightly + torchaudio (cu130)
- transformers 5.12.1, accelerate, onnxruntime, safetensors, numpy, protobuf
- torchcodec metadata stub
- Compatibility patches: transformers stubs, parler-tts patch

**stack-mid** (19.4 GB)
- torch 2.12 nightly + torchaudio (cu128)
- transformers 4.51.3, accelerate, onnxruntime, safetensors, numpy, protobuf
- No compatibility patches needed (transformers 4.x APIs are stable)

---

## 4. Image Layer Hierarchy

### 4.1 Tiered Inheritance (Shared Base)

```
Tier 1: tts-lab-base  (7.5 GB — stored ONCE, shared by ALL containers)
├── FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04
├── System: espeak-ng, ffmpeg, MeCab, python3.10, python3-dev, git, wget, curl
├── Python: fastapi, uvicorn, httpx, soundfile, huggingface_hub, psutil, requests
├── NLTK:   punkt, punkt_tab, cmudict, averaged_perceptron_tagger
├── App:    tts_lab.py, tts_lab_shims.py, tts_lab_config.py,
│           tts_lab_engines.py, tts_lab_dispatch.py, tts_lab_ui.py,
│           tts_lab_utils.py, tts_lab_engine_server.py
├── Symlink: /opt/arthur/models → /opt/models/tts
└── Env:     COQUI_TOS_AGREED=1, HF_HOME=/opt/models/huggingface, etc.

    ├── Tier 2: stack-current (+8.7 GB → 16.2 GB)
    │   torch nightly cu130, transformers 5.12.1, accelerate, onnxruntime
    │   Used by: engine-current
    │
    ├── Tier 2: stack-mid (+11.9 GB → 19.4 GB)
    │   torch nightly cu128, transformers 4.51.3
    │   Used by: engine-mid, engine-qwen
    │
    └── Tier 2: stack-legacy (not built)
        torch 1.13 cu117, transformers 4.46
        Used by: engine-legacy (deferred)

Tier 3: Engine Images (thin additions on top of stack)
├── engine-current (+55.3 GB → 71.5 GB)
│   21 engines + MeCab/unidic + zonos backbone + CSM clone
│   + python3-dev + torchcodec stub + numpy<2.0 pin
│
├── engine-mid (+1.3 GB → 20.7 GB)
│   vibevoice package + scipy/librosa/soundfile
│
├── engine-qwen (+0.9 GB → 20.3 GB)
│   qwen-tts 0.1.1 + hf-hub pin + scipy/librosa/soundfile
│
└── engine-legacy (not built)
    indextts + parler-tts + qwen-tts (torch 1.13)
```

### 4.2 Total Disk Usage

| Category | Size |
|----------|------|
| Docker images (all) | ~123 GB |
| /opt/models (HF cache, ONNX, GGUF) | 137 GB |
| Docker build cache | ~50 GB |
| **Total deployment** | **~310 GB** |

Note that shared base layers are stored once on disk. The engine-current image (71.5 GB) includes ~55 GB of engine packages on top of stack-current (16.2 GB). The effective unique disk for engine-current is ~55 GB, not 71.5 GB.

---

## 5. Network Layout

### 5.1 Bridge Network: `tts-lab-net`

```
                         Host: 192.168.0.87
                              │
                    ┌─────────┴─────────┐
                    │  Docker Bridge    │
                    │  172.18.0.0/16    │
                    │                   │
         ┌──────────┼──────────┬────────┼──────────┐
         │          │          │        │          │
    172.18.0.5  172.18.0.2  172.18.0.3  172.18.0.4  (future: .6, .7)
    orchestrator engine-     engine-     engine-
                current      qwen        mid
    Port 8001    Port 8101   Port 8104   Port 8103
    (external)   (internal)  (internal)  (internal)
```

### 5.2 Orchestrator Routing

| Engine Group | URL Pattern | Container Target |
|-------------|------------|-----------------|
| 21 current engines | `http://engine-current:8101` | `tts-lab-engine-current` |
| VibeVoice, Higgs | `http://engine-mid:8103` | `tts-lab-engine-mid` |
| Qwen3TTS | `http://engine-qwen:8104` | `tts-lab-engine-qwen` |
| IndexTTS, Parler | `http://engine-legacy:8102` | `tts-lab-engine-legacy` (not deployed) |
| Orpheus | `http://orpheus:8002` | `tts-lab-orpheus` (not deployed) |
| S2-Pro | `http://s2pro:8000/v1/audio/speech` | `tts-lab-s2pro` (not deployed) |

### 5.3 API Flow

```
Browser (192.168.0.87:8001)
    │
    ▼
orchestrator:/synthesize/{engine}
    │
    ├── tts_lab_dispatch._do_synth_remote()
    │
    ├── HTTP POST → engine-current:8101/synthesize   (15 engines)
    ├── HTTP POST → engine-qwen:8104/synthesize       (qwen3tts)
    ├── HTTP POST → engine-mid:8103/synthesize        (future: vibevoice, higgs)
    ├── HTTP POST → s2pro:8000/v1/audio/speech        (future: SGLang API)
    │
    ▼
engine server (lazy-load)
    ├── _ensure_loaded(name) → evict current → load new → synthesize
    └── Response: {audio_b64, sample_rate, synth_time_ms, audio_dur_ms, rtf}
```

---

## 6. Volume Mounts

All engine containers and the orchestrator share these volumes:

| Host Path | Container Path | Contents | Size |
|-----------|---------------|----------|------|
| `/opt/models` | `/opt/models` | HuggingFace cache, ONNX models, GGUF files, TTS voices | 137 GB |
| `/tmp/tts_uploads` | `/tmp/tts_uploads` | Reference WAV files for voice cloning | Variable |
| `/opt/arthur/reference_voices` | `/opt/arthur/reference_voices` | Additional reference voices | Variable |

**Key paths inside /opt/models:**
| Path | Contents |
|------|----------|
| `/opt/models/huggingface/` | All HF model caches (89 GB) |
| `/opt/models/cache/` | Pip, suno, whisper caches (33 GB) |
| `/opt/models/tts/` | Piper/Kokoro ONNX files (641 MB) |
| `/opt/models/CosyVoice/` | CosyVoice2 pretrained models (4.6 GB) |
| `/opt/models/tts_coqui/` | Coqui TTS models (1.8 GB) |
| `/opt/models/outetts-gguf/` | OuteTTS GGUF files (1.5 GB) |
| `/opt/models/csm/` | CSM source clone (340 KB) |

---

## 7. Engine Distribution

### 7.1 Per-Container Engine Map

```
engine-current (21 engines — torch 2.14 nightly + tf 5.12.1, CUDA 13.0)
├── SUPPORTED (15)
│   ├── piper           ONNX CPU, RTF 3.7×
│   ├── kokoro          ONNX, RTF 6.5×
│   ├── melo            Needs MeCab+unidic, RTF 9.8×
│   ├── matcha          ONNX flow-matching, RTF 8.0×
│   ├── chattts         RTF 11.5×
│   ├── outetts         LLM-based, RTF 34.9×
│   ├── bark            Heavy VRAM, RTF 19.0×
│   ├── styletts2       Needs langchain<0.3.0, RTF 35.8×
│   ├── f5tts           Voice cloning, RTF 13.4×
│   ├── chatterbox      AR+diffusion, RTF 25.2×
│   ├── chatterboxturbo One-step distilled, RTF 13.6×
│   ├── fishspeech      Voice cloning, RTF 9.4×
│   ├── omnivoice       600+ languages, RTF 4.5×
│   ├── zonos           Voice cloning, RTF 19.0×
│   └── xtts            RTF 44.7×
├── EXPERIMENTAL (5)
│   ├── cosyvoice       Needs model download + hyperpyyaml
│   ├── csm             Meta license gated
│   ├── manatts         Persian TTS, not configured
│   ├── neutts          Not configured
│   └── openvoice       Checkpoints not downloaded
└── Probe only (1)
    └── orpheus         Runs in separate container (blocked)

engine-mid (2 engines — torch 2.12 nightly + tf 4.51.3, CUDA 12.8)
├── EXPERIMENTAL (2)
│   ├── vibevoice       BLOCKED at config_load (vibevoice not in CONFIG_MAPPING)
│   └── higgs           Not yet tested

engine-qwen (1 engine — torch 2.12 nightly + tf 4.57.3, CUDA 12.8)
└── SUPPORTED (1)
    └── qwen3tts        RTF 4.5×, TransformGetItemToIndex patched

engine-legacy (not deployed)
└── BLOCKED (2)
    ├── indextts        Needs torch 1.13 + tf 4.46
    └── parler          Needs torch 1.13 + tf 4.46

orpheus (not deployed)
└── BLOCKED (1)
    └── orpheus         vllm incompatible with torch nightly

s2pro (not deployed)
└── BLOCKED (1)
    └── s2pro           SGLang transformers 5.6.0 too old
```

### 7.2 Engine Counts

| Container | Total | Supported | Experimental | Blocked | Deployed |
|-----------|:-----:|:---------:|:------------:|:-------:|:--------:|
| `engine-current` | **21** | 15 | 5 | 0 | ✅ |
| `engine-mid` | **2** | 0 | 2 | 0 | ✅ |
| `engine-qwen` | **1** | 1 | 0 | 0 | ✅ |
| `engine-legacy` | **2** | 0 | 0 | 2 | ❌ |
| `orpheus` | **1** | 0 | 0 | 1 | ❌ |
| `s2pro` | **1** | 0 | 0 | 1 | ❌ |
| **Total** | **28** | **16** | **7** | **4** | 4/7 |

---

## 8. Runtime Fingerprints

### 8.1 Deployed Containers

| Container | torch | torchvision | transformers | CUDA | Python | Driver |
|-----------|-------|-------------|:------------:|:----:|:------:|:------:|
| `tts-lab-engine-current` | `2.14.0.dev20260622+cu130` | `0.29.0.dev20260623+cu130` | `5.12.1` | 13.0 | 3.10.12 | 580.159.03 |
| `tts-lab-engine-mid` | `2.12.0.dev20260408+cu128` | — | `4.51.3` | 12.8 | 3.10.12 | 580.159.03 |
| `tts-lab-engine-qwen` | `2.12.0.dev20260408+cu128` | — | `4.57.3` | 12.8 | 3.10.12 | 580.159.03 |
| `tts-lab-orchestrator` | — | — | — | — | 3.10.12 | — |

### 8.2 Key Engine Pins

| Engine | Dependency | Constraint | Reason |
|--------|-----------|:----------:|--------|
| qwen3tts | transformers | `==4.57.3` (qwen_tts metadata) | Patched: TransformGetItemToIndex stub |
| qwen3tts | huggingface-hub | `<1.0` | is_offline_mode removed in 1.0 |
| styletts2 | langchain | `<0.3.0` | text_splitter removed in 1.x |
| f5tts | hf-hub | `>=1.0` | is_offline_mode restored |
| xtts, f5tts | torchcodec | Stub module | Not available for torch nightly |
| vibevoice | transformers | `>=5.x` or registration | Architecture not in CONFIG_MAPPING |

---

## 9. Synthesis Performance

### 9.1 All 16 Working Engines — Measured on RTX 5060 Ti

| # | Engine | Container | RTF | Audio Dur | Latency | sr | Tier |
|---|--------|-----------|:---:|----------:|--------:|:----:|------|
| 1 | piper | current | 3.7× | 743ms | 3s | 22050 | ⚡ Near real-time |
| 2 | kokoro | current | 6.5× | 1344ms | 9s | 24000 | Fast |
| 3 | melo | current | 9.8× | 1893ms | 19s | 44100 | Fast |
| 4 | matcha | current | 8.0× | 1183ms | 9s | 22050 | Fast |
| 5 | chattts | current | 11.5× | 902ms | 10s | 24000 | Fast |
| 6 | outetts | current | 34.9× | 1293ms | 45s | 24000 | 🐌 Slow |
| 7 | bark | current | 19.0× | 2653ms | 50s | 24000 | Moderate |
| 8 | styletts2 | current | 35.8× | 1772ms | 63s | 24000 | 🐌 Slow |
| 9 | f5tts | current | 13.4× | 682ms | 9s | 24000 | Fast |
| 10 | chatterbox | current | 25.2× | 840ms | 21s | 24000 | Moderate |
| 11 | chatterboxturbo | current | 13.6× | 1240ms | 17s | 24000 | Fast |
| 12 | fishspeech | current | 9.4× | 2972ms | 28s | 44100 | Fast |
| 13 | omnivoice | current | 4.5× | 1640ms | 7s | 24000 | ⚡ Near real-time |
| 14 | zonos | current | 19.0× | 1532ms | 29s | 24000 | Moderate |
| 15 | xtts | current | 44.7× | 1387ms | 62s | 24000 | 🐌 Slow |
| 16 | qwen3tts | qwen | 4.7× | 800ms | 4s | 24000 | ⚡ Near real-time |

**Test text:** "Hello world." (short prompt)
**Test method:** HTTP POST to orchestrator:8001/synthesize/{engine}

### 9.2 RTF Distribution

| Tier | RTF Range | Count | % | Engines |
|------|:---------:|:-----:|:---:|---------|
| Real-time | < 1.0× | 0 | 0% | — |
| Near real-time | 1.0–5.0× | 3 | 19% | piper, omnivoice, qwen3tts |
| Fast | 5.0–15.0× | 7 | 44% | kokoro, melo, matcha, chattts, f5tts, chatterboxturbo, fishspeech |
| Moderate | 15.0–30.0× | 3 | 19% | bark, chatterbox, zonos |
| Slow | 30.0–50.0× | 3 | 19% | outetts, styletts2, xtts |

### 9.3 Latency Distribution

| Latency | Count | Engines |
|---------|:-----:|---------|
| < 5s | 2 | piper, qwen3tts |
| 5–15s | 6 | kokoro, matcha, chattts, f5tts, omnivoice |
| 15–30s | 4 | melo, chatterbox, chatterboxturbo, fishspeech |
| 30–60s | 3 | outetts, bark, zonos |
| 60s+ | 1 | styletts2 |

---

## 10. Maturity Classification

### 10.1 States

| State | Icon | Meaning | CI: Build | CI: Smoke Test | Required |
|-------|:----:|---------|:---------:|:--------------:|:--------:|
| **SUPPORTED** | ✅ | Synthesis confirmed on target hardware | Yes | Yes | Pass |
| **DEPRECATED** | ⚠ | Still works, no longer recommended | Yes | Yes | Warn |
| **EXPERIMENTAL** | 🧪 | Container defined, not yet validated | Yes | Best effort | Warn |
| **BLOCKED** | ❌ | Cannot work — upstream missing/incompatible | No | No | Skip |

### 10.2 Lifecycle

```
EXPERIMENTAL  ──(all promotion gates pass)──▶  SUPPORTED
EXPERIMENTAL  ──(blocker found)─────────────▶  BLOCKED
BLOCKED       ──(upstream releases fix)─────▶  EXPERIMENTAL
SUPPORTED     ──(superseded by better)──────▶  DEPRECATED
DEPRECATED    ──(eventually breaks)─────────▶  BLOCKED
```

### 10.3 Current State

```
Supported:    16  (piper, kokoro, melo, matcha, chattts, outetts, bark,
                   styletts2, f5tts, chatterbox, chatterboxturbo,
                   fishspeech, omnivoice, zonos, xtts, qwen3tts)
Experimental:  7  (cosyvoice, csm, manatts, neutts, openvoice, vibevoice, higgs)
Blocked:       4  (orpheus, indextts, parler, s2pro)
Deprecated:    0
Total:        28
```

### 10.4 Promotion Gates

| Engine | Gates Required | Status |
|--------|---------------|:------:|
| qwen3tts | build_import, model_load, synthesis, vram_measured | ✅ All passed |
| vibevoice | config_load, model_load, inference, vram_measured | ❌ config_load failed |
| higgs | config_load, model_load, inference, vram_measured | Pending |
| cosyvoice | (not defined) | — |
| csm | (not defined) | — |

---

## 11. Docker Compose Profiles

### 11.1 Profile Map

| Profile | Containers | Engines | Command |
|:-------:|------------|---------|---------|
| *(default)* | orchestrator, engine-current, engine-qwen | 16 SUPP + 5 EXPER + 1 SUPP | `docker compose up -d` |
| `mid` | + engine-mid | + 2 EXPER (VibeVoice, Higgs) | `docker compose --profile mid up -d` |
| `gpu` | + orpheus | + 1 BLOCKED | `docker compose --profile gpu up -d` |
| `sglang` | + s2pro, vibevoice-sglang, higgs-sglang | + 3 (BLOCKED) | `docker compose --profile sglang up -d` |
| `legacy` | + engine-legacy | + 2 BLOCKED | `docker compose --profile legacy up -d` |
| *(all)* | All 7 | All 28 | `docker compose --profile mid --profile gpu --profile sglang --profile legacy up -d` |

### 11.2 Startup Order

```
Default start:
  engine-current ──(healthy)──┐
  engine-qwen    ──(healthy)──┼── orchestrator ──(port 8001)── ready
                              │
With --profile mid:           │
  engine-mid     ──(healthy)──┘
```

---

## 12. Environment Variables

### 12.1 Common (all containers)

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/opt/models/huggingface` | HuggingFace cache directory |
| `XDG_CACHE_HOME` | `/opt/models/cache` | Pip/suno/whisper caches |
| `COQUI_TOS_AGREED` | `1` | Accept Coqui TTS license |
| `TOKENIZERS_PARALLELISM` | `false` | Prevent tokenizer deadlocks |
| `PYTHONUNBUFFERED` | `1` | Real-time log output |

### 12.2 Container-Specific

| Container | Variable | Purpose |
|-----------|----------|---------|
| orchestrator | `ORCHESTRATOR_MODE=1` | Skip ML imports, HTTP-dispatch only |
| orchestrator | `PIPER_URL=http://engine-current:8101` | Engine routing |
| orchestrator | `QWEN3TTS_URL=http://engine-qwen:8104` | Dedicated container routing |
| orchestrator | `S2PRO_SGLANG_URL=http://s2pro:8000/v1/audio/speech` | SGLang API path |
| engine-current | `SUNO_USE_SMALL_MODELS=False` | Full Bark model |
| engine-qwen | `HF_TOKEN=${HF_TOKEN:-}` | Gated model access |
| orpheus | `HF_TOKEN=${HF_TOKEN:-}` | Gated model access |

### 12.3 .env File (gitignored)

```bash
# Required for gated models (qwen3tts, orpheus, csm)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## 13. Health Checks

| Container | Port | Endpoint | Interval | Timeout | Start Period |
|-----------|:----:|----------|:--------:|:-------:|:------------:|
| orchestrator | 8001 | `/status` | 30s | 10s | 15s |
| engine-current | 8101 | `/health` | 30s | 10s | 180s |
| engine-mid | 8103 | `/health` | 30s | 10s | 120s |
| engine-qwen | 8104 | `/health` | 30s | 10s | 120s |
| engine-legacy | 8102 | `/health` | 30s | 10s | 120s |
| orpheus | 8002 | `/health` | 30s | 10s | 120s |

### 13.1 Health Response Format (engine servers)

```json
{
    "status": "ok",
    "stack": "current",
    "engines_available": 21,
    "engines_loaded": 0,
    "current_engine": null,
    "engines": {
        "piper": {"loaded": false, "load_time_s": 0},
        ...
    },
    "gpu": {
        "name": "NVIDIA GeForce RTX 5060 Ti",
        "total_mb": 16311,
        "used_mb": 404
    }
}
```

---

## 14. VRAM Budget

### 14.1 Lazy-Load Engine Server

The engine server starts with **0 engines loaded**. On synthesis:
1. Evict the currently loaded engine (clear `_state`, `gc.collect()`, `torch.cuda.empty_cache()`)
2. Load the requested engine
3. Synthesize
4. Keep engine loaded (lazy — don't evict until another synthesis needs the VRAM)

### 14.2 Per-Engine VRAM (Measured)

| Engine | VRAM | Can co-reside with? |
|--------|:----:|---------------------|
| piper | ~200 MB | Anything |
| kokoro | ~200 MB | Anything |
| matcha | ~200 MB | Anything |
| melo | ~1.4 GB | All except bark |
| chattts | ~2 GB | All except bark |
| styletts2 | ~1.5 GB | All except bark |
| f5tts | ~3 GB | Light engines |
| fishspeech | ~3 GB | Light engines |
| omnivoice | ~2 GB | All except bark |
| zonos | ~3 GB | Light engines |
| outetts | ~2 GB | All except bark |
| chatterboxturbo | ~1.5 GB | All except bark |
| chatterbox | ~2 GB | All except bark |
| xtts | ~2 GB | All except bark |
| dia | ~4 GB | Light engines only |
| bark | ~12 GB | Alone |
| qwen3tts | ~4.2 GB | Light engines only |

### 14.3 Budget Scenarios (16 GB Total)

| Scenario | VRAM Used | Free | Viable? |
|----------|:---------:|:----:|:--------:|
| Idle (4 containers, 0 engines) | ~1.2 GB | 14.8 GB | ✅ |
| piper + idle + idle | ~1.4 GB | 14.6 GB | ✅ |
| f5tts + idle + idle | ~4.2 GB | 11.8 GB | ✅ |
| bark + idle + idle | ~13 GB | 3 GB | ✅ (tight) |
| matcha + qwen3tts + idle | ~4.6 GB | 11.4 GB | ✅ |
| piper + VibeVoice + idle | ~6.7 GB | 9.3 GB | ✅ (if VibeVoice ≤ 6.5 GB) |
| bark + qwen3tts | ~17 GB | OOM | ❌ |
| bark + VibeVoice | ~19 GB | OOM | ❌ |

---

## 15. Deployment Commands

### 15.1 First-Time Build

```bash
# Tier 1 — base (shared, build once):
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .

# Tier 2 — stacks:
docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current:latest .
docker build -f docker/Dockerfile.stack.mid -t tts-lab-stack-mid:latest .

# Tier 3 — engines:
docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
docker build -f docker/Dockerfile.engine-mid -t tts-lab-engine-mid:latest .
docker build -f docker/Dockerfile.engine-qwen -t tts-lab-engine-qwen:latest .

# Orchestrator:
docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .

# Total build time: ~35 minutes (mostly pip installs)
# Subsequent rebuilds: ~5–30 seconds (cached layers)
```

### 15.2 Start Services

```bash
# Default (orchestrator + engine-current + engine-qwen):
docker compose up -d

# + engine-mid (VibeVoice + Higgs):
docker compose --profile mid up -d

# + GPU-only engines:
docker compose --profile gpu up -d

# + SGLang engines:
docker compose --profile sglang up -d

# Everything:
docker compose --profile mid --profile gpu --profile sglang --profile legacy up -d
```

### 15.3 Maintenance

```bash
# Check status:
docker compose ps
docker compose logs -f --tail 20

# Restart a misbehaving engine:
docker compose restart engine-current

# Rebuild and update engine-current:
docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
docker compose up -d --force-recreate engine-current

# Evict loaded model (free VRAM):
curl -X POST http://localhost:8101/unload

# Health check:
curl http://localhost:8001/status

# Synthesis test:
curl -X POST http://localhost:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world.","params":{}}' -o /tmp/test.wav
```

### 15.4 Validation

```bash
# Check engine fingerprints:
python3 scripts/update_engine_status.py --fingerprint --container tts-lab-engine-current
python3 scripts/update_engine_status.py --fingerprint --container tts-lab-engine-mid
python3 scripts/update_engine_status.py --fingerprint --container tts-lab-engine-qwen

# Check promotion eligibility:
python3 scripts/update_engine_status.py qwen3tts --check

# Recompute summary:
python3 scripts/update_engine_status.py --recompute
```

---

## 16. File Inventory

```
TTS-LAB/
├── docker/
│   ├── Dockerfile.base                 Tier 1: universal foundation (7.5 GB)
│   ├── Dockerfile.stack.current        Tier 2: torch nightly + tf 5.12 (16.2 GB)
│   ├── Dockerfile.stack.mid            Tier 2: torch nightly + tf 4.x (19.4 GB)
│   ├── Dockerfile.stack.legacy         Tier 2: torch 1.13 + tf 4.46 (not built)
│   ├── Dockerfile.engine-current       Tier 3: 21 engines (71.5 GB)
│   ├── Dockerfile.engine-mid           Tier 3: 2 engines (20.7 GB)
│   ├── Dockerfile.engine-qwen          Tier 3: 1 engine (20.3 GB)
│   ├── Dockerfile.engine-legacy        Tier 3: 2 engines (not built)
│   ├── Dockerfile.orpheus              Tier 3: vllm + CUDA 12.1 (not built)
│   ├── Dockerfile.orchestrator         Orchestrator: no ML libs (7.5 GB)
│   ├── Dockerfile.base-py311           Future: CUDA 13 + Python 3.11 base
│   ├── Dockerfile.stack-py311          Future: CUDA 13 + Python 3.11 stack
│   ├── Dockerfile.engine-py311         Future: py311 engine container
│   └── Dockerfile.engine-omni          Future: omnivoice on latest tf
├── docker-compose.yml                  7 services, profiles, volumes, health checks
├── docs/
│   ├── engine_compatibility.yaml       Machine-readable single source of truth
│   ├── reference/
│   │   └── ARCHITECTURE_REFERENCE.md   This document
│   ├── containerization/
│   │   ├── 01-ARCHITECTURE.md          Architecture design document
│   │   ├── 02-ENGINE-FIXES.md          Engine fixes reference
│   │   ├── 03-METHODOLOGY.md           IaC methodology & lessons
│   │   ├── 04-ADHOC-LOG.md             Ad-hoc deployment day-by-day log
│   │   ├── 05-STATE-2026-06-21.md      Pre-IaC deployment snapshot
│   │   └── archive/                    Superseded plans (4 docs)
│   └── issues/
│       ├── chattts-encode-prompt-decode-bug.md
│       └── deployment-fixes-2026-06-23.md
├── scripts/
│   ├── update_engine_status.py          Validation gate updater
│   └── elevenlabs_persian_batch.py
├── tts_lab.py                           Orchestrator FastAPI
├── tts_lab_engine_server.py             Engine server (lazy-load)
├── tts_lab_engines.py                   Loaders + synthesizers (28 engines)
├── tts_lab_dispatch.py                  Availability + dispatch logic
├── tts_lab_shims.py                     Compatibility patches
├── tts_lab_config.py                    Engine config + constants
├── tts_lab_ui.py                        Web UI
├── tts_lab_utils.py                     Audio utilities
├── patches/                             Build-time compatibility patches
├── .github/workflows/
│   ├── build-images.yml                 CI: builds all 7 images
│   └── deploy.yml                       CI: SSH deploy via workflow_dispatch
└── ansible/
    ├── site.yml                         Main playbook
    ├── inventory.yml                    VM connection details
    └── roles/                           docker, disk, deploy, monitoring
```

---

## 17. Quick Reference

| Question | Answer |
|----------|--------|
| **Target hardware** | RTX 5060 Ti, 16 GB VRAM, Ubuntu 22.04 |
| **VM address** | 192.168.0.87 |
| **Web UI** | http://192.168.0.87:8001 |
| **Containers deployed** | 4 of 7 |
| **Engines working** | 16 of 28 (all SUPPORTED) |
| **Total image size** | ~123 GB (all images) |
| **Model cache** | 137 GB (/opt/models) |
| **Free disk** | 33 GB (/opt/models), 59 GB (root) |
| **Torch version** | 2.14 nightly cu130 (current), 2.12 nightly cu128 (mid/qwen) |
| **CUDA version** | 13.0 (current), 12.8 (mid/qwen) |
| **Driver** | 580.159.03 |
| **Transformers (current)** | 5.12.1 |
| **Transformers (mid)** | 4.51.3 |
| **Transformers (qwen)** | 4.57.3 |
| **Network** | Bridge: tts-lab-net (172.18.0.0/16) |
| **Best RTF** | piper 3.7×, omnivoice 4.5×, qwen3tts 4.7× |
| **Worst RTF** | xtts 44.7×, styletts2 35.8×, outetts 34.9× |
| **Image digests** | See [§18 Image Digests](#18-image-digests) |

---

## 18. Image Digests (2026-06-23)

Reproducible image references — torch nightly is not reproducible, but image digests are.

| Image | Digest (sha256) |
|-------|-----------------|
| `tts-lab-base:latest` | `5e0730e0f5eb8dd264084f6cf9b3f5c50bed779d5bda294d0aa75081f2a0e66e` |
| `tts-lab-stack-current:latest` | `4e9cad18a53ecee5cfab04c4444eaa2e24562fac16b8189ad01f408468cf8572` |
| `tts-lab-stack-mid:latest` | `d6e3fda2397897558dfba6660c456d61a17993c9e7f1d420207e2500d323950b` |
| `tts-lab-engine-current:latest` | `967f76c5b7ed4721a1af4185791c4784057718775e2469466df9f42d40713482` |
| `tts-lab-engine-mid:latest` | `8aca47588a3003de6aa3e5fe45abb90b504c0f7d6c925af03b0893d69a5fc114` |
| `tts-lab-engine-qwen:latest` | `cad1765179db1e1644e6f6937cda6c1600a58a17beb8b63d23e3dad8244e3713` |
| `tts-lab-orchestrator:latest` | `8d4329477074aa7adcedc83517f465769ba9fabd98f4d54ef6154b04db3e20aa` |

## 19. Disk Monitoring

**Current usage (2026-06-23):**

| Path | Used | Total | Free |
|------|------|-------|------|
| `/opt/models` (data disk) | 137 GB | 177 GB | **33 GB** |
| `/` (root, Docker images) | 572 GB | 630 GB | 59 GB |

**First resource to exhaust:** `/opt/models` — 33 GB free with active model downloads.
Docker build cache adds ~50 GB to root disk after image builds.

**Cleanup commands:**
```bash
# Reclaim Docker build cache (> 30 days old):
docker builder prune --filter "until=720h" --force

# Reclaim unused Docker images:
docker image prune -a --force

# Full cleanup (aggressive):
docker system prune -a --volumes --force

# Check largest model directories:
du -sh /opt/models/huggingface/hub/models--* | sort -rh | head -10
```
| **Fastest response** | piper 3s, qwen3tts 4s |
| **Slowest response** | styletts2 63s, xtts 62s |
