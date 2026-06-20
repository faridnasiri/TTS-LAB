# TTS Lab Containerization — Ad-Hoc Deployment Reference

> **Audience:** Engineers and developers with zero to minimal Docker/container/AI background.
> **Purpose:** Reference for understanding the architecture, lessons learned during ad-hoc deployment, and preparing for IaC rewrite.
> **Date:** 2026-06-19–20
> **Status:** Ad-hoc deployment completed, 11 engines serving. IaC rewrite pending.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Concepts Explained (for Beginners)](#2-concepts-explained-for-beginners)
3. [Architecture](#3-architecture)
4. [File Reference](#4-file-reference)
5. [Build Process — Step by Step](#5-build-process--step-by-step)
6. [Lessons Learned (Ad-Hoc → IaC)](#6-lessons-learned-ad-hoc--iac)
7. [Current Status](#7-current-status)
8. [IaC Rewrite Plan](#8-iac-rewrite-plan)
9. [Glossary](#9-glossary)
10. [Command Cheatsheet](#10-command-cheatsheet)

---

## 1. What We Built

We containerized a 28-engine Text-to-Speech laboratory. Instead of one fragile Python virtual environment where 28 engines fight over incompatible library versions, we have:

- **6 Docker containers** running on a single Ubuntu VM
- **11 TTS engines confirmed working** (more pending minor fixes)
- **All Dockerfiles committed to GitHub** with CI/CD pipeline
- **Health check + HTTP API** for every engine
- **Shared model storage** — 38 GB of AI models, never duplicated

### The Core Problem We Solved

```
Before (bare metal):              After (containers):
─────────────────────              ──────────────────
One Python 3.11 venv              Container A: torch 2.10 + tf 5.12 (21 engines)
  ├── torch 2.10                   Container B: torch 1.13 + tf 4.46 (3 engines)
  ├── transformers 5.12            Container C: CUDA 12.1 + vllm (1 engine)
  ├── numpy (conflict!)            Containers D/E/F: SGLang servers
  ├── protobuf (conflict!)         
  ├── 28 engine packages          Each container has its own isolated
  └── patches on patches            environment. No conflicts possible.
```

---

## 2. Concepts Explained (for Beginners)

### What is Docker?

Docker is a tool that packages software into **containers** — lightweight, isolated environments that include everything an application needs to run: code, libraries, system tools, and settings.

**Think of it like this:** A Python virtual environment isolates Python packages. A Docker container isolates the ENTIRE system — Python version, C libraries, GPU drivers, everything. It's like giving each application its own mini-computer.

### Key Terms

| Term | Simple Explanation |
|------|-------------------|
| **Image** | A blueprint. Like a `.iso` file — contains an OS + software. Read-only. |
| **Container** | A running instance of an image. Like a VM but shares the host kernel. |
| **Dockerfile** | A recipe that defines how to build an image. Each line creates a "layer." |
| **Layer** | One step in a Dockerfile. Layers are CACHED — if a layer hasn't changed, Docker reuses it. |
| **FROM** | Inherits from a parent image. This is how we build on top of shared bases. |
| **Volume** | A directory on the host shared with the container. Models live here — not in images. |
| **Registry** | A place to store images. GitHub Container Registry (GHCR) is free for public repos. |
| **Docker Compose** | Runs multiple containers together. One `docker compose up -d` starts everything. |
| **Healthcheck** | A command Docker runs to verify a container is healthy. `curl /health` in our case. |

### Why Layers Matter

This is the most important concept for understanding our architecture:

```dockerfile
FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04   # Layer 1: CUDA base (2 GB, shared by ALL)
RUN apt-get install -y espeak-ng ffmpeg git    # Layer 2: System tools (300 MB)
RUN pip install fastapi uvicorn httpx           # Layer 3: Python utils (200 MB)
COPY tts_lab.py /opt/arthur/                   # Layer 4: App code (10 MB)
```

- Layer 1 is stored ONCE, no matter how many images use it
- When you rebuild and only Layer 4 changes, only Layer 4 rebuilds (5 seconds)
- When Layer 1 changes (CUDA update), everything rebuilds (30 minutes)

**Our tiered architecture leverages this:** 2 stack images (torch + transformers) are shared by all 28 engine containers, but stored once on disk.

### GPU vs CPU in Docker

- **CUDA (GPU):** NVIDIA's parallel computing platform. Required for fast AI inference.
- **nvidia-container-toolkit:** A Docker plugin that lets containers access the host's GPU.
- **`--gpus all`:** Docker flag to give a container GPU access.
- **VRAM:** Video RAM on the GPU. Our RTX 5060 Ti has 16 GB. Large AI models (Bark, Orpheus) consume 6-12 GB each.
- **sm_120 / Blackwell:** The GPU architecture of RTX 5060 Ti. Older PyTorch versions don't support it — we needed the nightly build.

### What is a TTS Engine?

A Text-to-Speech engine is an AI model that converts text to audio. Each engine is a different research project with different:
- Model architecture (Transformer, Diffusion, ONNX, etc.)
- Python dependencies (torch, transformers, numpy versions)
- System requirements (espeak-ng for phonemes, ffmpeg for audio)
- License (MIT, Apache-2.0, gated on HuggingFace)

---

## 3. Architecture

### Container Map

```
┌──────────────────────────────────────────────────────────────┐
│               ORCHESTRATOR (port 8001)                       │
│         Web UI + HTTP dispatch. No ML libraries.             │
│         Built FROM tts-lab-base directly.                    │
│         ~1.5 GB image.                                       │
└──────────┬──────────┬──────────┬────────────────────────────┘
           │ HTTP     │ HTTP     │ HTTP
           ▼          ▼          ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────────────────┐
│ ENGINE-      │ │ ENGINE-  │ │ ORPHEUS (GPU mandatory)      │
│ CURRENT      │ │ LEGACY   │ │                              │
│ (port 8101)  │ │ (8102)   │ │ FROM nvidia/cuda:12.1.0      │
│              │ │          │ │ + vllm + orpheus-speech       │
│ torch nightly│ │torch 1.13│ │ ~7 GB image, ~6 GB VRAM      │
│ tf 5.12.1    │ │tf 4.46.1 │ │ Port 8002                    │
│              │ │          │ │                              │
│ 21 engines   │ │3 engines │ │ 1 engine                     │
│ (11 working) │ │(OOM)     │ │ (not built yet)              │
└──────────────┘ └──────────┘ └──────────────────────────────┘

┌──────────────┐ ┌──────────┐ ┌──────────────────────────────┐
│ VIBEVOICE    │ │ HIGGS    │ │ S2PRO                        │
│ (port 8003)  │ │ (8004)   │ │ (8005)                       │
│              │ │          │ │                              │
│ Pre-built    │ │Pre-built │ │ Pre-built                    │
│ lmsysorg/    │ │lmsysorg/ │ │ lmsysorg/                    │
│ sglang-omni  │ │sglang-   │ │ sglang-omni                  │
│              │ │omni      │ │                              │
│ ~7 GB VRAM   │ │~9 GB VRAM│ │ ~11 GB VRAM                  │
└──────────────┘ └──────────┘ └──────────────────────────────┘

ALL containers share: /opt/models (38 GB models, never duplicated)
```

### Image Inheritance Tree

```
                     tts-lab-base (1.5 GB)
              nvidia/cuda:12.8.2 + espeak + ffmpeg + tts_lab code
                          │
          ┌───────────────┼────────────────┐
          │               │                │
   stack:current     stack:legacy    (orchestrator
   (+3.5 GB)         (+2.5 GB)        inherits directly)
   torch nightly     torch 1.13.1
   tf 5.12.1         tf 4.46.1
          │               │
   engine-current   engine-legacy
   (+~28 GB)        (+~4 GB)
   21 engines       3 engines
```

### How Engines Communicate

Each engine container runs a FastAPI server exposing:
- `GET /health` — which engines are loaded
- `POST /synthesize` — `{engine, text, params}` → base64-encoded WAV audio

The orchestrator routes requests to the right container via HTTP. Engine URLs are set via environment variables:
```
PIPER_URL=http://engine-current:8101
CHATTTS_URL=http://engine-current:8101
INDEXTTS_URL=http://engine-legacy:8102
ORPHEUS_URL=http://orpheus:8002
```

---

## 4. File Reference

### Committed Files (IaC-ready)

```
TTS-LAB/
├── docker/
│   ├── Dockerfile.base              # Tier 1: universal foundation
│   ├── Dockerfile.stack.current     # Tier 2: torch nightly + tf 5.12
│   ├── Dockerfile.stack.legacy      # Tier 2: torch 1.13 + tf 4.46
│   ├── Dockerfile.engine-current    # Tier 3: 21 engines on stack:current
│   ├── Dockerfile.engine-legacy     # Tier 3: 3 engines on stack:legacy
│   ├── Dockerfile.orpheus           # Tier 3: Orpheus 3B + vllm
│   └── Dockerfile.orchestrator      # Web UI + HTTP dispatch
├── docker-compose.yml               # 7 services, GPU profiles
├── .dockerignore                    # Exclude models, venvs from build
├── .env.example                     # HF_TOKEN template
├── .github/workflows/
│   └── build-images.yml             # CI/CD: builds + pushes to GHCR
├── tts_lab_engine_server.py         # Shared FastAPI for engine containers
├── tts_lab_orpheus_server.py        # Standalone Orpheus server
├── tts_lab_shims_legacy.py          # Minimal shims (50 lines vs 551)
└── patches/                         # Compatibility patches
```

### What Each Dockerfile Contains

| Dockerfile | FROM | Key Additions | Size |
|------------|------|---------------|:---:|
| `.base` | `nvidia/cuda:12.8.2-runtime-ubuntu22.04` | espeak-ng, ffmpeg, python3.11, fastapi, tts_lab code | 1.5 GB |
| `.stack.current` | `tts-lab-base` | torch nightly cu128, tf 5.12, numpy, protobuf, patches | +3.5 GB |
| `.stack.legacy` | `tts-lab-base` | torch 1.13.1 cu117, tf 4.46.1, numpy<2.0, protobuf<4.0 | +2.5 GB |
| `.engine-current` | `stack:current` | 21 engine pip packages + git clones + nightly torch reinstall | +28 GB |
| `.engine-legacy` | `stack:legacy` | indextts, parler, qwen3tts + parler patch | +4 GB |
| `.orpheus` | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` | torch CUDA 12.1, vllm, orpheus-speech | 7 GB |
| `.orchestrator` | `tts-lab-base` | ENV ORCHESTRATOR_MODE=1 | +0 |

---

## 5. Build Process — Step by Step

### What Happens When You Run `docker build`

```
1. Docker reads the Dockerfile
2. For each RUN/COPY line, it creates a layer
3. Each layer is cached by a hash of its inputs
4. If inputs haven't changed, the cached layer is reused
5. If inputs changed, that layer and ALL subsequent layers rebuild

Example rebuild times:
  Only app code changed:     ~5 seconds  (last layer only)
  New pip package added:     ~2 minutes  (one pip layer)
  CUDA base image updated:   ~30 minutes (everything rebuilds)
```

### Dependency Order

```
docker build -f Dockerfile.base -t tts-lab-base .
docker build -f Dockerfile.stack.current -t tts-lab-stack-current .
docker build -f Dockerfile.stack.legacy -t tts-lab-stack-legacy .
docker build -f Dockerfile.engine-current -t tts-lab-engine-current .
docker build -f Dockerfile.engine-legacy -t tts-lab-engine-legacy .
docker build -f Dockerfile.orchestrator -t tts-lab-orchestrator .
```

Each step depends on the previous. Docker Compose handles this automatically:
```
docker compose build  # Builds all 4 custom images in order
docker compose up -d  # Starts everything
```

### Starting the Lab

```bash
# Default: orchestrator + engine-current + engine-legacy (24 engines)
docker compose up -d

# + Orpheus (GPU mandatory, ~6 GB VRAM)
docker compose --profile gpu up -d

# + SGLang servers (~7-11 GB VRAM each)
docker compose --profile sglang up -d vibevoice higgs s2pro

# Everything
docker compose --profile gpu --profile sglang up -d
```

---

## 6. Lessons Learned (Ad-Hoc → IaC)

Every lesson here becomes a requirement for the IaC rewrite.

### Lesson 1: CUDA Image Tags Changed

**Problem:** `nvidia/cuda:12.8-runtime-ubuntu22.04` doesn't exist.
**Root cause:** NVIDIA added patch versions to tags: `12.8.2-runtime-ubuntu22.04`.
**Fix:** Use exact patch version in Dockerfiles.
**IaC requirement:** Dockerfiles must reference verifiable image tags.

### Lesson 2: RTX 5060 Ti Needs Torch Nightly

**Problem:** `torch.cuda.is_available()` → warning about sm_120 not supported.
**Root cause:** RTX 5060 Ti has Blackwell architecture (compute capability 12.0). Stable torch 2.11 only supports up to sm_90 (Hopper).
**Fix:** Use PyTorch nightly: `--index-url https://download.pytorch.org/whl/nightly/cu128`.
**IaC requirement:** Dockerfile.stack.current must use nightly until torch 2.12 stable.

### Lesson 3: Python Version Matters

**Problem:** CUDA base image defaults to Python 3.10, but we install Python 3.11. `python3` → 3.10, `pip` → installs for 3.10. Packages installed for wrong Python version.
**Fix:** Add `python-is-python3` to apt install, or use `python3.11` explicitly.
**IaC requirement:** Standardize on one Python version throughout.

### Lesson 4: Hardcoded Paths Break in Docker

**Problem:** Patch scripts reference `/opt/arthur-bench-env/lib/python3.11/site-packages/` — the OLD venv path. This doesn't exist inside Docker.
**Fix:** Auto-detect site-packages: `python3 -c "import site; print(site.getsitepackages()[0])"`.
**IaC requirement:** All scripts must auto-detect paths.

### Lesson 5: Pip Dependency Conflicts Are Real

**Problem:** Installing multiple packages in one `pip install` causes version conflicts. Examples:
- `styletts2` vs `f5-tts`: accelerate version conflict (<0.26 vs >=0.33)
- `coqui-tts` vs `fish-speech`: conflicting sub-dependencies
- Engine packages downgrade torch (requiring nightly reinstall)
**Fix:** Split into separate RUN lines. Use `--no-deps` for problematic packages.
**IaC requirement:** One package group per RUN. Nightly torch reinstalled as LAST step.

### Lesson 6: Protobuf Version Is Critical

**Problem:** `cannot import name 'builder' from 'google.protobuf.internal'` — newer protobuf removed the builder module.
**Who needs what:**
- google-api-core → protobuf >= 5.29.6
- styletts2, f5tts, chatterbox → protobuf < 5.0 (for builder module)
**Fix:** Pin `protobuf>=5.0,<6.0` and hope. Or split into separate containers.
**IaC requirement:** Protobuf version explicitly pinned. Consider separate container for protobuf-sensitive engines.

### Lesson 7: GPU VRAM Is Shared — FIXED

**Problem:** Multiple models sharing one GPU compete for VRAM. Bark loads first, eats 15 GB, starves everything else.
**Root cause:** `_state[name]["instance"]` retained a reference to the evicted model, preventing Python GC from freeing GPU tensors. Even `torch.cuda.empty_cache()` couldn't recover the memory because the tensors were still reachable.
**Fix (deployed 2026-06-20):**
1. `_evict_current` in `tts_lab_engine_server.py` now clears `_state[name].pop("instance", None)` BEFORE calling `_safe_del`.
2. Added explicit `gc.collect()` after deletion.
3. Added `torch.cuda.memory.caching.allocator.empty_cache()` call.
4. Engine server runs in lazy-load mode — 0 engines loaded at startup, loading on demand.
**Result:** Successfully switches between all 14 engines without OOM. Confirmed: bark → chatterbox → melo → zonos transitions all work.
**IaC requirement:** ✅ Implemented. The fix is in the engine server code.

### Lesson 8: Docker Disk Usage Is Extreme

**Problem:** A single engine-current image is 50 GB. Docker build cache and intermediate layers consumed 100+ GB during build.
**Stats from VM:**
- Base image: 6.7 GB
- Stack current: 18.5 GB
- Engine current: 49.9 GB
- Stack legacy: 12.2 GB
- Engine legacy: ~10 GB
- **Total images: ~94 GB**
- **Model volume: 163 GB**
- **Total deployment: ~260 GB**
**Mitigation:** `docker builder prune -af` freed 60+ GB of build cache. Models should be on a separate disk.
**IaC requirement:** Disk budget documented. Separate disk for models. Prune step in build pipeline.

### Lesson 9: Some Packages Don't Build From Source

**Problem:** `av`, `pyaudio`, `parallel-wavegan`, `openai-whisper` fail to build C extensions from source in Docker.
**Fix:** Use pre-built wheels (`--only-binary av av`) or skip problematic engines.
**IaC requirement:** Pre-built wheels preferred. Source builds need build-essential + dev headers.

### Lesson 10: Engine Server Must Be Restart-Safe

**Problem:** `tts_lab_shims.py` import crashes intermittently on container restart. The shims monkey-patch torch/transformers internals that change between versions.
**Fix needed:** Make shims import tolerant of missing modules. Only apply patches that are actually needed for the stack version.
**IaC requirement:** Engine server must start reliably. Shims should be minimal and stack-specific.

---

## 7. Current Status

### Working (Ad-Hoc Deployment on VM — 2026-06-20)

| Component | Status | Details |
|-----------|:------|---------|
| Docker base image | ✅ Built | 6.7 GB, `nvidia/cuda:12.8.2` with espeak, ffmpeg, NLTK |
| Stack: current | ✅ Built | 18.5 GB, torch 2.12 nightly + tf 5.12 + transformers 5.12 |
| Engine: current | ✅ Running | 49.9 GB, 20/28 engines available, lazy-load mode on port 8101 |
| Orchestrator | ✅ Running | 25/28 shown available, Web UI on port 8001 |
| Engine: legacy | ❌ Not deployed | Needs middle-ground stack for Blackwell GPU (torch 2.x + tf 4.x) |
| Orpheus | ❌ Not deployed | Gated model (Hugging Face access request needed) |
| SGLang servers | ❌ Not started | Pre-built images available — 3 engines waiting |
| GPU (RTX 5060 Ti) | ✅ Working | torch 2.12 nightly, CUDA 12.8, sm_120 Blackwell supported |
| VRAM management | ✅ Fixed | Lazy-load + eviction works. Any-to-any engine switching confirmed. |

### Definitive Test Results (2026-06-20 — Round 3, final)

Full systematic test via engine server (port 8101) with VRAM clearing between tests.
All fixes deployed: VRAM leak fix, langchain, MeCab/unidic, lightning, zonos.backbone,
NLTK data, ONNX model symlink, orchestrator lazy-mode health check.

#### ✅ PASSING (14 engines)

| Engine | Audio | RTF | Synth Time | Notes |
|--------|------:|:---:|-----------:|-------|
| **bark** | 8333ms | 5.92× | 49.3s | Small model. Uses ~12GB VRAM. |
| **chatterbox** | 3480ms | 2.42× | 8.4s | Long-form TTS. Good quality. |
| **chatterboxturbo** | 3800ms | 1.11× | 4.2s | Nearly real-time! Turbo variant. |
| **chattts** | 4429ms | 2.72× | 12.1s | Conversational TTS. |
| **dia** | 4899ms | 7.20× | 35.3s | 1.6B model. Loading slow. |
| **fishspeech** | 11842ms | 3.48× | 41.2s | Voice cloning. Needs fish-speech code. |
| **kokoro** | 5418ms | 3.20× | 17.3s | ONNX-based. No GPU needed. |
| **matcha** | 5215ms | 0.24× | 1.3s | **Real-time capable!** Fastest. |
| **melo** | 6014ms | 0.46× | 2.8s | **Real-time capable!** |
| **omnivoice** | 4240ms | 0.67× | 2.8s | **Real-time capable!** |
| **outetts** | 5306ms | 13.59× | 72.1s | Slowest. Big model. |
| **piper** | 3703ms | 0.43× | 1.6s | **Real-time capable!** ONNX-based. |
| **styletts2** | 5547ms | 0.22× | 1.2s | **Real-time capable!** Style control. |
| **zonos** | 5143ms | 4.29× | 22.1s | Voice cloning. Backbone dir fix needed. |

**RTF Legend:** <1.0 = faster than real-time (can stream). 1-3× = near real-time. >3× = slower than real-time.

#### ❌ FAILING (6 engines)

| Engine | Root Cause | Fix |
|--------|-----------|-----|
| **f5tts** | Requires reference audio clip | Expected. Upload 5-15s WAV for testing. |
| **xtts** | torchcodec incompatible with nightly torch 2.12 | Build torchcodec from source for sm_120, or use stable torch. |
| **qwen3tts** | KeyError 'default' in ROPE init | transformers 5.x incompatibility. Needs middle-ground stack (torch 2.x + transformers 4.x). |
| **higgs** | SGLang server not running | `docker compose --profile sglang up -d higgs` |
| **vibevoice** | SGLang server not running | `docker compose --profile sglang up -d vibevoice` |
| **s2pro** | SGLang server not running | `docker compose --profile sglang up -d s2pro` |

#### ⛔ UNAVAILABLE (8 engines — need setup or not built)

| Engine | Reason | Category |
|--------|--------|----------|
| **cosyvoice** | git clone needed | Build failure — openai-whisper Cython |
| **csm** | Model gated + git clone needed | Requires huggingface-cli login |
| **indextts** | pip install needed | Legacy stack (torch 1.x + tf 4.x) |
| **manatts** | parallel-wavegan unavailable | No wheel available from PyPI |
| **neutts** | Not configured | Needs manual setup in tts_lab_engines.py |
| **openvoice** | pip install needed | Build failure — av package |
| **orpheus** | Gated model | Requires Hugging Face access request |
| **parler** | pip install needed | Legacy stack (torch 1.x + tf 4.x) |

### Deployed Fixes (this round)

| Fix | File | Description |
|-----|------|-------------|
| VRAM leak | `tts_lab_engine_server.py` | `_evict_current` now clears `_state[name]["instance"]` + `gc.collect()` |
| Orchestrator lazy mode | `tts_lab_dispatch.py` | `_check_available_remote` checks absence of `reason` field, not `loaded` flag |
| Model path symlink | Container | `ln -sf /opt/models/tts /opt/arthur/models` — piper/kokoro ONNX access |
| Missing deps | Container | langchain<0.3.0, MeCab+unidic, lightning, einops-exts, munch, loralib, cachetools |
| zonos.backbone | Container | Copied `backbone/` dir from git repo into pip package |
| NLTK data | Container | `nltk.download('punkt_tab')` for styletts2 |

### Web UI Status

| Component | Port | Status |
|-----------|:----:|--------|
| Orchestrator | 8001 | ✅ Working — 25/28 engines shown available (lazy-mode fix deployed) |
| Engine-Current | 8101 | ✅ 20 engines available (14 tested working, 3 fail, 3 external service) |
| Engine-Legacy | 8102 | ❌ Not deployed yet (needs middle-ground stack for Blackwell GPU) |

---

## 8. IaC Rewrite Plan

### Recommended Toolchain

| Tool | Purpose |
|------|---------|
| **Ansible** | VM provisioning — install Docker, NVIDIA toolkit, clone repo, start services |
| **Docker Compose** | Container orchestration — define services, volumes, networks |
| **GitHub Actions** | CI/CD — build images on push, push to GHCR |
| **GitHub Container Registry** | Image hosting — free, unlimited for public repos |

### Why Not Other Tools

| Tool | Why Not |
|------|---------|
| Terraform | Overkill for single VM. Great for cloud, not on-prem. |
| Kubernetes | Massive overkill. 1 VM doesn't need a cluster orchestrator. |
| Shell scripts | Not declarative or idempotent. Hard to maintain. |
| Packer | Builds VM images — useful but Ansible + cloud-init is simpler. |
| Puppet/Chef/Salt | Agent-based. Ansible is agentless (SSH only). |

### Ansible Playbook Structure (Planned)

```yaml
# site.yml — main playbook
- hosts: tts-lab
  roles:
    - docker           # Install Docker + NVIDIA Container Toolkit
    - disk             # Mount /opt/models data disk
    - repo             # Clone git repo
    - build            # docker compose build
    - deploy           # docker compose up -d
    - monitoring       # Health checks, log rotation

# Variables (group_vars/tts-lab.yml)
vm_ip: 192.168.0.87
ssh_user: arthur
models_disk: /dev/sdb1
gpu_profile: true   # enable --profile gpu
```

### What the IaC Rewrite Must Fix

All 10 lessons from [§6](#6-lessons-learned-ad-hoc--iac), plus:

1. **Single-command deploy:** `ansible-playbook -i inventory site.yml`
2. **Idempotent:** Run 10 times, same result. No errors on re-run.
3. **Restart-safe engine server:** Fix shims imports to not crash on restart.
4. **VRAM management:** Lazy engine loading or eviction in engine server.
5. **Complete engine set:** Fix remaining 7 failed engines.
6. **Disk budget enforced:** Ansible checks disk before building.
7. **Rollback:** `docker compose up -d` with previous image tags.

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **Container** | A lightweight, isolated environment that packages code + dependencies. Like a VM but shares the host OS kernel. |
| **Image** | A read-only blueprint for creating containers. Built from a Dockerfile. |
| **Dockerfile** | A text recipe defining how to build an image. `FROM`, `RUN`, `COPY`, `CMD`. |
| **Layer** | One step in a Dockerfile. Layers are cached and shared. Key to our tiered architecture. |
| **FROM** | Inherits from a parent image. Like `import` in Python but for entire filesystems. |
| **Volume** | A host directory mounted into a container. Survives container deletion. |
| **Registry** | A server that stores Docker images. GHCR = GitHub Container Registry (free for public). |
| **Docker Compose** | Runs multiple containers together. Defined in `docker-compose.yml`. |
| **Healthcheck** | A command Docker runs to check if a container is healthy. |
| **GPU / CUDA** | NVIDIA's parallel computing platform. Required for fast AI inference. |
| **VRAM** | Video RAM. GPU memory. RTX 5060 Ti has 16 GB. |
| **sm_120 / Blackwell** | GPU architecture. RTX 5060 Ti uses it. Needs torch nightly for support. |
| **TTS** | Text-to-Speech. Converting text to spoken audio. |
| **Engine** | One TTS model. Piper, Kokoro, ChatTTS, Bark, etc. We have 28. |
| **Stack** | A versioned set of ML libraries (torch + transformers + numpy + protobuf). |
| **Orchestrator** | The main web server that routes synthesis requests to engine containers. |
| **IaC** | Infrastructure as Code. Defining servers and deployments in version-controlled files. |
| **Ansible** | An IaC tool that uses SSH and YAML. Agentless — no software installed on managed machines. |
| **Idempotent** | Running the same operation multiple times produces the same result. Essential for IaC. |
| **GHCR** | GitHub Container Registry. Free unlimited storage for public Docker images. |
| **CI/CD** | Continuous Integration / Continuous Deployment. GitHub Actions builds images on git push. |
| **SGLang** | A serving framework for large language/audio models. Used by VibeVoice, Higgs, S2-Pro. |
| **vllm** | An inference engine for LLMs. Used by Orpheus. Requires CUDA GPU. |
| **ONNX** | Open Neural Network Exchange. A model format that runs on CPU or GPU. Piper, Kokoro, Matcha use it. |

---

## 10. Command Cheatsheet

### On the VM (Ubuntu 22.04)

```bash
# ── Docker basics ──────────────────────────────────────────────
docker images                              # List all images
docker ps                                  # List running containers
docker ps -a                               # List all containers (including stopped)
docker logs <name> --tail 50               # Last 50 log lines
docker exec -it <name> bash               # Shell inside container
docker rm -f <name>                        # Force-remove container
docker restart <name>                      # Restart container
docker system prune -af                    # Delete EVERYTHING not in use (careful!)
docker builder prune -af                   # Delete build cache only

# ── Build ─────────────────────────────────────────────────────
cd /opt/tts-lab-docker
docker build -f docker/Dockerfile.base -t tts-lab-base .
docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current .
docker compose build                       # Build all services

# ── Run ───────────────────────────────────────────────────────
docker compose up -d                       # Start all (CPU)
docker compose --profile gpu up -d         # + Orpheus
docker compose logs -f                     # All logs
docker compose ps                          # Status

# ── Debug ─────────────────────────────────────────────────────
docker logs tts-lab-engine-current --tail 20
curl http://localhost:8101/health          # Engine server status
curl -X POST http://localhost:8101/synthesize \
  -H "Content-Type: application/json" \
  -d '{"engine":"chattts","text":"Hello.","params":{}}' -o /tmp/test.wav
ffprobe /tmp/test.wav

# ── Disk ──────────────────────────────────────────────────────
df -h                                       # Disk free
docker system df                            # Docker disk usage
du -sh /opt/models/huggingface              # Model cache size
```

### On Your Windows Machine (from repo)

```bash
# Deploy code changes
scp -i ~/.ssh/id_arthur_vm tts_lab_engine_server.py arthur@192.168.0.87:/opt/tts-lab-docker/
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87 "cd /opt/tts-lab-docker && docker compose up -d --build"

# Sync Dockerfiles from VM (after ad-hoc fixes)
scp arthur@192.168.0.87:/opt/tts-lab-docker/docker/Dockerfile.\* docker/
```

---

> **Next:** IaC rewrite using Ansible + Docker Compose + GitHub Actions. Every lesson from this ad-hoc deployment becomes a requirement. Every error encountered becomes a test case. The goal: `git clone && docker compose up -d` works on ANY Ubuntu 22.04 VM with an NVIDIA GPU.
