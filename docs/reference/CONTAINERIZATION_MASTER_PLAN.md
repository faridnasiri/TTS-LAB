# TTS Lab Containerization — Master Plan

> **Audience:** Developers with zero to basic Docker knowledge.
> **Goal:** Understand *why* containers, *how* the tiered architecture works, and *what* to do day-to-day.
> **Date:** 2026-06-19
> **Status:** Plan — implementation pending
> **Current Environment:** transformers 5.12.1, torch 2.10.0+cu128, Python 3.11, CUDA GPU
> **Related Docs:** [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md) — current engine status & fixes

---

## Table of Contents

- [Part 1: The Problem Containers Solve](#part-1-the-problem-containers-solve)
  - [1.1 What is Dependency Hell?](#11-what-is-dependency-hell)
  - [1.2 The TTS Lab's Specific Problem](#12-the-tts-labs-specific-problem)
  - [1.3 Why Not Just Use Virtual Environments?](#13-why-not-just-use-virtual-environments)
- [Part 2: Docker Concepts (from Zero)](#part-2-docker-concepts-from-zero)
  - [2.1 What is a Container?](#21-what-is-a-container)
  - [2.2 What is an Image?](#22-what-is-an-image)
  - [2.3 What are Layers?](#23-what-are-layers)
  - [2.4 What is a Volume?](#24-what-is-a-volume)
  - [2.5 What is a Registry?](#25-what-is-a-registry)
  - [2.6 What is Docker Compose?](#26-what-is-docker-compose)
- [Part 3: The Tiered Architecture](#part-3-the-tiered-architecture)
  - [3.1 The Big Idea: Inheritance](#31-the-big-idea-inheritance)
  - [3.2 The Three Tiers](#32-the-three-tiers)
  - [3.3 Visual: The Full Inheritance Tree](#33-visual-the-full-inheritance-tree)
  - [3.4 Why This Solves Everything](#34-why-this-solves-everything)
- [Part 4: Complete File Reference](#part-4-complete-file-reference)
  - [4.1 Dockerfile.base](#41-dockerfilebase)
  - [4.2 Dockerfile.stack.v1](#42-dockerfilestackv1)
  - [4.3 Dockerfile.stack.v2 (Future-Proof)](#43-dockerfilestackv2-future-proof)
  - [4.4 Engine Dockerfiles (26× thin images)](#44-engine-dockerfiles-26-thin-images)
  - [4.5 Dockerfile.orpheus (CUDA)](#45-dockerfileorpheus-cuda)
  - [4.6 docker-compose.yml](#46-docker-composeyml)
- [Part 5: Step-by-Step Build & Deploy](#part-5-step-by-step-build--deploy)
  - [5.1 Prerequisites](#51-prerequisites)
  - [5.2 First-Time Build](#52-first-time-build)
  - [5.3 Starting the Lab](#53-starting-the-lab)
  - [5.4 Verifying Everything Works](#54-verifying-everything-works)
- [Part 6: Day-to-Day Maintenance](#part-6-day-to-day-maintenance)
  - [6.1 Adding a New Engine](#61-adding-a-new-engine)
  - [6.2 Updating an Engine's Dependencies](#62-updating-an-engines-dependencies)
  - [6.3 Adding a New ML Stack](#63-adding-a-new-ml-stack)
  - [6.4 Migrating an Engine to a New Stack](#64-migrating-an-engine-to-a-new-stack)
  - [6.5 Removing an Engine](#65-removing-an-engine)
  - [6.6 Monitoring & Debugging](#66-monitoring--debugging)
- [Part 7: Edge Cases & Special Situations](#part-7-edge-cases--special-situations)
  - [7.1 GPU Engines](#71-gpu-engines)
  - [7.2 Gated HuggingFace Models](#72-gated-huggingface-models)
  - [7.3 SGLang External Servers](#73-sglang-external-servers)
  - [7.4 Disk Is Full](#74-disk-is-full)
  - [7.5 Engine Crashes on Startup](#75-engine-crashes-on-startup)
  - [7.6 Multiple GPUs](#76-multiple-gpus)
  - [7.7 Air-Gapped / No-Internet Deployment](#77-air-gapped--no-internet-deployment)
- [Part 8: Image Hosting & Distribution](#part-8-image-hosting--distribution)
  - [8.1 Where to Host](#81-where-to-host)
  - [8.2 Push/Pull Sizes](#82-pushpull-sizes)
  - [8.3 CI/CD Pipeline](#83-cicd-pipeline)
- [Part 9: Migration from Bare Metal](#part-9-migration-from-bare-metal)
- [Part 10: Quick Reference](#part-10-quick-reference)
  - [10.1 Command Cheatsheet](#101-command-cheatsheet)
  - [10.2 Glossary](#102-glossary)
  - [10.3 Disk Budget Summary](#103-disk-budget-summary)

---

## Part 1: The Problem Containers Solve

### 1.1 What is Dependency Hell?

Imagine you're cooking in a kitchen. Recipe A needs salt with exactly 3% iodine. Recipe B needs salt with zero iodine — it ruins the dish if iodine is present. You have one salt shaker. You can't satisfy both recipes.

In software, **dependency hell** is the same problem. Two programs need different versions of the same library, and you only have one installation of that library on your system.

A real example from the TTS Lab:

```
Engine "Bark" needs:   numpy version 1.26.x (anything newer breaks it)
Engine "Orpheus" needs: numpy version 2.x   (anything older doesn't work)
```

You cannot install **both** numpy 1.26 and numpy 2.0 on the same system. One of these engines will be broken. This is dependency hell.

### 1.2 The TTS Lab's Specific Problem

The TTS Lab has **28 text-to-speech engines**. Each engine was written by a different research team at a different time, using whatever versions of libraries were current when they published.

The current VM runs **transformers 5.12.1 + torch 2.10.0 on CUDA**. Of the 28 engines:

| Status | Count | Engines |
|--------|:-----:|---------|
| **Working** | 21 | piper, kokoro, melo, matcha, chattts, outetts, bark, styletts2, f5tts, dia, xtts, cosyvoice, fishspeech, chatterbox, openvoice, zonos, manatts, chatterboxturbo, omnivoice |
| **Broken — needs older stack** | 3 | indextts (176 imports removed in tf 5.x), parler (meta tensor removed in torch 2.10), qwen3tts (config API changed in tf 5.x) |
| **Gated — needs auth** | 2 | orpheus, csm |
| **SGLang server required** | 3 | vibevoice, higgs, s2pro |
| **Not configured** | 1 | neutts |

The critical dependency conflict is no longer just Orpheus. **Three engines need a completely different ML stack:**

| Library | 21 engines need | 3 broken engines need | Conflict? |
|----------|--------------------------|---------------------------|:---------:|
| **torch** | 2.10.0 | 1.13.x | ✅ **Hard conflict** |
| **transformers** | 5.12.1 | 4.46.x | ✅ **Hard conflict** |
| **numpy** | 2.x (via torch 2.10) | 1.x (via torch 1.13) | ✅ **Hard conflict** |

And the CUDA/Orpheus isolation remains:

| Library | 21 engines | Orpheus (vllm) | Conflict? |
|----------|-----------|----------------|:---------:|
| **numpy** | 2.x | 2.x (also 2.x, but vllm pins differently) | ⚠️ Version coupling risk |
| **protobuf** | 5.x (via tf 5.12) | 5.x (vllm) | ⚠️ Version coupling risk |
| **CUDA** | 12.8 (via torch 2.10) | 12.1 (vllm requirement) | ⚠️ Different CUDA versions |

The core insight: **21 of 28 engines work on the current stack (torch 2.10 + tf 5.12).** Three need a legacy stack (torch 1.13 + tf 4.46). One needs CUDA isolation (Orpheus/vllm). Three are external SGLang servers.

### 1.3 Why Not Just Use Virtual Environments?

Python virtual environments (`venv`) isolate Python packages. They solve the *easy* version of the problem.

However, they do **not** solve:

| Problem | venv handles it? | Container handles it? |
|---------|:---:|:---:|
| Different Python versions (3.10 vs 3.11 vs 3.12) | ❌ One Python per system | ✅ Each container has its own Python |
| System packages (espeak-ng, ffmpeg, CUDA drivers) | ❌ Shared across all venvs | ✅ Each container has its own system |
| GPU library versions (CUDA 11.8 vs 12.1 vs 12.4) | ❌ One CUDA toolkit per machine | ✅ Each container has its own CUDA |
| "It works on my machine" | ❌ Different OS, different bugs | ✅ Same OS image everywhere |
| Reproducing a setup from 2 years ago | ❌ Good luck | ✅ Dockerfile is a time capsule |
| Running on a different server tomorrow | ❌ Re-do everything | ✅ `docker compose up` |

The TTS Lab has system-level dependencies (espeak-ng, ffmpeg, CUDA) that venvs cannot isolate. Containers wrap the **entire environment** — operating system, system packages, Python, pip packages, everything.

---

## Part 2: Docker Concepts (from Zero)

> If you already know Docker, skip to [Part 3](#part-3-the-tiered-architecture).

### 2.1 What is a Container?

A **container** is a running instance of a self-contained environment. Think of it as a lightweight virtual machine — but instead of virtualizing hardware, it shares the host's operating system kernel and isolates everything else.

```
┌──────────────────────────────────────────────┐
│  HOST MACHINE (your VM or laptop)            │
│                                              │
│  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Container A     │  │ Container B     │   │
│  │                 │  │                 │   │
│  │ Python 3.11     │  │ Python 3.12     │   │
│  │ numpy 1.26      │  │ numpy 2.1       │   │
│  │ espeak-ng 1.50  │  │ (no espeak)     │   │
│  │                 │  │                 │   │
│  │ "24 TTS engines"│  │ "Orpheus only"  │   │
│  └─────────────────┘  └─────────────────┘   │
│                                              │
│  Shared: Linux kernel, /opt/models (models)  │
└──────────────────────────────────────────────┘
```

A container is:
- **Isolated:** What happens in Container A cannot affect Container B.
- **Ephemeral:** Delete a container, its internal state is gone. (Unless you use volumes.)
- **Lightweight:** Starts in seconds, uses MB of RAM overhead (vs. GB for a full VM).
- **Reproducible:** The same image produces the same container on any machine.

### 2.2 What is an Image?

An **image** is the blueprint from which containers are created. It's a read-only snapshot of a filesystem — the operating system files, installed packages, application code, and configuration.

```
Image  ──(docker run)──▶  Container (running)
        ──(docker run)──▶  Container (another one, same image)
        ──(docker run)──▶  Container (third one)
```

You build an image from a **Dockerfile** (a recipe). You run containers from images. You never modify an image — you build a new version.

### 2.3 What are Layers?

This is the most important concept for understanding why our architecture works.

Every line in a Dockerfile creates a **layer**. Layers are stacked on top of each other:

```
┌─────────────────────────────────────┐
│ Layer 6: COPY app code (10 MB)      │  ← Changes weekly
├─────────────────────────────────────┤
│ Layer 5: pip install chatterbox     │  ← Changes monthly
├─────────────────────────────────────┤
│ Layer 4: pip install torch audio    │  ← Changes rarely
├─────────────────────────────────────┤
│ Layer 3: pip install numpy protobuf │
├─────────────────────────────────────┤
│ Layer 2: apt-get install espeak     │  ← Changes every 6 months
├─────────────────────────────────────┤
│ Layer 1: FROM python:3.11-slim      │  ← Almost never changes
└─────────────────────────────────────┘
```

**Critical fact: Layers are cached and shared.** If two images both start with `FROM python:3.11-slim`, that layer is stored on disk **once**, no matter how many images use it. If you rebuild after changing only the app code, only Layer 6 rebuilds — Layers 1–5 are reused from cache.

This is why **one image per engine** does NOT use 28× the disk space. The heavy layers (Python, PyTorch, system packages) are stored once and shared by all 28 images.

### 2.4 What is a Volume?

A **volume** is a directory on the host machine that is mounted into a container. It outlives the container.

```
Host machine:                       Container:
/opt/models/                  ←───▶ /opt/models/
  ├── huggingface/                    (same files, same writes)
  ├── tts/
  └── outetts-gguf/
```

When the container writes a downloaded model to `/opt/models/huggingface/`, it's actually writing to the host's `/opt/models/huggingface/`. Delete the container, the models survive. Start a new container, the models are already there.

**All TTS Lab containers share one model volume.** No model is downloaded twice. No model is duplicated.

### 2.5 What is a Registry?

A **registry** is a place to store and share Docker images. Think of it as "GitHub for Docker images."

```
Your machine                    Registry (ghcr.io)               Another machine
─────────────                   ──────────────────               ────────────────
docker build ──▶ docker push ──▶ image is stored    ──▶ docker pull ──▶ docker run
```

The most common registries:

| Registry | URL | Best For |
|----------|-----|----------|
| Docker Hub | `docker.io` | General public images |
| GitHub Container Registry (GHCR) | `ghcr.io` | Images for GitHub repos — **our choice** |
| GitLab Container Registry | `registry.gitlab.com` | GitLab-hosted projects |
| Amazon ECR Public | `public.ecr.aws` | AWS ecosystem |

### 2.6 What is Docker Compose?

**Docker Compose** is a tool for running multiple containers together. You define all your containers in a single `docker-compose.yml` file, then start them all with one command:

```bash
docker compose up -d    # Start everything defined in docker-compose.yml
docker compose down     # Stop and remove everything
docker compose logs -f  # Tail logs from all containers
```

Without Compose, you'd need to run `docker run` for each container individually, remembering all the ports, volumes, and environment variables each time. Compose stores all that in a file.

---

## Part 3: The Tiered Architecture

### 3.1 The Big Idea: Inheritance

Just as a child class inherits from a parent class in programming, a Docker image can inherit from a parent image using `FROM`. This is the foundation of the entire architecture:

```dockerfile
# Image A: the universal base
FROM python:3.11-slim
RUN apt-get install -y espeak-ng ffmpeg
# (120 MB — no ML libraries yet)

# Image B: ML stack v1 — inherits from A
FROM tts-lab-base:latest
RUN pip install torch==2.2.0 numpy==1.26 protobuf==3.20 transformers==4.41
# (1.2 GB added on top of A)

# Image C: Piper engine — inherits from B
FROM tts-lab-stack:v1
RUN pip install piper-tts
# (50 MB added on top of B)
```

The result: Images B and C both contain everything from A. Image C contains everything from B. But on disk, the layers of A and B are stored **once** and shared.

### 3.2 The Three Tiers

```
Tier 1:  tts-lab-base            ~120 MB   Changes every 6–12 months
         ↓                       ↓
Tier 2:  tts-lab-stack:current   ~3.5 GB   torch 2.10 + transformers 5.12 (21 engines)
         tts-lab-stack:legacy    ~2.5 GB   torch 1.13 + transformers 4.46 (3 engines)
         tts-lab-stack:cuda      ~7.0 GB   vllm + CUDA 12.1 (Orpheus)
         ↓                       ↓
Tier 3:  tts-lab-piper           ~50 MB    Changes when engine updates
         tts-lab-kokoro          ~80 MB
         tts-lab-melo            ~150 MB
         ... (2 engine containers — current with 21 engines, legacy with 3 engines)
```

#### Tier 1 — `tts-lab-base` (Universal Foundation)

Everything that **all 28 engines agree on**. This tier contains zero ML libraries — just the operating system, system tools, and the TTS Lab application code. Uses a CUDA base image (switched from CPU in the original plan to match the real deployment).

**Key difference from original plan:** The base now uses `nvidia/cuda:12.8-runtime-ubuntu22.04` instead of `python:3.11-slim` because the real environment runs torch 2.10 on CUDA 12.8.

| What | Why Here |
|------|----------|
| `nvidia/cuda:12.8-runtime-ubuntu22.04` | CUDA 12.8 + Ubuntu 22.04 (matches real deployment) |
| `python3.11`, `python3.11-venv`, `python3.11-dev` | Python 3.11 (installed on CUDA base) |
| `espeak-ng`, `espeak-ng-data` | Text-to-phoneme conversion (Kokoro, Zonos, ManaTTS) |
| `ffmpeg` | Audio format conversion |
| `libsndfile1`, `libsndfile1-dev` | WAV file I/O |
| `git`, `curl`, `wget` | Downloading models and repos |
| `soundfile`, `httpx`, `requests` | Python utilities every engine uses |
| `huggingface_hub` | Model downloading (nearly every engine) |
| `packaging` | Version parsing utilities |
| `psutil` | System resource monitoring |
| `tts_lab_*.py`, `patches/` | The application code |

This tier changes when: CUDA base image updates, or the app code changes.

#### Tier 2 — `tts-lab-stack:current`, `tts-lab-stack:legacy`, `tts-lab-stack:cuda` (ML Stacks)

Each stack is a **versioned snapshot of the ML ecosystem**. Engines inherit from whichever stack matches their dependency requirements.

| Stack | Components | Used By |
|-------|-----------|---------|
| **current** | `torch 2.10.0, transformers 5.12.1, numpy 2.x, protobuf 5.x, onnxruntime latest` | **21 engines** — all working engines |
| **legacy** | `torch 1.13.1, transformers 4.46.1, numpy 1.x, protobuf 3.x` | **3 engines** — indextts, parler, qwen3tts |
| **cuda** | `torch CUDA 12.1, vllm, cuda-toolkit` | **1 engine** — Orpheus 3B |

**Why legacy exists:** indextts needs 176 imports from transformers internals that were removed in 5.x. parler needs `torch.Tensor.__getitem__` behavior removed in torch 2.x (meta tensor). qwen3tts needs the old transformers config API (`pad_token_id` on config objects). These are NOT things you can patch — the APIs were removed.

A stack changes when: you intentionally upgrade the ML stack for a set of engines.

**You choose when to create a new stack.** It's not automatic.

#### Tier 3 — Engine Images (Thin)

Each engine image inherits from a stack and adds **only that engine's pip packages**. These are small (30–300 MB) and fast to build.

```dockerfile
# Engines on the current stack:
FROM tts-lab-stack:current
RUN pip install piper-tts      # Just one engine's worth of packages

# Engines on the legacy stack:
FROM tts-lab-stack:legacy
RUN pip install git+https://github.com/index-tts/IndexTTS.git
```

### 3.3 Visual: The Full Architecture (6 Containers)

```
┌──────────────────────────────────────────────────────────────────┐
│ Orchestrator: tts-lab-orchestrator (port 8001)                   │
│                                                                  │
│ FastAPI web UI + dispatch. Imports 21 engines in-process.        │
│ Routes 3 legacy engines → HTTP to legacy container.              │
│ Routes Orpheus → HTTP to orpheus container.                      │
│ Routes SGLang engines → HTTP to SGLang containers.              │
│                                                                  │
│ NO nginx needed — the FastAPI server IS the orchestrator.        │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP (internal Docker network)
          ┌────────────────┼───────────────────────┐
          │                │                       │
   ┌──────┴──────────┐ ┌───┴──────────┐ ┌─────────┴──────────────┐
   │ Container 1     │ │ Container 2  │ │ Container 3             │
   │ stack:current   │ │ stack:legacy │ │ orpheus (GPU mandatory) │
   │ ~5 GB           │ │ ~4 GB        │ │ ~7 GB                   │
   │                 │ │              │ │                         │
   │ torch 2.10.0    │ │ torch 1.13.1 │ │ vllm + CUDA 12.1        │
   │ tf 5.12.1       │ │ tf 4.46.1    │ │                         │
   │                 │ │              │ │ 1 engine                 │
   │ 21 engines      │ │ 3 engines    │ │ Port: 8002              │
   │ Port: 8101      │ │ Port: 8102   │ │                         │
   └─────────────────┘ └──────────────┘ └─────────────────────────┘

   ┌──────────────────┐ ┌──────────────┐ ┌────────────────────────┐
   │ Container 4      │ │ Container 5  │ │ Container 6             │
   │ vibevoice        │ │ higgs        │ │ s2pro                   │
   │ SGLang-Omni      │ │ SGLang-Omni  │ │ SGLang-Omni             │
   │ ~7 GB VRAM       │ │ ~9 GB VRAM   │ │ ~11 GB VRAM             │
   │ Port: 8003       │ │ Port: 8004   │ │ Port: 8005              │
   └──────────────────┘ └──────────────┘ └─────────────────────────┘

   All containers share: /opt/models (38 GB models, never duplicated)
```

### 3.4 Why 6 Containers, Not 28 Images

**Expert feedback on the per-engine image approach:** Even though Docker deduplicates layers, creating 28 Dockerfiles, 28 build targets, and 28 CI paths is overengineering for a single-VM lab. Most engines are just `pip install` away from each other.

**Instead: 2 engine containers (current + legacy).** All 21 compatible engines live in ONE container with ONE Python process — exactly how `tts_lab_dispatch.py` works today. Only the 3 engines that genuinely need a different torch/transformers get their own container. The orchestrator (main FastAPI) imports 21 engines in-process and routes the remaining 7 via HTTP.

```
Main FastAPI (orchestrator)
  ├── In-process: 21 engines (imported, same Python process)
  ├── HTTP → legacy container (3 engines: indextts, parler, qwen3tts)
  ├── HTTP → orpheus container (1 engine)
  └── HTTP → SGLang containers (3 engines: vibevoice, higgs, s2pro)
```

**No nginx needed.** The FastAPI server already serves the UI on port 8001. Adding nginx adds a moving part with no benefit for an internal lab deployment.

---

## Part 4: Complete File Reference

### File Inventory

```
TTS-LAB/
├── docker/
│   ├── Dockerfile.base              ← Tier 1: universal foundation
│   ├── Dockerfile.stack.current     ← Tier 2: torch 2.10 + tf 5.12 (21 engines)
│   ├── Dockerfile.stack.legacy      ← Tier 2: torch 1.13 + tf 4.46 (3 engines)
│   ├── Dockerfile.engine-current    ← Engine container: stack:current + 21 pip installs
│   ├── Dockerfile.engine-legacy     ← Engine container: stack:legacy + 3 pip installs
│   └── Dockerfile.orpheus           ← Orpheus with own CUDA 12.1 base
├── docker-compose.yml               ← 6 services (orchestrator + 2 engine + orpheus + 3 SGLang)
├── docker-compose.gpu.yml           ← GPU profile overrides
├── .github/workflows/
│   └── build-images.yml             ← CI/CD pipeline (6 images, not 28)
└── scripts/
    └── seed-models.sh               ← Pre-download models into volume
```

### 4.1 Dockerfile.base

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 1 — tts-lab-base
# ═══════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Universal foundation shared by EVERY engine image.
#   Contains only what all engines agree on:
#     - Python 3.11 on minimal Debian
#     - System tools (espeak-ng, ffmpeg, git, etc.)
#     - Python utilities (httpx, soundfile, huggingface_hub, etc.)
#     - TTS Lab application code
#
#   DELIBERATELY EXCLUDES:
#     - PyTorch (engines differ on CUDA version — 11.7 vs 12.1 vs 12.8)
#     - numpy, protobuf (engines differ on version)
#     - transformers (engines differ on version)
#     - Any engine-specific pip package
#
# WHAT THIS IMAGE CONTAINS:
#   /opt/arthur/          — TTS Lab Python modules
#   /opt/arthur/patches/  — Compatibility patches
#   /usr/bin/espeak-ng    — Phoneme converter (system)
#   /usr/bin/ffmpeg       — Audio processor (system)
#   /usr/lib/x86_64-linux-gnu/espeak-ng-data/ — espeak voice data
#
# SIZE: ~120 MB
# REBUILT: Every 6–12 months (OS security updates, app code changes)
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="TTS Lab - Base"
LABEL org.opencontainers.image.description="Universal foundation for all TTS Lab engine images"
LABEL tts-lab.tier="1-base"

# ── System packages ──────────────────────────────────────────────
# These are the C/C++ libraries and command-line tools that Python
# packages depend on. They must be installed at the OS level.
#
# build-essential  → C compiler (gcc, make) for building Python C extensions
# libsndfile1      → Audio file format library (WAV, FLAC, etc.)
# libsndfile1-dev  → Headers so Python soundfile can compile against it
# ffmpeg           → Audio/video converter (resampling, format conversion)
# espeak-ng        → Text-to-phoneme engine (Kokoro, Zonos, ManaTTS need this)
# espeak-ng-data   → Voice data files for espeak-ng (phoneme dictionaries)
# sox              → Simple audio processor (format detection, trimming)
# git              → Clone engine repos from GitHub
# wget, curl       → Download models and files
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    sox \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify espeak-ng data is in the right place.
# Phonemizer (used by Kokoro, Zonos, ManaTTS) looks for data at this exact path.
RUN ls /usr/lib/x86_64-linux-gnu/espeak-ng-data/ > /dev/null \
    && echo "espeak-ng data OK at /usr/lib/x86_64-linux-gnu/espeak-ng-data/"

# ── Python utilities (no ML libraries) ───────────────────────────
# These are pure-Python or thin-C-extension packages that every engine uses.
# Installing them here avoids repeating this step in every engine image.
#
# Why version pins?
#   packaging  — lightweight, no conflicts. Pinned for reproducibility.
#   setuptools  — needed by git-cloned packages during pip install.
RUN pip install --no-cache-dir \
    pip>=23.0 \
    setuptools>=65.0 \
    wheel>=0.38.0

RUN pip install --no-cache-dir \
    packaging>=21.0 \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "httpx>=0.27.0" \
    "websockets>=12.0" \
    "psutil>=5.9.0" \
    "requests>=2.28.0" \
    "soundfile>=0.12.1" \
    "huggingface_hub>=0.20.0"

# ── NLTK data (needed by MeloTTS, OpenVoice v2) ──────────────────
# These are linguistic databases used for text tokenization and
# part-of-speech tagging. Downloaded once here so engines don't
# re-download them at startup.
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('cmudict', quiet=True)"

# ── Application code ─────────────────────────────────────────────
# The TTS Lab Python modules. These are the same files for all engines.
# When code changes, only this layer rebuilds (~10 MB, ~5 seconds).
COPY tts_lab.py /opt/arthur/
COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_config.py /opt/arthur/
COPY tts_lab_utils.py /opt/arthur/
COPY tts_lab_engines.py /opt/arthur/
COPY tts_lab_dispatch.py /opt/arthur/
COPY tts_lab_ui.py /opt/arthur/
COPY patches/ /opt/arthur/patches/

# ── Runtime configuration ────────────────────────────────────────
ENV COQUI_TOS_AGREED=1
ENV HF_HOME=/opt/models/huggingface
ENV XDG_CACHE_HOME=/opt/models/cache
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/arthur
```

### 4.2 Dockerfile.stack.current (21 Engines — The Real Environment)

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 2 — tts-lab-stack:current
# ═══════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   The ML stack running on the production VM RIGHT NOW.
#   21 of 28 engines work with this stack.
#
# ENVIRONMENT (matches real deployment):
#   torch 2.10.0+cu128    — CUDA 12.8 GPU
#   transformers 5.12.1   — latest major version
#   numpy 2.x             — pulled by torch 2.10
#   protobuf 5.x          — pulled by transformers 5.12
#   onnxruntime            — latest
#
# ENGINES ON THIS STACK (21):
#   piper, kokoro, melo, matcha, chattts, outetts, bark, styletts2,
#   f5tts, dia, xtts, cosyvoice, fishspeech, chatterbox,
#   chatterboxturbo, omnivoice, openvoice, zonos, manatts
#
#   Plus 3 SGLang clients (thin HTTP wrappers, no ML deps):
#   vibevoice, higgs, s2pro
#
# SIZE: ~3.5 GB added on top of base
# REBUILT: When PyTorch or transformers release new versions
# ═══════════════════════════════════════════════════════════════════

FROM tts-lab-base:latest

LABEL org.opencontainers.image.title="TTS Lab - ML Stack current"
LABEL org.opencontainers.image.description="torch 2.10, transformers 5.12, CUDA 12.8"
LABEL tts-lab.tier="2-stack"
LABEL tts-lab.stack-version="current"

# ── PyTorch CUDA 12.8 ────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch>=2.10.0 \
    torchaudio>=2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── ML ecosystem (current versions on the production VM) ─────────
RUN pip install --no-cache-dir \
    "transformers>=5.12.0" \
    "accelerate>=0.30.0" \
    "onnxruntime>=1.18.0" \
    "safetensors>=0.4.0" \
    "numpy" \
    "protobuf"

# ── BUILD-TIME PATCHES (from ENGINE_FIXES_AND_CONTAINERIZATION.md) ─
# These MUST be applied at build time. They fix:
#   1. torchcodec package metadata (chatterbox fails without it)
#   2. transformers 5.x compatibility stubs

# torchcodec metadata fix — chatterbox/chatterboxturbo need this
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") \
    && mkdir -p "${SITE_PACKAGES}/torchcodec-99.0.0.dist-info" \
    && printf "Metadata-Version: 2.1\nName: torchcodec\nVersion: 99.0.0\n" \
       > "${SITE_PACKAGES}/torchcodec-99.0.0.dist-info/METADATA" \
    && echo "torchcodec metadata stub created"

# transformers 5.x compatibility patches (from the deploy pipeline)
RUN python /opt/arthur/patches/patch_transformers_stubs.py \
    && python /opt/arthur/patches/fix_transformers_shims.py \
    && python /opt/arthur/patches/patch_parler_tts.py \
    && echo "Compatibility patches applied"

# NOTE: tts_lab_shims.py is imported at RUNTIME by the engine server,
# not at build time. It contains inspect.getsourcefile patches and
# transformers 5.x stubs that fix:
#   - isin_mps_friendly, ExtensionsTrie, AddedToken (removed in 5.x)
#   - check_model_inputs compat (changed signature in 5.x)
#   - PretrainedConfig defaults (attrs removed in 5.x)
#   - torch._dynamo._trace_wrapped_higher_order_op (crash fix)
```

### 4.3 Dockerfile.stack.legacy (3 Broken Engines)

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 2 — tts-lab-stack:legacy
# ═══════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   ML stack for engines broken by the torch 2.10 / transformers 5.12
#   upgrade. These engines need OLDER versions where their required
#   APIs still exist.
#
# WHY THIS STACK EXISTS:
#   - indextts: 176 imports from removed transformers internals
#   - parler: torch.meta_tensor removed in torch 2.x
#   - qwen3tts: config.pad_token_id removed in transformers 5.x
#
#   These are NOT patchable. The APIs were deleted from the libraries.
#   These engines need the actual old library versions.
#
# ENVIRONMENT:
#   torch 1.13.1+cu117   — last version with meta tensor support
#   transformers 4.46.1  — last version with the old config API
#   numpy 1.x            — pinned by torch 1.x compatibility
#   protobuf 3.x         — pinned by transformers 4.x compatibility
#
# ENGINES ON THIS STACK (3):
#   indextts, parler, qwen3tts
#
# SIZE: ~2.5 GB added on top of base
# REBUILT: Almost never — these are frozen legacy versions
# ═══════════════════════════════════════════════════════════════════

FROM tts-lab-base:latest

LABEL org.opencontainers.image.title="TTS Lab - ML Stack legacy"
LABEL org.opencontainers.image.description="torch 1.13, transformers 4.46 — for indextts, parler, qwen3tts"
LABEL tts-lab.tier="2-stack"
LABEL tts-lab.stack-version="legacy"

# ── PyTorch 1.13 CUDA 11.7 ──────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117

# ── ML ecosystem (pinned to known-compatible versions) ──────────
RUN pip install --no-cache-dir \
    "numpy>=1.21,<2.0" \
    "protobuf>=3.20,<4.0" \
    "transformers==4.46.1" \
    "accelerate==0.26.0" \
    "onnxruntime>=1.15.0" \
    "safetensors>=0.3.0"

# Compatibility patches — only the ones relevant to transformers 4.x
# parler-tts needs its compat patch regardless of transformers version
RUN python /opt/arthur/patches/patch_parler_tts.py \
    && echo "Legacy compatibility patches applied"
```

### 4.4 Dockerfile.stack.cuda (Orpheus — unchanged from original plan)

See [4.5 Dockerfile.orpheus](#45-dockerfileorpheus-cuda) — the Orpheus image combines stack + engine since it's the only engine on this stack. Same design as the original plan.

### 4.5 Engine Containers (2, not 28)

> **Expert feedback:** Per-engine images are overengineering. "Even though Docker layers deduplicate, you're creating 28 Dockerfiles, 28 build targets, 28 image tags, 28 CI build paths — for almost no gain." Instead: 2 engine containers. All 21 compatible engines in one container (one Python process). Only the 3 that need a different stack get their own container.

#### Dockerfile.engine-current (21 engines)

```dockerfile
# docker/Dockerfile.engine-current
FROM tts-lab-stack:current

LABEL tts-lab.tier="3-engine"
LABEL tts-lab.stack="current"
LABEL tts-lab.engines="21"

# All 21 engines that work with torch 2.10 + tf 5.12
RUN pip install --no-cache-dir \
    piper-tts>=1.2.0 \
    kokoro-onnx>=0.4.0 \
    git+https://github.com/myshell-ai/MeloTTS.git \
    ChatTTS>=0.2.1 \
    outetts>=0.3.0 \
    bark>=1.0.0 \
    styletts2>=0.0.4 \
    f5-tts>=0.3.4 \
    git+https://github.com/nari-labs/dia.git \
    coqui-tts \
    chatterbox-tts>=0.1.0 perth>=1.0.0 \
    fish-speech \
    omnivoice \
    git+https://github.com/Zyphra/Zonos.git \
    git+https://github.com/myshell-ai/OpenVoice.git \
    sherpa-onnx \
    scipy librosa parallel-wavegan \
    phonemizer>=3.2.1

# CosyVoice2 — needs repo clone
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice \
    && pip install --no-cache-dir hyperpyyaml \
    && pip install --no-cache-dir -r /opt/CosyVoice/requirements.txt \
    && pip install --no-cache-dir /opt/CosyVoice/third_party/Matcha-TTS

# ManaTTS repo
RUN git clone --depth 1 \
    https://github.com/MahtaFetrat/Persian-MultiSpeaker-Tacotron2 \
    /opt/models/Persian-MultiSpeaker-Tacotron2

# CSM repo
RUN git clone --depth 1 https://github.com/SesameAILabs/csm /opt/models/csm

EXPOSE 8101
CMD ["uvicorn", "tts_lab_engine_server:app", "--host", "0.0.0.0", "--port", "8101"]
```

#### Dockerfile.engine-legacy (3 engines)

```dockerfile
# docker/Dockerfile.engine-legacy
FROM tts-lab-stack:legacy

LABEL tts-lab.tier="3-engine"
LABEL tts-lab.stack="legacy"
LABEL tts-lab.engines="3"

# indextts, parler, qwen3tts — broken on torch 2.10 / tf 5.12
RUN pip install --no-cache-dir \
    git+https://github.com/index-tts/IndexTTS.git \
    parler-tts==0.2.3 \
    qwen-tts

EXPOSE 8102
CMD ["uvicorn", "tts_lab_engine_server:app", "--host", "0.0.0.0", "--port", "8102"]
```

### 4.6 Dockerfile.orpheus (CUDA)

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 3 — Orpheus 3B Engine (CUDA-required)
# ═══════════════════════════════════════════════════════════════════
#
# WHY THIS IS SEPARATE:
#   Orpheus requires vllm, which requires:
#     - CUDA GPU (ONLY engine that refuses to run without GPU)
#     - numpy >= 2.0 (conflicts with 3 legacy engines that need numpy 1.x)
#     - protobuf >= 5.0 (conflicts with 3 legacy engines that need protobuf 3.x)
#     - CUDA toolkit 12.1 (different CUDA version than main stack's 12.8)
#
#   This image uses a CUDA base image instead of python:3.11-slim.
#   It runs its own Python environment with its own ML stack.
#   It is NOT based on tts-lab-stack:v1 or tts-lab-base.
#
# SIZE: ~7.0 GB
# GPU:  Required (nvidia-container-toolkit must be installed on host)
# VRAM: ~6 GB
# ═══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="TTS Lab - Orpheus 3B"
LABEL tts-lab.tier="3-engine"
LABEL tts-lab.engine="orpheus"
LABEL tts-lab.gpu="required"

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential libsndfile1 libsndfile1-dev ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CUDA ─────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── vllm + Orpheus ───────────────────────────────────────────────
# vllm is the inference engine. It pulls numpy>=2.0 and protobuf>=5.0
# as transitive dependencies — which is WHY this is a separate image.
RUN pip install --no-cache-dir vllm \
    && pip install --no-cache-dir orpheus-speech>=0.1.0

# ── Web server dependencies ──────────────────────────────────────
RUN pip install --no-cache-dir fastapi uvicorn httpx soundfile huggingface_hub

# ── Application code (only what Orpheus needs) ───────────────────
COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_config.py /opt/arthur/
COPY tts_lab_utils.py /opt/arthur/
COPY tts_lab_engines.py /opt/arthur/
COPY tts_lab_orpheus_server.py /opt/arthur/

ENV HF_HOME=/opt/models/huggingface
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/arthur
EXPOSE 8002

CMD ["uvicorn", "tts_lab_orpheus_server:app", "--host", "0.0.0.0", "--port", "8002"]
```

### 4.6 docker-compose.yml

```yaml
version: "3.9"

# ═══════════════════════════════════════════════════════════════════
# TTS Lab — Multi-Container Deployment
# ═══════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
#   24 lightweight engine containers + 1 Orpheus GPU container
#   + 3 SGLang external containers.
#
#   Each engine container runs a thin FastAPI server that loads
#   its model and exposes /health and /synthesize endpoints.
#
#   The main orchestrator (tts-lab-orchestrator) serves the web UI
#   at port 8001 and routes synthesis requests to engine containers.
#
# USAGE:
#   Default (24 engines — main + legacy, all CUDA-capable):
#     docker compose up -d
#
#   + Orpheus (GPU mandatory):
#     docker compose --profile gpu up -d
#
#   Everything (all 28 engines, needs GPU for Orpheus + SGLang):
#     docker compose --profile gpu --profile sglang up -d
#
#   Single engine for testing:
#     docker compose up -d piper
#
#   View logs:
#     docker compose logs -f [service-name]
#
#   Stop everything:
#     docker compose --profile gpu --profile sglang down
# ═══════════════════════════════════════════════════════════════════

# ── Reusable configuration blocks ────────────────────────────────

x-common-env: &common-env
  HF_HOME: /opt/models/huggingface
  XDG_CACHE_HOME: /opt/models/cache
  COQUI_TOS_AGREED: "1"
  TOKENIZERS_PARALLELISM: "false"
  PYTHONUNBUFFERED: "1"

x-model-volume: &model-volume
  volumes:
    - /opt/models:/opt/models
    - /tmp/tts_uploads:/tmp/tts_uploads

x-engine-defaults: &engine-defaults
  <<: *model-volume
  environment:
    <<: *common-env
  restart: unless-stopped
  networks:
    - tts-lab-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s

x-gpu: &gpu-config
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]

# ── Services ─────────────────────────────────────────────────────

services:
  # ═══════════════════════════════════════════════════════════════
  # Orchestrator — Web UI + API Gateway (port 8001)
  # ═══════════════════════════════════════════════════════════════
  orchestrator:
    build:
      context: .
      dockerfile: docker/Dockerfile.base
    image: tts-lab-base:latest
    container_name: tts-lab-orchestrator
    ports:
      - "8001:8001"
    <<: *model-volume
    environment:
      <<: *common-env
      SUNO_USE_SMALL_MODELS: "False"
      # Engine URLs — each points to its container's internal port
      PIPER_URL: http://piper:8100
      KOKORO_URL: http://kokoro:8100
      MELO_URL: http://melo:8100
      CHATTTS_URL: http://chattts:8100
      OUTETTS_URL: http://outetts:8100
      BARK_URL: http://bark:8100
      STYLETTS2_URL: http://styletts2:8100
      F5TTS_URL: http://f5tts:8100
      DIA_URL: http://dia:8100
      XTTS_URL: http://xtts:8100
      COSYVOICE_URL: http://cosyvoice:8100
      PARLER_URL: http://parler:8100
      CHATTERBOX_URL: http://chatterbox:8100
      CHATTERBOXTURBO_URL: http://chatterboxturbo:8100
      FISHSPEECH_URL: http://fishspeech:8100
      CSM_URL: http://csm:8100
      QWEN3TTS_URL: http://qwen3tts:8100
      INDEXTTS_URL: http://indextts:8100
      ZONOS_URL: http://zonos:8100
      OPENVOICE_URL: http://openvoice:8100
      MATCHA_URL: http://matcha:8100
      MANATTS_URL: http://manatts:8100
      OMNIVOICE_URL: http://omnivoice:8100
      ORPHEUS_URL: http://orpheus:8002
      VIBEVOICE_SGLANG_URL: http://vibevoice:8000/v1/audio/speech
      HIGGS_SGLANG_URL: http://higgs:8000/v1/audio/speech
      S2PRO_SGLANG_URL: http://s2pro:8000/v1/audio/speech
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - tts-lab-net
    depends_on:
      piper:
        condition: service_healthy
      kokoro:
        condition: service_healthy
      # ... (all 24 engine containers)
      # In practice, the orchestrator starts before all engines
      # are healthy — it shows "loading" until they report ready.

  # ═══════════════════════════════════════════════════════════════
  # Engine Containers — One per TTS engine (24 of these)
  # ═══════════════════════════════════════════════════════════════

  piper:
    build:
      context: .
      dockerfile: docker/engines/Dockerfile.piper
    image: tts-lab-piper:latest
    container_name: tts-lab-piper
    <<: *engine-defaults
    # Piper uses ONNX Runtime (no torch dependency)

  kokoro:
    build:
      context: .
      dockerfile: docker/engines/Dockerfile.kokoro
    image: tts-lab-kokoro:latest
    container_name: tts-lab-kokoro
    <<: *engine-defaults

  melo:
    build:
      context: .
      dockerfile: docker/engines/Dockerfile.melo
    image: tts-lab-melo:latest
    container_name: tts-lab-melo
    <<: *engine-defaults

  chattts:
    build:
      context: .
      dockerfile: docker/engines/Dockerfile.chattts
    image: tts-lab-chattts:latest
    container_name: tts-lab-chattts
    <<: *engine-defaults

  # ... (20 more engine services follow the same pattern)
  # Full list: outetts, bark, styletts2, f5tts, dia, xtts,
  # cosyvoice, parler, chatterbox, chatterboxturbo, fishspeech,
  # csm, qwen3tts, indextts, zonos, openvoice, matcha, manatts,
  # omnivoice, vibevoice, higgs, s2pro

  # Example of a future engine on a different stack:
  # future-engine:
  #   build:
  #     context: .
  #     dockerfile: docker/engines/Dockerfile.future-engine
  #   image: tts-lab-future-engine:latest
  #   <<: *engine-defaults
  #   # Note: this inherits from stack:v2, while the 24 above
  #   # inherit from stack:v1. They coexist because each container
  #   # has its own isolated filesystem.

  # ═══════════════════════════════════════════════════════════════
  # Orpheus 3B — CUDA-required, isolated ML stack
  # ═══════════════════════════════════════════════════════════════
  orpheus:
    build:
      context: .
      dockerfile: docker/Dockerfile.orpheus
    image: tts-lab-orpheus:latest
    container_name: tts-lab-orpheus
    ports:
      - "8002:8002"
    <<: *model-volume
    environment:
      <<: *common-env
      HF_TOKEN: ${HF_TOKEN:-}
    <<: *gpu-config
    restart: unless-stopped
    profiles:
      - gpu
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    networks:
      - tts-lab-net

  # ═══════════════════════════════════════════════════════════════
  # SGLang External Servers — Pre-built images, no custom Dockerfile
  # ═══════════════════════════════════════════════════════════════

  vibevoice:
    image: lmsysorg/sglang-omni:dev
    container_name: tts-lab-vibevoice
    ports:
      - "8003:8000"
    volumes:
      - /opt/models/huggingface:/root/.cache/huggingface
    command:
      - --model
      - microsoft/VibeVoice-1.5B
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
    <<: *gpu-config
    restart: unless-stopped
    profiles:
      - sglang
    networks:
      - tts-lab-net

  higgs:
    image: lmsysorg/sglang-omni:dev
    container_name: tts-lab-higgs
    ports:
      - "8004:8000"
    volumes:
      - /opt/models/huggingface:/root/.cache/huggingface
    command:
      - --model
      - bosonai/higgs-audio-v3-tts-4b
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
    <<: *gpu-config
    restart: unless-stopped
    profiles:
      - sglang
    networks:
      - tts-lab-net

  s2pro:
    image: lmsysorg/sglang-omni:dev
    container_name: tts-lab-s2pro
    ports:
      - "8005:8000"
    volumes:
      - /opt/models/huggingface:/root/.cache/huggingface
    command:
      - --model
      - fishaudio/s2-pro
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
    <<: *gpu-config
    restart: unless-stopped
    profiles:
      - sglang
    networks:
      - tts-lab-net

networks:
  tts-lab-net:
    driver: bridge
```

---

## Part 4b: Alternative — Single Container with Per-Engine venvs

> **From:** [ENGINE_FIXES_AND_CONTAINERIZATION.md §4.4](ENGINE_FIXES_AND_CONTAINERIZATION.md)

If running 28 separate containers feels like too much orchestration overhead, there is a middle ground: **one Docker container with multiple Python virtual environments inside it.**

### Concept

```
┌─────────────────────────────────────────────────────────┐
│  ONE Container (tts-lab:latest)                         │
│                                                         │
│  /opt/arthur/                                           │
│  ├── venvs/                                             │
│  │   ├── current/     ← torch 2.10 + tf 5.12 (21 eng)  │
│  │   ├── legacy/      ← torch 1.13 + tf 4.46 (3 eng)   │
│  │   └── sglang/      ← HTTP clients only (3 eng)      │
│  ├── tts_lab.py        ← orchestrator (uses current venv)│
│  └── dispatch.py       ← routes legacy engines to        │
│                           subprocess in legacy venv      │
└─────────────────────────────────────────────────────────┘
```

### How It Works

1. **21 engines** run in-process in the `current` venv (fast, no network overhead)
2. **3 legacy engines** (indextts, parler, qwen3tts) are called via **subprocess** — the dispatch layer spawns a short-lived Python process in the `legacy` venv, passes the text + params via stdin JSON, gets WAV back via stdout
3. **3 SGLang engines** (vibevoice, higgs, s2pro) are HTTP clients — no ML deps needed
4. **1 CUDA engine** (orpheus) either runs in a separate container or as a subprocess in its own venv

### Subprocess Dispatch (for legacy engines)

Instead of in-process `_load_indextts()` / `_synth_indextts()`, the dispatch layer runs:

```python
import subprocess, json, base64

def _do_synth_legacy(engine_name: str, text: str, params: dict, venv_path: str) -> dict:
    """Synthesize by spawning a subprocess in the legacy venv."""
    payload = json.dumps({"text": text, "params": params})
    result = subprocess.run(
        [f"{venv_path}/bin/python", "-m", "tts_lab_legacy_worker", engine_name],
        input=payload,
        capture_output=True,
        text=True,
        timeout=300,
    )
    result.check_returncode()
    data = json.loads(result.stdout)
    return {
        "audio_b64": data["audio_b64"],
        "sample_rate": data["sample_rate"],
        "synth_time_ms": data["synth_time_ms"],
        "audio_dur_ms": data["audio_dur_ms"],
        "rtf": data["rtf"],
        "load_time_s": data["load_time_s"],
    }
```

The legacy worker (`tts_lab_legacy_worker.py`) is a tiny script (~40 lines) that loads one engine, synthesizes, and prints JSON to stdout. The model stays loaded between calls if the worker is kept alive (using a long-running daemon) or is reloaded each time (slower but simpler).

### Per-Engine-venv vs. Per-Engine-Container

|   | Per-Engine venvs (in one container) | Per-Engine Containers |
|---|:---:|:---:|
| **Containers to manage** | 1 | 28 |
| **docker-compose complexity** | Low (~40 lines) | Medium (~200 lines) |
| **RAM overhead** | ~2 GB (one Python process) | ~4–5 GB (28 processes) |
| **Isolation** | Good (separate venvs) | Perfect (separate containers) |
| **Dependency pinning** | Per-venv requirements.txt | Per-Dockerfile |
| **GPU sharing** | All in one container, one CUDA context | Each container gets its own CUDA context |
| **Subprocess overhead** | ~500ms per legacy synthesis call | ~10ms HTTP call |
| **Best for** | Single admin, one VM, simpler ops | Multi-machine, frequent engine additions |

### Recommendation

- **Start with per-engine-venvs** during initial Docker migration — it's the smallest change from the current architecture
- **Graduate to the 6-container approach** when you want full isolation without the overhead of 28 images
- **The approaches build on each other** — venv (1 container) → 6 containers (current + legacy + orpheus + 3 SGLang) → add containers only when genuinely needed

---

## Part 4c: Expert Review — Concerns & Refinements

> The architecture was reviewed by an independent expert. Here are their key concerns and how they affect the plan.

### Concern 1: Verify Legacy Stack Versions Before Freezing

**The concern:** The plan assumes parler needs torch 1.13. This may be too conservative. Parler might work on torch 2.1 + transformers 4.46 — a much more maintainable stack.

**Action before building Dockerfiles:**

```bash
# Test if the 3 broken engines work on a middle-ground stack
python3.10 -m venv /tmp/legacy-test
source /tmp/legacy-test/bin/activate
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.1

# Test each engine
pip install git+https://github.com/index-tts/IndexTTS.git
python -c "from indextts import IndexTTS; print('indextts OK')"

pip install parler-tts==0.2.3
python -c "from parler_tts import ParlerTTS; print('parler OK')"

pip install qwen-tts
python -c "from qwen_tts import Qwen3TTS; print('qwen3tts OK')"
```

**If all 3 pass on torch 2.1:** Use torch 2.1 + tf 4.46 as the legacy stack — much closer to the current stack, easier to maintain, same CUDA 11.8 base.

**If only parler fails on torch 2.1:** Try torch 2.0 for parler only. Goal: find the HIGHEST torch version that works, not the lowest.

### Concern 2: Reduce Monkey Patches After Containerization

**The concern:** `tts_lab_shims.py` contains emergency fixes that should not become permanent architecture:

- `inspect.getsourcefile` patch
- `transformers.masking_utils` stub
- `torch._dynamo._trace_wrapped_higher_order_op` stub
- Fake `torchcodec` metadata

**Principle:** Containerization should let you run the versions engines expect, reducing the need for patches.

**After containerization, audit each patch:**

| Patch | Can It Be Removed? | How |
|-------|:---:|------|
| inspect.getsourcefile | Possibly | Only needed for torch 2.10 + Python 3.11 combo. Legacy container has torch 1.13 — doesn't need it. |
| masking_utils stub | ✅ Yes | Needed because tf 5.x removed it. Legacy container runs tf 4.46 which HAS it. |
| _trace_wrapped_higher_order_op | Possibly | Torch 2.10 bug. May be fixed in later torch versions. |
| torchcodec metadata | ❌ No | Chatterbox checks for `torchcodec` package metadata. Until chatterbox drops this check, the stub stays. |

**Goal:** Each container should have only the patches it actually needs. The legacy container should need ZERO patches — it runs the versions the engines were built for.

### Concern 3: No nginx Needed

**The concern:** The plan mentioned a reverse proxy. For an internal lab deployment, the FastAPI server already serves the UI on port 8001. Adding nginx adds a moving part with no benefit unless you need TLS termination or rate limiting.

**Decision:** Removed nginx from the architecture. The main FastAPI server IS the orchestrator. It imports 21 engines in-process and routes the other 7 via HTTP to their containers.

---

## Part 4d: Patch Lifecycle — What Happens After Containerization

> The TTS Lab currently has ~15 monkey-patches in `tts_lab_shims.py`. Containerization changes their lifecycle dramatically.

### The Principle

**Patches exist because you forced a newer tf/torch on engines that need older ones.** Containerization removes that forcing. The legacy container runs the exact versions the legacy engines were built for. It should need zero patches.

### Where Patches Live After Containerization

```
Before (bare metal):               After (containerized):
                                   
tts_lab_shims.py                   docker/Dockerfile.stack.current
(15 patches, all engines)          ├── RUN: torchcodec metadata stub
                                   ├── RUN: patch_transformers_stubs.py
                                   ├── RUN: fix_transformers_shims.py
                                   └── RUN: patch_parler_tts.py
                                   
                                   tts_lab_shims.py
                                   ├── inspect.getsourcefile fix
                                   ├── masking_utils stub
                                   ├── _trace_wrapped_higher_order_op stub
                                   ├── isin_mps_friendly stub
                                   ├── ExtensionsTrie, AddedToken stubs
                                   ├── check_model_inputs compat
                                   ├── PretrainedConfig defaults
                                   ├── perth watermarker stub
                                   └── thread-pool env vars
                                   ↑ Imported ONLY by engine-current
                                   
                                   tts_lab_shims_legacy.py (NEW, minimal)
                                   └── thread-pool env vars (only if needed)
                                   ↑ Imported by engine-legacy — likely EMPTY
                                   
                                   (nothing)
                                   ↑ Orpheus container — no tf, no patches needed
```

### Patch-by-Patch: Engine-Current (torch 2.10 + tf 5.12)

| Patch | Why It Exists | Fate After Containerization | On Stack Upgrade |
|-------|---------------|----------------------------|------------------|
| `inspect.getsourcefile` | torch 2.10 bug on Python 3.11 | **Stays** — needed as long as torch 2.10 is used | torch 2.11 fixed it? Remove. Still broken? Keep. |
| `masking_utils` stub | Removed in tf 5.x | **Stays** — engines reference it, it's gone | Never coming back. Permanent until engines update. |
| `_trace_wrapped_higher_order_op` | torch 2.10 import crash | **Stays** | May be fixed in future torch. Test on each upgrade. |
| `isin_mps_friendly` | Removed in tf 5.x | **Stays** | Permanent until engines migrate to the new API. |
| `ExtensionsTrie`, `AddedToken` | Removed in tf 5.x | **Stays** | Used by indextts — but indextts is in legacy! Test: does any current engine need these? If not, remove. |
| `torch.isin` wrapper | parler needs old API | **Stays for now** | parler is in legacy. Test: does any current engine use this? If not, remove. |
| `check_model_inputs` compat | Signature changed in tf 5.x | **Stays** | Used by qwen3tts — which is in legacy. Audit, possibly remove. |
| `PretrainedConfig` defaults | Attrs removed in tf 5.x | **Stays** | May affect parler/qwen3tts config loading. Both in legacy now. Audit. |
| `torchcodec` metadata | chatterbox checks for it | **Build-time** (`Dockerfile.stack.current` RUN step) | Remove when chatterbox drops the check. |
| `perth` watermarker stub | chatterbox depends on perth | **Stays** — runtime stub | Remove when perth fixes. |
| Thread-pool env vars | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. | **Stays** | Always needed. Not a hack — system configuration. |
| `patch_parler_tts.py` | parler + tf 5.x incompatibility | **Stays** — applied at build time | parler is in legacy now. But the patch modifies parler_tts site-packages. If parler is in legacy, this patch may not be needed in current. **Audit after split.** |
| `librosa`/`soundfile` compatibility layer | API changes across versions | **Stays** | Always needed — not version-specific. |
| Deep recursion limit (`sys.setrecursionlimit(10000)`) | chattterbox import chain | **Stays** | May not be needed in older torch. Test and tune per container. |

### Patch-by-Patch: Engine-Legacy (torch 1.13 + tf 4.46)

| Patch | Needed? | Why |
|-------|:---:|------|
| `masking_utils` stub | ❌ **No** | `masking_utils` EXISTS in tf 4.46 |
| `isin_mps_friendly` | ❌ **No** | EXISTS in tf 4.46 |
| `ExtensionsTrie`, `AddedToken` | ❌ **No** | EXIST in tf 4.46 |
| `download_url` | ❌ **No** | EXISTS in tf 4.46 |
| `find_pruneable_heads_and_indices` | ❌ **No** | EXISTS in tf 4.46 |
| `PretrainedConfig.pad_token_id` | ❌ **No** | EXISTS in tf 4.46 config objects |
| `check_model_inputs` compat | ❌ **No** | Old signature is the default in tf 4.46 |
| `_trace_wrapped_higher_order_op` | ❌ **No** | torch 1.13 doesn't have this import chain |
| `inspect.getsourcefile` | ❌ **No** | torch 1.13 + Python 3.11 doesn't trigger this crash |
| `torch.isin` wrapper for parler | ⚠️ Maybe | Depends on exact torch 1.13 API. Test. |
| `patch_parler_tts.py` | ⚠️ Maybe | parlor was designed for tf 4.x. May not need patching at all with tf 4.46. |
| Thread-pool env vars | ✅ Yes | Always needed — system configuration. |
| `librosa`/`soundfile` compatibility | ✅ Yes | If used by any legacy engine. |

**Target: 2–3 patches max in legacy, down from ~15.** The ones that remain are system configuration (thread pools, audio compat), not framework monkey-patches.

### Stack Upgrade Workflow

Containerization changes the upgrade cycle from "one big bang" to "tested, incremental, reversible":

```
1. Edit Dockerfile.stack.current
   Change: torch 2.10 → 2.11, tf 5.12 → 5.13

2. docker build -f docker/Dockerfile.stack.current -t tts-lab-stack:current-test .

3. docker run --rm -v /opt/models:/opt/models tts-lab-stack:current-test \
      python -c "from tts_lab_dispatch import _sweep_availability; _sweep_availability()"
   → 18 engines pass, 3 fail (let's say f5tts, dia, zonos)

4. Investigate the 3 failures:
   - f5tts: needs a version pin → add to Dockerfile.engine-current
   - dia: needs a new compatibility patch → add to tts_lab_shims.py
   - zonos: known issue, fix already in zonos upstream → upgrade zonos pip version

5. Rebuild, retest → 21/21 pass

6. docker tag tts-lab-stack:current-test tts-lab-stack:current
   docker compose up -d engine-current   # Rolling update, zero downtime

7. The legacy container was NEVER touched. torch 1.13 + tf 4.46, sealed forever.
```

### Key Difference

| | Today (Bare Metal) | After Containerization |
|---|:---:|:---:|
| Number of patches | ~15 applied to EVERYTHING | ~10 in current, ~2 in legacy, ~0 in orpheus |
| Patch surface | All 28 engines inherit all patches | Each container only gets patches it needs |
| Upgrade impact | One `pip install` can break anything | Stack rebuild + automated sweep → only the rebuilt container |
| Legacy stability | Fragile — patches must satisfy both old engines AND new tf | Sealed capsule — runs the versions engines were built for |
| New engine risk | Can conflict with existing patch assumptions | Only affects its own container |
| Rollback | Reinstall old packages, re-apply old patches | `docker compose up -d engine-current` (previous image tag, instant) |
| Patch audit | Manual, error-prone | `grep -r "patch\|stub\|fix"` per Dockerfile — scoped and reviewable |

### 5.1 Prerequisites

On your Ubuntu 22.04 VM:

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# Log out and back in for group membership to take effect

# 2. Install NVIDIA Container Toolkit (GPU machines only)
# Skip this step if no GPU is available
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify Docker works
docker run hello-world

# 4. Verify GPU access (GPU machines only)
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# 5. Clone the repo
cd /opt
git clone https://github.com/farid-nasiri/TTS-LAB tts-lab-docker
cd tts-lab-docker
```

### 5.2 First-Time Build

```bash
# Build the base image first (120 MB, ~2 minutes)
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .

# Build stack v1 (1.2 GB, ~10 minutes)
docker build -f docker/Dockerfile.stack.v1 -t tts-lab-stack:v1 .

# Build all engine images (24 engines, ~15 minutes total)
# These build in parallel — slowest engine sets the pace
docker compose build

# If you have a GPU, build the CUDA stack and Orpheus
docker build -f docker/Dockerfile.orpheus -t tts-lab-orpheus:latest .
```

**Total first build time: ~25–35 minutes** (mostly downloading pip packages).

The good news: after the first build, only changed layers rebuild. Adding a new engine: ~2 minutes. Changing app code: ~30 seconds.

### 5.3 Starting the Lab

```bash
# Default: main (21 engines) + legacy (3 engines) — all CUDA-capable
docker compose up -d

# + Orpheus (the only GPU-mandatory engine):
docker compose --profile gpu up -d

# + SGLang engines (VibeVoice, Higgs, S2-Pro):
docker compose --profile gpu --profile sglang up -d

# Watch logs during startup:
docker compose logs -f

# Check which engines are healthy:
docker compose ps
```

### 5.4 Verifying Everything Works

```bash
# 1. Check the orchestrator UI
curl http://localhost:8001/
# Should return the HTML UI

# 2. Check engine availability
curl http://localhost:8001/status | python -m json.tool
# Lists all 28 engines with their availability status

# 3. Quick synthesis test (Piper — fastest engine)
curl -X POST http://localhost:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Docker deployment is operational.","params":{}}' \
  -o /tmp/test.wav
ffprobe /tmp/test.wav
# Should show a valid WAV file with audio

# 4. Check individual engine health
curl http://localhost:8100/health    # Piper
curl http://localhost:8002/health    # Orpheus (if GPU profile active)
```

---

## Part 6: Day-to-Day Maintenance

### 6.1 Adding a New Engine

You found a new TTS engine called "NewTTS" on PyPI. It works with the current stack v1.

**Step 1:** Create `docker/engines/Dockerfile.newtts`:

```dockerfile
FROM tts-lab-stack:v1
LABEL tts-lab.engine="newtts"
RUN pip install --no-cache-dir newtts>=1.0.0
EXPOSE 8100
CMD ["uvicorn", "tts_lab_engine_server:app", "--host", "0.0.0.0", "--port", "8100"]
```

**Step 2:** Add engine config to `tts_lab_config.py`:

```python
# Add to MODEL_ORDER list
MODEL_ORDER = [..., "newtts"]

# Add to MODEL_INFO dict
MODEL_INFO["newtts"] = {
    "label": "NewTTS v1",
    "ram_est_mb": 1500,
    "heavy": True,
    "supports_voice_cloning": False,
    "languages": ["en"],
}
```

**Step 3:** Add the container to `docker-compose.yml`:

```yaml
  newtts:
    build:
      context: .
      dockerfile: docker/engines/Dockerfile.newtts
    image: tts-lab-newtts:latest
    container_name: tts-lab-newtts
    <<: *engine-defaults
```

And add the URL to the orchestrator's environment:

```yaml
    environment:
      # ... existing URLs ...
      NEWTTS_URL: http://newtts:8100
```

**Step 4:** Rebuild and restart just this engine:

```bash
docker compose build newtts
docker compose up -d newtts
```

**What did NOT happen:**
- ❌ The other 27 engines were not touched, rebuilt, or restarted
- ❌ No risk of breaking existing engines
- ❌ No dependency conflict resolution needed
- ❌ No "pip install" on your host machine

**Total work: ~5 minutes. Risk to existing engines: ZERO.**

### 6.2 Updating an Engine's Dependencies

Chatterbox released version 0.3.0 with a bug fix. You want to update.

```bash
# 1. Edit the engine Dockerfile (change version pin)
# docker/engines/Dockerfile.chatterbox:
#   RUN pip install chatterbox-tts>=0.3.0 perth>=1.1.0

# 2. Rebuild just this engine
docker compose build chatterbox

# 3. Test it
docker compose up -d chatterbox
curl -X POST http://localhost:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"Testing the update.","params":{}}' -o /tmp/test.wav

# 4. If the test sounds wrong, roll back:
git checkout docker/engines/Dockerfile.chatterbox
docker compose build chatterbox
docker compose up -d chatterbox
```

Only the Chatterbox container was restarted. The other 27 engines were unaffected.

### 6.3 Adding a New ML Stack

A new engine called "HotTTS" requires `torch>=2.6.0` and `numpy>=2.0`. This conflicts with stack v1 (which has torch 2.2 and numpy 1.26). You need stack v2.

```bash
# 1. Stack v2 Dockerfile already exists at docker/Dockerfile.stack.v2
#    (It was created as a future-proof template.)

# 2. Build stack v2
docker build -f docker/Dockerfile.stack.v2 -t tts-lab-stack:v2 .

# 3. Create the engine Dockerfile
# docker/engines/Dockerfile.hottts:
#   FROM tts-lab-stack:v2
#   LABEL tts-lab.engine="hottts"
#   LABEL tts-lab.stack="v2"
#   RUN pip install --no-cache-dir hottts
#   EXPOSE 8100
#   CMD ["uvicorn", "tts_lab_engine_server:app", "--host", "0.0.0.0", "--port", "8100"]

# 4. Build and run
docker compose build hottts
docker compose up -d hottts
```

**What happened:**
- Stack v2 (1.2 GB) was built — it's a new set of layers on disk
- The hottts engine image (thin, ~100 MB) was built on top of v2
- Stack v1 and its 24 engines were NOT touched
- HotTTS has its own isolated environment with torch 2.6 and numpy 2.x
- Engine #25 (hottts) and engine #1 (piper) run in different containers with different stacks

**Disk cost: ~1.3 GB (stack v2 + engine layer). Risk: ZERO.**

### 6.4 Migrating an Engine to a New Stack

You've tested that Chatterbox works fine with stack v2. You want to move it from v1 to v2.

```bash
# 1. Change the FROM line in the engine Dockerfile
#    FROM tts-lab-stack:v1  →  FROM tts-lab-stack:v2

# 2. Rebuild
docker compose build chatterbox

# 3. Test thoroughly
docker compose up -d chatterbox
curl -X POST http://localhost:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"Comprehensive test sentence with various phonemes.","params":{}}' \
  -o /tmp/test.wav

# 4. If it works, keep it. If not, revert the FROM line and rebuild.
```

Over time, as engines prove compatible with v2, migrate them one by one. When all engines have moved, archive stack v1. This is a gradual, low-risk migration — not a flag day.

### 6.5 Removing an Engine

An engine is permanently broken or no longer needed.

```bash
# 1. Remove the Dockerfile
rm docker/engines/Dockerfile.old-engine

# 2. Remove from docker-compose.yml (delete the service block)

# 3. Remove from tts_lab_config.py (delete from MODEL_ORDER and MODEL_INFO)

# 4. Stop and remove the container
docker compose stop old-engine
docker compose rm old-engine

# 5. Optionally delete the image
docker rmi tts-lab-old-engine:latest
```

The engine's model files in `/opt/models/` remain — delete them manually if you want the disk space back.

### 6.6 Monitoring & Debugging

```bash
# See all running containers and their status
docker compose ps

# Tail logs from all containers
docker compose logs -f

# Tail logs from one engine
docker compose logs -f chatterbox

# See logs from the last 5 minutes
docker compose logs --since 5m

# Check if an engine's health endpoint is responding
curl http://localhost:8100/health

# Get a shell inside a running container (for debugging)
docker exec -it tts-lab-chatterbox bash

# Check disk usage of Docker images
docker images | grep tts-lab
docker system df

# Check container resource usage (CPU, RAM)
docker stats --no-stream

# Restart a misbehaving engine
docker compose restart chatterbox

# Completely rebuild and restart one engine
docker compose build --no-cache chatterbox
docker compose up -d chatterbox
```

---

## Part 7: Edge Cases & Special Situations

### 7.1 GPU Engines

**Problem:** Orpheus, VibeVoice, Higgs, and S2-Pro need a GPU. Your VM doesn't have one.

**Solution:** The GPU containers are in Docker Compose **profiles**. They are not started by default.

```bash
# Default (no GPU-mandatory containers) — Orpheus/SGLang show "unavailable"
docker compose up -d

# GPU deployment — add profiles for whatever fits in VRAM
docker compose --profile gpu up -d           # Orpheus
docker compose --profile sglang up -d        # SGLang engines
docker compose --profile gpu --profile sglang up -d  # All 28 engines
```

The orchestrator probes each engine URL at startup. If an engine URL is configured but the container is not running, that engine shows as "unavailable — container not running" in the UI. This is graceful degradation — the other engines work normally.

**Running GPU engines on a different machine:** Set the `*_URL` environment variables on the orchestrator to point to the GPU machine's IP:

```yaml
# In docker-compose.yml, orchestrator environment:
ORPHEUS_URL: http://192.168.0.99:8002
VIBEVOICE_SGLANG_URL: http://192.168.0.99:8003/v1/audio/speech
```

Then start the GPU containers on that machine. The orchestrator (on the CPU VM) routes to them over the network.

### 7.2 Gated HuggingFace Models

**Problem:** Orpheus, CSM, and Qwen3-TTS models are "gated" on HuggingFace — you need to request access and authenticate.

**Solution:** Pass `HF_TOKEN` through docker-compose:

```bash
# Set the token on your host
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Start with the token
HF_TOKEN=$HF_TOKEN docker compose --profile gpu up -d
```

Or store it in a `.env` file (never commit this to git):

```bash
# .env (gitignored)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Then in docker-compose:
```yaml
environment:
  HF_TOKEN: ${HF_TOKEN:-}
```

The `${HF_TOKEN:-}` syntax means "use the HF_TOKEN variable if set, otherwise empty string." If not set, the engine shows "gated — run huggingface-cli login" in the UI.

### 7.3 SGLang External Servers

**Problem:** SGLang containers (`lmsysorg/sglang-omni:dev`) are 8 GB images from Docker Hub. They need to download 6–10 GB of model weights on first run.

**Solution:** The model weights go to the shared `/opt/models/huggingface` volume. They download once — subsequent starts are fast.

First startup of an SGLang container:
```bash
# This will download the model (~6 GB for VibeVoice):
docker compose --profile sglang up -d vibevoice
docker logs -f tts-lab-vibevoice
# Wait for: "INFO: Started server process" and "Uvicorn running on..."
# This can take 10–20 minutes on first run (downloading + loading model).
```

Subsequent startups: ~30 seconds (model cached in volume).

### 7.4 Disk Is Full

**Problem:** Docker images and model files together can exceed available disk space.

**Diagnosis:**
```bash
# What's using disk?
docker system df -v
du -sh /opt/models/*
df -h
```

**Cleanup strategies (least to most aggressive):**

```bash
# 1. Remove old unused images
docker image prune -a
# Frees build cache and untagged images (~5 GB typical)

# 2. Remove all unused Docker data
docker system prune -a --volumes
# WARNING: This deletes ALL unused images, containers, volumes, and caches.
# Only run this if you're sure.

# 3. Remove models for engines you never use
# Delete specific model directories from /opt/models/huggingface/hub/
rm -rf /opt/models/huggingface/hub/models--suno--bark  # Example: 2.5 GB

# 4. Use small model variants where available
# Set in docker-compose environment:
SUNO_USE_SMALL_MODELS: "True"   # Bark: 1.3 GB instead of 2.5 GB
# For OuteTTS, use the Q4_K_M GGUF (384 MB) instead of Q8 (650 MB)
```

**Prevention:**
- Mount `/opt/models` on a dedicated disk (your deploy script already does this)
- Set up a cron job to prune old Docker images monthly
- Use `docker builder prune --filter "until=720h"` to keep 30 days of build cache

### 7.5 Engine Crashes on Startup

**Problem:** An engine container exits immediately after starting.

**Diagnosis:**
```bash
# Check the container status
docker compose ps
# Look for "Exited (1)" or "Restarting"

# See the crash logs
docker compose logs chatterbox --tail 50

# Common causes and fixes:
# "ModuleNotFoundError: No module named 'xxx'"
#   → Missing pip package. Add to engine Dockerfile. Rebuild.
#
# "CUDA out of memory"
#   → GPU VRAM exhausted. Stop other GPU containers first.
#
# "Permission denied: '/opt/models/...'"
#   → Volume permission issue. chmod 777 /opt/models or use named volume.
#
# "Connection refused" (health check)
#   → Model is still loading. Increase start_period in healthcheck.
```

**Auto-recovery:** All containers have `restart: unless-stopped`. If an engine crashes, Docker restarts it automatically. If it keeps crashing (restart loop), Docker backs off exponentially.

### 7.6 Multiple GPUs

**Problem:** You have 2 GPUs (e.g., two RTX 4090s). You want Orpheus on GPU 0 and SGLang engines on GPU 1.

**Solution:** Use `CUDA_VISIBLE_DEVICES` to pin containers to specific GPUs:

```yaml
# In docker-compose.yml:
orpheus:
  environment:
    CUDA_VISIBLE_DEVICES: "0"    # Only sees GPU 0

vibevoice:
  environment:
    CUDA_VISIBLE_DEVICES: "1"    # Only sees GPU 1

higgs:
  environment:
    CUDA_VISIBLE_DEVICES: "1"    # Shares GPU 1 with vibevoice
```

This way, Orpheus gets dedicated VRAM on GPU 0, and the SGLang engines share GPU 1 (you decide which ones to run based on available VRAM).

### 7.7 Air-Gapped / No-Internet Deployment

**Problem:** Your server has no internet access. You need to deploy from a USB drive or internal network.

**Solution:** Build on a machine with internet, then transfer:

```bash
# On internet-connected build machine:
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
docker build -f docker/Dockerfile.stack.v1 -t tts-lab-stack:v1 .
# Build all engine images...

# Export images to files:
docker save -o tts-lab-images.tar \
  tts-lab-base:latest \
  tts-lab-stack:v1 \
  tts-lab-piper:latest \
  tts-lab-kokoro:latest \
  # ... (all engine images)

# Compress (5 GB → ~1.5 GB):
gzip tts-lab-images.tar

# Transfer to air-gapped server (USB, internal network, etc.)
# On air-gapped server:
gunzip tts-lab-images.tar.gz
docker load -i tts-lab-images.tar

# Models need to be transferred separately:
# Copy /opt/models/ from the internet-connected machine
# to /opt/models/ on the air-gapped server
tar -czf models.tar.gz /opt/models/
# Transfer and extract on destination

# Start the lab:
docker compose up -d
```

---

## Part 8: Image Hosting & Distribution

### 8.1 Where to Host

| Registry | Free Storage (Public) | Pull Limits | Why Use |
|----------|:---:|:---:|------|
| **GitHub Container Registry (GHCR)** | **Unlimited, free** | None for public | Same platform as your code. Best choice. |
| Docker Hub | Unlimited | 100/hr (authenticated), 10/hr (anonymous) | Most users already have Docker Hub access |
| Quay.io (Red Hat) | Unlimited, free | None | Good alternative if GHCR has issues |

**Recommended: GHCR.** Your repo is already on GitHub. Images live next to code. One `GITHUB_TOKEN` authenticates everything.

Image naming on GHCR:
```
ghcr.io/farid-nasiri/tts-lab-base:latest
ghcr.io/farid-nasiri/tts-lab-stack:v1
ghcr.io/farid-nasiri/tts-lab-piper:latest
ghcr.io/farid-nasiri/tts-lab-kokoro:latest
...
```

### 8.2 Push/Pull Sizes

| Image | Uncompressed | Compressed (push/pull) | How Often |
|-------|:---:|:---:|:---:|
| `tts-lab-base` | 120 MB | ~40 MB | Every 6–12 months |
| `tts-lab-stack:v1` | 1.2 GB | ~400 MB | Every 3–6 months |
| `tts-lab-stack:v2` | 1.2 GB | ~400 MB | When created |
| Each engine image | 30–300 MB | ~10–100 MB | When engine updates |
| `tts-lab-orpheus` | 7.0 GB | ~2.5 GB | When vllm/Orpheus updates |
| SGLang (upstream) | 8.0 GB | ~3 GB | Pulled from Docker Hub, not re-hosted |

**Typical CI pipeline push:**
- Code change only → pushes base (40 MB compressed) + the engine images with code (10 MB each)
- Stack upgrade → pushes stack v1 (400 MB) + all engine images that inherited from it
- New engine → pushes the new engine image (10–100 MB)

### 8.3 CI/CD Pipeline

`.github/workflows/build-images.yml`:

```yaml
name: Build and Push Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'docker/**'
      - 'tts_lab_*.py'
      - 'patches/**'
      - '.github/workflows/build-images.yml'
  workflow_dispatch:     # Allow manual trigger

jobs:
  # ── Detect what changed ────────────────────────────────────────
  detect-changes:
    runs-on: ubuntu-22.04
    outputs:
      base_changed: ${{ steps.filter.outputs.base }}
      stack_v1_changed: ${{ steps.filter.outputs.stack_v1 }}
      engines_changed: ${{ steps.filter.outputs.engines }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            base:
              - 'docker/Dockerfile.base'
              - 'tts_lab_*.py'
              - 'patches/**'
            stack_v1:
              - 'docker/Dockerfile.stack.v1'
            engines:
              - 'docker/engines/**'

  # ── Build & push base image ────────────────────────────────────
  build-base:
    needs: detect-changes
    if: needs.detect-changes.outputs.base_changed == 'true'
    runs-on: ubuntu-22.04
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.base
          push: true
          tags: ghcr.io/${{ github.repository }}/tts-lab-base:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Build & push stack images ──────────────────────────────────
  build-stack-v1:
    needs: [detect-changes, build-base]
    if: |
      always() &&
      (needs.detect-changes.outputs.stack_v1_changed == 'true' ||
       needs.detect-changes.outputs.base_changed == 'true')
    runs-on: ubuntu-22.04
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.stack.v1
          push: true
          tags: ghcr.io/${{ github.repository }}/tts-lab-stack:v1
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Build & push engine images (matrix build — parallel) ───────
  build-engines:
    needs: [detect-changes, build-stack-v1]
    if: always() && !cancelled()
    runs-on: ubuntu-22.04
    permissions:
      packages: write
    strategy:
      fail-fast: false     # One engine failing doesn't cancel others
      matrix:
        engine:
          - piper
          - kokoro
          - melo
          - chattts
          - outetts
          - bark
          - styletts2
          - f5tts
          - dia
          - xtts
          - cosyvoice
          - parler
          - chatterbox
          - chatterboxturbo
          - fishspeech
          - csm
          - qwen3tts
          - indextts
          - zonos
          - openvoice
          - matcha
          - manatts
          - omnivoice
          - vibevoice
          - higgs
          - s2pro
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/engines/Dockerfile.${{ matrix.engine }}
          push: true
          tags: ghcr.io/${{ github.repository }}/tts-lab-${{ matrix.engine }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**CI/CD behavior:**
- Push to `main` → GitHub Actions runs
- `detect-changes` figures out which Dockerfiles were modified
- Only changed images rebuild — unchanged images skip
- Engine images build in parallel (matrix strategy)
- All push to GHCR automatically

**GitHub Actions limits:** 2,000 free minutes/month for public repos. A full build of everything is ~300 minutes (most of it parallelized). You can do ~6 full rebuilds per month on the free tier. Incremental builds (single engine) take ~5 minutes.

---

## Part 9: Migration from Bare Metal

Your current setup is a systemd service on Ubuntu 22.04 at `192.168.0.87:8001`. Here's how to migrate with minimal risk:

### Phase 1: Preparation (day 1, 30 min)

```bash
# Install Docker alongside the running service
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker arthur
# Log out and back in

# Clone the repo with Docker files
cd /opt
git clone https://github.com/farid-nasiri/TTS-LAB tts-lab-docker
cd tts-lab-docker

# Build all images (runs in background, doesn't affect the running service)
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
docker build -f docker/Dockerfile.stack.v1 -t tts-lab-stack:v1 .
docker compose build
# This takes ~30 min but the systemd service on port 8001 keeps running.
```

### Phase 2: Test on alternate port (day 1, 10 min)

```bash
# Start the Docker deployment on port 8009 for testing
# (Edit docker-compose.yml temporarily: change "8001:8001" to "8009:8001")
docker compose up -d

# Test the Docker deployment without affecting the live service
curl http://localhost:8009/status
curl -X POST http://localhost:8009/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Docker test.","params":{}}' -o /tmp/docker-test.wav

# Is everything working? Proceed to cutover.
# Is something wrong? Debug, then: docker compose down
```

### Phase 3: Cutover (day 2, 2 min downtime)

```bash
# 1. Stop the old systemd service
sudo systemctl stop arthur-lab

# 2. Fix the port back to 8001 in docker-compose.yml

# 3. Start the Docker deployment on 8001
docker compose up -d

# 4. Verify
curl http://localhost:8001/status
curl http://localhost:8001/

# 5. If something went wrong — instant rollback:
docker compose down
sudo systemctl start arthur-lab
# Total downtime: ~2 minutes

# 6. If all good, disable the old service:
sudo systemctl disable arthur-lab
```

### Phase 4: Cleanup (day 8, after confirming stability)

```bash
# Remove the old venv (frees ~5 GB)
rm -rf /opt/arthur-bench-env

# Remove the old systemd service file
sudo rm /etc/systemd/system/arthur-lab.service
sudo systemctl daemon-reload

# Prune Docker build cache (frees ~5 GB)
docker builder prune -a
```

---

## Part 10: Quick Reference

### 10.1 Command Cheatsheet

```bash
# ── Building ─────────────────────────────────────────────────────
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .  # Build base
docker build -f docker/Dockerfile.stack.v1 -t tts-lab-stack:v1 . # Build stack
docker compose build                          # Build all engines
docker compose build chatterbox               # Build one engine

# ── Running ──────────────────────────────────────────────────────
docker compose up -d                          # Start all (CPU)
docker compose --profile gpu up -d            # Start CPU + Orpheus
docker compose --profile sglang up -d         # Start CPU + SGLang
docker compose up -d piper kokoro             # Start specific engines
docker compose down                           # Stop everything
docker compose restart chatterbox             # Restart one engine

# ── Monitoring ───────────────────────────────────────────────────
docker compose ps                             # List containers + health
docker compose ps -a                          # Include stopped containers
docker compose logs -f                        # All logs (follow)
docker compose logs chatterbox --tail 50      # One engine, last 50 lines
docker stats --no-stream                      # CPU/RAM per container

# ── Debugging ────────────────────────────────────────────────────
docker exec -it tts-lab-chatterbox bash       # Shell in container
docker inspect tts-lab-chatterbox             # Full container config
curl http://localhost:8100/health             # Engine health check
docker compose down && docker compose up -d   # Hard restart everything

# ── Cleanup ──────────────────────────────────────────────────────
docker image prune -a                         # Remove unused images
docker builder prune                          # Remove build cache
docker system df                              # Disk usage summary
docker system prune -a --volumes              # Nuclear option — clean everything
```

### 10.2 Glossary

| Term | What It Means |
|------|---------------|
| **Image** | A read-only blueprint for creating containers. Built from a Dockerfile. |
| **Container** | A running instance of an image. Isolated, lightweight, ephemeral. |
| **Dockerfile** | A text recipe that defines how to build an image. Each line creates a layer. |
| **Layer** | One step in a Dockerfile. Layers are cached and shared between images. |
| **Volume** | A directory on the host mounted into a container. Outlives the container. |
| **Registry** | A server that stores Docker images. Like GitHub for images. |
| **GHCR** | GitHub Container Registry — free, unlimited storage for public images. |
| **FROM** | Dockerfile instruction to inherit from a parent image. The foundation of the tiered architecture. |
| **Build context** | The directory sent to Docker during build. Usually `.` (current directory). |
| **Docker Compose** | Tool for defining and running multi-container applications. |
| **Profile** | A Compose feature to conditionally include services (`--profile gpu`). |
| **Orchestrator** | The main container that serves the web UI and routes synthesis to engine containers. |
| **Stack** | A versioned set of ML libraries (PyTorch + numpy + transformers + etc.). |
| **Tier** | A level in the inheritance hierarchy (Tier 1 = base, Tier 2 = stack, Tier 3 = engine). |
| **Healthcheck** | A command Docker runs periodically to check if a container is healthy. |
| **`restart: unless-stopped`** | Docker automatically restarts a crashed container unless you explicitly stop it. |
| **`depends_on`** | Compose directive to control startup order (but does NOT wait for readiness — use healthchecks for that). |

### 10.3 Disk Budget Summary

| What | Size | Notes |
|------|:----:|-------|
| `tts-lab-base` (Tier 1) | ~1.5 GB | CUDA 12.8 base + espeak + ffmpeg + app code |
| `tts-lab-stack:current` (Tier 2) | ~3.5 GB | torch 2.10 + transformers 5.12 (shared by 21 engines) |
| `tts-lab-stack:legacy` (Tier 2) | ~2.5 GB | torch 1.13 + transformers 4.46 (shared by 3 engines) |
| `tts-lab-stack:cuda` (Tier 2, Orpheus) | ~3.5 GB | torch CUDA 12.1 + vllm (one engine) |
| 2 engine images (Tier 3) | ~5 GB + ~4 GB | engine-current (~5 GB with 21 engines), engine-legacy (~4 GB with 3 engines) |
| SGLang base (pre-built) | ~8.0 GB | Shared by 3 SGLang containers |
| Model files (shared volume) | ~38.5 GB | `/opt/models/` — mounted, not in images |
| Docker build cache | ~5 GB | Can be pruned |
| **Grand total** | **~66 GB** | Images (~23 GB) + models (~38 GB) + cache (~5 GB) |
| **Minimum (without SGLang/Orpheus)** | **~48 GB** | Current + legacy stacks + models |

| Stack | Inherits From | Engines |
|-------|:---:|---------|
| `current` | `tts-lab-base` | 21 engines in ONE container (engine-current) |
| `legacy` | `tts-lab-base` | 3 engines in ONE container (engine-legacy) |
| `cuda` | `nvidia/cuda:12.1` (separate base) | 1 engine (orpheus) |
| SGLang (pre-built) | `lmsysorg/sglang-omni:dev` | 3 engines in 3 containers |

**Key difference from original plan:** 6 containers, not 28 images. Per expert review — per-engine images are overengineering for a single-VM lab. 21 engines coexist in one container (same Python process), exactly as they do today. Only 3 legacy engines + Orpheus + 3 SGLang get separate containers.

---

> **End of Master Plan.** This document is a living reference. Update it when the architecture changes or new patterns emerge.
