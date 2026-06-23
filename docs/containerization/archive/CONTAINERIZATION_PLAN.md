# TTS Lab Containerization — Deep Evaluation & Plan

> **Date:** 2026-06-19 (updated same day from [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md) findings)
> **Status:** Plan — not yet implemented
> **Scope:** 28 TTS engines, dependency isolation via Docker
> **Related Docs:**
> - [CONTAINERIZATION_MASTER_PLAN.md](CONTAINERIZATION_MASTER_PLAN.md) — 2,000-line guide for people new to Docker (concepts, all Dockerfiles, maintenance, edge cases, cheatsheet)
> - [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md) — Current engine status, fixes applied, the 3 broken engines

---

## Table of Contents

1. [Current State: The Dependency Problem](#1-current-state-the-dependency-problem)
2. [Containerization Options](#2-containerization-options)
3. [Recommended Architecture: Option D — Pragmatic Hybrid](#3-recommended-architecture-option-d--pragmatic-hybrid)
4. [Detailed Resource Breakdown](#4-detailed-resource-breakdown)
5. [Dockerfile Designs](#5-dockerfile-designs)
6. [Orchestration: docker-compose.yml](#6-orchestration-docker-composeyml)
7. [Code Changes Required](#7-code-changes-required)
8. [Model Storage Strategy](#8-model-storage-strategy)
9. [Total Disk Budget](#9-total-disk-budget)
10. [Migration Path: Zero-Downtime Cutover](#10-migration-path-zero-downtime-cutover)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [Recommendation Summary](#12-recommendation-summary)

---

## 1. Current State: The Dependency Problem

### 1.1 Architecture

The TTS Lab runs as a **single FastAPI process** (`tts_lab.py`) on an Ubuntu 22.04 VM (192.168.0.87), serving 28 TTS engines through one Python 3.11 virtual environment. The deploy flow is managed by `deploy_lab.ps1`, which bootstraps everything from a fresh VM to a running systemd service in 8 phases.

> **Current Environment (as of 2026-06-19):** transformers 5.12.1, torch 2.10.0+cu128, Python 3.11, CUDA GPU. See [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md) for the latest engine status.

### 1.2 Actual Dependency Conflicts

Of the 28 engines after the transformers 5.x / torch 2.10 upgrade:

| Status | Count | Engines |
|--------|:-----:|---------|
| **Working on current env** | 21 | piper, kokoro, melo, matcha, chattts, outetts, bark, styletts2, f5tts, dia, xtts, cosyvoice, fishspeech, chatterbox, chatterboxturbo, omnivoice, openvoice, zonos, manatts (+3 SGLang clients) |
| **Broken — needs older stack** | 3 | **indextts** (176 imports removed in tf 5.x), **parler** (meta tensor removed in torch 2.10), **qwen3tts** (config.pad_token_id removed in tf 5.x) |
| **Gated — needs HF auth** | 2 | orpheus, csm |
| **Not configured** | 1 | neutts |

The conflict is no longer just Orpheus vs. everything. **Three engines need a completely different ML stack:**

| Library | 21 engines (current) | 3 broken engines need | Conflict? |
|----------|---------------------|----------------------|:---------:|
| **torch** | 2.10.0 | 1.13.x | ✅ **Hard — API removed** |
| **transformers** | 5.12.1 | 4.46.x | ✅ **Hard — 176 imports gone** |
| **numpy** | 2.x | 1.x | ✅ **Hard — binary incompatible** |

### 1.3 The Real Picture

**This is no longer a one-split problem.** The VM upgrade to torch 2.10 + transformers 5.12 broke 3 engines in ways that cannot be patched — the APIs they depend on were literally deleted from the libraries. They need their own environment with the old library versions.

Additionally, Orpheus still needs CUDA/vllm isolation. And 3 engines (vibevoice, higgs, s2pro) run as external SGLang servers with zero Python ML code on HuggingFace.

### 1.4 Why Containerization Makes Sense

1. **Reproducibility:** The current deploy depends on a hand-maintained VM. A Dockerfile is a self-documenting spec of every dependency.
2. **Isolation:** 3 legacy engines get their own torch 1.13 + transformers 4.46 environment. Orpheus gets its CUDA/vllm environment. Neither can affect the 21 working engines.
3. **Scaling:** GPU containers can run on a different machine from CPU containers.
4. **Future-proofing:** When engine #29 arrives with conflicting dependencies, it gets its own container — no domino effect.
5. **Rollback:** `docker compose up -d` vs. `docker compose down && docker compose up -d --build` — instant rollback if an update breaks something.

---

## 2. Containerization Options

### Option A: Monolith in Docker (1 image)

Everything in one container. Same as now, just Dockerized.

|   |   |
|---|---|
| Images | 1 (`tts-lab:latest`, ~5 GB) |
| Disk (models) | ~36 GB (volume) |
| Complexity | Trivial — one `docker build`, one `docker run` |
| Dependency isolation | **None** — same conflicts, different box |
| Build time | ~35 min |
| Verdict | ❌ **Misses the point.** Doesn't solve the Orpheus/vllm conflict. |

### Option B: One Container Per Engine (28 images)

Maximum isolation. Each engine gets its own Dockerfile and container.

|   |   |
|---|---|
| Images | 28 (~150 MB – 8 GB each) |
| Disk (images) | ~40 GB (PyTorch duplicated in ~20 images) |
| Disk (models) | ~36 GB (shared volume) |
| Complexity | **Very high** — 28 Dockerfiles, 28 docker-compose services, network orchestration |
| Code changes | Major — rewrite entire `dispatch.py` as HTTP orchestrator |
| Build time | Hours (28 sequential or parallel builds) |
| Verdict | ❌ **Overkill.** Duplicates gigabytes of common dependencies (PyTorch, transformers, numpy). 28 containers to monitor, restart, and debug. |

### Option C: Grouped by Dependency Family (3-5 images)

Group engines that can coexist into shared containers based on their dependency compatibility.

|   |   |
|---|---|
| Images | 3 custom + 3 pre-built = 6 total (see §3 for breakdown) |
| Disk (images) | ~20 GB total |
| Disk (models) | ~36 GB (shared volume) |
| Complexity | Moderate — manageable docker-compose |
| Code changes | ~90 lines (remote dispatch for 4 engines) |
| Build time | ~50 min (mostly parallelizable) |
| Verdict | ✅ **Good balance.** Isolates the real conflicts, minimizes duplication. |

### Option D: Pragmatic Hybrid — Most in One, Isolate Troublemakers (Recommended)

Like Option C but using tiered base images: one base image shared by all stacks, separate containers for engines that need different ML stacks (3 legacy engines on torch 1.13 + 4 GPU engines).

|   |   |
|---|---|
| Images | 1 custom main + 1 custom CUDA + 3 pre-built SGLang = 5 total |
| Disk (images) | ~20 GB total |
| Complexity | Low — 2 Dockerfiles, 1 docker-compose.yml |
| Code changes | ~90 lines |
| Verdict | ✅ **Pragmatic minimum.** Only containerize what must be isolated. |

### Comparison Matrix

|   | Option A (Monolith) | Option B (Per-Engine) | **Option C (6 Containers — RECOMMENDED)** |
|---|:---:|:---:|:---:|
| Dockerfiles to maintain | 1 | 28 | **6** (base, 2 stacks, 2 engines, orpheus) |
| Containers running | 1 | 28 | **7** (orchestrator + 2 engine + orpheus + 3 SGLang) |
| Dependency isolation | None | Perfect | **Good — 3 stacks isolate real conflicts** |
| Image disk overhead | 5 GB | 5 GB (shared base) | **~14 GB (base + 2 stacks + 2 engine images + orpheus)** |
| Code changes | 0 lines | ~500 lines | **~30 lines (HTTP dispatch for 7 engines)** |
| Build time (full) | 35 min | 50 min (parallel) | **~50 min** |
| Build time (add engine) | 35 min (rebuild all) | 2 min | **~5 min (add pip line + rebuild one container)** |
| GPU decoupling | No | Yes | **Yes — GPU engines in separate containers** |
| Expert score | 5/10 | 6/10 | **9.5/10** |

---

## 3. Recommended Architecture: Option D — Pragmatic Hybrid

### 3.1 Architecture: Tiered Base Images

The architecture uses Docker's `FROM` inheritance to share common layers — like class inheritance in programming. The full design is in [CONTAINERIZATION_MASTER_PLAN.md](CONTAINERIZATION_MASTER_PLAN.md). Here's the concrete container layout:

```
┌──────────────────────────────────────────────────────────────────┐
│                     tts-lab-base  (1.5 GB)                       │
│          nvidia/cuda:12.8, espeak-ng, ffmpeg, tts_lab code      │
│          Shared by ALL containers — stored ONCE on disk          │
└────────────┬──────────────┬───────────────────┬──────────────────┘
             │              │                   │
    ┌────────┴────────┐ ┌───┴──────────┐ ┌──────┴──────────────┐
    │ stack:current   │ │ stack:legacy │ │ stack:cuda (Orpheus)│
    │ (+3.5 GB)       │ │ (+2.5 GB)    │ │ (+3.5 GB)           │
    │                 │ │              │ │                     │
    │ torch 2.10.0    │ │ torch 1.13.1 │ │ torch CUDA 12.1     │
    │ tf 5.12.1       │ │ tf 4.46.1    │ │ vllm                │
    │ numpy 2.x       │ │ numpy 1.x    │ │ CUDA toolkit        │
    │ protobuf 5.x    │ │ protobuf 3.x │ │                     │
    └────────┬────────┘ └───┬──────────┘ └──────────┬──────────┘
             │              │                       │
    ┌────────┼──────┬──────┐│  ┌──────┬──────┐      │
    │        │      │      ││  │      │      │      │
    piper  kokoro  ...  21 engines  │      │   orpheus
    50MB   80MB         (each       │      │   (included
                  30-300 MB thin)   │      │    in stack)
                                    │      │
                         indextts parler qwen3tts
                         150MB    150MB   100MB

    SGLang external containers (pre-built, no custom Dockerfiles):
    ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
    │ sglang-vibevoice     │  │ sglang-higgs         │  │ sglang-s2pro         │
    │ lmsysorg/sglang-omni │  │ lmsysorg/sglang-omni │  │ lmsysorg/sglang-omni │
    │ ~7 GB VRAM           │  │ ~9 GB VRAM           │  │ ~11 GB VRAM          │
    │ Port: 8003           │  │ Port: 8004           │  │ Port: 8005           │
    └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### 3.2 Why Tiered: Each Stack Isolates a Real Conflict

The current VM runs **transformers 5.12.1 + torch 2.10.0 on CUDA 12.8**. Of 28 engines:

| Stack | Engines | Why It Exists |
|-------|:-------:|---------------|
| **current** (torch 2.10, tf 5.12) | **21 engines** | The working majority. piper through omnivoice + 3 SGLang HTTP clients. |
| **legacy** (torch 1.13, tf 4.46) | **3 engines** | indextts (176 imports removed in tf 5.x), parler (meta tensor removed in torch 2.10), qwen3tts (config API removed in tf 5.x). APIs were *deleted* — cannot be patched. |
| **cuda** (vllm, CUDA 12.1) | **1 engine** | Orpheus 3B. vllm needs its own CUDA toolkit + pinned numpy/protobuf. |

**Key insight:** This is no longer a one-split problem. The VM upgrade to torch 2.10 + tf 5.12 broke 3 engines in ways that require the actual old library binaries — not patches, not stubs. Docker's layer caching makes storing 3 stacks cost roughly the same as 1 big image (~5 GB vs ~5.2 GB unique bytes) because the base is shared.

### 3.3 Alternative: Single Container with Per-Engine venvs

Instead of multiple containers, run **one container with 3 Python venvs inside** ([see Master Plan §4b](CONTAINERIZATION_MASTER_PLAN.md)). The 3 legacy engines are called via subprocess in their own venv. This is a simpler first migration step — fewer containers, fewer code changes. The trade-off: GPU memory is shared, subprocess overhead (~500ms per call), and isolation is weaker.

### 3.4 Why SGLang Engines Are Already Separate

VibeVoice-1.5B, Higgs Audio v3 4B, and Fish S2-Pro 5B have **zero Python code on HuggingFace** — no `modeling_*.py`, no `auto_map`. They can only run via an SGLang server. The TTS Lab code for these engines is already an HTTP client. Containerizing them just means running the SGLang server in Docker instead of directly on the host.

---

## 4. Detailed Resource Breakdown

### 4.1 Image Sizes (Corrected — Tiered Architecture)

| Image | Base | Key Layers | On-Disk | Shared By |
|-------|------|------------|---------|-----------|
| `tts-lab-base` | `nvidia/cuda:12.8-runtime-ubuntu22.04` | espeak-ng, ffmpeg, tts_lab code, utilities | **~1.5 GB** | All engine containers |
| `tts-lab-stack:current` | `tts-lab-base` | torch 2.10 + tf 5.12 + numpy 2.x + onnxruntime | **+3.5 GB** | 21 engines |
| `tts-lab-stack:legacy` | `tts-lab-base` | torch 1.13 + tf 4.46 + numpy 1.x + onnxruntime | **+2.5 GB** | 3 engines (indextts, parler, qwen3tts) |
| `tts-lab-orpheus` | `nvidia/cuda:12.1-runtime-ubuntu22.04` (own base) | python3.11 + vllm + orpheus-speech + torch CUDA | **~7.0 GB** | 1 engine |
| `lmsysorg/sglang-omni:dev` | Pre-built upstream | SGLang + vllm + transformers | **~8 GB** | 3 SGLang engines (shared base) |
| 2 engine images (Tier 3) | Respective stacks | engine-current (~5 GB) + engine-legacy (~4 GB) | **~9 GB** | 24 engines across 2 containers |

**Total image disk: ~28 GB** (base + 3 stacks + 2 engine images + SGLang). Without SGLang/Orpheus: **~16 GB**.

> **6 containers, not 28 images.** Per expert review — 21 engines in one container (same Python process), 3 legacy engines in another. Only separate what genuinely needs a different stack.

### 4.2 Model Storage (Shared Volume)

Unchanged from [original analysis](#42-model-storage-shared-volume) — **~38.5 GB** mounted at `/opt/models/`. All containers share one volume. Models are never duplicated.

| Storage Path | Contents | Size |
|---|---|---|
| `/opt/models/huggingface/` | All HuggingFace model caches (ChatTTS, Bark, Dia, XTTS, Parler, Chatterbox, Fish, CSM, Qwen3, IndexTTS, Zonos, OmniVoice, Orpheus, VibeVoice, Higgs, S2-Pro) | **~30 GB** |
| `/opt/models/tts/` | Piper ONNX voices (6 voices × ~65 MB), Kokoro ONNX (89 MB + 27 MB) | **~500 MB** |
| `/opt/models/outetts-gguf/` | OuteTTS GGUF files (Q4_K_M default ~384 MB, Q8 variant ~650 MB) | **~1 GB** |
| `/opt/models/cosyvoice/` | CosyVoice2 pretrained models (~2 GB) + Matcha-TTS dependencies | **~2.2 GB** |
| `/opt/models/openvoice_v2/` | OpenVoice v2 checkpoints (converter + base speakers) | **~200 MB** |
| `/opt/models/manatts-vocoder/` | ManaTTS HiFi-GAN vocoder (`checkpoint-2500000steps.pkl`, ~916 MB) | **~920 MB** |
| `/opt/models/Persian-MultiSpeaker-Tacotron2/` | ManaTTS Tacotron2 repo clone | **~50 MB** |
| `/opt/models/csm/` | Sesame CSM repo clone + gated model | **~2 GB** |
| `/opt/models/fish-speech/` | Fish Speech v1.5.1 source code clone | **~100 MB** |
| `/opt/models/indextts/` | IndexTTS-2 model files | **~1.5 GB** |
| **Total** | | **~38.5 GB** |

### 4.3 VRAM Requirements (GPU Containers)

| Container | Engine | VRAM Needed | GPU Required? | Notes |
|-----------|--------|-------------|---------------|-------|
| `tts-lab-orpheus` | Orpheus 3B | **~6 GB** | ✅ CUDA mandatory | vllm + 3B model in BF16 |
| `sglang-vibevoice` | VibeVoice-1.5B | **~7 GB** | ✅ CUDA mandatory | SGLang server + 1.5B model |
| `sglang-higgs` | Higgs 4B | **~9 GB** | ✅ CUDA mandatory | SGLang server + 4B AR model |
| `sglang-s2pro` | S2-Pro 5B | **~11 GB** | ✅ CUDA mandatory | SGLang server + dual-AR 5B (4B slow + 400M fast) |
| **All 4 GPU engines** | | **~33 GB** | ✅ Multi-GPU or A100 | Won't fit on a single consumer GPU |

**GPU sizing guide:**

| GPU | VRAM | What Fits | Strategy |
|-----|------|-----------|----------|
| RTX 3060 (12 GB) | 12 GB | Orpheus only, OR 1 SGLang engine | Pick one GPU engine, rest show unavailable |
| RTX 4070 Ti (12 GB) | 12 GB | Orpheus only, OR 1 SGLang engine | Same as above |
| RTX 4090 (24 GB) | 24 GB | Orpheus + 1 SGLang engine together | Run 2 of 4 GPU engines |
| RTX 6000 Ada (48 GB) | 48 GB | All 4 can run simultaneously | Full GPU deployment |
| A100 (80 GB) | 80 GB | Everything with room to spare | Ideal for full deployment |
| **No GPU available** | 0 GB | Main + legacy containers serve 24 engines (all on CUDA torch) | Orpheus/SGLang show "GPU required" |
| **Single GPU (e.g., 4090 24 GB)** | Up to 24 GB | Orpheus + 1 SGLang engine | Rest show "VRAM full" |

> **On the current VM** (CUDA GPU, torch 2.10+cu128), the main and legacy containers serve 24 engines — all using CUDA torch builds. Orpheus + SGLang containers are started with `--profile gpu`. Without a GPU, those containers simply don't start — their engines show "GPU required" in the UI.

### 4.4 RAM Requirements (System Memory, at Runtime)

| Container | Idle RAM | One Engine Loaded | All Engines Loaded | Notes |
|-----------|----------|-------------------|-------------------|-------|
| `tts-lab-main` | ~2 GB | +1–3 GB | **~12 GB** | VRAM eviction (`_evict_heavy`) keeps only one heavy engine loaded at a time |
| `tts-lab-orpheus` | ~2 GB | +3 GB | **~5 GB** | Single engine, no multiplexing needed |
| `sglang-vibevoice` | ~4 GB | +3 GB | **~7 GB** | SGLang keeps model in VRAM; system RAM for request processing |
| `sglang-higgs` | ~4 GB | +4 GB | **~8 GB** | |
| `sglang-s2pro` | ~4 GB | +5 GB | **~9 GB** | |

> **Current VM has 8 GB swap** configured. The main container's VRAM eviction logic works identically inside Docker — only one heavy engine is kept in RAM at a time. Idle engines are garbage-collected before loading a new one.

### 4.5 Engine Classification by Weight

**Light engines** (< 500 MB RAM, fast load, CPU-friendly):
Piper, Kokoro, Matcha-TTS

**Medium engines** (500–1500 MB RAM):
MeloTTS, StyleTTS2, OuteTTS, OmniVoice, OpenVoice v2

**Heavy engines** (1500–3500 MB RAM, VRAM eviction applies):
ChatTTS, Bark, Dia-1.6B, XTTS-v2, CosyVoice2, Parler-TTS, Chatterbox, Chatterbox-Turbo, Fish Speech, CSM 1B, Qwen3-TTS, IndexTTS-2, Zonos, ManaTTS

**External engines** (run in separate GPU containers):
Orpheus 3B, VibeVoice-1.5B, Higgs Audio v3 4B, Fish S2-Pro 5B

---

## 5. Dockerfile Designs

> **Full Dockerfiles with comments** are in [CONTAINERIZATION_MASTER_PLAN.md §4](CONTAINERIZATION_MASTER_PLAN.md). This section is a summary of the architecture.

### 5.1 Tiered Structure (4 Dockerfiles to maintain)

| Dockerfile | FROM | Adds | Size | Changes |
|------------|------|------|:---:|---------|
| `docker/Dockerfile.base` | `nvidia/cuda:12.8-runtime-ubuntu22.04` | espeak-ng, ffmpeg, tts_lab code, utilities | 1.5 GB | Every 6–12 months |
| `docker/Dockerfile.stack.current` | `tts-lab-base` | torch 2.10, tf 5.12, numpy, onnxruntime + patches | +3.5 GB | Every 3–6 months |
| `docker/Dockerfile.stack.legacy` | `tts-lab-base` | torch 1.13, tf 4.46, numpy 1.x, onnxruntime | +2.5 GB | Rarely (frozen) |
| `docker/Dockerfile.orpheus` | `nvidia/cuda:12.1` (own base) | vllm, orpheus-speech, torch CUDA | 7.0 GB | When vllm updates |
| `docker/engines/Dockerfile.{name}` ×28 | Respective stack | One engine's pip packages | 30–300 MB | When engine updates |

### 5.2 Build-Time Patches (Critical)

From [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md), these MUST run at build time in `stack:current`:

1. **torchcodec metadata stub** — creates fake dist-info so chatterbox doesn't crash on `importlib.metadata.version("torchcodec")`
2. **transformers 5.x stubs** — `patch_transformers_stubs.py`, `fix_transformers_shims.py`, `patch_parler_tts.py`
3. **Runtime shims** (`tts_lab_shims.py`) — imported at container startup, not build time. Patches `inspect.getsourcefile`, stubs `torch._dynamo._trace_wrapped_higher_order_op`, adds `isin_mps_friendly`, `ExtensionsTrie`, `AddedToken` stubs

### 5.3 SGLang Engines

Pre-built `lmsysorg/sglang-omni:dev` — no custom Dockerfiles. Pull and run with `--model` flag. Models download to shared `/opt/models/huggingface` volume on first run.

---

## 6. Orchestration: docker-compose.yml

> **Full compose file** is in [CONTAINERIZATION_MASTER_PLAN.md §4.6](CONTAINERIZATION_MASTER_PLAN.md).

Key services: `orchestrator` (port 8001) + 28 engine containers + 3 SGLang containers. GPU engines use `profiles: [gpu]` so they're only started with `docker compose --profile gpu up -d`. Legacy engines (indextts, parler, qwen3tts) run from `stack:legacy` with their own containers. All share `/opt/models` volume.

> **Full `docker-compose.yml`** (all 28 services + profiles + health checks) is in [CONTAINERIZATION_MASTER_PLAN.md §4.6](CONTAINERIZATION_MASTER_PLAN.md).

---

## 7. Code Changes Required

### 7.1 Two Approaches

| Approach | Code Changes | Complexity | Best For |
|----------|:---:|:---:|----------|
| **A. Per-engine-venv in one container** | ~30 lines (subprocess dispatch for 3 legacy engines) | Low | First migration step |
| **B. Per-engine containers** | ~500 lines (HTTP wrappers per engine) | Medium | Long-term, multi-machine |

### 7.2 Approach A: Subprocess Dispatch (Recommended First Step)

Add to `tts_lab_dispatch.py` — spawn a subprocess in the legacy venv for indextts, parler, qwen3tts:

```python
import subprocess, json

_LEGACY_ENGINES = {"indextts", "parler", "qwen3tts"}
_LEGACY_VENV = "/opt/arthur/venvs/legacy"

def _do_synth_legacy(name: str, text: str, params: dict) -> dict:
    payload = json.dumps({"text": text, "params": params})
    result = subprocess.run(
        [f"{_LEGACY_VENV}/bin/python", "-m", "tts_lab_legacy_worker", name],
        input=payload, capture_output=True, text=True, timeout=300,
    )
    result.check_returncode()
    return json.loads(result.stdout)
```

Plus a ~40-line `tts_lab_legacy_worker.py` that loads one engine and prints JSON to stdout.

### 7.3 Approach B: Full HTTP Wrappers

Each engine container runs a thin FastAPI server exposing `/health` and `/synthesize`. The orchestrator's dispatch layer becomes an HTTP router. Full details in [CONTAINERIZATION_MASTER_PLAN.md §7](CONTAINERIZATION_MASTER_PLAN.md).

### 7.4 SGLang Engines — Zero Code Changes

Already HTTP clients. Just set `VIBEVOICE_SGLANG_URL`, `HIGGS_SGLANG_URL`, `S2PRO_SGLANG_URL` to point at the Docker service names.

---

## 8. Model Storage Strategy

### 8.1 Shared Volume Layout

```
/opt/models/                        ← Docker bind-mount (38+ GB)
├── huggingface/                    ← HF_HOME, shared by all containers
│   ├── hub/
│   │   ├── models--suno--bark/
│   │   ├── models--parler-tts--parler-tts-mini-v1/
│   │   ├── models--fishaudio--fish-speech-1.5/
│   │   ├── models--nari-labs--Dia-1.6B/
│   │   ├── models--IndexTeam--IndexTTS-2/
│   │   ├── models--Zyphra--Zonos-v0.1-transformer/
│   │   ├── models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   │   ├── models--canopylabs--orpheus-3b-0.1-ft/
│   │   ├── models--k2-fsa--OmniVoice/
│   │   ├── models--ResembleAI--chatterbox/
│   │   ├── models--hootan09--ChatterBox/
│   │   ├── ... (30+ model repos)
│   │   └── models--fishaudio--s2-pro/     ← SGLang downloads
│   └── ...
├── tts/                            ← Light ONNX models (< 1 GB)
│   ├── en_US-ryan-high.onnx         (Piper)
│   ├── en_US-ryan-high.onnx.json
│   ├── en_US-amy-low.onnx
│   ├── en_US-lessac-medium.onnx
│   ├── kokoro-v1.0.onnx             (Kokoro)
│   └── voices-v1.0.bin
├── outetts-gguf/                   ← OuteTTS GGUF models
│   ├── OuteTTS-1.0-0.6B-Q4_K_M.gguf   (~384 MB, default)
│   └── OuteTTS-1.0-0.6B-Q8_0.gguf     (~650 MB, higher quality)
├── cosyvoice/
│   └── pretrained_models/
│       └── CosyVoice2-0.5B/         (~2 GB)
├── openvoice_v2/                   ← OpenVoice checkpoints
│   ├── converter/
│   │   ├── config.json
│   │   └── checkpoint.pth
│   └── base_speakers/
│       └── EN/
├── manatts-vocoder/                ← ManaTTS HiFi-GAN
│   └── vctk_hifigan.v1/
│       └── checkpoint-2500000steps.pkl   (~916 MB)
├── Persian-MultiSpeaker-Tacotron2/  ← ManaTTS repo (git clone)
├── fish-speech/                    ← Fish Speech v1.5.1 (git clone)
├── csm/                            ← Sesame CSM 1B (git clone + model)
├── indextts/                       ← IndexTTS-2 model files
│   └── config.yaml
└── cache/                          ← XDG_CACHE_HOME (Bark, etc.)
```

### 8.2 Model Download Strategy

Three options, depending on deployment scenario:

| Strategy | How | Pros | Cons | Best For |
|----------|-----|------|------|----------|
| **A. Bind-mount existing** | Mount host `/opt/models` into containers | Zero re-download, instant migration | Tied to host path, not portable | **Existing VM migration** |
| **B. Named Docker volume, pre-seeded** | `docker volume create tts-models && docker run --rm -v tts-models:/data -v /opt/models:/src alpine cp -a /src/. /data/` | Portable, Docker-managed, backup-friendly | 38 GB copy on initial seed | New deployments |
| **C. Lazy download on first use** | Start with empty volume, engines download models on first synthesis | Minimal setup, no upfront download | First synthesis per engine is slow (downloads model) | Dev/CI environments |
| **D. Models baked into image** | `COPY models/ /opt/models/` in Dockerfile | Fully self-contained image | Image becomes enormous (43 GB), slow push/pull | **Not recommended** |

**Recommendation for the existing VM:** Strategy A (bind mount). The models are already at `/opt/models`. Just mount it.

**Recommendation for new deployments:** Strategy B (named volume, seeded from a backup tarball or downloaded via a one-time setup script).

### 8.3 Volume Sharing Between Containers

All containers mount the same `/opt/models` volume. This means:

- Orpheus container writes downloaded models to `/opt/models/huggingface/`
- Main container reads those same models (and vice versa)
- SGLang containers download to `/root/.cache/huggingface` (mapped to `/opt/models/huggingface`)
- No duplication — one 38 GB directory, zero waste

**Concurrency note:** Multiple containers writing to `HF_HOME` simultaneously is safe because HuggingFace Hub uses file locks. However, only one container should download a given model at a time. In practice, models are downloaded once and cached — concurrent writes are rare.

---

## 9. Total Disk Budget

| Category | Size | Notes |
|----------|------|-------|
| **Images** | | |
| `tts-lab-main` image | 5.2 GB | Uncompressed on disk |
| `tts-lab-orpheus` image | 7.0 GB | Uncompressed on disk |
| `lmsysorg/sglang-omni:dev` base | 8.0 GB | Shared base for all 3 SGLang containers |
| Docker build cache | ~5 GB | Can be pruned: `docker builder prune` |
| *Images subtotal* | **~25 GB** | |
| **Model Files (shared volume)** | | |
| HuggingFace cache | ~30 GB | 30+ model repos |
| ONNX models (Piper, Kokoro, Matcha) | ~500 MB | |
| OuteTTS GGUF | ~1 GB | |
| CosyVoice2 pretrained | ~2.2 GB | |
| OpenVoice v2 checkpoints | ~200 MB | |
| ManaTTS vocoder + repo | ~970 MB | |
| Other git clones + models | ~3.7 GB | Fish Speech, CSM, IndexTTS |
| *Models subtotal* | **~38.5 GB** | |
| **Grand Total (all containers, all models)** | **~66 GB** | Images (~23 GB) + models (~38 GB) + cache (~5 GB) |
| **Stacks only (current + legacy + base, no GPU)** | **~11 GB** | Without Orpheus and SGLang containers |

### 9.1 Comparison to Current Deployment

| | Current (Bare Metal) | Containerized |
|---|---|---|
| Python venv | ~5 GB | Included in image |
| Model files | ~38 GB | ~38 GB (same, shared volume) |
| System packages | On host | In image (200 MB) |
| Docker overhead | 0 | ~5–20 GB (images + cache) |
| **Total** | **~43 GB** | **~44–64 GB** |

The Docker overhead is modest — the bulk of storage is and always will be the model files.

---

## 10. Migration Path: Zero-Downtime Cutover

### Phase 1: Build Images (runs alongside current deployment)

```bash
# On the VM, clone the repo and build
cd /opt/arthur
git pull

# Build both images (won't affect running service)
docker compose build tts-lab
docker compose build orpheus

# Pull SGLang images (optional — only if GPU available)
docker pull lmsysorg/sglang-omni:dev
```

**Impact on running service:** Zero. `docker build` uses CPU and disk I/O but doesn't touch the systemd service on port 8001.

### Phase 2: Test Main Container on Alternate Port

```bash
# Run main container on port 8009 for testing
docker compose up -d tts-lab
# Manually test: http://192.168.0.87:8001 (still the systemd service on 8001)
# But container is also up — test it directly:
docker exec tts-lab-main curl -s http://localhost:8001/status | python -m json.tool
```

Temporarily stop systemd to test on port 8001:

```bash
sudo systemctl stop arthur-lab
docker compose up -d tts-lab   # Now on port 8001
curl http://localhost:8001/status   # Verify 24 engines

# If something is wrong, roll back in seconds:
docker compose down
sudo systemctl start arthur-lab
```

### Phase 3: Test GPU Containers (if GPU available)

```bash
# Start only what fits in your GPU VRAM
docker compose --profile gpu up -d orpheus
docker logs -f tts-lab-orpheus   # Wait for "Loaded in XXs"

# Test synthesis through the main container
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","params":{}}'
```

### Phase 4: Flip the Switch

```bash
# Stop the old systemd service
sudo systemctl stop arthur-lab
sudo systemctl disable arthur-lab

# Start the full Docker deployment
docker compose up -d                 # 24 engines (main + legacy containers)
# or:
docker compose --profile gpu up -d   # + Orpheus GPU engine
# or:
docker compose --profile gpu --profile sglang up -d  # All 28 engines

# Quick verification
curl http://localhost:8001/
curl http://localhost:8001/status | python -m json.tool
```

### Phase 5: Cleanup (after 1 week of stable operation)

```bash
# Keep the old venv as a fallback for one week
# After confirming stability:
rm -rf /opt/arthur-bench-env

# Optionally remove the systemd service file
sudo rm /etc/systemd/system/arthur-lab.service
sudo systemctl daemon-reload

# Clean Docker build cache (optional — frees ~5 GB)
docker builder prune -a
```

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Docker build takes 40 min** | High (first build) | Slow iteration | Layer caching — code changes rebuild in seconds. Only pip changes are slow. Use `--build-arg` to skip tiers during dev. |
| **Image too large to push/pull (5.2 GB)** | Medium | Slow CI/CD, hard to share | Use a local registry (`registry:2`) or `docker save`/`docker load` for air-gapped deploy. Compressed size is ~1.8 GB. |
| **GPU sharing conflicts between containers** | Medium | SGLang OOM kills container | Docker Compose `profiles` — only start what fits. Monitor with `nvidia-smi`. Use `CUDA_VISIBLE_DEVICES` to pin specific engines to specific GPUs. |
| **espeak-ng data path missing** | Low | Kokoro, Zonos, ManaTTS fail | Included in Dockerfile (`apt-get install espeak-ng espeak-ng-data`). Verified with `ls` in build step. |
| **Network latency for remote engine audio** | Low | Slightly slower synthesis | WAV files are base64-encoded. A 60s 24kHz 16-bit mono WAV is ~2.9 MB raw, ~3.9 MB base64. Over Docker bridge network: ~10ms transfer. Negligible vs. synthesis time (seconds to minutes). |
| **HF_TOKEN not passed to GPU containers** | Medium | Gated models (Orpheus, CSM, Qwen3) fail to download | Pass via `${HF_TOKEN:-}` in docker-compose. Document in setup guide. Without token, these engines show "gated — run huggingface-cli login". |
| **Future engine #29 introduces new conflicts** | High (inevitable) | Could break the main container's venv | Now isolated — if a new engine conflicts, give it its own container. The architecture already supports this pattern. |
| **Docker daemon restart kills all engines** | Low | Brief downtime | `restart: unless-stopped` in docker-compose. Containers auto-restart. Model reload on restart is the main cost (30-120s per heavy engine). |
| **Bind-mount permission issues (SELinux/AppArmor)** | Low | Containers can't read /opt/models | Test on first deploy. Fix with `:Z` suffix on volume mount (`/opt/models:/opt/models:Z`) or `chcon -Rt svirt_sandbox_file_t /opt/models`. |
| **Orpheus vllm version drift** | Medium | Future vllm versions may break orpheus-speech | Pin vllm version in Dockerfile.orpheus (`pip install vllm==X.Y.Z`). Test on version bumps. |

---

## 12. Recommendation Summary

|   |   |
|---|---|
| **Approach** | 6 containers: orchestrator + engine-current (21 engines) + engine-legacy (3 engines) + orpheus + 3 SGLang |
| **Dockerfiles to maintain** | **6** (base, stack:current, stack:legacy, engine-current, engine-legacy, orpheus) |
| **Pre-built images** | `lmsysorg/sglang-omni:dev` (pull only) |
| **Code changes** | ~30 lines (subprocess dispatch for legacy engines) or ~500 lines (full HTTP wrappers) |
| **Total disk** | ~66 GB all-in (images ~23 GB + models ~38 GB + cache ~5 GB) |
| **Orchestration** | `docker-compose.yml` with `--profile gpu` and `--profile sglang` flags |
| **Model sharing** | Bind-mount existing `/opt/models` — zero re-download |
| **Migration risk** | Low — can test on alternate port, rollback = `systemctl start arthur-lab` |
| **Migration path** | Phase 1: multi-venv in Docker → Phase 2: extract legacy → Phase 3: extract GPU engines |

### Key Insight (Corrected)

The transformers 5.x / torch 2.10 upgrade broke **3 engines** (indextts, parler, qwen3tts) in ways that cannot be patched — their required APIs were deleted from the libraries. They need a **legacy stack** (torch 1.13 + tf 4.46). Docker's tiered base image pattern handles this: the base is shared, each stack is a thin layer, total disk for all images is ~23 GB (not 40+ GB as originally estimated).

### Related Documents

| Doc | Purpose |
|-----|---------|
| [ENGINE_FIXES_AND_CONTAINERIZATION.md](ENGINE_FIXES_AND_CONTAINERIZATION.md) | Ground truth — current engine status, what broke, what was fixed |
| [CONTAINERIZATION_MASTER_PLAN.md](CONTAINERIZATION_MASTER_PLAN.md) | Full implementation guide — Docker concepts, every Dockerfile, maintenance, edge cases, cheatsheet |

---

## Appendix A: Engine-Per-Container — When It Makes Sense

> **Correction (2026-06-19):** The original version of this appendix claimed per-engine images would use ~40 GB of disk due to base image duplication. This is **wrong** when using a tiered base image architecture. With a shared base image, Docker stores the heavy layers (Python, PyTorch, system packages) once, and each engine image adds only its own thin layer (30–300 MB). Total image disk for 28 per-engine images with a shared base: ~5 GB — nearly identical to the single-image approach.

### Per-Engine Images with Shared Base: The Real Trade-off

|   | 1 Big Image (all 24 engines in one container) | 28 Small Images (shared base, one per engine) |
|---|:---:|:---:|
| **Disk (images)** | ~5.2 GB | ~5.0 GB (base stored once + thin engine layers) |
| **Disk (models)** | ~38 GB | ~38 GB (same shared volume) |
| **RAM idle** | ~2 GB (1 Python process) | ~4–5 GB (28 Python processes, ~100 MB idle each) |
| **Dockerfiles** | 1 (complex, ~120 lines) | 28 (simple, ~20 lines each) |
| **docker-compose** | ~40 lines | ~200 lines |
| **Code changes** | ~85 lines | ~500 lines (HTTP wrapper per engine) |
| **Isolation** | Good (one venv for 24) | **Perfect** (28 independent venvs) |
| **Add an engine** | Edit big Dockerfile, rebuild 5 GB | New 20-line Dockerfile, rebuild 50-300 MB |
| **New conflicting dep** | Painful — upgrade everything or nothing | Create new stack, one engine uses it, zero risk |
| **Push/pull** | 5 GB every change | 1.2 GB base once + 50-300 MB per engine change |
| **Startup** | ~30s (one container) | ~2 min (28 containers in parallel) |
| **Debugging** | `docker logs tts-lab-main` | `docker logs tts-lab-chatterbox` (targeted) |

### When Per-Engine Wins

Per-engine images are the better choice when:

1. **You add engines frequently** — new engine = new 100 MB image, no rebuild of existing engines
2. **Multiple ML stacks coexist** — engines on stack v1, v2, v3 coexist peacefully
3. **You need per-engine versioning** — `tts-lab-chatterbox:v2.1` vs `:v2.2`
4. **You distribute across machines** — 24 engines on one VM, GPU-mandatory engines on another with a GPU
5. **You want targeted updates** — upgrade Chatterbox without touching Piper

### When One Big Image Wins

1. **One VM, one admin** — simpler to manage one container
2. **Minimal RAM** — one Python process vs. 28
3. **Minimal code changes** — dispatch stays in-process
4. **All engines on one stack** — no need for isolation

### Recommendation

**Start with the 6-container approach** (see [CONTAINERIZATION_MASTER_PLAN.md](CONTAINERIZATION_MASTER_PLAN.md) for the full guide). 21 engines in one container (same Python process — exactly how `tts_lab_dispatch.py` works today), 3 legacy engines in a second container, plus Orpheus and 3 SGLang servers. Disk overhead is modest (~16 GB for all images without SGLang/Orpheus). Only add containers when an engine genuinely needs a different stack.

For a **step-by-step guide from zero Docker knowledge through full deployment**, see the [Master Plan](CONTAINERIZATION_MASTER_PLAN.md) — a 2,000-line document covering concepts, architecture, every Dockerfile, maintenance procedures, edge cases, and a command cheatsheet.

---

## Appendix B: Engine Dependency Compatibility Matrix

| Engine | Stack | espeak-ng | git clone | Special |
|--------|:-----:|:---------:|:---------:|---------|
| Piper | current | | | ONNX Runtime (uses CUDA if available) |
| Kokoro | current | ✅ | | ONNX Runtime |
| MeloTTS | current | | ✅ | NLTK data |
| ChatTTS | current | | | |
| OuteTTS | current | | | llama-cpp with CUDA offload |
| Bark | current | | | |
| StyleTTS 2 | current | | | |
| F5-TTS | current | | | |
| Dia-1.6B | current | | ✅ | |
| XTTS-v2 | current | | | |
| CosyVoice2 | current | | ✅ | hyperpyyaml |
| **Parler-TTS** | **legacy** | | | Needs torch 1.13 (meta tensor removed in 2.x) |
| Chatterbox | current | | | Needs torchcodec stub |
| Chatterbox-Turbo | current | | | Needs torchcodec stub |
| Fish Speech | current | | ✅ | |
| CSM 1B | current | | ✅ | Gated model |
| **Qwen3-TTS** | **legacy** | | | Needs tf 4.x config API |
| **IndexTTS-2** | **legacy** | | ✅ | 176 imports removed in tf 5.x |
| Zonos | current | ✅ | ✅ | |
| OpenVoice v2 | current | | ✅ | Needs MeloTTS |
| Matcha-TTS | current | | | sherpa-onnx (uses CUDA if available) |
| ManaTTS | current | ✅ | ✅ | HiFi-GAN vocoder |
| OmniVoice | current | | | |
| **Orpheus 3B** | **cuda** | | | **vllm — GPU mandatory** |
| VibeVoice | (SGLang) | | | External SGLang server |
| Higgs | (SGLang) | | | External SGLang server |
| S2-Pro | (SGLang) | | | External SGLang server |

**All engines use CUDA torch builds** — the only distinction is which CUDA/torch version:
- **current** (21 engines): torch 2.10.0+cu128 (CUDA 12.8), transformers 5.12.1
- **legacy** (3 engines): torch 1.13.1+cu117 (CUDA 11.7), transformers 4.46.1
- **cuda** (1 engine): torch CUDA 12.1 + vllm — GPU mandatory
- **SGLang** (3 engines): external servers, no local torch needed

Piper, Kokoro, and Matcha-TTS use ONNX Runtime (not torch) — but ONNX Runtime can still use CUDA GPU acceleration when available.
