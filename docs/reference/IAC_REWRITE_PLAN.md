# TTS Lab IaC Rewrite — Comprehensive Plan

> **Date:** 2026-06-21
> **Status:** Plan — pending review
> **Related:**
> - [CONTAINERIZATION_CURRENT_STATE.md](CONTAINERIZATION_CURRENT_STATE.md) — What's running now (ad-hoc)
> - [CONTAINERIZATION_PLAN.md](CONTAINERIZATION_PLAN.md) — Original containerization plan
> - [CONTAINERIZATION_ADHOC_REFERENCE.md](CONTAINERIZATION_ADHOC_REFERENCE.md) — Day-by-day log of every fix

---

## Goal

Push to git → GitHub Actions builds images → pushes to GHCR → Ansible pulls to VM → `docker compose up -d`. Everything version-controlled, reproducible, zero manual steps. No more flat images, no more `docker cp` hotfixes.

---

## 1. The Architecture

### 1.1 Docker Image Hierarchy (Tiered — 3 Layers → 4 Images)

Docker images can inherit from each other using `FROM`. Think of it like class inheritance: a base class with shared code, subclasses that add specialized functionality. The base layer is stored once on disk and shared by all child images.

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: tts-lab-base  (~1.5 GB)                                │
│ FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04                     │
│                                                                  │
│ System: espeak-ng, ffmpeg, MeCab, Python 3.11, git, wget, curl  │
│ Python: fastapi, uvicorn, httpx, soundfile, huggingface_hub      │
│ NLTK:   punkt, punkt_tab, cmudict, averaged_perceptron_tagger    │
│ Code:   tts_lab.py, tts_lab_shims.py, tts_lab_config.py, ...     │
│ Symlink: /opt/arthur/models → /opt/models/tts                    │
│                                                                  │
│ Shared by ALL containers. Stored ONCE on disk.                   │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌────────────────────────────┐  ┌──────────────────────────────┐
│ LAYER 2: stack-current     │  │ LAYER 1 (reuse): orchestrator│
│ (~+3.5 GB → 5 GB total)    │  │ (~1.5 GB total)              │
│                             │  │                              │
│ FROM tts-lab-base:latest    │  │ FROM tts-lab-base:latest     │
│ torch 2.12 nightly (cu128)  │  │ ENV ORCHESTRATOR_MODE=1      │
│ transformers 5.12.1         │  │ No ML libraries              │
│ numpy, protobuf, safetensors│  │ Port 8001                    │
│ onnxruntime, accelerate     │  │                              │
│ Patches: transformers stubs │  │                              │
└──────────┬─────────────────┘  └──────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ LAYER 3: engine-current  (~+12 GB → 17 GB total)             │
│                                                               │
│ FROM tts-lab-stack-current:latest                             │
│ All 22 engine pip installs WITH FIXES BAKED IN                │
│ MeCab + unidic download                                       │
│ zonos backbone directory copy                                 │
│ CSM clone + .pth file                                         │
│ huggingface-hub >= 1.0                                        │
│ Nightly torch reinstalled as FINAL step                       │
│ Port 8101                                                     │
└──────────────────────────────────────────────────────────────┘
```

**Total image disk: ~19 GB** (base 1.5 + stack 3.5 + engine 12 + orchestrator 1.5)

Compare to the current ad-hoc monolith: **57 GB** (50 GB engine + 7 GB orchestrator, no layer sharing).

### 1.2 Container Map (7 Containers)

| Container | Image | Port | GPU | Status |
|-----------|-------|:----:|:---:|--------|
| `orchestrator` | `tts-lab-orchestrator` | 8001 | No | ✅ Ready to build |
| `engine-current` | `tts-lab-engine-current` | 8101 | Yes | ✅ Ready to build |
| `engine-legacy` | `tts-lab-engine-legacy` | 8102 | Yes | 🔧 Deferred (user skipped legacy engines) |
| `orpheus` | `tts-lab-orpheus` | 8002 | Yes | 🔧 Blocked (vllm vs torch nightly) |
| `vibevoice` | `lmsysorg/sglang-omni:dev` | 8003 | Yes | ❌ Blocked upstream |
| `higgs` | `lmsysorg/sglang-omni:dev` | 8004 | Yes | ❌ Blocked upstream |
| `s2pro` | `lmsysorg/sglang-omni:dev` | 8005 | Yes | ❌ Blocked upstream |

### 1.3 GPU Strategy

The RTX 5060 Ti has 16 GB VRAM. The engine-current container uses lazy-load: only ONE engine in VRAM at a time (~300 MB to 12 GB depending on model). The engine-mid container also uses lazy-load. Only ONE engine-mid engine can be loaded at a time alongside engine-current.

**VRAM budgeting with engine-mid:**
- engine-current (lazy, 1 engine): ~300 MB – 12 GB
- engine-mid (lazy, 1 engine): ~3 GB (qwen3tts), ~6 GB (VibeVoice), ~9 GB (Higgs)
- Total with both: can fit a light engine-current engine + one engine-mid engine (13-15 GB of 16 GB)

### 1.4 engine-mid — The Middle-Ground Stack (NEW)

Three engines need transformers 4.x but can't use the legacy stack (torch 1.13 is too old for them). They also run as **local models** — no SGLang needed.

| Engine | Why Not engine-current | Why Not engine-legacy | Solution |
|--------|----------------------|----------------------|----------|
| **qwen3tts** | transformers 5.x removed `ROPE_INIT_FUNCTIONS["default"]` | torch 1.13 too old | transformers 4.x + torch 2.x |
| **VibeVoice** | `vibevoice` pip package conflicts with tf 5.12.1 | torch 1.13 too old | tf 4.x + `vibevoice` package |
| **Higgs** | `higgs` architecture not in tf 5.12 | torch 1.13 too old | tf 4.x with `higgs` support |

**What about S2-Pro?** S2-Pro is the only engine that truly needs SGLang — it's deeply integrated with paged KV cache, RadixAttention, and CUDA graph replay. It remains blocked until SGLang updates.

**The engine-mid stack:**
```
Layer 1: tts-lab-base (~1.5 GB) ← SAME base, shared
Layer 2: tts-lab-stack-mid (~+3 GB → ~4.5 GB total)
  FROM tts-lab-base:latest
  torch 2.10.0 stable (cu121)
  transformers 4.51.3

Layer 3: tts-lab-engine-mid (~+8 GB → ~12.5 GB total)
  FROM tts-lab-stack-mid:latest
  qwen-tts, vibevoice, higgs
  Port 8103
```

**New Dockerfiles needed:**
- `docker/Dockerfile.stack.mid` — torch 2.10 stable + transformers 4.x
- `docker/Dockerfile.engine-mid` — FROM stack.mid, 3 engines

**Updated Container Map:**

| Container | Image | Port | GPU | Status |
|-----------|-------|:----:|:---:|--------|
| orchestrator | `tts-lab-orchestrator` | 8001 | No | ✅ |
| engine-current | `tts-lab-engine-current` | 8101 | Yes | ✅ (15 engines) |
| **engine-mid** | **`tts-lab-engine-mid`** | **8103** | **Yes** | **🆕 (3 engines)** |
| engine-legacy | `tts-lab-engine-legacy` | 8102 | Yes | 🔧 Deferred |
| orpheus | `tts-lab-orpheus` | 8002 | Yes | 🔧 Blocked |
| S2-Pro | SGLang | 8005 | Yes | ❌ Blocked upstream |

---

## 2. What Already Works (Reuse, Don't Rewrite)

| File | Status | Notes |
|------|:------:|-------|
| `docker/Dockerfile.base` | 90% | Missing MeCab, punkt_tab, model symlink, hf-hub>=1.0 |
| `docker/Dockerfile.stack.current` | ✅ | Correct: torch nightly + transformers 5.12 |
| `docker/Dockerfile.orchestrator` | ✅ | Correct: FROM base, ORCHESTRATOR_MODE=1 |
| `docker-compose.yml` | ✅ | Correct: 7 services, profiles, volumes, health checks |
| `.github/workflows/build-images.yml` | ✅ | Fully working: path detection, dependency builds, GHCR push |
| All `.py` code fixes | ✅ | 15 commits on main, all fixes permanent |

---

## 3. What We Must Change

### Part 1: Dockerfile.base — 4 Additions

These items are needed by multiple engines. Adding them to the base means they're installed once, not duplicated.

**Addition 1 — MeCab (Japanese text processor for melo, xtts):**
```dockerfile
# Add to the apt-get install line:
mecab libmecab-dev
```

**Addition 2 — huggingface-hub version (for f5tts + transformers 5.12 compatibility):**
```dockerfile
# Change the pip install line to pin version:
"huggingface_hub>=1.0"
# (0.36.x removed is_offline_mode, 1.x restored it)
```

**Addition 3 — NLTK data (for styletts2):**
```dockerfile
# Add to the NLTK download RUN:
nltk.download('punkt_tab', quiet=True)
```

**Addition 4 — Model path symlink (for piper, kokoro):**
```dockerfile
# Add after WORKDIR:
RUN ln -sf /opt/models/tts /opt/arthur/models
```

### Part 2: Dockerfile.engine-current — Full Rewrite

This is the main work. The current flat 25-line Dockerfile must become a ~80-line Dockerfile with every ad-hoc fix baked in as a proper RUN step. Each engine section includes the fix that was discovered during testing.

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 3 — engine-current  (22 engines on torch nightly + tf 5.12)
# ═══════════════════════════════════════════════════════════════════

FROM tts-lab-stack-current:latest
LABEL tts-lab.tier="3-engine"
LABEL tts-lab.stack="current"

# ── Lightweight ONNX engines ─────────────────────────────────────
RUN pip install --no-cache-dir piper-tts>=1.2.0 kokoro-onnx>=0.4.0 sherpa-onnx

# ── MeloTTS + MeCab ──────────────────────────────────────────────
# melo and xtts need MeCab C library + Python bindings + unidic dictionary
RUN apt-get update && apt-get install -y --no-install-recommends \
    mecab libmecab-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git \
    mecab-python3 unidic
RUN python3 -m unidic download

# ── StyleTTS2 — needs OLD langchain (<0.3.0) ─────────────────────
# langchain 1.x removed text_splitter module.
# styletts2 is installed --no-deps to prevent version conflicts
RUN pip install --no-cache-dir --no-deps styletts2
RUN pip install --no-cache-dir "langchain<0.3.0" einops-exts munch f5-tts>=0.3.4

# ── Standard pip engines ─────────────────────────────────────────
RUN pip install --no-cache-dir ChatTTS>=0.2.1 bark>=1.0.0 outetts>=0.3.0
RUN pip install --no-cache-dir chatterbox-tts>=0.1.0 perth>=1.0.0

# XTTS — coqui-tts won't work with torch nightly (torchcodec issue)
# but install the package so import probes pass
RUN pip install --no-cache-dir coqui-tts || true

# ── Fish Speech — needs lightning (NOT pytorch_lightning) ────────
# fish-speech imports from "lightning", not "pytorch_lightning"
# --no-deps because it pins old versions of everything
RUN pip install --no-cache-dir --no-deps fish-speech
RUN pip install --no-cache-dir lightning loralib cachetools kui \
    silero-vad opencc-python-reimplemented pyrootutils

RUN pip install --no-cache-dir qwen-tts omnivoice

# ── Dia 1.6B ────────────────────────────────────────────────────
RUN pip install --no-cache-dir git+https://github.com/nari-labs/dia.git

# ── Zonos — pip wheel is missing backbone/ subpackage ────────────
# The git repo has it, the pip wheel doesn't. Copy it in.
RUN pip install --no-cache-dir git+https://github.com/Zyphra/Zonos.git
RUN SITE_PKGS=$$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    git clone --depth 1 https://github.com/Zyphra/Zonos.git /tmp/zonos-src && \
    cp -r /tmp/zonos-src/zonos/backbone "$$SITE_PKGS/zonos/backbone" && \
    rm -rf /tmp/zonos-src

# ── CSM — needs git clone + heavy deps ───────────────────────────
RUN git clone https://github.com/SesameAILabs/csm.git /opt/models/csm
RUN pip install --no-cache-dir torchtune torchao moshi silentcipher
RUN echo /opt/models/csm > $$(python3 -c "import site; print(site.getsitepackages()[0])")/csm-path.pth

# Orpheus — install for availability probe, runs in separate container
RUN pip install --no-cache-dir orpheus-speech || true

# ── Shared audio processing ─────────────────────────────────────
RUN pip install --no-cache-dir phonemizer>=3.2.1 scipy librosa hyperpyyaml soundfile

# ═══════════════════════════════════════════════════════════════════
# CRITICAL: Reinstall torch nightly as the FINAL step.
# Many engine packages above install torch as a dependency,
# downgrading our nightly build. This restores it.
# ═══════════════════════════════════════════════════════════════════
RUN pip install --no-cache-dir --upgrade torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Pin versions that engine packages may have changed
RUN pip install --no-cache-dir "protobuf>=5.0,<6.0" "huggingface-hub>=1.0"

# Delete broken Dia model cache — only keep Dia-1.6B-0626
RUN rm -rf /opt/models/huggingface/hub/models--nari-labs--Dia-1.6B

# Engine server code (with VRAM leak fix, auto-retry, GPU health)
COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_engine_server.py /opt/arthur/

EXPOSE 8101
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=180s \
    CMD curl -f http://localhost:8101/health || exit 1
CMD ["python3", "/opt/arthur/tts_lab_engine_server.py", "--port", "8101", "--stack", "current"]
```

### Part 3: GitHub Actions — Deploy Workflow

The existing `build-images.yml` needs only minor path updates. The new piece is a **deploy workflow** that pulls images to the VM and restarts containers.

**New file: `.github/workflows/deploy.yml`**

This workflow is triggered manually (`workflow_dispatch`) or when a release tag is pushed. It:
1. Connects to the VM via SSH
2. Pulls the latest images from GHCR
3. Restarts containers with `docker compose up -d`
4. Runs a health check
5. Reports the result

### Part 4: Ansible Playbooks

Ansible is an infrastructure-as-code tool that uses SSH and YAML. Unlike Puppet/Chef, Ansible is **agentless** — nothing needs to be installed on the VM except Python (which Ubuntu already has). You run it from your laptop or a CI runner.

#### How Ansible Works (For Beginners)

1. You write a **playbook** (YAML file) describing the desired state
2. You create an **inventory** (list of VMs to manage)
3. You run `ansible-playbook -i inventory.yml site.yml`
4. Ansible SSH's into each VM and executes tasks to reach the desired state
5. Playbooks are **idempotent** — running them 10 times produces the same result as running once

#### File Structure

```
ansible/
├── site.yml                          # Main playbook — entry point
├── inventory.yml                     # VM IP, user, SSH key
├── group_vars/
│   └── tts-lab.yml                   # Variables (disk path, GPU profile, HF token)
└── roles/
    ├── docker/
    │   └── tasks/main.yml            # Install Docker + NVIDIA Container Toolkit
    ├── disk/
    │   └── tasks/main.yml            # Mount /opt/models data disk
    ├── deploy/
    │   └── tasks/main.yml            # Pull images → docker compose up
    └── monitoring/
        └── tasks/main.yml            # Log rotation, health checks
```

#### Role: docker

```
Purpose: Install Docker Engine and NVIDIA Container Toolkit on a fresh VM.

Steps:
1. Add Docker's official GPG key and apt repository
2. apt-get install docker-ce docker-ce-cli containerd.io
3. Add NVIDIA Container Toolkit GPG key and repository
4. apt-get install nvidia-container-toolkit
5. Configure Docker daemon (/etc/docker/daemon.json):
   - Log rotation: max-size=10m, max-file=3
   - Storage driver: overlay2
6. Add user 'arthur' to docker group (so sudo not needed)
7. systemctl enable --now docker
8. Verify: docker run --rm --gpus all nvidia/cuda:12.8.2-base-ubuntu22.04 nvidia-smi
```

#### Role: disk

```
Purpose: Mount the secondary data disk at /opt/models.

Steps:
1. Check for /dev/sdb (or vdb) — the 200-650 GB data disk
2. If disk has no filesystem: mkfs.ext4 -L models /dev/sdb
3. Create mount point: mkdir -p /opt/models
4. Mount: mount /dev/sdb /opt/models
5. Add to /etc/fstab: UUID=xxx /opt/models ext4 defaults,nofail 0 2
6. Create subdirectories: tts, huggingface, cache
7. Set permissions: chown -R arthur:arthur /opt/models
```

#### Role: deploy

```
Purpose: Pull Docker images from GHCR and start containers.

Steps:
1. Create /opt/arthur directory
2. Copy docker-compose.yml to /opt/arthur/
3. Create .env file with HF_TOKEN from environment variable
4. Login to GHCR: echo $GHCR_TOKEN | docker login ghcr.io -u $GITHUB_ACTOR --password-stdin
5. Pull images: docker compose -f /opt/arthur/docker-compose.yml pull
6. Stop old systemd service if running: systemctl stop arthur-lab && systemctl disable arthur-lab
7. Start containers: docker compose -f /opt/arthur/docker-compose.yml up -d
8. Wait for health checks: sleep 30, check docker compose ps
9. Verify: curl -f http://localhost:8001/status
```

#### Role: monitoring

```
Purpose: Ensure logs don't fill the disk and health is tracked.

Steps:
1. Create /etc/docker/daemon.json with log rotation config
2. Reload Docker daemon if changed
3. Create cron job: */5 * * * * curl -sf http://localhost:8001/status || echo "TTS Lab down" | logger
4. Optional: Set up nvidia-smi monitoring for VRAM tracking
```

### Part 5: Runbook for Beginners

A new file `docs/operations/RUNBOOK.md` — step-by-step instructions for someone who has never used Docker or Ansible before.

Covers:
- Prerequisites (VM specs, Ubuntu version, GPU)
- First-time setup (clone repo, install Ansible, set tokens)
- Running the playbook
- Verifying everything works
- Day-to-day operations (restart, view logs, test engines, update images)
- Troubleshooting common issues (OOM, disk full, engine not available)
- Glossary of terms (Docker, Ansible, GHCR, CI/CD, etc.)

---

## 4. Complete File Checklist

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `docker/Dockerfile.base` | **Edit** | Add MeCab, hf-hub>=1.0, punkt_tab, model symlink |
| 2 | `docker/Dockerfile.stack.current` | No change | Already correct |
| 3 | `docker/Dockerfile.engine-current` | **Rewrite** | All 12 ad-hoc fixes as RUN steps |
| 4 | `docker/Dockerfile.orchestrator` | No change | Already correct |
| 5 | `docker/Dockerfile.stack.mid` | **New** | torch 2.10 stable + transformers 4.x |
| 6 | `docker/Dockerfile.engine-mid` | **New** | qwen3tts, VibeVoice, Higgs |
| 7 | `docker-compose.yml` | **Edit** | Add engine-mid service, update orchestrator port to 8001 |
| 8 | `.github/workflows/build-images.yml` | **Edit** | Add engine-mid build job, update path triggers |
| 9 | `.github/workflows/deploy.yml` | **New** | SSH deploy via workflow_dispatch |
| 10 | `ansible/site.yml` | **New** | Main playbook |
| 11 | `ansible/inventory.yml` | **New** | VM connection details |
| 12 | `ansible/group_vars/tts-lab.yml` | **New** | Variables |
| 13 | `ansible/roles/docker/tasks/main.yml` | **New** | Install Docker + NVIDIA toolkit |
| 14 | `ansible/roles/disk/tasks/main.yml` | **New** | Mount data disk |
| 15 | `ansible/roles/deploy/tasks/main.yml` | **New** | Pull images + start |
| 16 | `ansible/roles/monitoring/tasks/main.yml` | **New** | Health checks + log rotation |
| 17 | `docs/operations/RUNBOOK.md` | **New** | Beginner-friendly operations guide |

---

## 5. Build Order and Timing

Dependency chain (must build in this order):

```
1. Dockerfile.base           (10 min)  — no dependencies
2. Dockerfile.stack.current  (15 min)  — depends on base
3. Dockerfile.engine-current (20 min)  — depends on stack.current
4. Dockerfile.orchestrator   (5 min)   — depends on base
5. docker-compose.yml        (─)       — depends on all images
─────────────────────────────────────────────────────────
Total: ~50 minutes on GitHub Actions (free for public repos)
```

---

## 6. What's NOT in This Plan

| Item | Reason |
|------|--------|
| **SGLang containers** | Blocked upstream — `lmsysorg/sglang-omni:dev` transformers 5.6.0 too old. Keep compose definitions, they'll work when SGLang updates. |
| **Orpheus container** | vllm is incompatible with torch 2.12 nightly. RTX 5060 Ti requires nightly. Deadlock until vllm supports sm_120 or torch stable adds it. |
| **engine-legacy** | Deferred per user request. indytts, parler, qwen3tts skipped. |
| **Kubernetes / Terraform** | Massive overkill for a single VM. |
| **Voice Library** | Separate feature, not container-related. |

---

## 7. Verification

After running the Ansible playbook or GitHub Actions deploy:

```bash
# 1. All containers running and healthy?
docker compose ps
# Expected: orchestrator (healthy), engine-current (healthy)

# 2. Web UI loads?
curl -f http://localhost:8001/
# Expected: HTML page with TTS Lab title

# 3. GPU detected and available?
curl -s http://localhost:8001/status | jq .gpu.name
# Expected: "NVIDIA GeForce RTX 5060 Ti"

# 4. Right number of engines available?
curl -s http://localhost:8001/status | jq '[.models[] | select(.available==true)] | length'
# Expected: 22+

# 5. Fastest engine works? (matcha — real-time, no GPU heavy load)
curl -X POST http://localhost:8001/synthesize/matcha \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world.","params":{}}' \
  -o /tmp/test.wav
file /tmp/test.wav
# Expected: RIFF (little-endian) WAVE audio, PCM, 22050 Hz

# 6. Voice cloning engine works? (f5tts — needs reference WAV)
curl -X POST http://localhost:8001/synthesize/f5tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","params":{"audio_prompt_id":"alex_wright"}}' \
  -o /tmp/test_f5.wav
file /tmp/test_f5.wav
# Expected: RIFF WAVE audio

# 7. VRAM bar shows real data?
curl -s http://localhost:8001/status | jq .gpu
# Expected: {"name":"...", "vram_total":15847, "vram_used":<number>, "vram_free":<number>}
```
