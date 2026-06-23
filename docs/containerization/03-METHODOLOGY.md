# TTS Lab — IaC Methodology, Issues & Lessons Learned

> **Date:** 2026-06-21
> **Status:** Living document — update as new issues are discovered
> **Related:**
> - [01-ARCHITECTURE.md](01-ARCHITECTURE.md) — Architecture design
> - [archive/IAC_REWRITE_PLAN.md](archive/IAC_REWRITE_PLAN.md) — Original IaC rewrite plan
> - [04-ADHOC-LOG.md](04-ADHOC-LOG.md) — Day-by-day ad-hoc log

---

## 1. The Two Approaches

### 1.1 Ad-Hoc Patching (What We Started With)

Ad-hoc patching means modifying a running container directly, without updating the Dockerfile or committing changes to version control.

**Workflow:**
```
1. Find a bug (engine crashes with 500)
2. SSH into VM
3. docker exec into container
4. pip install missing-package
5. docker cp fixed-file.py container:/opt/arthur/
6. docker restart container
7. Test — works!
8. Move on to next bug
```

**What's wrong with this:**
- The Docker image doesn't have the fix. If the container is recreated from the image, all fixes are lost.
- No version history. You can't `git blame` to see why a change was made.
- The fixes are invisible to CI/CD. GitHub Actions builds images without them.
- Python `.pyc` bytecode cache masks source file updates. You change the `.py` file but Python keeps running the old compiled version.
- Disk fills up with untracked pip installs and build cache.
- Rebuilding from scratch requires manually re-applying every fix.

**When ad-hoc is acceptable:**
- Emergency hotfix in production while the proper fix is being prepared
- One-time debugging to understand a problem
- Testing a hypothesis before committing to code changes

**When ad-hoc is NOT acceptable:**
- As the primary development methodology
- For anything that needs to survive a container restart
- For fixes that took more than 5 minutes to discover

### 1.2 IaC (Infrastructure as Code — What We Moved To)

Every fix is committed to git, baked into a Dockerfile, and the image is rebuilt from a clean build context. Nothing is done directly to a running container.

**Workflow:**
```
1. Edit code in local repo
2. git commit && git push
3. On VM: cd /opt/tts-lab-docker && sudo git pull
4. sudo docker build -f docker/Dockerfile.<target> -t <image>:latest .
5. sudo docker stop <container> && sudo docker rm <container>
6. sudo docker run ... <image>:latest
7. Test via curl
```

**Why this matters:**
- Every fix is version-controlled. `git log` shows the complete history.
- Images are reproducible. Anyone can clone the repo and build the same image.
- CI/CD (GitHub Actions) automatically builds and pushes images on push.
- Rollback is trivial: `git revert` + rebuild.
- No `.pyc` cache issues because Docker builds start from a clean filesystem.
- No disk bloat from ad-hoc pip installs inside running containers.

---

## 2. The IaC Implementation Pattern

### 2.1 Docker Image Hierarchy (Tiered Architecture)

```
Layer 1: tts-lab-base           (~7 GB)   Shared by ALL containers
Layer 2: tts-lab-stack-current  (~19 GB)  torch nightly + transformers 5.12
Layer 2: tts-lab-stack-mid      (~16 GB)  torch 2.10 stable + transformers 4.x
Layer 3: tts-lab-engine-current (~60 GB)  15 engines (all fixes baked in)
Layer 3: tts-lab-engine-mid     (~17 GB)  VibeVoice + Higgs
Layer 3: tts-lab-engine-qwen    (~8 GB)   Qwen3-TTS (hf-hub < 1.0)
Layer 1: tts-lab-orchestrator   (~7 GB)   Web UI + HTTP dispatch
```

Every layer is a separate `docker build` step. Changes to a layer only require rebuilding that layer and the ones above it. Docker caches unchanged layers.

### 2.2 The Build Context Must Be Git-Cloned

**Critical discovery:** The VM must have a `git clone` of the repo as the build context. If you manually copy files with `scp`, you will eventually miss one and the Docker image will have stale code.

```bash
# One-time setup on VM:
sudo rm -rf /opt/tts-lab-docker
sudo git clone https://github.com/faridnasiri/TTS-LAB.git /opt/tts-lab-docker

# For every rebuild:
cd /opt/tts-lab-docker && sudo git pull
sudo docker build -f docker/Dockerfile.<target> -t <image>:latest .
```

### 2.3 BuildKit Cache — The Silent Saboteur

Docker BuildKit maintains its own cache separate from Docker image layers. Even with `--no-cache`, BuildKit can serve stale file contents from its cache.

**Symptom:** You change a file in git, push, git pull on VM, rebuild with `--no-cache`, but the running container STILL has the old code.

**Root cause:** BuildKit has a separate cache (`docker builder prune` shows it). This cache is keyed by file content hash — but if BuildKit cached the file before your change, it might serve the old version even after `git pull`.

**Fix:**
```bash
# Before rebuilding after code changes:
sudo docker builder prune -af   # Clear BuildKit cache
sudo docker build --no-cache -f docker/Dockerfile.<target> -t <image>:latest .
```

**Lesson:** If a `--no-cache` rebuild doesn't pick up your code changes, the BuildKit cache is the culprit. Clear it.

### 2.4 Which Files Get Copied? — The COPY Layer Problem

**Critical discovery:** Engine Dockerfiles only copied `tts_lab_shims.py` and `tts_lab_engine_server.py`. The `tts_lab_engines.py` and `tts_lab_dispatch.py` files were inherited from the base image (Dockerfile.base has `COPY tts_lab_engines.py .`). When we changed engine code, the engine images didn't pick it up because the base image (built weeks ago) had the old version.

**Fix:** Engine Dockerfiles must COPY all four files:
```dockerfile
COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_engine_server.py /opt/arthur/
COPY tts_lab_engines.py /opt/arthur/
COPY tts_lab_dispatch.py /opt/arthur/
```

**Lesson:** Always check which files are inherited from parent images vs copied in the current Dockerfile.

---

## 3. Issues Encountered & Resolved

### 3.1 Torch Nightly Volatility

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| CUDA library mismatch | `libcudart.so.13: cannot open` | Nightly index moved from CUDA 12.8 to CUDA 13 toolchain | Added `cuda-nvrtc-13-0` package to base image + ldconfig |
| torchvision crash | `operator torchvision::nms does not exist` | Torch nightly lacks torchvision operator | Catch `RuntimeError` in shims import |
| `--no-deps` breaks deps | `libcupti.so.12: undefined symbol` | `--force-reinstall --no-deps` skips CUDA runtime libraries | Use `--force-reinstall` WITHOUT `--no-deps` |

### 3.2 Disk Space

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| Build fails | `no space left on device` | Docker images + BuildKit cache = 200+ GB | `docker builder prune -af` freed 128 GB |
| Repeated issue | Every few builds, disk fills | BuildKit accumulates cache for every layer | Add `docker builder prune -af` to pre-build cleanup |

### 3.3 Dependency Conflicts Between Engines

| Conflict | Engines | Resolution |
|----------|---------|------------|
| `huggingface-hub` version | f5tts needs ≥ 1.0, qwen3tts needs < 1.0 | Separate container (engine-qwen) |
| `transformers` version | 15 engines need 5.x, 3 engines need 4.x | Separate container (engine-mid) |
| `torch` version | 15 engines need nightly (sm_120), orpheus needs stable | Separate container (orpheus — blocked) |

### 3.4 Engine-Specific Issues

| Engine | Issue | Root Cause | IaC Fix |
|--------|-------|-----------|---------|
| **ChatTTS** | `LZMAError: Corrupt input data` | Library bug: `encode_prompt` format ≠ `_decode` format | Fall back to random speaker when ref voice requested |
| **Dia** | KV cache 128 vs 540 | `Dia-1.6B` (no suffix) config format incompatible | Delete broken model, use `Dia-1.6B-0626` only |
| **Dia** | `libnvrtc-builtins.so.13.0` missing | Torch nightly links CUDA 13 | Added `cuda-nvrtc-13-0` + ldconfig |
| **OuteTTS** | `soundfile.LibsndfileError: BytesIO` | Shim's `torchaudio.load` fallback called `str(path)` on BytesIO | Added `isinstance(path, io.BytesIO)` check |
| **OuteTTS** | 55s for "Hello world" | Default `max_length=8192` regardless of text | Auto-cap at `min(max(len*50, 2048), 4096)` |
| **OuteTTS** | `libnvrtc-builtins.so.13.0` missing | Same as Dia | Same fix |
| **f5tts** | `requires reference audio clip` | No default ref WAV | Auto-select first available WAV from `/tmp/tts_uploads/` |
| **qwen3tts** | `KeyError: 'default'` | ROPE_INIT_FUNCTIONS removed in transformers 5.x | Register `"default"` → `"llama3"` mapping |
| **qwen3tts** | `huggingface-hub>=0.34,<1.0 required` | f5tts needs ≥ 1.0 | Separate container (engine-qwen) |
| **qwen3tts** | `unexpected keyword 'inputs_embeds'` | qwen_tts 0.1.1 incompatible with transformers 5.12 | UNRESOLVED — needs transformers 5.0-5.11 |

### 3.5 General IaC Issues

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| Stale build context | Rebuilt images have old code | VM had manually-copied files, not git clone | `git clone` the repo as build context |
| BuildKit cache | `--no-cache` doesn't pick up changes | BuildKit has separate cache | `docker builder prune -af` before rebuild |
| Missing COPY layers | Code changes don't appear in container | Engine Dockerfiles only copied 2 of 4 .py files | COPY all four .py files in engine Dockerfiles |
| `.pyc` bytecode | `docker cp` fixes don't take effect | Python loads stale `.pyc` | Not an issue in IaC (clean build), but MUST document |

---

## 4. When to Add a New Container

The decision tree for adding a new container vs fixing in an existing one:

```
Does the engine have a dependency that CONFLICTS with existing engines?
├── YES → New container
│   ├── Different transformers major version? → New container
│   ├── Different huggingface-hub version range? → New container
│   ├── Different torch version? → New container (or new stack)
│   └── Needs SGLang/vllm? → New container (pre-built image)
└── NO → Fix in existing container
    ├── Missing pip package? → Add to Dockerfile RUN
    ├── Missing system library? → Add to Dockerfile.base apt-get
    ├── Code bug? → Fix in .py file, COPY in Dockerfile
    └── Wrong default params? → Fix in _synth_* function
```

**Rule of thumb:** If fixing the engine would break another engine, it needs its own container. If not, fix it in the existing container.

---

## 5. The IaC Checklist

Before starting any engine fix:

- [ ] Is the fix committed to git? (not just `docker cp`)
- [ ] Is the fix in the correct Dockerfile? (not applied to running container)
- [ ] Are all required `.py` files COPY'd in that Dockerfile?
- [ ] Have I pushed to GitHub? (`git push`)
- [ ] On VM: `git pull` in `/opt/tts-lab-docker/`
- [ ] On VM: `docker builder prune -af` (clear BuildKit cache)
- [ ] On VM: `docker build -f docker/Dockerfile.<target> -t <image>:latest .`
- [ ] Did the build actually rebuild the changed layers? (check output for "CACHED" vs new layer IDs)
- [ ] `docker stop` + `docker rm` old container
- [ ] `docker run` new container
- [ ] Test with `curl` — does it pass?
- [ ] Run full sweep: `python3 sweep.py`

---

## 6. Container Inventory

| # | Container | Port | Stack | Engines | Status |
|---|-----------|:----:|-------|---------|:------:|
| 1 | `tts-lab-orchestrator` | 8009 | base | Web UI | ✅ |
| 2 | `tts-lab-engine-current` | 8101 | current | 15 (bark→zonos + f5tts + qwen3tts) | ✅ |
| 3 | `tts-lab-engine-mid` | 8103 | mid | VibeVoice, Higgs | 🔧 |
| 4 | `tts-lab-engine-qwen` | 8104 | current | Qwen3-TTS | 🔧 |
| 5 | `tts-lab-engine-legacy` | 8102 | legacy | indextts, parler | 🔧 Deferred |
| 6 | `tts-lab-orpheus` | 8002 | cuda | Orpheus 3B | ❌ Blocked |
| 7 | `tts-lab-s2pro` | 8005 | SGLang | S2-Pro | ❌ Blocked |

---

## 7. IaC Mitigation Improvements (2026-06-22)

This section documents the targeted improvements made to eliminate the core build pipeline and dependency problems identified in the first IaC implementation.

### 7.1 Makefile — Automated Build Pipeline

**Problem:** The 11-step manual checklist (pull → clean cache → build → stop → rm → run → wait → test) was error-prone and slow. Every rebuild required remembering all steps.

**Solution:** `Makefile` at repo root. Single-command builds.

```bash
# Build one engine image:
make build-engine ENGINE=current    # tts-lab-engine-current:latest
make build-engine ENGINE=mid        # tts-lab-engine-mid:latest
make build-engine ENGINE=qwen       # tts-lab-engine-qwen:latest

# Build + deploy in one command:
make deploy-engine ENGINE=current

# Full 7-image chain rebuild:
make rebuild

# Run engine sweep test:
make sweep
```

**Key design decisions:**
- `clean-cache` uses `--filter until=24h` instead of `-af` — preserves base layers, only clears stale cache
- `pull` runs before every build — ensures latest committed code
- `TORCH_VER` and `TORCHAUDIO_VER` variables at the top of the Makefile are the **single source of truth** for torch versions. Both `build-engine` and `rebuild` targets pass them as `--build-arg` to Docker. No hardcoded strings duplicated across files.
- Deploy targets include full `docker run` with all required mounts and env vars
- `ARG CACHEBUST=0` is placed **inside the Dockerfile immediately before COPY** — this invalidates ONLY the COPY layer, not heavy upstream layers (apt, pip installs). The Makefile's `--build-arg CACHEBUST=$(date +%s)` is no longer needed at the CLI level.
- Editable install (`pip install -e .`) is safe for this lab environment. The `docker run` mounts (`-v /opt/models:/opt/models`, etc.) do not overlap with `/opt/arthur/` where the source code lives, so mounts cannot accidentally shadow the installed package.

### 7.2 Python Package Structure — Single COPY

**Problem:** Engine Dockerfiles had individual `COPY tts_lab_X.py` statements that went out of sync with the actual files. When `tts_lab_engines.py` was changed but the Dockerfile didn't COPY it, the base image's old version was used. This caused the f5tts fix to silently not deploy for 3 rebuild cycles.

**Solution:** `pyproject.toml` registers all 10 core modules. Dockerfiles use a single glob pattern:

```dockerfile
COPY pyproject.toml /opt/arthur/
COPY tts_lab*.py /opt/arthur/
RUN cd /opt/arthur && pip install --no-deps -e .
```

**Why this works:**
- `pip install -e .` creates an editable install in site-packages pointing back to `/opt/arthur/`
- Python imports resolve correctly regardless of working directory
- Impossible to forget a file — all `tts_lab*.py` files are copied
- Adding a new module just requires adding it to `pyproject.toml`

### 7.3 Pinned Torch Nightly Snapshots

**Problem:** `pip install torch --index-url .../nightly/cu128` installed whatever the latest nightly build was. When the index moved from CUDA 12.8 to CUDA 13 toolchain, it broke Dia, OuteTTS, and required adding CUDA 13 NVRTC libraries. A future index update could break things again.

**Solution:** Pin to a known-working snapshot using Docker `ARG`:

```dockerfile
# In Dockerfile.stack.current:
ARG TORCH_VERSION=2.12.0.dev20260408+cu128
ARG TORCHAUDIO_VERSION=2.11.0.dev20260407+cu128
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

```dockerfile
# In Dockerfile.engine-current (final restore step):
ARG TORCH_VERSION=2.12.0.dev20260408+cu128
ARG TORCHAUDIO_VERSION=2.11.0.dev20260407+cu128
RUN pip install --no-cache-dir --force-reinstall \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

**How to update:** Test a new nightly snapshot on the VM first. If all 15 engines pass, update the `TORCH_VER` and `TORCHAUDIO_VER` variables at the top of the Makefile (the single source of truth). The Makefile passes them to both `Dockerfile.stack.current` and `Dockerfile.engine-current` via `--build-arg`. No need to edit Dockerfiles directly.

```bash
# Test a new snapshot:
make build-engine ENGINE=current TORCH_VER=2.12.0.dev20260409+cu128

# If all engines pass, update the defaults in Makefile:
#   TORCH_VER      ?= 2.12.0.dev20260409+cu128
```

### 7.4 Pinned qwen3tts Transformers Version

**Problem:** qwen_tts 0.1.1 needs transformers >= 5.0 (for ROPE API and auto_docstring) but < 5.12 (check_model_inputs signature changed in 5.12). The engine-qwen container inherited transformers 5.12.1 from stack.current, causing `unexpected keyword argument 'inputs_embeds'`.

**Solution:** Pin transformers in engine-qwen's Dockerfile:

```dockerfile
# Pin to a version range compatible with qwen_tts 0.1.1
RUN pip install --no-cache-dir "transformers>=5.0,<5.12" --force-reinstall
```

This overrides the stack's transformers version for this specific container only.

### 7.5 BuildKit Cache — Targeted Clearing

**Problem:** `docker builder prune -af` cleared ALL BuildKit cache (128+ GB), forcing full rebuilds of every layer including pip installs. This made every rebuild take 20+ minutes.

**Solution:** Use time-based filtering to only clear stale cache:

```makefile
clean-cache:
	docker builder prune -f --filter until=24h
```

This clears cache older than 24 hours — enough to pick up code changes while keeping recent pip installs cached. Combined with `--build-arg CACHEBUST=$(date +%s)` to force COPY layer invalidation.

### 7.6 Results After Mitigation

| Metric | Before | After |
|--------|:------:|:-----:|
| Steps to build + deploy | 11 manual | 1 command (`make deploy-engine`) |
| Build time (cached) | 20+ min | 8-9 min |
| COPY layer misses | 3+ cycles | 0 (glob pattern) |
| Torch version | Floating (breakable) | Pinned (stable) |
| BuildKit cache | Full prune (128 GB) | Time-filtered (stale only) |
| qwen3tts tf version | 5.12 (broken) | 5.0-5.11 (compatible) |

---

## 8. engine-py311 Session — Findings & Solutions (2026-06-22)

### 8.1 The CUDA 12/13 Deadlock

**Problem:** After the PyTorch nightly index moved to CUDA 13 toolchain, `torch` wheels (cu128) compiled for CUDA 12.8 could not coexist with `torchvision` wheels compiled for CUDA 13. The failure chain was:

```
Engine load → transformers → image_utils → torchvision
→ RuntimeError: PyTorch CUDA 12.8 vs torchvision CUDA 13.0
```

This blocked 10+ engines in engine-current (4 worked: matcha, piper, kokoro, outetts — all ONNX/lightweight engines that don't use torchvision).

**Failed approaches (2 days of attempts):**
- `--upgrade torch` without `--no-deps`: pulled CUDA 13 nvidia packages, broke torchvision
- `--upgrade --no-deps`: missing CUDA runtime libraries (libcupti.so.12)
- `--force-reinstall`: same CUDA 13/12 mismatch
- `PIP_CONSTRAINT`: too restrictive, prevented engine package installs
- Multiple rebuilts with slight flag variations: Docker cache masked real issues

**Root cause:** `pip install torch --index-url .../nightly/cu128` installs torch (CUDA 12.8) but its dependencies pull nvidia-* CUDA 13 packages. Engine packages that need torchvision get a CUDA 13 version. Mixing CUDA 12 and 13 in one container is fundamentally broken.

**Solution: engine-py311 container.** Separate container with unified CUDA 13 stack:

```dockerfile
FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04
# Python 3.11 (required by torchvision nightly >= 2026-06)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# All torch/torchvision/torchaudio from SAME index (cu130)
RUN pip install --pre torch torchvision torchaudio --index-url .../nightly/cu130
```

**Key insight:** When two dependencies require different CUDA major versions that cannot coexist, a new container is the correct IaC solution — not trying to force them together.

### 8.2 LD_LIBRARY_PATH Trap

**Problem:** `update-alternatives` to Python 3.11 causes pip to install nvidia libraries to `/usr/local/lib/python3.11/dist-packages/nvidia/*/lib/`. The dynamic linker can't find these paths at runtime. Docker ignores `ENV LD_LIBRARY_PATH` during `RUN` (build) steps.

**Symptom:** `import torch` succeeds during `docker run` but fails during `docker build` on subsequent RUN steps.

**Fix:** Set `LD_LIBRARY_PATH` explicitly inline during RUN commands that need GPU libraries:

```dockerfile
ENV NVIDIA_LIB_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:..."
ENV LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:..."
# For build-time validation:
RUN LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}" pip install --pre torch ...
```

Also add `cuda-nvrtc-13-0` system package for `libnvrtc-builtins.so.13.0` (needed by outetts, dia).

### 8.3 check_model_inputs Global Fix

**Problem:** `transformers` 5.12 `check_model_inputs` decorator validates kwargs against function signatures. Older engine code (chatterbox, chattts) passes `inputs_embeds`, `attention_mask`, `position_ids` as kwargs. The decorator rejects these → all model inferences fail.

**Failed approaches:**
- Per-model `__wrapped__` bypass: saved `_orig_llama_forward` was already wrapped, and the wrapper was applied before our patch
- Per-engine pass-through in `_load_*`: too late — decorators run at import time, not call time
- Stripping kwargs in `_patched_llama_forward`: playing whack-a-mole with kwarg names

**Solution:** Replace the `check_model_inputs` factory function itself in `tts_lab_shims.py`, BEFORE any model is imported:

```python
# In tts_lab_shims.py, before any engine imports:
import transformers.utils.generic as _tug

def _noop_check_model_inputs(*args, **kwargs):
    """Identity pass-through — returns function unchanged."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]  # used as @check_model_inputs without ()
    def _identity(func):
        return func
    return _identity

_tug.check_model_inputs = _noop_check_model_inputs
```

This runs once at engine server startup. All subsequent `@check_model_inputs` decorators are no-ops. Fixed chatterbox, chattts, and partially qwen3tts.

### 8.4 Container Strategy Decisions

After this session, the production architecture is:

| Container | Port | Engines | Rationale |
|-----------|:----:|:-------:|-----------|
| **engine-py311** | 8105 | 14 | Primary: CUDA 13 + Python 3.11 unified stack |
| **engine-omni** | 8106 | 1 | omnivoice needs transformers upgrade (FROM engine-py311) |
| **engine-qwen** | 8104 | 1 | qwen3tts needs tf 4.x + hf-hub <1.0 (FROM engine-py311) |
| **orchestrator** | 8009 | UI | Web interface + HTTP dispatch |

**Decommissioned:**
- engine-current (74 GB, 4 engines) — replaced by engine-py311 (52 GB, 14 engines)
- engine-mid (17 GB, 0 engines) — never worked, engines moved to engine-py311
- Old systemd arthur-lab service (port 8001) — stale venv, all engines broken

**Design principle:** One container = one incompatible dependency set. If an engine needs a different transformers version, CUDA version, or Python version than the primary container, it gets its own lightweight container derived FROM the primary.

### 8.5 qwen3tts Remaining Issues

qwen3tts requires a very specific combination:
- transformers >= 5.0 (for `auto_docstring`)
- transformers < 5.12 (before `check_model_inputs` signature change)
- transformers WITHOUT `TransformGetItemToIndex` (class added in 4.54, incompatible with torch 2.11)
- `ROPE_INIT_FUNCTIONS["default"]` (removed in tf 5.x)
- `huggingface-hub < 1.0` (qwen_tts 0.1.1 requirement)

These five constraints span three different transformers version ranges (4.50-4.53, 5.0-5.11, 5.12+). No single version satisfies all five. The `check_model_inputs` issue is resolved by our global shims fix, but `TransformGetItemToIndex` remains. qwen3tts needs one of:
- An older transformers without TransformGetItemToIndex (< 4.50) — but then `auto_docstring` is missing
- A newer transformers with TransformGetItemToIndex fixed for torch 2.11
- A qwen_tts update from upstream

### 8.6 VRAM Management Across Containers

**Discovery:** Multiple GPU containers share the SAME physical GPU. A heavy engine loaded in one container consumes VRAM that other containers cannot use. With 16 GB total:
- engine-py311 alone: 14 engines work (one at a time via lazy-load)
- engine-py311 + engine-qwen: conflicts (qwen3tts may OOM)
- engine-py311 + engine-omni: works (omnivoice is lightweight)

**Best practice:** Run only the primary engine container (engine-py311). Start auxiliary containers (engine-qwen, engine-omni) on demand and stop them when done.

### 8.7 Docker Build Cache Can Mask Code Changes

**Re-confirmed:** Even `--no-cache` rebuilds can serve stale code if BuildKit cache is not explicitly cleared. Always run before code-change rebuilds:

```bash
docker builder prune -f --filter until=1h
```

---

## 9. Tiered py311 Architecture — Final Design (2026-06-22)

### 9.1 The Mistake Repeated

engine-py311 was initially built as a 52 GB monolith (`FROM nvidia/cuda`). engine-omni and engine-qwen inherited FROM it, each adding another 52 GB. Total unique: ~157 GB.

This was the same mistake made with engine-current. The root cause: focus on "make it work" over "make it right" during debugging.

### 9.2 The Fix: 3-Tier Architecture

```
base-py311 (7.2 GB)              ← Python 3.11, system pkgs, NVRTC, LD_LIBRARY_PATH
  └── stack-py311 (16.4 GB)      ← torch CUDA 13, tf 5.12, ML utils, NLTK
        ├── engine-py311 (52 GB, ~12 GB unique)  ← 14 engine packages
        ├── engine-omni  (17 GB,  ~1 GB unique)  ← omnivoice + upgraded tf
        └── engine-qwen  (18 GB,  ~1 GB unique)  ← qwen-tts + tf 4.x
```

**Total unique on disk: ~37 GB** (vs 157 GB before — 4.2× smaller).

### 9.3 Key Design Rules (Learned Twice Now)

1. **Base image = what ALL containers agree on.** System packages, Python version, NVIDIA libs, LD paths. Never engine-specific code.

2. **Stack image = what a STACK of engines agrees on.** Torch version, transformers version, shared ML utilities. A stack is defined by a CUDA version + Python version + transformers major version.

3. **Engine image = what ONE engine or engine GROUP needs that differs from the stack.** Pip packages, model files, engine server code. Thin layer (~1-12 GB) on top of the stack.

4. **When two engines need different transformers major versions, they go in different containers — but share the same base+stack when possible.**

5. **When an engine needs a completely different CUDA version → different base → different stack → different container.**

### 9.4 Container Decision Matrix

| If engine needs... | Action |
|-------------------|--------|
| Same torch, same tf, different pip packages | Add to engine-py311 |
| Same torch, different tf major version | New engine container FROM stack-py311 |
| Same torch, different hf-hub version | New engine container FROM stack-py311 |
| Different torch/CUDA version | New base + new stack + new engine |
| SGLang / vllm server | Pre-built external container |

### 9.5 Final Container Inventory

| # | Image | Size | Shared | Engines | Status |
|---|-------|------|:------:|:-------:|:------:|
| 1 | `base-py311` | 7.2 GB | All 3 | — | ✅ |
| 2 | `stack-py311` | 16.4 GB | All 3 | — | ✅ |
| 3 | `engine-py311` | 52 GB (12 unique) | — | **14** | ✅ |
| 4 | `engine-omni` | 17 GB (1 unique) | — | omnivoice | ✅ |
| 5 | `engine-qwen` | 18 GB (1 unique) | — | qwen3tts | 🔧 |
| 6 | `orchestrator` | 7.5 GB | — | UI | ✅ |

### 9.6 Engine Status — Final

| Status | Count | Engines |
|--------|:-----:|---------|
| ✅ Working | **15** | matcha, piper, kokoro, melo, styletts2, xtts, chatterboxturbo, chatterbox, chattts, fishspeech, zonos, bark, dia, outetts, omnivoice |
| 🔧 Fixable | 2 | f5tts (torchcodec), csm (model path) |
| 🔧 Partial | 1 | qwen3tts (TransformGetItemToIndex) |
| ❌ SGLang | 3 | higgs, vibevoice, s2pro |
| ❌ vllm | 1 | orpheus |
| ⏭️ Skip | 6 | cosyvoice, indextts, manatts, neutts, openvoice, parler |

### 9.7 Dockerfile Inventory (Complete)

```
docker/
├── Dockerfile.base              # Tier 1 — CUDA 12.8, Python 3.10 (legacy)
├── Dockerfile.base-py311        # Tier 1 — CUDA 12.8, Python 3.11 (current) 🆕
├── Dockerfile.stack.current     # Tier 2 — torch nightly CUDA 12
├── Dockerfile.stack.mid         # Tier 2 — torch 2.10 + tf 4.x
├── Dockerfile.stack-py311       # Tier 2 — torch nightly CUDA 13 🆕
├── Dockerfile.engine-current    # Tier 3 — 22 engines (CUDA 12)
├── Dockerfile.engine-mid        # Tier 3 — VibeVoice, Higgs
├── Dockerfile.engine-py311      # Tier 3 — 14 engines (CUDA 13) 🆕
├── Dockerfile.engine-omni       # Tier 3 — omnivoice 🆕
├── Dockerfile.engine-qwen       # Tier 3 — Qwen3-TTS
├── Dockerfile.orchestrator      # Web UI
└── Dockerfile.orpheus           # Orpheus vllm
```

**Active (used in deployment):** base-py311, stack-py311, engine-py311, engine-omni, engine-qwen, orchestrator.
**Legacy (kept for reference):** base, stack.current, stack.mid, engine-current, engine-mid, orpheus.
