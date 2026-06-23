# Containerization — Current State vs Original Plan

> **Date:** 2026-06-21
> **Status:** Ad-hoc deployment on VM, IaC rewrite pending
> **Related:**
> - [CONTAINERIZATION_PLAN.md](CONTAINERIZATION_PLAN.md) — Original plan (6 containers, tiered Dockerfiles)
> - [CONTAINERIZATION_ADHOC_REFERENCE.md](CONTAINERIZATION_ADHOC_REFERENCE.md) — Day-by-day log of every fix applied
> - [CONTAINERIZATION_MASTER_PLAN.md](CONTAINERIZATION_MASTER_PLAN.md) — 2,000-line educational guide

---

## 1. What's Actually Running Right Now

### 1.1 Container Inventory

| Container | Image | Port | Network | Status | Size |
|-----------|-------|:----:|---------|:------:|------|
| `tts-lab-orchestrator` | `tts-lab-orchestrator:latest` | 8009 | host | ✅ Running | ~6.7 GB |
| `tts-lab-engine-current` | `tts-lab-engine-current:latest` | 8101 | host | ✅ Running | ~49.9 GB |
| `tts-lab-engine-legacy` | — | — | — | ❌ Not deployed | — |
| `tts-lab-orpheus` | — | — | — | ❌ Not deployed | — |
| `tts-lab-vibevoice` | — | — | — | ❌ Not deployed (blocked) | — |
| `tts-lab-higgs` | — | — | — | ❌ Not deployed (blocked) | — |
| `tts-lab-s2pro` | — | — | — | ❌ Not deployed (blocked) | — |

**2 of 7 planned containers are running.**

The old systemd service (`arthur-lab.service`) still runs on port 8001 — a bare-metal deployment with a stale Python 3.11 venv. This is why the Docker orchestrator uses port 8009 instead of the planned 8001.

### 1.2 What Each Running Container Does

**`tts-lab-orchestrator` (port 8009):**
- Serves the Web UI (HTML/CSS/JS)
- Routes all synthesis requests to `tts-lab-engine-current:8101` via HTTP
- Queries engine server `/health` for GPU/VRAM stats
- No ML libraries installed — `ORCHESTRATOR_MODE=1`
- Shows 25/28 engines available in the UI

**`tts-lab-engine-current` (port 8101):**
- FastAPI engine server — lazy-load mode
- 22 engines probed as available at startup
- Only ONE engine loaded in VRAM at a time
- On synthesis: evict current engine → load requested → synthesize → return audio
- Auto-evict on synthesis failure + retry once with fresh load
- GPU: RTX 5060 Ti, 15847 MB VRAM, CUDA 12.8, torch 2.12 nightly

### 1.3 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `/opt/models` | `/opt/models` | HuggingFace cache, ONNX models, GGUF files |
| `/tmp/tts_uploads` | `/tmp/tts_uploads` | Reference WAV files for voice cloning |
| `/opt/arthur/reference_voices` | `/opt/arthur/reference_voices` | Additional reference voices |

---

## 2. Engine Status — Plan vs Reality

### 2.1 Planned (from CONTAINERIZATION_PLAN.md §3)

| Stack | Engine Count | Engines |
|-------|:-----------:|---------|
| **current** | 21 | piper → omnivoice + 3 SGLang HTTP clients |
| **legacy** | 3 | indextts, parler, qwen3tts |
| **cuda (orpheus)** | 1 | orpheus |
| **SGLang external** | 3 | vibevoice, higgs, s2pro |

### 2.2 Actual (2026-06-21)

| Status | Count | Engines |
|--------|:-----:|---------|
| ✅ **Working (synthesis confirmed)** | 15 | bark, chatterbox, chatterboxturbo, chattts, dia, f5tts, fishspeech, kokoro, matcha, melo, omnivoice, outetts, piper, styletts2, zonos |
| 🔧 **Partially working (one blocker remains)** | 2 | csm (Meta license), orpheus (vllm vs torch nightly) |
| ❌ **Blocked upstream** | 3 | vibevoice, higgs, s2pro (SGLang transformers too old) |
| ⛔ **Not built** | 4 | cosyvoice, manatts, neutts, openvoice |
| 🚫 **Skipped by request** | 4 | xtts, qwen3tts, indextts, parler |

**15 of 28 engines work. 13 are missing due to blockers outside the current container's scope.**

### 2.3 Engine-by-Engine RTF (Measured)

| Engine | RTF | Audio | Notes |
|--------|:---:|------:|-------|
| **matcha** | 0.24× ⚡ | 5.2s | Real-time! ONNX flow-matching |
| **styletts2** | 0.22× ⚡ | 5.5s | Real-time! Needs langchain<0.3.0 |
| **piper** | 0.43× ⚡ | 3.7s | Real-time! ONNX CPU |
| **melo** | 0.46× ⚡ | 6.0s | Real-time! Needs MeCab+unidic |
| **omnivoice** | 0.67× ⚡ | 4.2s | Real-time! 600+ languages |
| **chatterboxturbo** | 1.11× | 3.8s | Near real-time! One-step distilled |
| **chatterbox** | 2.42× | 3.5s | Near real-time |
| **chattts** | 2.14× | 2.2s | Near real-time. Ref voice: library bug |
| **kokoro** | 3.20× | 5.4s | ONNX-based |
| **fishspeech** | 3.48× | 11.8s | Voice cloning |
| **zonos** | 4.29× | 5.1s | Voice cloning. Backbone fix needed |
| **dia** | 6-7× | 23.6s | 1.6B model. Monkey-patched KV cache |
| **f5tts** | 5.45× | 1.6s | Voice cloning. Needs hf-hub>=1.0 |
| **bark** | 5.92× | 8.3s | ~12 GB VRAM |
| **outetts** | 15-26× 🐌 | 3.0s | LLM-based. 19 tok/s. Auto-capped |

---

## 3. Ad-Hoc Fixes Applied (Not in the Docker Image)

Every fix below was applied via `docker cp` + `docker restart`. **They are not baked into the Docker image.** The IaC rewrite must incorporate each one.

### 3.1 Code Fixes (Committed to Git)

| # | File | Fix | Commit |
|---|------|-----|--------|
| 1 | `tts_lab_engine_server.py` | VRAM leak: clear `_state[name]["instance"]` before `_safe_del` | `8eab298` |
| 2 | `tts_lab_engine_server.py` | Orchestrator lazy-mode health check fix | `8eab298` |
| 3 | `tts_lab_engine_server.py` | Auto-evict + retry on synthesis failure | `9fb627d` |
| 4 | `tts_lab_engine_server.py` | GPU info in /health response | `b874ce3` |
| 5 | `tts_lab_dispatch.py` | CSM probe: use huggingface_hub for auth | `56649f4` |
| 6 | `tts_lab_dispatch.py` | Lazy-mode: check `reason` absence not `loaded` flag | `8eab298` |
| 7 | `tts_lab.py` | Orchestrator queries engine server for GPU data | `b874ce3` |
| 8 | `tts_lab_engines.py` | ChatTTS ref voice fallback (LZMA encode/decode bug) | `009cd4d` |
| 9 | `tts_lab_engines.py` | Dia: prefer 0626 model, monkey-patch KV cache | `040b8bf` |
| 10 | `tts_lab_engines.py` | OuteTTS: auto-cap max_length | `46a7417` |
| 11 | `tts_lab_engines.py` | Dia: max_tokens floor 3072 | `46cd734` |
| 12 | `tts_lab_shims.py` | BytesIO handling in torchaudio.load fallback | `1467465` |
| 13 | `tts_lab_ui.py` | GPU badge: remote mode placeholder + JS update | `b874ce3` |
| 14 | `tts_lab_ui.py` | Per-engine description textarea (localStorage) | `437631f` |
| 15 | `tts_lab_config.py` | Updated rtf_est with measured values | `437631f` |

### 3.2 In-Container Fixes (Not in Git, Must Be Dockerfile RUN steps)

| # | Fix | IaC Location |
|---|-----|-------------|
| 1 | `apt-get install mecab libmecab-dev` | `Dockerfile.base` |
| 2 | `pip install mecab-python3 unidic && python3 -m unidic download` | `Dockerfile.engine-current` |
| 3 | `pip install "langchain<0.3.0" einops-exts munch` | `Dockerfile.engine-current` |
| 4 | `pip install lightning loralib cachetools kui silero-vad opencc-python-reimplemented pyrootutils` | `Dockerfile.engine-current` |
| 5 | `pip install torchtune torchao moshi silentcipher` (CSM) | `Dockerfile.engine-current` |
| 6 | `pip install "huggingface-hub>=1.0"` | `Dockerfile.engine-current` |
| 7 | `git clone Zonos && cp -r zonos/backbone site-packages/zonos/` | `Dockerfile.engine-current` |
| 8 | `git clone SesameAILabs/csm /opt/models/csm && echo /opt/models/csm > site-packages/csm-path.pth` | `Dockerfile.engine-current` |
| 9 | `nltk.download('punkt_tab')` | `Dockerfile.base` |
| 10 | `ln -sf /opt/models/tts /opt/arthur/models` | `Dockerfile.base` |
| 11 | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128` (MUST be last RUN in engine-current) | `Dockerfile.engine-current` |
| 12 | Delete `Dia-1.6B` cache (broken config), keep only `Dia-1.6B-0626` | Post-build or model download script |

### 3.3 Critical: .pyc Bytecode Cache

When deploying Python files via `docker cp`, Python's `.pyc` bytecode cache is NOT invalidated. The old code keeps running until `__pycache__/*.pyc` is deleted and the container restarted. This wasted hours debugging Dia.

**IaC impact:** Not an issue for Docker image builds (clean filesystem). But MUST be documented for anyone doing ad-hoc hotfixes.

---

## 4. Architecture Comparison: Plan vs Actual

### 4.1 Container Count

| | Planned | Actual | Delta |
|---|:---:|:---:|-------|
| Orchestrator | 1 | 1 | ✅ |
| Engine containers | 2 (current + legacy) | 1 (current only) | ❌ Legacy not built |
| GPU containers | 1 (orpheus) | 0 | ❌ Not deployed |
| SGLang containers | 3 | 0 | ❌ Blocked upstream |
| **Total** | **7** | **2** | **-5** |

### 4.2 Image Architecture

| | Planned | Actual |
|---|---------|--------|
| **Tiers** | base → stack → engine (3 layers) | Single monolithic image (~50 GB) |
| **Base reuse** | base shared by current + legacy + orpheus | No base reuse |
| **Image count** | 5 custom + 1 pre-built | 2 custom (orchestrator, engine-current) |
| **Total image size** | ~28 GB planned | ~57 GB actual (50 engine + 7 orchestrator) |

**Why the actual image is 50 GB vs planned ~16 GB:** The planned architecture used tiered images (base → stack → engine) where the base is shared. The actual build created a single flat `engine-current` image with all 21 engines' pip packages installed inline. Additionally, torch nightly is larger than torch stable.

### 4.3 Network Model

| | Planned | Actual |
|---|---------|--------|
| **Network** | `tts-lab-net` bridge | `host` network |
| **Orchestrator port** | 8001 | 8009 (8001 taken by old systemd) |
| **Engine server port** | 8101 | 8101 ✅ |
| **Legacy port** | 8102 | Not deployed |

**Why host network:** The bridge network failed during ad-hoc setup (orchestrator was already on host network from initial `docker run`). Using host network means containers communicate via `localhost` — simpler for ad-hoc, but less isolated. The IaC rewrite should use the planned bridge network.

### 4.4 Engine Distribution

| Stack | Planned Engines | Actual Engines | Status |
|-------|:---:|:---:|--------|
| **current** | 21 | 22 (all non-legacy, non-SGLang engines ended up here) | Running |
| **legacy** | 3 (indextts, parler, qwen3tts) | 0 | Not built |
| **orpheus** | 1 | 0 | Not deployed |
| **SGLang** | 3 | 0 | Blocked |

---

## 5. What Changed from the Original Plan

### 5.1 Torch Nightly (Unplanned)

**Plan assumed:** torch 2.10 stable on CUDA 12.8

**Reality:** RTX 5060 Ti (Blackwell, sm_120) is NOT supported by torch 2.10 stable. Required switching to **torch 2.12 nightly** from `https://download.pytorch.org/whl/nightly/cu128`.

**Impact:**
- Nightly torch must be reinstalled as the LAST step in any Dockerfile (engine packages downgrade it)
- Several engines became incompatible: xtts (torchcodec), orpheus (vllm)
- The `Dockerfile.stack.current` must use nightly index URL instead of stable

### 5.2 Lazy-Load Engine Server (Unplanned)

**Plan assumed:** All 21 engines loaded at startup in engine-current, or a simpler loading model.

**Reality:** With 16 GB VRAM, loading even 3 heavy engines simultaneously causes OOM. Built a lazy-load engine server with thread-safe VRAM eviction:
- 0 engines loaded at startup
- On synthesis: evict current → load requested → synthesize
- Auto-evict + retry on failure
- `/unload` endpoint for manual VRAM clearing

**Impact:** Added ~200 lines to `tts_lab_engine_server.py`. The orchestrator health check needed updating to handle lazy mode (engines available but not loaded).

### 5.3 Engine RTF Reality vs Estimates

Original estimates were largely theoretical. Measured RTF values are significantly different:

| Engine | Plan Estimate | Measured | Delta |
|--------|:---:|:---:|-------|
| fishspeech | 0.14× | 3.48× | 25× slower than estimated |
| outetts | 1.45× | 15-26× | 10-18× slower |
| omnivoice | 0.025× | 0.67× | 27× slower than estimated |
| f5tts | "needs ref WAV" | 5.45× | Now measured |
| matcha | 0.05× | 0.24× | Still real-time |

The original estimates were optimistic. Several "real-time" estimates turned out to be 3-25× slower in practice.

### 5.4 SGLang Engines — Plan Was Correct, Reality Blocked It

**Plan:** Run `lmsysorg/sglang-omni:dev` containers with `--model` flag.

**Reality:** The SGLang image bundles transformers 5.6.0. VibeVoice, Higgs, and S2-Pro architectures require transformers ≥ 5.12 (or a version that includes their model definitions). As of 2026-06-21:
- No released transformers version has them
- transformers GitHub main (5.13.0.dev0) doesn't have them yet
- The `vibevoice` pip package (0.0.1) conflicts with transformers 5.12.1
- SGLang 0.5.12 is pinned to transformers==5.6.0

**Status:** The plan was sound — the upstream just hasn't caught up yet. When SGLang releases a new image with updated transformers, these 3 engines will work with zero code changes.

### 5.5 Engine-Legacy — Not Yet Needed

**Plan:** A separate container with torch 1.13 + tf 4.46 for indextts, parler, qwen3tts.

**Reality:** The user marked indextts, parler, and qwen3tts as intentionally skipped. Additionally, qwen3tts needs a "middle-ground" stack (torch 2.x + tf 4.x), not the full legacy stack (torch 1.13 + tf 4.46). The legacy container is still planned but not urgent.

---

## 6. VRAM Management — Working

### 6.1 Lazy-Load Flow

```
Startup:
  probe availability → 22 engines available
  0 engines loaded in VRAM
  VRAM: 15.5 GB free

Synthesis request for "bark":
  evict current (none loaded)
  load bark → 12 GB VRAM used
  synthesize → return audio
  bark stays loaded (lazy — don't evict until needed)

Synthesis request for "matcha":
  evict bark → clear _state ref → gc.collect() → torch.cuda.empty_cache()
  VRAM: 15.5 GB free
  load matcha → 0.4 GB VRAM used
  synthesize → return audio

Synthesis error (e.g., corrupted model):
  auto-evict → fresh load from disk → retry
  if retry succeeds: return audio transparently
  if retry fails: return 500
```

### 6.2 VRAM Budget (16 GB Total)

| Scenario | VRAM Used | Free |
|----------|:---------:|:----:|
| Idle (0 engines) | ~300 MB | 15.5 GB |
| Light engine (matcha, piper, kokoro) | ~700 MB | 15.1 GB |
| Medium engine (melo, styletts2) | ~1.5 GB | 14.3 GB |
| Heavy engine (bark) | ~12 GB | 3.7 GB |
| Heavy engine (dia) | ~4 GB | 11.7 GB |
| One SGLang engine (if available) | ~7-11 GB | Would need to stop engine-current |

**One SGLang engine could fit if engine-current is stopped.** Running both simultaneously would OOM on a 16 GB card.

---

## 7. What the IaC Rewrite Must Fix

### 7.1 Dockerfile Changes

The current `Dockerfile.engine-current` is a flat 50 GB image. The IaC rewrite must:

1. **Restore tiered architecture:** base → stack.current → engine-current
2. **Add all 12 in-container fixes** from §3.2 as RUN steps
3. **Reinstall torch nightly as last step** (prevents engine packages downgrading it)
4. **Delete Dia-1.6B cache** (keep only 0626) in a post-install step
5. **Add `huggingface-hub>=1.0`** (needed by f5tts + transformers 5.12)

### 7.2 docker-compose.yml Changes

1. **Use bridge network** (`tts-lab-net`) instead of host network
2. **Add HF_TOKEN** from env file for gated model access (CSM, Orpheus)
3. **Add `restart: unless-stopped`** to all services
4. **Add health checks** with appropriate start_period (engine-current needs 180s)
5. **Keep SGLang profiles** — they'll work when upstream updates

### 7.3 Operational Changes

1. **Stop systemd service** (`arthur-lab.service`) — port 8001 conflict
2. **Move orchestrator to port 8001** after systemd is stopped
3. **Add model download script** for ONNX files (piper, kokoro)
4. **Add HF token injection** from host or secrets file

### 7.4 Engine-Specific IaC Notes

| Engine | IaC Requirement |
|--------|----------------|
| **f5tts** | Needs `huggingface-hub>=1.0` + reference WAV in volume |
| **chattts** | Ref voice broken (ChatTTS library bug). Fallback to random speaker. |
| **dia** | Use `Dia-1.6B-0626` only. Monkey-patch KV cache min size. |
| **outetts** | Auto-cap `max_length` to text-proportional value. BytesIO fix in shims. |
| **styletts2** | Pin `langchain<0.3.0`. Version conflicts are cosmetic (works despite warnings). |
| **zonos** | Copy `backbone/` directory from git repo into site-packages. |
| **fishspeech** | Install `lightning` (not `pytorch_lightning`) + loralib + cachetools + kui etc. |
| **melo** | `apt-get install mecab libmecab-dev` + `pip install mecab-python3 unidic` |
| **csm** | Git clone SesameAILabs/csm + torchtune torchao moshi silentcipher + `.pth` file. Also needs Meta license for llama-3.2-1b. |
| **orpheus** | Separate container per `Dockerfile.orpheus`. vllm incompatible with torch nightly. |
| **vibevoice/higgs/s2pro** | Wait for SGLang image update. Containers already defined in docker-compose. |

---

## 8. Disk Usage — Current VM

| Path | Size | Notes |
|------|------|-------|
| `/opt/models/huggingface/` | 89 GB | All HF model caches |
| `/opt/models/cache/` | 33 GB | Pip, suno, whisper caches |
| `/opt/models/tts/` | 641 MB | Piper/Kokoro ONNX files |
| `/opt/models/CosyVoice/` | 4.6 GB | CosyVoice2 pretrained models |
| `/opt/models/tts_coqui/` | 1.8 GB | Coqui TTS models |
| `/opt/models/outetts-gguf/` | 1.5 GB | OuteTTS GGUF files |
| `/opt/models/fish-speech/` | 184 MB | Fish Speech source clone |
| `/opt/models/csm/` | 340 KB | CSM source clone (model not downloaded) |
| `/opt/models/swapfile` | 33 GB → **deleted 2026-06-21** | Was on wrong disk |
| `/opt/models/indextts` | 11 GB → **deleted 2026-06-21** | Skipped engine |
| **Total /opt/models** | **~133 GB** (was 177 GB before cleanup) | |
| Docker images | ~57 GB | engine-current (50) + orchestrator (7) |
| Docker build cache | ~53 GB | Reclaimable with `docker builder prune` |
| **Total deployment** | **~190 GB** | |

---

## 9. Key Lessons for IaC

1. **Nightly torch is non-negotiable** for RTX 5060 Ti. Must be last RUN step.
2. **Test every engine after build** — the systematic sweep found issues that individual tests missed.
3. **Clear .pyc after any hotfix** — Python bytecode cache masks source file updates.
4. **Auto-evict on failure** — prevents "broken model stuck in VRAM forever" class of bugs.
5. **Check available vs loaded** — lazy-load mode means engines are available but not in VRAM.
6. **SGLang is blocked upstream** — the plan was correct, the timing was wrong. Keep the compose definitions, they'll work when SGLang updates.
7. **Model configs change upstream** — Dia-1.6B changed its config format between releases. Pin model versions.
8. **BytesIO is not a file path** — soundfile can't open `<_io.BytesIO object at 0x...>`.
9. **huggingface-hub version matters** — 0.36.x removed `is_offline_mode`, 1.x restored it. Pin >=1.0 for f5tts compatibility.
10. **VRAM eviction must clear _state references** — not just `del instance` and `torch.cuda.empty_cache()`.

---

## 10. Next Steps

1. **Complete IaC rewrite** — bake all 15 code fixes + 12 in-container fixes into Dockerfiles
2. **Build engine-legacy** — for indextts, parler, qwen3tts (low priority, user skipped)
3. **Build orpheus container** — needs CUDA 12.1 + stable torch + vllm
4. **Wait for SGLang update** — monitor `lmsysorg/sglang-omni` for new tags
5. **Stop systemd service** — move Docker orchestrator to port 8001
6. **Push images to GHCR** — enable CI/CD pipeline in GitHub Actions
7. **Write Ansible playbook** — provision VM from scratch
