# IaC Fix Plan — Remaining Work

> **Date:** 2026-06-21
> **Status:** Plan — pending review
> **Related:** [IAC_REWRITE_PLAN.md](IAC_REWRITE_PLAN.md) — Full architecture plan

---

## Context

We built the tiered Docker architecture. Results from the last test run:

| Status | Count | Engines |
|--------|:-----:|---------|
| ✅ Pass | **12** | bark, chatterbox, chatterboxturbo, chattts, fishspeech, kokoro, matcha, melo, omnivoice, piper, styletts2, zonos |
| ❌ CUDA 13 | 2 | dia, outetts — `libnvrtc-builtins.so.13.0` missing |
| ❌ SGLang | 3 | higgs, vibevoice, s2pro — blocked upstream |
| ❌ Other | 3 | f5tts (needs ref WAV), orpheus (vllm vs nightly), xtts (torchcodec, skipped) |
| ⏭️ Skip | 8 | cosyvoice, csm, indextts, manatts, neutts, openvoice, parler, qwen3tts |

The build process had friction because:
1. The VM build context (`/opt/tts-lab-docker/`) had stale `.py` files — code fixes weren't included in images
2. Torch nightly index moved to CUDA 13-linked builds, breaking dia and outetts (fix committed, needs rebuild)
3. We don't have a proper git clone on the VM

---

## Step 1: Fix the Build Context (One-Time)

**Problem:** The VM at `/opt/tts-lab-docker/` has old code. Every rebuild misses the latest fixes.

**Fix:** Clone the repo on the VM. Use it as the build context.

```bash
# On VM, run once:
cd /opt && rm -rf tts-lab-docker
git clone https://github.com/faridnasiri/TTS-LAB.git /opt/tts-lab-docker
```

After this, `docker build` uses the latest committed code. No more SCP + manual copy. When code changes are committed and pushed, `git pull` on the VM updates the build context.

---

## Step 2: Full Rebuild Chain

All Dockerfiles are already correct in the repo. Rebuild in dependency order:

```
1. docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
   → CUDA 12.8 + CUDA 13 NVRTC + MeCab + hf-hub>=1.0 + punkt_tab + model symlink + python-multipart
   ~10 minutes

2. docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current:latest .
   → torch 2.12 nightly + transformers 5.12 + accelerate + onnxruntime
   ~15 minutes

3. docker build -f docker/Dockerfile.stack.mid -t tts-lab-stack-mid:latest .
   → torch 2.10 stable + transformers 4.x
   ~12 minutes

4. docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
   → 22 engines with ALL fixes baked in (MeCab, langchain, lightning, zonos backbone, CSM, nightly restore)
   ~20 minutes

5. docker build -f docker/Dockerfile.engine-mid -t tts-lab-engine-mid:latest .
   → 3 engines (qwen3tts, VibeVoice, Higgs) on middle-ground stack
   ~10 minutes

6. docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .
   → Web UI + HTTP dispatch, ORCHESTRATOR_MODE=1
   ~5 minutes
```

**Total: ~72 minutes** (base shared, stacks parallel, engines parallel)

### What Changed from Last Build

| File | Change |
|------|--------|
| `Dockerfile.base` | Added: `cuda-nvrtc-13-0` (fixes dia + outetts), MeCab, hf-hub>=1.0, punkt_tab, python-multipart |
| `Dockerfile.engine-current` | Rewritten: all 12 ad-hoc fixes as RUN steps, `--force-reinstall` torch nightly LAST step with deps |
| `Dockerfile.stack.mid` | New: torch 2.10 stable + transformers 4.x |
| `Dockerfile.engine-mid` | New: qwen3tts, VibeVoice, Higgs |
| `tts_lab_shims.py` | Fixed: catch RuntimeError in modeling_layers import (torchvision nms operator) |
| `tts_lab_engine_server.py` | Fixed: VRAM leak, auto-retry, GPU info in /health |
| `tts_lab_dispatch.py` | Fixed: lazy-mode health check, CSM auth, vibevoice/higgs URLs |

---

## Step 3: Deploy and Verify

```bash
# Stop old containers
docker stop tts-lab-engine-current tts-lab-orchestrator
docker rm tts-lab-engine-current tts-lab-orchestrator

# Start new ones
cd /opt/tts-lab-docker
docker compose -f docker-compose.yml --profile mid up -d

# Wait for startup (engine server probes 20 engines, takes ~60s)
sleep 70

# Verify orchestrator
curl -s http://localhost:8009/status | python3 -c \
  'import sys,json; d=json.load(sys.stdin); print("Orchestrator:", sum(1 for m in d["models"].values() if m["available"]), "/", len(d["models"]))'
# Expected: 22 / 28 (or more after sweep)

# Run full engine sweep
python3 sweep.py
```

---

## Expected Final Results

| Status | Count | Engines |
|--------|:-----:|---------|
| ✅ Pass | **14** | bark, chatterbox, chatterboxturbo, chattts, **dia**, fishspeech, kokoro, matcha, melo, omnivoice, **outetts**, piper, styletts2, zonos |
| ❌ Ref WAV | 1 | f5tts — works with `audio_prompt_id` param |
| ❌ SGLang | 3 | higgs, vibevoice, s2pro — blocked upstream |
| ❌ Nightly | 1 | orpheus — vllm incompatible with torch nightly |
| ⏭️ Skip | 9 | cosyvoice, csm, indextts, manatts, neutts, openvoice, parler, qwen3tts, xtts |

---

## What's Already Done (No Changes Needed)

| File | Status |
|------|:------:|
| `docker/Dockerfile.base` | ✅ All fixes committed |
| `docker/Dockerfile.stack.current` | ✅ Correct |
| `docker/Dockerfile.stack.mid` | ✅ New — ready |
| `docker/Dockerfile.engine-current` | ✅ All fixes baked in |
| `docker/Dockerfile.engine-mid` | ✅ New — ready |
| `docker/Dockerfile.orchestrator` | ✅ Correct |
| `docker-compose.yml` | ✅ engine-mid service, URLs updated |
| `.github/workflows/build-images.yml` | ✅ Path-based CI/CD |
| `.github/workflows/deploy.yml` | ✅ SSH deploy |
| `ansible/` | ✅ All 4 roles |
| All `.py` files | ✅ All fixes committed |

---

## Files Changed in This Plan

| # | File | Action |
|---|------|--------|
| 1 | VM: `/opt/tts-lab-docker/` | `git clone` the repo (one-time setup) |
| 2 | Nothing else | All code already committed |

---

## Verification Checklist

```bash
# 1. Engine server healthy?
curl -s http://localhost:8101/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["engines_available"], "available")'
# Expected: 20+

# 2. GPU info flowing?
curl -s http://localhost:8009/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print("GPU:", d["gpu"].get("name","?"), d["gpu"].get("vram_total","?"), "MB")'
# Expected: NVIDIA GeForce RTX 5060 Ti 15847 MB

# 3. All 14 working engines pass?
python3 sweep.py
# Expected: 14 pass, 6 fail, 8 skip

# 4. Docker images — tiered, not monolithic?
docker images | grep tts-lab
# Expected: base (~7 GB), stack-current (~19 GB), stack-mid (~16 GB), engine-current (~60 GB), engine-mid (~17 GB), orchestrator (~7 GB)
```
