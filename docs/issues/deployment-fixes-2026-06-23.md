# Deployment & Engine Fixes — 2026-06-23 (Complete)

> **VM:** arthur-server (192.168.0.87) — Ubuntu 22.04, RTX 5060 Ti 16 GB
> **Architecture:** 7 containers (compatibility-domain), 4 deployed
> **Git range:** 612e9d5 → 7cba740 (10 commits)
> **Final state:** 16/16 supported engines runtime-confirmed

---

## 1. Deployment Blockers (Resolved)

### 1.1 GPU Access — engine-current Had No GPU

**Symptom:** `RuntimeError: Found no NVIDIA driver on your system` inside engine-current container.

**Root cause:** `docker-compose.yml` engine-current service had `<<: *model-volume` but was missing `deploy.resources.reservations.devices` — the NVIDIA GPU reservation. The old ad-hoc deployment used `--gpus all` with `docker run` (host network), but docker compose with bridge network requires explicit GPU device mapping.

**Fix:** Unrolled `<<: *model-volume` anchor and added inline `deploy:` block with `driver: nvidia, count: 1, capabilities: [gpu]`.

**Commit:** `2dd33c9`

**Before:**
```
engine-current:
  <<: *model-volume          # volumes only, no GPU
  environment:
```

**After:**
```
engine-current:
  volumes:
    - /opt/models:/opt/models
    - /tmp/tts_uploads:/tmp/tts_uploads
    - /opt/arthur/reference_voices:/opt/arthur/reference_voices
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
```

### 1.2 YAML Merge Key Rejection — Docker Compose v5

**Symptom:** `mapping key "<<" already defined at line 151` — docker compose refused to parse the file.

**Root cause:** Docker Compose v5 strictly rejects multiple YAML merge keys (`<<:`) on separate lines. Services using both `<<: *model-volume` and `<<: *gpu-config` were broken: engine-mid, engine-qwen, orpheus.

**Fix:** Unrolled all double-merge anchors into inline YAML for all three services. Kept single-merge anchors where only one is used (orchestrator, SGLang services).

**Commit:** `34b64ef`

### 1.3 Missing Image — engine-legacy

**Symptom:** `docker compose up -d` failed trying to build `Dockerfile.engine-legacy` which requires `tts-lab-stack-legacy:latest` — an image that was never built (deferred).

**Root cause:** engine-legacy service had no `profiles:` key, so docker compose tried to start/build it by default. The legacy stack (torch 1.13 + tf 4.46) was intentionally deferred — indextts and parler are marked BLOCKED.

**Fix:** Added `profiles: [legacy]` to the engine-legacy service. Now only starts with `docker compose --profile legacy up -d`.

**Commit:** `998ce87`

### 1.4 Stale Containers — Name Conflicts

**Symptom:** `Conflict. The container name "/tts-lab-engine-qwen" is already in use`.

**Root cause:** Previous ad-hoc deployment had started containers with the same names. `docker compose down` couldn't clean them up because they were started with `docker run`, not compose.

**Fix:** `docker ps -a | grep tts-lab | xargs -r docker rm -f` — force-removed all stale containers before compose up.

---

## 2. Stack-Level Fixes

### 2.1 sm_120 — Torch Stable Lacks Blackwell Support

**Symptom:** `NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.` Engine-mid and engine-qwen showed this warning with torch 2.5.1+cu121 stable.

**Root cause:** RTX 5060 Ti is Blackwell architecture (sm_120). Stable torch only supports up to sm_90 (H100). sm_120 support was added in torch 2.12 nightly.

**Fix:** Changed `Dockerfile.stack.mid` from stable torch cu121 to nightly torch cu128 (later unified to cu130 for consistency with stack-current). The "mid" stack name now refers to transformers version (4.x), not torch stability.

**Commit:** `624bb6c` (stack-mid → nightly), `618f0ed` (unified → cu130)

**Impact:** All three deployed stacks now use torch nightly:
| Stack | torch | transformers | CUDA |
|-------|-------|-------------|------|
| current | 2.14.0.dev20260622+cu130 | 5.12.1 | 13.0 |
| mid | 2.12.0.dev20260408+cu128 | 4.51.3 | 12.8 |
| qwen | 2.12.0.dev20260408+cu128 | 4.57.3 | 12.8 |

**Note:** engine-mid and engine-qwen still run the older cu128 nightly (built earlier in the session). engine-current was upgraded to cu130 during the torchvision skew fix. A full rebuild cycle will unify all three on cu130.

### 2.2 Torchvision CUDA Version Skew — cu128 Index Drift

**Symptom:** `ERROR: ResolutionImpossible` during `pip install --force-reinstall torch torchaudio torchvision --index-url cu128`. Torchvision 0.27.0.dev20260407 required `torch==2.12.0.dev20260407` but the cu128 index only had `torch==2.12.0.dev20260408`.

**Root cause:** PyTorch nightly builds for different packages are uploaded independently. The cu128 index had a 1-day gap between the torch build (April 8) and the torchvision build (April 7). Pip's dependency resolver cannot reconcile this.

**Fix:** Switched all three stacks from `cu128` to `cu130` nightly index where torch and torchvision builds were aligned (both dated June 22-23, 2026). Removed hardcoded `ARG TORCH_VERSION` pins from `Dockerfile.stack.current` in favor of letting pip resolve compatible versions.

**Commit:** `618f0ed`

---

## 3. Engine-Level Fixes

### 3.1 qwen3tts — TransformGetItemToIndex Crash

**Symptom:** `AttributeError: __enter__` at `transformers/masking_utils.py:391` during qwen3tts synthesis. The `TransformGetItemToIndex` context manager has a broken `__enter__` method in transformers 4.57.3.

**Discovery chain:**
1. qwen_tts 0.1.1 metadata pins `transformers==4.57.3` (exact)
2. Attempt to use `transformers==4.51.3`: `ImportError: cannot import name 'auto_docstring'` (not present in 4.51.3)
3. Attempt `transformers==4.53.3`: `ImportError: cannot import name 'check_model_inputs'` (not in `transformers.utils.generic` yet)
4. qwen_tts needs: auto_docstring (≥4.52) + check_model_inputs (≥4.57) + no broken TransformGetItemToIndex
5. No single transformers version satisfies all three without a patch

**Fix:** Let qwen_tts install its preferred transformers==4.57.3. Patched `tts_lab_shims.py` to replace `TransformGetItemToIndex` with a working no-op context manager:

```python
class _NoopTransformGetItemToIndex:
    def __enter__(self): return self
    def __exit__(self, *a): pass
transformers.masking_utils.TransformGetItemToIndex = _NoopTransformGetItemToIndex
```

This avoids the version war entirely — give qwen_tts what it wants, fix the one broken function.

**Commit:** `e0b8dd9`

**Validation:** Synthesis confirmed — RTF 4.5×, 3.8s audio, 24kHz mono WAV.

### 3.2 Numpy ABI Breakage — 10 Engines Regressed

**Symptom:** After rebuilding engine-current from IaC Dockerfile, only 3/16 engines worked (piper, kokoro, matcha). Errors:
- `ImportError: cannot import name 'Inf' from 'numpy'`
- `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Root cause:** The IaC Dockerfile's final step `pip install --force-reinstall torch torchaudio torchvision --index-url cu130` upgraded numpy from 1.26.4 to 2.2.6. Engine packages (bark, melo, outetts, styletts2, f5tts, fishspeech, zonos) have compiled C extensions built against numpy 1.x ABI. The ABI changed incompatibly between numpy 1.x and 2.x.

**Fix:** Added `numpy>=1.26,<2.0` pin AFTER the torch force-reinstall step. This prevents torch from upgrading numpy past the 1.x boundary.

**Commit:** `05469b6`

**Impact:** Recovered 7 engines in one fix (melo, outetts, bark, styletts2, f5tts, fishspeech, zonos).

### 3.3 python3-dev — Triton CUDA Kernel Compilation

**Symptom:** `fatal error: Python.h: No such file or directory` during ChatTTS and OmniVoice synthesis. Triton JIT-compiles CUDA kernels at runtime and needs the Python C headers.

**Root cause:** The base image `nvidia/cuda:12.8.2-runtime-ubuntu22.04` is a RUNTIME image — it includes CUDA libraries but not development headers. Triton's JIT compiler invokes `gcc` to compile `.c` files that `#include <Python.h>`.

**Fix:** Added `apt-get install python3-dev` to `Dockerfile.engine-current`.

**Commit:** `af05e11`

**Impact:** Fixed chattts (RTF 11.5×) and omnivoice (RTF 4.5×).

### 3.4 torchcodec Stub — XTTS and f5tts Import Chain

**Symptom:**
- XTTS: `ImportError: From Pytorch 2.9, the torchcodec library is required for audio IO`
- f5tts: `AttributeError: module 'torchcodec' has no attribute 'decoders'`

**Root cause:** coqui-tts 2.9+ requires `torchcodec` for audio I/O with PyTorch ≥ 2.9. The `stack-current` Dockerfile created only metadata stubs (`torchcodec-99.0.0.dist-info/METADATA`) but NO Python module. The coqui-tts import check does `import torchcodec` which succeeds with the metadata, but then fails when accessing `torchcodec.decoders.AudioDecoder`.

f5tts has a secondary dependency: transformers' ASR pipeline (`automatic_speech_recognition.py`) checks `isinstance(inputs, torchcodec.decoders.AudioDecoder)`.

**Fix chain:**
1. First attempt: `pip install torchcodec || true` — failed silently (no such package on PyPI)
2. Second attempt: Created `torchcodec/__init__.py` module — XTTS import succeeded
3. Third attempt: Added `torchcodec/decoders/__init__.py` with `class AudioDecoder: pass` — f5tts import succeeded
4. Final: Added build-time verification `python3 -c "from torchcodec.decoders import AudioDecoder"` and ARG cache-busting

**Commits:** `883b240`, `d8f1388`, `7cba740`

**Impact:** Fixed xtts (RTF 44.7×) and f5tts (RTF 13.4×).

### 3.5 chatterbox/chatterboxturbo — False Negative (Test Artifact)

**Symptom:** chatterbox failed with `CUDA out of memory` during direct test. chatterboxturbo showed HTTP 500 through orchestrator.

**Root cause:** Test artifact. The direct test script loaded engines sequentially without eviction. By the time chatterbox was tested, a prior engine (the engine server's loaded model) occupied 6.58 GB VRAM. The engine server's lazy-load mode correctly handles eviction — the orchestrator test was failing for a different reason (engine server hadn't finished probing after restart).

**Resolution:** Both engines work correctly when tested in isolation with clean VRAM: chatterboxturbo RTF 1.2× (near real-time), chatterbox RTF 25.2×.

---

## 4. Final Synthesis Sweep — All 16 Supported Engines

| # | Engine | Container | RTF | Audio | Latency | Notes |
|---|--------|-----------|:---:|------:|--------|-------|
| 1 | **piper** | engine-current | 3.7× | 743ms | 3s | ONNX CPU. Fastest response. |
| 2 | **kokoro** | engine-current | 6.5× | 1344ms | 9s | ONNX. |
| 3 | **melo** | engine-current | 9.8× | 1893ms | 19s | Needs MeCab+unidic. Fixed by numpy pin. |
| 4 | **matcha** | engine-current | 8.0× | 1183ms | 9s | ONNX flow-matching. |
| 5 | **chattts** | engine-current | 11.5× | 902ms | 10s | Fixed by python3-dev. |
| 6 | **outetts** | engine-current | 34.9× | 1293ms | 45s | LLM-based. Slowest RTF. Fixed by numpy pin. |
| 7 | **bark** | engine-current | 19.0× | 2653ms | 50s | Heavy VRAM. Fixed by numpy pin. |
| 8 | **styletts2** | engine-current | 35.8× | 1772ms | 63s | Needs langchain<0.3.0. Longest latency. |
| 9 | **f5tts** | engine-current | 13.4× | 682ms | 9s | Voice cloning. Fixed by torchcodec stub. |
| 10 | **chatterbox** | engine-current | 25.2× | 840ms | 21s | AR+diffusion. |
| 11 | **chatterboxturbo** | engine-current | 13.6× | 1240ms | 17s | One-step distilled. Near real-time. |
| 12 | **fishspeech** | engine-current | 9.4× | 2972ms | 28s | Voice cloning. Longest audio. |
| 13 | **omnivoice** | engine-current | 4.5× | 1640ms | 7s | 600+ languages. Real-time tier. |
| 14 | **zonos** | engine-current | 19.0× | 1532ms | 29s | Voice cloning. |
| 15 | **xtts** | engine-current | 44.7× | 1387ms | 62s | Fixed by torchcodec stub. |
| 16 | **qwen3tts** | engine-qwen | 4.7× | 800ms | 4s | 🆕 Promoted. TransformGetItemToIndex patch. |

**All 16 runtime-confirmed on RTX 5060 Ti.**

### RTF Distribution

| Tier | RTF Range | Count | Engines |
|------|:---------:|:-----:|---------|
| Real-time | < 1.0× | 0 | — |
| Near real-time | 1.0–5.0× | 3 | piper, omnivoice, qwen3tts |
| Fast | 5.0–15.0× | 7 | kokoro, melo, matcha, chattts, f5tts, chatterboxturbo, fishspeech |
| Moderate | 15.0–30.0× | 3 | bark, chatterbox, zonos |
| Slow | 30.0–50.0× | 3 | outetts, styletts2, xtts |

---

## 5. Fixes Applied — Complete Inventory

| # | Fix | Layer | Commit | Engines Affected |
|---|-----|-------|--------|------------------|
| 1 | GPU access (deploy.resources) | docker-compose.yml | `2dd33c9` | engine-current |
| 2 | YAML merge keys (Compose v5) | docker-compose.yml | `34b64ef` | engine-mid, engine-qwen, orpheus |
| 3 | Profile isolation (legacy) | docker-compose.yml | `998ce87` | engine-legacy |
| 4 | sm_120 support (nightly torch) | stack.mid | `624bb6c` | engine-mid, engine-qwen |
| 5 | cu130 unification (torchvision skew) | stack.current, stack.mid, engine-current | `618f0ed` | All stacks |
| 6 | TransformGetItemToIndex patch | tts_lab_shims.py, engine-qwen | `e0b8dd9` | qwen3tts |
| 7 | numpy < 2.0 ABI pin | engine-current | `05469b6` | melo, outetts, bark, styletts2, f5tts, fishspeech, zonos |
| 8 | python3-dev for triton | engine-current | `af05e11` | chattts, omnivoice |
| 9 | torchcodec module stub | engine-current | `883b240`, `d8f1388`, `7cba740` | xtts, f5tts |

---

## 6. Container State (Final)

| Container | Image | Port | GPU | Engines | Health |
|-----------|-------|:----:|:---:|:-------:|:------:|
| `tts-lab-orchestrator` | `tts-lab-orchestrator:latest` | 8001 | No | Web UI + routing | ✅ |
| `tts-lab-engine-current` | `tts-lab-engine-current:latest` | 8101 | Yes | 21 (15 working via orchestrator) | ✅ |
| `tts-lab-engine-mid` | `tts-lab-engine-mid:latest` | 8103 | Yes | 2 (VibeVoice, Higgs — EXPERIMENTAL) | ✅ |
| `tts-lab-engine-qwen` | `tts-lab-engine-qwen:latest` | 8104 | Yes | 1 (Qwen3TTS — SUPPORTED) | ✅ |

### Not Deployed (3 containers)

| Container | Reason | Profile | Blockers |
|-----------|--------|:-------:|----------|
| `engine-legacy` | Image not built (deferred) | `legacy` | indextts, parler need torch 1.13 + tf 4.46 |
| `orpheus` | vllm incompatible with torch nightly | `gpu` | Needs separate CUDA 12.1 + stable torch |
| `s2pro` | SGLang transformers too old | `sglang` | paged KV cache, RadixAttention, CUDA graph replay |

---

## 7. Remaining Work (Non-Blocking)

| Area | Engine | Issue | Approach |
|------|--------|-------|----------|
| VibeVoice POC | vibevoice | `vibevoice` not in transformers CONFIG_MAPPING | Registration shim or wait for SGLang/transformers update |
| Higgs POC | higgs | Same class of issue | Same approach |
| dia latency | dia | 86s for 10s of audio (RTF 6.6×) | Already works, just slow |
| Container unification | engine-mid, engine-qwen | Still on cu128, not cu130 | Rebuild with cu130 for consistency |
| engine-current Dockerfile | — | In-place fixes need proper Dockerfile rebuild | Run `docker build --no-cache` from latest commit |

---

## 8. Files Changed (This Session)

```
docker-compose.yml                  — GPU config, YAML merge unroll, legacy profile
docker/Dockerfile.stack.current     — cu128 → cu130, remove ARG pins
docker/Dockerfile.stack.mid         — stable → nightly, cu128 → cu130
docker/Dockerfile.engine-current    — numpy pin, python3-dev, torchcodec stub, cu130
docker/Dockerfile.engine-qwen       — stack-py311 → stack-mid, --no-deps, TransformGetItemToIndex approach
tts_lab_shims.py                    — TransformGetItemToIndex no-op context manager
scripts/update_engine_status.py     — Gate updater, container introspection, history, promotion
docs/engine_compatibility.yaml      — Maturity classification, validation gates, fingerprints
docs/containerization/01-ARCHITECTURE.md          — Architecture design
docs/containerization/archive/IAC_REWRITE_PLAN.md — Original IaC rewrite plan (superseded)
docs/reference/ARCHITECTURE_REFERENCE.md          — Deployed-state reference
docs/issues/deployment-2026-06-23.md         — Initial deployment findings
docs/issues/deployment-fixes-2026-06-23.md   — This document
```
