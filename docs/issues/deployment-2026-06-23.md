# Deployment Findings — 2026-06-23

> **VM:** 192.168.0.87 — RTX 5060 Ti, 16 GB VRAM, Ubuntu 22.04
> **Architecture:** 7 containers (compatibility-domain), 4 deployed
> **Git commit:** 05469b6

---

## 1. Engine Synthesis Results

### Confirmed Working (11/16 supported engines)

| Engine | Container | RTF | Audio | Latency | Notes |
|--------|-----------|:---:|------:|--------|-------|
| **piper** | engine-current | 2.2× | 777ms | 2s | ONNX CPU. Fastest engine. |
| **kokoro** | engine-current | 6.5× | 1344ms | 9s | ONNX. |
| **melo** | engine-current | 11.7× | 1811ms | 21s | Fixed by numpy < 2.0 pin. |
| **matcha** | engine-current | 8.1× | 1183ms | 10s | ONNX flow-matching. |
| **outetts** | engine-current | 36.3× | 1293ms | 47s | LLM-based. Slowest working. Fixed by numpy pin. |
| **bark** | engine-current | 27.7× | 1760ms | 49s | Heavy VRAM. Fixed by numpy pin. |
| **styletts2** | engine-current | 35.5× | 1772ms | 63s | Needs langchain<0.3.0. |
| **f5tts** | engine-current | 31.0× | 682ms | 21s | Voice cloning. |
| **fishspeech** | engine-current | 18.4× | 1160ms | 21s | Voice cloning. |
| **zonos** | engine-current | 19.5× | 1637ms | 32s | Voice cloning. |
| **qwen3tts** | engine-qwen | 4.5× | 3840ms | 17s | **Newly promoted.** Patch applied. |

### Failing (5 engines)

| Engine | Error | Root Cause | Fix Path |
|--------|-------|------------|----------|
| **chattts** | HTTP 500 | ChatTTS library bug (LZMA encode/decode) | Needs fallback code path in tts_lab_engines.py |
| **chatterbox** | HTTP 500 | `DacModel` import missing | Needs DAC/descript-audio-codec pip package |
| **chatterboxturbo** | HTTP 500 | `DacModel` import missing | Same as chatterbox |
| **omnivoice** | HTTP 500 | `HiggsAudioV2TokenizerModel` not in tf 5.12.1 | Needs newer transformers or engine-omni container |
| **xtts** | HTTP 500 | torchcodec + torch nightly incompatibility | Coqui-TTS/torchcodec known issue |

### Timeout (1 engine)

| Engine | Error | Root Cause |
|--------|-------|------------|
| **dia** | 120s timeout | 1.6B model, may need more time or KV cache fix |

---

## 2. Fixes Applied During Deployment

### 2.1 GPU Access (engine-current)
**Problem:** engine-current had no `deploy.resources` in docker-compose.yml — no GPU visible.
**Fix:** Added inline GPU config (nvidia driver, count: 1, capabilities: [gpu]).
**Commit:** 2dd33c9

### 2.2 YAML Merge Keys (Docker Compose v5)
**Problem:** Double `<<:` merge directives rejected by Docker Compose v5.
**Fix:** Unrolled all `<<: *model-volume` + `<<: *gpu-config` anchors into inline YAML for engine-mid, engine-qwen, and orpheus.
**Commit:** 34b64ef

### 2.3 Profile Isolation (engine-legacy)
**Problem:** engine-legacy image doesn't exist — docker compose tried to build it on every `up -d`.
**Fix:** Added `profiles: [legacy]` to engine-legacy service.
**Commit:** 998ce87

### 2.4 sm_120 Support (stack-mid)
**Problem:** RTX 5060 Ti (Blackwell, sm_120) not supported by torch 2.10 stable.
**Fix:** Switched stack-mid from torch stable cu121 to torch nightly cu128 (later cu130).
**Commit:** 624bb6c

### 2.5 TransformGetItemToIndex (qwen3tts)
**Problem:** qwen_tts 0.1.1 requires transformers==4.57.3 (exact), but TransformGetItemToIndex class in masking_utils has broken __enter__.
**Fix:** Updated tts_lab_shims.py to replace TransformGetItemToIndex with a working no-op context manager (__enter__/__exit__). qwen-tts now installs its preferred transformers version naturally — no version war.
**Commit:** e0b8dd9

### 2.6 Torchvision CUDA Version Skew
**Problem:** Nightly cu128 index had torch 20260408 but torchvision 20260407 — 1-day skew caused `ResolutionImpossible`.
**Fix:** Unified all three stacks (current, mid) on cu130 nightly index where torch+torchvision builds are aligned.
**Commit:** 618f0ed

### 2.7 Numpy ABI Breakage
**Problem:** Torch nightly force-reinstall upgraded numpy to 2.x, breaking compiled C extensions in engine packages (bark, melo, outetts, styletts2, etc.) that were built against numpy 1.x ABI.
**Fix:** Added `numpy>=1.26,<2.0` pin AFTER the torch force-reinstall step.
**Commit:** 05469b6
**Impact:** Recovered 7 additional engines (was 3/16, now 10/15).

---

## 3. Container Fingerprints (Deployed)

| Stack | Container | torch | transformers | cuda | driver |
|-------|-----------|-------|-------------|------|--------|
| current | engine-current | 2.14.0.dev20260622+cu130 | 5.12.1 | 13.0 | 580.159.03 |
| mid | engine-mid | 2.12.0.dev20260408+cu128 | 4.51.3 | 12.8 | 580.159.03 |
| qwen | engine-qwen | 2.12.0.dev20260408+cu128 | 4.57.3 | 12.8 | 580.159.03 |

**Note:** engine-current upgraded to CUDA 13 during the cu130 switch. engine-mid and engine-qwen remain on CUDA 12.8 (built earlier, not yet rebuilt with cu130). This divergence should be resolved in the next rebuild cycle.

---

## 4. Architecture Validation

### What worked
- **Compatibility-domain partitioning:** Qwen3TTS isolation in engine-qwen was the right call. Its unique dependency constraints (qwen_tts==0.1.1, transformers==4.57.3, TransformGetItemToIndex patch) would have caused drift in any shared container.
- **Bridge networking:** Container DNS resolution (engine-current, engine-qwen) works correctly. Orchestrator → engine HTTP routing functional.
- **Lazy-load mode:** Engine server starts with 0 engines loaded. Synthesis triggers load → synthesize → optional eviction. VRAM management correct.
- **Promotion framework:** Qwen3TTS promoted from EXPERIMENTAL → SUPPORTED via automated gate validation. Summary recomputed on every write.

### What needs attention
- **engine-current torch version drift:** Upgraded from CUDA 12.8 to CUDA 13 during deployment troubleshooting. Needs full rebuild cycle to stabilize.
- **stack-mid/engine-mid not rebuilt:** Still on old torch 2.12.0.dev20260408. Needs rebuild with cu130 for consistency.
- **engine-qwen transformers version:** 4.57.3 (not in the documented 4.51-4.53 range). Works with the shim patch but documentation should be updated.

---

## 5. State After Deployment

```
Supported:    17  (11 runtime-confirmed, 6 pending verification)
Experimental:  7  (cosyvoice, csm, manatts, neutts, openvoice, vibevoice, higgs)
Blocked:       4  (orpheus, indextts, parler, s2pro)
Total:        28
```

Runtime-confirmed: piper, kokoro, melo, matcha, outetts, bark, styletts2, f5tts, fishspeech, zonos, qwen3tts (11)

---

## 6. Remaining Engine-Level Fixes

| Engine | Issue | Fix |
|--------|-------|-----|
| chattts | LZMA encode/decode bug | Use random speaker fallback (already in code, may need import path fix) |
| chatterbox | DacModel missing | `pip install descript-audio-codec` |
| chatterboxturbo | DacModel missing | Same as chatterbox |
| omnivoice | HiggsAudioV2TokenizerModel | Needs transformers > 5.12 or dedicated engine-omni container |
| xtts | torchcodec incompatible | Known torch nightly issue — coqui-tts installs but can't synth |
| dia | KV cache / timeout | Increase timeout or use Dia-1.6B-0626 with monkey-patch |
| vibevoice | vibevoice not in CONFIG_MAPPING | Registration shim or wait for SGLang/transformers update |
| higgs | Not tested | Same class of issue as vibevoice |
