# Ideogram 4 — Performance Optimizations

**Date:** 2026-06-09
**Target:** Arthur Image Lab (port 8002), RTX 5060 Ti 16 GB, Ubuntu VM

---

## Optimization Summary

| # | Optimization | Impact | Status |
|---|-------------|--------|--------|
| 1 | `asyncio.to_thread` — non-blocking generation | `/status` and `/logs` work during generation | ✅ |
| 2 | Pre-load at startup (background thread) | First API call is instant after load | ✅ |
| 3 | `HF_HUB_OFFLINE=1` — skip HF network checks | Saves ~30-60s per load | ✅ |
| 4 | Qwen3-VL 4-bit disk cache | Skips bitsandbytes re-quantization on restart | ✅ |
| 5 | Flash Attention 2 | 2-3x faster attention, 30% less VRAM | 🔄 |
| 6 | Request logging → ring buffer → Web UI Logs tab | Real-time monitoring | ✅ |

---

## 1. Non-Blocking Generation

**Problem:** `engines.generate()` ran synchronously, blocking the FastAPI event loop. `/status` and `/logs` timed out during generation.

**Fix:** Wrapped in `await asyncio.to_thread(engines.generate, engine_key, params)`

**File:** `image_lab_dispatch.py` line ~118

```python
results = await asyncio.to_thread(engines.generate, engine_key, params)
```

---

## 2. Pre-Load at Startup

**Problem:** First request after restart triggered 6-8 min model load → 503 errors.

**Fix:** `asyncio.create_task(_preload_ideogram4())` in `lifespan()`. Model loads in background thread at startup. Service responds immediately.

**File:** `image_lab.py` — `_preload_ideogram4()` function

---

## 3. HF_HUB_OFFLINE — Skip Network Checks

**Problem:** Every `from_pretrained()` call made HTTP HEAD requests to huggingface.co for each model file (~10 files = 30-60s).

**Fix:** `HF_HUB_OFFLINE=1` in `/opt/arthur-img/.env`. Forces HF library to use only local cache.

**Caveat:** All model files must be pre-cached. Remove `HF_HUB_OFFLINE` if adding new models.

---

## 4. Qwen3-VL 4-Bit Disk Cache

**Problem:** bitsandbytes re-quantizes Qwen3-VL (8B params, 5.2 GB) from bf16 to 4-bit on every restart (~30-60s CPU time).

**Fix:** After first load, save quantized model to `/opt/arthur-img-models/qwen3-vl-4bit-cache/`. On subsequent loads, load from cache directly.

**File:** `ideogram4_lab_engine.py` — `_cache_quantized_text_encoder()` and `_load_cached_text_encoder()`

**Cache path:** `/opt/arthur-img-models/qwen3-vl-4bit-cache/`

---

## 5. Flash Attention 2 (IN PROGRESS)

**Expected impact:**
- 2-3x faster attention computation in DiT transformer
- 30% less VRAM during denoising (~12 GB → ~8-9 GB)
- Generation time: ~60s → ~35-40s for QUALITY_48

**Installation:**
```bash
sudo systemctl stop arthur-imglab.service
MAX_JOBS=2 /opt/arthur-img-env/bin/pip install flash-attn --no-build-isolation
sudo systemctl start arthur-imglab.service
```

**Compatibility:** RTX 5060 Ti (Blackwell, sm_120). FA2 2.7+ has Blackwell support. Compiles from source — one-time cost of 10-15 min.

**Safety:** FA2 integrates via PyTorch's SDPA backend. Drop-in replacement for `F.scaled_dot_product_attention`. Affects ALL models that use SDPA — but is backwards compatible. No code changes needed.

---

## 6. Request Logging + Web UI Logs Tab

**Problem:** No visibility into what parameters external tools were sending.

**Fix:** 
- `_RingLogHandler` captures last 200 log entries in memory
- `GET /logs?n=200` API endpoint returns JSON
- "Logs" tab in Web UI with color-coded terminal-style viewer
- Manual refresh only (no auto-polling)

**Files:** `image_lab.py` (handler + endpoint), `image_lab_ui.py` (UI tab)

---

## Magic Prompt Priority Chain

```
1. Ideogram hosted API (FREE)  → IDEOGRAM_API_KEY
2. DeepSeek native + v1.txt    → DEEPSEEK_API_KEY
3. OpenRouter → DeepSeek       → OPENROUTER_API_KEY
```

v1.txt: Ideogram's 28 KB, 296-line system prompt with 19 formatting rules.

---

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Release freed VRAM back to OS |
| `HF_HUB_OFFLINE` | `1` | Skip HF network checks |
| `HF_HOME` | `/opt/arthur-img-models/huggingface` | Model cache location |
| `IDEOGRAM_API_KEY` | `***` | Free magic prompt API |
| `DEEPSEEK_API_KEY` | `***` | Fallback magic prompt |
| `OPENROUTER_API_KEY` | `***` | Last-resort fallback |

---

## Known External Tool Issues

From request logs (2026-06-09):
- `magic_prompt_aspect_ratio=1:1` but `width=1280 height=720` (16:9) — aspect mismatch
- `num_inference_steps=28` overriding QUALITY_48's 48 steps
- Prompt is meta-instructions, not scene description
- No retry on 503 (model loading)

---

## VRAM Budget (NF4, with offloading)

| Component | VRAM | State |
|-----------|:----:|-------|
| Conditional transformer | ~5.0 GB | GPU (during denoising) |
| Unconditional transformer | ~5.0 GB | CPU (offloaded, restored for CFG) |
| Qwen3-VL text encoder | ~1.3 GB | CPU (offloaded, restored once) |
| VAE | ~0.2 GB | CPU (offloaded, restored for decode) |
| CUDA overhead + buffers | ~2-3 GB | GPU |
| **Idle total** | **~10.6 GB** | |
| **Peak (generating)** | **~11.9 GB** | |
| **With FA2 (projected)** | **~8-9 GB** | |

---

## Pipeline Patches (ideogram4/src/ideogram4/pipeline_ideogram4.py)

8 patches for VRAM offloading during denoising loop:
- Lines ~320, 572, 612-657: Per-step component offload to CPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required
- Post-load offload in engine after `from_pretrained()`
- Defensive auto-restore in `_get_qwen3_vl_embeddings`
