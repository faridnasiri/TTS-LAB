# Ideogram 4 Integration — VRAM Fix & Deployment

**Date:** 2026-06-08 to 2026-06-09  
**Target:** Arthur Image Lab (port 8002) on Ubuntu VM (RTX 5060 Ti 16 GB, 96 GB RAM)  
**Goal:** Deploy Ideogram 4 (9.3B DiT) text-to-image model that fits in 16 GB VRAM and generates images via the existing web UI and API.

---

## 1. Architecture & Problem Space

### Model components
| Component | Disk Size | Expected VRAM (NF4) |
|-----------|:--------:|:-------------------:|
| Conditional transformer (DiT) | 4.9 GB | ~5.0 GB (NF4 = uint8 storage) |
| Unconditional transformer (DiT) | 4.9 GB | ~5.0 GB |
| Qwen3-VL 8B text encoder | 5.2 GB | ~1.3 GB (true bnb 4-bit) |
| VAE (autoencoder) | 161 MB | ~0.2 GB |
| CUDA allocator + PyTorch overhead | — | ~3 GB |
| **Total load** | — | **~15.8 GB** |

A 16 GB RTX 5060 Ti has ~15.85 GiB usable. That leaves **~13 MB free** — not enough for inference compute buffers (activations, attention scores, etc.).

### Key insight
The official `ideogram-ai/ideogram-4-nf4` repo uses:
- **NF4 = uint8 storage** for the DiT transformers (not true 4-bit — each weight is 8 bits)
- **bitsandbytes NF4** for Qwen3-VL (true 4-bit via `load_in_4bit=True`)
- Two separate DiT transformers (conditional + unconditional) for Classifier-Free Guidance (CFG)

---

## 2. Approach: Why Not Just Load Larger Quant?

The user asked: *"why not use nvfp4 or skip the larger model?"*

We were already using the smallest official quant (NF4). The issue was not model size — it was that all 4 components were simultaneously resident in VRAM. The solution was **aggressive CPU offloading of components not needed during the denoising loop**.

---

## 3. Initial Integration (Before VRAM Fix)

### Files modified for engine integration
| File | Changes |
|------|---------|
| `image_lab_config.py` | Added `ideogram4` EngineInfo with all 15 params; removed `client_only=True` from magic prompt fields |
| `image_lab_dispatch.py` | Added 6 Form fields (preset, mu, std, use_magic_prompt, magic_prompt_input, magic_prompt_aspect_ratio) + params dict entries |
| `image_lab_engines.py` | Added `_load_ideogram4()`, `_generate_ideogram4()`, `_probe_ideogram4()`, dispatch tables, unload entry |
| `image_lab_ui.py` | Added engine tab, gallery filter, checkbox param rendering, magic prompt show/hide JS |
| `ideogram4_lab_engine.py` | Full engine wrapper: load, generate, probe, DeepSeek/OpenRouter magic prompt |
| `scripts/deploy/deploy_image_lab.ps1` | ideogram4 pip install, file copy, OPENROUTER_API_KEY in .env |
| `secrets.env` | Added OPENROUTER_API_KEY placeholder |

### Pre-existing bugs fixed
- Indentation errors in `image_lab_engines.py` (`_unload_current()` and `probe_availability()`)
- Missing `try:` block in `_load_flux2()` text encoder offloading

---

## 4. VRAM Fix — What Failed

### Attempt 1: Increase system RAM + swap
- **Hypothesis:** OOM killer was killing the process during model load (30 GB CPU RAM needed)
- **Action:** Upgraded VM from 31 GB → 62 GB RAM, increased swap to 31 GB
- **Result:** Model loaded successfully (no OOM kill), but still **15.8 GB VRAM / 13 MB free** → inference OOM

### Attempt 2: Pre-offload VAE and Qwen3-VL in our engine wrapper
- **Hypothesis:** Components not needed during denoising can be offloaded
- **Action:** Added `.to('cpu')` calls in `generate_ideogram4()`
- **Result:** The pipeline's internal `from_pretrained()` loads all components to GPU first, then our offload happens — but PyTorch caching allocator doesn't release memory back to OS with default config

### Attempt 3: Pipeline-level offload patches
- **Hypothesis:** Offloading must happen inside the pipeline's `__call__` method
- **Action:** Patched `ideogram4/src/ideogram4/pipeline_ideogram4.py` to offload components
- **Bug:** `import torch` / `import torch as _torch` inside function body caused Python to shadow the module-level import → `UnboundLocalError: cannot access local variable 'torch'`
- **Result:** Fixed by removing all function-level `import torch`, using only module-level import at line 9

---

## 5. VRAM Fix — What Worked ✅

### Fix 5A: Environment change
```bash
# /opt/arthur-img/.env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
This tells PyTorch's caching allocator to release memory segments back to the OS when freed, rather than hoarding them.

### Fix 5B: Engine-level post-load offload (`ideogram4_lab_engine.py`)
Added after `from_pretrained()` returns — immediately offloads Qwen3-VL, unconditional transformer, and VAE to CPU:
```python
for comp_name in ['text_encoder', 'unconditional_transformer', 'autoencoder']:
    if hasattr(pipe, comp_name) and getattr(pipe, comp_name) is not None:
        dev = next(comp.parameters()).device
        if dev.type == 'cuda':
            setattr(pipe, comp_name, comp.to('cpu'))
gc.collect()
torch.cuda.empty_cache()
```
This prevents the initial 15.8 GB OOM on the first `__call__`.

### Fix 5C: Defensive guard in `_get_qwen3_vl_embeddings`
Added before `language_model = self.text_encoder.language_model`:
```python
if next(self.text_encoder.parameters()).device.type != self.device.type:
    self.text_encoder = self.text_encoder.to(self.device)
    torch.cuda.empty_cache()
```
Auto-restores Qwen3-VL to GPU when it was left on CPU by a previous generation.

### Fix 5D: Pipeline offload patches (`ideogram4/src/ideogram4/pipeline_ideogram4.py`)

| Line | Patch | What it does | VRAM saved |
|------|-------|-------------|:---:|
| 572 | `self.text_encoder.to('cpu')` | Offload Qwen3-VL after `_encode_text()` (never used again) | ~3 GB |
| 574 | `torch.cuda.empty_cache()` | Release allocator cache back to OS | — |
| 612 | `self.autoencoder.to('cpu')` | Offload VAE before denoising loop | ~0.2 GB |
| 614 | `self.unconditional_transformer.to('cpu')` | Offload unconditional DiT before loop | ~5 GB |
| 616 | `torch.cuda.empty_cache()` | Release allocator cache | — |
| 632-638 | Per-step swap for CFG | Bring unconditional to GPU for its forward, offload after | enables CFG |
| 650 | `self.autoencoder.to(self.device)` | Restore VAE for `_decode()` | — |
| 657 | `self.text_encoder.to(self.device)` | Restore Qwen3-VL after decode | — |

### Fix 5E: Pipeline loading order fix
The original `from_pretrained()` loaded both transformers simultaneously → peak RAM spike. Patched to load one at a time with `gc.collect()` between:
```python
# Load conditional transformer first
conditional_state_dict = _load_indexed_or_single_state_dict(...)
conditional_transformer = _build_transformer(...)
del conditional_state_dict
gc.collect()

# Then load unconditional transformer
unconditional_state_dict = _load_indexed_or_single_state_dict(...)
unconditional_transformer = _build_transformer(...)
del unconditional_state_dict
gc.collect()
```

---

## 6. Results

### 6A. Initial Fix (guidance=1.0, no CFG)

```
Before:  VRAM 15,836 MiB / 13 MiB free  →  HTTP 503: CUDA OOM
After:   VRAM  5,838 MiB / 10,011 MiB free  →  HTTP 200 ✅
```

**VRAM reduction: 10 GB (63%) — from 15.8 GB to 5.8 GB**

### 6B. Benchmark (guidance=1.0, TURBO 12 steps)

| Resolution | Time | VRAM Peak | Base64 | Status |
|-----------:|:-----|:---------|:------|:------|
| 256×256 | 378.5s* | 5,838 MiB | 101 KB | ✅ |
| 384×384 | 11.6s | ~8,200 MiB | 257 KB | ✅ |
| 512×512 | 13.4s | ~9,800 MiB | 500 KB | ✅ |
| 768×768 | 22.2s | 11,546 MiB | 1.2 MB | ✅ |

\*256×256 includes model loading (~6 min). Subsequent: pure inference 11-22s.

### 6C. Newspaper Text Render Test

**Test prompt:** Newspaper front page "THE GLOBAL TRAVELLER" with 6 text elements, 768×1024, DEFAULT_20 preset.

| Guidance | ID | Result | Quality |
|----------|----|--------|---------|
| 1.0 (no CFG) | `88ce69a9` | ✅ Generated | ❌ Garbled/mixed text |
| 7.0 (full CFG) | `a48b6908` | ✅ Generated | ✅ Proper text rendering |

**Key finding:** guidance_scale=1.0 is sufficient for simple images but **CFG (guidance≥7.0) is essential for text rendering/typography**. The per-step transformer swap enables CFG while staying under 16 GB VRAM.

**VRAM with CFG enabled:** 11,944 MiB loaded, 3,905 MiB free (75% utilization)

---

## 7. Magic Prompt (External AI)

Instead of running Qwen3-VL's generative head locally for JSON caption expansion, the engine uses:

| Provider | API | Status |
|----------|-----|--------|
| DeepSeek (via OpenRouter) | `sk-or-v1-...` | ✅ Configured |
| Claude Sonnet (via OpenRouter) | `sk-or-v1-...` | ⚠️ Fallback |
| Local Qwen3-VL | `ideogram4.magic_prompt` | ❌ Disabled (too heavy) |

The `OPENROUTER_API_KEY` is read from `secrets.env` and injected into the VM's `/opt/arthur-img/.env`.

---

## 8. Benchmark Results

Tested 2026-06-09 with nf4 quant, TURBO 12 steps, guidance=1.0, seed=42.

| Resolution | Time | VRAM Peak | Base64 Size | Status |
|-----------:|:-----|:---------|:-----------|:------|
| 256×256 | 378.5s* | ~5,800 MiB | 101 KB | ✅ |
| 384×384 | 11.6s | ~8,200 MiB | 257 KB | ✅ |
| 512×512 | 13.4s | ~9,800 MiB | 500 KB | ✅ |
| 768×768 | 22.2s | 11,546 MiB | 1.2 MB | ✅ |

\* 256×256 includes model loading time (~6 min). Subsequent gens are pure inference.

**Key observations:**
- **Pure inference: 11–22s** per image across all resolutions
- **VRAM headroom: ~4.3 GB free** at 768×768 (72% utilization)
- **1024×1024 should fit** based on the linear scaling (~650 KB at 512 → ~1.2 MB at 768)
- **Subsequent generations reuse the loaded model** — no reload needed
- **Restore fix confirmed working** — all 4 gens succeeded without device mismatch errors

---

## 9. Remaining Work

| Task | Priority |
|------|:--------:|
| Test at 512×512 and 1024×1024 resolution | High |
| Test with guidance_scale > 1.0 (brings unconditional back to GPU) | High |
| Test FP8 quant (should fit even better) | Medium |
| Test magic prompt via DeepSeek API | Medium |
| Persist pipeline patches across ideogram4 package updates | Low |
| Add OOM retry logic in engine | Low |

---

## 9. Remaining Work

| Task | Priority | Status |
|------|:--------:|:------|
| Test 1024×1024 resolution | High | ⬜ |
| Test with guidance_scale > 1.0 (brings unconditional back) | High | ⬜ |
| Test FP8 quant | Medium | ⬜ |
| Test magic prompt via DeepSeek API | Medium | ⬜ |
| Persist pipeline patches across ideogram4 package updates | Low | ⬜ |
| Add OOM retry logic in engine | Low | ⬜ |

---

## 10. Key Lessons

1. **`import torch` inside a function body is a bug** — it shadows the module-level import and causes `UnboundLocalError`
2. **PyTorch caching allocator hoards freed memory** — use `expandable_segments:True`
3. **`.to('cpu')` + `gc.collect()` + `torch.cuda.empty_cache()`** is the reliable VRAM offload pattern
4. **Must restore offloaded components after `__call__`** — otherwise next gen fails with device mismatch
5. **Offload immediately after `from_pretrained()`** — prevents initial 15.8 GB OOM before first `__call__`
6. **Defensive guard in `_encode_text`** — auto-restores Qwen3-VL if left on CPU by previous failure
7. **guidance_scale=1.0 ≠ text quality** — CFG is essential for Ideogram 4's text rendering
8. **Per-step transformer swap enables CFG** — bring unconditional to GPU only for its forward pass
9. **NF4 DiT weights are uint8 (8-bit), not true 4-bit** — each transformer ~5 GB in VRAM
10. **Two DiT transformers = double VRAM** — conditional (denoising) + unconditional (CFG)
11. **62 GB system RAM needed** — safetensors loaded to CPU first, then quantized
12. **Engine offload + pipeline guard + per-step swap = 16 GB viable for Ideogram 4 NF4**


## 11. Automation Tooling

**Status checker** deployed at `/tmp/check_status.py` on VM:
```bash
# One-time status check
ssh VM "python3 /tmp/check_status.py [result_file]"

# Continuous monitoring (every 30s)
ssh VM "python3 /tmp/check_status.py --watch 30"
```
Shows: VRAM, service status, CPU/RAM, loading state, test results.

**Benchmark script** at `c:\repos\TTS-LAB\_bench_ig4.py` (SCP to VM):
```bash
ssh VM "python3 /tmp/bench_ig4.py"
```
Tests 4 resolutions (256/384/512/768) sequentially with timing per gen.

## 12. Final Configuration Summary

| Component | Location | What |
|-----------|----------|------|
| Service | `/etc/systemd/system/arthur-imglab.service` | Runs `image_lab.py` on port 8002 |
| Env vars | `/opt/arthur-img/.env` | HF_TOKEN, OPENROUTER_API_KEY, PYTORCH_CUDA_ALLOC_CONF |
| Engine | `/opt/arthur-img/ideogram4_lab_engine.py` | Post-load offload, DeepSeek magic prompt |
| Pipeline patches | `/opt/arthur-img/ideogram4/src/ideogram4/pipeline_ideogram4.py` | 8 patches (lines 320, 572, 612-657) |
| Web UI | `image_lab_ui.py` | ideogram4 tab, gallery filter, magic prompt UX |
| Deploy script | `scripts/deploy/deploy_image_lab.ps1` | Installs ideogram4, copies files, sets env |
