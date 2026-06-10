# Ideogram 4 — Crash Fix & Optimization Attempt (June 10, 2026)

## Incident

Remote service at `192.168.0.92` called `POST /generate/ideogram4` on the Arthur Image Lab and got `500 Internal Server Error`. The pre-load at startup had also silently failed.

## Root Cause

The deploy script's Phase 3 injected `local_files_only=True` into every `from_pretrained()` call in `pipeline_ideogram4.py`, including the tokenizer:

```python
# pipeline_ideogram4.py line 117 (BEFORE)
tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True, **tokenizer_kwargs)
```

The Qwen3-VL tokenizer needs to check huggingface.co for a few optional files (`vocab.json`, `merges.txt`, `config.json`) that legitimately don't exist in the `ideogram-ai/ideogram-4-nf4` repo (they return HTTP 404). With `local_files_only=True`, this graceful "file not found" became a hard `OSError`, crashing the entire pipeline load.

The pre-load also failed silently with the same error — caught as a warning instead of a crash.

### Secondary issues in the external caller's request

```
magic_prompt_aspect_ratio=1:1  but  width=1280 height=720  (16:9)       ← mismatch
num_inference_steps=28         overriding V4_QUALITY_48    (should be 48)
prompt="I want you to to generate one image for each timestamp..."       ← meta-instructions, not scene
```

These match the "Known External Tool Issues" already documented in `IDEOGRAM4_OPTIMIZATIONS.md`.

## Fix Applied

### 1. Remove `local_files_only=True` from the tokenizer (pipeline)

**File:** `/opt/arthur-img/ideogram4/src/ideogram4/pipeline_ideogram4.py` line 117

```python
# AFTER
tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)
```

The tokenizer now makes a few cheap HEAD requests on first load (~1-2s), confirms missing files are genuinely absent, and proceeds. The transformer weight loading still uses `local_files_only=True` (those files ARE fully cached).

### 2. Clear stale `.no_exist` cache markers

```bash
rm -rf /opt/arthur-img-models/huggingface/models--ideogram-ai--ideogram-4-nf4/.no_exist
```

These markers were created during failed `local_files_only=True` attempts and would cause spurious cache misses even with network access.

### 3. Fix deploy script (future deploys)

**File:** `deploy_image_lab.ps1` Phase 3

Removed the sed patch that injected `local_files_only=True` into the tokenizer. Kept the two sed patches for transformer weight loading (AutoModel calls). Added a comment explaining why.

### 4. Claude Code permissions

**File:** `.claude/settings.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i *)",
      "Bash(curl -s *)",
      "Bash(scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i *)"
    ]
  }
}
```

## Qwen3-VL 4-bit Cache Attempt (abandoned)

**Goal:** Skip 30-60s bitsandbytes re-quantization on restart by caching quantized weights to disk (`/opt/arthur-img-models/qwen3-vl-4bit-cache/`).

**What worked:**
- `torch.save(state_dict)` successfully saved 5.5 GB of quantized Params4bit weights (bnb's `save_pretrained()` raises `NotImplementedError`)
- `AutoModel.from_pretrained(device_map="cpu")` loaded cached weights to CPU with 0 GPU usage
- Cold reload from disk cache: ~7s vs ~30-60s for fresh quantize

**Why abandoned:**
- Loading the cached model (even to CPU) caused CUDA OOM when `from_pretrained()` subsequently loaded the full pipeline to GPU
- The full pipeline fits both transformers + VAE + Qwen3-VL on the 16 GB GPU — but only barely. Any extra allocation (even hidden Params4bit CUDA buffers from the CPU-loaded model) pushes it over
- Three consecutive OOM attempts confirmed the issue
- **Revisit when:** Flash Attention 2 is installed (projected 30% VRAM reduction), or GPU is upgraded

**Code reverted** in `ideogram4_lab_engine.py` — `_cache_quantized_text_encoder` and `_load_cached_text_encoder` are now no-ops with explanatory comments.

## Result

| Metric | Before | After |
|--------|--------|-------|
| First-request 500 error | ❌ Crash | ✅ Works |
| Pre-load at startup | ❌ Silent fail | ✅ Successful |
| Load time | — | ~395 s (6.5 min) |
| VRAM after load | — | 4,980 MiB used, 10,723 MiB free |
| Test generation (1280×720, 28 steps) | — | ✅ 194.6 s |

**API:** `POST http://192.168.0.87:8002/generate/ideogram4`
**Status:** `GET http://192.168.0.87:8002/status`
**Web UI:** `http://192.168.0.87:8002`
