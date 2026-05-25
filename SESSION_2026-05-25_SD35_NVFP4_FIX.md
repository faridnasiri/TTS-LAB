# Session Summary — SD 3.5 Large NVFP4 Fix
**Date:** 2026-05-25  
**VM:** `arthur@192.168.0.87` | Service: `arthur-imglab.service` on port 8002  
**Stack:** Python 3.11.0rc1, PyTorch 2.11.0+cu128, torchao 0.17.0+cu128, CUDA 12.8.90  
**GPU:** RTX 5060 Ti (15.48 GB)

---

## Problem

SD 3.5 Large with `quant=nvfp4` was returning **HTTP 503** with the error:

```
Cannot copy out of meta tensor; no data!
```

The engine appeared in the dropdown (nvfp4 option shown), but every generation request failed on load.

---

## Root Cause Analysis

### Original save (530 MB, corrupt)
The initial sd35 nvfp4 save was made while `arthur-imglab.service` was running with another model loaded (~14 GB VRAM used). When `nvfp4_save.py` ran `device_map="auto"`, the accelerate library offloaded layers to disk (meta tensors) because the GPU was nearly full.

**Result:** 706 of 815 weight tensors in the saved `.bin` shards were meta tensors — placeholders containing no actual data. Loading these with `device_map="cuda"` triggered the `Cannot copy out of meta tensor` crash.

### Second attempt (16.29 GB, wrong format)
The attempted workaround was to use `device_map={"":"cpu"}` to force CPU-only loading, avoiding the meta-tensor issue. However, NVFP4 quantization (torchao MX formats) **requires CUDA kernels** — it cannot run on CPU. With a CPU-only device map, `quantize_()` silently did nothing, and the full BF16 model (~16 GB) was saved instead.

Loading a 16 GB BF16 checkpoint as NVFP4 with `device_map="cuda"` caused **CUDA OOM**.

### Why wan-t2v nvfp4 worked
The Wan T2V NVFP4 save was done earlier when the GPU was free. With `device_map="auto"` and a free GPU, quantization runs on CUDA and produces the correct ~7.5 GB output. Zero meta tensors.

---

## Fixes

### 1. `image_lab_engines.py` — `_strip_missing_nvfp4_options()`
Previously this function only checked for the presence of `config.json`. It was updated to also validate shard file sizes: if the total `.bin` shard size for a model is less than 1 GB (`MIN_SHARD_BYTES`), the save is treated as missing or corrupted and removed from the dropdown.

This correctly caught the 530 MB corrupt sd35 save and (hypothetically) would catch the 16 GB BF16 save as a wrong-format save (too large to be a valid NVFP4).

### 2. `nvfp4_save.py` — device_map fix + meta tensor guard
- Changed device_map from `{"":"cpu"}` to `"auto"` so quantization runs on CUDA
- Added a pre-save meta tensor check: if any tensor in the model is still a meta tensor after loading, the script aborts with a clear error message explaining that the GPU was likely occupied during the run

### 3. Operational fix — stop service before saving
The save must be run with the service stopped so the GPU is completely free for `device_map="auto"` to load and quantize on GPU without offloading.

---

## Re-save Procedure

```bash
# On the VM:
sudo systemctl stop arthur-imglab.service

# Delete corrupted save
rm -rf /opt/arthur-img-models/nvfp4/sd35/

# Run re-save (in screen session for safety)
screen -dmS nvfp4-sd35 bash -c \
  'source /opt/arthur-img/.env; \
   /opt/arthur-img-env/bin/python /opt/arthur-img/nvfp4_save_sd35.py \
   > /tmp/nvfp4_sd35_save.log 2>&1'

# Monitor progress
tail -f /tmp/nvfp4_sd35_save.log

# Restart when done
sudo systemctl start arthur-imglab.service
```

**Timing (with fresh HF download):**
| Phase | Duration |
|---|---|
| HF download (2 BF16 shards, ~16 GB) | 2 min 23 s |
| Load + quantize on GPU | ~147 s |
| Save to disk | ~25 s |
| **Total** | **~172 s** |

---

## Results

| Metric | Value |
|---|---|
| Save size | **4.72 GB** (vs. 530 MB corrupt / 16 GB BF16) |
| Model load time | **21.6 s** (vs. 63.9 s for Q4_0 GGUF) |
| Inference (10 steps, 1024×1024) | **52 s** (5.24 s/it) |
| VRAM during inference | ~9.7 GB |
| Output | Valid 1024×1024 PNG |
| HTTP status | **200 OK** |

Journal log confirming success:
```
SD 3.5 Large ready (quant=nvfp4) in 21.6 s
100%|██████████| 10/10 [00:52<00:00,  5.24s/it]
Saved image sd35_c6f811ec-c34f-47d2-bef8-a6a57a64685b.png (1024x1024)
POST /generate/sd35 HTTP/1.1" 200 OK
```

---

## Key Lessons

- **`device_map="auto"` with occupied GPU → meta tensor saves.** Always stop the service before running nvfp4_save scripts.
- **`device_map={"":"cpu"}` bypasses NVFP4 quantization.** CUDA is required for torchao MX/NVFP4 formats. CPU-only device_map silently produces a BF16 save at full model size.
- **Validate shard size, not just config presence.** A corrupt or wrong-format save can pass a file-existence check. Size thresholds catch both cases.
- **HF_TOKEN location on VM:** `/opt/arthur-img/.env` — must `source` this before running save scripts.

---

## NVFP4 Model State (post-fix)

| Model | Path | Size | Status |
|---|---|---|---|
| sd35 | `/opt/arthur-img-models/nvfp4/sd35/transformer/` | 4.72 GB | ✅ Valid |
| wan-t2v | `/opt/arthur-img-models/nvfp4/wan-t2v/transformer/` | 7.5 GB | ✅ Valid |
| flux2 | `/opt/arthur-img-models/nvfp4/flux2/transformer/` | (empty) | ⚠️ Not saved — stripped from dropdown |

---

## API Response Format

The `/generate/sd35` endpoint returns:

```json
{
  "results": [
    {
      "id": "<uuid>",
      "engine": "sd35",
      "filename": "sd35_<uuid>.png",
      "url": "<relative-url>",
      "base64": "<base64-encoded-PNG>",
      "type": "image",
      "width": 1024,
      "height": 1024,
      "params": {
        "prompt": "...",
        "quant": "nvfp4",
        "num_inference_steps": 10
      },
      "created_at": 1779729145.3251185
    }
  ]
}
```

Note: The `url` field is a relative path, not a base64 string. Use the `base64` field to decode the image directly.
