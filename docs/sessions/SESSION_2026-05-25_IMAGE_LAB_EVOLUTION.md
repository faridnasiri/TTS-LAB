# Arthur Image Lab — Evolution After Initial Reference Document
**Date:** 2026-05-25  
**VM:** `arthur@192.168.0.87` | Service: `arthur-imglab.service` on port **8002**  
**GPU:** NVIDIA RTX 5060 Ti 16 GB GDDR7 (driver 580.159.03, CUDA 12.8)  
**Covers:** All work done after `ARTHUR_IMAGE_LAB_REFERENCE.md` was created

---

## Table of Contents

1. [Overview — What Changed and Why](#1-overview--what-changed-and-why)
2. [Architecture Shift: BnB NF4 → GGUF Quantization](#2-architecture-shift-bnb-nf4--gguf-quantization)
3. [FLUX.2 Group Offloading — OOM Root Cause and Fix](#3-flux2-group-offloading--oom-root-cause-and-fix)
4. [Stable Diffusion 3.5 — GGUF Migration + preq_save.py](#4-stable-diffusion-35--gguf-migration--preq_savepy)
5. [Wan2.2 — GGUF Migration](#5-wan22--gguf-migration)
6. [New Engine: FLUX.2 Klein 4B](#6-new-engine-flux2-klein-4b)
7. [New Quantization Pathway: NVFP4 (Blackwell Native)](#7-new-quantization-pathway-nvfp4-blackwell-native)
8. [Utility Scripts Added](#8-utility-scripts-added)
9. [VM Infrastructure: Swap Space](#9-vm-infrastructure-swap-space)
10. [UI Evolution](#10-ui-evolution)
11. [Engine Testing Results](#11-engine-testing-results)
12. [Current Disk Layout](#12-current-disk-layout)
13. [Current Engine Catalogue](#13-current-engine-catalogue)
14. [Deployment Workflow (Current)](#14-deployment-workflow-current)
15. [Complete File Inventory](#15-complete-file-inventory)

---

## 1. Overview — What Changed and Why

The initial reference document (`ARTHUR_IMAGE_LAB_REFERENCE.md`) described a version of the image lab that used:
- **FLUX.2**: Loaded from `diffusers/FLUX.2-dev-bnb-4bit` using BitsAndBytes NF4 pre-quantization with `device_map="balanced"`
- **SD 3.5 Large**: Loaded from `stabilityai/stable-diffusion-3.5-large` in full bfloat16 with `enable_model_cpu_offload()`
- **Wan2.2**: Loaded from the official Diffusers repos in full bfloat16 with `enable_model_cpu_offload()`
- **3 engines total** (no FLUX.2 Klein)

The codebase has since been substantially rearchitected. The primary drivers were:

| Problem | Solution |
|---|---|
| BnB NF4 FLUX.2 was still OOMing despite `device_map="balanced"` | Switch to GGUF quantization + group offloading |
| SD 3.5 text encoders (T5-XXL ~9 GB) caused slow sequential offloading | Save shared components once via `preq_save.py`, load transformer-only via GGUF |
| No way to select quantization level interactively | GGUF exposes Q3_K_M / Q4_K_M / Q5_K_M / Q8_0 options in the UI |
| RTX 5060 Ti is a Blackwell GPU (SM100+) — NVFP4 is native 4-bit | Added `nvfp4_save.py` quantization pathway |
| Missing compact/fast image model | Added FLUX.2 Klein 4B (step-distilled, ~13 GB VRAM) |

---

## 2. Architecture Shift: BnB NF4 → GGUF Quantization

### What GGUF Is

GGUF (GPT-Generated Unified Format) is the quantization format used by llama.cpp. diffusers 0.38.0 added native GGUF transformer loading for Flux2, SD3.5, and Wan models via `from_single_file()` + `GGUFQuantizationConfig`.

GGUF quantization levels available:

| Level | Bits/weight | FLUX.2 size | SD3.5 size | Quality |
|---|---|---|---|---|
| Q3_K_M | ~3.35 bits | 16 GB | 4.0 GB | Lowest |
| Q4_K_M | ~4.45 bits | 20 GB | 4.8 GB | Good (recommended) |
| Q5_K_M | ~5.45 bits | 24 GB | 5.8 GB | Better |
| Q8_0 | ~8.5 bits | 35 GB | 8.8 GB | Near-lossless |

GGUF files are single flat files (no sharded safetensors). They are sourced from:
- FLUX.2: `city96/FLUX.2-dev-gguf` on HuggingFace
- SD 3.5: `city96/stable-diffusion-3.5-large-gguf`
- Wan2.2: `QuantStack/Wan2.2-T2V-A14B-GGUF` and `QuantStack/Wan2.2-I2V-A14B-GGUF`

### GGUF Loading Pattern

```python
from diffusers import GGUFQuantizationConfig, Flux2Transformer2DModel

gguf_path = "/opt/arthur-img-models/gguf/flux2/flux2-dev-Q4_K_M.gguf"

transformer = Flux2Transformer2DModel.from_single_file(
    gguf_path,
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype         = torch.bfloat16,
)
```

The `from_single_file()` call loads only the transformer weights (no text encoder, no VAE). The other pipeline components are loaded separately and assembled into the full pipeline.

### Key Difference from BnB NF4

| Aspect | BnB NF4 | GGUF |
|---|---|---|
| Source repo | `diffusers/FLUX.2-dev-bnb-4bit` | `city96/FLUX.2-dev-gguf` |
| File format | Sharded safetensors with quantization metadata | Single `.gguf` file |
| Loading API | `from_pretrained(repo_id, device_map="balanced")` | `from_single_file(path, GGUFQuantizationConfig)` |
| Device placement | `device_map` via accelerate | Manual (group offloading) |
| AlignDevicesHook | Added by accelerate (conflicts with group offloading) | Not added |
| Selectable quality | Fixed (one checkpoint) | Q3_K_M / Q4_K_M / Q5_K_M / Q8_0 |
| Non-transformer components | In same repo (BnB 4-bit) | From `diffusers/FLUX.2-dev-bnb-4bit` (text encoder) or `preq_save.py` output (SD3.5/Wan) |

### GGUF File Cache on Disk

GGUF files are stored separately from the HuggingFace cache:

```
/opt/arthur-img-models/gguf/
├── flux2/
│   ├── flux2-dev-Q3_K_M.gguf   (~16 GB)
│   ├── flux2-dev-Q4_K_M.gguf   (~20 GB)
│   ├── flux2-dev-Q5_K_M.gguf   (~24 GB)
│   └── flux2-dev-Q8_0.gguf     (~35 GB)
├── sd35/
│   ├── sd3.5_large-Q4_0.gguf   (~4.8 GB)
│   ├── sd3.5_large-Q5_0.gguf   (~5.8 GB)
│   └── sd3.5_large-Q8_0.gguf   (~8.8 GB)
├── wan-t2v/
│   ├── HighNoise/               (8 files — Q3_K_M/Q4_K_M/Q5_K_M/Q8_0)
│   └── LowNoise/                (8 files — same variants)
└── wan-i2v/
    ├── HighNoise/               (8 files)
    └── LowNoise/                (8 files)
```

Files are downloaded on first use via `_ensure_gguf()` in `image_lab_engines.py`. The `gguf_download.py` utility script pre-downloads all variants to avoid delays on first generation.

---

## 3. FLUX.2 Group Offloading — OOM Root Cause and Fix

### The Problem

Even after switching from BnB NF4 to GGUF quantization, FLUX.2 was still producing CUDA OOM errors. The reason:

When `Flux2Transformer2DModel.from_single_file()` loads a GGUF file, it loads the transformer weights into CPU RAM. There is no `device_map` involved — the model is on CPU after loading.

When `Flux2Pipeline.from_pretrained("diffusers/FLUX.2-dev-bnb-4bit")` is then called with `transformer=...`, it assembles the pipeline. If `device_map="balanced"` was then applied to the pipeline as a whole, accelerate added `AlignDevicesHook` to the transformer.

The problem: `apply_group_offloading()` (diffusers 0.38.0+) conflicts with `AlignDevicesHook`. Calling `apply_group_offloading()` on a module that already has `AlignDevicesHook` raises:

```
ValueError: The model has already been dispatched via `device_map` or similar ...
```

Attempting to work around this by calling `remove_hook_from_module()` then `apply_group_offloading()` succeeded for the transformer but the resulting execution path was unstable.

### The Root Cause: CPU → CUDA Allocation of the Full GGUF Transformer

At Q4_K_M, the FLUX.2 transformer is ~20 GB. Without any offloading:
1. It sits in CPU RAM after `from_single_file()` (~20 GB CPU)
2. During the first `pipe()` call, PyTorch needs to run a forward pass
3. Without device placement hooks, it tries to execute on whichever device its `device` attribute points to
4. Moving a 20 GB model to CUDA in one shot → OOM (GPU only has 15.48 GB)

### The Fix: Load on CPU, Apply Leaf-Level Group Offloading Before Pipeline Assembly

```python
# Step 1: Load transformer from GGUF — stays on CPU
transformer = Flux2Transformer2DModel.from_single_file(
    gguf_path,
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype         = torch.bfloat16,
    # No device_map, no device argument — loaded to CPU
)

# Step 2: Apply group offloading BEFORE assembling into pipeline
#         (before accelerate can add AlignDevicesHook)
from diffusers.hooks import apply_group_offloading
apply_group_offloading(
    transformer,
    onload_device  = torch.device("cuda"),
    offload_device = torch.device("cpu"),
    offload_type   = "leaf_level",   # One leaf layer at a time → peak VRAM ~300–500 MB per step
    use_stream     = False,
)

# Step 3: Load text encoder (from BnB NF4 repo) — also on CPU initially
text_encoder = AutoModel.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    subfolder  = "text_encoder",
    device_map = "cpu",              # Stays on CPU (NF4 BnB 4-bit)
    dtype      = torch.bfloat16,
    token      = HF_TOKEN,
)
apply_group_offloading(
    text_encoder,
    onload_device  = torch.device("cuda"),
    offload_device = torch.device("cpu"),
    offload_type   = "leaf_level",
    use_stream     = False,
)

# Step 4: Assemble pipeline — passes already-offloaded models
pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    transformer  = transformer,      # already has group offload hooks
    text_encoder = text_encoder,     # already has group offload hooks
    torch_dtype  = torch.bfloat16,
    token        = HF_TOKEN,
)

# Step 5: VAE — small enough (~300 MB) to keep on GPU permanently
pipe.vae = pipe.vae.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

**Why `device_map="cpu"` for the text encoder works with group offloading:**  
`device_map="cpu"` with a single CPU destination does NOT add `AlignDevicesHook` (confirmed empirically). It simply places all modules on CPU. `apply_group_offloading()` can then be applied cleanly.

**Why `use_stream=False`:**  
CUDA streams for async offloading require pinned memory. The GGUF quantized tensors and BnB NF4 tensors are not always pinnable. `use_stream=False` uses synchronous CPU↔GPU transfers, which is slower but always safe.

### Group Offloading Memory Profile

With `leaf_level` group offloading, at any point during the transformer forward pass:
- **On GPU**: Only the current leaf layer being executed (~10–50 MB typical)
- **On CPU**: All other transformer layers + text encoder (pending their forward pass)

Peak VRAM during FLUX.2 generation at Q4_K_M:
- CUDA driver overhead: ~0.5 GB
- VAE (permanently on GPU): ~0.3 GB
- Current leaf layer(s) during transformer forward: ~0.3–0.5 GB
- Current component during text encoder forward: ~0.2–0.5 GB
- **Total peak: ~1.5–2 GB** (far below the 15.48 GB limit)

This is the tradeoff vs. BnB NF4 with `device_map="balanced"` (~10 GB peak VRAM):
- **Group offloading**: Lower peak VRAM, slower (more CPU↔GPU transfers during forward)
- **BnB NF4 balanced**: Higher peak VRAM (~10 GB), faster (less movement)

With a 16 GB GPU, group offloading is the safe choice for the 32B FLUX.2 transformer.

### Generator Device Fix

Because group offloading makes `transformer.device` point to `"cuda"` (the `onload_device`), the pipeline creates latents on CUDA. The generator must also be on CUDA:

```python
# Before fix (broken):
generator = torch.Generator("cpu").manual_seed(seed)
# Raises: RuntimeError: Expected all tensors to be on the same device

# After fix (correct):
generator = torch.Generator("cuda").manual_seed(seed)
```

### AlignDevicesHook Guard

A safety guard was added for the text encoder case, since `device_map="cpu"` *sometimes* adds an `AlignDevicesHook` depending on accelerate version:

```python
try:
    apply_group_offloading(text_encoder, ...)
except ValueError as exc:
    if "AlignDevicesHook" in str(exc) or "CpuOffload" in str(exc):
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(text_encoder, recurse=True)
        apply_group_offloading(text_encoder, ...)
    else:
        raise
```

---

## 4. Stable Diffusion 3.5 — GGUF Migration + preq_save.py

### Problem with Original SD 3.5 Loading

The original approach (`from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=bfloat16, enable_model_cpu_offload())`) required downloading the full 40 GB checkpoint including:
- T5-XXL text encoder: ~9 GB in bfloat16
- CLIP-L + CLIP-G text encoders: ~1 GB total
- MMDiT transformer: ~15 GB in bfloat16
- VAE: ~330 MB

Every load cycle (switching engines) re-loaded all of these from the HF cache.

### The `preq_save.py` Approach

`preq_save.py` is a one-time setup script that:

1. **Copies shared pipeline components** (text encoders, VAE, tokenizers, scheduler, configs) from the HF cache to a dedicated `shared/` directory using `rsync -avL` (follows symlinks to copy real files, not HF cache symlinks).

2. **Saves them once to `/opt/arthur-img-models/quantized/<model>/shared/`**

After running `preq_save.py`, loading SD 3.5 with GGUF requires only:
- Load GGUF transformer via `from_single_file()` (Q4_0: ~4.8 GB file, fast)
- Load pipeline from `shared/` path with `transformer=<gguf_transformer>`

```python
# SD 3.5 loading (GGUF, after preq_save.py)
transformer = SD3Transformer2DModel.from_single_file(
    "/opt/arthur-img-models/gguf/sd35/sd3.5_large-Q4_0.gguf",
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype         = torch.bfloat16,
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/opt/arthur-img-models/quantized/sd35/shared",   # local dir, no network
    transformer = transformer,
    torch_dtype = torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
```

`enable_model_cpu_offload()` is still used for SD 3.5 because its text encoders (T5-XXL) need sequential GPU offloading — there is no group offloading complication since the SD 3.5 transformer GGUF is not pre-quantized with BnB NF4.

### SD 3.5 VRAM Budget at Q4_0

| Component | VRAM when active | Notes |
|---|---|---|
| MMDiT transformer (GGUF Q4_0) | ~5 GB | On GPU during denoising steps |
| T5-XXL text encoder | ~9 GB | On GPU during text encoding only |
| CLIP-L + CLIP-G | ~1 GB | On GPU during text encoding |
| VAE | ~0.3 GB | On GPU during image decode |
| **Peak** | **~12 GB** | (T5 + transformer simultaneously during first step) |

The `sd35` engine is configured with `vram_gb=12.0` in `image_lab_config.py`.

---

## 5. Wan2.2 — GGUF Migration

Wan2.2 uses two transformers per pipeline variant (HighNoise + LowNoise), and each transformer is a separate GGUF file from `QuantStack/Wan2.2-{T2V|I2V}-A14B-GGUF`.

```
wan-t2v:
  HighNoise → transformer    (primary denoiser for high-noise timesteps)
  LowNoise  → transformer_2  (refiner for low-noise timesteps)

wan-i2v: same structure
```

Loading pattern:
```python
t2v_tf  = WanTransformer3DModel.from_single_file("HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", ...)
t2v_tf2 = WanTransformer3DModel.from_single_file("LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf", ...)

pipe_t2v = WanPipeline.from_pretrained(
    "/opt/arthur-img-models/quantized/wan-t2v/shared",  # via preq_save.py
    transformer   = t2v_tf,
    transformer_2 = t2v_tf2,
    torch_dtype   = torch.bfloat16,
)
pipe_t2v.enable_model_cpu_offload()
pipe_t2v.vae.enable_slicing()
```

Both T2V and I2V pipelines are loaded simultaneously into `STATE.loaded_model` (T2V) and `STATE.loaded_pipe2` (I2V). If I2V fails to load (insufficient disk or VRAM), T2V continues to work.

**Wan VRAM at Q4_K_M:**

| Component | VRAM |
|---|---|
| T2V HighNoise transformer | ~5 GB |
| T2V LowNoise transformer | ~5 GB |
| Text encoder (T5) | ~2.5 GB (offloaded) |
| VAE | ~0.5 GB |
| **Peak** | **~14 GB** |

---

## 6. New Engine: FLUX.2 Klein 4B

### What It Is

FLUX.2 Klein 4B is a compact 4-billion parameter flow transformer from Black Forest Labs. It was released under Apache 2.0 (unlike FLUX.2 [dev] which is non-commercial).

Key differences from FLUX.2 [dev]:

| Property | FLUX.2 [dev] | FLUX.2 Klein 4B |
|---|---|---|
| Parameters | 32B | 4B |
| Text encoder | Mistral3 24B VLM | Qwen3 (compact) |
| Inference steps | 28 typical | 4 optimal (step-distilled) |
| VRAM needed | ~15 GB (group offload) | ~13 GB (CPU offload) |
| HuggingFace repo | `black-forest-labs/FLUX.2-dev` | `black-forest-labs/FLUX.2-klein-4B` |
| BnB NF4 checkpoint | `diffusers/FLUX.2-dev-bnb-4bit` | No pre-quantized version |
| GGUF available | Yes (city96) | No |
| License | FLUX [dev] Non-Commercial | Apache 2.0 |
| Speed | ~9 min (256×256, 4 steps, Q3_K_M) | ~seconds (256×256, 4 steps) |

### Why 4 Steps

FLUX.2 Klein 4B uses **step distillation** — it was trained to produce outputs matching FLUX.2 [dev] 28-step results in just 4 denoising steps. The distilled version's guidance schedule is calibrated for 4 steps; running more steps does not reliably improve quality and may degrade it.

The UI sets default `num_inference_steps=4` and `max=20` for this engine.

### Loading

```python
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype = torch.bfloat16,
    token       = HF_TOKEN,
)
pipe.enable_model_cpu_offload()   # Sequential offload (no BnB conflict)
pipe.vae.enable_slicing()
```

`Flux2KleinPipeline` is a new pipeline class in diffusers 0.38.0. It uses a different attention implementation than `Flux2Pipeline` and handles the Qwen3 text encoder natively.

No group offloading is needed because:
- The 4B transformer is small enough that `enable_model_cpu_offload()` manages it efficiently
- No BnB 4-bit weights are involved (loaded in bfloat16), so there is no `.to()` dequantization hazard

### Generator for FLUX.2 Klein

Unlike FLUX.2 [dev] (which requires `torch.Generator("cuda")`), FLUX.2 Klein uses CPU offload without group offloading, so latents may be on CPU at creation time:

```python
generator = torch.Generator("cpu").manual_seed(seed)
```

---

## 7. New Quantization Pathway: NVFP4 (Blackwell Native)

### What NVFP4 Is

NVFP4 (NVIDIA FP4) is a hardware-native 4-bit floating-point format supported by NVIDIA Blackwell GPUs (SM100+, i.e., RTX 5060 Ti and similar). It differs from GGUF quantization:

- **GGUF**: Integer quantization (k-quants) computed in software, compatible with all CUDA hardware
- **NVFP4**: True FP4 arithmetic accelerated by Blackwell tensor cores — theoretically the fastest on RTX 50xx

`torchao` provides `NVFP4WeightOnlyConfig` for this:

```python
from torchao.quantization import quantize_, NVFP4WeightOnlyConfig
quantize_(transformer, NVFP4WeightOnlyConfig())
```

### nvfp4_save.py

`nvfp4_save.py` quantizes transformers from their BF16 checkpoint and saves the result to disk at `/opt/arthur-img-models/nvfp4/`. This is a one-time pre-computation step.

**Jobs defined:**

| Model | Source repo | Output path |
|---|---|---|
| FLUX.2 [dev] | `black-forest-labs/FLUX.2-dev` | `nvfp4/flux2/transformer/` |
| SD 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | `nvfp4/sd35/transformer/` |
| Wan2.2 T2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `nvfp4/wan-t2v/transformer/` and `.../transformer_2/` |
| Wan2.2 I2V | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `nvfp4/wan-i2v/transformer/` and `.../transformer_2/` |

**Key design**: The BF16 weights are downloaded to a temporary directory, quantized in-place layer-by-layer (keeping peak RAM low), saved as safetensors, then the temporary BF16 cache is deleted.

### NVFP4 at Runtime

When `quant=nvfp4` is selected in the UI, `_load_nvfp4_transformer()` is called:

```python
def _load_nvfp4_transformer(model_key: str, subfolder: str):
    path = os.path.join("/opt/arthur-img-models/nvfp4", model_key, subfolder)
    return AutoModel.from_pretrained(
        path,
        torch_dtype     = torch.bfloat16,
        use_safetensors = False,
    )
```

The NVFP4 transformer is already on GPU (Blackwell hardware handles the FP4 arithmetic). No group offloading is applied to it — it is small enough (~8 GB for FLUX.2).

**Important**: `nvfp4_save.py` must be run before the first NVFP4 generation. If the saved directory does not exist, the UI shows an error:

```
RuntimeError: NVFP4 transformer not found at /opt/arthur-img-models/nvfp4/flux2/transformer.
Run nvfp4_save.py first to download and quantize it.
```

The UI shows this option in the Quantization dropdown as:
```
NVFP4 — ~8 GB transformer ⚡ Blackwell native (run nvfp4_save.py first)
```

---

## 8. Utility Scripts Added

### `gguf_download.py`

Pre-downloads all GGUF variants for all engines to the local disk cache before starting the service. Avoids first-generation delays.

**Usage:**
```bash
/opt/arthur-img-env/bin/python /opt/arthur-img/gguf_download.py
```

Skips already-downloaded files. Prints progress including file size and disk free space. Downloads 24 files total across all engines and all quantization levels.

**GGUF source repos:**
- `city96/FLUX.2-dev-gguf`
- `city96/stable-diffusion-3.5-large-gguf`
- `QuantStack/Wan2.2-T2V-A14B-GGUF`
- `QuantStack/Wan2.2-I2V-A14B-GGUF`

### `preq_save.py`

One-time setup: copies shared pipeline components (text encoders, VAE, tokenizers, scheduler, config files) from the HuggingFace hub cache to a flat directory structure at `/opt/arthur-img-models/quantized/`. This is necessary because GGUF loading via `from_single_file()` only loads the transformer — the rest of the pipeline must come from a `from_pretrained()` call, either a remote repo or a local directory.

**Why a local directory instead of the remote repo**: Loading from the HF remote repo would force downloading all weights (including the transformer) even though we only need the non-transformer components. The local `shared/` directory contains only what's needed.

**Usage:**
```bash
sudo systemctl stop arthur-imglab.service
nohup /opt/arthur-img-env/bin/python /opt/arthur-img/preq_save.py \
      > /var/log/preq_save.log 2>&1 &
tail -f /var/log/preq_save.log
sudo systemctl start arthur-imglab.service
```

**Output structure:**
```
/opt/arthur-img-models/quantized/
├── sd35/shared/
│   ├── text_encoder/      (CLIP-L)
│   ├── text_encoder_2/    (CLIP-G)
│   ├── text_encoder_3/    (T5-XXL, ~9 GB bfloat16)
│   ├── tokenizer/
│   ├── tokenizer_2/
│   ├── tokenizer_3/
│   ├── vae/
│   ├── scheduler/
│   └── model_index.json
├── wan-t2v/shared/
│   └── (similar structure)
└── wan-i2v/shared/
    └── (similar structure)
```

### `nvfp4_save.py`

Quantizes BF16 transformer weights to NVFP4 format and saves them to disk. One-time operation; must be run on the VM with `torchao` installed and the Blackwell GPU available.

**Usage:**
```bash
screen -S nvfp4
/opt/arthur-img-env/bin/python /opt/arthur-img/nvfp4_save.py
# Takes 30-90 minutes per transformer
```

### `create_grafana_dashboard.py`

Standalone script that creates/updates a Grafana dashboard named "Model Load Monitor" via the Grafana REST API. Uses the explicit datasource UID (`ffjmsi0wmmpdsf`) rather than `"__default__"` to avoid Grafana 13.0.1 resolution issues. Sets `refresh: 1` so template variables auto-populate on every dashboard load.

**Usage:**
```bash
python3 /opt/arthur-img/create_grafana_dashboard.py
```

---

## 9. VM Infrastructure: Swap Space

### Why Swap Was Needed

GGUF group offloading with FLUX.2 moves transformer layers from CPU RAM to GPU during the forward pass. At Q4_K_M, the FLUX.2 transformer is ~20 GB in CPU RAM (plus the BnB NF4 text encoder at ~14 GB). This requires:
- ~20 GB transformer in CPU RAM
- ~14 GB text encoder in CPU RAM  
- OS + Python overhead: ~4 GB
- **Total: ~38 GB RAM required**

If the VM has only 32 GB physical RAM, the kernel will OOM-kill the process. Swap prevents this by allowing the least-recently-used pages to be swapped to disk.

### Swap Configuration

```bash
# Added 8 GB swap file (on root disk /dev/sda1)
sudo fallocate -l 8G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab

# Result:
# /swapfile  (existing) — 2 GB
# /swapfile2 (new)      — 8 GB
# Total swap: 10 GB
```

After adding swap, check with:
```bash
swapon --show     # Shows all swap devices
free -h           # Shows total RAM + swap
```

**Note**: Using swap for ML workloads means disk I/O during generation if CPU RAM is genuinely exhausted. The RTX 5060 Ti VM should have at minimum 48 GB RAM to avoid swap for FLUX.2 Q4_K_M. Swap is a safety net, not a performance solution.

---

## 10. UI Evolution

### 10.1 Engine Tab List — Added FLUX.2 Klein

The original `buildEngineTabs()` function had the engine list hardcoded to 3 engines. The list was updated to include `flux2klein`:

```javascript
// Before:
const keys = ['flux2', 'sd35', 'wan'];
const labels = { flux2: 'FLUX.2', sd35: 'SD 3.5', wan: 'Wan2.2' };

// After:
const keys = ['flux2', 'flux2klein', 'sd35', 'wan'];
const labels = { flux2: 'FLUX.2', flux2klein: 'FLUX.2 Klein', sd35: 'SD 3.5', wan: 'Wan2.2' };
```

The dynamic `renderParams()` function already handled any engine — it reads the param schema from the `/status` API. Only the tab list needed to be explicitly defined.

### 10.2 Sidebar Overflow — CSS Fix

**Problem:** When the log panel was expanded (max-height: 220px), it pushed the parameter form out of the visible sidebar area. The params were cut off and inaccessible even though the sidebar had a scrollbar.

**Root cause:** The `.sidebar` element had `overflow-y: auto`. This created an outer scrollbar on the entire sidebar, so scrolling moved the whole sidebar (tabs + params + log). The log panel was `flex-shrink: 0` and `max-height: 220px`, meaning it took its space unconditionally and pushed params down below the viewport. Since the params area also had `overflow-y: auto` but no `min-height: 0`, flex layout gave it zero height when the log panel took all the space.

**Fix:**

```css
/* Before */
.sidebar { background: var(--panel); border-right: 1px solid var(--border);
           display: flex; flex-direction: column; overflow-y: auto; }

/* After */
.sidebar { background: var(--panel); border-right: 1px solid var(--border);
           display: flex; flex-direction: column; overflow: hidden; }
```

```css
/* Before */
.params-area { flex: 1; overflow-y: auto; padding: 16px; }

/* After */
.params-area { flex: 1; overflow-y: auto; padding: 16px; min-height: 0; }
```

**Why `min-height: 0`:** In CSS flexbox, flex children default to `min-height: auto` (the intrinsic minimum height of their content). This prevents them from shrinking below their content height. Setting `min-height: 0` allows the params area to shrink to zero height and then rely on `overflow-y: auto` to make the content within it scroll. Without `min-height: 0`, a flex child with `overflow-y: auto` and `flex: 1` will not actually scroll — it will instead expand to fit all its content.

**Combined effect:** The sidebar is now `overflow: hidden` (no outer sidebar scrollbar), and the params area has `min-height: 0` (can shrink to zero and scroll internally). The log panel is `flex-shrink: 0` so it takes its space, and the params area adjusts accordingly.

### 10.3 Availability Badges — Color Coding

**Problem:** Engine availability badges showed `...` initially and updated to either `✓ available` or `✗ unavailable` after the first `/status` poll (~2 seconds). However, both states used the same gray color (`var(--muted)`), providing no visual distinction.

**Fix in `updateStatus()`:**

```javascript
// Before:
badge.textContent = e.available ? '✓ available' : '✗ unavailable';
// (no color change)

// After:
badge.textContent = e.available ? '✓ available' : '✗ unavailable';
badge.style.color  = e.available ? 'var(--ok)' : 'var(--err)';
```

CSS variables used:
- `--ok: #34d399` (green) — available engine
- `--err: #f87171` (red) — unavailable engine

### 10.4 Log Panel

The log panel was confirmed working. It shows structured entries in the format:
```
[HH:MM:SS] [LEVEL] message
```

Log levels and colors:
- `INFO` — blue (`#7cb8ff`)
- `WARN` — amber (`var(--warn)`)
- `ERROR` — red (`var(--err)`)
- `DBG` — gray (`#6b7280`)

The `uiLog()` JavaScript function both writes to the DOM panel and mirrors to the browser console. Server-side events (generation start/complete, model load, VRAM stats) are reflected in the log via the `/status` poll responses.

### 10.5 Current UI Layout (After All Fixes)

```
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Sidebar (300 px)             │  Main pane                                   │
│                              │                                               │
│ ┌────────────────────────┐   │  Status: ● Ready | VRAM 1.2 / 15.5 GB       │
│ │ [FLUX.2][Klein][SD3.5] │   │  [Generate] [Gallery]                        │
│ │ [Wan2.2]               │   │  ───────────────────────────────────────     │
│ │  ✓ available (green)   │   │  ┌────────────────────────────────────────┐ │
│ └────────────────────────┘   │  │  Output pane (scrollable)              │ │
│                              │  │  [Result card with image/video]        │ │
│ ┌────────────────────────┐   │  │  prompt | seed | steps | ↓ download    │ │
│ │  Params area           │   │  └────────────────────────────────────────┘ │
│ │  (scrollable, flex:1,  │   │                                               │
│ │   min-height:0)        │   │                                               │
│ │  Prompt [textarea]     │   │                                               │
│ │  Ref image [drop]      │   │                                               │
│ │  Width ──────── 1024   │   │                                               │
│ │  Height ─────── 1024   │   │                                               │
│ │  Steps ──────── 28     │   │                                               │
│ │  Guidance ───── 3.5    │   │                                               │
│ │  Seed ────────── -1    │   │                                               │
│ │  Quantization [select] │   │                                               │
│ └────────────────────────┘   │                                               │
│  ▸ curl API snippet          │                                               │
│  ⚡ Generate                 │                                               │
│  Engine description text     │                                               │
│  ┌────────────────────────┐  │                                               │
│  │ ▸ Log [Clear]          │  │                                               │
│  │ (max-height: 220px,    │  │                                               │
│  │  flex-shrink: 0)       │  │                                               │
│  └────────────────────────┘  │                                               │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 11. Engine Testing Results

All engine tests run via the web UI at `http://192.168.0.87:8002`.

### FLUX.2 [dev] — Q3_K_M

| Parameter | Value |
|---|---|
| Resolution | 256 × 256 |
| Steps | 4 |
| Quantization | Q3_K_M (~16 GB transformer) |
| Load time | ~58 seconds |
| Generation time | ~9 minutes |
| VRAM peak | ~1.5 GB (group offloading active) |
| Result | ✅ Image saved (`flux2_43bf8ef0...png`) |

**Note:** Q3_K_M was selected for testing because the Q4_K_M transformer (~20 GB) was not yet cached. Q3_K_M is the smallest available GGUF that fits in CPU RAM alongside the BnB NF4 text encoder.

**False positive during testing:** A test script checked `response["images"]` but the API returns `response["results"]`. The image was confirmed via `journalctl` logs showing `Saved image flux2_43bf8ef0...`. The test script was not updated (low priority).

### FLUX.2 Klein 4B

| Parameter | Value |
|---|---|
| Resolution | 256 × 256 |
| Steps | 4 (distilled model default) |
| Quantization | BF16 (no GGUF available) |
| Load time | ~3.1 seconds (cached in HF hub) |
| Generation time | Seconds (fast distilled) |
| VRAM peak | ~13 GB |
| Result | ✅ Image generated and displayed |

### Stable Diffusion 3.5 Large

| Parameter | Value |
|---|---|
| Resolution | 256 × 256 |
| Steps | 4 |
| Quantization | Q4_0 (~4.8 GB transformer) |
| Load time | ~35.5 seconds |
| Generation time | ~1.5 minutes |
| VRAM peak | ~12 GB |
| Result | ✅ Image generated and displayed |

### Wan2.2 (Video)

Not tested in this session. Wan2.2 requires large GGUF downloads (HighNoise + LowNoise × Q4_K_M = ~4 × 9.7 GB = ~38.8 GB total for T2V alone). The engine shows as `✓ available` in the UI (probe succeeded) but a full test generation was deferred.

---

## 12. Current Disk Layout

### Updated Storage Structure

```
/opt/arthur-img/                      ~1 MB   (Python source code)
├── image_lab.py
├── image_lab_config.py
├── image_lab_dispatch.py
├── image_lab_engines.py
├── image_lab_ui.py
├── image_lab_utils.py
├── nvfp4_save.py
├── preq_save.py
├── gguf_download.py
├── create_grafana_dashboard.py
└── .env                              (HF_TOKEN, paths — chmod 600)

/opt/arthur-img-models/
├── huggingface/hub/                  ~50 GB  (HF cache — text encoders, VAEs, configs)
│   ├── models--diffusers--FLUX.2-dev-bnb-4bit/   (text encoder NF4 + VAE)
│   └── models--black-forest-labs--FLUX.2-klein-4B/
├── gguf/                             ~100+ GB (varies by downloaded variants)
│   ├── flux2/                        (Q3_K_M: 16GB, Q4_K_M: 20GB, etc.)
│   ├── sd35/                         (Q4_0: 4.8GB, Q5_0: 5.8GB, Q8_0: 8.8GB)
│   ├── wan-t2v/                      (16 GGUF files across HighNoise/LowNoise variants)
│   └── wan-i2v/                      (16 GGUF files)
├── quantized/                        ~35 GB  (preq_save.py output — shared components)
│   ├── sd35/shared/                  (text encoders + VAE for SD3.5, ~20 GB)
│   ├── wan-t2v/shared/               (text encoder + VAE for Wan T2V, ~8 GB)
│   └── wan-i2v/shared/               (~8 GB)
└── nvfp4/                            (nvfp4_save.py output — empty until script is run)
    ├── flux2/transformer/
    ├── sd35/transformer/
    ├── wan-t2v/transformer/
    ├── wan-t2v/transformer_2/
    ├── wan-i2v/transformer/
    └── wan-i2v/transformer_2/

/opt/arthur-gen/
├── images/                           (Generated PNG files)
├── videos/                           (Generated MP4 files)
└── gallery.json

/swapfile                             2 GB  (original swap)
/swapfile2                            8 GB  (added this session)
```

---

## 13. Current Engine Catalogue

### Summary Table

| Engine | Key | Architecture | Quantization | VRAM | Load Time | HF Repo |
|---|---|---|---|---|---|---|
| FLUX.2 [dev] | `flux2` | 32B DiT | GGUF (Q3_K_M–Q8_0) + NVFP4 | 15–16 GB (group offload) | ~58 s (Q3_K_M) | `diffusers/FLUX.2-dev-bnb-4bit` (text enc), `city96/FLUX.2-dev-gguf` (transformer) |
| FLUX.2 Klein 4B | `flux2klein` | 4B flow transformer | BF16 (no GGUF) | ~13 GB | ~3 s (cached) | `black-forest-labs/FLUX.2-klein-4B` |
| SD 3.5 Large | `sd35` | 8B MMDiT | GGUF (Q4_0–Q8_0) + NVFP4 | ~12 GB | ~35 s (Q4_0) | `stabilityai/stable-diffusion-3.5-large` (shared), `city96/stable-diffusion-3.5-large-gguf` (transformer) |
| Wan2.2 | `wan` | 14B video DiT | GGUF (Q3_K_M–Q8_0) + NVFP4 | ~14 GB | ~2–5 min | `QuantStack/Wan2.2-{T2V|I2V}-A14B-GGUF` |

### API Keys

- `POST /generate/flux2` — Text-to-image or image editing (with reference_image)
- `POST /generate/flux2klein` — Text-to-image or image-to-image
- `POST /generate/sd35` — Text-to-image (up to 4 images per request)
- `POST /generate/wan` — Text-to-video or image-to-video (mode=t2v|i2v)

### Quantization Selection in UI

The Quantization dropdown is shown per-engine. When the user changes quantization while the same engine is already loaded, a yellow warning banner appears:
```
⚠️ Quantization change — model will reload (~60 s)
```

---

## 14. Deployment Workflow (Current)

### Quick Code Deploy (Most Common)

```powershell
# From Windows dev machine in C:\repos\TTS-LAB\:
.\deploy_image_lab.ps1 -Phase 5   # SCP all Python files
.\deploy_image_lab.ps1 -Phase 6   # Restart service
```

### Manual Deploy (One-liners)

```powershell
# SCP individual files:
scp -i "$env:USERPROFILE\.ssh\id_arthur_vm" image_lab_ui.py arthur@192.168.0.87:/opt/arthur-img/

# Restart service:
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 "sudo systemctl restart arthur-imglab.service"

# Verify new UI is live (grep for a known CSS string):
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 "curl -s http://localhost:8002/ | grep -c 'min-height: 0'"

# Check HTTP status:
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 "curl -s -o /dev/null -w '%{http_code}' http://localhost:8002/"
```

### Service Monitoring

```bash
# Live logs
sudo journalctl -u arthur-imglab.service -f

# Last N lines
sudo journalctl -u arthur-imglab.service -n 50 --no-pager

# Check for OOM kill
sudo journalctl -u arthur-imglab.service | grep -E "kill|OOM|oom"

# Check VRAM
nvidia-smi
```

### Troubleshooting Service Won't Start

```bash
# Check if previous process is holding GPU
sudo lsof /dev/nvidia*

# Force kill old process
sudo kill -9 <PID>

# Check swap usage
free -h
swapon --show
```

---

## 15. Complete File Inventory

Files in `C:\repos\TTS-LAB\` relevant to the image lab:

### Core Service Files

| File | Purpose | Deploy to VM |
|---|---|---|
| `image_lab.py` | FastAPI entry point | Yes (`/opt/arthur-img/`) |
| `image_lab_config.py` | Engine catalogue, paths, LabState | Yes |
| `image_lab_dispatch.py` | HTTP route handlers | Yes |
| `image_lab_engines.py` | Load/unload/generate per engine | Yes |
| `image_lab_ui.py` | Inline HTML/CSS/JS web UI | Yes |
| `image_lab_utils.py` | VRAM, I/O, gallery helpers | Yes |

### Utility Scripts (Run on VM)

| File | Purpose | When to Run |
|---|---|---|
| `preq_save.py` | Pre-save shared pipeline components (SD35, Wan) | Once after first model download |
| `nvfp4_save.py` | Quantize transformers to NVFP4 for Blackwell | Once before using nvfp4 option |
| `gguf_download.py` | Pre-download all GGUF variants | Once (or add new variants later) |
| `create_grafana_dashboard.py` | Create/update Model Load Monitor dashboard | After Grafana install or dashboard changes |

### Deployment Automation

| File | Purpose |
|---|---|
| `deploy_image_lab.ps1` | 8-phase idempotent deploy script (Windows → VM via SSH) |
| `secrets.env` | HF_TOKEN and other secrets (NOT committed) |
| `secrets.env.example` | Template for secrets.env (committed, no real values) |

### Reference Documents

| File | Contents |
|---|---|
| `ARTHUR_IMAGE_LAB_REFERENCE.md` | Initial comprehensive reference (original architecture with BnB NF4) |
| `SESSION_2026-05-25_IMAGE_LAB_EVOLUTION.md` | This document — post-reference evolution |

---

## Appendix: Key Code Snippets

### A. Full FLUX.2 Load Sequence (Current)

```python
def _load_flux2(quant: str = "Q4_K_M"):
    import torch
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    from diffusers.hooks import apply_group_offloading
    from transformers import AutoModel

    quant = quant or "Q4_K_M"

    if quant == "nvfp4":
        transformer = _load_nvfp4_transformer("flux2", "transformer")
        _apply_group_offload = False
    else:
        gguf_path = _ensure_gguf(*_FLUX2_GGUF[quant], os.path.join(GGUF_ROOT, "flux2"))
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config = _gguf_quant_config(),
            torch_dtype         = torch.bfloat16,
        )
        _apply_group_offload = True

    if _apply_group_offload:
        apply_group_offloading(transformer,
            onload_device="cuda", offload_device="cpu",
            offload_type="leaf_level", use_stream=False)

    text_encoder = AutoModel.from_pretrained(
        ENGINES["flux2"].hf_repo,   # diffusers/FLUX.2-dev-bnb-4bit
        subfolder="text_encoder", device_map="cpu",
        dtype=torch.bfloat16, token=HF_TOKEN or True)

    try:
        apply_group_offloading(text_encoder,
            onload_device="cuda", offload_device="cpu",
            offload_type="leaf_level", use_stream=False)
    except ValueError as exc:
        if "AlignDevicesHook" in str(exc) or "CpuOffload" in str(exc):
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(text_encoder, recurse=True)
            apply_group_offloading(text_encoder,
                onload_device="cuda", offload_device="cpu",
                offload_type="leaf_level", use_stream=False)
        else:
            raise

    pipe = Flux2Pipeline.from_pretrained(
        ENGINES["flux2"].hf_repo,
        transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16, token=HF_TOKEN or True)

    pipe.vae = pipe.vae.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    STATE.loaded_model  = pipe
    STATE.active_engine = "flux2"
    STATE.active_quant  = quant
    ENGINES["flux2"].loaded = True
```

### B. CSS Layout Fix (Sidebar + Params)

```css
/* overflow: hidden prevents outer sidebar scrollbar */
.sidebar {
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;          /* was: overflow-y: auto */
}

/* min-height: 0 allows flex child to shrink below content height,
   enabling overflow-y: auto to actually create an internal scrollbar */
.params-area {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  min-height: 0;             /* added */
}

/* Log panel stays at bottom, takes fixed space, does not shrink */
.log-panel {
  border-top: 1px solid var(--border);
  background: #10121a;
  flex-shrink: 0;            /* unchanged — this is correct */
}

.log-body {
  display: none;
  max-height: 220px;         /* unchanged — constrains log height */
  overflow-y: auto;
}
```

### C. Badge Color Update in `updateStatus()`

```javascript
async function refreshStatus() {
  const s = await apiFetch('/status');
  // ... other status updates ...

  for (const e of s.engines) {
    const el = document.getElementById('tab-' + e.key);
    if (el) {
      const badge = el.querySelector('.badge');
      badge.textContent = e.available ? '✓ available' : '✗ unavailable';
      badge.style.color  = e.available ? 'var(--ok)' : 'var(--err)';
      // --ok: #34d399 (green)  |  --err: #f87171 (red)
    }
  }
}
```

### D. Deployment Verification Commands

```powershell
# 1. Confirm service is running
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 `
    "sudo systemctl is-active arthur-imglab.service"
# Expected: "active"

# 2. Confirm new UI is being served (count occurrences of known new string)
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 `
    "curl -s http://localhost:8002/ | grep -c 'min-height: 0'"
# Expected: "1" (or more)

# 3. Full status JSON
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 `
    "curl -s http://localhost:8002/status | python3 -m json.tool"

# 4. Check recent logs
ssh -i "$env:USERPROFILE\.ssh\id_arthur_vm" arthur@192.168.0.87 `
    "sudo journalctl -u arthur-imglab.service --no-pager -n 20"
```
