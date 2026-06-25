# Arthur Image & Video Generation Lab — Complete Engineering Reference

> **Version:** May 2026  
> **Service:** `arthur-imglab.service` — FastAPI on port **8002**  
> **Host VM:** Ubuntu 22.04 — `192.168.0.87`  
> **GPU:** NVIDIA RTX 5060 Ti 16 GB GDDR7 (driver 580.159.03, CUDA 12.8)  

---

## Table of Contents

1. [What Is This Lab?](#1-what-is-this-lab)
2. [Architecture Overview](#2-architecture-overview)
3. [Source Files Reference](#3-source-files-reference)
4. [AI Engines — Supported Models](#4-ai-engines--supported-models)
5. [Infrastructure & Environment](#5-infrastructure--environment)
6. [Deployment Guide (Step-by-Step)](#6-deployment-guide-step-by-step)
7. [API Reference](#7-api-reference)
8. [Web UI Guide](#8-web-ui-guide)
9. [Monitoring — Grafana / Prometheus / nvidia_gpu_exporter](#9-monitoring--grafana--prometheus--nvidia_gpu_exporter)
10. [Problems Encountered & Solutions](#10-problems-encountered--solutions)
11. [VRAM & Memory Management Deep-Dive](#11-vram--memory-management-deep-dive)
12. [OOM (Out-of-Memory) Error — Root Cause & Fix](#12-oom-out-of-memory-error--root-cause--fix)
13. [HuggingFace Model Storage](#13-huggingface-model-storage)
14. [Disk Layout & Storage Planning](#14-disk-layout--storage-planning)
15. [Systemd Service Configuration](#15-systemd-service-configuration)
16. [Security Notes](#16-security-notes)
17. [Maintenance & Day-to-Day Operations](#17-maintenance--day-to-day-operations)
18. [Troubleshooting Runbook](#18-troubleshooting-runbook)
19. [Grafana Dashboard — Model Load Monitor](#19-grafana-dashboard--model-load-monitor)
20. [Known Limitations & Future Work](#20-known-limitations--future-work)
21. [Glossary](#21-glossary)

---

## 1. What Is This Lab?

The **Arthur Image & Video Generation Lab** is a self-hosted AI generation service that runs on a local Ubuntu virtual machine with GPU passthrough. It provides:

- **Text-to-Image** generation using state-of-the-art diffusion models
- **Image-to-Image editing** (provide a reference image, describe changes)
- **Text-to-Video** generation (cinematic motion from a text description)
- **Image-to-Video animation** (make a still image come alive)

The entire system is a **single Python FastAPI process** (`image_lab.py`) that listens on port 8002. It serves both a browser-based Web UI and a JSON REST API. Models are loaded into GPU VRAM on demand and swapped as needed.

### Non-Engineer Summary

Think of it as a private version of services like Midjourney or Runway — running on your own hardware, with no usage limits, no cloud costs per generation, and full privacy. You open a web browser, type a description of what you want, and within 30–120 seconds the GPU renders the image or video and displays it on the page.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Windows Dev Machine (192.168.x.x)                              │
│  VS Code + scripts/deploy/deploy_image_lab.ps1                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSH / SCP (id_arthur_vm key)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ubuntu 22.04 VM  (192.168.0.87)                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  arthur-imglab.service  (systemd)                      │     │
│  │  /opt/arthur-img-env/bin/python /opt/arthur-img/       │     │
│  │                                                         │     │
│  │  image_lab.py         ← FastAPI entry point            │     │
│  │  image_lab_config.py  ← Engine catalogue + state       │     │
│  │  image_lab_engines.py ← Load / Unload / Generate       │     │
│  │  image_lab_dispatch.py← HTTP route handlers            │     │
│  │  image_lab_ui.py      ← Inline HTML/CSS/JS UI          │     │
│  │  image_lab_utils.py   ← VRAM, I/O, gallery helpers     │     │
│  └────────────────────────────────────────────────────────┘     │
│              │ Port 8002                                         │
│              ▼                                                   │
│  ┌─────────────────────────┐   ┌──────────────────────────┐    │
│  │  GPU: RTX 5060 Ti 16 GB │   │  Model cache (sda1)      │    │
│  │  CUDA 12.8 / PyTorch    │   │  /opt/arthur-img-models/ │    │
│  │  diffusers 0.38.0       │   │  ~32 GB FLUX.2           │    │
│  │  BitsAndBytes 4-bit     │   │  ~40 GB SD 3.5 Large     │    │
│  │  accelerate 1.13.0      │   │  ~49 GB Wan2.2 T2V       │    │
│  └─────────────────────────┘   │  ~50 GB Wan2.2 I2V       │    │
│                                 └──────────────────────────┘    │
│                                                                  │
│  Monitoring Stack (port 3000 / 9090 / 9835 / 9100)             │
│  Grafana ← Prometheus ← nvidia_gpu_exporter + node-exporter    │
└─────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

1. Browser sends `POST /generate/flux2` (multipart form, optional image upload)
2. `image_lab_dispatch.py` validates the engine key, reads form fields
3. `engines.generate("flux2", params)` is called — this is synchronous (blocks)
4. `_ensure_engine("flux2")` evicts any loaded model, loads FLUX.2 into VRAM
5. `_generate_flux2(params)` runs the diffusion pipeline; PyTorch uses CUDA
6. The output image/video is written to `/opt/arthur-gen/images/` or `.../videos/`
7. A JSON entry is appended to `gallery.json`
8. The file path + metadata is returned to the browser as JSON
9. The browser renders the image/video card in the output pane

---

## 3. Source Files Reference

All source files live on the **Windows dev machine** at `C:\repos\TTS-LAB\` and are deployed to the VM at `/opt/arthur-img/` via SCP.

### `image_lab.py` — Entry Point

| Responsibility | Details |
|---|---|
| Load `.env` file | Reads `/opt/arthur-img/.env` before any imports |
| Set HF cache env vars | `HF_HOME`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE` must be set **before** importing diffusers/transformers |
| Configure logging | `%(asctime)s [%(levelname)s] %(name)s — %(message)s` to stdout (systemd captures it) |
| Start FastAPI app | Mounts `image_lab_dispatch.router`, serves UI at `GET /` |
| Run uvicorn | Binds to `0.0.0.0:8002` |

**Critical detail:** `_load_dotenv()` and the `os.environ.setdefault()` calls for HF paths happen at the **top of the file**, before any `import diffusers` or `import transformers`. If these are set after the library imports, the libraries have already resolved their cache directories and the setting has no effect.

---

### `image_lab_config.py` — Engine Catalogue & Global State

This is the **single source of truth** for what models exist, what parameters they accept, and what the service-wide state is.

**Key objects:**

```python
@dataclass
class EngineInfo:
    key: str          # "flux2" | "sd35" | "wan"
    label: str        # Human-readable display name
    description: str  # Shown in the UI sidebar
    output_type: str  # "image" | "video"
    vram_gb: float    # VRAM estimate when loaded (for display only)
    hf_repo: str      # Primary HuggingFace repo ID
    hf_repo_alt: str  # Secondary repo (Wan I2V variant)
    params: list      # Parameter schema (drives the UI dynamically)
    available: bool   # Set True at startup after import checks
    loaded: bool      # Set True when model is in VRAM
    error: str        # Last error message if unavailable
```

```python
@dataclass
class LabState:
    active_engine: str   # Which engine is currently in VRAM
    loaded_model: Any    # The loaded pipeline object
    loaded_pipe2: Any    # Second pipeline (Wan T2V + I2V pair)
    loading: bool        # True during model load
    generating: bool     # True during inference
```

**Path constants:**

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/opt/arthur-img-models/huggingface` | HuggingFace model cache |
| `MODELS_ROOT` | `/opt/models/image` | Legacy path (unused now) |
| `OUTPUT_ROOT` | `/opt/arthur-gen` | Generated images/videos output |
| `IMAGES_DIR` | `$OUTPUT_ROOT/images` | PNG outputs |
| `VIDEOS_DIR` | `$OUTPUT_ROOT/videos` | MP4 outputs |
| `GALLERY_DB` | `$OUTPUT_ROOT/gallery.json` | Gallery index file |

---

### `image_lab_engines.py` — Model Loading & Inference

The largest and most complex file. Contains one `_load_*` function and one `_generate_*` function per engine, plus shared VRAM lifecycle helpers.

**VRAM lifecycle:**

```
_ensure_engine(key)
    ├── if active_engine == key: return (already loaded)
    ├── _unload_current()
    │       ├── STATE.loaded_model = None
    │       ├── STATE.loaded_pipe2 = None
    │       ├── STATE.active_engine = None
    │       └── free_vram()  (gc.collect + torch.cuda.empty_cache)
    └── _LOADERS[key]()  → _load_flux2() / _load_sd35() / _load_wan()
```

**Public entry point:**

```python
def generate(engine_key: str, params: dict) -> list[dict]:
    _ensure_engine(engine_key)
    STATE.generating = True
    try:
        return _GENERATORS[engine_key](params)
    finally:
        STATE.generating = False
```

---

### `image_lab_dispatch.py` — HTTP Routes

| Route | Method | Purpose |
|---|---|---|
| `/status` | GET | JSON: engines list, VRAM stats, active engine, loading/generating flags |
| `/generate/{engine_key}` | POST | Multipart form submission; triggers generation |
| `/outputs/{filename}` | GET | Serve generated image/video file |
| `/gallery` | GET | Return gallery JSON array |
| `/gallery/{entry_id}` | DELETE | Remove a gallery entry + file |
| `/` | GET | Returns the full Web UI (HTML) |

---

### `image_lab_ui.py` — Browser Interface

A single Python string constant `UI_HTML` containing the entire frontend — HTML, CSS, and JavaScript — returned by `GET /`. No build step, no npm, no separate static files.

**UI capabilities:**
- Engine selector tabs (FLUX.2 / SD 3.5 / Wan2.2)
- Dynamic parameter form (generated from `engine.params` schema via the `/status` API)
- VRAM usage bar in the header (live, polled every 3 s)
- Status dot (green=idle, amber=loading or generating)
- Output gallery (images displayed inline, videos with playback controls)
- Reference image drag-and-drop upload for I2I / I2V modes
- Download button for each result
- Dark theme with accent colours (`--accent: #6c8ef7`, `--accent2: #a78bfa`)

---

### `image_lab_utils.py` — Shared Helpers

| Function | Purpose |
|---|---|
| `ensure_dirs()` | Create output directories and empty gallery.json if missing |
| `vram_stats()` | Returns `{available, allocated_gb, reserved_gb, total_gb, free_gb, device_name}` |
| `free_vram()` | `gc.collect()` + `torch.cuda.empty_cache()` + `ipc_collect()` |
| `save_image(pil_img, engine, params)` | Save PIL image as PNG, append to gallery |
| `save_images(images, engine, params)` | Batch version of save_image |
| `save_video(frames, fps, engine, params)` | Save frames as MP4 via imageio-ffmpeg |
| `read_gallery()` | Read and return gallery.json as a list |
| `delete_gallery_entry(entry_id)` | Remove entry from gallery + delete file |
| `random_seed()` | Cryptographically random seed (0–2³¹-1) |

---

### `scripts/deploy/deploy_image_lab.ps1` — Deployment Automation

A PowerShell script that runs from the **Windows dev machine**. Connects to the VM over SSH using `~/.ssh/id_arthur_vm`. All 8 phases are idempotent (safe to re-run).

| Phase | Name | Duration |
|---|---|---|
| 1 | System packages + directory layout | ~2 min |
| 2 | Python 3.11 venv + PyTorch CUDA 12.8 | ~10 min |
| 3 | Engine Python packages | ~5 min |
| 4 | Model pre-download (large, optional) | 30–90 min |
| 5 | SCP code files to VM | ~5 s |
| 6 | Write systemd service + restart | ~15 s |
| 7 | HuggingFace CLI token cache | ~5 s |
| 8 | Health check (`/status` endpoint) | ~5 s |

**Common deployment commands:**

```powershell
# First-time full deploy
.\scripts/deploy/deploy_image_lab.ps1

# Re-deploy code only (fastest iteration cycle)
.\scripts/deploy/deploy_image_lab.ps1 -Phase 5; .\scripts/deploy/deploy_image_lab.ps1 -Phase 6

# Skip model download on a machine with existing cache
.\scripts/deploy/deploy_image_lab.ps1 -SkipPhases "4"

# Override target VM
.\scripts/deploy/deploy_image_lab.ps1 -VM 192.168.0.99

# Provide HF token explicitly
.\scripts/deploy/deploy_image_lab.ps1 -HFToken hf_xxxxxxxxxxxxxxxx
```

---

### `create_grafana_dashboard.py` — Monitoring Dashboard Creator

A standalone Python script that POSTs a pre-built dashboard JSON to Grafana's REST API (`/api/dashboards/db`). Idempotent — uses `"overwrite": true`. Run on the VM to create/update the "Model Load Monitor" dashboard.

---

## 4. AI Engines — Supported Models

### 4.1 FLUX.2 [dev] — `flux2`

| Property | Value |
|---|---|
| **HuggingFace repo** | `diffusers/FLUX.2-dev-bnb-4bit` |
| **Architecture** | 32B rectified flow transformer (DiT) |
| **Text encoder** | Mistral3ForConditionalGeneration (VLM, multimodal) |
| **Quantization** | Both transformer and text encoder are **pre-quantized NF4 BnB 4-bit** in the checkpoint |
| **Disk size** | ~32 GB (transformer: 17 GB on disk as BnB uint8, text encoder: 14.5 GB) |
| **VRAM when loaded** | ~10 GB (4-bit transformer ~6 GB + 4-bit Mistral3 ~4 GB + VAE ~0.3 GB) |
| **Output type** | Image (PNG) |
| **Supports I2I** | Yes — pass `reference_image` for image editing mode |
| **License** | FLUX [dev] Non-Commercial License |
| **Requires HF token** | Yes (gated model) |

**Default parameters:**

| Parameter | Default | Range |
|---|---|---|
| `width` / `height` | 1024 × 1024 | 256–2048 (step 64) |
| `num_inference_steps` | 28 | 1–50 |
| `guidance_scale` | 4.0 | 1.0–20.0 |
| `seed` | -1 (random) | -1 to 2³¹-1 |

**Loading strategy (critical — see also section 12):**

```python
pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    torch_dtype = torch.bfloat16,
    device_map  = "balanced",   # accelerate places 4-bit components on CUDA directly
    token       = HF_TOKEN,
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.enable_attention_slicing(1)
```

**Why `device_map="balanced"` is mandatory:** The model weights are stored as BnB uint8 (4-bit packed). Calling `.to("cuda:0")` on a BnB model attempts to **dequantize** the weights back to bfloat16 before moving — that requires 17 GB for the transformer alone, causing OOM. `device_map="balanced"` uses accelerate to place each component on CUDA natively without dequantization.

---

### 4.2 Stable Diffusion 3.5 Large — `sd35`

| Property | Value |
|---|---|
| **HuggingFace repo** | `stabilityai/stable-diffusion-3.5-large` |
| **Architecture** | 8B MMDiT (Multimodal Diffusion Transformer) |
| **Text encoders** | CLIP-L, CLIP-G, T5-XXL |
| **Quantization** | None — runs in bfloat16 |
| **Disk size** | ~40 GB |
| **VRAM when loaded** | ~12 GB (uses `enable_model_cpu_offload()`) |
| **Output type** | Image (PNG), up to 4 per request |
| **Supports I2I** | No (text-to-image only in current implementation) |
| **License** | Stability AI Community License |
| **Requires HF token** | Yes (gated model) |

**Loading strategy:**

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype = torch.bfloat16,
    token       = HF_TOKEN,
)
pipe.enable_model_cpu_offload()   # T5-XXL lives in CPU RAM; moves to GPU only during encoding
```

The T5-XXL text encoder is ~9 GB in bfloat16. Without `enable_model_cpu_offload()`, it would OOM a 16 GB card alongside the 8B transformer. With offloading, the T5 runs on GPU during text encoding, then transfers back to CPU; the transformer and VAE then run on GPU.

---

### 4.3 Wan2.2 — `wan`

| Property | Value |
|---|---|
| **T2V HuggingFace repo** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| **I2V HuggingFace repo** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| **Architecture** | 14B causal video diffusion model (Alibaba) |
| **Disk size** | ~49 GB (T2V) + ~50 GB (I2V) = ~99 GB total |
| **VRAM when loaded** | ~14 GB (both pipelines with CPU offload + VAE slicing) |
| **Output type** | Video (MP4) |
| **Modes** | `t2v` (text-to-video) and `i2v` (image-to-video) |
| **License** | Apache 2.0 |
| **Requires HF token** | No |

**Loading strategy:**

```python
pipe_t2v = WanPipeline.from_pretrained(t2v_repo, torch_dtype=torch.bfloat16)
pipe_t2v.enable_model_cpu_offload()
pipe_t2v.vae.enable_slicing()

pipe_i2v = WanImageToVideoPipeline.from_pretrained(i2v_repo, torch_dtype=torch.bfloat16)
pipe_i2v.enable_model_cpu_offload()
pipe_i2v.vae.enable_slicing()
```

Both pipelines are kept in RAM simultaneously (`STATE.loaded_model` = T2V, `STATE.loaded_pipe2` = I2V), allowing mode switching without reloading. If I2V fails to load (e.g., insufficient disk), T2V continues to work.

**Default parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `mode` | `t2v` | `t2v` or `i2v` |
| `num_frames` | 49 | At 16 fps ≈ 3 seconds |
| `fps` | 16 | Output video frame rate |
| `resolution` | `720p` | `720p` (1280×720) or `480p` (854×480) |
| `guidance_scale` | 5.0 | |

---

### Engine Comparison Summary

| Feature | FLUX.2 [dev] | SD 3.5 Large | Wan2.2 |
|---|---|---|---|
| Output | Image | Image | Video |
| Model size | 32B params | 8B params | 14B params |
| VRAM needed | ~10 GB | ~12 GB | ~14 GB |
| Speed (1024px) | ~60 s | ~30 s | ~3 min (49 frames) |
| Image quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Prompt following | Excellent | Very good | Good |
| Reference image | ✓ I2I editing | ✗ | ✓ I2V animation |
| License | Non-commercial | Community | Apache 2.0 |

---

## 5. Infrastructure & Environment

### VM Specifications

| Component | Value |
|---|---|
| **Hypervisor** | Proxmox VE (GPU passthrough / DDA) |
| **OS** | Ubuntu 22.04 LTS |
| **IP** | 192.168.0.87 |
| **SSH user** | `arthur` |
| **SSH key** | `~/.ssh/id_arthur_vm` (on dev machine) |
| **Sudo** | Passwordless |
| **GPU** | NVIDIA RTX 5060 Ti 16 GB GDDR7 |
| **Driver** | 580.159.03 |
| **CUDA** | 12.8 |
| **RAM** | ≥64 GB (CPU RAM used for model offload) |

### Disk Layout

| Device | Mount | Size | Contents |
|---|---|---|---|
| `/dev/sda1` | `/` (root) | 650 GB | OS + `/opt/arthur-img-models/` (image model cache) + `/opt/arthur-img/` (code) |
| `/dev/sdb1` | `/opt/models` | 180 GB | TTS models (100% full — unrelated to image lab) |

> **Important:** The image lab models are stored on the root disk (`sda1`) at `/opt/arthur-img-models/`, NOT on `/opt/models` (`sdb1`). The `/opt/models` disk is full with TTS service data.

### Python Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 | System-installed, venv at `/opt/arthur-img-env/` |
| PyTorch | 2.11.0+cu128 | CUDA 12.8 build |
| diffusers | 0.38.0 | Includes Flux2Pipeline, WanPipeline |
| transformers | latest | Includes Mistral3ForConditionalGeneration |
| accelerate | 1.13.0 | Required for `device_map="balanced"` |
| bitsandbytes | latest | BnB NF4 4-bit quantization |
| FastAPI | latest | Web framework |
| uvicorn | latest (standard) | ASGI server with websocket support |

### Environment Variables (`.env` file at `/opt/arthur-img/.env`)

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx    # Required for FLUX.2 and SD 3.5 (gated)
HF_HOME=/opt/arthur-img-models/huggingface
IMGLAB_MODELS_ROOT=/opt/models/image   # Legacy, not actively used
IMGLAB_OUTPUT_ROOT=/opt/arthur-gen
IMGLAB_PORT=8002
```

The `.env` file has `chmod 600` permissions. It is **not** committed to source control. The `secrets.env` file on the dev machine contains the tokens for use by the deployment script.

---

## 6. Deployment Guide (Step-by-Step)

### Prerequisites (Dev Machine — Windows)

1. OpenSSH client installed (comes with Windows 10+)
2. SSH key pair generated:
   ```powershell
   ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_arthur_vm"
   ```
3. Public key copied to VM:
   ```powershell
   type "$env:USERPROFILE\.ssh\id_arthur_vm.pub" | ssh arthur@192.168.0.87 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ```
4. `secrets.env` file in `C:\repos\TTS-LAB\` containing:
   ```
   HF_TOKEN=hf_your_token_here
   ```

### Phase 1 — System Packages & Directories

Installs: `ffmpeg`, `libglib2.0-0`, `libsm6`, `libxext6`, `libgl1`, `python3.11`, `python3.11-venv`, `git-lfs`

Creates: `/opt/arthur-img`, `/opt/models/image`, `/opt/arthur-gen/images`, `/opt/arthur-gen/videos`

### Phase 2 — Python venv + PyTorch

Creates a Python 3.11 virtual environment at `/opt/arthur-img-env/`. Installs PyTorch with CUDA 12.8 support from the PyTorch wheel index.

**Verification:** The phase prints `torch.__version__` and `torch.cuda.is_available()` — both must be correct before proceeding.

### Phase 3 — Python Packages

Installs the full ML inference stack:
- `diffusers transformers accelerate safetensors sentencepiece protobuf`
- `bitsandbytes` (for FLUX.2 NF4 4-bit)
- `fastapi uvicorn[standard] python-multipart`
- `Pillow imageio imageio-ffmpeg opencv-python-headless`
- `huggingface_hub requests`

### Phase 4 — Model Download

Downloads all four models (~170 GB total) to `/opt/arthur-img-models/huggingface/hub/`:

| Model | Download Size | Destination |
|---|---|---|
| `diffusers/FLUX.2-dev-bnb-4bit` | ~32 GB | `models--diffusers--FLUX.2-dev-bnb-4bit` |
| `stabilityai/stable-diffusion-3.5-large` | ~40 GB | `models--stabilityai--stable-diffusion-3.5-large` |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ~49 GB | `models--Wan-AI--Wan2.2-T2V-A14B-Diffusers` |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ~50 GB | `models--Wan-AI--Wan2.2-I2V-A14B-Diffusers` |

Uses `snapshot_download()` with `ignore_patterns=['*.msgpack','*.h5','flax_model*']` to skip non-PyTorch weights. All files in the HF hub format are symlinks pointing to content-addressed blobs in a `blobs/` directory.

> **Note:** This phase uses SCP to transfer the download script to `/tmp/imglab_download.py` first, then executes it via SSH. This was necessary because multi-line heredocs in PowerShell SSH commands caused quoting failures.

### Phase 5 — SCP Code Files

Copies the 6 Python source files to `/opt/arthur-img/` and writes the `.env` file with the HF token and path configuration.

### Phase 6 — Systemd Service

Writes `/etc/systemd/system/arthur-imglab.service`, enables it, and restarts it. The service runs as `root` (required for some GPU operations and file creation in `/opt/arthur-gen`).

### Phase 7 — HF Token Cache

Runs `huggingface-cli login` on the VM to cache the token in `~/.cache/huggingface/token`. This allows `from_pretrained()` to find the token even if the `.env` variable is not set.

### Phase 8 — Health Check

Polls `http://192.168.0.87:8002/status` and prints:
- Available engines (✓ / ✗)
- VRAM stats
- Web UI and API URLs

---

## 7. API Reference

### `GET /status`

Returns the current state of all engines and hardware.

**Response (JSON):**

```json
{
  "engines": [
    {
      "key": "flux2",
      "label": "FLUX.2 [dev]",
      "description": "...",
      "output_type": "image",
      "vram_gb": 10.0,
      "available": true,
      "loaded": false,
      "error": "",
      "params": [ ... ]
    }
  ],
  "active_engine": null,
  "generating": false,
  "loading": false,
  "vram": {
    "available": true,
    "allocated_gb": 0.0,
    "reserved_gb": 0.0,
    "total_gb": 15.48,
    "free_gb": 15.48,
    "device_name": "NVIDIA GeForce RTX 5060 Ti"
  }
}
```

---

### `POST /generate/{engine_key}`

Triggers image or video generation. Accepts multipart/form-data.

**URL parameters:** `engine_key` = `flux2` | `sd35` | `wan`

**Common form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text description of desired output |
| `negative_prompt` | string | `""` | What NOT to include (SD35, Wan) |
| `width` | int | 1024 | Output width in pixels |
| `height` | int | 1024 | Output height in pixels |
| `num_inference_steps` | int | 28 | Denoising steps |
| `guidance_scale` | float | 4.0 | Prompt adherence strength |
| `seed` | int | -1 | -1 for random, fixed value for reproducibility |
| `reference_image` | file | null | Optional image upload (FLUX.2 I2I, Wan I2V) |

**Wan-specific fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `t2v` | `t2v` (text-to-video) or `i2v` (image-to-video) |
| `num_frames` | int | 49 | Number of video frames (49 ≈ 3 s at 16 fps) |
| `fps` | int | 16 | Output video frame rate |
| `resolution` | string | `720p` | `720p` (1280×720) or `480p` (854×480) |

**Response (success, 200):**

```json
[
  {
    "id": "uuid-string",
    "engine": "flux2",
    "output_type": "image",
    "filename": "flux2_20260524_2314_abc123.png",
    "url": "/outputs/flux2_20260524_2314_abc123.png",
    "params": { "prompt": "...", "seed": 1234567 },
    "created_at": 1779663000.0
  }
]
```

**Response (error, 503):**

```json
{ "detail": "CUDA out of memory. Tried to allocate 15.01 GiB..." }
```

---

### `GET /outputs/{filename}`

Returns the raw PNG image or MP4 video file.

### `GET /gallery`

Returns the full gallery as a JSON array of result objects (same schema as generate response items).

### `DELETE /gallery/{entry_id}`

Removes a gallery entry and deletes the associated file from disk. The `entry_id` is the `id` field from the gallery entry (UUID string).

---

## 8. Web UI Guide

Access the UI at **`http://192.168.0.87:8002`** from any browser on the local network.

### Layout

```
┌─────────────────────┬──────────────────────────────────────────────┐
│  Sidebar (300 px)   │  Main pane                                   │
│                     │                                               │
│  [FLUX.2][SD35][Wan]│  ← Engine tabs                               │
│                     │  Status bar: ● Idle | FLUX.2 loaded | 8.2 GB│
│  ┌───────────────┐  │  ┌──────────────────────────────────────┐   │
│  │  Parameter    │  │  │  Output Gallery                      │   │
│  │  Form         │  │  │                                       │   │
│  │               │  │  │  [Image Card 1]                       │   │
│  │  Prompt:      │  │  │  prompt | seed | steps | ↓ download  │   │
│  │  [textarea]   │  │  │                                       │   │
│  │               │  │  │  [Image Card 2]  ...                 │   │
│  │  Width: [1024]│  │  │                                       │   │
│  │  Steps: [28]  │  │  └──────────────────────────────────────┘   │
│  │               │  │                                               │
│  └───────────────┘  │                                               │
│  [ Generate ▶ ]     │                                               │
│  ─────────────────  │                                               │
│  Engine description │                                               │
└─────────────────────┴──────────────────────────────────────────────┘
```

### VRAM Bar

The top-right of the header shows a live VRAM usage bar that updates every 3 seconds. It displays reserved VRAM / total VRAM. When a model is loading or generating, the bar climbs to 10–14 GB.

### Status Indicator

- 🟢 **Green dot** — Service idle, no model loaded or model loaded and ready
- 🟡 **Amber pulsing dot** — Model currently loading or generation in progress
- 🔴 **Red dot** — Last generation failed

### Reference Image Upload

For FLUX.2 and Wan I2V: click the dashed file drop zone to upload a reference image. The file is sent as multipart form data with the generation request. In FLUX.2 mode, it enables image editing. In Wan I2V mode, it becomes the first frame to animate.

---

## 9. Monitoring — Grafana / Prometheus / nvidia_gpu_exporter

### Stack Components

| Service | Port | Binary / Package |
|---|---|---|
| Grafana | 3000 | `grafana` v13.0.1 (APT package) |
| Prometheus | 9090 | `prometheus` (APT package) |
| nvidia_gpu_exporter | 9835 | `/usr/bin/nvidia_gpu_exporter` |
| prometheus-node-exporter | 9100 | `prometheus-node-exporter` (APT package) |

### Prometheus Configuration (`/etc/prometheus/prometheus.yml`)

```yaml
scrape_configs:
  - job_name: 'nvidia_gpu'
    static_configs:
      - targets: ['localhost:9835']
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### GPU Metrics (nvidia_smi_* prefix)

The `nvidia_gpu_exporter` exposes all nvidia-smi metrics:

| Metric | Description |
|---|---|
| `nvidia_smi_memory_used_bytes` | VRAM currently used |
| `nvidia_smi_memory_free_bytes` | VRAM currently free |
| `nvidia_smi_memory_total_bytes` | Total VRAM (constant: 15.48 GB) |
| `nvidia_smi_utilization_gpu_ratio` | GPU core utilization (0.0–1.0) |
| `nvidia_smi_utilization_memory_ratio` | Memory controller utilization |
| `nvidia_smi_temperature_gpu` | GPU temperature in °C |
| `nvidia_smi_power_draw_instant_watts` | Instantaneous power draw |
| `nvidia_smi_power_draw_watts` | Average power draw |
| `nvidia_smi_clocks_current_graphics_clock_hz` | Current GPU clock |
| `nvidia_smi_index` | GPU index (used as label source for `uuid` and `instance`) |

### Node Exporter Metrics Used

| Metric | Description |
|---|---|
| `node_disk_read_bytes_total{device="sda"}` | Total bytes read from root disk |
| `node_disk_written_bytes_total{device="sda"}` | Total bytes written to root disk |
| `node_cpu_seconds_total{mode="idle"}` | CPU idle time (used to derive utilization) |
| `node_memory_MemTotal_bytes` | Total system RAM |
| `node_memory_MemAvailable_bytes` | Available system RAM |

### Grafana Admin Access

- **URL:** `http://192.168.0.87:3000`
- **Admin password:** `newpass2026` (reset during session — change this!)
- **Datasource UID:** `ffjmsi0wmmpdsf` (Prometheus, `http://localhost:9090`, default datasource)

---

## 10. Problems Encountered & Solutions

This section documents every significant complication during development and deployment, in chronological order.

---

### Problem 1: `step_=8` Keyword Argument Typo

**File:** `image_lab_config.py`  
**Symptom:** Service crashed immediately on startup with a `TypeError`.  
**Root cause:** A parameter definition used `step_=8` instead of `step=8` in the `_p()` helper call. The trailing underscore `_` was silently passed as an unknown keyword.  
**Fix:** Changed `step_=8` to `step=8` in the Wan parameter schema.  
**Lesson:** Python does not warn about unknown kwargs in functions that accept `**kwargs`; test by actually starting the service.

---

### Problem 2: NVML Driver/Kernel Version Mismatch (Error 804)

**Symptom:** `nvidia_gpu_exporter.service` logs showed:
```
Failed to initialize NVML: Driver/library version mismatch NVML library version: 580.159
```
PyTorch could not see the GPU (`torch.cuda.is_available()` returned `False`).

**Root cause:** The VM kernel or NVIDIA driver modules were updated while the system was running. The running kernel module and the userspace NVML library were from different driver versions.

**Fix:** Full VM reboot (`sudo reboot`). After reboot, both the kernel module and userspace library loaded the same version (580.159.03).

**Lesson:** NVML mismatches always require a reboot; they cannot be fixed by restarting just the affected service.

---

### Problem 3: PowerShell Variable `$VMHost` Undefined

**File:** `scripts/deploy/deploy_image_lab.ps1` Phase 4  
**Symptom:** SSH commands in Phase 4 failed silently or used an empty string for the VM host.  
**Root cause:** The script used `$VMHost` in the SSH helper function but the parameter was named `$VM`. PowerShell's strict mode raised an error.  
**Fix:** Changed all references to `$VMHost` → `$VM`.  
**Lesson:** Always use `Set-StrictMode -Version Latest` at the top of deployment scripts — it catches undefined variables.

---

### Problem 4: PowerShell Heredoc Misinterpreted by SSH

**Symptom:** When Phase 4 tried to write a Python download script via SSH heredoc (`<< PYEOF`), the shell in PowerShell interpreted the heredoc markers itself and sent garbled data to SSH.  
**Root cause:** PowerShell does not support POSIX heredoc syntax. The `<<` operator is interpreted by PowerShell, not passed to the remote shell.  
**Fix:** Write the Python script to a local temp file (`$env:TEMP\imglab_download.py`), SCP it to `/tmp/`, then execute it via SSH.  
**Lesson:** Never use heredocs in SSH commands from PowerShell. Always SCP scripts first.

---

### Problem 5: Wrong Wan Model Repository IDs

**Symptom:** Phase 4 `snapshot_download()` raised a `RepositoryNotFoundError` for the Wan models.  
**Root cause:** The initial repo IDs were `Wan-AI/Wan2.2-T2V-14B` and `Wan-AI/Wan2.2-I2V-14B`. The actual correct IDs on HuggingFace include the architecture suffix `-A14B-Diffusers`.  
**Fix:** Updated to `Wan-AI/Wan2.2-T2V-A14B-Diffusers` and `Wan-AI/Wan2.2-I2V-A14B-Diffusers` in both the deploy script and `image_lab_config.py`.  
**Lesson:** Always verify HuggingFace repo IDs against the actual hub page before embedding them in code.

---

### Problem 6: Permission Denied on `/opt/models/image/`

**Symptom:** Phase 4 download script failed with `PermissionError: [Errno 13] Permission denied: '/opt/models/image'`.  
**Root cause:** The directory `/opt/models/` was created by root and owned by root. The download script ran as the `arthur` user but couldn't write into it.  
**Fix:** Added `sudo chown -R arthur:arthur /opt/models/` before the download step in Phase 4.  
**Lesson:** When creating directories with `sudo mkdir`, always also set ownership with `sudo chown` for the user who will write to them.

---

### Problem 7: `/opt/models` Disk 100% Full

**Symptom:** FLUX.2 download completed but immediately wrote corrupt/incomplete files. `df -h` showed `/dev/sdb1` at 100% (177/177 GB used — entirely by TTS service models).  
**Root cause:** The `/opt/models` mount point (`/dev/sdb1`, 180 GB) was already fully occupied by TTS audio models from a separate service. The image lab was attempting to store model files there.  
**Fix:**
1. Moved FLUX.2 (32 GB) and SD 3.5 (40 GB) downloads from `/opt/models/image/` to `/opt/arthur-img-models/` on the root disk (`/dev/sda1`, 650 GB, 503 GB free)
2. Updated `HF_HOME` in all relevant locations:
   - `image_lab.py` default: `/opt/arthur-img-models/huggingface`
   - `image_lab_config.py` default: `/opt/arthur-img-models/huggingface`
   - `/opt/arthur-img/.env`: `HF_HOME=/opt/arthur-img-models/huggingface`
   - Phase 4 download script env var: same path
3. Re-downloaded Wan models (~99 GB) directly to the correct path

**Lesson:** Plan disk capacity before downloading large models. Separate disks for separate services prevents this situation.

---

### Problem 8: Grafana Admin Password Unknown

**Symptom:** API calls to `http://localhost:3000/api/datasources` returned `401 Unauthorized`. Default password `admin` was rejected. Previous reset attempts with `sqlite3` failed (not installed).  
**Root cause:** The Grafana admin password had been changed from the default at some earlier point. Since no password manager was used, it was lost.  
**Fix:** Used the Grafana CLI to reset it:
```bash
sudo grafana cli admin reset-admin-password newpass2026
```
The Grafana CLI binary is at `/usr/sbin/grafana` and accepts the `cli admin reset-admin-password` subcommand even when the Grafana server is running.  
**Note:** Grafana 13.0.1 uses "unified storage" — the `dashboard` table in `grafana.db` is empty because dashboards are stored via the new Kubernetes-style resource API.

---

### Problem 9: Grafana GPU Dashboard "No Data"

**Symptom:** The "Nvidia GPU Metrics" dashboard (uid `vlvPlrgnk`) showed no data in all panels.  
**Root cause (multi-factor):**
1. Before the VM reboot, `nvidia_gpu_exporter` was failing with the NVML mismatch (Problem 2). No GPU metrics existed in Prometheus for that period.
2. After the reboot, metrics were flowing but the dashboard's `$gpu`, `$job`, `$node` template variables had `"current": {}` (no saved value) and `refresh: 2` (only refreshes on time-range change, not on initial load).
3. The dashboard's time range included the pre-reboot "no data" period.

**Fix:**
1. VM reboot fixed the metrics pipeline
2. User was instructed to set time range to "Last 30 minutes" (post-reboot only) and let the variables auto-populate
3. Created a new "Model Load Monitor" dashboard (uid `model-load-monitor`) with `refresh: 1` (on load) variables that auto-select on every page open

---

### Problem 10: FLUX.2 Generation OOM — The Full Story

See **Section 12** for complete technical details. Summary: The original implementation was fundamentally broken in two ways — using the wrong text encoder architecture (T5 instead of Mistral3) and calling `.to("cuda:0")` which dequantized 4-bit weights causing OOM.

---

### Problem 11: `create_grafana_dashboard.py` Used `__default__` UID

**Symptom:** Model Load Monitor dashboard panels showed "No data" even though Prometheus had data.  
**Root cause:** The dashboard JSON specified `"uid": "__default__"` for the Prometheus datasource reference. While this is a Grafana special keyword, it did not resolve correctly in Grafana 13.0.1.  
**Fix:** Updated all datasource references in the dashboard to use the explicit UID `ffjmsi0wmmpdsf` (obtained from `GET /api/datasources`).

---

### Problem 12: Variable Refresh in New Dashboard Didn't Populate on Load

**Symptom:** After opening the Model Load Monitor dashboard, the `$gpu` dropdown was empty and all GPU panels showed "No data".  
**Root cause:** Template variables with `refresh: 2` only re-query their values when the time range changes. On first load, they stay empty.  
**Fix:** Changed to `refresh: 1` which triggers the variable query on every dashboard load.

---

## 11. VRAM & Memory Management Deep-Dive

### Why VRAM Management Is Critical

The RTX 5060 Ti has 15.48 GB of VRAM. The three models have these approximate VRAM requirements:
- FLUX.2: ~10 GB
- SD 3.5: ~12 GB  
- Wan2.2: ~14 GB

None of these can coexist in VRAM simultaneously. The lab uses a **single-model-at-a-time** strategy: before loading a new model, the previous one is fully evicted.

### The Eviction Cycle

```python
def _unload_current():
    STATE.loaded_model = None   # Drop Python reference
    STATE.loaded_pipe2 = None
    STATE.active_engine = None
    gc.collect()                # Python garbage collector
    torch.cuda.empty_cache()    # Release PyTorch's CUDA memory pool
    torch.cuda.ipc_collect()    # Clean up inter-process CUDA handles
```

After eviction, `nvidia-smi` should show ~0.5 GB used (CUDA driver overhead only).

### BitsAndBytes (BnB) 4-bit Quantization

BnB NF4 (NormalFloat4) quantization stores model weights in 4-bit format:
- Each weight value is mapped to one of 16 possible values in the NF4 codebook
- Groups of 64 weights share a quantization scale (bfloat16 per block)
- Storage format: `uint8` tensors (two 4-bit values packed per byte)

**Memory savings:**
- bfloat16: 2 bytes/weight
- NF4: 0.5 bytes/weight (stored) + ~0.03 bytes/weight (scale metadata)
- Net reduction: ~75% memory vs bfloat16

**Critical constraint:** BnB 4-bit models **must** be loaded to CUDA. The quantization/dequantization kernels are GPU-only. Loading to CPU first and then calling `.to("cuda")` attempts to convert `uint8` → `bfloat16` first, which **expands** memory usage dramatically before the transfer.

### The `device_map="balanced"` Strategy for FLUX.2

accelerate's `device_map="balanced"` works as follows:
1. Calls `infer_auto_device_map()` to compute the memory footprint of each submodule
2. Assigns each submodule to a device (GPU or CPU) fitting within the VRAM budget
3. For BnB-quantized submodules, it honours the quantized size (not the bfloat16 size)
4. Inserts `dispatch_model` hooks so tensors automatically move between devices during the forward pass

For our case (both transformer and text encoder pre-quantized):
- FLUX.2 transformer (4-bit): ~6 GB on GPU
- Mistral3 text encoder (4-bit): ~4 GB on GPU  
- VAE (bfloat16): ~0.3 GB on GPU
- Total: ~10.3 GB → comfortably within 15.48 GB

### The `enable_model_cpu_offload()` Strategy (SD 3.5, Wan)

For models where individual components exceed VRAM:
1. All model submodules start in CPU RAM
2. accelerate hooks `forward()` to move each top-level component to GPU when needed
3. After the component runs, it moves back to CPU
4. Only one major component (text encoder OR transformer OR VAE) is on GPU at a time

For SD 3.5: T5-XXL (~9 GB bfloat16) runs on GPU during text encoding, then moves to CPU. The MMDiT transformer and VAE then run on GPU.

### Memory Optimization Techniques Applied

| Technique | Benefit | Applied To |
|---|---|---|
| `pipe.vae.enable_slicing()` | VAE decodes in slices, reducing peak VRAM by 2–3 GB | FLUX.2, SD 3.5 |
| `pipe.vae.enable_tiling()` | VAE processes large images in tiles, flat memory cost | FLUX.2, SD 3.5 |
| `pipe.enable_attention_slicing(1)` | Attention computed one head at a time, reduces peak VRAM | FLUX.2, SD 3.5 |
| `pipe.vae.enable_slicing()` | Same as above | Wan2.2 |
| `enable_model_cpu_offload()` | Sequential CPU↔GPU movement of major components | SD 3.5, Wan2.2 |
| `device_map="balanced"` | accelerate-managed placement (BnB-safe) | FLUX.2 |

---

## 12. OOM (Out-of-Memory) Error — Root Cause & Fix

This section gives a complete technical account of the CUDA OOM error that blocked FLUX.2 generation.

### The Error Message

```
Generation failed: CUDA out of memory. Tried to allocate 15.01 GiB.
GPU 0 has a total capacity of 15.48 GiB of which 14.98 GiB is free.
Process 939 has 364.00 MiB memory in use.
Including non-PyTorch memory, this process has 128.00 MiB memory in use.
Of the allocated memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved
by PyTorch but unallocated.
```

### Interpreting the Error

Key facts from the error:
- **14.98 GiB free** — The GPU is nearly empty. No model is currently loaded in VRAM.
- **0 bytes allocated by PyTorch** — PyTorch hasn't moved any model weights to GPU yet.
- **364 MiB in use (non-PyTorch)** — This is CUDA driver/runtime overhead only.
- **Tried to allocate 15.01 GiB** — A single contiguous allocation of ~15 GB was attempted.

This error happened **during model loading**, not during inference. Specifically, it happened when calling `.to("cuda:0")` on the pipeline.

### Bug #1: Wrong Text Encoder Architecture

The original `_load_flux2` code:

```python
pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=None,    # WRONG assumption: "T5 is remote"
    torch_dtype=torch_dtype,
).to(device)              # WRONG: dequantizes BnB weights
```

And during generation:
```python
def _remote_text_encoder(prompts):
    resp = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        ...
    )
    embeds = torch.load(BytesIO(resp.content), weights_only=True)
    return embeds.to(device)   # T5 embeddings — WRONG format for FLUX.2!
```

**FLUX.1** uses T5-XXL as the text encoder. The old code used a HuggingFace remote endpoint that served T5 embeddings, and this worked for FLUX.1.

**FLUX.2** uses `Mistral3ForConditionalGeneration` — a multimodal VLM, not T5. The `Flux2Pipeline` processes prompts through Mistral3's hidden states at layers 10, 20, and 30. T5 embeddings have a completely different shape and meaning. Passing T5 embeddings to FLUX.2 would have produced garbage images (or a runtime error on shape mismatch).

### Bug #2: `.to("cuda:0")` Dequantizes BnB 4-bit Weights

The model `diffusers/FLUX.2-dev-bnb-4bit` has:
- Transformer (DiT): Pre-quantized NF4 BnB 4-bit in the checkpoint
- Text encoder (Mistral3): Pre-quantized NF4 BnB 4-bit in the checkpoint

Both have `quantization_config` in their respective `config.json` files:
```json
{
  "quantization_config": {
    "_load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    ...
  }
}
```

When `from_pretrained()` is called **without** `device_map`, the default behavior is:
1. Load model weights to CPU RAM
2. The BnB quantization config is recognized
3. Weights are stored as `uint8` tensors (4-bit packed)

When `.to("cuda:0")` is then called on this BnB model:
- PyTorch's `.to()` method does **not** understand BnB quantization
- It sees `uint8` tensors and tries to convert them to the pipeline's `torch_dtype` (bfloat16)
- `uint8` → `bfloat16` dequantization of the text encoder (14.5 GB disk, ~14.5 GB in bfloat16) requires **15 GiB** of temporary GPU memory
- The GPU only has 15.48 GB total → OOM

The 15.01 GiB allocation attempt corresponds precisely to the Mistral3 text encoder being dequantized to bfloat16 in a single call.

### Why `text_encoder=None` Didn't Help

With the original code (`text_encoder=None`), the Mistral3 encoder was not loaded at all. The transformer (4-bit, ~5-6 GB GPU) and VAE were loaded. Then `.to("cuda:0")` was called on the whole pipeline. Since `text_encoder=None`, only the transformer was subject to the `.to()` call. The transformer's BnB dequantization (17 GB on disk) at ~8-9 GB bfloat16 should have been marginal. But even this caused OOM because:
- The transformer has 2 safetensors shards totalling 17 GB on disk
- In BnB uint8 format in CPU RAM, it's ~8-9 GB
- When `.to("cuda:0")` triggers dequantization to bfloat16, it temporarily needs ~17 GB GPU

### The Fix

```python
pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    torch_dtype = torch.bfloat16,
    device_map  = "balanced",   # ← KEY CHANGE
    token       = HF_TOKEN,
)
```

With `device_map="balanced"`:
1. `accelerate` calls `infer_auto_device_map()` to inspect each submodule
2. It detects the BnB quantization and uses the **4-bit size** for memory planning
3. Components are placed **directly on CUDA** without going through CPU intermediately
4. `.to()` is **never called** — accelerate's dispatch mechanism handles device placement
5. Peak VRAM: ~6 GB (transformer) + ~4 GB (text encoder) + ~0.3 GB (VAE) = **~10.3 GB**

And the generate function now uses standard prompt passing:

```python
result = pipe(
    prompt    = params["prompt"],   # ← Mistral3 handles this natively
    image     = reference_image,    # ← Optional I2I reference
    ...
)
```

### Model Size Accounting

```
FLUX.2-dev-bnb-4bit components (disk vs VRAM):

Component          Disk Size   VRAM (4-bit)   VRAM (bf16)
─────────────────────────────────────────────────────────
Transformer (DiT)  17 GB       ~6 GB          ~34 GB
Text enc (Mistral3)14.5 GB     ~4 GB          ~14.5 GB
VAE                321 MB      321 MB (bf16)  321 MB
Tokenizer/config   ~5 MB       N/A            N/A
─────────────────────────────────────────────────────────
Total              ~32 GB      ~10.3 GB       ~49 GB
```

*Disk size for BnB 4-bit is larger than expected (not 25% of bf16) because BnB stores uint8 packed values (50% of bf16 size) PLUS quantization scales (one bf16 per 64 weights adds ~3%) PLUS quantization offsets and lookup tables.*

---

## 13. HuggingFace Model Storage

### Cache Structure

HuggingFace Hub uses a content-addressed cache at `$HF_HOME/hub/`:

```
/opt/arthur-img-models/huggingface/hub/
├── models--diffusers--FLUX.2-dev-bnb-4bit/
│   ├── blobs/          ← Actual weight files (content-addressed)
│   │   ├── ef849d9660...  (9.4 GB — transformer shard 1)
│   │   ├── 43bf95cffd...  (7.6 GB — transformer shard 2)
│   │   ├── 4875062f3f...  (4.6 GB — text encoder shard 1)
│   │   └── ...
│   ├── refs/main       ← Current commit hash
│   └── snapshots/
│       └── c30ad107.../  ← Symlinks to blobs
│           ├── transformer/
│           │   ├── config.json → ../../../blobs/...
│           │   ├── diffusion_pytorch_model-00001-of-00002.safetensors → ../../../blobs/...
│           │   └── diffusion_pytorch_model-00002-of-00002.safetensors → ../../../blobs/...
│           ├── text_encoder/
│           │   ├── config.json → ...
│           │   └── model-000{01..04}-of-00004.safetensors → ...
│           ├── vae/
│           ├── tokenizer/
│           ├── scheduler/
│           └── model_index.json
├── models--stabilityai--stable-diffusion-3.5-large/
│   └── (similar structure, ~40 GB)
├── models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/
│   └── (similar structure, ~49 GB)
└── models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/
    └── (similar structure, ~50 GB)
```

### Why HF_HOME Must Be Set Before Imports

`diffusers` and `transformers` resolve `HF_HOME` at import time. If you set `os.environ["HF_HOME"]` after `import diffusers`, the library has already cached the path (typically `~/.cache/huggingface`). The `image_lab.py` entry point sets all HF env vars before any library import using `os.environ.setdefault()`.

### `from_pretrained()` Cache Lookup

When `Flux2Pipeline.from_pretrained("diffusers/FLUX.2-dev-bnb-4bit")` is called:
1. Checks `$HF_HOME/hub/models--diffusers--FLUX.2-dev-bnb-4bit/refs/main` for cached commit hash
2. If found, resolves the snapshot directory without hitting the network
3. Loads weights from the blob files via the symlinks in the snapshot

This means model loading is **offline-capable** once downloaded. The HF_TOKEN is only needed for gated models and the initial download.

---

## 14. Disk Layout & Storage Planning

### Current Usage

| Path | Contents | Size |
|---|---|---|
| `/opt/arthur-img/` | Python source code | ~1 MB |
| `/opt/arthur-img/.env` | Secrets + paths | <1 KB |
| `/opt/arthur-img-models/` | Image model cache | ~172 GB |
| `/opt/arthur-gen/` | Generated outputs | Growing |
| `/opt/models/` | TTS models (separate service) | 177 GB (full) |

### Capacity Planning

| Model | Disk | VRAM | RAM (offload) |
|---|---|---|---|
| FLUX.2-dev-bnb-4bit | 32 GB | 10 GB | Minimal (BnB on GPU) |
| SD 3.5 Large | 40 GB | 12 GB | ~9 GB (T5 offload) |
| Wan2.2 T2V + I2V | 99 GB | 14 GB | ~15 GB (offload) |
| **Total models** | **~171 GB** | — | — |

Recommended root disk size: **≥500 GB** for model cache + OS + outputs. The current 650 GB setup has ~330 GB free after models (enough for months of output accumulation).

### Output Storage

Generated files are not automatically cleaned up. Each 1024×1024 PNG is ~2–5 MB. Each Wan video (49 frames, 720p MP4) is ~15–40 MB. At high usage, implement a cron job to purge old files.

---

## 15. Systemd Service Configuration

### Service File (`/etc/systemd/system/arthur-imglab.service`)

```ini
[Unit]
Description=Arthur Image & Video Generation Lab
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur-img
ExecStart=/opt/arthur-img-env/bin/python /opt/arthur-img/image_lab.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-/opt/arthur-img/.env

[Install]
WantedBy=multi-user.target
```

### Key Design Choices

- **`User=root`**: Required because generating files in `/opt/arthur-gen/`, CUDA driver initialization in some configurations, and potential file permission issues with the HF cache. For hardened production use, consider creating a dedicated `imglab` user with appropriate group memberships (`video`, `render`).
- **`EnvironmentFile=-/opt/arthur-img/.env`**: The leading `-` makes this non-fatal if the file doesn't exist.
- **`IMGLAB_GPU_ONLY=1`**: Set in `/opt/arthur-img/.env` to force GPU-only execution and disable CPU model offloading.
- **`PYTHONUNBUFFERED=1`**: Ensures Python's stdout is not buffered, so log lines appear in `journalctl` in real-time.
- **`Restart=on-failure`**: Automatically restarts the service if it crashes, but not if it exits cleanly. `RestartSec=5` prevents rapid restart loops.

### Service Management Commands

```bash
# View live logs (follow mode)
sudo journalctl -u arthur-imglab.service -f

# View last 100 lines
sudo journalctl -u arthur-imglab.service -n 100 --no-pager

# Check status
sudo systemctl status arthur-imglab.service

# Restart after code update
sudo systemctl restart arthur-imglab.service

# Disable auto-start
sudo systemctl disable arthur-imglab.service

# Check if enabled
sudo systemctl is-enabled arthur-imglab.service
```

---

## 16. Security Notes

### What Is Exposed

The service binds to `0.0.0.0:8002` — accessible from any machine on the local network. There is **no authentication** on the API or Web UI. Anyone on the local network (or VPN) can:
- Generate unlimited images and videos
- View the gallery of all previously generated content
- Delete gallery entries

### Secrets Management

- `HF_TOKEN` is stored in `/opt/arthur-img/.env` with `chmod 600` (root-readable only)
- The deploy script reads the token from `secrets.env` on the dev machine, which is in the repo directory — ensure this file is in `.gitignore` and never committed
- SSH private key `id_arthur_vm` should never leave the dev machine

### Recommendations for Production Hardening

1. Add HTTP Basic Auth or OAuth to the FastAPI app (or put nginx in front)
2. Run the service as a non-root user
3. Use a secrets manager instead of `.env` files
4. Restrict SSH access to specific source IPs (`AllowUsers` in `/etc/ssh/sshd_config`)
5. Enable ufw or iptables to restrict port 8002 to the local subnet

---

## 17. Maintenance & Day-to-Day Operations

### Updating Source Code

```powershell
# Edit files locally on dev machine, then:
.\scripts/deploy/deploy_image_lab.ps1 -Phase 5   # SCP code
.\scripts/deploy/deploy_image_lab.ps1 -Phase 6   # Restart service
```

### Checking Service Health

```powershell
# From Windows dev machine:
curl http://192.168.0.87:8002/status | python -m json.tool
```

```bash
# From VM:
sudo systemctl status arthur-imglab.service
curl -s http://localhost:8002/status | python3 -m json.tool
```

### Viewing GPU Status

```bash
watch -n 1 nvidia-smi
```

### Clearing Generated Output

```bash
# WARNING: this deletes all generated images and videos
sudo rm -f /opt/arthur-gen/images/* /opt/arthur-gen/videos/*
sudo sh -c 'echo "[]" > /opt/arthur-gen/gallery.json'
```

### Updating Python Packages

```bash
ssh arthur@192.168.0.87
source /opt/arthur-img-env/bin/activate
pip install --upgrade diffusers transformers accelerate
sudo systemctl restart arthur-imglab.service
```

### Grafana Dashboard Management

```bash
# SSH to VM
python3 /opt/arthur-img/create_grafana_dashboard.py
# This recreates/updates the Model Load Monitor dashboard
```

---

## 18. Troubleshooting Runbook

### Service Won't Start

```bash
sudo journalctl -u arthur-imglab.service -n 50 --no-pager
```

Common causes:
- **ImportError** for diffusers/transformers: re-run Phase 3 of deploy script
- **SyntaxError in Python file**: check recently edited source files
- **Port 8002 already in use**: `sudo lsof -i :8002` to find the occupying process

### `torch.cuda.is_available()` Returns `False`

Causes and fixes:
1. **NVML mismatch** (Error 804): `sudo reboot`
2. **Wrong PyTorch build** (CPU-only): Re-run Phase 2 with the correct CUDA wheel URL
3. **GPU not passed through**: Check Proxmox DDA/passthrough config

### Model Loads but Generation Fails

```bash
# Check VRAM state
nvidia-smi
# Check service logs for stack trace
sudo journalctl -u arthur-imglab.service -n 200 --no-pager | grep -A 20 "ERROR\|Exception\|Traceback"
```

### OOM During Loading

If you see `Tried to allocate X GiB` during model load:
- If X ≈ 15 GB for FLUX.2: The `device_map="balanced"` fix was not applied. Ensure the latest `image_lab_engines.py` is deployed.
- If X ≈ 9 GB for SD 3.5: `enable_model_cpu_offload()` was not called. Check the loader.
- Any engine: Check if another process is using VRAM (`nvidia-smi`). The service should have evicted its previous model.

### OOM During Inference (Generation)

If you see OOM after the model loads successfully:
- Try reducing resolution (width/height)
- Reduce `num_inference_steps` 
- For FLUX.2: Confirm `enable_attention_slicing(1)`, `vae.enable_slicing()`, `vae.enable_tiling()` are called after loading
- Check if `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps (set in `.env`)

### Grafana Shows No GPU Data

1. Verify exporter is running: `sudo systemctl status nvidia_gpu_exporter.service`
2. Verify Prometheus scrapes it: `curl -s http://localhost:9090/api/v1/query?query=nvidia_smi_index | python3 -m json.tool`
3. Check for NVML mismatch in exporter logs (fix: reboot VM)
4. Open "Model Load Monitor" dashboard, set time range to "Last 30 minutes" (post-reboot), wait for variables to populate
5. Grafana admin password: `newpass2026` — log in at `http://192.168.0.87:3000`

### Wan Video Is Corrupt / Won't Play

- Ensure `ffmpeg` is installed on the VM: `which ffmpeg`
- Check `imageio-ffmpeg` is installed in the venv: `pip show imageio-ffmpeg`
- Test ffmpeg: `ffmpeg -version`

### HF Token Errors

```
HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/...
```
- Token expired or revoked: generate a new token at https://huggingface.co/settings/tokens
- Update `/opt/arthur-img/.env` and restart service
- Re-run Phase 7 of the deploy script to cache the new token

---

## 19. Grafana Dashboard — Model Load Monitor

**URL:** `http://192.168.0.87:3000/d/model-load-monitor`  
**Auto-refresh:** Every 5 seconds  
**Default time range:** Last 10 minutes  
**UID:** `model-load-monitor`  
**Datasource UID:** `ffjmsi0wmmpdsf` (Prometheus)

### Dashboard Purpose

This dashboard was created specifically to observe what happens during AI model loading. When you click "Generate" in the UI and the lab must load a model from disk into VRAM, the dashboard shows:

1. **Disk Read MB/s spike** — model weights are read from SSD into CPU RAM (typically 200–800 MB/s)
2. **System RAM increase** — model lives briefly in RAM during loading
3. **VRAM Used climb** — model transfers from RAM to GPU (PCIe bandwidth ~10–15 GB/s)
4. **GPU Utilization spike** — model quantization and first inference step

### Panels

#### GPU Section

| Panel | Type | Query |
|---|---|---|
| VRAM Used (GB) | Timeseries | `nvidia_smi_memory_used_bytes{uuid="$gpu"} / 1073741824` + total line |
| GPU Utilization % | Timeseries | `nvidia_smi_utilization_gpu_ratio{uuid="$gpu"} * 100` |
| GPU Power Draw (W) | Timeseries | `nvidia_smi_power_draw_instant_watts{uuid="$gpu"}` |
| GPU Temp °C | Stat | `nvidia_smi_temperature_gpu{uuid="$gpu"}` |
| VRAM Free (GB) | Stat | `nvidia_smi_memory_free_bytes{uuid="$gpu"} / 1073741824` |
| GPU Utilization % | Stat | `nvidia_smi_utilization_gpu_ratio{uuid="$gpu"} * 100` |
| Power Draw (W) | Stat | `nvidia_smi_power_draw_instant_watts{uuid="$gpu"}` |

#### Disk I/O Section

| Panel | Type | Query |
|---|---|---|
| Disk Read MB/s ($disk) | Timeseries | `rate(node_disk_read_bytes_total{device="$disk"}[15s]) / 1048576` |
| Disk Write MB/s ($disk) | Timeseries | `rate(node_disk_written_bytes_total{device="$disk"}[15s]) / 1048576` |

#### System Section

| Panel | Type | Query |
|---|---|---|
| CPU Utilization % | Timeseries | `(1 - avg(rate(node_cpu_seconds_total{mode="idle"}[15s]))) * 100` |
| RAM Used (GB) | Timeseries | `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1073741824` |

### Template Variables

| Variable | Query | Auto-selects | Purpose |
|---|---|---|---|
| `$gpu` | `label_values(nvidia_smi_index, uuid)` | GPU UUID | Filter all GPU panels to specific GPU |
| `$disk` | `label_values(node_disk_read_bytes_total, device)` | `sda` (regex filtered) | Select which disk to show I/O for |

Both variables use `refresh: 1` (populate on dashboard load), so they auto-select without user interaction.

### Reading the Dashboard During a Model Load

A typical FLUX.2 load from cold (no model in VRAM) on the RTX 5060 Ti:

```
T=0s   User clicks Generate
T=1s   Disk Read spikes to 400-800 MB/s (loading transformer weights from NVMe)
T=8s   RAM Used increases by ~8 GB (transformer in CPU RAM)
T=12s  Disk Read spikes again (loading text encoder weights)
T=20s  RAM Used increases by ~6 GB more (full model in RAM)
T=22s  VRAM Used climbs from 0 to 10 GB (GPU loading via PCIe, BnB quantization active)
T=30s  VRAM at ~10 GB, Disk I/O returns to baseline
T=35s  GPU Utilization spikes (inference begins)
T=90s  GPU Utilization drops (inference complete), VRAM stays at 10 GB (model stays loaded)
T=90s  Result appears in browser
```

---

## 20. Known Limitations & Future Work

### Current Limitations

1. **Single request at a time**: The generation function is synchronous and blocks the FastAPI event loop. Concurrent requests will time out or queue. Fix: move generation to a background thread or process.

2. **No authentication**: Anyone on the local network can use the service. Fix: add FastAPI `HTTPBasicAuth` or an API key middleware.

3. **No persistent gallery beyond gallery.json**: If the file is lost, generated images are orphaned. Fix: use SQLite for the gallery index.

4. **Model eviction on every engine switch**: If a user alternates between FLUX.2 and SD 3.5, each switch takes 30–60 s to reload. Fix: implement LRU caching or allow both to coexist if VRAM permits.

5. **No progress reporting during inference**: The browser shows "Generating…" but has no step-by-step progress. Fix: use diffusers `callback_on_step_end` to emit SSE or WebSocket progress events.

6. **Wan loads both T2V and I2V simultaneously**: Takes 2x the loading time and 2x the RAM even if only T2V is used. Fix: make I2V load lazy.

7. **No batching for video generation**: Wan generates one video per request. Fix: batch multiple requests into one pipeline call.

8. **Generated files accumulate indefinitely**: No TTL or cleanup. Fix: add a background task that purges files older than N days.

9. **FLUX.2 reference image format**: The `image=` parameter in `Flux2Pipeline.__call__` is expected to be a PIL Image for the reference frame, but the current code passes raw bytes — this may fail depending on how `_load_ref_image()` handles it.

### Planned Improvements

- **LoRA support**: Load and apply LoRA weights to any engine for style fine-tuning
- **Multi-GPU support**: When a second GPU is added, assign one model per GPU permanently
- **REST API client library**: A Python client for programmatic access from other services
- **Prompt history**: Save prompts in localStorage so users can recall previous sessions
- **Image upscaler**: Add a lightweight Real-ESRGAN 4x pass as a post-processing step
- **Video-to-Video**: Wan supports animating based on a source video with motion control
- **Grafana alerting**: Alert when VRAM > 95% or GPU temperature > 85°C

---

## 21. Glossary

| Term | Meaning |
|---|---|
| **BnB** | BitsAndBytes — a library by Tim Dettmers for 4-bit and 8-bit quantization of neural network weights |
| **NF4** | NormalFloat4 — BitsAndBytes' 4-bit quantization format optimized for normally-distributed weights. Uses a pre-defined codebook of 16 values |
| **DiT** | Diffusion Transformer — a class of diffusion model that uses transformer blocks instead of U-Net blocks |
| **MMDiT** | Multimodal Diffusion Transformer — the architecture used by SD 3.5, which processes image and text tokens jointly |
| **VRAM** | Video RAM — the dedicated memory on the GPU. Currently 15.48 GB on the RTX 5060 Ti |
| **CPU offload** | Technique where model weights live in CPU RAM and are moved to GPU only during the forward pass, then moved back |
| **device_map** | An accelerate feature that automatically distributes model layers across available devices (GPU, CPU) based on memory budget |
| **HF Hub** | HuggingFace Hub — the model repository hosting service at huggingface.co |
| **safetensors** | A safe, fast file format for storing PyTorch tensors, developed by HuggingFace |
| **snapshot_download** | HuggingFace Hub function that downloads all files of a model revision to the local cache |
| **model_index.json** | A file in diffusers model repos that declares which pipeline class and component classes the model uses |
| **systemd** | The init system and service manager for Linux used to manage the image lab service |
| **accelerate** | HuggingFace library that handles multi-device model distribution and mixed-precision training/inference |
| **NVML** | NVIDIA Management Library — the low-level C API that nvidia-smi and monitoring tools use |
| **DDA** | Discrete Device Assignment — Microsoft's name for GPU passthrough in Hyper-V. Used here in Proxmox context to refer to GPU passthrough generally |
| **Proxmox** | The hypervisor that hosts the Ubuntu VM. Provides KVM virtualization with PCIe passthrough for the GPU |
| **PCIe bandwidth** | The data transfer rate between CPU and GPU over the PCIe bus (~10–16 GB/s for PCIe 4.0 x16) |
| **rectified flow** | The mathematical framework used by FLUX models for the diffusion process. Different from DDPM used by older Stable Diffusion models |
| **Mistral3** | A multimodal VLM (Vision-Language Model) used as the text encoder in FLUX.2, replacing the T5 encoder used in FLUX.1 |
| **VAE** | Variational Autoencoder — the component that compresses images to/from the latent space where diffusion operates |
| **cfg** / **guidance scale** | Classifier-Free Guidance scale. Higher values make the model follow the prompt more closely but can reduce variety. 3.5–5.0 is typical |
| **Prometheus** | Open-source time-series metrics database and query engine |
| **Grafana** | Open-source metrics visualization and dashboarding tool |
| **nvidia_gpu_exporter** | A Prometheus exporter that reads nvidia-smi metrics and exposes them at `:9835/metrics` |
| **node_exporter** | A Prometheus exporter for Linux system metrics (CPU, RAM, disk, network) at `:9100/metrics` |
| **LRU** | Least Recently Used — a caching eviction policy where the item not accessed for the longest time is evicted first |
| **SSE** | Server-Sent Events — a web standard for pushing real-time updates from server to browser over HTTP |
| **idempotent** | An operation that produces the same result whether run once or many times. All 8 deploy phases are designed to be idempotent |

---

*Document generated: May 2026*  
*Maintained by: Arthur Engineering Team*  
*For issues: check the service logs first (`sudo journalctl -u arthur-imglab.service -f`)*
