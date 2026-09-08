# Arthur Image & Video Generation Lab — Complete Engineering Reference

> **Version:** May 2026 — **revised 2026-09-07** (sd35 + wan removed; Z-Image, Qwen-Image 2512, HiDream O1, ERNIE-Image added — see `docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md`)
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
- ~~Text-to-Video / Image-to-Video~~ — video engines (Wan2.2) **removed 2026-09-07**; the video plumbing (`save_video`, `/files/videos`, UI type-driven rendering) stays for a future video engine

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
│  │  diffusers ≥0.40.0      │   │  SANA ×2 (9.1 G each)    │    │
│  │  BitsAndBytes 4-bit     │   │  Boogu (20 G)            │    │
│  │  accelerate 1.13.0      │   │  ideogram-4-nf4 (16 G)   │    │
│  └─────────────────────────┘   │  + zimage/qwenimage/ernie│    │
│                                │  + ComfyUI sidecar models │    │
│                                └──────────────────────────┘    │
│                                                                  │
│  Monitoring Stack (port 3000 / 9090 / 9835 / 9100)             │
│  Grafana ← Prometheus ← nvidia_gpu_exporter + node-exporter    │
└─────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

1. Browser sends `POST /generate/flux2klein` (multipart form, optional image upload)
2. `image_lab_dispatch.py` validates the engine key, reads form fields
3. `engines.generate("flux2klein", params)` is called via `asyncio.to_thread` — the HTTP connection stays open until done, but `/status`, `/logs` and the management endpoints keep serving (`generating: true` in the poll)
4. `_ensure_engine("flux2klein")` evicts any loaded model, loads FLUX.2 Klein into VRAM
5. `_generate_flux2klein(params)` runs the diffusion pipeline; PyTorch uses CUDA
6. The output image/video is written to `/opt/arthur-gen/images/` or `.../videos/`
7. A JSON entry (with per-image `stats` — started/finished timestamps + load/generation split) is appended to `gallery.json`
8. The file path + metadata is returned to the browser as JSON
9. The browser renders the image/video card in the output pane with its stat block (prompt, timings, params)

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
    key: str          # "flux2klein" | "flux2klein9b" | "ideogram4" | "sana" | "boogu"
                      #  | "zimage" | "qwenimage" | "hidream" | "ernie"
    label: str        # Human-readable display name
    description: str  # Shown in the UI sidebar
    output_type: str  # "image" | "video"
    vram_gb: float    # VRAM estimate when loaded (for display only)
    hf_repo: str      # Primary HuggingFace repo ID
    hf_repo_alt: str  # Secondary repo (GGUF / quantised variant)
    params: list      # Parameter schema (drives the UI dynamically)
    available: bool   # Set True at startup after import checks
    loaded: bool      # Set True when model is in VRAM
    error: str        # Last error message if unavailable
```

```python
@dataclass
class LabState:
    active_engine: str   # Which engine is currently in VRAM
    active_quant: str    # Quant of the loaded engine ("" = BF16/default)
    loaded_model: Any    # The loaded pipeline object
    loading: bool        # True during model load
    generating: bool     # True during inference
    run_started: float   # wall clock when a generate() run began (incl. load)
    run_loaded_at: float # wall clock when the model finished loading (0.0 if not reached)
                         #   → read by save_image/save_video to stamp each entry's stats
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
    │       ├── STATE.active_engine = None
    │       └── free_vram()  (gc.collect + torch.cuda.empty_cache)
    └── _LOADERS[key]()  → _load_flux2klein() / _load_flux2klein9b() / _load_ideogram4()
                          / _load_sana() / _load_boogu() / _load_zimage()
                          / _load_qwenimage() / _load_ernie() / _load_hidream()
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
| `/status[?brief=1]` | GET | JSON: engines list, torch VRAM, device-wide `gpu` report + host `system` RAM, active engine, loading/generating flags. `brief=1` drops per-engine `params`/`description` for the 4 s UI poll |
| `/generate/{engine_key}` | POST | Multipart form submission; runs generation in a worker thread |
| `/engines/{engine_key}/load` | POST | Preload engine (async, optional `quant` form field) |
| `/engines/unload` | POST | Evict the resident engine |
| `/engines/{engine_key}/evict` | POST | Evict ONE engine if it is resident |
| `/evict-all` | POST | Whole-card eviction — Image Lab engine + all TTS engines via orchestrator `localhost:8009/evict-all` |
| `/refresh` | POST | Re-probe engine availability |
| `/files/images/{filename}` | GET | Serve generated PNG (same for `/files/videos/...` MP4) |
| `/gallery` | GET | Return gallery JSON array |
| `/gallery/{entry_id}` | DELETE | Remove a gallery entry + file |
| `/` | GET | Returns the full Web UI (HTML) |

---

### `image_lab_ui.py` — Browser Interface

A single Python string constant `UI_HTML` containing the entire frontend — HTML, CSS, and JavaScript — returned by `GET /`. No build step, no npm, no separate static files.

**UI capabilities:**
- Engine selector tabs (FLUX.2 Klein / Klein 9B-KV / Ideogram 4 / SANA 1.6B / Boogu Turbo / Z-Image / Qwen-Image 2512 / HiDream O1 / ERNIE-Image)
- Dynamic parameter form (generated from `engine.params` schema via the `/status` API)
- **VRAM/system report strip** (TTS-Lab-style, polled every 4 s with `?brief=1`): host RAM bar, device-wide VRAM bar with % + "hot" state, GPU badge, per-process "who holds the VRAM" line (container names resolved via `/proc/<pid>/cgroup` + `docker ps`), resident-engine chip with ✕ evict, **Evict VRAM** (whole card incl. TTS containers) and **🔄 Refresh** buttons
- **⬇ Preload / ⏏ Unload** buttons per engine tab (load honors the selected quant; load is async so the UI stays live — the status dot pulses amber while `loading: true`)
- Status dot (green=idle, amber=loading or generating)
- Per-image stat cards (prompt + started/finished times + duration with load/gen split + all params; seed click-to-copy)
- Gallery detail modal — click any thumbnail for the full media + same stat block
- Output gallery (images displayed inline, videos with playback controls)
- Reference image drag-and-drop upload for I2I / I2V modes
- Toast notifications, Download button for each result
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

### 4.1 FLUX.2 [dev] — `flux2` — 🗑️ REMOVED (was: BLOCKED)

| Property | Value |
|---|---|
| **HuggingFace repo** | `diffusers/FLUX.2-dev-bnb-4bit` (gated) |
| **Architecture** | 32B rectified flow transformer (DiT) |
| **Text encoder** | Mistral3ForConditionalGeneration (VLM, multimodal) |
| **Quantization** | Transformer: **GGUF Q4_K_M** (`city96/FLUX.2-dev-gguf`, ~20 GB). Text encoder: **NF4 BnB 4-bit** |
| **Disk size** | ~19 GB GGUF + ~32 GB BnB encoder cache (~51 GB total) |
| **Status** | **REMOVED 2026-08-13** — engine entry, loaders, UI tab, and all model files deleted from the VM |
| **Output type** | Image (PNG) |
| **Supports I2I** | Yes — pass `reference_image` for image editing mode |
| **License** | FLUX [dev] Non-Commercial License |
| **Requires HF token** | Yes (gated model) — was `HF_TOKEN` in `/opt/arthur-img/.env` |

**🗑️ REMOVED (2026-08-13):** the Q4_K_M GGUF is ~20 GB and the NF4 text
encoder another ~6 GB — **~27 GB total, larger than the whole 15.5 GiB card**.
No amount of TTS-engine eviction helps, and the GPU-only policy forbids the
CPU offloading that would be required (user directive: *"never use CPU
rendering / system-RAM offloading — if it doesn't fit even with TTS models
removed, disable it, maybe remove it completely"*). Deleted: the engine entry
(`image_lab_config.py`), the three loader/synth functions + probe
(`image_lab_engines.py`), the UI tab, the 19 GB GGUF directory
(`/opt/arthur-img-models/gguf/flux2/`), the 32 GB BnB cache
(`huggingface/hub/models--diffusers--FLUX.2-dev-bnb-4bit/`), and the
`nvfp4/flux2` stub. The Klein siblings (`flux2klein`, `flux2klein9b`) are
unaffected — they fit and run pure-GPU.

History (for reference): a leaf-level group-offload implementation was tested
2026-08-13 and *worked* (~15 min/image at 28 steps — CPU streaming), but was
rejected: too slow, and CPU streaming is against the GPU-only policy.
Section 12 holds the even older BnB-era account. To restore on a 32 GB+ GPU:
re-add the loader/probe with the structural VRAM guard, re-download the GGUF
and BnB cache, and unblock the probe check.

---

### 4.2 Stable Diffusion 3.5 Large — `sd35` — 🗑️ REMOVED

| Property | Value |
|---|---|
| **Status** | **REMOVED 2026-09-07** — superseded by the Z-Image / Qwen-Image-2512 / HiDream-O1 / ERNIE-Image round (docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md) |
| **Why** | "dated — latent VAE blurs glyphs" (user research doc); the four additions all out-type it on the typography criteria the lab cares about |

Historical: 8B MMDiT + CLIP-L/CLIP-G/T5-XXL encoders, Stability AI Community License (gated), ran city96 GGUF-quantised transformer with pre-saved shared encoders (`preq_save.py`) or torchao NVFP4 (`nvfp4_save.py` — both scripts SUPERSEDED). Code, engine entry, UI tab, GGUF/NVFP4/quantized VM files all deleted; HF cache cleanup under Phase F.

---

### 4.3 Wan2.2 — `wan` — 🗑️ REMOVED

| Property | Value |
|---|---|
| **Status** | **REMOVED 2026-09-07** — video engines dropped from the lab (user decision; see the T2I-swap session doc) |
| **Why** | Wan2.2 is a video model, not text-first — the lab is now all-image. Video *plumbing* (save_video, /files/videos, UI type-driven rendering) remains for a future video engine. |

Historical: 14B causal video diffusion (Alibaba), two GGUF-quantised transformers (HighNoise + LowNoise) per mode from QuantStack, T2V (`STATE.loaded_model`) + I2V (`STATE.loaded_pipe2`) pair. `loaded_pipe2` removed from `LabState`; the Wan Form fields (`mode`/`num_frames`/`fps`/`resolution`) removed from `image_lab_dispatch.py`.

---

### 4.4 SANA 1.6B — `sana` (Sprint + 1.5 variants)

One engine key, two checkpoints selected via the existing `quant` form field
(the string that drives reload on change):

| Property | Value |
|---|---|
| **Sprint repo** | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` |
| **1.5 repo** | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` |
| **Architecture** | 1.6B DiT + Gemma-2-2B-IT text encoder + DC-AE (32×) VAE |
| **Quantization** | None — whole pipeline bf16, fully GPU-resident |
| **Disk size** | 9.74 GB per repo; ~14 GB net for both (Gemma-2-2B shards dedup across the pair) |
| **VRAM (measured 2026-09-07)** | resident 8,968 MiB; gen device peak 10,866 MiB (1024²) |
| **Output type** | Image (PNG), up to 4 per request |
| **Resolution** | 256–2048 px, step 32 (DC-AE 32× compression) |
| **License** | Apache 2.0 + Gemma terms |
| **Requires HF token** | No (ungated) |

**Variants (`quant` field):**

| Value | Pipeline | Steps | CFG |
|---|---|---|---|
| `sprint-1.6b` (default) | `SanaSprintPipeline` + SCMScheduler | 1–4 (default 4; >4 clamped server-side with a log line) | none (guidance-free) |
| `1.5-1.6b` | `SanaPipeline` + DPM scheduler | 1–24 (default 20; UI preset sets this on variant switch) | 4.5 default; `negative_prompt` supported |

**Sprint SCM schedule quirk (fix baked into `_generate_sana`):** diffusers'
`SanaSprintPipeline` passes `intermediate_timesteps=1.3` (its signature
default) to the SCMScheduler, which accepts that only at exactly 2 steps (the
SCM max→1.3→0 jump). For 1/3/4 steps the code passes
`intermediate_timesteps=None`, making the scheduler fall back to the linear
`max_timesteps→0` schedule the Sprint distiller was trained on. The 2-step
path keeps the default 1.3 jump.

**Loading strategy:**

```python
pipe = SanaPipeline / SanaSprintPipeline.from_pretrained(
    repo, torch_dtype=torch.bfloat16, token=None,  # ungated
)
pipe.to("cuda")                  # whole pipeline resident incl. Gemma encoder
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()  # protects 2048² gens
```

---

### 4.5 Boogu-Image 0.1 Turbo — `boogu` ⚠️ CPU-OFFLOAD EXCEPTION

> **The ONLY engine exempt from the lab's GPU-only policy** (user-approved
> 2026-09-06). The vendor's own 16 GB guidance is fp8 weights +
> `enable_model_cpu_offload()`, and the fp8 mllm + bf16 DiT together exceed
> the card. Every other engine still honors `IMGLAB_GPU_ONLY` — this one
> stages modules from CPU RAM across a request by design.

| Property | Value |
|---|---|
| **HuggingFace repo** | `Boogu/Boogu-Image-0.1-Turbo-fp8` (~21 GB disk) |
| **mllm** | Qwen3-VL-8B-class, **fp8** (`quant_method` in config — the name's "fp8" is the mllm + a runtime flag, not the DiT) |
| **transformer** | custom `BooguImageTransformer2DModel`, **bf16-stored** `.bin` shards (loaded with `use_safetensors=False`) |
| **vae / scheduler** | FLUX.1 VAE (335 MB) + custom in-repo scheduler (`pip` package `boogu`, cloned from `github.com/boogu-project/Boogu-Image`) |
| **Output type** | Image (PNG), up to 2 per request |
| **Resolution** | ≤ 1536×1536, step 16 (FLUX.1 VAE) |
| **Steps / CFG** | 1–8, default 4; CFG forced 1.0 — **no** `negative_prompt` field |
| **License** | Apache 2.0 (repo states research use only) |
| **Requires HF token** | No (ungated) |
| **VRAM (measured 2026-09-07)** | transient GPU peak 12,640 MiB (1024², 4 steps); ~0 resident between requests |
| **RAM (measured)** | peak 26,814 MB of 64 GB host (60% gate) |
| **Latency (measured)** | first load 383 s incl. ~14 GB cache top-up; first gen ~100 s (module staging); warm gens 37–41 s |

**Loading strategy (thin wrappers in `boogu_lab_engine.py`):**

```python
os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"  # before ANY transformers import
transformer = BooguImageTransformer2DModel.from_pretrained(
    repo_snapshot, torch_dtype=torch.bfloat16, use_safetensors=False)
pipe = BooguImageTurboPipeline.from_pretrained(
    repo_snapshot, torch_dtype=torch.bfloat16,
    trust_remote_code=True, transformer=transformer)
pipe.enable_model_cpu_offload(device="cuda")   # THE approved offload exception
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
```

Do **not** enable `torch.compile` (documented all-black outputs). The fused-op
gate in `block_lumina2.py` keys off the lowercase `device` env var (unset in
the lab) → torch RMSNorm fallback, benign — same path the upstream direct run
validated. A failed generation can't strand components GPU-pinned: `mllm` and
`processor` were added to the shared `_gpu_attrs` unload list.

---

### Engine Comparison Summary

| Feature | FLUX.2 Klein 4B | FLUX.2 Klein 9B-KV | Ideogram 4 | SANA 1.6B | Boogu Turbo | Z-Image | Qwen-Image 2512 | HiDream O1 | ERNIE-Image |
|---|---|---|---|---|---|---|---|---|---|
| Output | Image | Image | Image | Image | Image | Image | Image | Image | Image |
| Model | 4B DiT + Qwen3-4B | 9B-KV DiT + Qwen3-8B | 9.3B DiT + Qwen3-VL | 1.6B DiT + Gemma-2-2B | mllm (fp8) + DiT (bf16) | Z-Image Turbo (MM-DiT) + Qwen3-4B | 20B DiT + Qwen2.5-VL | HiDream-O1-Dev (UiT) | ERNIE-Image-Turbo (DiT + VL encoder) |
| Route | diffusers | diffusers + GGUF | diffusers (module) | diffusers | boogu module | diffusers + GGUF | diffusers + GGUF | **ComfyUI sidecar** (port 8188) | diffusers + NVFP4 |
| VRAM when loaded | ~10 GB | ~10 GB | 6–10 GB (quant) | ~11 GB (measured 10.9 GB peak) | ~13 GB transient peak, CPU-offloaded | ~10–11 GB† | ~14 GB† (encoder-park peak) | ~11–12 GB† (fp8_scaled 8.1 GB + activations) | ~10–11 GB† |
| Steps | distilled (4) | distilled (4) | — | Sprint 1–4 / 1.5 ≤ 24 | 1–8 (default 4) | 1–8 (default 8, CFG 0.0) | 1–50 (default 20, guidance 4.0) | 28 (fixed, CFG 0.0) | 8 (fixed, CFG 1.0) |
| Text rendering | poor | poor | **native** | poor | poor | **native** | **native** | **native** | **native** |
| Reference image | ✓ I2I | ✓ I2I (KV) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Negative prompt | ✓ | ✓ | ✗ | 1.5 only (Sprint is CFG-free) | ✗ (CFG 1.0) | ✗ (CFG 0.0) | ✓ | ✗ (CFG 0.0) | ✗ (CFG 1.0) |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 + Gemma terms | Apache 2.0 (research) | Apache 2.0 | Apache 2.0 | MIT | Apache 2.0 |
| Quantization | GGUF | GGUF Q6_K | NF4/FP8/BF16 | none (bf16) | fp8 mllm + bf16 DiT | GGUF (Q4_K_M default) | GGUF (Q4_K_M default) | fp8_scaled | NVFP4 (+bnb4 encoder) |

† = initial estimate — **calibrate live** (measured numbers recorded in the T2I-swap session doc after verification).

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
| `/dev/sda1` | `/` (root) | 630 GB (one single disk) | OS + `/opt/arthur-img-models/` (image model cache) + `/opt/arthur-img/` (code) + **all TTS docker images/containers** |
| `/dev/sdb1` | `/opt/models` | 180 GB | TTS model weights (100% full — unrelated to image lab) |

> **Important:** `/opt/arthur-img-models/` is NOT a separate mount — it shares the
> 630 GB root disk with the TTS docker stack (measured 2026-09-07: the disk hit
> 100% during the Boogu download; `docker builder prune` + removing a stale
> unreferenced cache entry freed ~87 GB — open-but-deleted container layers can
> pin tens of GB until pruned). Watch `df -h /` before large model downloads.
> After the 2026-09-07 cleanup: 133 GB free (79%).

### Python Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 | System-installed, venv at `/opt/arthur-img-env/` |
| PyTorch | 2.11.0+cu128 | CUDA 12.8 build |
| diffusers | ≥ 0.40.0 | Bumped in the 2026-09-07 swap deploy — ZImagePipeline (≥0.37), QwenImagePipeline (≥0.35), ErnieImagePipeline (≥0.38); see session doc |
| transformers | latest | Includes Mistral3ForConditionalGeneration |
| accelerate | 1.13.0 | Required for `device_map="balanced"` |
| bitsandbytes | latest | BnB NF4 4-bit quantization (qwenimage bnb-4bit encoder) |
| FastAPI | latest | Web framework |
| uvicorn | latest (standard) | ASGI server with websocket support |

### Environment Variables (`.env` file at `/opt/arthur-img/.env`)

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx    # Optional — all current engines are public repos (sd35 gate removed 2026-09-07)
HF_HOME=/opt/arthur-img-models/huggingface
IMGLAB_MODELS_ROOT=/opt/models/image   # Legacy, not actively used
IMGLAB_OUTPUT_ROOT=/opt/arthur-gen
IMGLAB_PORT=8002
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 # Boogu — must be set BEFORE any transformers import
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

Pre-downloads the new models into the HF cache at
`/opt/arthur-img-models/huggingface/` (top-level `models--*` directories —
hf_hub ≥ 1.0 layout, NOT a `hub/` subdirectory). Updated 2026-09-07: sd35/Wan
entries removed; Z-Image / Qwen-Image 2512 / ERNIE / HiDream repos added.
The old `diffusers/FLUX.2-dev-bnb-4bit` entry was removed earlier (flux2
deleted 2026-08-13):

| Model | Download Size | Destination |
|---|---|---|
| `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | ~9.7 GB | `models--Efficient-Large-Model--Sana_Sprint_1.6B_1024px_diffusers` |
| `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | ~9.7 GB (~4.5 GB net — shares the Gemma-2-2B encoder shards already cached) | `models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers` |
| `Boogu/Boogu-Image-0.1-Turbo-fp8` | ~21 GB | `models--Boogu--Boogu-Image-0.1-Turbo-fp8` |
| `Tongyi-MAI/Z-Image-Turbo` | text encoder + VAE only (transformer excluded — GGUF route) | `models--Tongyi-MAI--Z-Image-Turbo` |
| `jayn7/Z-Image-Turbo-GGUF` | Q4_K_M ~5 GB (default tier) | `gguf/zimage/` |
| `Qwen/Qwen-Image-2512` | VL encoder + VAE (transformer excluded — GGUF route) | `models--Qwen--Qwen-Image-2512` |
| `unsloth/Qwen-Image-2512-GGUF` | Q4_K_M ~12.3 GB (default tier) | `gguf/qwenimage/` |
| encoder-quant repo (chosen in Phase B) | ~4–8 GB | HF cache |
| `lite-infer/ERNIE-Image-Turbo-…-nvfp4-…` | ~9.6 GB | `models--lite-infer--…` |
| `Comfy-Org/HiDream-O1-Image` | Dev fp8_scaled ~8.1 GB | comfy `models/checkpoints/` |

**New-model total (2026-09-07 additions) ≈ 45–55 GB.** Uses
`snapshot_download()` with `ignore_patterns=['*.msgpack','*.h5','flax_model*']`
(+ transformer exclusions for the GGUF-route repos; exact ignore list lives in
the deploy script, Phase 4). Snapshot files are symlinks into each repo's
content-addressed `blobs/` directory.

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

> **Full, current API documentation lives in [`IMAGE_LAB_API_REFERENCE.md`](IMAGE_LAB_API_REFERENCE.md)** (all endpoints, response schemas, error reference, curl/Python cookbooks). This section is a condensed overview and can drift.

### `GET /status`

Returns the current state of all engines and hardware. Supports `?brief=1` (drops per-engine `params`/`description` — what the UI polls every 4 s).

**Response (JSON, condensed):**

```json
{
  "engines": [ { "key": "flux2klein", "label": "FLUX.2 Klein", "available": true, "loaded": true, "error": "" } ],
  "active_engine": "flux2klein",
  "active_quant": "",
  "generating": false,
  "loading": false,
  "vram": { "available": true, "allocated_gb": 0.0, "reserved_gb": 6.1, "total_gb": 15.48, "free_gb": 9.3, "device_name": "NVIDIA GeForce RTX 5060 Ti" },
  "system": { "total": 31914, "used": 15022, "free": 16892 },
  "gpu": {
    "available": true, "name": "NVIDIA GeForce RTX 5060 Ti",
    "vram_total_mb": 16280, "vram_used_mb": 9216, "vram_free_mb": 7064,
    "source": "nvidia-smi",
    "processes": [
      { "pid": 5121, "mb": 6144, "process": "python", "container": "" },
      { "pid": 2093, "mb": 2970, "process": "python3", "container": "tts-lab-engine-current" }
    ]
  }
}
```

`gpu` is the **device-wide** view (`nvidia-smi`): it includes the TTS engine containers sharing the card, with per-process container attribution. `vram` is this process's torch view (kept for compatibility).

### Management endpoints

| Endpoint | Purpose |
|---|---|
| `POST /engines/{key}/load` | Preload engine — async (poll `/status` for `loading:true`), optional `quant` form field |
| `POST /engines/unload` | Evict the resident engine |
| `POST /engines/{key}/evict` | Evict one engine (only if resident) |
| `POST /evict-all` | **Whole-card** eviction — Image Lab engine + all TTS engines via `localhost:8009/evict-all` |
| `POST /refresh` | Re-probe availability |

### `POST /generate/{engine_key}`

Triggers image or video generation. Accepts multipart/form-data. Runs in a worker thread — the HTTP response stays open, but `/status` keeps serving.

**URL parameters:** `engine_key` = `flux2klein` | `flux2klein9b` | `ideogram4` | `sana` | `boogu` | `zimage` | `qwenimage` | `hidream` | `ernie`

**Common form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text description of desired output |
| `negative_prompt` | string | `""` | What NOT to include (flux2klein, flux2klein9b, qwenimage) |
| `width` | int | 1024 | Output width in pixels |
| `height` | int | 1024 | Output height in pixels |
| `num_inference_steps` | int | engine default | Denoising steps (clamped per engine; fixed for zimage/hidream/ernie — see §4) |
| `guidance_scale` | float | engine default | Prompt adherence strength (fixed 0.0 for zimage/hidream, 1.0 for ernie/boogu) |
| `seed` | int | -1 | -1 for random, fixed value for reproducibility |
| `quant` | string | engine default | Quantization level (see the engine tables in §4) |
| `reference_image` | file | null | Optional image upload (FLUX.2 Klein I2I) |

**Response (success, 200):**

```json
{
  "results": [
    {
      "id": "uuid-string",
      "engine": "flux2klein",
      "type": "image",
      "filename": "flux2klein_uuid.png",
      "url": "/files/images/flux2klein_uuid.png",
      "params": { "prompt": "...", "seed": 1234567 },
      "stats": { "started_at": 1779663000.0, "finished_at": 1779663001.9, "load_s": 1.6, "total_s": 1.9 },
      "created_at": 1779663000.0
    }
  ]
}
```

**Response (error, 503):**

```json
{ "detail": "CUDA out of memory. Tried to allocate 15.01 GiB..." }
```

### Files & gallery

| Endpoint | Purpose |
|---|---|
| `GET /files/images/{filename}` | Raw PNG (videos: `/files/videos/{filename}`) |
| `GET /gallery` | Gallery JSON — entries same shape as generate results, `stats` on rows saved since 2026-09-06 |
| `DELETE /gallery/{entry_id}` | Remove a gallery entry and delete its file from disk |

---

## 8. Web UI Guide

Access the UI at **`http://192.168.0.87:8002`** from any browser on the local network.

### Layout

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ Sidebar (300 px)         │ Main pane                                      │
│                          │  ┌──────────────────────────────────────────┐ │
│ [FLUX.K][9B][Ideogram4]  │  │ VRAM/system report strip                 │ │
│ [SANA][Boogu][ZImg] …    │  │ ● Ready · Loaded: FLUX.2 Klein ...        │ │
│                          │  │            [Evict VRAM][🔄 Refresh]       │ │
│ ┌──────────────────────┐ │  │ RAM ▓▓▓░░ 8.1/31.2 GiB                   │ │
│ │ Parameter form       │ │  │ VRAM ▓▓▓▓▓ 6.4/15.9 GiB (40%)  🟢RTX5060 │ │
│ │  Prompt: [textarea]  │ │  │ GPU: Image Lab 5.8 GiB · ... · 🧠 chip ✕  │ │
│ │  ...  Quant: [Q4_K_M]│ │  └──────────────────────────────────────────┘ │
│ └──────────────────────┘ │  ┌──────────────────────────────────────────┐ │
│ [⬇ Preload][⏏ Unload]    │  │ Generate │ Gallery │ Logs                │ │
│ [⚡ Generate]            │  │ ┌──────────────────────────────────────┐ │ │
│ ──────────────────────── │  │ │ [image] ⬇ Download                  │ │ │
│ Engine description       │  │ │ ┌──────────────────────────────────┐ │ │ │
│ ▸ Log panel              │  │ │ │ Prompt used (expanded)           │ │ │ │
└──────────────────────────┘  │ │ │ "the fox, golden hour..."  ───────│ │ │ │
                              │ │ │ Started 14:02:11 · Finished 14:02:13 │ │ │ │
                              │ │ │ ⏱ 1.9 s (1.6 s load · 0.3 s gen)     │ │ │ │
                              │ │ │ Seed · Width · Height · Steps · ... │ │ │ │
                              │ │ └──────────────────────────────────┘ │ │ │
                              │ └──────────────────────────────────────┘ │ │
                              └──────────────────────────────────────────┘
```

### VRAM / System Report Strip

A full-width strip above the view tabs (polled every **4 s** via `/status?brief=1`), mirroring the TTS Lab's VRAM report:

- **Row 1** — status dot + text, `Loaded: <engine> · <quant>`, then **Evict VRAM** (red ghost button) and **🔄 Refresh**.
- **Row 2** — **RAM bar** (host memory, MB from `/status.system`), **VRAM bar** (device-wide, MB from `/status.gpu` — includes the TTS containers, not just this process), GPU badge (`🟢 RTX 5060 Ti · 15.9 GiB`), and the **🧠 In VRAM** chip line.
- The **detail line under the VRAM bar** shows *who* holds the card, e.g. `GPU: Image Lab 5.8 GiB · tts-lab-engine-editx 12.9 GiB` (bare-metal host python → "Image Lab"; container PIDs resolved to `tts-lab-*` names via `/proc/<pid>/cgroup` + `docker ps`).
- The resident engine chip (`🧠 FLUX.2 Klein · Q4_K_M ✕`) evicts just that engine — ✕ calls `POST /engines/{key}/evict`.
- **Evict VRAM** asks for confirmation, then calls `POST /evict-all`: unloads the Image Lab engine **and** POSTs `localhost:8009/evict-all` to evict every TTS engine container sharing the card. The toast reports each side's outcome (TTS counts/freed MB come from the orchestrator payload; a down orchestrator shows as a warning, not an error).
- All management buttons disable while `loading`/`generating` is true (re-enabled on the next poll).

### Preload / Unload

Each engine tab has **⬇ Preload** and **⏏ Unload** buttons above Generate. Preload sends the currently-selected `quant` value (`POST /engines/{key}/load`, `quant` form field) and is fully **asynchronous** — the load runs server-side while the UI keeps polling (`loading: true`, amber dot); the button re-enables when the load finishes. Unload evicts whatever is resident. The ⚠️ quantization-change warning still appears when the selected quant differs from the loaded one.

### Per-Image Stat Cards & Gallery Detail

Every result card now carries a full stat block:

- **Header** — engine chip, quant chip, finished time, ⬇ Download.
- **Prompt** — the text actually sent (Ideogram 4 shows the magic-prompt-expanded caption as *"Prompt used (expanded)"* with the raw *"Submitted prompt"* beneath when they differ). Long prompts clamp to 3 lines with a *Show more* toggle.
- **Timing grid** — Started / Finished (local time), and a **Duration pill** (`⏱ 1.9 s`) with the load/gen split (`1.6 s load · 0.3 s gen`) when the model had to load. Rows saved before 2026-09-06 show *"not recorded"*.
- **Parameter grid** — seed (click to copy), width, height, quant, steps, guidance, frames/FPS for videos, preset, magic-prompt flags, negative prompt, … — rendered from the persisted `params`.

Clicking a **gallery thumbnail** opens the same stats in a detail **modal** (larger media + full stat block + Download / 🗑 Delete). Old gallery rows without `stats` render with what they have.

### Status Indicator

- 🟢 **Green dot** — Service idle ("Ready")
- 🟡 **Amber pulsing dot** — "Loading model…" (preload or quant reload) or "Generating…"
- 🔴 **Red dot** — Server unreachable (poll failed)

### Reference Image Upload

For FLUX.2 Klein (I2I / image editing): click the dashed file drop zone to upload a reference image. The file is sent as multipart form data with the generation request. (Wan I2V — the only other upload consumer — was removed 2026-09-07; the drop zone still renders per engine schema.)

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
    STATE.active_engine = None
    gc.collect()                # Python garbage collector
    torch.cuda.empty_cache()    # Release PyTorch's CUDA memory pool
    torch.cuda.ipc_collect()    # Clean up inter-process CUDA handles
```

After eviction, `nvidia-smi` should show ~0.5 GB used (CUDA driver overhead only).

This cycle only manages **this process**. The Image Lab shares the card with the TTS engine containers, so whole-card management goes through `POST /evict-all` (UI: **Evict VRAM**): it runs `_unload_current()` here and then POSTs `localhost:8009/evict-all` on the TTS orchestrator (the same call `_evict_tts_engines()` makes when it needs headroom). The `/status` `gpu.processes` block (container-attributed via `/proc/<pid>/cgroup` + `docker ps`) shows whether anything else still holds the card.

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

### The `enable_model_cpu_offload()` Strategy (historical — SD 3.5, Wan; both REMOVED 2026-09-07)

For models where individual components exceed VRAM:
1. All model submodules start in CPU RAM
2. accelerate hooks `forward()` to move each top-level component to GPU when needed
3. After the component runs, it moves back to CPU
4. Only one major component (text encoder OR transformer OR VAE) is on GPU at a time

Historical example: SD 3.5's T5-XXL (~9 GB bfloat16) ran on GPU during text encoding, then moved to CPU. No remaining engine uses full-pipeline CPU offload — the current pattern is encode-then-park (klein encoders, SANA), boogu's CPU-offload exception, or whole-component streaming.

### Memory Optimization Techniques Applied

| Technique | Benefit | Applied To |
|---|---|---|
| `pipe.vae.enable_slicing()` | VAE decodes in slices, reducing peak VRAM by 2–3 GB | FLUX.2 |
| `pipe.vae.enable_tiling()` | VAE processes large images in tiles, flat memory cost | FLUX.2 |
| `pipe.enable_attention_slicing(1)` | Attention computed one head at a time, reduces peak VRAM | FLUX.2 |
| `device_map="balanced"` | accelerate-managed placement (BnB-safe) | FLUX.2 (historical) |
| encode-then-park | Text encoder produces embeds, then drops to CPU / ref-dropped | klein engines, SANA |
| encode-headroom eviction | TTS evict-all before encoder loads when VRAM is short | qwenimage (planned) |

(SD 3.5 / Wan rows removed with the engines — 2026-09-07.)

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

huggingface_hub ≥ 1.0 (the version in the imglab venv) stores the
content-addressed cache **directly under `$HF_HOME`** — top-level
`models--<org>--<repo>` directories, no `hub/` subdirectory. (A legacy
`hub/` subdir holding ~69 GB of duplicates — downloaded by a one-off upstream
script run — was deleted 2026-09-07 after confirming no code path reads it.)

```
/opt/arthur-img-models/huggingface/          (92 GB total)
├── models--Boogu--Boogu-Image-0.1-Turbo-fp8/
│   ├── blobs/          ← Actual weight files (content-addressed)
│   │   ├── 9f41d2aa...  (10.3 GB — transformer .bin shards, bf16)
│   │   ├── 6e0cdd51...  (10.6 GB — mllm fp8 shards)
│   │   └── ...
│   ├── refs/main       ← Current commit hash
│   └── snapshots/
│       └── 6e7d02c1.../  ← Symlinks to blobs
│           ├── mllm/  ├── transformer/  ├── vae/
│           ├── processor/  ├── scheduler/
│           └── model_index.json
├── models--Efficient-Large-Model--Sana_Sprint_1.6B_1024px_diffusers/   (9.1 GB)
├── models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers/       (9.1 GB — Gemma shards dedup against the Sprint repo)
├── models--ideogram-ai--ideogram-4-nf4/    (16 GB)
├── models--ideogram-ai--ideogram-4-fp8/    (8.7 GB)
├── models--Qwen--Qwen3-8B/                 (16 GB — shared by klein9b + ideogram)
├── models--black-forest-labs--FLUX.2-klein-4B/   (15 GB)
└── models--Wan-AI--Wan2.1-T2V-14B-Diffusers/    (stale leftover — Wan2.2 used GGUF; cleanup target, Phase F)
```

(sd35/wan model dirs removed from this tree with the engines — 2026-09-07; any
surviving `models--stabilityai--*` / `models--Wan-AI--*` blobs are Phase-F
cleanup targets.)

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
| `/opt/arthur-img-models/` | Image model cache | ~227 GB at 2026-09-07 pre-cleanup (huggingface/ 92 G + gguf/ 71 G + quantized/ 44 G + nvfp4/ 20 G); sd35/wan dirs deleted under Phase F |
| `/opt/arthur-gen/` | Generated outputs | Growing |
| `/opt/models/` | TTS models (separate service) | 177 GB (full) |

### Capacity Planning

| Model | On-disk copies (2026-09-07 du) | VRAM | RAM (offload) |
|---|---|---|---|
| FLUX.2 Klein 4B / 9B-KV | hf-cache 15 G / gguf/ 22 G (Q6_K + quant ladder) | ~10 GB each | ~1.5 GB (Qwen encoders lazy) |
| Ideogram 4 | hf-cache 25 G (nf4 16 G + fp8 8.7 G; Qwen3-8B 16 G shared with klein9b) | 6–10 GB | — |
| SANA Sprint + 1.5 | hf-cache 18.2 G (Gemma shards deduped) | ~11 GB (measured device peak 10.9 GB) | — (GPU-only) |
| Boogu Turbo fp8 | hf-cache 20 G | ~13 GB transient | **~27 GB (CPU offload exception)** |
| Z-Image Turbo | hf-cache (encoder+VAE) + gguf/zimage/ Q4_K_M ~5 G | ~10–11 GB † | ~2 GB (Qwen3-4B encoder parks) |
| Qwen-Image 2512 | hf-cache (VL encoder+VAE) + gguf/qwenimage/ Q4_K_M ~12.3 G | ~14 GB † | encoder 4–8 GB staged then parked |
| HiDream O1-Dev | comfy `models/checkpoints/` fp8_scaled ~8.1 G | ~11–12 GB † (separate process!) | comfy sidecar RAM +~12 GB |
| ERNIE-Image-Turbo | hf-cache ~9.6 G (NVFP4 repo) | ~10–11 GB † | — |

(SD 3.5 + Wan rows deleted 2026-09-07 — their VM files are Phase-F cleanup targets. † = estimate, calibrate live.)

Recommended root disk size: **≥500 GB**. The single 630 GB root disk (shared
with the TTS docker stack) had 133 GB free after the 2026-09-07 cleanup.

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

1. **Single generation at a time**: Generation and model loading are single-flight by design (one resident model, `generating`/`loading` flags → second `/generate` or `/load` returns `503 "Server is busy"`). Since 2026-09-06 both run in worker threads (`asyncio.to_thread`), so `/status`, `/logs`, gallery and eviction keep working mid-run — the event loop is never blocked.

2. **No authentication**: Anyone on the local network can use the service. Fix: add FastAPI `HTTPBasicAuth` or an API key middleware.

3. **No persistent gallery beyond gallery.json**: If the file is lost, generated images are orphaned. Fix: use SQLite for the gallery index.

4. **Model eviction on every engine switch**: Alternating engines means a 30–60 s reload per switch. Fix: implement LRU caching or allow both to coexist if VRAM permits.

5. **No progress reporting during inference**: The browser shows "Generating…" but has no step-by-step progress. Fix: use diffusers `callback_on_step_end` to emit SSE or WebSocket progress events.

6. ~~**Wan loads both T2V and I2V simultaneously**~~ — Wan removed 2026-09-07.

7. ~~**No batching for video generation**~~ — video engines removed 2026-09-07; plumbing retained.

8. **Generated files accumulate indefinitely**: No TTL or cleanup. Fix: add a background task that purges files older than N days.

9. **FLUX.2 reference image format**: The `image=` parameter in `Flux2Pipeline.__call__` is expected to be a PIL Image for the reference frame, but the current code passes raw bytes — this may fail depending on how `_load_ref_image()` handles it.

### Planned Improvements

- **LoRA support**: Load and apply LoRA weights to any engine for style fine-tuning (Qwen-Image 2512 Lightning 4-step LoRA is a candidate)
- **HiDream quality ladder**: fp8_scaled → bf16 full model if VRAM ever allows
- **Multi-GPU support**: When a second GPU is added, assign one model per GPU permanently
- **REST API client library**: A Python client for programmatic access from other services
- **Prompt history**: Save prompts in localStorage so users can recall previous sessions
- **Image upscaler**: Add a lightweight Real-ESRGAN 4x pass as a post-processing step
- ~~**Video-to-Video**~~ — Wan removed 2026-09-07; revisit only with a future video engine
- **Grafana alerting**: Alert when VRAM > 95% or GPU temperature > 85°C

---

## 21. Glossary

| Term | Meaning |
|---|---|
| **BnB** | BitsAndBytes — a library by Tim Dettmers for 4-bit and 8-bit quantization of neural network weights |
| **NF4** | NormalFloat4 — BitsAndBytes' 4-bit quantization format optimized for normally-distributed weights. Uses a pre-defined codebook of 16 values |
| **DiT** | Diffusion Transformer — a class of diffusion model that uses transformer blocks instead of U-Net blocks |
| **MMDiT** | Multimodal Diffusion Transformer — the architecture used by SD 3.5 (removed 2026-09-07), processing image and text tokens jointly |
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
