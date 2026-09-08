"""
image_lab_config.py — Engine catalogue, paths, VRAM estimates, and global state
for the Arthur Image & Video Generation Lab (port 8002).
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# All large files live on the 650 GB data disk
MODELS_ROOT      = os.environ.get("IMGLAB_MODELS_ROOT",  "/opt/models/image")
HF_HOME          = os.environ.get("HF_HOME",              "/opt/arthur-img-models/huggingface")
OUTPUT_ROOT      = os.environ.get("IMGLAB_OUTPUT_ROOT",  "/opt/arthur-gen")
IMAGES_DIR       = os.path.join(OUTPUT_ROOT, "images")
VIDEOS_DIR       = os.path.join(OUTPUT_ROOT, "videos")
GALLERY_DB       = os.path.join(OUTPUT_ROOT, "gallery.json")

# HuggingFace auth token (required for gated models and the remote T5 encoder)
HF_TOKEN         = os.environ.get("HF_TOKEN", "")

# Service config
PORT             = int(os.environ.get("IMGLAB_PORT", "8002"))
HOST             = os.environ.get("IMGLAB_HOST", "0.0.0.0")

# Runtime behavior flags
USE_COMFYUI      = os.environ.get("IMGLAB_USE_COMFYUI", "").lower() in ("1", "true", "yes")
GPU_ONLY         = os.environ.get("IMGLAB_GPU_ONLY", "").lower() in ("1", "true", "yes")
NO_MODEL_OFFLOAD = os.environ.get("IMGLAB_NO_MODEL_OFFLOAD", "").lower() in ("1", "true", "yes")

# Normalize GPU-only behavior so either flag works
GPU_ONLY = GPU_ONLY or NO_MODEL_OFFLOAD

# ---------------------------------------------------------------------------
# Engine descriptor
# ---------------------------------------------------------------------------

@dataclass
class EngineInfo:
    key: str                          # API key used in URL  e.g. "flux2klein"
    label: str                        # Human label
    description: str
    output_type: str                  # "image" | "video"
    vram_gb: float                    # Estimated VRAM when loaded
    hf_repo: str                      # Primary HuggingFace repo
    hf_repo_alt: Optional[str]        # Secondary repo (e.g. I2V variant)
    params: list[dict]                # Parameter schema for the UI
    image_input: str = "none"         # "none" | "reference" — can this engine
                                      # natively consume an uploaded reference image?
    available: bool = False           # Set at startup after import checks
    loaded: bool    = False           # Set when model is resident in VRAM
    error: str      = ""              # Last error if unavailable

# ---------------------------------------------------------------------------
# Parameter schema helpers
# ---------------------------------------------------------------------------

def _p(name, type_, default, label, min_=None, max_=None, step=None,
       options=None, tooltip="", required=False, client_only=False):
    d = dict(name=name, type=type_, default=default, label=label,
             tooltip=tooltip, required=required)
    if min_  is not None: d["min"]     = min_
    if max_  is not None: d["max"]     = max_
    if step  is not None: d["step"]    = step
    if options is not None: d["options"] = options
    if client_only: d["client_only"] = True
    return d

# ---------------------------------------------------------------------------
# Engine catalogue
# ---------------------------------------------------------------------------

ENGINES: dict[str, EngineInfo] = {

    "flux2klein": EngineInfo(
        key         = "flux2klein",
        label       = "FLUX.2 Klein 4B",
        description = (
            "FLUX.2 Klein 4B — compact 4B flow transformer from Black Forest Labs. "
            "Apache 2.0 license. Uses Qwen3 text encoder (far smaller than the 32B "
            "dev variant's Mistral 24B). Fits in ~10 GiB VRAM — runs on our RTX 5060 Ti. "
            "Loaded directly from HuggingFace (no pre-saved shared dir needed). "
            "Supports text-to-image and image-to-image (reference image)."
        ),
        output_type = "image",
        image_input = "reference",   # FLUX.2 klein natively conditions on the ref image
        vram_gb     = 13.0,
        hf_repo     = "black-forest-labs/FLUX.2-klein-4B",
        hf_repo_alt = None,
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate.", required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="Describe what you do NOT want in the image."),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Upload a reference image for I2I / style-transfer mode."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output width in pixels. Must be a multiple of 64."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output height in pixels. Must be a multiple of 64."),
            _p("num_inference_steps", "int",      4,      "Steps",
               min_=1, max_=20, step=1,
               tooltip="4 steps is optimal — FLUX.2-klein-4B is a step-distilled model. More steps rarely help."),
            _p("guidance_scale",      "float",    3.5,    "Guidance scale",
               min_=1.0, max_=10.0, step=0.5,
               tooltip="Ignored for this step-distilled model; included for UI consistency."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1,
               tooltip="Fixed seed for reproducible results."),
        ],
    ),

    "flux2klein9b": EngineInfo(
        key         = "flux2klein9b",
        label       = "FLUX.2 Klein 9B-KV",
        description = (
            "FLUX.2 Klein 9B-KV — the mid-size Klein variant from Black Forest Labs "
            "(official 9B-KV repo is gated). Runs the Q6_K GGUF (default) from "
            "QuantStack/FLUX.2-Klein-9B-KV-GGUF with a locally-derived transformer "
            "config (8 double + 24 single blocks, hidden 4096). Uses the Qwen3-8B "
            "text encoder (NF4-quantised) and the shared FLUX.2 Klein VAE/scheduler. "
            "KV reference-token caching enables efficient image editing. "
            "Quant ladder Q3_K_M…Q8_0 (default raised from Q4_K_M on 2026-09-04)."
        ),
        output_type = "image",
        image_input = "reference",   # FLUX.2 klein natively conditions on the ref image
        vram_gb     = 13.5,
        hf_repo     = "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        hf_repo_alt = "black-forest-labs/FLUX.2-klein-4B",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate.", required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="Describe what you do NOT want in the image."),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Upload a reference image for I2I / style-transfer mode (KV-cached)."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output width in pixels. Must be a multiple of 64."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output height in pixels. Must be a multiple of 64."),
            _p("num_inference_steps", "int",      4,      "Steps",
               min_=1, max_=20, step=1,
               tooltip="4 steps is optimal — FLUX.2-klein models are step-distilled. More steps rarely help."),
            _p("guidance_scale",      "float",    3.5,    "Guidance scale",
               min_=1.0, max_=10.0, step=0.5,
               tooltip="Ignored for this step-distilled model; included for UI consistency."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1,
               tooltip="Fixed seed for reproducible results."),
            _p("quant",               "select",   "Q6_K", "Quantization",
               options=[
                   {"value": "Q3_K_M", "label": "Q3_K_M — 4.6 GB  (smallest usable for faces)"},
                   {"value": "Q4_K_M", "label": "Q4_K_M — 5.7 GB  (previous default)"},
                   {"value": "Q5_K_M", "label": "Q5_K_M — 6.8 GB"},
                   {"value": "Q6_K",   "label": "Q6_K   — 7.9 GB  ✓ default"},
                   {"value": "Q8_0",   "label": "Q8_0   — 10.0 GB (near-lossless — exceeds 16 GB card)"},
               ],
               tooltip=(
                   "GGUF quantisation uses QuantStack/FLUX.2-Klein-9B-KV-GGUF, downloaded "
                   "on first use. All quants share one transformer config. Default Q6_K since "
                   "2026-09-04 (Q4_K_M passed the identity A/B; Q6_K is the quality-per-GB "
                   "sweet spot). Q8_0 loads but OOMs at generation on the 16 GB card even "
                   "with TTS containers evicted (verified 2026-09-05) — kept for larger-GPU "
                   "or text-encoder-offload deployments."
               )),
        ],
    ),

            "ideogram4": EngineInfo(
                key         = "ideogram4",
                label       = "Ideogram 4",
                description = (
                    "Ideogram 4 — 9.3B flow-matching DiT with Qwen3-VL text encoder. "
                    "Native text rendering. State-of-the-art prompt adherence. "
                    "Uses a structured JSON caption format (not plain text). "
                    "nf4 quant = ~6 GB VRAM for transformer (bitsandbytes, CUDA only). "
                    "fp8 quant = ~10 GB VRAM for transformer (weight-only float8, any device). "
                    "Optional magic-prompt expansion via OpenRouter (Claude Sonnet)."
                ),
                output_type = "image",
                vram_gb     = 10.0,
                hf_repo     = "ideogram-ai/ideogram-4-nf4",
                hf_repo_alt = "ideogram-ai/ideogram-4-fp8",
                params      = [
                    _p("prompt",                "textarea", "",     "Caption (JSON or plain text)",
                       required=True,
                       tooltip="JSON caption for Ideogram 4, or plain text if Magic Prompt is enabled (auto-expands via DeepSeek)."),
                    _p("reference_image",       "file",     None,   "Reference image (optional)",
                       tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                               "no native image conditioning exists."),
                    _p("use_magic_prompt",      "checkbox", False,  "Magic Prompt (expand via DeepSeek)",
                       tooltip="When enabled, your plain-text prompt is expanded into a structured JSON caption via DeepSeek API. When disabled, your prompt is sent directly (must be valid JSON)."),
                    _p("width",                 "int",      1024,   "Width (px)",
                       min_=256, max_=2048, step=16,
                       tooltip="Must be multiple of 16. Range: 256-2048."),
                    _p("height",                "int",      1024,   "Height (px)",
                       min_=256, max_=2048, step=16,
                       tooltip="Must be multiple of 16. Range: 256-2048."),
                    _p("preset",                "select",   "V4_DEFAULT_20", "Sampler preset",
                       options=[
                           {"value": "V4_QUALITY_48", "label": "V4_QUALITY_48 - 48 steps (best quality)"},
                           {"value": "V4_DEFAULT_20", "label": "V4_DEFAULT_20 - 20 steps (recommended)"},
                           {"value": "V4_TURBO_12",   "label": "V4_TURBO_12 - 12 steps (fastest)"},
                       ],
                       tooltip="Named sampler preset. V4_DEFAULT_20 is best balance."),
                    _p("num_inference_steps",   "int",      0,      "Override steps (0 = preset default)",
                       min_=0, max_=128, step=1,
                       tooltip="0 uses the preset's step count. Set to override."),
                    _p("guidance_scale",        "float",    7.0,    "Guidance scale",
                       min_=1.0, max_=30.0, step=0.5,
                       tooltip="Constant CFG weight. 7.0 is default."),
                    _p("mu",                    "float",    0.0,    "Schedule mean (mu)",
                       min_=-5.0, max_=5.0, step=0.1,
                       tooltip="Logit-normal schedule mean."),
                    _p("std",                   "float",    1.75,   "Schedule std (sigma)",
                       min_=0.1, max_=5.0, step=0.1,
                       tooltip="Logit-normal schedule standard deviation."),
                    _p("seed",                  "int",      -1,     "Seed (-1 = random)",
                       min_=-1, max_=2**31-1, step=1),
                    _p("quant",                 "select",   "nf4",  "Quantization",
                       options=[
                           {"value": "nf4", "label": "nf4 - 6 GB VRAM, CUDA only (recommended)"},
                           {"value": "fp8", "label": "fp8 - 10 GB VRAM, any device"},
                       ],
                       tooltip="nf4 is smaller/faster on CUDA. fp8 works on any device."),
                    _p("magic_prompt_aspect_ratio", "select", "1:1", "Aspect ratio (magic prompt)",
                       options=[
                           {"value": "1:1",  "label": "1:1 - Square"},
                           {"value": "3:2",  "label": "3:2 - Landscape"},
                           {"value": "2:3",  "label": "2:3 - Portrait"},
                           {"value": "16:9", "label": "16:9 - Widescreen"},
                           {"value": "9:16", "label": "9:16 - Phone"},
                       ],
                       tooltip="Target aspect ratio for magic-prompt expansion."),
                ],
            ),

    "sana": EngineInfo(
        key         = "sana",
        label       = "SANA 1.6B",
        description = (
            "SANA 1.6B — efficient DC-AE text-to-image model from NVIDIA "
            "(Efficient-Large-Model). Two variants share one engine key: "
            "Sprint 1.6B (1-4 steps, no CFG — step-distilled) and SANA 1.5 "
            "1.6B (~20 steps, CFG 4.5 — quality). Gemma-2-2B-IT text encoder "
            "stays resident. Whole pipeline bf16 ≈ 10.4 GiB VRAM. "
            "Apache-2.0 + Gemma terms."
        ),
        output_type = "image",
        vram_gb     = 11.0,
        hf_repo     = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        hf_repo_alt = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate.", required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="Used by the SANA 1.5 variant only — Sprint is trained "
                       "guidance-free and ignores it."),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=2048, step=32,
               tooltip="Output width in pixels. Must be a multiple of 32 (DC-AE "
                       "32x compression)."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=2048, step=32,
               tooltip="Output height in pixels. Must be a multiple of 32 (DC-AE "
                       "32x compression)."),
            _p("num_inference_steps", "int",      4,      "Steps",
               min_=1, max_=24, step=1,
               tooltip="Sprint: 1-4 steps (clamped server-side; 4 is best). "
                       "SANA 1.5: ~20 steps recommended."),
            _p("guidance_scale",      "float",    4.5,    "Guidance scale",
               min_=1.0, max_=20.0, step=0.5,
               tooltip="SANA 1.5 uses CFG (4.5 default). Sprint is trained "
                       "guidance-free — value ignored by that variant."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=4, step=1),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "sprint-1.6b", "Variant",
               options=[
                   {"value": "sprint-1.6b", "label": "Sprint 1.6B — 1-4 steps, no CFG (default)"},
                   {"value": "1.5-1.6b",    "label": "SANA 1.5 1.6B — ~20 steps, CFG 4.5"},
               ],
               tooltip=(
                   "Which SANA checkpoint to run. Switching variant unloads the "
                   "current one and reloads from the HF cache (~60 s)."
               )),
        ],
    ),

    "boogu": EngineInfo(
        key         = "boogu",
        label       = "Boogu Turbo",
        description = (
            "Boogu-Image-0.1-Turbo-fp8 — DMD few-step (4) image model from "
            "Boogu Team with a Qwen3-VL-8B-class instruction encoder. Runs "
            "with fp8 mllm weights and CPU model offload — the ONE engine "
            "with a user-approved CPU-offload exception to the lab's "
            "GPU-only policy (nothing stays resident; each generate stages "
            "components on the card, ~20-21 GB of host RAM peak). "
            "Apache-2.0, research-only. No CFG (1.0)."
        ),
        output_type = "image",
        vram_gb     = 12.0,
        hf_repo     = "Boogu/Boogu-Image-0.1-Turbo-fp8",
        hf_repo_alt = None,
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate. Long, detailed "
                       "instructions work best with the VLM encoder.",
               required=True),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=16,
               tooltip="Output width in pixels. Must be a multiple of 16 "
                       "(FLUX.1 VAE)."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=16,
               tooltip="Output height in pixels. Must be a multiple of 16 "
                       "(FLUX.1 VAE)."),
            _p("num_inference_steps", "int",      4,      "Steps",
               min_=1, max_=8, step=1,
               tooltip="DMD few-step model — 4 steps is the training target. "
                       "Clamped to [1, 8]."),
            _p("guidance_scale",      "float",    1.0,    "Guidance (fixed 1.0)",
               min_=1.0, max_=1.0, step=0.1,
               tooltip="Boogu Turbo is a DMD student model: no classifier-free "
                       "guidance. The pipeline requires 1.0 and the server "
                       "forces it."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=2, step=1,
               tooltip="Max 2 — each image stages the full pipeline on the "
                       "card under CPU offload."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
        ],
    ),

    "zimage": EngineInfo(
        key         = "zimage",
        label       = "Z-Image Turbo",
        description = (
            "Z-Image Turbo — Tongyi ~6B single-stream S3-DiT with a Qwen3-4B "
            "text encoder, distilled to 8 steps at CFG 0.0 (no negative "
            "prompt). Apache-2.0, plain-prompt friendly, strong EN/ZH "
            "in-image text (LongText-Bench 0.922). Runs GGUF-quantised "
            "transformer (jayn7/Z-Image-Turbo-GGUF); the Qwen3-4B encoder is "
            "loaded on demand for uncached prompts, then parked (embed "
            "cache)."
        ),
        output_type = "image",
        vram_gb     = 10.0,
        hf_repo     = "Tongyi-MAI/Z-Image-Turbo",
        hf_repo_alt = "jayn7/Z-Image-Turbo-GGUF",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate. Plain prompt "
                       "friendly — no JSON caption required.",
               required=True),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=16),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=16),
            _p("num_inference_steps", "int",      8,      "Steps",
               min_=1, max_=8, step=1,
               tooltip="Distilled turbo model — 8 steps is the training target."),
            _p("guidance_scale",      "float",    0.0,    "Guidance (fixed 0.0)",
               min_=0.0, max_=0.0, step=0.1,
               tooltip="Z-Image Turbo is distilled guidance-free — CFG is "
                       "disabled. The server forces 0.0."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=4, step=1),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "Q4_K_M", "Quantization",
               options=[
                   {"value": "Q3_K_S", "label": "Q3_K_S — ~3.5 GB transformer  fastest"},
                   {"value": "Q3_K_M", "label": "Q3_K_M — ~3.8 GB transformer"},
                   {"value": "Q4_K_S", "label": "Q4_K_S — ~4.3 GB transformer"},
                   {"value": "Q4_K_M", "label": "Q4_K_M — ~4.6 GB transformer  ✓ recommended"},
                   {"value": "Q5_K_S", "label": "Q5_K_S — ~4.8 GB transformer"},
                   {"value": "Q5_K_M", "label": "Q5_K_M — ~5.1 GB transformer"},
                   {"value": "Q6_K",   "label": "Q6_K — ~5.5 GB transformer"},
                   {"value": "Q8_0",   "label": "Q8_0 — ~6.7 GB transformer  best quality"},
               ],
               tooltip="GGUF quantisation via jayn7/Z-Image-Turbo-GGUF. Any tier "
                       "fits the 16 GB card; Q4_K_M is the quality/speed "
                       "sweet spot, Q8_0 the near-lossless ceiling."),
        ],
    ),

    "qwenimage": EngineInfo(
        key         = "qwenimage",
        label       = "Qwen-Image 2512",
        description = (
            "Qwen-Image-2512 — Alibaba ~20B MMDiT, Apache-2.0. Strongest "
            "independently-verified bilingual (EN/ZH) long-text rendering of "
            "any fully-open model (LongText-Bench 0.956 EN / 0.965 ZH). "
            "Runs a GGUF Q4_K_S transformer (~11.5 GB, unsloth) with the "
            "in-repo Qwen2.5-VL-family text encoder quantised for the 16 GB "
            "card; encoder loads on demand and parks after encoding (embed "
            "cache). Measured 2026-09-08: ~5 s/it → ~110 s at the 20-step "
            "default; Q4_K_S is the only tier that fits (Q4_K_M's load alone "
            "OOMs the card)."
        ),
        output_type = "image",
        vram_gb     = 13.8,  # measured gen process peak 14,136 MiB (2026-09-08)
        hf_repo     = "Qwen/Qwen-Image-2512",
        hf_repo_alt = "unsloth/Qwen-Image-2512-GGUF",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image. Dense bilingual layouts (posters, "
                       "slides, infographics) are this model's strength.",
               required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="CFG path — what to avoid in the image."),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=16),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=16),
            _p("num_inference_steps", "int",      20,     "Steps",
               min_=1, max_=50, step=1,
               tooltip="20 steps ≈ fast daily tier; up to 50 for max quality."),
            _p("guidance_scale",      "float",    4.0,    "Guidance scale",
               min_=1.0, max_=8.0, step=0.5),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=2, step=1,
               tooltip="Max 2 — a full generation runs the whole resident "
                       "transformer for ~1-3 min."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "Q4_K_S", "Quantization",
               options=[
                   {"value": "Q4_K_S", "label": "Q4_K_S — ~11.5 GB transformer  ✓ recommended"},
               ],
               tooltip="GGUF quantisation via unsloth/Qwen-Image-2512-GGUF. "
                       "Q4_K_S is the ONLY tier that fits the 16 GB card — "
                       "Q4_K_M (12.3 GB) was removed 2026-09-08: its load "
                       "alone OOMs (14.2 GiB process floor vs 14.28 GiB "
                       "usable with the TTS containers' CUDA contexts)."),
        ],
    ),

    "qwenimage-edit": EngineInfo(
        key         = "qwenimage-edit",
        label       = "Qwen-Image Edit",
        description = (
            "Qwen-Image-Edit-2511 — the instruction-following edit sibling "
            "of Qwen-Image-2512. Consumes a reference image NATIVELY (identity "
            "consistency + in-image text retention); the model card recommends "
            "40 steps, true_cfg_scale 4.0. Runs the unsloth GGUF Q4_K_S "
            "transformer (~11.56 GB) with the base engine's bnb-4bit "
            "Qwen2.5-VL encoder for the image-conditioned encode (vision "
            "tokens via the Qwen2VL processor; cache-miss encode parks the "
            "transformer). Measured 2026-09-08: 12.06 s/step at 1024² → ~4 min "
            "at the 20-step default; canvas ceiling ~1.05M px (1024² and "
            "720×1440 both verified); driver peak 15.40 GiB — the whole card."
        ),
        output_type = "image",
        image_input = "reference",  # edit pipeline natively conditions on the ref
        vram_gb     = 15.0,  # measured gen process peak 14.64 GiB driver (2026-09-08)
        hf_repo     = "Qwen/Qwen-Image-Edit-2511",
        hf_repo_alt = "unsloth/Qwen-Image-Edit-2511-GGUF",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Instruction to edit the reference image (identity, "
                       "pose, expression, background, headline text …). The "
                       "edit checkpoint applies it to the uploaded image — "
                       "describe the change, not a fresh scene.",
               required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="CFG path — what to avoid in the edited image. Also "
                       "image-conditioned (encoded against the reference)."),
            _p("reference_image",     "file",     None,   "Reference image (required)",
               tooltip="Consumed natively — the edit checkpoint conditions "
                       "on this image (identity + in-image text retention). "
                       "Upload the base image with every request.",
               required=True),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=16,
               tooltip="Canvas area is capped at ~1.05M px on the 16 GB card — "
                       "1024×1024 or 720×1440 both verified; larger canvases "
                       "exceed the measured VRAM ceiling."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=16,
               tooltip="Canvas area is capped at ~1.05M px on the 16 GB card — "
                       "1024×1024 or 720×1440 both verified; larger canvases "
                       "exceed the measured VRAM ceiling."),
            _p("num_inference_steps", "int",      20,     "Steps",
               min_=1, max_=50, step=1,
               tooltip="20 steps ≈ ~4 min per image; the model card "
                       "recommends 40 (~8 min) for max quality."),
            _p("guidance_scale",      "float",    4.0,    "Guidance scale",
               min_=1.0, max_=8.0, step=0.5,
               tooltip="Maps to the edit pipeline's true_cfg_scale — 4.0 is "
                       "the model-card setting."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=2, step=1,
               tooltip="Max 2 — each image runs ~4 min at the 20-step "
                       "default (12 s/step, measured)."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "Q4_K_S", "Quantization",
               options=[
                   {"value": "Q4_K_S", "label": "Q4_K_S — ~11.56 GB transformer  ✓ recommended"},
               ],
               tooltip="GGUF quantisation via unsloth/Qwen-Image-Edit-2511-GGUF. "
                       "Q4_K_S is the ONLY tier that fits the 16 GB card — "
                       "the edit pipeline's two-channel conditioning (vision "
                       "tokens + VAE ref latents) peaks at 15.40 GiB driver "
                       "at 1024²/20 st/CFG 4.0 (measured 2026-09-08)."),
        ],
    ),

    "hidream": EngineInfo(
        key         = "hidream",
        label       = "HiDream O1",
        description = (
            "HiDream-O1-Image-Dev — 8B pixel-space unified UiT (no VAE → no "
            "latent glyph blur). MIT. Runs OUT-OF-PROCESS via a headless "
            "ComfyUI sidecar on port 8188 (hidream_comfy_bridge.py). Dev "
            "checkpoint: fp8_scaled (~8.1 GB), 28 steps, cfg 1.0 (CFG-free "
            "with 7.6 noise scaling). The lab unloads its own resident "
            "engine first and hands VRAM back to ComfyUI after each "
            "generation."
        ),
        output_type = "image",
        vram_gb     = 11.5,
        hf_repo     = "HiDream-ai/HiDream-O1-Image-Dev",
        hf_repo_alt = "Comfy-Org/HiDream-O1-Image",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image. Long text prompts render at "
                       "pixel-level sharpness.",
               required=True),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=2048, step=64),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=2048, step=64),
            _p("num_inference_steps", "int",      28,     "Steps (fixed 28)",
               min_=28, max_=28, step=1,
               tooltip="Dev checkpoint — fixed 28-step sampler."),
            _p("guidance_scale",      "float",    1.0,    "Guidance (fixed 1.0)",
               min_=1.0, max_=1.0, step=0.1,
               tooltip="HiDream-O1-Dev is CFG-free — the ComfyUI graph runs "
                       "SamplerCustom at cfg 1.0 with 7.6 noise scaling "
                       "(ModelNoiseScale). The server forces 1.0."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=4, step=1),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
        ],
    ),

    "ernie": EngineInfo(
        key         = "ernie",
        label       = "ERNIE-Image",
        description = (
            "ERNIE-Image-Turbo — Baidu ~8B single-stream DiT + Ministral-3-3B "
            "text encoder, distilled to 8 steps at CFG 1.0. Apache-2.0. "
            "Poster/layout bilingual CN+EN text at low latency. Primary route "
            "is the pre-quantised Nunchaku-Lite NVFP4 transformer with a "
            "bnb-4bit text encoder (lite-infer repo); falls back to an fp8 / "
            "bf16 diffusers layout when the NVFP4 route can't load."
        ),
        output_type = "image",
        vram_gb     = 10.0,
        hf_repo     = "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder",
        hf_repo_alt = "baidu/ERNIE-Image-Turbo",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the poster / layout you want. CN+EN text is "
                       "the model's strength.",
               required=True),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Reference uploads are rejected: this checkpoint is text-only — "
                       "no native image conditioning exists."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=16),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=16),
            _p("num_inference_steps", "int",      8,      "Steps (fixed 8)",
               min_=8, max_=8, step=1,
               tooltip="Distilled turbo — fixed 8-step sampler."),
            _p("guidance_scale",      "float",    1.0,    "Guidance (fixed 1.0)",
               min_=1.0, max_=1.0, step=0.1,
               tooltip="ERNIE-Image-Turbo is distilled guidance-free — CFG is "
                       "forced 1.0. The server enforces it."),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=4, step=1),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
        ],
    ),
}

        # ---------------------------------------------------------------------------
        # Global runtime state  (mutated by engines + dispatch at runtime)
        # ---------------------------------------------------------------------------

class LabState:
    active_engine: Optional[str]  = None   # key of model currently in VRAM
    active_quant:  str            = ""     # quantization of the loaded model
    loaded_model:  Optional[Any]  = None   # the pipeline object
    loading:       bool           = False  # True while a load is in progress
    generating:    bool           = False  # True while generation runs
    last_used:     float          = 0.0   # time.time() of last generate call
    # Run-timing context for per-image stats: stamped by engines.generate()
    # around load+inference; save_image/save_video read them to record each
    # entry's started/finished timestamps and load/generation split.
    run_started:   float          = 0.0   # time.time() when a generate() run began
    run_loaded_at: float          = 0.0   # time.time() when the model finished loading

STATE = LabState()
