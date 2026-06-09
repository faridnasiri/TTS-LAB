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
    key: str                          # API key used in URL  e.g. "flux2"
    label: str                        # Human label
    description: str
    output_type: str                  # "image" | "video"
    vram_gb: float                    # Estimated VRAM when loaded
    hf_repo: str                      # Primary HuggingFace repo
    hf_repo_alt: Optional[str]        # Secondary repo (e.g. I2V variant)
    params: list[dict]                # Parameter schema for the UI
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

    "flux2": EngineInfo(
        key         = "flux2",
        label       = "FLUX.2 [dev]",
        description = (
            "32B rectified flow transformer. State-of-the-art text-to-image and "
            "image editing (reference image supported). GGUF quantised transformer "
            "loaded from city96/FLUX.2-dev-gguf. Non-transformer components "
            "(T5 encoder, VAE) reused from the pre-quantised BnB NF4 cache."
        ),
        output_type = "image",
        vram_gb     = 16.0,
        hf_repo     = "diffusers/FLUX.2-dev-bnb-4bit",
        hf_repo_alt = None,
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate.", required=True),
            _p("reference_image",     "file",     None,   "Reference image (optional)",
               tooltip="Upload a reference image for style/content transfer (I2I mode)."),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output width in pixels. Must be a multiple of 64."),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=2048, step=64,
               tooltip="Output height in pixels. Must be a multiple of 64."),
            _p("num_inference_steps", "int",      28,     "Steps",
               min_=1, max_=50, step=1,
               tooltip="28 is a good trade-off; fewer = faster but lower quality."),
            _p("guidance_scale",      "float",    3.5,    "Guidance scale",
               min_=1.0, max_=20.0, step=0.5,
               tooltip="How strongly the model follows the prompt. 3.5–4.0 is typical."),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1,
               tooltip="Fixed seed for reproducible results."),
            _p("quant",               "select",   "Q4_K_M", "Quantization",
               options=[
                   {"value": "Q3_K_M", "label": "Q3_K_M — 16 GB transformer  (smallest GGUF)"},
                   {"value": "Q4_K_M", "label": "Q4_K_M — 20 GB transformer  ✓ recommended GGUF"},
                   {"value": "Q5_K_M", "label": "Q5_K_M — 24 GB transformer  (higher quality GGUF)"},
                   {"value": "Q8_0",   "label": "Q8_0   — 35 GB transformer  (near-lossless GGUF)"},
                   {"value": "nvfp4",  "label": "NVFP4  — ~8 GB transformer  ⚡ Blackwell native (run nvfp4_save.py first)"},
               ],
               tooltip=(
                   "GGUF quantisation uses city96/FLUX.2-dev-gguf, downloaded on first use. "
                   "NVFP4 uses torchao NVFP4WeightOnlyConfig baked from BF16 by nvfp4_save.py — "
                   "fastest on RTX 5060 Ti (Blackwell SM100+). Q4_K_M is best GGUF trade-off."
               )),
        ],
    ),

    "sd35": EngineInfo(
        key         = "sd35",
        label       = "SD 3.5 Large",
        description = (
            "Stable Diffusion 3.5 Large — 8B MMDiT text-to-image model. "
            "GGUF quantised transformer (city96/stable-diffusion-3.5-large-gguf). "
            "Text encoders and VAE reused from pre-saved shared directory on disk. "
            "Q4_0 uses ~5 GB VRAM for the transformer alone. "
            "Supports speed-presets for turbo / lightning fast runs on the same model weights."
        ),
        output_type = "image",
        vram_gb     = 12.0,
        hf_repo     = "stabilityai/stable-diffusion-3.5-large",
        hf_repo_alt = None,
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the image you want to generate.", required=True),
            _p("negative_prompt",     "textarea", "",     "Negative prompt",
               tooltip="Describe what you do NOT want in the image."),
            _p("speed_preset",        "select",   "standard", "Mode",
               options=[
                   {"value": "standard",  "label": "Standard — 28 steps (default quality)"},
                   {"value": "turbo",     "label": "Turbo — 4 steps (fast preview)"},
                   {"value": "lightning", "label": "Lightning — 8 steps (fast)"},
               ],
               tooltip="Choose a speed/quality profile for SD 3.5 Large. Turbo and Lightning use fewer steps on the same model weights.",
               client_only=True),
            _p("width",               "int",      1024,   "Width (px)",
               min_=256, max_=1536, step=64),
            _p("height",              "int",      1024,   "Height (px)",
               min_=256, max_=1536, step=64),
            _p("num_inference_steps", "int",      28,     "Steps",
               min_=1, max_=100, step=1),
            _p("guidance_scale",      "float",    4.5,    "Guidance scale",
               min_=1.0, max_=20.0, step=0.5),
            _p("num_images",          "int",      1,      "Images per request",
               min_=1, max_=4, step=1),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "Q4_0", "Quantization",
               options=[
                   {"value": "Q4_0",  "label": "Q4_0  — 4.8 GB transformer  ✓ recommended GGUF"},
                   {"value": "Q5_0",  "label": "Q5_0  — 5.8 GB transformer  (higher quality GGUF)"},
                   {"value": "Q8_0",  "label": "Q8_0  — 8.8 GB transformer  (near-lossless GGUF)"},
                   {"value": "nvfp4", "label": "NVFP4 — ~2 GB transformer  ⚡ Blackwell native (run nvfp4_save.py first)"},
               ],
               tooltip=(
                   "GGUF quantisation uses city96/stable-diffusion-3.5-large-gguf, downloaded on first use. "
                   "NVFP4 uses torchao NVFP4WeightOnlyConfig baked from BF16 by nvfp4_save.py — "
                   "fastest on RTX 5060 Ti (Blackwell SM100+)."
               )),
        ],
    ),

    "flux2klein": EngineInfo(
        key         = "flux2klein",
        label       = "FLUX.2 Klein 4B",
        description = (
            "FLUX.2 Klein 4B — compact 4B flow transformer from Black Forest Labs. "
            "Apache 2.0 license. Uses Qwen3 text encoder (far smaller than the Mistral 24B "
            "in FLUX.2-dev). Fits in ~13 GB VRAM at BF16 — runs on our RTX 5060 Ti. "
            "Loaded directly from HuggingFace (no pre-saved shared dir needed). "
            "Supports text-to-image and image-to-image (reference image)."
        ),
        output_type = "image",
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

    "wan": EngineInfo(
        key         = "wan",
        label       = "Wan2.2",
        description = (
            "Wan2.2 text-to-video and image-to-video model from Alibaba. "
            "T2V-A14B generates up to 5 s of cinematic video from a text prompt. "
            "I2V-A14B animates a reference image. Uses two GGUF-quantised transformers "
            "(HighNoise + LowNoise) from QuantStack, loaded with model_cpu_offload."
        ),
        output_type = "video",
        vram_gb     = 14.0,
        hf_repo     = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        hf_repo_alt = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        params      = [
            _p("prompt",              "textarea", "",     "Prompt",
               tooltip="Describe the motion and scene you want to generate.", required=True),
            _p("negative_prompt",     "textarea",
               "low quality, blurry, distorted", "Negative prompt"),
            _p("mode",                "select",   "t2v",  "Mode",
               options=["t2v", "i2v"],
               tooltip="t2v = text-to-video | i2v = image-to-video (requires reference image)."),
            _p("reference_image",     "file",     None,   "Reference image (I2V mode)",
               tooltip="Required when mode=i2v. First frame to animate."),
            _p("num_frames",          "int",      49,     "Frames",
               min_=16, max_=120, step=8,
               tooltip="Number of video frames. At 16 fps, 49 frames ≈ 3 s."),
            _p("fps",                 "int",      16,     "FPS",
               min_=8, max_=24, step=1),
            _p("resolution",          "select",   "720p", "Resolution",
               options=["480p", "720p"]),
            _p("seed",                "int",      -1,     "Seed (-1 = random)",
               min_=-1, max_=2**31-1, step=1),
            _p("quant",               "select",   "Q4_K_M", "Quantization",
               options=[
                   {"value": "Q3_K_M", "label": "Q3_K_M — 7.2 GB × 2 transformers  (smallest GGUF)"},
                   {"value": "Q4_K_M", "label": "Q4_K_M — 9.7 GB × 2 transformers  ✓ recommended GGUF"},
                   {"value": "Q5_K_M", "label": "Q5_K_M — 10.8 GB × 2 transformers  (higher quality GGUF)"},
                   {"value": "Q8_0",   "label": "Q8_0   — 15.4 GB × 2 transformers  (near-lossless GGUF)"},
                   {"value": "nvfp4",  "label": "NVFP4  — ~4 GB × 2 transformers  ⚡ Blackwell native (run nvfp4_save.py first)"},
               ],
               tooltip=(
                   "GGUF quantisation uses QuantStack repos, downloaded on first use. "
                   "NVFP4 uses torchao NVFP4WeightOnlyConfig baked from BF16 by nvfp4_save.py — "
                   "fastest on RTX 5060 Ti (Blackwell SM100+). Both HighNoise + LowNoise transformers quantized."
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
                    _p("prompt",                "textarea", "",     "Caption (JSON)",
                       required=True,
                       tooltip="Structured JSON caption. Use Magic Prompt to auto-generate from plain text."),
                    _p("use_magic_prompt",      "checkbox", False,  "Magic Prompt (auto-expand plain text)",
                       tooltip="Enable to expand a plain-text idea into a JSON caption via OpenRouter."),
                    _p("magic_prompt_input",    "textarea", "",     "Plain idea (for magic prompt)",
                       tooltip="Write a plain idea here if magic prompt is enabled."),
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
        }

        # ---------------------------------------------------------------------------
        # Global runtime state  (mutated by engines + dispatch at runtime)
        # ---------------------------------------------------------------------------

class LabState:
    active_engine: Optional[str]  = None   # key of model currently in VRAM
    active_quant:  str            = ""     # quantization of the loaded model
    loaded_model:  Optional[Any]  = None   # the pipeline object
    loaded_pipe2:  Optional[Any]  = None   # second pipeline (Wan I2V variant)
    loading:       bool           = False  # True while a load is in progress
    generating:    bool           = False  # True while generation runs
    last_used:     float          = 0.0   # time.time() of last generate call

STATE = LabState()
