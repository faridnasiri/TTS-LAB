"""
image_lab_engines.py — Load / unload / generate functions for all three engines:
  flux2  — FLUX.2 [dev] GGUF (Q3_K_M / Q4_K_M / Q5_K_M / Q8_0) via city96
  sd35   — Stable Diffusion 3.5 Large GGUF (Q4_0 / Q5_0 / Q8_0) via city96
  wan    — Wan2.2 T2V / I2V GGUF (Q3_K_M / Q4_K_M / Q5_K_M / Q8_0) via QuantStack

GGUF files are downloaded from HuggingFace on first use and cached under GGUF_ROOT.
Non-transformer pipeline components (text encoders, VAE, scheduler, tokenizers)
are loaded from the pre-saved shared directories written by preq_save.py.
"""

from __future__ import annotations
import logging
import os
import time
from typing import Any, Optional

from image_lab_config import ENGINES, STATE, HF_TOKEN, HF_HOME
from image_lab_utils import free_vram, random_seed, save_image, save_images, save_video

# Local directory for cached GGUF model files
GGUF_ROOT = "/opt/arthur-img-models/gguf"

# Pre-saved shared pipeline components (text encoders, VAE, configs)
# These were written by preq_save.py and contain everything except the transformer.
PREQ_ROOT = "/opt/arthur-img-models/quantized"

# NVFP4-quantized transformers saved by nvfp4_save.py (torchao NVFP4WeightOnlyConfig)
NVFP4_ROOT = "/opt/arthur-img-models/nvfp4"

# ---------------------------------------------------------------------------
# GGUF file catalogue
# ---------------------------------------------------------------------------

# (repo_id, filename_in_repo)  — for flat-layout repos (FLUX.2, SD35)
_FLUX2_GGUF: dict[str, tuple[str, str]] = {
    "Q3_K_M": ("city96/FLUX.2-dev-gguf", "flux2-dev-Q3_K_M.gguf"),
    "Q4_K_M": ("city96/FLUX.2-dev-gguf", "flux2-dev-Q4_K_M.gguf"),
    "Q5_K_M": ("city96/FLUX.2-dev-gguf", "flux2-dev-Q5_K_M.gguf"),
    "Q8_0":   ("city96/FLUX.2-dev-gguf", "flux2-dev-Q8_0.gguf"),
}
_SD35_GGUF: dict[str, tuple[str, str]] = {
    "Q4_0": ("city96/stable-diffusion-3.5-large-gguf", "sd3.5_large-Q4_0.gguf"),
    "Q5_0": ("city96/stable-diffusion-3.5-large-gguf", "sd3.5_large-Q5_0.gguf"),
    "Q8_0": ("city96/stable-diffusion-3.5-large-gguf", "sd3.5_large-Q8_0.gguf"),
}

# Wan has HighNoise (=transformer) and LowNoise (=transformer_2) sub-directories
def _wan_gguf(variant: str, noise: str, quant: str) -> tuple[str, str]:
    """Return (repo_id, filename_in_repo) for a Wan GGUF file."""
    # variant: "t2v" | "i2v"    noise: "HighNoise" | "LowNoise"
    tag = "T2V" if variant == "t2v" else "I2V"
    repo = f"QuantStack/Wan2.2-{tag}-A14B-GGUF"
    fname = f"Wan2.2-{tag}-A14B-{noise}-{quant}.gguf"
    return (repo, f"{noise}/{fname}")

log = logging.getLogger("image_lab")

# ---------------------------------------------------------------------------
# GGUF download helper
# ---------------------------------------------------------------------------

def _ensure_gguf(repo_id: str, filename_in_repo: str, local_dir: str) -> str:
    """
    Return the local path to a GGUF file.  If not present, downloads it from
    HuggingFace Hub into `local_dir` (preserving any sub-folder in the name).
    """
    # filename_in_repo may include a sub-folder, e.g. "HighNoise/Wan2.2-...gguf"
    local_path = os.path.join(local_dir, filename_in_repo)
    if os.path.isfile(local_path):
        log.info("GGUF cached locally: %s", local_path)
        return local_path

    log.info("Downloading GGUF %s/%s → %s …", repo_id, filename_in_repo, local_dir)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id   = repo_id,
        filename  = filename_in_repo,
        local_dir = local_dir,
        local_dir_use_symlinks = False,
        token     = HF_TOKEN or None,
    )
    log.info("GGUF downloaded: %s", downloaded)
    return downloaded


def _gguf_quant_config(dtype=None):
    """Return a GGUFQuantizationConfig, importing from wherever diffusers exposes it."""
    import torch
    compute_dtype = dtype or torch.bfloat16
    try:
        from diffusers import GGUFQuantizationConfig
    except ImportError:
        from diffusers.quantizers.gguf import GGUFQuantizationConfig
    return GGUFQuantizationConfig(compute_dtype=compute_dtype)


def _load_nvfp4_transformer(model_key: str, subfolder: str):
    """
    Load a pre-saved NVFP4-quantized transformer from disk.
    The transformer must have been saved by nvfp4_save.py first.
    `model_key`  — e.g. "flux2", "sd35", "wan-t2v", "wan-i2v"
    `subfolder`  — "transformer" or "transformer_2"
    """
    import torch
    from diffusers import AutoModel

    path = os.path.join(NVFP4_ROOT, model_key, subfolder)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(
            f"NVFP4 transformer not found at {path}.\n"
            f"Run nvfp4_save.py first to download and quantize it."
        )
    log.info("Loading NVFP4 transformer from %s …", path)
    return AutoModel.from_pretrained(
        path,
        torch_dtype     = torch.bfloat16,
        use_safetensors = False,
    )


# ---------------------------------------------------------------------------
# VRAM lifecycle helpers
# ---------------------------------------------------------------------------

def _unload_current():
    """Destroy the currently-loaded pipeline and free VRAM."""
    if STATE.active_engine is None:
        return
    log.info("Unloading engine: %s (quant=%s)", STATE.active_engine, STATE.active_quant)
    STATE.loaded_model  = None
    STATE.loaded_pipe2  = None
    STATE.active_engine = None
    STATE.active_quant  = ""
    free_vram()
    ENGINES["flux2"].loaded      = False
    ENGINES["flux2klein"].loaded = False
    ENGINES["sd35"].loaded       = False
    ENGINES["wan"].loaded        = False


def _ensure_engine(key: str, quant: str = ""):
    """Load engine `key` with `quant` into VRAM, evicting whatever is currently loaded."""
    if STATE.active_engine == key and STATE.active_quant == quant:
        return  # already loaded with the same quantization
    _unload_current()
    loader = _LOADERS.get(key)
    if loader is None:
        raise RuntimeError(f"No loader for engine '{key}'")
    STATE.loading = True
    try:
        loader(quant)
    finally:
        STATE.loading = False

# ---------------------------------------------------------------------------
# FLUX.2 [dev]
# ---------------------------------------------------------------------------

def _load_flux2(quant: str = "Q4_K_M"):
    import torch
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    from transformers import BitsAndBytesConfig as TrfBnBConfig

    quant = quant or "Q4_K_M"
    t0    = time.time()
    token = HF_TOKEN or True

    if quant == "nvfp4":
        transformer = _load_nvfp4_transformer("flux2", "transformer")
    else:
        if quant not in _FLUX2_GGUF:
            raise RuntimeError(
                f"FLUX.2 quant '{quant}' not recognised. "
                f"Valid options: {list(_FLUX2_GGUF)} + ['nvfp4']"
            )
        repo_id, fname = _FLUX2_GGUF[quant]
        gguf_path = _ensure_gguf(repo_id, fname, os.path.join(GGUF_ROOT, "flux2"))
        log.info("Loading FLUX.2 [dev] transformer from GGUF — quant=%s …", quant)
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config = _gguf_quant_config(),
            torch_dtype         = torch.bfloat16,
        )

    # Load the text encoder (Mistral 3.1 8B) in 4-bit NF4 to keep RAM under 31 GB.
    # Without this, the text encoder alone uses ~15 GB BF16; with NF4 it's ~4 GB.
    bnb_4bit = TrfBnBConfig(
        load_in_4bit           = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_quant_type    = "nf4",
    )

    # Load the rest of the pipeline (text encoder, VAE, scheduler) from the
    # cached BnB repo — only the transformer is swapped out for our GGUF version.
    log.info("Loading FLUX.2 [dev] pipeline (quant=%s) …", quant)
    pipe = Flux2Pipeline.from_pretrained(
        ENGINES["flux2"].hf_repo,   # diffusers/FLUX.2-dev-bnb-4bit (already in HF cache)
        transformer          = transformer,
        torch_dtype          = torch.bfloat16,
        quantization_config  = bnb_4bit,
        token                = token,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    STATE.loaded_model  = pipe
    STATE.active_engine = "flux2"
    STATE.active_quant  = quant
    ENGINES["flux2"].loaded = True
    log.info("FLUX.2 [dev] ready (quant=%s) in %.1f s", quant, time.time() - t0)


def _generate_flux2(params: dict) -> list[dict]:
    import torch

    pipe  = STATE.loaded_model
    seed  = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    # Use CPU generator — with device_map the pipeline manages device placement
    generator = torch.Generator("cpu").manual_seed(seed)

    result = pipe(
        prompt              = params["prompt"],
        image               = _load_ref_image(params.get("reference_image")),
        width               = int(params.get("width",  1024)),
        height              = int(params.get("height", 1024)),
        num_inference_steps = int(params.get("num_inference_steps", 28)),
        guidance_scale      = float(params.get("guidance_scale", 4.0)),
        generator           = generator,
    )

    final_params = {**params, "seed": seed}
    return save_images(result.images, "flux2", final_params)


# ---------------------------------------------------------------------------
# FLUX.2 Klein 4B
# ---------------------------------------------------------------------------

def _load_flux2klein(quant: str = ""):
    import torch
    from diffusers import Flux2KleinPipeline

    t0    = time.time()
    token = HF_TOKEN or True

    log.info("Loading FLUX.2 Klein 4B from HuggingFace (%s) …",
             ENGINES["flux2klein"].hf_repo)
    pipe = Flux2KleinPipeline.from_pretrained(
        ENGINES["flux2klein"].hf_repo,   # black-forest-labs/FLUX.2-klein-4B
        torch_dtype = torch.bfloat16,
        token       = token,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()

    STATE.loaded_model       = pipe
    STATE.active_engine      = "flux2klein"
    STATE.active_quant       = ""
    ENGINES["flux2klein"].loaded = True
    log.info("FLUX.2 Klein 4B ready in %.1f s", time.time() - t0)


def _generate_flux2klein(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    generator = torch.Generator("cpu").manual_seed(seed)

    result = pipe(
        prompt              = params["prompt"],
        image               = _load_ref_image(params.get("reference_image")),
        width               = int(params.get("width",  1024)),
        height              = int(params.get("height", 1024)),
        num_inference_steps = int(params.get("num_inference_steps", 20)),
        guidance_scale      = float(params.get("guidance_scale", 3.5)),
        generator           = generator,
    )

    final_params = {**params, "seed": seed}
    return save_images(result.images, "flux2klein", final_params)


# ---------------------------------------------------------------------------
# Stable Diffusion 3.5 Large
# ---------------------------------------------------------------------------

def _load_sd35(quant: str = "Q4_0"):
    import torch
    from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

    quant = quant or "Q4_0"
    shared_path = f"{PREQ_ROOT}/sd35/shared"

    if not os.path.isdir(shared_path):
        raise RuntimeError(
            f"SD 3.5 shared pipeline components not found at: {shared_path}\n"
            f"Run preq_save.py first to create this directory."
        )

    t0 = time.time()

    if quant == "nvfp4":
        transformer = _load_nvfp4_transformer("sd35", "transformer")
    else:
        if quant not in _SD35_GGUF:
            raise RuntimeError(
                f"SD 3.5 quant '{quant}' not recognised. "
                f"Valid options: {list(_SD35_GGUF)} + ['nvfp4']"
            )
        repo_id, fname = _SD35_GGUF[quant]
        gguf_path = _ensure_gguf(repo_id, fname, os.path.join(GGUF_ROOT, "sd35"))
        log.info("Loading SD 3.5 Large transformer from GGUF — quant=%s …", quant)
        transformer = SD3Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config = _gguf_quant_config(),
            torch_dtype         = torch.bfloat16,
        )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        shared_path,
        transformer = transformer,
        torch_dtype = torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()

    STATE.loaded_model  = pipe
    STATE.active_engine = "sd35"
    STATE.active_quant  = quant
    ENGINES["sd35"].loaded = True
    log.info("SD 3.5 Large ready (quant=%s) in %.1f s", quant, time.time() - t0)


def _generate_sd35(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    n = int(params.get("num_images", 1))
    generator = [
        torch.Generator(device="cpu").manual_seed(seed + i)
        for i in range(n)
    ]

    result = pipe(
        prompt              = params["prompt"],
        negative_prompt     = params.get("negative_prompt", ""),
        width               = int(params.get("width",  1024)),
        height              = int(params.get("height", 1024)),
        num_inference_steps = int(params.get("num_inference_steps", 28)),
        guidance_scale      = float(params.get("guidance_scale", 4.5)),
        num_images_per_prompt = n,
        generator           = generator,
    )

    final_params = {**params, "seed": seed}
    return save_images(result.images, "sd35", final_params)


# ---------------------------------------------------------------------------
# Wan2.2  (T2V + I2V)
# ---------------------------------------------------------------------------

_WAN_VALID_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0")


def _load_wan(quant: str = "Q4_K_M"):
    import torch
    from diffusers import WanPipeline, WanImageToVideoPipeline
    try:
        from diffusers import WanTransformer3DModel
    except ImportError:
        from diffusers.models import WanTransformer3DModel

    quant = quant or "Q4_K_M"
    log.info("Loading Wan2.2 (%s) — quant=%s …", "NVFP4" if quant == "nvfp4" else "GGUF", quant)
    t0 = time.time()

    # Inner helper — only defined (and used) for GGUF paths
    def _load_wan_gguf_transformer(variant: str, noise: str) -> Any:
        repo_id, fname_in_repo = _wan_gguf(variant, noise, quant)
        gguf_path = _ensure_gguf(
            repo_id, fname_in_repo,
            os.path.join(GGUF_ROOT, f"wan-{variant}"),
        )
        log.info("  Loading Wan %s %s transformer from %s …", variant.upper(), noise, gguf_path)
        return WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config = _gguf_quant_config(),
            torch_dtype         = torch.bfloat16,
        )

    if quant == "nvfp4":
        t2v_tf  = _load_nvfp4_transformer("wan-t2v", "transformer")
        t2v_tf2 = _load_nvfp4_transformer("wan-t2v", "transformer_2")
    else:
        if quant not in _WAN_VALID_QUANTS:
            raise RuntimeError(
                f"Wan quant '{quant}' not recognised. "
                f"Valid options: {_WAN_VALID_QUANTS} + ['nvfp4']"
            )
        t2v_tf  = _load_wan_gguf_transformer("t2v", "HighNoise")
        t2v_tf2 = _load_wan_gguf_transformer("t2v", "LowNoise")

    # T2V pipeline — HighNoise = transformer, LowNoise = transformer_2
    t2v_shared = f"{PREQ_ROOT}/wan-t2v/shared"
    if not os.path.isdir(t2v_shared):
        raise RuntimeError(
            f"Wan T2V shared pipeline components not found at: {t2v_shared}\n"
            f"Run preq_save.py first to create this directory."
        )
    pipe_t2v = WanPipeline.from_pretrained(
        t2v_shared,
        transformer   = t2v_tf,
        transformer_2 = t2v_tf2,
        torch_dtype   = torch.bfloat16,
    )
    pipe_t2v.enable_model_cpu_offload()
    pipe_t2v.vae.enable_slicing()

    # I2V pipeline — same structure, separate weights
    pipe_i2v = None
    i2v_shared = f"{PREQ_ROOT}/wan-i2v/shared"
    try:
        if quant == "nvfp4":
            i2v_tf  = _load_nvfp4_transformer("wan-i2v", "transformer")
            i2v_tf2 = _load_nvfp4_transformer("wan-i2v", "transformer_2")
        else:
            i2v_tf  = _load_wan_gguf_transformer("i2v", "HighNoise")
            i2v_tf2 = _load_wan_gguf_transformer("i2v", "LowNoise")
        pipe_i2v = WanImageToVideoPipeline.from_pretrained(
            i2v_shared,
            transformer   = i2v_tf,
            transformer_2 = i2v_tf2,
            torch_dtype   = torch.bfloat16,
        )
        pipe_i2v.enable_model_cpu_offload()
        pipe_i2v.vae.enable_slicing()
    except Exception as exc:
        log.warning("Wan I2V load failed (T2V still available): %s", exc)

    STATE.loaded_model  = pipe_t2v
    STATE.loaded_pipe2  = pipe_i2v
    STATE.active_engine = "wan"
    STATE.active_quant  = quant
    ENGINES["wan"].loaded = True
    log.info("Wan2.2 ready (quant=%s) in %.1f s", quant, time.time() - t0)


def _generate_wan(params: dict) -> list[dict]:
    import torch
    from diffusers.utils import export_to_video

    mode = params.get("mode", "t2v")
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    fps       = int(params.get("fps", 16))
    n_frames  = int(params.get("num_frames", 49))
    res_str   = params.get("resolution", "720p")
    width, height = (1280, 720) if res_str == "720p" else (854, 480)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    if mode == "i2v" and STATE.loaded_pipe2 is not None:
        pipe = STATE.loaded_pipe2
        ref  = _load_ref_image(params.get("reference_image"))
        if ref is None:
            raise ValueError("I2V mode requires a reference_image upload.")
        ref_resized = ref.resize((width, height))
        output = pipe(
            image               = ref_resized,
            prompt              = params["prompt"],
            negative_prompt     = params.get("negative_prompt", ""),
            num_frames          = n_frames,
            guidance_scale      = float(params.get("guidance_scale", 5.0)),
            generator           = generator,
        )
    else:
        pipe   = STATE.loaded_model
        output = pipe(
            prompt              = params["prompt"],
            negative_prompt     = params.get("negative_prompt", ""),
            height              = height,
            width               = width,
            num_frames          = n_frames,
            guidance_scale      = float(params.get("guidance_scale", 5.0)),
            generator           = generator,
        )

    frames      = output.frames[0]
    final_params = {**params, "seed": seed, "fps": fps,
                    "width": width, "height": height}
    entry = save_video(frames, fps, "wan", final_params)
    return [entry]


# ---------------------------------------------------------------------------
# Availability probe (called at startup)
# ---------------------------------------------------------------------------

def probe_availability():
    """
    Check which engines can be loaded (packages importable, not that models
    are downloaded — that happens lazily on first generate call).
    """
    _probe_flux2()
    _probe_flux2klein()
    _probe_sd35()
    _probe_wan()


def _probe_flux2():
    try:
        from diffusers import Flux2Pipeline      # noqa: F401
        import bitsandbytes                      # noqa: F401
        import requests                          # noqa: F401
        if not HF_TOKEN:
            from huggingface_hub import get_token
            tok = get_token()
            if not tok:
                raise RuntimeError("No HF_TOKEN and no cached HF token found")
        ENGINES["flux2"].available = True
    except Exception as exc:
        ENGINES["flux2"].available = False
        ENGINES["flux2"].error     = str(exc)
        log.warning("FLUX.2 [dev] unavailable: %s", exc)


def _probe_flux2klein():
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
        ENGINES["flux2klein"].available = True
    except Exception as exc:
        ENGINES["flux2klein"].available = False
        ENGINES["flux2klein"].error     = str(exc)
        log.warning("FLUX.2 Klein 4B unavailable: %s", exc)


def _probe_sd35():
    try:
        from diffusers import StableDiffusion3Pipeline  # noqa: F401
        ENGINES["sd35"].available = True
    except Exception as exc:
        ENGINES["sd35"].available = False
        ENGINES["sd35"].error     = str(exc)
        log.warning("SD 3.5 Large unavailable: %s", exc)


def _probe_wan():
    try:
        from diffusers import WanPipeline  # noqa: F401
        import imageio                     # noqa: F401
        ENGINES["wan"].available = True
    except Exception as exc:
        ENGINES["wan"].available = False
        ENGINES["wan"].error     = str(exc)
        log.warning("Wan2.2 unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Public generate dispatcher
# ---------------------------------------------------------------------------

def generate(engine_key: str, params: dict) -> list[dict]:
    """
    Load `engine_key` into VRAM (evicting current if needed) and generate.
    Returns a list of result dicts (always a list; images may have multiple).
    """
    if engine_key not in ENGINES:
        raise ValueError(f"Unknown engine: {engine_key}")
    if not ENGINES[engine_key].available:
        raise RuntimeError(
            f"Engine '{engine_key}' is not available: {ENGINES[engine_key].error}"
        )
    if STATE.generating:
        raise RuntimeError("Another generation is already in progress.")

    quant = params.get("quant", "")
    STATE.generating = True
    try:
        _ensure_engine(engine_key, quant)
        generator_fn = _GENERATORS[engine_key]
        results = generator_fn(params)
        STATE.last_used = time.time()
        return results
    finally:
        STATE.generating = False


# ---------------------------------------------------------------------------
# Public load / unload for the API
# ---------------------------------------------------------------------------

def load_engine(engine_key: str):
    _ensure_engine(engine_key)


def unload_engine():
    _unload_current()


# ---------------------------------------------------------------------------
# Helper — load a reference image from bytes or path
# ---------------------------------------------------------------------------

def _load_ref_image(ref) -> Optional[Any]:
    if ref is None:
        return None
    from PIL import Image
    if isinstance(ref, bytes):
        import io as _io
        return Image.open(_io.BytesIO(ref)).convert("RGB")
    if isinstance(ref, str) and os.path.exists(ref):
        return Image.open(ref).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_LOADERS = {
    "flux2":      _load_flux2,
    "flux2klein": _load_flux2klein,
    "sd35":       _load_sd35,
    "wan":        _load_wan,
}

_GENERATORS = {
    "flux2":      _generate_flux2,
    "flux2klein": _generate_flux2klein,
    "sd35":       _generate_sd35,
    "wan":        _generate_wan,
}
