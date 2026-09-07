"""
boogu_lab_engine.py — Boogu-Image-0.1-Turbo-fp8 engine for Arthur Image Lab.

Integrates the upstream `boogu` package (boogu-project/Boogu-Image, installed
as an editable `--no-deps` package at /opt/arthur-img/Boogu-Image) with the
diffusers-layout fp8 checkpoint from HuggingFace.

What the "fp8" repo actually is (verified 2026-09-06/07):
  * mllm/      — Qwen3-VL-8B-class instruction encoder, fp8 weights
                 (transformers finegrained-fp8 quant_method config).
  * transformer/ — custom BooguImageTransformer2DModel, ~10.3 GB of .bin
                 shards stored BF16 — the fp8 name applies to the mllm plus
                 a runtime deepgemm toggle, NOT to the DiT weights.
  * vae/       — FLUX.1 VAE. scheduler/ + processor/ configs; the custom
                 scheduler/pipeline classes live in the installed boogu
                 package (the repo ships no pipeline_*.py files).

CPU-offload EXCEPTION (user-approved 2026-09-06 — the only engine with one):
the vendor's 16 GB guidance is fp8 + `enable_model_cpu_offload`. Loading both
the ~10.3 GB bf16 DiT and the ~10.6 GB fp8 mllm on the card at once cannot
fit 16 GB with room to generate, so this engine runs the pipeline with model
CPU offload: nothing stays resident between requests and each generate
stages one module on the card at a time (accelerate offload hooks). Host RAM
peak is ~20-21 GB (62 GB box). Every other engine honours IMGLAB_GPU_ONLY;
this one is the sanctioned exception.

DMD few-step (turbo) path: 4 steps, NO CFG — the pipeline requires
text_guidance_scale == image_guidance_scale == 1.0 and
empty_instruction_guidance_scale == 0.0, and raises otherwise. The loader
forces these; any caller-supplied guidance is clamped to 1.0.
NEVER call torch.compile on this pipeline — upstream documents
occasional all-black outputs (test_turbo_fp8.sh).

Usage:
    from boogu_lab_engine import load_boogu, generate_boogu, probe_boogu
    pipeline = load_boogu()
    images, seed_used = generate_boogu(pipeline, params)
"""

from __future__ import annotations
import logging
import os
import random
import time
from typing import Any, Optional

log = logging.getLogger("image_lab")

# Single checkpoint — the "quant" machinery rides with "" (see image_lab_config).
BOOGU_REPO = "Boogu/Boogu-Image-0.1-Turbo-fp8"

# Import paths resolved 2026-09-07 from the installed boogu package (the
# package's top-level __init__.py is empty — classes live in submodules).
_TURBO_MODULE      = "boogu.pipelines.boogu.pipeline_boogu_turbo"
_TRANSFORMER_MODULE = "boogu.models.transformers.transformer_boogu"


def _disable_deepgemm_for_fp8_vlm() -> None:
    """Force the Triton finegrained-fp8 fallback for the fp8 mllm.

    Mirrors the upstream `_disable_deepgemm_for_fp8_vlm()` helper in
    inference_turbo.py: newer transformers integrate DeepGEMM for fp8 linear
    layers, but the DeepGEMM kernel (sm100+) build is not present on this VM,
    so without the patch the fp8 mllm load fails. The env var is set first as
    belt-and-braces (transformers reads it if the integration has not been
    imported yet); the monkeypatch covers the case where transformers'
    finegrained_fp8 module was already imported by an earlier engine.
    """
    os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "DeepGEMM disabled for fp8 VLM — forcing the Triton "
            "finegrained-fp8 fallback."
        )

    try:
        import transformers.integrations.finegrained_fp8 as fg_fp8
    except ImportError:
        log.warning(
            "transformers.integrations.finegrained_fp8 not present — "
            "deepgemm patch skipped (transformers %s)",
            __import__("transformers").__version__,
        )
        return
    if hasattr(fg_fp8, "deepgemm_fp8_fp4_linear"):
        fg_fp8.deepgemm_fp8_fp4_linear = _raise_import_error
    elif hasattr(fg_fp8, "_load_deepgemm_kernel"):
        fg_fp8._load_deepgemm_kernel = _raise_import_error
    else:
        log.info("finegrained_fp8 exposes no deepgemm hook — nothing to patch")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_boogu(quant: str = ""):
    """Load the Boogu turbo fp8 pipeline (CPU-offloaded — the sanctioned
    exception to the lab's GPU-only policy; see the module docstring)."""
    import torch

    _disable_deepgemm_for_fp8_vlm()

    from boogu.models.transformers.transformer_boogu import (  # noqa: F401
        BooguImageTransformer2DModel,
    )
    from boogu.pipelines.boogu.pipeline_boogu_turbo import (
        BooguImageTurboPipeline,
    )

    t0 = time.time()

    # Local snapshot inside the shared HF cache (HF_HOME on the VM) — same
    # cache the deploy script pre-downloads into, so this is a no-op when
    # Phase 4 ran. Diffusers' from_pretrained would also fetch it, but the
    # transformer load below needs a real directory path.
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(BOOGU_REPO)
    log.info("Boogu checkpoint snapshot: %s", snapshot)

    # The fp8 repo stores the DiT weights as .bin shards (bf16) — safetensors
    # do not exist for it. Loaded separately exactly like the upstream loader
    # (inference_turbo.py --use_fp8_weights True) so the rest of the pipeline
    # resolves from model_index.json.
    log.info("Loading Boogu transformer (bf16 .bin shards, ~10.3 GB) …")
    transformer = BooguImageTransformer2DModel.from_pretrained(
        os.path.join(snapshot, "transformer"),
        torch_dtype     = torch.bfloat16,
        use_safetensors = False,
    )

    log.info("Assembling BooguImageTurboPipeline (fp8 mllm, FLUX.1 VAE) …")
    pipe = BooguImageTurboPipeline.from_pretrained(
        snapshot,
        torch_dtype      = torch.bfloat16,
        trust_remote_code = True,
        transformer      = transformer,
    )

    # ── CPU-offload exception (user-approved) — see the module docstring ──
    # Nothing stays resident: accelerate moves each module to the card for
    # its forward and back to CPU afterwards (model_cpu_offload_seq
    # mllm->transformer->vae). Host RAM peak ~20-21 GB on the 62 GB box.
    pipe.enable_model_cpu_offload_flag = True
    pipe.enable_model_cpu_offload(device="cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    log.info("Boogu turbo pipeline ready in %.1f s (CPU-offloaded)",
             time.time() - t0)
    return pipe


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_boogu(pipe, params: dict) -> tuple[list[Any], int]:
    """Generate turbo images with the loaded Boogu pipeline.

    `params` keys: prompt, width, height, num_inference_steps (clamped to
    [1, 8], default 4), seed (-1 → random drawn here, returned). Guidance is
    FORCED to 1.0 — the DMD path has no CFG branch and the pipeline raises on
    anything else. One pipeline call per image (each call re-stages modules
    under CPU offload, so batching would only buy latency — not VRAM).

    Returns (images, seed_used).
    """
    import torch

    seed = int(params.get("seed", -1))
    if seed < 0:
        seed = random.randrange(2**31 - 1)
    t0 = time.time()

    prompt = params["prompt"]
    steps  = max(1, min(int(params.get("num_inference_steps", 4)), 8))
    width  = int(params.get("width",  1024))
    height = int(params.get("height", 1024))
    n      = max(1, min(int(params.get("num_images", 1)), 2))
    # DMD student constraints — any other value raises inside the pipeline.
    guidance = 1.0

    images: list[Any] = []
    for i in range(n):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        out = pipe(
            instruction                    = prompt,
            width                         = width,
            height                        = height,
            num_inference_steps           = steps,
            text_guidance_scale           = guidance,
            image_guidance_scale          = guidance,
            empty_instruction_guidance_scale = 0.0,
            num_images_per_instruction    = 1,
            generator                     = generator,
        )
        images.extend(out.images)

    log.info("Boogu turbo generated %d image(s) in %.1f s (%d steps, %dx%d)",
             len(images), time.time() - t0, steps, width, height)
    return images, seed


# ---------------------------------------------------------------------------
# Probe availability (called at startup)
# ---------------------------------------------------------------------------

def probe_boogu() -> dict:
    """Check whether the boogu package imports cleanly on this system."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False,
                    "error": "CUDA not available — Boogu requires the CUDA offload path"}
        try:
            import importlib
            importlib.import_module(_TRANSFORMER_MODULE)
            importlib.import_module(_TURBO_MODULE)
        except ImportError as exc:
            return {"available": False,
                    "error": f"boogu package not installed ({exc})"}
        return {"available": True}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
