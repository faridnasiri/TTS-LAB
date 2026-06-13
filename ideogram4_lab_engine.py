"""
ideogram4_lab_engine.py — Ideogram 4 text-to-image engine for Arthur Image Lab.

Integrates the ideogram4 package as a local PyPI-installed dependency.
Supports both nf4 (CUDA-only, bitsandbytes) and fp8 (any device) quantizations.
Optional magic-prompt expansion (no local model — all API-based, zero VRAM):

  1. Ideogram hosted API (free, best quality)  — IDEOGRAM_API_KEY
  2. DeepSeek native API (v1.txt prompt)        — DEEPSEEK_API_KEY
  3. OpenRouter → DeepSeek V3 (v1.txt fallback) — OPENROUTER_API_KEY

Usage:
    from ideogram4_lab_engine import load_ideogram4, generate_ideogram4
    pipeline = load_ideogram4(quant="nf4")
    images   = generate_ideogram4(pipeline, prompt="a cat", ...)
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("image_lab")

# ---------------------------------------------------------------------------
# Lazy imports — delay torch/ideogram4 imports until load time
# ---------------------------------------------------------------------------

# Magic-prompt system prompts shipped with the ideogram4 package
_MAGIC_PROMPT_SYSTEM_DIR = (
    Path(__file__).resolve().parent
    / "ideogram4"
    / "src"
    / "ideogram4"
    / "magic_prompt_system_prompts"
)

# Fallback: v1.txt copied alongside ideogram4_lab_engine.py (used when
# editable install layout differs or package is in site-packages)
_MAGIC_PROMPT_FALLBACK_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Magic-prompt: which LLM provider to use
# ---------------------------------------------------------------------------
# Priority: DEEPSEEK_API_KEY > OPENROUTER_API_KEY
# DeepSeek key can be native (sk-...) or via OpenRouter (sk-or-v1-...)
_MAGIC_PROMPT_MODE = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QWEN3_CACHE_DIR = "/opt/arthur-img-models/qwen3-vl-4bit-cache"


def _cache_quantized_text_encoder(pipe) -> None:
    """Save the bitsandbytes-quantized Qwen3-VL to disk so we skip re-quantization on restart.

    NOTE: Currently disabled — bitsandbytes Params4bit tensors cannot be
    serialized via save_pretrained() (NotImplementedError) and the torch.save
    workaround causes CUDA OOM on reload. Revisit when GPU VRAM situation changes.
    """
    return  # Disabled — known OOM issue with cached Params4bit reload


def _load_cached_text_encoder(device: str = "cuda", torch_dtype=None):
    """Load the pre-quantized Qwen3-VL from disk cache. Returns (text_encoder, tokenizer) or (None, None).

    NOTE: Currently disabled — reloading cached Params4bit weights causes CUDA OOM
    due to hidden GPU buffer allocations. Revisit when VRAM situation changes.
    """
    return None, None  # Disabled — known OOM issue

def _resolve_hf_token() -> str:
    """Return HF_TOKEN from environment (set in deploy script)."""
    tok = os.environ.get("HF_TOKEN", "")
    return tok.strip()


def _resolve_openrouter_key() -> str | None:
    """Return OpenRouter API key from OPENROUTER_API_KEY env var or config."""
    return os.environ.get("OPENROUTER_API_KEY", None)


def _resolve_deepseek_key() -> str | None:
    """Return DeepSeek API key from DEEPSEEK_API_KEY env var."""
    return os.environ.get("DEEPSEEK_API_KEY", None)


def _load_caption_hint() -> str:
    """Return a hint about the caption JSON format for the UI."""
    return (
        "Ideogram 4 expects a structured JSON caption, not a plain-text prompt. "
        "You can write one manually (example below) or use 'Magic prompt' to expand "
        "a plain idea into the proper format using an LLM via OpenRouter.\n\n"
        'Example:\n'
        '{\n'
        '  "high_level_description": "a serene mountain lake at sunset",\n'
        '  "style_description": {\n'
        '    "aesthetics": "cinematic, photorealistic",\n'
        '    "lighting": "golden hour warm light",\n'
        '    "photo": "35mm, f/2.8, shallow depth of field",\n'
        '    "medium": "photograph",\n'
        '    "color_palette": ["#FF6B35", "#1A237E", "#FFD54F", "#4A148C"]\n'
        '  },\n'
        '  "compositional_deconstruction": {\n'
        '    "background": "snow-capped mountains reflecting on a calm lake under a vibrant sunset sky",\n'
        '    "elements": [\n'
        '      {"type": "obj", "desc": "a tall pine tree silhouetted against the sunset"},\n'
        '      {"type": "obj", "desc": "a wooden dock extending into the lake"},\n'
        '      {"type": "obj", "desc": "a rowboat tied to the dock"}\n'
        '    ]\n'
        '  }\n'
        '}\n\n'
        "Tip: Keep high_level_description concise (like a search query). "
        "The compositional_deconstruction elements describe specific image components."
    )


# ---------------------------------------------------------------------------
# Magic-prompt expansion
# ---------------------------------------------------------------------------

# OpenRouter endpoint — use for DeepSeek via OpenRouter too
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The OpenRouter model slug for DeepSeek
_OPENROUTER_DEEPSEEK_MODEL = "deepseek/deepseek-chat"

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Ideogram's own hosted magic-prompt API (free, server-side — no local model needed)
_IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"


# ---------------------------------------------------------------------------
# v1.txt system prompt loader (Ideogram's hand-crafted 28KB recipe)
# ---------------------------------------------------------------------------

def _load_ideogram_v1_messages(prompt: str, aspect_ratio: str = "1:1") -> list[dict]:
    """
    Build chat messages using Ideogram's official v1.txt system prompt.
    Uses the ideogram4 package's own build_messages() if available,
    otherwise loads v1.txt from the shipped directory or fallback path.

    Returns [{"role": "system", ...}, {"role": "user", ...}]
    """
    # Try the ideogram4 package's build_messages first
    try:
        from ideogram4.magic_prompt import build_messages
        return build_messages("v1.txt", prompt, aspect_ratio)
    except (ImportError, Exception):
        pass

    # Fallback: load v1.txt manually
    v1_path = _MAGIC_PROMPT_SYSTEM_DIR / "v1.txt"
    if not v1_path.exists():
        v1_path = _MAGIC_PROMPT_FALLBACK_DIR / "v1.txt"
    if not v1_path.exists():
        log.warning("v1.txt not found — falling back to simple system prompt")
        return [
            {"role": "system", "content": "Expand the user's plain-text idea into an Ideogram 4 JSON caption."},
            {"role": "user", "content": f"Aspect ratio: {aspect_ratio}\n\n{prompt}"},
        ]

    raw = v1_path.read_text(encoding="utf-8")
    # Parse [SYSTEM] and [USER] sections
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()

    system = sections.get("system", "Expand the user's plain-text idea into an Ideogram 4 JSON caption.")
    user_template = sections.get("user", "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).")
    user = user_template.replace("{{aspect_ratio}}", aspect_ratio)
    if "{{original_prompt}}" in user:
        user = user.replace("{{original_prompt}}", prompt)
    else:
        user = f"{user}\n\n{prompt}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _expand_via_ideogram(
    prompt: str,
    aspect_ratio: str = "1:1",
) -> str | None:
    """
    Expand a plain-text prompt via Ideogram's own hosted magic-prompt API (FREE).

    This is the highest-quality option — Ideogram's servers run the expansion
    with their own tuned pipeline. No system prompt needed.

    Requires IDEOGRAM_API_KEY env var.
    """
    api_key = os.environ.get("IDEOGRAM_API_KEY", "")
    if not api_key:
        return None  # Silently skip — not configured

    import requests

    # Ideogram API uses "WxH" format
    ideogram_ratio = aspect_ratio.replace(":", "x")

    try:
        resp = requests.post(
            _IDEOGRAM_MAGIC_PROMPT_URL,
            headers={
                "Api-Key": api_key,
                "Content-Type": "application/json",
            },
            json={"text_prompt": prompt, "aspect_ratio": ideogram_ratio},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        json_prompt = data.get("json_prompt")
        if not json_prompt:
            log.warning("Ideogram API returned no json_prompt: %s", data)
            return None
        # Return minified JSON
        import json as _json
        caption = _json.dumps(json_prompt, ensure_ascii=False, separators=(",", ":"))
        log.info("Magic-prompt expanded via Ideogram hosted API (free)")
        return caption
    except Exception as exc:
        log.warning("Ideogram hosted magic-prompt failed: %s", exc)
        return None


def _expand_via_deepseek(
    prompt: str,
    aspect_ratio: str = "1:1",
) -> str | None:
    """
    Expand a plain-text prompt into a structured Ideogram 4 caption via DeepSeek API.

    Uses Ideogram's official v1.txt system prompt for maximum caption quality.
    Returns the JSON caption string, or None if expansion fails.
    """
    api_key = _resolve_deepseek_key()
    if not api_key:
        log.warning("DeepSeek API key not set — skipping magic prompt expansion")
        return None

    import requests

    messages = _load_ideogram_v1_messages(prompt, aspect_ratio)

    body = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(
            _DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            log.warning("DeepSeek returned empty content")
            return None
        # Strip code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return content
    except Exception as exc:
        log.warning("DeepSeek magic-prompt expansion failed: %s", exc)
        return None

def _expand_via_openrouter(
    prompt: str,
    aspect_ratio: str = "1:1",
) -> str | None:
    """
    Expand a plain-text prompt into a structured Ideogram 4 caption
    via OpenRouter using DeepSeek model.
    """
    api_key = _resolve_openrouter_key()
    if not api_key:
        log.warning("OpenRouter API key not set — skipping magic prompt expansion")
        return None

    import requests

    messages = _load_ideogram_v1_messages(prompt, aspect_ratio)

    body = {
        "model": _OPENROUTER_DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://arthur-lab.local",
                "X-Title": "Arthur Image Lab",
            },
            json=body,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            log.warning("OpenRouter returned empty content")
            return None
        # Strip code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return content
    except Exception as exc:
        log.warning("OpenRouter DeepSeek magic-prompt expansion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ideogram4(
    quant: str = "nf4",
    device: str | None = None,
    dtype: str = "bfloat16",
    **kwargs,
):
    """
    Load the Ideogram 4 pipeline from HuggingFace.

    Args:
        quant: "nf4" (bitsandbytes 4-bit, CUDA-only, ~6 GB VRAM) or
               "fp8" (weight-only float8, any device, ~10 GB VRAM)
        device: "cuda" or "cpu". Auto-detected if None.
        dtype: "bfloat16" (recommended), "float16", or "float32"

    Returns:
        Ideogram4Pipeline instance ready for __call__.
    """
    import torch

    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve dtype
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    # Select repo + pipeline config based on quantization
    if quant == "nf4":
        weights_repo = "ideogram-ai/ideogram-4-nf4"
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, Ideogram4Config
        config = Ideogram4PipelineConfig(weights_repo=weights_repo)
        transformer_config = Ideogram4Config()

        log.info(
            "Loading Ideogram 4 (nf4) from %s on %s — ~6 GB VRAM for transformer",
            weights_repo, device,
        )

        pipe = Ideogram4Pipeline.from_pretrained(
            config=config,
            device=device,
            dtype=torch_dtype,
            transformer_config=transformer_config,
        )
    elif quant == "fp8":
        weights_repo = "ideogram-ai/ideogram-4-fp8"
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, Ideogram4Config

        # fp8 variant needs a custom config pointing at the fp8 repo
        config = Ideogram4PipelineConfig(
            weights_repo=weights_repo,
            conditional_index_filename=(
                "transformer/diffusion_pytorch_model.safetensors.index.json"
            ),
            unconditional_index_filename=(
                "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json"
            ),
            autoencoder_filename="vae/diffusion_pytorch_model.safetensors",
            text_encoder_subfolder="text_encoder",
            tokenizer_subfolder="tokenizer",
        )
        transformer_config = Ideogram4Config()

        log.info(
            "Loading Ideogram 4 (fp8) from %s on %s — ~10 GB VRAM for transformer",
            weights_repo, device,
        )

        pipe = Ideogram4Pipeline.from_pretrained(
            config=config,
            device=device,
            dtype=torch_dtype,
            transformer_config=transformer_config,
        )
    else:
        raise ValueError(f"Unknown quantization: {quant}. Use 'nf4' or 'fp8'.")

    log.info("Ideogram 4 loaded successfully")

    # Cache the quantized Qwen3-VL to disk — skips bitsandbytes re-quantization on next load
    _cache_quantized_text_encoder(pipe)

    # Immediately offload non-essential components to CPU to free VRAM
    # The pipeline __call__ will bring them back when needed
    import gc
    # Aggressive cleanup before offloading (GPU may be at 99% capacity)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    for comp_name in ['text_encoder', 'unconditional_transformer', 'autoencoder']:
        if hasattr(pipe, comp_name):
            comp = getattr(pipe, comp_name)
            if comp is not None:
                try:
                    dev = next(comp.parameters()).device
                    if dev.type == 'cuda':
                        setattr(pipe, comp_name, comp.to('cpu'))
                        log.info("Offloaded %s to CPU after loading", comp_name)
                        # Free immediately so next offload has room
                        gc.collect()
                        torch.cuda.empty_cache()
                except StopIteration:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    free_mib = torch.cuda.mem_get_info()[0] / 1024**2
    log.info("VRAM after offload: %.0f MiB free", free_mib)

    return pipe


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_ideogram4(
    pipe,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    preset: str = "V4_DEFAULT_20",
    num_steps: int | None = None,
    guidance_scale: float = 7.0,
    guidance_schedule: list[float] | None = None,
    mu: float | None = None,
    std: float | None = None,
    seed: int = -1,
    use_magic_prompt: bool = False,
    magic_prompt_aspect_ratio: str = "1:1",
    **kwargs,
):
    """
    Generate images with Ideogram 4.

    Args:
        pipe: Loaded Ideogram4Pipeline.
        prompt: JSON caption string (or plain text if use_magic_prompt).
        width/height: Output dimensions (multiples of 16, 256–2048).
        preset: Sampler preset name from PRESETS dict.
        num_steps: Override steps from preset if given.
        guidance_scale: CFG scale (used if no guidance_schedule given).
        guidance_schedule: Per-step CFG weights (overrides guidance_scale).
        mu, std: Logit-normal schedule params (overrides preset).
        seed: Random seed (-1 = random).
        use_magic_prompt: If True, expand prompt via OpenRouter.
        magic_prompt_aspect_ratio: Target aspect ratio for magic prompt.
        **kwargs: Ignored extra params for compatibility.

    Returns:
        list of PIL Image objects.
    """
    import torch
    from ideogram4.sampler_configs import PRESETS
    from ideogram4.scheduler import SamplerParameters

    # --- Handle magic prompt expansion ---
    if use_magic_prompt and prompt.strip():
        # Priority: Ideogram hosted API (free, best quality) →
        #           Native DeepSeek API (v1.txt prompt) →
        #           OpenRouter → DeepSeek (v1.txt prompt, fallback)
        expanded = _expand_via_ideogram(prompt, magic_prompt_aspect_ratio)
        if not expanded:
            expanded = _expand_via_deepseek(prompt, magic_prompt_aspect_ratio)
        if not expanded:
            expanded = _expand_via_openrouter(prompt, magic_prompt_aspect_ratio)
        if expanded:
            log.info("Magic-prompt expanded plain text to structured caption")
            prompt = expanded
        else:
            log.warning(
                "Magic-prompt expansion failed — using prompt as-is. "
                "Ideogram 4 expects a JSON caption for best results."
            )

    # --- Resolve preset params ---
    if preset in PRESETS:
        sp: SamplerParameters = PRESETS[preset]
        effective_num_steps = num_steps if num_steps is not None else sp.num_steps
        effective_mu = mu if mu is not None else sp.mu
        effective_std = std if std is not None else sp.std
        effective_guidance_schedule = guidance_schedule if guidance_schedule else list(sp.guidance_schedule)
    else:
        effective_num_steps = num_steps or 20
        effective_mu = mu if mu is not None else 0.0
        effective_std = std if std is not None else 1.75
        effective_guidance_schedule = guidance_schedule

    # If guidance_schedule was provided as a list, use it; otherwise fall back
    # to the per-step constant guidance_scale.
    if effective_guidance_schedule and len(effective_guidance_schedule) != effective_num_steps:
        log.warning(
            "guidance_schedule length (%d) != num_steps (%d) — falling back to constant guidance_scale=%.1f",
            len(effective_guidance_schedule), effective_num_steps, guidance_scale,
        )
        effective_guidance_schedule = None

    # --- Generate ---
    start = time.time()
    gen_seed = seed if seed >= 0 else None

    # VRAM optimisation: offload components not needed during denoising
    import gc
    vae_offloaded = False
    uc_offloaded = False
    te_offloaded = False

    # 1. Offload VAE (only needed for final decode)
    if hasattr(pipe, 'autoencoder') and pipe.autoencoder is not None:
        ae_dev = next(pipe.autoencoder.parameters()).device
        if ae_dev.type == 'cuda':
            pipe.autoencoder = pipe.autoencoder.to('cpu')
            gc.collect(); torch.cuda.empty_cache()
            vae_offloaded = True
            log.info("Offloaded VAE to CPU (~160 MB VRAM freed)")

    # 2. Offload unconditional transformer (only needed for CFG steps)
    #    We pre-offload and let the pipeline bring it back for CFG
    if hasattr(pipe, 'unconditional_transformer') and pipe.unconditional_transformer is not None:
        uc_dev = next(pipe.unconditional_transformer.parameters()).device
        if uc_dev.type == 'cuda':
            pipe.unconditional_transformer = pipe.unconditional_transformer.to('cpu')
            gc.collect(); torch.cuda.empty_cache()
            uc_offloaded = True
            log.info("Offloaded unconditional transformer to CPU (~6 GB VRAM freed)")

    # 3. Offload Qwen3-VL text encoder (only needed once for token encoding)
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        te_dev = next(pipe.text_encoder.parameters()).device
        if te_dev.type == 'cuda':
            pipe.text_encoder = pipe.text_encoder.to('cpu')
            gc.collect(); torch.cuda.empty_cache()
            te_offloaded = True
            log.info("Offloaded Qwen3-VL to CPU (~3 GB VRAM freed)")

    # Log VRAM state
    free_mib = torch.cuda.mem_get_info()[0] / 1024**2
    log.info("VRAM before generation: %.0f MiB free, %.0f MiB used",
             free_mib, torch.cuda.get_device_properties(0).total_memory / 1024**2 - free_mib)

    try:
        images = pipe(
            prompts=prompt,
            height=height,
            width=width,
            num_steps=effective_num_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=tuple(effective_guidance_schedule) if effective_guidance_schedule else None,
            mu=effective_mu,
            std=effective_std,
            seed=gen_seed,
            raise_on_caption_issues=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Ideogram 4 generation failed. Check your caption JSON format.\n"
            f"Error: {exc}\n\n"
            f"Tip: Use Magic Prompt to auto-generate the JSON caption from plain text, "
            f"or ensure your prompt follows the Ideogram 4 caption schema."
        ) from exc
    finally:
        # Restore offloaded components
        if vae_offloaded and hasattr(pipe, 'autoencoder'):
            pipe.autoencoder = pipe.autoencoder.to('cuda')
            gc.collect(); torch.cuda.empty_cache()
        if uc_offloaded and hasattr(pipe, 'unconditional_transformer'):
            pipe.unconditional_transformer = pipe.unconditional_transformer.to('cuda')
            gc.collect(); torch.cuda.empty_cache()
        if te_offloaded and hasattr(pipe, 'text_encoder'):
            pipe.text_encoder = pipe.text_encoder.to('cuda')
            gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - start
    log.info(
        "Ideogram 4 generated %d image(s) in %.1f s (%d steps, %dx%d, preset=%s)",
        len(images), elapsed, effective_num_steps, width, height, preset,
    )

    return images, prompt  # Return final caption for API response


# ---------------------------------------------------------------------------
# Probe availability (called at startup)
# ---------------------------------------------------------------------------

def probe_ideogram4() -> dict:
    """
    Check whether Ideogram 4 can be loaded on this system.
    Returns a dict with 'available' bool and optional 'error' string.
    """
    try:
        import torch

        # Check torch + CUDA
        if not torch.cuda.is_available():
            return {"available": False, "error": "CUDA not available — RTX 5060 (Blackwell) required for nf4 variant"}

        # Check bitsandbytes for nf4
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            return {"available": False, "error": "bitsandbytes not installed — required for nf4 quantization"}

        # Check HF token
        hf_token = _resolve_hf_token()
        if not hf_token:
            return {
                "available": False,
                "error": "HF_TOKEN not set in environment — required for gated model ideogram-ai/ideogram-4-nf4",
            }

        # Check ideogram4 package
        try:
            import ideogram4  # noqa: F401
        except ImportError:
            return {"available": False, "error": "ideogram4 package not installed"}

        # Check VRAM (rough minimum for nf4: ~8 GB including Qwen3-VL + VAE)
        device_id = 0
        total_vram = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        if total_vram < 8:
            return {
                "available": False,
                "error": f"Only {total_vram:.1f} GB VRAM detected — need at least 8 GB for Ideogram 4 nf4",
            }

        return {"available": True}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
