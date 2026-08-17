"""
image_lab_engines.py — Load / unload / generate functions for all engines:
  flux2klein  — FLUX.2 Klein 4B (custom NF4 encoder assembly)
  flux2klein9b — FLUX.2 Klein 9B-KV GGUF (Q4_K_M) via QuantStack
  sd35        — Stable Diffusion 3.5 Large GGUF (Q4_0 / Q5_0 / Q8_0) via city96
  wan         — Wan2.2 T2V / I2V GGUF (Q3_K_M / Q4_K_M / Q5_K_M / Q8_0) via QuantStack
  ideogram4   — Ideogram 4 (API)

Note: flux2 (FLUX.2 [dev] 32B) was REMOVED 2026-08-13 — its ~27 GB footprint
cannot fit the 15.5 GiB card and the GPU-only policy forbids CPU fallback.

GGUF files are downloaded from HuggingFace on first use and cached under GGUF_ROOT.
Non-transformer pipeline components (text encoders, VAE, scheduler, tokenizers)
are loaded from the pre-saved shared directories written by preq_save.py.
"""

from __future__ import annotations
import gc
import hashlib
import logging
import os
import time
from typing import Any, Optional

from image_lab_config import ENGINES, STATE, OUTPUT_ROOT, HF_TOKEN, HF_HOME, GPU_ONLY
from image_lab_utils import free_vram, random_seed, save_image, save_images, save_video

# Local directory for cached GGUF model files
GGUF_ROOT = "/opt/arthur-img-models/gguf"

# Pre-saved shared pipeline components (text encoders, VAE, configs)
# These were written by preq_save.py and contain everything except the transformer.
PREQ_ROOT = "/opt/arthur-img-models/quantized"

# NVFP4-quantized transformers saved by nvfp4_save.py (torchao NVFP4WeightOnlyConfig)
NVFP4_ROOT = "/opt/arthur-img-models/nvfp4"

# ── Prompt-embedding disk cache ─────────────────────────────────────────────
# prompt_embeds depend ONLY on the prompt text (and the fixed encoder) — they
# are deterministic. Caching them lets repeat prompts skip the ~4.5 GiB Qwen3
# text encoder entirely: it loads lazily, only for prompts never seen before.
# Lives under the output root (writable by the service); bump the version to
# invalidate after an encoder change.
# Optional override: point the embed cache at a RAM disk (e.g. /dev/shm) to
# keep the ~12.5 MB-per-prompt writes off the NVMe. Trade-off: tmpfs is
# volatile — a VM reboot wipes the cache and previously-seen prompts re-encode
# once (+10-15 s each). Default (disk) keeps the cache across reboots.
EMBED_CACHE_ROOT    = os.environ.get("EMBED_CACHE_ROOT", "").strip() or os.path.join(OUTPUT_ROOT, "embed_cache")
EMBED_CACHE_VERSION = 1

# ── Gemini Flash prompt expansion (optional, free tier) ─────────────────────
# When GEMINI_API_KEY is set, klein prompts are expanded by Gemini BEFORE the
# local Qwen3 encode — richer captions, zero extra VRAM (the embeddings are
# still produced by the local encoder). Any failure falls back to the original
# prompt; expansion never blocks generation.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

_GEMINI_EXPAND_INSTRUCTION = (
    "You are a prompt engineer for a text-to-image model (FLUX.2 Klein). "
    "Rewrite the user's prompt into ONE vivid paragraph of at most 60 words "
    "that the image model will render well.\n\n"
    "Hard rules:\n"
    "- Preserve the user's literal requests exactly: any text they ask to "
    "WRITE or display must appear VERBATIM in your output.\n"
    "- If the prompt references an attached reference image, describe what to "
    "do WITH it — do not invent new scene content that replaces it.\n"
    "- Add concrete visual detail (lighting, palette, style, composition) but "
    "never contradict or add to literal text requests.\n"
    "- Plain English. No preamble. No markdown. Only the rewritten prompt."
)


def _embed_cache_path(engine_key: str, prompt: str) -> str:
    """Disk path for the cached embeddings of `prompt` (hash-keyed)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
    return os.path.join(EMBED_CACHE_ROOT, f"v{EMBED_CACHE_VERSION}", engine_key, digest + ".pt")


def _embed_cache_load(path: str) -> Optional[Any]:
    """Load cached prompt_embeds (None if absent/corrupt)."""
    import torch
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)["prompt_embeds"]
    except Exception as ex:
        log.warning("Embedding cache unreadable (%s) — ignoring", ex)
        return None


def _embed_cache_save(path: str, prompt_embeds: Any) -> None:
    """Save prompt_embeds to disk; failures are non-fatal (cache is an optimisation)."""
    import torch
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"prompt_embeds": prompt_embeds.detach().cpu()}, path)
    except Exception as ex:
        log.warning("Embedding cache write failed (%s) — continuing without it", ex)


def _gemini_expand_prompt(prompt: str) -> Optional[str]:
    """Expand `prompt` via the free Gemini Flash tier. Returns None (use the
    original prompt) on any failure — no key, HTTP error, unparseable reply."""
    if not GEMINI_API_KEY:
        return None
    try:
        import httpx
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
        r = httpx.post(url, json={
            "contents": [{"parts": [{"text": _GEMINI_EXPAND_INSTRUCTION + "\n\nPrompt: " + prompt}]}],
            # gemini-2.5-flash thinks by default (~190 hidden thought tokens)
            # and maxOutputTokens caps THOUGHTS + visible text combined —
            # at 200 the visible answer was cut at ~8 tokens. thinkingBudget 0
            # is ignored by the API, so the cap must simply leave room:
            # 1024 covers ~190 thoughts + the ≤60-word rewrite with headroom.
            "generationConfig": {"temperature": 0.8, "maxOutputTokens": 1024,
                                 "thinkingConfig": {"thinkingBudget": 0}},
        }, timeout=30.0)
        if r.status_code != 200:
            log.warning("Gemini expansion HTTP %s — using original prompt", r.status_code)
            return None
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "").strip() if parts else ""
        if not text:
            log.warning("Gemini returned no text — using original prompt")
            return None
        return text
    except Exception as ex:
        log.warning("Gemini expansion failed (%s) — using original prompt", ex)
        return None


def _ensure_klein_encoder(pipe: Any, engine_key: str) -> None:
    """Load the Qwen3 text encoder into the pipeline on demand (embed-cache
    miss) and place it on CUDA. With the cache, the encoder is only needed
    for prompts never seen before — leaving it unloaded otherwise keeps the
    resident set at ~5.7 GiB instead of ~10 GiB."""
    import torch
    enc = pipe.text_encoder
    if enc is not None:
        if enc.device != torch.device("cuda"):
            enc.to("cuda")
        return
    if engine_key == "flux2klein":
        repo, extra, label = "black-forest-labs/FLUX.2-klein-4B", {"subfolder": "text_encoder"}, "Qwen3 (klein-4B)"
    else:
        repo, extra, label = "Qwen/Qwen3-8B", {}, "Qwen3-8B"
    from transformers import AutoModel, BitsAndBytesConfig
    log.info("Loading %s text encoder (NF4 4-bit) on demand — brand-new prompt …", label)
    pipe.text_encoder = AutoModel.from_pretrained(
        repo,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        ),
        device_map = "cuda",
        **extra,
    )


def _klein_prompt_embeds(pipe: Any, engine_key: str, prompts: dict) -> dict:
    """Return {label: prompt_embeds} for the given {label: prompt_text} map.

    Cache hit → embeds from disk, the text encoder never even loads.
    Cache miss → lazy-load the encoder, encode under torch.no_grad() (the
    Qwen3 forward with output_hidden_states=True otherwise keeps ~2.4 GiB of
    per-layer activations live through the returned embeds' autograd graph —
    the pipeline's own __call__ is no_grad-decorated; our manual call isn't),
    save to cache, then park the encoder on CPU for the transformer pass.
    Non-GPU_ONLY keeps the historical direct-encode path (group-offload hooks)."""
    import torch
    if not GPU_ONLY:
        out: dict = {}
        for label, text in prompts.items():
            with torch.no_grad():
                embeds, _ = pipe.encode_prompt(prompt=text, device="cuda")
            out[label] = embeds
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        return out

    result: dict = {}
    missing: list = []
    for label, text in prompts.items():
        cached = _embed_cache_load(_embed_cache_path(engine_key, text))
        if cached is not None:
            result[label] = cached.to("cuda")
        else:
            missing.append((label, text))
    if not missing:
        log.info("Prompt-embedding cache hit (%s) — skipping the text encoder", engine_key)
        return result
    _ensure_klein_encoder(pipe, engine_key)
    for label, text in missing:
        with torch.no_grad():
            embeds, _ = pipe.encode_prompt(prompt=text, device="cuda")
        _embed_cache_save(_embed_cache_path(engine_key, text), embeds)
        result[label] = embeds
    _park_klein_encoder(pipe)
    return result


def _park_klein_encoder(pipe) -> None:
    """Release the NF4 text encoder from VRAM after a cache-miss encode.

    The encoder is dropped (reference + gc), exactly like _unload_current —
    .to("cpu") alone left it alive on GPU. Note: even freed, its blocks stay
    pinned at the DRIVER level (~4.4 GiB, shared segments with the live
    transformer — see the note in image_lab.py) but remain reusable by the
    torch allocator, so later generations and even the next encoder load
    absorb into the pool. Only a full unload/restart returns it to the
    driver. _ensure_klein_encoder re-loads it for brand-new prompts.
    """
    import torch
    enc = getattr(pipe, "text_encoder", None)
    if enc is None:
        return
    pipe.text_encoder = None
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# GGUF file catalogue
# ---------------------------------------------------------------------------

# (repo_id, filename_in_repo)  — for flat-layout repos (SD35)
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
    `model_key`  — e.g. "sd35", "wan-t2v", "wan-i2v"
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
        device_map      = "cuda",
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

    # Explicitly move GPU components to CPU before dropping references.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prevents the caching
    # allocator from releasing memory unless we explicitly offload first.
    _gpu_attrs = [
        'conditional_transformer', 'unconditional_transformer',
        'text_encoder', 'autoencoder', 'transformer',
        'vae', 'text_encoder_2',
    ]
    for ref in [STATE.loaded_model, STATE.loaded_pipe2]:
        if ref is None:
            continue
        for attr in _gpu_attrs:
            comp = getattr(ref, attr, None)
            if comp is not None:
                try:
                    setattr(ref, attr, comp.to('cpu'))
                except Exception:
                    pass
                try:
                    setattr(ref, attr, None)
                except Exception:
                    pass

    STATE.loaded_model  = None
    STATE.loaded_pipe2  = None
    STATE.active_engine = None
    STATE.active_quant  = ""
    free_vram()
    for k in ENGINES:
        ENGINES[k].loaded = False


def _ensure_engine(key: str, quant: str = ""):
    """Load engine `key` with `quant` into VRAM, evicting whatever is currently loaded."""
    if STATE.active_engine == key and STATE.active_quant == quant:
        return  # already loaded with the same quantization
    _unload_current()
    # GPU-only policy: make room before allocating — evict the TTS engine
    # containers, then the LLM container if still short (see
    # _ensure_vram_headroom). Runs after _unload_current so the previous
    # image engine's VRAM is already accounted for.
    if GPU_ONLY:
        need = _VRAM_NEED_MB.get(key)
        if need is not None:
            _ensure_vram_headroom(need, key)
    loader = _LOADERS.get(key)
    if loader is None:
        raise RuntimeError(f"No loader for engine '{key}'")
    STATE.loading = True
    try:
        loader(quant)
    finally:
        STATE.loading = False

# ---------------------------------------------------------------------------
# Shared-GPU helper
# ---------------------------------------------------------------------------

def _evict_tts_engines() -> int:
    """POST /evict to every TTS engine container to free shared GPU memory.

    GPU-only policy (2026-08-13): image engines NEVER fall back to CPU
    rendering or system-RAM offloading. When a loader needs more VRAM than is
    free, the TTS containers' resident models are evicted first — they
    lazy-reload on their next TTS request (a few seconds of added latency).
    Returns the free VRAM in MiB after the evictions settle.
    """
    import time as _t
    import urllib.request as _urllib
    import torch
    for port in (8101, 8102, 8103, 8104):
        try:
            req = _urllib.Request(
                f"http://localhost:{port}/evict", data=b"", method="POST")
            with _urllib.urlopen(req, timeout=5) as resp:
                log.info("TTS engine container :%d evicted (%s)",
                         port, resp.read().decode().strip()[:120])
        except Exception:
            pass   # container down / nothing loaded — fine
    _t.sleep(2)  # let CUDA return the freed memory to the driver
    return torch.cuda.mem_get_info()[0] // (1024 * 1024)


# ---------------------------------------------------------------------------
# VRAM headroom enforcement — every loader makes room before allocating
# ---------------------------------------------------------------------------

# Minimum free VRAM (MiB) required before each engine's loader starts.
# Calibrated for the RTX 5060 Ti 16 GB card (~15350 MiB max free once the
# TTS containers' CUDA contexts release). The GPU-only policy (2026-08-13)
# forbids CPU fallback, so a short load makes room instead of OOMing mid-load.
_VRAM_NEED_MB: dict[str, int] = {
    # Klein engines use a LAZY text encoder — only the transformer is resident
    # at load (~5.6 GiB); the NF4 Qwen3 encoder (+4.5 GiB) loads on demand for
    # uncached prompts and is released afterwards (its blocks stay pooled at
    # the driver level until a full unload — reusable by torch, invisible to
    # mem_get_info). 10500 lets the engine reload even while TTS containers
    # hold their CUDA contexts (~10832 MiB free observed), yet still leaves
    # headroom for the on-demand encode peak (~10.6 GiB). Old values
    # (12500/14800) assumed the encoder was resident.
    "flux2klein":   10500,
    "flux2klein9b": 10500,
    "sd35":         12500,  # GGUF transformer + shared encoders/VAE ≈ 11-12 GiB
    "wan":          14800,  # two A14B transformers — needs the card to itself
    "ideogram4":    12000,  # nf4 transformer + Qwen3-VL encoder + VAE; load peak ≈ 11 GiB
}

_LLM_CONTAINER_NAME = "tts-lab-llm-qwen36"
_DOCKER_SOCK = "/var/run/docker.sock"


def _stop_llm_container() -> bool:
    """Stop the Qwen 3.6 LLM container to free its ~13.6 GiB VRAM.

    Last-resort eviction for image loads. Reversible: the TTS orchestrator
    restarts the container before LLM inference (tts_lab_dispatch Phase 0).
    Returns True only if this call issued the stop (HTTP 204); False if the
    container was already stopped (304), the docker socket is missing, or
    the request failed.
    """
    if not os.path.exists(_DOCKER_SOCK):
        log.warning("No docker socket (%s) — cannot stop LLM container", _DOCKER_SOCK)
        return False
    try:
        import httpx
        transport = httpx.HTTPTransport(uds=_DOCKER_SOCK)
        with httpx.Client(transport=transport, timeout=10.0) as client:
            r = client.post(
                f"http://localhost/v1.49/containers/{_LLM_CONTAINER_NAME}/stop")
        stopped = r.status_code == 204
        log.info("Stop LLM container → HTTP %s %s", r.status_code,
                 "✓" if stopped else "✗ (already stopped or failed)")
        return stopped
    except Exception as exc:
        log.warning("Failed to stop LLM container: %s", exc)
        return False


def _ensure_vram_headroom(need_mb: int, key: str) -> None:
    """Make sure `need_mb` MiB of VRAM are free before loading engine `key`.

    Escalation chain (GPU-only policy — image engines never fall back to CPU):
      1. Evict the TTS engine containers (they lazy-reload on their next
         TTS request, a few seconds of added latency).
      2. Stop the LLM container as a last resort (~13.6 GiB).
      3. Raise a clear error instead of OOMing mid-load.
    """
    import torch
    if not torch.cuda.is_available():
        return  # CPU / CPU-offload mode — nothing to enforce
    free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mb >= need_mb:
        return
    log.info("VRAM free %d MiB < %d — evicting TTS engine containers …",
             free_mb, need_mb)
    free_mb = _evict_tts_engines()
    if free_mb < need_mb:
        log.info("Still %d MiB free — stopping LLM container (~13.6 GiB) …", free_mb)
        if _stop_llm_container():
            # Wait (bounded) for the driver to reclaim the LLM's memory.
            t0 = time.monotonic()
            while free_mb < need_mb and time.monotonic() - t0 < 30:
                time.sleep(1)
                free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mb < need_mb:
        label = ENGINES.get(key).label if ENGINES.get(key) else key
        raise RuntimeError(
            f"{label} needs ~{need_mb // 1024} GiB free VRAM; only {free_mb} MiB "
            f"available after evicting the TTS engine containers and stopping "
            f"the LLM container. (GPU-only policy — no CPU offloading.)")



# ---------------------------------------------------------------------------
# FLUX.2 Klein 4B
# ---------------------------------------------------------------------------

def _load_flux2klein(quant: str = ""):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from diffusers.models import AutoencoderKLFlux2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, Qwen2TokenizerFast

    t0    = time.time()
    # Same hf_hub 1.16.1 issue as the 9B-KV loader: `token=True` REQUIRES a
    # token and the systemd service has none — and this is a PUBLIC repo,
    # so no token is needed at all.
    token = None
    repo  = ENGINES["flux2klein"].hf_repo   # black-forest-labs/FLUX.2-klein-4B

    if not GPU_ONLY:
        # GPU-less machines (IMGLAB_GPU_ONLY=0) — historical path.
        log.info("Loading FLUX.2 Klein 4B from HuggingFace (%s) …", repo)
        pipe = Flux2KleinPipeline.from_pretrained(
            repo, torch_dtype = torch.bfloat16, token = token,
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()

        STATE.loaded_model       = pipe
        STATE.active_engine      = "flux2klein"
        STATE.active_quant       = ""
        ENGINES["flux2klein"].loaded = True
        log.info("FLUX.2 Klein 4B ready in %.1f s", time.time() - t0)
        return

    # ── GPU-only policy (2026-08-13) ─────────────────────────────────────────
    # Never fall back to CPU. The full bf16 repo measures ~15.1 GiB with the
    # Qwen3-4B text encoder — no room left for 1024² generation on a 15.5 GiB
    # card. The encoder is therefore loaded NF4 4-bit on CUDA (as in the 9B-KV
    # loader); the transformer stays full bf16. VRAM headroom is enforced
    # centrally in _ensure_engine → _ensure_vram_headroom (evicts the TTS
    # containers, then the LLM container, before allocating).

    log.info("Loading FLUX.2 Klein 4B transformer (BF16) …")
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo, subfolder = "transformer", torch_dtype = torch.bfloat16,
        token = token,
    )

    # Text encoder loads LAZILY — prompt embeddings are cached to disk (see
    # _klein_prompt_embeds), so the ~4 GiB Qwen3 encoder only loads for
    # prompts never seen before. Resident set without it: ~5.5 GiB. Its freed
    # blocks stay pooled at the driver level after a miss (see image_lab.py
    # note) — reusable by torch, invisible to nvidia-smi until a full unload.
    text_encoder = None

    log.info("Loading FLUX.2 Klein 4B tokenizer / VAE / scheduler …")
    tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder = "tokenizer")
    vae       = AutoencoderKLFlux2.from_pretrained(
        repo, subfolder = "vae", torch_dtype = torch.bfloat16, token = token,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder = "scheduler", torch_dtype = torch.bfloat16, token = token,
    )

    log.info("Assembling FLUX.2 Klein 4B pipeline …")
    pipe = Flux2KleinPipeline(
        transformer  = transformer,
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        vae          = vae,
        scheduler    = scheduler,
    ).to("cuda")
    pipe.vae.enable_slicing()

    # See the 9B-KV loader: the pipeline's `_execution_device` falls back to
    # `self.device`, which reads the __init__ signature order (text_encoder
    # first) — so when generation offloads the encoder to CPU, latent
    # placement would flip to "cpu" and the CUDA VAE would die with a conv
    # weight/input mismatch. With precomputed embeds the encoder is never
    # called during generation, so the pipeline genuinely executes on cuda.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    STATE.loaded_model       = pipe
    STATE.active_engine      = "flux2klein"
    STATE.active_quant       = ""
    ENGINES["flux2klein"].loaded = True
    log.info("FLUX.2 Klein 4B ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _generate_flux2klein(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    generator = torch.Generator("cpu").manual_seed(seed)
    guidance  = float(params.get("guidance_scale", 3.5))

    # Optional Gemini prompt expansion (needs GEMINI_API_KEY). Runs BEFORE
    # encoding, so the expanded caption flows through the normal local Qwen3
    # encoder — richer text, identical VRAM. Expansion also changes the
    # embed-cache key, so expanded prompts cache under their own hash.
    prompt = params["prompt"]
    expanded = _gemini_expand_prompt(prompt) if GPU_ONLY else None
    if expanded:
        log.info("Gemini expanded prompt: %r → %r", prompt, expanded)
        prompt = expanded

    # Prompt-embedding cache — see _klein_prompt_embeds. CFG mode needs the
    # empty-prompt (negative) embeds too; the negative prompt is a constant,
    # so it caches as a single file per engine.
    embeds = _klein_prompt_embeds(pipe, "flux2klein",
                                  {"prompt": prompt, "negative": ""})

    # Ref token count is the other budget lever: each ref token costs ~0.5 MiB
    # of retained K/V across every denoising step. Cap the ref area so a full
    # landscape screenshot (1024×576, 9216 tokens) passes untouched while
    # square 1024² refs can't blow the card. (PIL thumbnail keeps aspect.)
    ref_img = _load_ref_image(params.get("reference_image"))
    if ref_img is not None:
        w, h = ref_img.size
        if w * h > 768 * 768:
            log.info("Reference image %dx%d exceeds the 16 GB card's KV budget — downscaling to ≤768² px", w, h)
            ref_img.thumbnail((768, 768))

    try:
        result = pipe(
            prompt_embeds           = embeds["prompt"],
            negative_prompt_embeds  = embeds["negative"],
            image                   = ref_img,
            width                   = int(params.get("width",  1024)),
            height                  = int(params.get("height", 1024)),
            num_inference_steps     = int(params.get("num_inference_steps", 20)),
            guidance_scale          = guidance,
            generator               = generator,
        )
    finally:
        # Release the encoder (reference drop + collect — .to("cpu") alone
        # doesn't free the device_map NF4 module). No-op if never loaded.
        _park_klein_encoder(pipe)

    final_params = {**params, "seed": seed}
    if expanded:
        final_params = {**final_params, "expanded_prompt": expanded}
    return save_images(result.images, "flux2klein", final_params)


# ---------------------------------------------------------------------------
# FLUX.2 Klein 9B-KV (GGUF Q4_K_M)
# ---------------------------------------------------------------------------

_FLUX2KLEIN9B_GGUF: dict[str, tuple[str, str]] = {
    "Q4_K_M": ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q4_K_M.gguf"),
}

# Official black-forest-labs/FLUX.2-klein-9b-kv repo is gated (token lacks
# access), so the transformer comes from the public QuantStack GGUF.  That
# GGUF carries no `mmdit_*` config metadata, and diffusers' auto-config
# inference (DIFFUSERS_DEFAULT_PIPELINE_PATHS) would pick the wrong FLUX.2-dev
# 32B config, so we derive the config from the actual GGUF tensor shapes and
# pass it explicitly:
#
#   double_blocks 0..7   (8 layers)   qkv [12288,4096] -> hidden 4096, 32 heads x 128
#   single_blocks 0..23  (24 layers)  linear1 [36864,4096] = qkv 12288 + mlp 24576
#   img_mlp.0 [24576,4096]            -> mlp_ratio 3.0 (diffusers SwiGLU doubles: 4096*3*2)
#   txt_in [12288,4096]               -> joint_attention_dim 12288 (= 3 x Qwen3-8B hidden 4096)
#   time_in [256,4096]                -> timestep_guidance_channels 256
#   guidance_in absent                -> guidance_embeds false (distilled)
_FLUX2KLEIN9B_CONFIG: dict = {
    "_class_name":              "Flux2Transformer2DModel",
    "_diffusers_version":       "0.37.0.dev0",
    "attention_head_dim":       128,
    "axes_dims_rope":           [32, 32, 32, 32],
    "eps":                      1e-06,
    "guidance_embeds":          False,
    "in_channels":              128,
    "joint_attention_dim":      12288,
    "mlp_ratio":                3.0,
    "num_attention_heads":      32,
    "num_layers":               8,
    "num_single_layers":        24,
    "out_channels":             None,
    "patch_size":               1,
    "rope_theta":               2000,
    "timestep_guidance_channels": 256,
}


def _flux2klein9b_config_dir() -> str:
    """Return (creating if needed) the local transformer config dir for the GGUF."""
    cfg_dir = os.path.join(GGUF_ROOT, "flux2klein9b", "transformer_cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    import json
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(_FLUX2KLEIN9B_CONFIG, f, indent=2)
    return cfg_dir


def _load_flux2klein9b(quant: str = "Q4_K_M"):
    import torch
    from diffusers import Flux2KleinKVPipeline, Flux2Transformer2DModel
    from diffusers.models import AutoencoderKLFlux2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, Qwen2TokenizerFast

    use_quant = quant or "Q4_K_M"   # resolve the caller's "" to the engine default…
    if use_quant not in _FLUX2KLEIN9B_GGUF:
        raise RuntimeError(
            f"FLUX.2 Klein 9B-KV quant '{use_quant}' not recognised. "
            f"Valid options: {list(_FLUX2KLEIN9B_GGUF)}"
        )
    t0    = time.time()
    # NOTE: hf_hub 1.16.1 treats `token=True` as "a token is REQUIRED" and
    # raises LocalTokenNotFoundError before any cache check — and the systemd
    # service has no HF_TOKEN (shell /etc/environment isn't loaded by systemd).
    # All components below come from PUBLIC repos, so no token is needed.
    token = None
    repo  = ENGINES["flux2klein9b"].hf_repo_alt   # black-forest-labs/FLUX.2-klein-4B (shared components)

    # ── VRAM budget decision ─────────────────────────────────────────────────
    # GPU-only policy (2026-08-13): never fall back to CPU or system-RAM
    # offloading. Full-GPU budget (transformer 5.7 + Qwen3-8B NF4 4.5 + VAE
    # 0.3 + activations ≈ 12-13 GiB) needs the GPU almost to itself — the TTS
    # engine containers (~3.4 GiB resident when loaded) are evicted first;
    # they lazy-reload on their next TTS request. VRAM headroom is enforced
    # centrally in _ensure_engine → _ensure_vram_headroom (14800 MiB need,
    # below the ~15350 MiB max achievable free — TTS containers' CUDA
    # contexts never fully release).

    # ── Transformer (GGUF Q4_K_M, explicit derived config) ────────────────
    repo_id, fname = _FLUX2KLEIN9B_GGUF[use_quant]
    gguf_path = _ensure_gguf(repo_id, fname, os.path.join(GGUF_ROOT, "flux2klein9b"))
    log.info("Loading FLUX.2 Klein 9B-KV transformer from GGUF — quant=%s …", use_quant)
    transformer = Flux2Transformer2DModel.from_single_file(
        gguf_path,
        config              = _flux2klein9b_config_dir(),
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    )
    if GPU_ONLY:
        transformer = transformer.to("cuda")
    else:
        from diffusers.hooks import apply_group_offloading
        log.info("Applying leaf-level group offloading to FLUX.2 Klein 9B-KV transformer …")
        apply_group_offloading(
            transformer,
            onload_device  = torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type   = "leaf_level",
            use_stream     = False,
        )

    # ── Text encoder (Qwen3-8B, hidden 4096 → 3 x 4096 = 12288 joint dim) ──
    # GPU-only mode loads it LAZILY: prompt embeddings are cached to disk
    # (see _klein_prompt_embeds), so the ~4.5 GiB encoder only loads for
    # prompts never seen before. Resident set without it: ~5.7 GiB. Its freed
    # blocks stay pooled at the driver level after a miss (see image_lab.py
    # note) — reusable by torch, invisible to nvidia-smi until a full unload.
    if GPU_ONLY:
        text_encoder = None
    else:
        log.info("Loading Qwen3-8B text encoder (BF16, CPU) …")
        text_encoder = AutoModel.from_pretrained(
            "Qwen/Qwen3-8B",
            device_map = "cpu",
            torch_dtype = torch.bfloat16,
        )
        from diffusers.hooks import apply_group_offloading
        log.info("Applying leaf-level group offloading to Qwen3-8B text encoder …")
        apply_group_offloading(
            text_encoder,
            onload_device  = torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type   = "leaf_level",
            use_stream     = False,
        )

    # ── Shared klein components (from the accessible klein-4B repo) ────────
    tokenizer  = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen3-8B")
    vae        = AutoencoderKLFlux2.from_pretrained(
        repo, subfolder = "vae", torch_dtype = torch.bfloat16, token = token,
    )
    scheduler  = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder = "scheduler", torch_dtype = torch.bfloat16, token = token,
    )

    # ── Pipeline assembly ──────────────────────────────────────────────────
    log.info("Assembling FLUX.2 Klein 9B-KV pipeline …")
    pipe = Flux2KleinKVPipeline(
        transformer  = transformer,
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        vae          = vae,
        scheduler    = scheduler,
        is_distilled = True,
    )
    if GPU_ONLY:
        # Everything is already on CUDA (transformer + VAE); the NF4 encoder
        # loads on demand for cache-miss prompts and is released afterwards.
        log.info("GPU-only mode enabled: moving FLUX.2 Klein 9B-KV to CUDA …")
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # The pipeline resolves its execution device from the FIRST module's
    # actual location — `_execution_device` falls back to `self.device`,
    # which reads the __init__ signature order (text_encoder first), not
    # `hf_device_map` or `_exclude_from_cpu_offload`. Generation offloads
    # the text encoder to CPU after encoding (VRAM budget for ref-image KV
    # caching); without this override, "cpu" propagates into latent
    # placement and the CUDA VAE dies with a conv weight/input mismatch.
    # Precomputed embeds mean the encoder is never called during
    # generation, so the pipeline genuinely executes on cuda while the
    # encoder sits on CPU — and reports cuda anyway when it is not
    # offloaded, so the override is never wrong.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    # Store the CALLER's quant ("" = default), not the resolved default —
    # _ensure_engine compares active_quant against the request verbatim, so
    # storing the resolved "Q4_K_M" made every subsequent default-quant
    # request "mismatch" and pay an unload+reload.
    STATE.loaded_model       = pipe
    STATE.active_engine      = "flux2klein9b"
    STATE.active_quant       = quant
    ENGINES["flux2klein9b"].loaded = True
    log.info("FLUX.2 Klein 9B-KV ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0,
             torch.cuda.memory_allocated() / 1024**3)


def _generate_flux2klein9b(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    generator = torch.Generator("cpu").manual_seed(seed)

    # Optional Gemini prompt expansion (needs GEMINI_API_KEY) — see the 4B
    # generator: richer captions through the same local encoder, identical
    # VRAM, and a different cache key.
    prompt = params["prompt"]
    expanded = _gemini_expand_prompt(prompt) if GPU_ONLY else None
    if expanded:
        log.info("Gemini expanded prompt: %r → %r", prompt, expanded)
        prompt = expanded

    # Prompt-embedding cache — see _klein_prompt_embeds. Step-distilled klein
    # runs without CFG, so only the positive embeds are needed.
    embeds = _klein_prompt_embeds(pipe, "flux2klein9b", {"prompt": prompt})

    # Ref token count is the other budget lever: each ref token costs ~0.5 MiB
    # of retained K/V across every denoising step. Cap the ref area so a full
    # landscape screenshot (1024×576, 9216 tokens) passes untouched while
    # square 1024² refs can't blow the card. (PIL thumbnail keeps aspect.)
    ref_img = _load_ref_image(params.get("reference_image"))
    if ref_img is not None:
        w, h = ref_img.size
        if w * h > 768 * 768:
            log.info("Reference image %dx%d exceeds the 16 GB card's KV budget — downscaling to ≤768² px", w, h)
            ref_img.thumbnail((768, 768))

    try:
        # Note: no guidance_scale — step-distilled klein models run without CFG
        # (guidance=None in the pipeline). The UI still shows the param for
        # consistency with the other klein engine.
        result = pipe(
            prompt_embeds       = embeds["prompt"],
            image               = ref_img,
            width               = int(params.get("width",  1024)),
            height              = int(params.get("height", 1024)),
            num_inference_steps = int(params.get("num_inference_steps", 4)),
            generator           = generator,
        )
    finally:
        # Release the encoder (reference drop + collect — .to("cpu") alone
        # doesn't free the device_map NF4 module). No-op if never loaded.
        _park_klein_encoder(pipe)

    final_params = {**params, "seed": seed}
    if expanded:
        final_params = {**final_params, "expanded_prompt": expanded}
    return save_images(result.images, "flux2klein9b", final_params)


def _probe_flux2klein9b():
    try:
        from diffusers import Flux2KleinKVPipeline, Flux2Transformer2DModel  # noqa: F401
        from transformers import AutoModel, Qwen2TokenizerFast               # noqa: F401
        ENGINES["flux2klein9b"].available = True
    except Exception as exc:
        ENGINES["flux2klein9b"].available = False
        ENGINES["flux2klein9b"].error     = str(exc)
        log.warning("FLUX.2 Klein 9B-KV unavailable: %s", exc)


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
    if GPU_ONLY:
        log.info("GPU-only mode enabled: moving SD 3.5 Large pipeline to CUDA …")
        pipe = pipe.to("cuda")
    else:
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
    if GPU_ONLY:
        log.info("GPU-only mode enabled: moving Wan T2V pipeline to CUDA …")
        pipe_t2v = pipe_t2v.to("cuda")
    else:
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
        if GPU_ONLY:
            log.info("GPU-only mode enabled: moving Wan I2V pipeline to CUDA …")
            pipe_i2v = pipe_i2v.to("cuda")
        else:
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
# Ideogram 4
# ---------------------------------------------------------------------------

def _load_ideogram4(quant: str = "nf4"):
    import importlib
    ideogram4_engine = importlib.import_module("ideogram4_lab_engine")
    t0 = time.time()
    log.info("Loading Ideogram 4 (quant=%s) …", quant)
    pipe = ideogram4_engine.load_ideogram4(quant=quant)
    STATE.loaded_model  = pipe
    STATE.active_engine = "ideogram4"
    STATE.active_quant  = quant
    ENGINES["ideogram4"].loaded = True
    log.info("Ideogram 4 ready (quant=%s) in %.1f s", quant, time.time() - t0)


def _generate_ideogram4(params: dict) -> list[dict]:
    import importlib
    ideogram4_engine = importlib.import_module("ideogram4_lab_engine")

    pipe   = STATE.loaded_model
    prompt = params["prompt"]

    # Magic prompt: when enabled, the engine expands plain text → JSON via DeepSeek
    use_magic = bool(params.get("use_magic_prompt", False))

    # Resolve steps: 0 means "use preset default"
    steps = int(params.get("num_inference_steps", 0))
    if steps == 0:
        steps = None

    images, caption, seed_used = ideogram4_engine.generate_ideogram4(
        pipe,
        prompt=prompt,
        width=int(params.get("width", 1024)),
        height=int(params.get("height", 1024)),
        preset=params.get("preset", "V4_DEFAULT_20"),
        num_steps=steps,
        guidance_scale=float(params.get("guidance_scale", 7.0)),
        mu=float(params.get("mu", 0.0)),
        std=float(params.get("std", 1.75)),
        seed=int(params.get("seed", -1)),
        use_magic_prompt=use_magic,
        magic_prompt_aspect_ratio=params.get("magic_prompt_aspect_ratio", "1:1"),
    )

    # Record the ACTUAL seed used — auto seed is now randomized server-side,
    # so the gallery/API shows the real seed for reproducibility.
    final_params = {**params, "seed": seed_used, "caption": caption}
    return save_images(images, "ideogram4", final_params)


# ---------------------------------------------------------------------------
# Availability probe (called at startup)
# ---------------------------------------------------------------------------

def probe_availability():
    """
    Check which engines can be loaded (packages importable, not that models
    are downloaded — that happens lazily on first generate call).
    """
    _probe_flux2klein()
    _probe_flux2klein9b()
    _probe_sd35()
    _probe_wan()
    _probe_ideogram4()
    _strip_missing_nvfp4_options()


def _nvfp4_shard_size_bytes(transformer_dir: str) -> int:
    """Return total bytes of all .bin shard files under transformer_dir."""
    total = 0
    for fname in os.listdir(transformer_dir) if os.path.isdir(transformer_dir) else []:
        if fname.endswith(".bin"):
            total += os.path.getsize(os.path.join(transformer_dir, fname))
    return total


def _strip_missing_nvfp4_options():
    """Remove the nvfp4 quant choice for any engine whose saved files are missing or corrupted.

    A save is considered corrupted if the total .bin shard size is < MIN_SHARD_BYTES,
    which catches the meta-tensor bug from nvfp4_save.py when device_map='auto' caused
    disk offloading and most weights were serialised as empty meta tensors.
    """
    # Minimum total .bin shard size (bytes) to consider a save valid.
    # Corrupted SD3.5 save had 530 MB (only non-meta tensors); valid NVFP4 saves are ≥ 2 GB.
    MIN_SHARD_BYTES = 1 * 1024 ** 3  # 1 GB

    checks = {
        "sd35":  os.path.join(NVFP4_ROOT, "sd35",    "transformer"),
        "wan":   os.path.join(NVFP4_ROOT, "wan-t2v", "transformer"),
    }
    for engine_key, transformer_dir in checks.items():
        config_path = os.path.join(transformer_dir, "config.json")
        missing = not os.path.isfile(config_path)
        if not missing:
            shard_size = _nvfp4_shard_size_bytes(transformer_dir)
            if shard_size < MIN_SHARD_BYTES:
                missing = True
                log.warning(
                    "NVFP4 save for %s appears corrupted (shard size %.0f MB < 1 GB) "
                    "— removing from quant options. Re-run nvfp4_save.py to fix.",
                    engine_key, shard_size / 1024 ** 2,
                )
        if missing:
            for p in ENGINES[engine_key].params:
                if p.get("name") == "quant" and "options" in p:
                    p["options"] = [o for o in p["options"] if o.get("value") != "nvfp4"]
            if not log.isEnabledFor(logging.WARNING):
                log.info("NVFP4 not saved for %s — removed from quant options", engine_key)


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


def _probe_ideogram4():
    try:
        import importlib
        mod = importlib.import_module("ideogram4_lab_engine")
        result = mod.probe_ideogram4()
        if result["available"]:
            ENGINES["ideogram4"].available = True
        else:
            ENGINES["ideogram4"].available = False
            ENGINES["ideogram4"].error     = result.get("error", "unknown error")
            log.warning("Ideogram 4 unavailable: %s", result.get("error"))
    except Exception as exc:
        ENGINES["ideogram4"].available = False
        ENGINES["ideogram4"].error     = str(exc)
        log.warning("Ideogram 4 unavailable: %s", exc)


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
    if STATE.loading:
        raise RuntimeError("Model is currently loading. Retry in a few seconds.")

    quant = params.get("quant", "")
    STATE.generating = True
    try:
        try:
            _ensure_engine(engine_key, quant)
        except Exception:
            # A failed load (e.g. CUDA OOM mid-load) leaves partially-loaded
            # tensors pinned in the caching allocator — release them so the
            # card isn't bricked until the service restarts.
            _unload_current()
            raise
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
    try:
        _ensure_engine(engine_key)
    except Exception:
        # Same cleanup as generate() — a failed load must not leave
        # partially-loaded tensors pinned on the GPU.
        _unload_current()
        raise
    # Mark the model as freshly used so the idle-eviction loop doesn't
    # instantly recycle an API-preloaded model (last_used is otherwise only
    # touched by generate()).
    STATE.last_used = time.time()


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
    "flux2klein":    _load_flux2klein,
    "flux2klein9b":  _load_flux2klein9b,
    "sd35":          _load_sd35,
    "wan":           _load_wan,
    "ideogram4":     _load_ideogram4,
}

_GENERATORS = {
    "flux2klein":    _generate_flux2klein,
    "flux2klein9b":  _generate_flux2klein9b,
    "sd35":          _generate_sd35,
    "wan":           _generate_wan,
    "ideogram4":     _generate_ideogram4,
}
