"""
image_lab_engines.py — Load / unload / generate functions for all engines:
  flux2klein  — FLUX.2 Klein 4B (custom NF4 encoder assembly)
  flux2klein9b — FLUX.2 Klein 9B-KV GGUF (Q4_K_M) via QuantStack
  ideogram4   — Ideogram 4 (API)
  sana        — SANA 1.6B (Sprint + SANA 1.5 variants, variant rides `quant`)
  boogu       — Boogu-Image-0.1-Turbo-fp8 (CPU-offload exception; boogu_lab_engine)
  zimage      — Z-Image Turbo (GGUF ladder via jayn7 + lazy bnb Qwen3-4B encoder)
  qwenimage   — Qwen-Image 2512 (GGUF ladder via unsloth + bnb4 VL encoder,
                encode-without-transformer on cache miss)
  qwenimage-edit — Qwen-Image-Edit 2511 (GGUF Q4_K_S via unsloth + the SAME
                bnb4 VL encoder; image-conditioned embeds — prompt + ref bytes
                key the cache; consumes the uploaded reference image natively)
  hidream     — HiDream O1-Dev (OUT-OF-PROCESS ComfyUI sidecar, port 8188 —
                hidream_comfy_bridge.py)
  ernie       — ERNIE-Image-Turbo (Nunchaku-Lite NVFP4 primary, fp8 fallback)

Removed: sd35 + wan 2026-09-07 (see docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md);
flux2 (FLUX.2 [dev] 32B) removed 2026-08-13 — ~27 GB footprint cannot fit the
15.5 GiB card and the GPU-only policy forbids CPU fallback.

Model weights are downloaded from HuggingFace on first use (GGUF files cached
under GGUF_ROOT, HF repos under HF_HOME); text encoders / VAEs / schedulers
come from the same repos. Nothing uses preq_save.py shared dirs any more.
"""

from __future__ import annotations
import gc
import hashlib
import logging
import os
import time
from typing import Any, Optional

from image_lab_config import ENGINES, STATE, OUTPUT_ROOT, HF_TOKEN, HF_HOME, GPU_ONLY
from image_lab_utils import free_vram, random_seed, save_image, save_images

# Local directory for cached GGUF model files
GGUF_ROOT = "/opt/arthur-img-models/gguf"

# TTS orchestrator's whole-card eviction endpoint — the engine containers
# publish no host ports, so the orchestrator (port 8009) is the canonical
# broker for evicting TTS models off the shared GPU (see _evict_tts_engines).
# Also used by the dispatch-layer /evict-all for the UI "Evict VRAM" button.
TTS_EVICT_ALL_URL = "http://localhost:8009/evict-all"
# Orchestrator status — the warm-gen tenant guard (_tts_tenant_loaded) asks
# the orchestrator whether any TTS model is resident instead of inferring it
# from free VRAM (a free-memory floor can't tell a foreign tenant apart from
# this process's own post-draw pooled blocks).
TTS_STATUS_URL    = "http://localhost:8009/status"

# Headless ComfyUI sidecar for the HiDream O1 engine (arthur-comfy.service,
# own venv, port 8188 — see hidream_comfy_bridge.py). The lab and the sidecar
# share the card OUT-OF-PROCESS: VRAM hand-off goes through the bridge module
# (_poke_comfy_free here, comfy_free in the bridge), never in-process tensors.
IMGLAB_COMFY_URL = os.environ.get("IMGLAB_COMFY_URL", "http://127.0.0.1:8188")

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


# Encoder-load headroom (MiB free BEFORE the NF4 quantize-on-load starts).
# BitsAndBytes streams bf16 shards to CUDA, quantises, frees — the GPU peak is
# the staging burst, ~2.75 GiB for klein-4B's Qwen3-4B, ~5.2 GiB for
# klein9b's Qwen3-8B (measured: the 8B load drove the process from 7.55 GiB
# to 12.58 GiB before OOMing on a 96 MiB alloc, 2026-09-05). Threshold sits
# between two measured post-transformer-load states on the shared card:
#   ~4,940 MiB free — a TTS container additionally holds its ~2.2 GiB model
#                     (e.g. engine-current omnivoice): cache-miss flows evict
#                     it via the orchestrator, landing back at ~7.1 GiB.
#   ~7,100 MiB free — TTS containers idle (contexts only, ~0.9 GiB, NOT
#                     further evictable): the 8B staging burst fits — fresh
#                     prompts must NOT be blocked here.
# So: evict when free < 6000 (frees any resident TTS model), proceed above.
# The transformer-load gate (_VRAM_NEED_MB 10500) passes with a TTS model
# resident (10,748 ≥ 10,500), leaving ~4.9 GiB for the encode — verified OOM
# 2026-09-05. Cache-miss flows pay a TTS eviction when a model is resident;
# cache hits never do.
_ENCODER_NEED_MB: dict[str, int] = {
    "flux2klein":   4000,  # Qwen3-4B — ~2.75 GiB staging + margin
    "flux2klein9b": 6000,  # Qwen3-8B — ~5.2 GiB staging; evicts a resident
                           # TTS model, spares the idle-contexts state
    "zimage":       4000,  # Qwen3-4B (Z-Image's encoder) — same ~2.75 GiB
                           # staging burst as flux2klein; encode runs with the
                           # ~4.6 GiB GGUF transformer resident
    "qwenimage":    8000,  # Qwen2.5-VL-7B-family bnb4 encoder — measured
                           # 7.3 GiB resident (2026-09-08); the miss flow
                           # unloads the ~12 GiB GGUF transformer first so
                           # free VRAM ≈ idle (~14.2 GiB) and this gate only
                           # guards the case where a TTS model is resident
    "qwenimage-edit": 8000,  # SAME OzzyGT bnb4 encoder as qwenimage (probe-
                             # verified 2026-09-08: edit shares the base's
                             # Qwen2.5-VL text encoder); encode peak 6.31 GiB
                             # torch-alloc with the transformer unloaded
}


def _ensure_encoder_headroom(engine_key: str) -> None:
    """Make room for the on-demand text-encoder load before it allocates.

    Same discipline as _ensure_vram_headroom but for the lazy NF4 encoder:
    evict the TTS containers when free VRAM is short, else raise a clear
    error instead of OOMing mid-quantise (GPU-only policy — no CPU
    fallback). Runs only on embed-cache misses.
    """
    if not GPU_ONLY:
        return
    need_mb = _ENCODER_NEED_MB.get(engine_key)
    if need_mb is None:
        return
    import torch
    free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mb >= need_mb:
        return
    log.info("Encoder load needs %d MiB free (only %d) — evicting TTS engine containers …",
             need_mb, free_mb)
    free_mb = _evict_tts_engines()
    if free_mb < need_mb:
        label = (ENGINES.get(engine_key).label if ENGINES.get(engine_key)
                 else engine_key)
        raise RuntimeError(
            f"Text-encoder load for {label} needs ~{need_mb // 1024} GiB free "
            f"VRAM; only {free_mb} MiB available after evicting the TTS engine "
            f"containers. (GPU-only policy — no CPU offloading.)")


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
    elif engine_key == "zimage":
        # Z-Image's encoder lives in-repo (Tongyi-MAI/Z-Image-Turbo subfolder
        # "text_encoder") and is the SAME Qwen3-4B architecture — the NF4
        # on-demand load + encode + park flow is identical to the klein path.
        repo, extra, label = "Tongyi-MAI/Z-Image-Turbo", {"subfolder": "text_encoder"}, "Qwen3-4B (Z-Image)"
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
    _ensure_encoder_headroom(engine_key)
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

log = logging.getLogger("image_lab")

# ---------------------------------------------------------------------------
# GGUF download helper
# ---------------------------------------------------------------------------

def _ensure_gguf(repo_id: str, filename_in_repo: str, local_dir: str) -> str:
    """
    Return the local path to a GGUF file.  If not present, downloads it from
    HuggingFace Hub into `local_dir` (preserving any sub-folder in the name).
    """
    # filename_in_repo may include a sub-folder; it is preserved on disk.
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


def _component_config_dir(repo_id: str, subfolder: str, engine_key: str) -> str:
    """Local transformer-config dir for a GGUF single-file load.

    GGUF files carry the raw weights but not always a diffusers-side
    `config.json` the from_single_file machinery can auto-derive (the klein9b
    GGUF needed an explicit hand-derived config for that reason). This helper
    fetches the ARCHITECTURE's own config.json from the model repo into
    GGUF_ROOT/<engine_key>/transformer_cfg/ and hands the dir to
    `<Model>2DModel.from_single_file(..., config=cfg_dir)`. All ladder quants
    share the one architecture → one config dir per engine.
    """
    import shutil as _shutil
    from huggingface_hub import hf_hub_download
    cfg_dir  = os.path.join(GGUF_ROOT, engine_key, "transformer_cfg")
    target   = os.path.join(cfg_dir, "config.json")
    if os.path.isfile(target):
        return cfg_dir
    os.makedirs(cfg_dir, exist_ok=True)
    log.info("Fetching %s %s/config.json → %s …", repo_id, subfolder, cfg_dir)
    fetched = hf_hub_download(
        repo_id   = repo_id,
        filename  = "config.json",
        subfolder = subfolder,
        token     = None,          # public repo — see the hf_hub 1.16.1 note
    )
    # Copy out of the HF snapshot cache into the GGUF dir, atomically.
    _shutil.copyfile(fetched, target + ".tmp")
    os.replace(target + ".tmp", target)
    return cfg_dir


# ---------------------------------------------------------------------------
# VRAM lifecycle helpers
# ---------------------------------------------------------------------------

def _unload_current():
    """Destroy the currently-loaded pipeline and free VRAM."""
    if STATE.active_engine is None:
        # Nothing resident in-process — but the ComfyUI sidecar may still hold
        # its ~8 GiB from before a lab restart, so hand the card back anyway.
        _poke_comfy_free()
        # Still force-release THIS process's pooled torch blocks: after a
        # heavy engine's encode/unload dance the caching allocator can hold
        # ~4.4 GiB while no engine is tracked as resident (2026-09-08
        # deadlock) — empty_cache returns those blocks to the driver, so an
        # evict/unload call must never no-op on them.
        log.info("Nothing resident — releasing pooled VRAM (gc + empty_cache + ipc_collect) …")
        free_vram()
        return
    log.info("Unloading engine: %s (quant=%s)", STATE.active_engine, STATE.active_quant)
    if STATE.active_engine == "hidream":
        log.info("Unloading HiDream: releasing the ComfyUI sidecar's models …")

    # Explicitly move GPU components to CPU before dropping references.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prevents the caching
    # allocator from releasing memory unless we explicitly offload first.
    _gpu_attrs = [
        'conditional_transformer', 'unconditional_transformer',
        'text_encoder', 'autoencoder', 'transformer',
        'vae', 'text_encoder_2',
        # Boogu pipeline components — under its CPU-offload exception nothing
        # is GPU-resident at rest, but a mid-generate failure can leave them
        # staged on the card; the failure path must release them.
        'mllm', 'processor',
    ]
    for attr in _gpu_attrs:
        comp = getattr(STATE.loaded_model, attr, None)
        if comp is not None:
            try:
                setattr(STATE.loaded_model, attr, comp.to('cpu'))
            except Exception:
                pass
            try:
                setattr(STATE.loaded_model, attr, None)
            except Exception:
                pass

    STATE.loaded_model  = None
    STATE.active_engine = None
    STATE.active_quant  = ""
    free_vram()
    for k in ENGINES:
        ENGINES[k].loaded = False
    # Any non-hidream unload frees the card for whoever loads next — if that
    # is the ComfyUI sidecar, hand it a clean card (its own models may also
    # still be resident from an earlier hidream generation).
    _poke_comfy_free()


def _ensure_engine(key: str, quant: str = ""):
    """Load engine `key` with `quant` into VRAM, evicting whatever is currently loaded."""
    if STATE.active_engine == key and STATE.active_quant == quant:
        return  # already loaded with the same quantization
    _unload_current()
    # GPU-only policy: make room before allocating — evict the TTS engine
    # containers (see _ensure_vram_headroom). Runs after _unload_current so
    # the previous image engine's VRAM is already accounted for.
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
    """Evict every TTS engine container's resident model to free GPU memory.

    GPU-only policy (2026-08-13): image engines NEVER fall back to CPU
    rendering or system-RAM offloading. When a loader needs more VRAM than is
    free, the TTS containers' resident models are evicted first — they
    lazy-reload on their next TTS request (a few seconds of added latency).
    Returns the free VRAM in MiB after the evictions settle.

    Routing: the engine containers live on the compose bridge and publish NO
    host ports, so direct localhost:PORT /evict calls have been connection-
    refused (and silently swallowed) since the move to containers. The
    orchestrator is the canonical broker — it sits on the bridge, knows the
    containers by service name, and publishes 8009 on the host.
    """
    import json as _json
    import time as _t
    import urllib.request as _urllib
    import torch
    try:
        req = _urllib.Request(TTS_EVICT_ALL_URL, data=b"", method="POST")
        with _urllib.urlopen(req, timeout=60) as resp:
            body = resp.read().decode().strip()
        log.info("TTS evict-all: %s", body[:200])
        info = _json.loads(body)
        if info.get("evicted_count"):
            log.info("TTS evict-all: %d container(s) evicted, %s freed",
                     info["evicted_count"],
                     info.get("freed_mb_total", "?"))
    except Exception as exc:
        # Orchestrator down / eviction failed — never OOM silently on top of
        # it: log loudly and let the VRAM gate raise its clear error.
        log.warning("TTS evict-all via orchestrator failed: %s", exc)
    _t.sleep(2)  # let CUDA return the freed memory to the driver
    return torch.cuda.mem_get_info()[0] // (1024 * 1024)


# ---------------------------------------------------------------------------
# Warm-gen tenant guard — whole-card engines whose gen needs a clear card
# ---------------------------------------------------------------------------

# Engines whose measured gen working set only fits on a cleared card (TTS
# tenants evicted; ambient ≤ ~1 GiB). A WARM draw (engine already resident)
# never re-runs the load gate, so a TTS model that loaded since the last
# draw would crowd the card mid-gen — for these engines the warm path evicts
# first (see _evict_tts_tenant_if_loaded). Only whole-card engines with
# measured clear-card requirements belong here; add as each is measured.
_WARM_TENANT_GUARD_KEYS: frozenset[str] = frozenset({
    "flux2klein9b-nvfp4",   # FHD-class gen needs the clear card (09-09 ladder)
})


def _tts_tenant_loaded() -> bool:
    """True when any TTS engine container has a model resident right now.

    Queries the orchestrator /status (the canonical broker for the shared
    16 GB card — engine containers publish no host ports). Deterministic
    tenant check: a free-VRAM floor is unusable warm because the engine's
    own pooled blocks from the previous draw read as "occupied".
    """
    import json as _json
    import urllib.request as _urllib
    try:
        req = _urllib.Request(TTS_STATUS_URL)
        with _urllib.urlopen(req, timeout=5) as resp:
            models = _json.loads(resp.read().decode()).get("models", {})
    except Exception as exc:
        # Orchestrator down → fail open (today's behavior); the cold load
        # gate still protects the next engine load.
        log.warning("TTS tenant check failed (%s) — assuming none loaded", exc)
        return False
    for e in models.values():
        if not isinstance(e, dict):
            continue
        if e.get("loaded_model") or e.get("status") in ("loaded", "loading"):
            return True
    return False


def _evict_tts_tenant_if_loaded(engine_key: str) -> None:
    """Evict a TTS tenant that loaded since the last draw (warm-gen guard).

    Only engines in _WARM_TENANT_GUARD_KEYS pay the check — one cheap
    orchestrator /status call per warm gen. Measured 2026-09-09 (OmniVoice
    resident beside NVFP4): card free drops ~3.1 → ~1.5 GiB and anything
    above ~1.04 MP OOMs mid-gen (503). Evicting BEFORE the gen keeps the
    engine on the clear card it was calibrated on; the TTS model pays a
    cold reload on its next synth (accepted trade).
    """
    if engine_key not in _WARM_TENANT_GUARD_KEYS:
        return
    if not _tts_tenant_loaded():
        return
    log.info("%s resident + TTS tenant loaded — evicting TTS before warm gen",
             engine_key)
    _evict_tts_engines()


def _compact_after_draw(engine_key: str) -> None:
    """Release a guard-set engine's post-draw working set; model stays resident.

    The gen's transient tensors die with the call but their blocks stay pooled
    in torch's caching allocator — after an FHD draw the card reads
    ~14.7-15.5 GiB used even though the engine is idle, and that pool OOM'd
    OmniVoice's reload (1.14 GiB chunk vs 794 MiB free, 2026-09-09).
    empty_cache returns the pool to the driver WITHOUT unloading the model,
    settling the card back toward the measured fresh-load state (~13.1-13.2
    GiB used / ~3.1 GiB free) so a ~2 GiB TTS tenant can co-load beside the
    resident engine — the cohabitation budget (user design, 2026-09-09).
    Nunchaku scratch sits OUTSIDE torch's allocator; the log's free-MiB delta
    (driver-level cudaMemGetInfo) shows what actually returned, and the next
    draw re-ascends from the cold pool (~0.1-0.2 s — negligible per draw).
    """
    if engine_key not in _WARM_TENANT_GUARD_KEYS:
        return
    import torch
    try:
        free_before = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        free_vram()
        free_after = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    except Exception as exc:
        # Never fail a draw that already succeeded — compaction is best-effort.
        log.warning("%s post-draw compaction failed: %s", engine_key, exc)
        return
    log.info("%s post-draw compaction: released %d MiB (free %d → %d MiB), "
             "engine kept resident", engine_key, free_after - free_before,
             free_before, free_after)


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
    # flux2klein9b-nvfp4 (nunchaku-lite NVFP4 + RESIDENT bnb4 Qwen3-8B): the
    # encoder never parks (unlike the Q6 lane's lazy encode) → whole-card
    # engine, qwenimage-edit class. Measured 2026-09-08/09 on the 5060 Ti:
    # load 9.1 s → 11.22 GiB torch-alloc at ready, but the driver footprint
    # at load is ~11.5-12.3 GiB (nunchaku scratch sits outside torch's
    # allocator — ernie precedent). The gen working set is canvas-scaled and
    # tops out at FULL-HD: in-service peaks 15,371-15,697 MiB on a clear
    # card (1920×1072 verified 2026-09-09, dev 4/4 + lab re-draw 2/2; every
    # out-of-band / second-process FHD draw OOMs at the 15.48 GiB per-process
    # torch wall). 2026-09-09 ladder with a TTS tenant resident (OmniVoice,
    # ~2.4 GiB ambient): 720×1440 / 1360×768 still fit (~15.7-15.8 GiB card
    # peaks, ≤ ~600 MiB margin) but 1024×1024 already died on a 512 MiB
    # alloc and 1536×1024 / FHD were guaranteed mid-gen OOMs — the old 13000
    # passed that state untouched (13,876 MiB free) and let the gen walk
    # into the OOM. 15000 demands a genuinely CLEAR card: it sits ~350-450
    # MiB under the post-eviction max free (~15.35-15.45 GiB) and above
    # every TTS-resident free reading, so any load auto-evicts the TTS
    # tenant (OmniVoice pays a ~30 s cold reload on its next synth —
    # accepted 2026-09-09) and a cleared card passes untouched. Warm draws
    # are covered by _evict_tts_tenant_if_loaded. Idle TTS containers whose
    # contexts alone hold ~1.5-2 GiB now 503 the load honestly instead of
    # OOMing mid-gen.
    "flux2klein9b-nvfp4": 15000,
    # ideogram4 (own pip package, bnb nf4): load-ready state is cheap — 5,953
    # MiB used / 9,895 free after the loader's offloads (2026-09-08), so the
    # gate governs LOAD and stays moderate. But CFG generation pulls the
    # unconditional transformer AND encoder back onto the card → process peak
    # 14,036 MiB (nvidia-smi) — the engine needs the whole card mid-gen
    # (single-engine-at-a-time dispatch handles coexistence). Warm gen 236 s
    # @1024²/28 steps; same-seed reruns differ (bnb nf4 matmul noise — an
    # engine property, like ernie's nunchaku). Cold load ~500 s the first run
    # after a cache heal (2×5.2 GB re-download); ~90 s thereafter.
    "ideogram4":    12500,
    # SANA: whole bf16 pipeline incl. Gemma-2-2B encoder resident. Measured
    # 2026-09-07 (1024², both variants): resident 8,968 MiB, gen proc peak
    # 9,886 MiB, device peak 10,866 MiB. 11500 still sits between the
    # TTS-resident free state (~10.8 GiB → evicts TTS on load) and the idle
    # free state (~14.3 GiB → passes untouched); the headroom above the device
    # peak covers 2048² activation growth.
    "sana":         11500,
    # Boogu (CPU-offload exception — the ONLY engine allowed to stage off
    # GPU): load is cheap (modules stage on CPU) but the first generate moves
    # the whole fp8 mllm AND bf16 DiT onto the card. Measured GPU peak
    # 2026-09-07: 12,640 MiB (1024², 4 steps, TTS evicted) → 13200 clears it
    # yet still sits under the idle free state, so an idle TTS is left alone.
    "boogu":        13200,
    # ── 2026-09-07 additions — all four "calibrate live" pending Phase E ──────
    "zimage":       10500,  # GGUF Q4_K_M transformer + VAE + activations.
                            # Measured 2026-09-08: 4.90 GiB CUDA at ready (16.3 s
                            # load), warm gen 48 s @1024²/8 steps on diffusers
                            # 0.40. Lazy Qwen3-4B encode staging on cache miss
                            # (~10.6 GiB worst-case peak, klein-4B precedent).
    "qwenimage":    14000,  # Q4_K_S only (see _QWENIMAGE_GGUF — higher tiers
                            # OOM). Resident ~12.0 GiB torch-alloc at ready;
                            # measured gen peak 12.19 GiB torch-alloc / 14,166
                            # MiB driver-accounted (nvidia-smi) at 1024²/20
                            # steps CFG 4.0 (2026-09-08) → the pipeline needs
                            # the whole card: idle free ~14.2 GiB passes the
                            # gate untouched, a resident TTS model trips it
                            # and evicts. Cache-miss encodes run with the
                            # transformer UNLOADED.
    "qwenimage-edit": 14000,  # Q4_K_S only (see _QWENIMAGE_EDIT_GGUF).
                            # Two-channel edit conditioning (vision tokens +
                            # VAE ref latents) makes the DiT seq ~8.2K tokens
                            # at 1024² → resident ~11.81 GiB torch-alloc, gen
                            # peak 12.70 GiB torch-alloc / 15.40 GiB driver-
                            # total / 14.64 GiB pid (nvidia-smi, 2026-09-08
                            # probe, 1024² AND 720×1440 / 20 st / CFG 4.0 —
                            # margin 0.53 GiB). The whole card, like
                            # qwenimage: idle free passes untouched, a
                            # resident TTS model trips the gate and evicts.
    "hidream":      11500,  # Out-of-process ComfyUI sidecar (fp8_scaled ~8.1
                            # GiB + pixel-space activations at ≤2048²). The
                            # gate runs after _unload_current pokes comfy
                            # /free, so it measures the true idle free state.
                            # Calibrate live against the sidecar's peak.
    "ernie":        13000,  # NVFP4 transformer 4.71 + bnb4 encoder 2.74 + VAE.
                            # Measured 2026-09-08 on the 5060 Ti: load 31.2 s,
                            # 7.18 GiB torch-alloc at ready BUT 12,418 MiB
                            # process resident (nunchaku scratch sits outside
                            # torch's allocator); card peak 12,993 MiB at
                            # 1024²/8 steps CFG 1.0. 13000 forces a TTS
                            # eviction when TTS holds ~10.8 GiB, passes the
                            # idle state (~14.2 GiB) untouched.
}

def _ensure_vram_headroom(need_mb: int, key: str) -> None:
    """Make sure `need_mb` MiB of VRAM are free before loading engine `key`.

    Escalation chain (GPU-only policy — image engines never fall back to CPU):
      1. Evict the TTS engine containers (they lazy-reload on their next
         TTS request, a few seconds of added latency).
      2. Still short? If nothing is resident in-process, force-release THIS
         process's pooled torch blocks (gc + empty_cache + ipc_collect) and
         re-check before giving up — the 2026-09-08 deadlock: after a heavy
         engine's encode/unload dance the caching allocator kept ~4.4 GiB
         pooled with STATE.active_engine None, so mem_get_info read
         ~10,452 MiB free and every gate failed ~48 MiB short (503) until a
         process restart. The reclaim returns those blocks to the driver
         when nothing live holds them.
      3. Raise a clear error instead of OOMing mid-load. (The LLM container
         was retired 2026-08-23 — the old stop-LLM last-resort is gone.)
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
    if free_mb < need_mb and STATE.active_engine is None:
        # Nothing left to unload in-process — the shortfall may be pooled
        # blocks the caching allocator can still return to the driver (see
        # docstring). Force the release once, then re-check.
        import time as _t
        log.info("Still short (%d MiB < %d) with no engine resident — forcing "
                 "gc + empty_cache + ipc_collect …", free_mb, need_mb)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        _t.sleep(1)   # let the driver settle after the release
        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        log.info("Post-reclaim free VRAM: %d MiB", free_mb)
    if free_mb < need_mb:
        label = ENGINES.get(key).label if ENGINES.get(key) else key
        raise RuntimeError(
            f"{label} requires ≥ {need_mb:,} MiB free; only {free_mb:,} MiB "
            f"available after evicting the TTS engine containers. "
            f"(GPU-only policy — no CPU offloading.)")



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
    # containers before allocating).

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
            log.info("reference_image scaled to %dx%d (what the model sees)", ref_img.width, ref_img.height)

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

# GGUF quant ladder for the same KV transformer — all entries share the
# architecture-derived config, only the file differs. Quant chosen via the
# API `quant` form field; "" resolves to the loader default (Q6_K since
# 2026-09-04 — the identity-preservation A/B pointed at Q4_K_M's 0.70-0.93
# cosine band; Q6_K is the quality-per-GB sweet spot on the 16 GB card).
# Q8_0 needs TTS containers evicted (~14 GiB total). Q2_K/Q3_K_S exist but
# blur facial detail — deliberately not offered.
_FLUX2KLEIN9B_GGUF: dict[str, tuple[str, str]] = {
    "Q3_K_M": ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q3_K_M.gguf"),
    "Q4_K_M": ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q4_K_M.gguf"),
    "Q5_K_M": ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q5_K_M.gguf"),
    "Q6_K":   ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q6_K.gguf"),
    "Q8_0":   ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q8_0.gguf"),
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

    use_quant = quant or "Q6_K"     # resolve the caller's "" to the engine default…
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
            log.info("reference_image scaled to %dx%d (what the model sees)", ref_img.width, ref_img.height)

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
# FLUX.2 Klein 9B NVFP4 (nunchaku-lite fast lane)
# ---------------------------------------------------------------------------

def _load_flux2klein9b_nvfp4(quant: str = ""):
    """FLUX.2 Klein 9B NVFP4 — lite-infer nunchaku-lite checkpoint.

    from_pretrained loads the WHOLE pipeline resident (NVFP4 transformer +
    bnb4 Qwen3-8B text encoder + VAE) — the encoder never parks, unlike the
    Q6_K GGUF lane's lazy encode. Whole-card engine (qwenimage-edit class);
    _VRAM_NEED_MB 15000 governs — the gate demands a genuinely CLEAR card,
    so any load auto-evicts a resident TTS tenant (OmniVoice pays a ~30 s
    cold reload on its next synth, accepted 2026-09-09); warm draws are
    guarded by _evict_tts_tenant_if_loaded. Verified 2026-09-08/09 on
    sm_120: load 9.1 s, 11.22 GiB torch-alloc at ready; gen 4.5-11.3 s @ 4
    steps; driver gen peaks up to 15,659 MiB at 1536×1024 with a reference
    attached. Full-HD 1920×1072 verified IN-SERVICE on a clear card
    2026-09-09 (dev 4/4 + lab 2/2, 13.4 s steady, peaks 15,371-15,697 MiB)
    — every out-of-band/second-process FHD draw OOMs, so >1536² canvases
    are service-context only; idle TTS containers holding ~1.5-2 GiB of
    contexts also 503 the load honestly (gate message) instead of letting
    the gen walk into a mid-gen OOM.
    Same-seed reruns are near-identical, not byte-identical (nunchaku
    numerics — measured MAE 3.66/255 on a 1024² rerun pair).
    """
    import torch
    from diffusers import Flux2KleinPipeline

    if not GPU_ONLY:
        raise RuntimeError(
            "FLUX.2 Klein 9B NVFP4 is CUDA-only (nunchaku-lite kernels on "
            "sm_120) — no CPU/offload fallback exists.")
    t0 = time.time()
    repo = ENGINES["flux2klein9b-nvfp4"].hf_repo
    log.info("Loading FLUX.2 Klein 9B NVFP4 (nunchaku-lite) …")
    pipe = Flux2KleinPipeline.from_pretrained(
        repo, torch_dtype=torch.bfloat16).to("cuda")
    for enable in ("enable_slicing", "enable_tiling"):   # decode safety
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()
    # Cement cuda as the execution device — the first-module signature-order
    # fallback can read cpu and kill the CUDA VAE on some pipelines; here
    # every component is already on cuda, so the override is never wrong.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    STATE.loaded_model       = pipe
    STATE.active_engine      = "flux2klein9b-nvfp4"
    STATE.active_quant       = quant
    ENGINES["flux2klein9b-nvfp4"].loaded = True
    log.info("FLUX.2 Klein 9B NVFP4 ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _generate_flux2klein9b_nvfp4(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    generator = torch.Generator("cpu").manual_seed(seed)

    # Same Gemini expansion policy as the Q6 lane (no-op without the key).
    prompt = params["prompt"]
    expanded = _gemini_expand_prompt(prompt) if GPU_ONLY else None
    if expanded:
        log.info("Gemini expanded prompt: %r → %r", prompt, expanded)
        prompt = expanded

    # Ref thumbnail — same KV-budget rule as the Q6 lane: cap the ref area at
    # ≤768² so a full landscape screenshot passes untouched while square
    # full-res refs can't blow the card.
    ref_img = _load_ref_image(params.get("reference_image"))
    if ref_img is not None:
        w, h = ref_img.size
        if w * h > 768 * 768:
            log.info("Reference image %dx%d exceeds the 16 GB card's KV budget — downscaling to ≤768² px", w, h)
            ref_img.thumbnail((768, 768))
            log.info("reference_image scaled to %dx%d (what the model sees)", ref_img.width, ref_img.height)

    # No guidance_scale — step-distilled klein runs without CFG (passing
    # guidance gets it ignored with a warning). Width/height that aren't
    # multiples of 16 are rounded by the pipeline (verified: 1366×768
    # renders 1360×768). The resident bnb4 encoder handles the text prompt
    # in-pipeline (no prompt-embed cache, no parking — whole-card engine).
    result = pipe(
        prompt              = prompt,
        image               = ref_img,
        width               = int(params.get("width",  1024)),
        height              = int(params.get("height", 1024)),
        num_inference_steps = int(params.get("num_inference_steps", 4)),
        generator           = generator,
    )
    final_params = {**params, "seed": seed}
    if expanded:
        final_params = {**final_params, "expanded_prompt": expanded}
    return save_images(result.images, "flux2klein9b-nvfp4", final_params)


def _probe_flux2klein9b_nvfp4():
    try:
        # diffusers 0.40 nunchaku-lite dispatches the checkpoint's pipeline
        # class from its own config (Flux2KleinPipeline, auto_encoder kwarg).
        from diffusers import Flux2KleinPipeline  # noqa: F401
        ENGINES["flux2klein9b-nvfp4"].available = True
    except Exception as exc:
        ENGINES["flux2klein9b-nvfp4"].available = False
        ENGINES["flux2klein9b-nvfp4"].error     = str(exc)
        log.warning("FLUX.2 Klein 9B NVFP4 unavailable: %s", exc)


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
# SANA 1.6B  (Sprint + SANA 1.5 — the variant rides the `quant` form field)
# ---------------------------------------------------------------------------

# Why quant carries the variant: _ensure_engine / generate() / the UI
# reload-warning banner all key on the quant string — a separate param would
# not trigger an unload+reload when the checkpoint switches. STATE.active_quant
# stores the caller's verbatim string ("sprint-1.6b" / "1.5-1.6b"), so a
# default request never mismatches a resident default.
# Variant → (repo, supports_CFG). Both repos ship identical Gemma-2-2B-IT
# encoder shards → the HF cache dedups ~5.2 GB between them.
_SANA_VARIANTS: dict = {
    "sprint-1.6b": ("Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", False),
    "1.5-1.6b":    ("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",    True),
}


def _load_sana(quant: str = ""):
    import torch
    from diffusers import SanaPipeline, SanaSprintPipeline

    variant = quant or "sprint-1.6b"
    if variant not in _SANA_VARIANTS:
        raise RuntimeError(
            f"SANA variant '{variant}' not recognised. "
            f"Valid options: {list(_SANA_VARIANTS)}"
        )
    repo, has_cfg = _SANA_VARIANTS[variant]
    pipe_cls = SanaPipeline if has_cfg else SanaSprintPipeline

    t0 = time.time()
    # Public (ungated) repos — token=None per hf_hub 1.16.1 semantics (a token
    # would only be needed for gated access; see the klein loaders).
    log.info("Loading SANA %s (%s) — whole bf16 pipeline to CUDA …",
             variant, pipe_cls.__name__)
    pipe = pipe_cls.from_pretrained(
        repo, torch_dtype=torch.bfloat16, token=None,
    ).to("cuda")

    # Tiling protects 2048² renders (no-op at 1024). Slicing is a no-op where
    # AutoencoderDC lacks it (diffusers-version dependent) — guard both.
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # Store the CALLER's quant verbatim ("" = default), never the resolved
    # variant — _ensure_engine compares active_quant to the request string,
    # so resolving here would make every default request pay an unload+reload.
    STATE.loaded_model  = pipe
    STATE.active_engine = "sana"
    STATE.active_quant  = quant
    ENGINES["sana"].loaded = True
    log.info("SANA %s ready in %.1f s (CUDA: %.2f GiB)",
             variant, time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _generate_sana(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    # Resident by construction — _ensure_engine guarantees the request's
    # variant is loaded when we get here.
    variant, has_cfg = _SANA_VARIANTS[STATE.active_quant or "sprint-1.6b"]

    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    n = int(params.get("num_images", 1))
    steps = int(params.get("num_inference_steps", 4))
    if not has_cfg:
        # Sprint is step-distilled: SCMScheduler runs 1-4 steps. Clamp instead
        # of erroring — a UI/API caller pasting 20 would otherwise silently
        # get garbage from an invalid schedule.
        if steps > 4:
            log.info("Sprint steps=%d exceeds the 1-4 range — clamping to 4", steps)
            steps = 4
        steps = max(1, steps)
    else:
        steps = max(1, min(steps, 24))

    generator = [
        torch.Generator(device="cpu").manual_seed(seed + i)
        for i in range(n)
    ]

    kw = dict(
        prompt                 = params["prompt"],
        width                  = int(params.get("width",  1024)),
        height                 = int(params.get("height", 1024)),
        num_inference_steps    = steps,
        num_images_per_prompt  = n,
        generator              = generator,
    )
    if has_cfg:
        kw["negative_prompt"] = params.get("negative_prompt", "")
        kw["guidance_scale"]  = float(params.get("guidance_scale", 4.5))
    elif steps != 2:
        # diffusers SanaSprintPipeline passes intermediate_timesteps=1.3 (its
        # signature default) to the SCMScheduler, which accepts that only at
        # exactly 2 steps (the SCM max->1.3->0 jump). For 1/3/4 steps the
        # scheduler requires intermediate_timesteps=None to fall back to the
        # linear max_timesteps->0 schedule the Sprint distiller was trained
        # on — otherwise EVERY non-2-step Sprint request errors out.
        kw["intermediate_timesteps"] = None

    result = pipe(**kw)

    final_params = {**params, "seed": seed, "variant": variant}
    return save_images(result.images, "sana", final_params)


def _probe_sana():
    try:
        from diffusers import SanaPipeline, SanaSprintPipeline  # noqa: F401
        ENGINES["sana"].available = True
    except Exception as exc:
        ENGINES["sana"].available = False
        ENGINES["sana"].error     = str(exc)
        log.warning("SANA unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Boogu-Image-0.1-Turbo-fp8  (CPU-offload exception — boogu_lab_engine)
# ---------------------------------------------------------------------------

def _load_boogu(quant: str = ""):
    import importlib
    boogu_engine = importlib.import_module("boogu_lab_engine")
    t0 = time.time()
    log.info("Loading Boogu turbo fp8 (CPU-offloaded — approved exception) …")
    pipe = boogu_engine.load_boogu(quant=quant)
    STATE.loaded_model  = pipe
    STATE.active_engine = "boogu"
    STATE.active_quant  = quant
    ENGINES["boogu"].loaded = True
    log.info("Boogu turbo ready in %.1f s", time.time() - t0)


def _generate_boogu(params: dict) -> list[dict]:
    import importlib
    boogu_engine = importlib.import_module("boogu_lab_engine")

    pipe   = STATE.loaded_model
    images, seed_used = boogu_engine.generate_boogu(pipe, params)

    # Record the ACTUAL seed used — auto seed is drawn server-side inside the
    # module, so the gallery/API shows the real seed for reproducibility.
    final_params = {**params, "seed": seed_used}
    return save_images(images, "boogu", final_params)


def _probe_boogu():
    try:
        import importlib
        mod = importlib.import_module("boogu_lab_engine")
        result = mod.probe_boogu()
        if result["available"]:
            ENGINES["boogu"].available = True
        else:
            ENGINES["boogu"].available = False
            ENGINES["boogu"].error     = result.get("error", "unknown error")
            log.warning("Boogu unavailable: %s", result.get("error"))
    except Exception as exc:
        ENGINES["boogu"].available = False
        ENGINES["boogu"].error     = str(exc)
        log.warning("Boogu unavailable: %s", exc)


# ---------------------------------------------------------------------------
# ComfyUI sidecar hand-off (HiDream O1 — out-of-process engine)
# ---------------------------------------------------------------------------

def _poke_comfy_free() -> None:
    """Ask the ComfyUI sidecar to release its models from VRAM (best effort).

    Runs on EVERY lab unload (both branches of _unload_current): the lab and
    the sidecar share the card across processes, so whichever engine loads
    next gets a clean card. Cheap when the sidecar is idle (it unloads
    nothing); a no-op when comfy is down (debug-log only — it fires on every
    unload, so failures must stay quiet).
    """
    try:
        import importlib
        bridge = importlib.import_module("hidream_comfy_bridge")
        bridge.comfy_free()
    except Exception as exc:
        log.debug("ComfyUI sidecar free skipped (%s)", exc)


# ---------------------------------------------------------------------------
# Z-Image Turbo  (GGUF ladder via jayn7 + lazy NF4 Qwen3-4B encoder)
# ---------------------------------------------------------------------------

# GGUF quant ladder — all tiers share one architecture-derived config dir
# (from Tongyi's own transformer/config.json via _component_config_dir); only
# the file differs. Quant rides the API `quant` form field; "" resolves to
# the loader default (Q4_K_M — the quality/speed sweet spot on the 16 GB
# card; Q8_0 is the near-lossless ceiling and still fits).
_ZIMAGE_GGUF: dict[str, tuple[str, str]] = {
    "Q3_K_S": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q3_K_S.gguf"),
    "Q3_K_M": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q3_K_M.gguf"),
    "Q4_K_S": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K_S.gguf"),
    "Q4_K_M": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K_M.gguf"),
    "Q5_K_S": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q5_K_S.gguf"),
    "Q5_K_M": ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q5_K_M.gguf"),
    "Q6_K":   ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q6_K.gguf"),
    "Q8_0":   ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf"),
}


def _load_zimage(quant: str = "Q4_K_M"):
    import torch
    from diffusers import ZImagePipeline
    from diffusers.models import AutoencoderKL, ZImageTransformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    if not GPU_ONLY:
        raise RuntimeError(
            "Z-Image Turbo only has a GPU-only path (GGUF transformer + lazy "
            "NF4 encoder) — no CPU/offload fallback was written for it.")
    use_quant = quant or "Q4_K_M"
    if use_quant not in _ZIMAGE_GGUF:
        raise RuntimeError(
            f"Z-Image Turbo quant '{use_quant}' not recognised. "
            f"Valid options: {list(_ZIMAGE_GGUF)}"
        )
    t0 = time.time()
    # Public repos (jayn7 GGUF + Tongyi components) — token=None per the
    # hf_hub 1.16.1 semantics documented in the klein loaders.
    repo_id, fname = _ZIMAGE_GGUF[use_quant]
    repo = ENGINES["zimage"].hf_repo       # Tongyi-MAI/Z-Image-Turbo

    log.info("Loading Z-Image Turbo transformer from GGUF — quant=%s …", use_quant)
    gguf_path = _ensure_gguf(repo_id, fname, os.path.join(GGUF_ROOT, "zimage"))
    transformer = ZImageTransformer2DModel.from_single_file(
        gguf_path,
        config              = _component_config_dir(repo, "transformer", "zimage"),
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    ).to("cuda")

    # Text encoder loads LAZILY (same pattern as the klein engines): Z-Image's
    # encoder is the in-repo Qwen3-4B, and prompt embeddings are cached to
    # disk (_zimage_prompt_embeds), so the NF4 encoder only loads for prompts
    # never seen before and is parked again afterwards. Resident set without
    # it: GGUF transformer (~4.6 GiB @ Q4_K_M) + VAE.
    text_encoder = None

    log.info("Loading Z-Image Turbo tokenizer / VAE / scheduler …")
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    vae       = AutoencoderKL.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.bfloat16, token=None,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder="scheduler", torch_dtype=torch.bfloat16, token=None,
    )

    log.info("Assembling Z-Image Turbo pipeline …")
    pipe = ZImagePipeline(
        scheduler    = scheduler,
        vae          = vae,
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        transformer  = transformer,
    ).to("cuda")
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    # See the klein loaders: the pipeline resolves its execution device from
    # the FIRST module's location, which falls back to self.device and reads
    # the __init__ signature order (text_encoder first). With precomputed
    # embeds the encoder is never called during generation, so the pipeline
    # genuinely executes on cuda — cement it to keep latent placement sane.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    # Store the CALLER's quant verbatim ("" = default) — _ensure_engine
    # compares active_quant against the request string.
    STATE.loaded_model       = pipe
    STATE.active_engine      = "zimage"
    STATE.active_quant       = quant
    ENGINES["zimage"].loaded = True
    log.info("Z-Image Turbo ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _zimage_prompt_embeds(pipe, text: str):
    """Return Z-Image's masked prompt embedding for `text` (CUDA tensor).

    Cache hit → embeds from disk, the Qwen3-4B encoder never loads. Cache
    miss → lazy-load the encoder (_ensure_klein_encoder's zimage branch),
    encode through the pipeline's own chat-template encode_prompt (returns a
    LIST of per-prompt attention-masked tensors — Z-Image's transformer takes
    variable-length rows, unlike the padded-batch klein/FLUX embeds), save
    the single masked row, then park the encoder (see _klein_prompt_embeds
    for why encode runs under no_grad).
    """
    import torch
    path = _embed_cache_path("zimage", text)
    cached = _embed_cache_load(path)
    if cached is not None:
        log.info("Z-Image prompt-embedding cache hit — skipping the text encoder")
        return cached.to("cuda")
    _ensure_encoder_headroom("zimage")
    _ensure_klein_encoder(pipe, "zimage")
    with torch.no_grad():
        embeds, _negative = pipe.encode_prompt(
            prompt=[text],
            device="cuda",
            do_classifier_free_guidance=False,   # distilled — no CFG encode
        )
    emb = embeds[0]                               # [masked_len, 2560] bf16
    log.info("Z-Image encoded %d-token prompt (Qwen3-4B, NF4) → embed cache",
             emb.shape[0])
    _embed_cache_save(path, emb)
    _park_klein_encoder(pipe)
    return emb.to("cuda")


def _generate_zimage(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    # Step-distilled turbo: 8 steps is the training target; clamp instead of
    # erroring (mirrors the _generate_sana clamp) — a caller pasting 20 would
    # otherwise silently get an invalid schedule.
    steps = int(params.get("num_inference_steps", 8))
    if steps > 8:
        log.info("Z-Image Turbo steps=%d exceeds the distilled 1-8 range — "
                 "clamping to 8", steps)
        steps = 8
    steps = max(1, steps)
    n = int(params.get("num_images", 1))

    emb = _zimage_prompt_embeds(pipe, params["prompt"])

    images = []
    for i in range(n):
        # Per-image loop with seed+i (single generator per call) — the
        # pipeline's num_images semantics differ per model; a loop is the
        # deterministic common denominator (see the qwenimage/ernie notes).
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        result = pipe(
            prompt_embeds       = [emb],   # masked row(s); prompt must be None
            width               = int(params.get("width",  1024)),
            height              = int(params.get("height", 1024)),
            num_inference_steps = steps,
            guidance_scale      = 0.0,     # distilled — CFG forced off
            generator           = generator,
        )
        images.append(result.images[0])

    final_params = {**params, "seed": seed}
    return save_images(images, "zimage", final_params)


def _probe_zimage():
    try:
        from diffusers import ZImagePipeline                    # noqa: F401
        from diffusers.models import ZImageTransformer2DModel   # noqa: F401
        ENGINES["zimage"].available = True
    except Exception as exc:
        ENGINES["zimage"].available = False
        ENGINES["zimage"].error     = str(exc)
        log.warning("Z-Image Turbo unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Qwen-Image 2512  (GGUF via unsloth; encoder encodes WITHOUT the transformer
# resident — the ~12 GiB Q4_K_S GGUF and the ~7.3 GiB bnb4 encoder never
# coexist on the 16 GB card)
# ---------------------------------------------------------------------------

# Q4_K_S is the ONLY tier that fits this card — verified live 2026-09-08:
# Q4_K_M (12.34 GB) loads to a ~14.2 GiB process floor on a 15.48 GiB card
# that already hosts ~1.2 GiB of TTS-container CUDA contexts — its load
# OOMs at the final pipeline .to("cuda") and generation is unreachable
# (measured 3×, standalone and in-lab). Q4_K_S (~11.5 GB) loads and
# generates at native CFG 4.0, 1024²/20 steps, peak 12.19 GiB torch-alloc.
# Q5_K_S/Q5_K_M sit between the two and have no path. Higher tiers removed
# 2026-09-08; keep the dict shape (loader + UI enumerate it) in case a
# smaller Qwen-Image sibling ever ships.
_QWENIMAGE_GGUF: dict[str, tuple[str, str]] = {
    "Q4_K_S": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q4_K_S.gguf"),
}

# The full in-repo Qwen2.5-VL-family encoder is ~16.6 GB bf16 — never fits.
# This is the pre-quantised bnb-4bit mirror of the same text encoder (hidden
# states identical for our purposes: Qwen-Image only reads hidden_states[-1]).
_QWENIMAGE_ENCODER_REPO = "OzzyGT/Qwen-Image-2512-bnb-4bit-text-encoder"

# The Qwen-Image prompt template + prefix-drop the diffusers pipeline
# hardcodes (verified against diffusers main, 2026-09-07). Encode replicates
# QwenImagePipeline._get_qwen_prompt_embeds statically — template-wrap →
# tokenize at 1024+34 → hidden_states[-1] → attention-mask extraction →
# drop the 34 template-prefix tokens → zero-pad stack. The pipeline's own
# __call__ re-slices [:1024], which the drop above already guarantees.
_QWENIMAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_QWENIMAGE_TEMPLATE_DROP_IDX = 34

# The tokenizer is tiny and CPU-only — keep one around for the encode-without-
# transformer path (the pipeline object is destroyed while the encoder is in
# VRAM). Loaded by _load_qwenimage; None before the first load.
_QWENIMAGE_TOKENIZER = None


def _qwenimage_encode_texts(encoder, tokenizer, texts: list, device, dtype):
    """Static replication of QwenImagePipeline._get_qwen_prompt_embeds.

    Returns the zero-padded stack (B, ≤1024, hidden) + attention mask; the
    pipeline's encode_prompt turns an all-ones mask into None, so a single
    unpadded text gets mask=None downstream (kept for parity either way).
    """
    import torch
    wrapped = [_QWENIMAGE_PROMPT_TEMPLATE.format(t) for t in texts]
    tokens = tokenizer(
        wrapped,
        max_length = 1024 + _QWENIMAGE_TEMPLATE_DROP_IDX,
        padding    = True,
        truncation = True,
        return_tensors = "pt",
    ).to(device)
    hidden = encoder(
        input_ids       = tokens.input_ids,
        attention_mask  = tokens.attention_mask,
        output_hidden_states = True,
    ).hidden_states[-1]
    mask = tokens.attention_mask.bool()
    valid_lengths = mask.sum(dim=1)
    rows = torch.split(hidden[mask], valid_lengths.tolist(), dim=0)
    rows = [r[_QWENIMAGE_TEMPLATE_DROP_IDX:] for r in rows]  # drop prefix
    max_len = max(r.size(0) for r in rows)
    prompt_embeds = torch.stack([
        torch.cat([r, r.new_zeros(max_len - r.size(0), r.size(1))])
        for r in rows
    ])
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds, mask


def _qwenimage_load_encoder():
    """Load the bnb-4bit Qwen2.5-VL text encoder onto CUDA.

    The OzzyGT repo is a full ForConditionalGeneration checkpoint
    (language_model.* / lm_head keys + vision tower). Loading it via
    AutoModel would build the BASE Qwen2_5_VLModel, whose key layout can
    never match — every weight comes back UNEXPECTED/MISSING (random init,
    then a 'normal_kernel_cuda' not implemented for 'Byte' crash on the
    mismatched 4-bit dispatch; measured ~2 min wasted + a ~7.7 GiB random
    model resident through the retry). Load the declared architecture
    directly (12 s, 729 shards, ~7.3 GiB resident — measured 2026-09-08).
    """
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _QWENIMAGE_ENCODER_REPO,
        device_map    = "cuda",
        torch_dtype   = torch.bfloat16,
    )


def _load_qwenimage(quant: str = "Q4_K_S"):
    import torch
    from diffusers import QwenImagePipeline
    from diffusers.models import (AutoencoderKLQwenImage,
                                  QwenImageTransformer2DModel)
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer

    if not GPU_ONLY:
        raise RuntimeError(
            "Qwen-Image 2512 only has a GPU-only path (GGUF transformer + "
            "bnb4 encoder) — no CPU/offload fallback was written for it.")
    use_quant = quant or "Q4_K_S"
    if use_quant not in _QWENIMAGE_GGUF:
        raise RuntimeError(
            f"Qwen-Image 2512 quant '{use_quant}' not recognised. "
            f"Valid options: {list(_QWENIMAGE_GGUF)}"
        )
    t0 = time.time()
    repo_id, fname = _QWENIMAGE_GGUF[use_quant]
    repo = ENGINES["qwenimage"].hf_repo    # Qwen/Qwen-Image-2512

    log.info("Loading Qwen-Image 2512 transformer from GGUF — quant=%s …",
             use_quant)
    gguf_path = _ensure_gguf(repo_id, fname, os.path.join(GGUF_ROOT, "qwenimage"))
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config              = _component_config_dir(repo, "transformer", "qwenimage"),
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    ).to("cuda")

    # Text encoder NEVER loads in-process with the transformer resident —
    # cache-miss encodes run with the pipeline unloaded (_generate_qwenimage
    # unloads, encodes via _qwenimage_load_encoder, saves, reloads).
    text_encoder = None

    log.info("Loading Qwen-Image 2512 tokenizer / VAE / scheduler …")
    # Slow Qwen2Tokenizer — the Qwen-Image-2512 repo ships no tokenizer.json
    # (verified 2026-09-07); this is exactly what the pipeline itself loads.
    tokenizer = Qwen2Tokenizer.from_pretrained(repo, subfolder="tokenizer")
    global _QWENIMAGE_TOKENIZER
    _QWENIMAGE_TOKENIZER = tokenizer
    # MUST load via the concrete AutoencoderKLQwenImage class: the 2512
    # repo's vae/config.json is Cosmos-style (base_dim/dim_mult keys, no
    # in_channels/block_out_channels), and generic AutoencoderKL.from_pretrained
    # does NOT dispatch on the config's _class_name — it builds the base
    # class from defaults and dies on decoder.conv_in.bias 64-vs-384
    # (verified 2026-09-08 on diffusers 0.38.0).
    vae       = AutoencoderKLQwenImage.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.bfloat16, token=None,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder="scheduler", torch_dtype=torch.bfloat16, token=None,
    )

    log.info("Assembling Qwen-Image 2512 pipeline …")
    pipe = QwenImagePipeline(
        scheduler    = scheduler,
        vae          = vae,
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        transformer  = transformer,
    ).to("cuda")
    for enable in ("enable_slicing", "enable_tiling"):   # 1536² decode safety
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()

    # See the klein loaders — with a None text encoder the pipeline's
    # _execution_device can fall back to a cpu reading of the signature
    # order; precomputed embeds mean generation never calls the encoder, so
    # cement cuda.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    # Store the CALLER's quant verbatim ("" = default), never the resolved
    # default — see the klein9b loader for the reload-loop rationale.
    STATE.loaded_model       = pipe
    STATE.active_engine      = "qwenimage"
    STATE.active_quant       = quant
    ENGINES["qwenimage"].loaded = True
    log.info("Qwen-Image 2512 ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _generate_qwenimage(params: dict) -> list[dict]:
    import torch

    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()
    steps = int(params.get("num_inference_steps", 20))
    steps = max(1, min(steps, 50))
    guidance = float(params.get("guidance_scale", 4.0))
    n = int(params.get("num_images", 1))
    quant = params.get("quant", "")

    # Prompts to encode: positive always; the negative (CFG side) only when
    # true_cfg_scale > 1 — at guidance ≤ 1 the pipeline warns and ignores it,
    # and encoding it would be wasted VRAM churn. The negative default "" is
    # a constant, so its cache file is written once per engine.
    need_neg = guidance > 1
    to_encode: dict = {"prompt": params["prompt"]}
    if need_neg:
        to_encode["negative"] = params.get("negative_prompt", "") or ""

    # ── Embedding cache ─────────────────────────────────────────────────────
    # Hit → embeds from disk, encoder never loads. Miss → the cache-miss
    # flow runs the encoder with the ~12 GiB Q4_K_S GGUF transformer
    # UNLOADED: the two never coexist on the 16 GB card. That means the
    # whole pipeline reloads afterwards (~15-25 s GGUF read) — paid once per
    # (quant, prompt).
    embeds: dict = {}
    missing: list = []
    for label, text in to_encode.items():
        cached = _embed_cache_load(_embed_cache_path("qwenimage", text))
        if cached is not None:
            embeds[label] = cached
        else:
            missing.append((label, text))
    if missing:
        log.info("Qwen-Image prompt-embedding cache miss (%d prompt(s)) — "
                 "unloading the transformer to encode …", len(missing))
        _unload_current()
        _ensure_encoder_headroom("qwenimage")
        encoder = _qwenimage_load_encoder()
        try:
            for label, text in missing:
                texts = [text]
                # no_grad is MANDATORY here: without it the output row keeps
                # the autograd graph alive, and the graph's leaf refs hold the
                # encoder's ~5.5 GiB of weights past `del encoder` — the reload
                # gate then measures ~8.4 GiB free and the load 503s (measured
                # 2026-09-08; a gate pass would OOM the gen instead).
                with torch.no_grad():
                    encoded, _mask = _qwenimage_encode_texts(
                        encoder, _QWENIMAGE_TOKENIZER, texts,
                        device="cuda", dtype=torch.bfloat16)
                # Single-text batch → keep the (1, ≤1024, hidden) row. The
                # all-ones mask collapses to None inside the pipeline.
                row = encoded[0]
                _embed_cache_save(_embed_cache_path("qwenimage", text), row)
                embeds[label] = row
        finally:
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        log.info("Qwen-Image encode done — reloading the transformer …")
        _ensure_engine("qwenimage", quant)
    pipe = STATE.loaded_model

    images = []
    for i in range(n):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        kw = dict(
            prompt              = None,     # precomputed embeds — no re-encode
            prompt_embeds       = embeds["prompt"].to("cuda").unsqueeze(0),
            negative_prompt     = None,
            negative_prompt_embeds = (embeds["negative"].to("cuda").unsqueeze(0)
                                      if need_neg else None),
            width               = int(params.get("width",  1024)),
            height              = int(params.get("height", 1024)),
            num_inference_steps = steps,
            true_cfg_scale      = guidance,
            generator           = generator,
        )
        result = pipe(**kw)
        images.append(result.images[0])

    final_params = {**params, "seed": seed}
    return save_images(images, "qwenimage", final_params)


def _probe_qwenimage():
    try:
        from diffusers import QwenImagePipeline                     # noqa: F401
        from diffusers.models import QwenImageTransformer2DModel    # noqa: F401
        ENGINES["qwenimage"].available = True
    except Exception as exc:
        ENGINES["qwenimage"].available = False
        ENGINES["qwenimage"].error     = str(exc)
        log.warning("Qwen-Image 2512 unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Qwen-Image-Edit-2511  (instruction edit — consumes the ref image natively)
# ---------------------------------------------------------------------------

# GGUF Q4_K_S (~11.56 GB) is the ONLY tier that fits the 16 GB card — the
# edit pipeline's TWO-channel conditioning (vision tokens through the VL
# encoder at 384² area + VAE-encoded ref latents at 1024² area concatenated
# to the noise latents) pushes the DiT sequence to ~8.2K tokens at 1024².
# Measured 2026-09-08 (probe, production fleet resident): gen driver peak
# 15.40 GiB at BOTH 1024² and 720×1440 / 20 st / CFG 4.0 — margin 0.53 GiB;
# 12.06 s/step (~4 min per image); same-seed rerun pixel-identical.
# Q4_K_M-class loads OOM (base-2512 precedent) — no higher tier has a path.
_QWENIMAGE_EDIT_GGUF: dict[str, tuple[str, str]] = {
    "Q4_K_S": ("unsloth/Qwen-Image-Edit-2511-GGUF",
               "qwen-image-edit-2511-Q4_K_S.gguf"),
}

# The Qwen2VL processor merges the condition image into the prompt text at
# encode (vision tokens + <|vision_start|> placeholder) — CPU-only, kept for
# the encode-without-transformer path exactly like the base engine's
# tokenizer. Loaded by _load_qwenimage_edit; None before the first load.
_QWENIMAGE_EDIT_PROCESSOR = None


def _load_qwenimage_edit(quant: str = "Q4_K_S"):
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.models import (AutoencoderKLQwenImage,
                                  QwenImageTransformer2DModel)
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    if not GPU_ONLY:
        raise RuntimeError(
            "Qwen-Image-Edit 2511 only has a GPU-only path (GGUF transformer "
            "+ bnb4 encoder) — no CPU/offload fallback was written for it.")
    use_quant = quant or "Q4_K_S"
    if use_quant not in _QWENIMAGE_EDIT_GGUF:
        raise RuntimeError(
            f"Qwen-Image-Edit 2511 quant '{use_quant}' not recognised. "
            f"Valid options: {list(_QWENIMAGE_EDIT_GGUF)}"
        )
    t0 = time.time()
    repo_id, fname = _QWENIMAGE_EDIT_GGUF[use_quant]
    repo = ENGINES["qwenimage-edit"].hf_repo   # Qwen/Qwen-Image-Edit-2511

    log.info("Loading Qwen-Image-Edit 2511 transformer from GGUF — "
             "quant=%s …", use_quant)
    gguf_path = _ensure_gguf(repo_id, fname,
                             os.path.join(GGUF_ROOT, "qwenimage-edit"))
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config              = _component_config_dir(repo, "transformer",
                                                    "qwenimage-edit"),
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    ).to("cuda")

    # Text encoder NEVER loads in-process with the transformer resident —
    # cache-miss encodes run with the pipeline unloaded
    # (_generate_qwenimage_edit unloads, encodes via the OzzyGT mirror,
    # saves, reloads).
    text_encoder = None

    log.info("Loading Qwen-Image-Edit 2511 tokenizer / processor / "
             "VAE / scheduler …")
    tokenizer = Qwen2Tokenizer.from_pretrained(repo, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(repo, subfolder="processor")
    global _QWENIMAGE_EDIT_PROCESSOR
    _QWENIMAGE_EDIT_PROCESSOR = processor
    # MUST load via the concrete AutoencoderKLQwenImage class — same
    # Cosmos-style vae/config.json gotcha as the base 2512 repo (generic
    # AutoencoderKL.from_pretrained mis-dispatches and dies on
    # decoder.conv_in.bias).
    vae       = AutoencoderKLQwenImage.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.bfloat16, token=None,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder="scheduler", torch_dtype=torch.bfloat16, token=None,
    )

    log.info("Assembling Qwen-Image-Edit 2511 pipeline …")
    pipe = QwenImageEditPlusPipeline(
        scheduler    = scheduler,
        vae          = vae,
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        processor    = processor,
        transformer  = transformer,
    ).to("cuda")
    for enable in ("enable_slicing", "enable_tiling"):   # decode safety
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()
    # See the qwenimage loader — with a None text encoder the pipeline's
    # _execution_device can fall back to a cpu reading of the signature
    # order; precomputed embeds mean generation never calls the encoder, so
    # cement cuda.
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    STATE.loaded_model       = pipe
    STATE.active_engine      = "qwenimage-edit"
    STATE.active_quant       = quant
    ENGINES["qwenimage-edit"].loaded = True
    log.info("Qwen-Image-Edit 2511 ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _qwenimage_edit_encode_texts(encoder, cond_img, texts: list) -> dict:
    """Image-conditioned encode via a PARTIAL edit pipeline (encoder +
    processor, no transformer/VAE) running the pipeline-native
    _get_qwen_prompt_embeds. Returns {text: single-row embeds}.

    No static replication is possible here (unlike the text-only base
    engine): the EditPlus system template + 64-token prefix drop, the
    per-image "<|vision_start|>…<|image_pad|>…" placeholder and the image
    merge all live inside the pipeline. The probe validated this
    partial-pipeline construction (register_modules accepts None modules;
    the init guards the VAE accessor) — the single-sample mask collapses to
    None, which the pipeline treats as all-valid.
    """
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions)

    enc_pipe = QwenImageEditPlusPipeline(
        scheduler    = None,
        vae          = None,
        text_encoder = encoder,
        tokenizer    = None,
        processor    = _QWENIMAGE_EDIT_PROCESSOR,
        transformer  = None,
    )
    enc_pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))
    # __call__-preprocess parity: the condition image is resized to the 384²
    # area BEFORE the processor merges it (CONDITION_IMAGE_SIZE = 384 * 384).
    cond_w, cond_h = calculate_dimensions(
        384 * 384, cond_img.size[0] / cond_img.size[1])
    cond_img = enc_pipe.image_processor.resize(cond_img, cond_h, cond_w)

    out: dict = {}
    try:
        for text in texts:
            # no_grad is MANDATORY — same autograd-graph trap as the base
            # engine (the output row would otherwise keep the encoder's
            # weights alive past `del encoder`).
            with torch.no_grad():
                pe, _mask = enc_pipe._get_qwen_prompt_embeds(
                    text, [cond_img], device="cuda", dtype=torch.bfloat16)
            out[text] = pe[0]              # single-row (drop batch dim)
    finally:
        del enc_pipe
        torch.cuda.empty_cache()
    return out


def _generate_qwenimage_edit(params: dict) -> list[dict]:
    import torch

    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()
    steps = int(params.get("num_inference_steps", 20))
    steps = max(1, min(steps, 50))
    guidance = float(params.get("guidance_scale", 4.0))
    n = int(params.get("num_images", 1))
    quant = params.get("quant", "")
    W = int(params.get("width",  1024))
    H = int(params.get("height", 1024))

    ref_bytes = params.get("reference_image")
    if ref_bytes is None:
        raise ValueError(
            "reference_image is required for Qwen-Image Edit — the edit "
            "checkpoint consumes a base image natively (no text-only path). "
            "Upload the image with this request.")
    # Measured ceiling 2026-09-08: 1024² (1.05M px) AND 720×1440 both fit at
    # 20 st/CFG 4.0 with ≤0.53 GiB margin; any larger canvas OOMs the card.
    if W * H > 1024 * 1024:
        raise ValueError(
            f"Canvas {W}×{H} = {W*H/1e6:.1f}M px exceeds the ~1.05M px ceiling "
            f"verified on the 16 GB card (1024² and 720×1440 both fit — "
            f"measured 2026-09-08).")
    ref = _load_ref_image(ref_bytes)       # corrupt upload → ValueError → 400
    if max(ref.size) > 1024:               # pipeline re-resizes internally
        from PIL import Image
        ref.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    # Prompts to encode: positive always; the negative (CFG side) only when
    # true_cfg_scale > 1 — at guidance ≤ 1 the pipeline warns and ignores it.
    # The edit embeds are image-conditioned, so the cache key covers the
    # prompt text AND the uploaded image bytes: same (prompt, image) pair →
    # identical decode → identical cond image → identical embeds.
    need_neg = guidance > 1
    to_encode: dict = {"prompt": params["prompt"]}
    if need_neg:
        to_encode["negative"] = params.get("negative_prompt", "") or ""
    ref_digest = hashlib.sha256(ref_bytes).hexdigest()[:24]

    def _cache_path(text: str) -> str:
        digest = hashlib.sha256(
            (ref_digest + "\x00" + text).encode("utf-8")).hexdigest()[:24]
        return os.path.join(EMBED_CACHE_ROOT, f"v{EMBED_CACHE_VERSION}",
                            "qwenimage-edit", digest + ".pt")

    # Hit → embeds from disk, encoder never loads. Miss → the flow runs the
    # encoder with the ~11.8 GiB Q4_K_S transformer UNLOADED (they never
    # coexist on the 16 GB card), then reloads the pipeline (~20-50 s GGUF
    # read) — paid once per (prompt, image bytes).
    embeds: dict = {}
    missing: list = []
    for label, text in to_encode.items():
        cached = _embed_cache_load(_cache_path(text))
        if cached is not None:
            embeds[label] = cached
        else:
            missing.append((label, text))
    if missing:
        log.info("Qwen-Image-Edit embed cache miss (%d prompt(s), "
                 "image-keyed) — unloading the transformer to encode …",
                 len(missing))
        _unload_current()
        _ensure_encoder_headroom("qwenimage-edit")
        # Shared OzzyGT mirror — the edit checkpoint uses the same
        # Qwen2.5-VL-family text encoder as the base 2512 (probe-verified).
        encoder = _qwenimage_load_encoder()
        try:
            rows = _qwenimage_edit_encode_texts(
                encoder, ref, [t for _, t in missing])
        finally:
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        for label, text in missing:
            _embed_cache_save(_cache_path(text), rows[text])
            embeds[label] = rows[text]
        log.info("Qwen-Image-Edit encode done — reloading the transformer …")
        _ensure_engine("qwenimage-edit", quant)
    pipe = STATE.loaded_model

    images = []
    for i in range(n):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        kw = dict(
            image               = [ref],
            prompt              = None,     # precomputed embeds — no re-encode
            prompt_embeds       = embeds["prompt"].to("cuda").unsqueeze(0),
            negative_prompt     = None,
            negative_prompt_embeds = (embeds["negative"].to("cuda").unsqueeze(0)
                                      if need_neg else None),
            width               = W,
            height              = H,
            num_inference_steps = steps,
            true_cfg_scale      = guidance,
            generator           = generator,
        )
        result = pipe(**kw)
        images.append(result.images[0])

    final_params = {**params, "seed": seed}
    return save_images(images, "qwenimage-edit", final_params)


def _probe_qwenimage_edit():
    try:
        from diffusers import QwenImageEditPlusPipeline              # noqa: F401
        from diffusers.models import QwenImageTransformer2DModel     # noqa: F401
        ENGINES["qwenimage-edit"].available = True
    except Exception as exc:
        ENGINES["qwenimage-edit"].available = False
        ENGINES["qwenimage-edit"].error     = str(exc)
        log.warning("Qwen-Image-Edit 2511 unavailable: %s", exc)


# ---------------------------------------------------------------------------
# ERNIE-Image-Turbo  (Nunchaku-Lite NVFP4 primary, fp8 mirror fallback)
# ---------------------------------------------------------------------------

# Primary: Baidu's pre-quantised suite — NVFP4 transformer (sm_120-native via
# nunchaku-lite) + bnb-4bit Mistral3 encoder + Flux2 VAE, ~9.6 GB total,
# measured 9.572 GB peak at 1024²/8 steps on an RTX PRO 6000. Fallback (load
# failure — e.g. no nunchaku backend in this venv): the fp8 diffusers-layout
# mirror of the same transformer/encoder pair. The bf16 original (baidu)
# does not fit alongside its encoder and is deliberately NOT in the ladder.
_ERNIE_FALLBACK_REPO = "rootlocalghost/ERNIE-Image-Turbo-FP8"


def _load_ernie(quant: str = ""):
    import gc as _gc
    import torch
    from diffusers import ErnieImagePipeline

    if not GPU_ONLY:
        raise RuntimeError(
            "ERNIE-Image-Turbo only has a GPU-only path (NVFP4/fp8 pipeline) "
            "— no CPU/offload fallback was written for it.")
    t0 = time.time()
    # quant is accepted for interface parity but unused — the ladder is
    # internal (repo fallback), not user-selectable. The VRAM headroom gate
    # ran before this loader (via _ensure_engine), so _VRAM_NEED_MB governs.
    repo = ENGINES["ernie"].hf_repo       # lite-infer nunchaku suite
    try:
        pipe = ErnieImagePipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16, token=None,
        )
    except Exception as exc:
        log.warning("ERNIE primary repo %s failed to load (%s) — "
                    "falling back to fp8 mirror %s", repo, exc,
                    _ERNIE_FALLBACK_REPO)
        pipe = ErnieImagePipeline.from_pretrained(
            _ERNIE_FALLBACK_REPO, torch_dtype=torch.bfloat16, token=None,
        )

    # Drop the PE prompt-enhancer (Ministral3-3B, ~7.66 GB bf16) BEFORE any
    # .to("cuda") — the lab generates from plain prompts and the pipeline
    # gates PE on self.pe being not None (it is in _optional_components), so
    # a None pe is fully supported. Saves the load path from ever staging it.
    pe = getattr(pipe, "pe", None)
    if pe is not None:
        log.info("ERNIE: dropping pe (Ministral3-3B prompt enhancer, ~7.7 GB) "
                 "— lab generation runs plain-prompt, use_pe path disabled")
        try:
            pipe.pe = None
        except Exception:
            pass
        del pe
        _gc.collect()

    log.info("Moving ERNIE-Image-Turbo pipeline to CUDA …")
    pipe = pipe.to("cuda")

    STATE.loaded_model       = pipe
    STATE.active_engine      = "ernie"
    STATE.active_quant       = quant
    ENGINES["ernie"].loaded  = True
    log.info("ERNIE-Image-Turbo ready in %.1f s (CUDA: %.2f GiB)",
             time.time() - t0, torch.cuda.memory_allocated() / 1024**3)


def _generate_ernie(params: dict) -> list[dict]:
    import torch

    pipe = STATE.loaded_model
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()

    # Distilled turbo: 8 steps / CFG 1.0 (do_classifier_free_guidance is
    # guidance > 1.0 → false at exactly 1.0, so no negative encode happens
    # and use_pe needs no pe). Fixed by the schema (min=max); clamp anyway so
    # a raw-API caller can't ask for an invalid schedule.
    steps = int(params.get("num_inference_steps", 8))
    if steps != 8:
        log.info("ERNIE steps=%d differs from the distilled 8-step schedule — "
                 "forcing 8", steps)
        steps = 8
    guidance = float(params.get("guidance_scale", 1.0))
    if guidance != 1.0:
        log.info("ERNIE guidance=%s differs from the distilled CFG-1.0 "
                 "schedule — forcing 1.0", guidance)
        guidance = 1.0
    n = int(params.get("num_images", 1))

    images = []
    for i in range(n):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        result = pipe(
            prompt               = params["prompt"],
            width                = int(params.get("width",  1024)),
            height               = int(params.get("height", 1024)),
            num_inference_steps  = steps,
            guidance_scale       = guidance,
            generator            = generator,
        )
        images.append(result.images[0])

    final_params = {**params, "seed": seed}
    return save_images(images, "ernie", final_params)


def _probe_ernie():
    try:
        # ErnieImagePipeline itself raises ImportError on transformers<5.0.0
        # (Ministral3ForCausalLM gate) — the probe surfaces that clearly.
        from diffusers import ErnieImagePipeline  # noqa: F401
        ENGINES["ernie"].available = True
    except Exception as exc:
        ENGINES["ernie"].available = False
        ENGINES["ernie"].error     = str(exc)
        log.warning("ERNIE-Image-Turbo unavailable: %s", exc)


# ---------------------------------------------------------------------------
# HiDream O1-Dev  (OUT-OF-PROCESS via the ComfyUI sidecar + bridge)
# ---------------------------------------------------------------------------

# Nothing diffusers-side: the Dev fp8_scaled checkpoint (~8.1 GB) runs inside
# the headless ComfyUI sidecar (own venv, arthur-comfy.service, port 8188)
# with the official Dev template graph (28 steps, cfg 1.0 + 7.6 noise scale).
# The lab delegates the whole graph to hidream_comfy_bridge.py and only
# collects the output PNGs. active_engine is still set (loaded_model stays
# None) so the idle-eviction path releases the sidecar's VRAM via the
# _unload_current comfy poke — see _load_hidream.

def _hidream_make_headroom(need_mb: int) -> int:
    """Bridge callback: evict the TTS containers, return free VRAM in MiB."""
    free_mb = _evict_tts_engines()
    if free_mb < need_mb:
        raise RuntimeError(
            f"HiDream (ComfyUI sidecar) needs ~{need_mb} MiB free VRAM; only "
            f"{free_mb} MiB after evicting the TTS engine containers.")
    return free_mb


def _load_hidream(quant: str = ""):
    # Thin loader — everything happens in the sidecar. Set active_engine so
    # the dispatcher's warm-path check skips reloads between hidream requests
    # and the idle-eviction loop (keyed on active_engine) can release comfy's
    # VRAM; loaded_model stays None (no in-process pipeline exists to unload).
    STATE.loaded_model  = None
    STATE.active_engine = "hidream"
    STATE.active_quant  = quant
    ENGINES["hidream"].loaded = True
    log.info("HiDream delegated to the ComfyUI sidecar (%s)", IMGLAB_COMFY_URL)


def _generate_hidream(params: dict) -> list[dict]:
    import importlib
    bridge = importlib.import_module("hidream_comfy_bridge")

    seed = params.get("seed", -1)
    if seed == -1:
        seed = random_seed()
    steps = int(params.get("num_inference_steps", 28))   # Dev is 28 fixed
    if steps != 28:
        log.info("HiDream steps=%d differs from the Dev 28-step sampler — "
                 "forcing 28", steps)
        steps = 28
    n = int(params.get("num_images", 1))

    images = []
    for i in range(n):
        images.append(bridge.generate(
            prompt        = params["prompt"],
            width         = int(params.get("width",  1024)),
            height        = int(params.get("height", 1024)),
            seed          = seed + i,
            steps         = steps,
            make_headroom = _hidream_make_headroom,
        ))

    final_params = {**params, "seed": seed}
    return save_images(images, "hidream", final_params)


def _probe_hidream():
    try:
        import importlib
        bridge = importlib.import_module("hidream_comfy_bridge")
        result = bridge.probe()
        if result.get("available"):
            ENGINES["hidream"].available = True
            ENGINES["hidream"].error     = ""
        else:
            ENGINES["hidream"].available = False
            ENGINES["hidream"].error     = result.get("error", "sidecar unreachable")
            log.warning("HiDream unavailable: %s", ENGINES["hidream"].error)
    except Exception as exc:
        ENGINES["hidream"].available = False
        ENGINES["hidream"].error     = str(exc)
        log.warning("HiDream unavailable: %s", exc)


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
    _probe_flux2klein9b_nvfp4()
    _probe_ideogram4()
    _probe_sana()
    _probe_boogu()
    _probe_zimage()
    _probe_qwenimage()
    _probe_qwenimage_edit()
    _probe_hidream()
    _probe_ernie()


def _probe_flux2klein():
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
        ENGINES["flux2klein"].available = True
    except Exception as exc:
        ENGINES["flux2klein"].available = False
        ENGINES["flux2klein"].error     = str(exc)
        log.warning("FLUX.2 Klein 4B unavailable: %s", exc)


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
            # Stamp the run-timing context: save_image/save_video read
            # run_started/run_loaded_at to record per-image stats. run_loaded_at
            # is only set when a load actually happens — a warm run (same
            # engine + quant already resident) leaves it 0.0 so the saved
            # entry's load_s is null, not a misleading 0.0.
            need_load = not (STATE.active_engine == engine_key and
                             (not quant or STATE.active_quant == quant))
            STATE.run_started = time.time()
            STATE.run_loaded_at = 0.0
            _ensure_engine(engine_key, quant)
            if need_load:
                STATE.run_loaded_at = time.time()
        except Exception:
            # A failed load (e.g. CUDA OOM mid-load) leaves partially-loaded
            # tensors pinned in the caching allocator — release them so the
            # card isn't bricked until the service restarts.
            STATE.run_started = 0.0
            STATE.run_loaded_at = 0.0
            _unload_current()
            raise
        if not need_load:
            # Warm draw: the engine is resident, so the load gate above did
            # not re-run — evict any TTS model that loaded since the last
            # draw so the gen gets the clear card it was calibrated on
            # (whole-card engines only; see _evict_tts_tenant_if_loaded).
            _evict_tts_tenant_if_loaded(engine_key)
        generator_fn = _GENERATORS[engine_key]
        try:
            results = generator_fn(params)
        except Exception:
            # A failed generation leaks just like a failed load: the NF4
            # text-encoder quantise-on-load (cache-miss prompts) or the
            # encode itself can OOM mid-allocation and pin partially-loaded
            # tensors in the caching allocator — bricks the card until the
            # service restarts (verified 2026-09-05: 3.4-5 GiB stuck after
            # an aborted encoder load; every later request 503'd on the
            # VRAM gate). Drop everything so the next request starts clean.
            _unload_current()
            raise
        STATE.last_used = time.time()
        # Guard-set engines (clear-card class) leave the gen's working set
        # pooled after a successful draw — release it while the engine stays
        # resident so a TTS tenant can co-load beside it (cohabitation;
        # the pooled state OOM'd OmniVoice's 1.14 GiB load chunk at 794 MiB
        # free, 2026-09-09). Best-effort; see _compact_after_draw.
        _compact_after_draw(engine_key)
        return results
    finally:
        STATE.generating = False
        # Run context is only meaningful while a generation is in flight
        STATE.run_started   = 0.0
        STATE.run_loaded_at = 0.0


# ---------------------------------------------------------------------------
# Public load / unload for the API
# ---------------------------------------------------------------------------

def load_engine(engine_key: str, quant: str = ""):
    try:
        _ensure_engine(engine_key, quant)
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
    """Decode a reference image from upload bytes or a filesystem path.

    Corrupt/undecodable uploads raise ValueError (surfaces as a 400 from the
    API) instead of silently degrading to None — a silently dropped reference
    would generate without the identity the caller asked for.

    Every successful decode logs exactly one "consumed reference_image" line,
    on purpose. A client whose reference never arrives (bad multipart framing,
    wrong field name, unterminated final part) still gets a 200 and a
    normal-looking image, so the *absence* of this line on a request that
    meant to send a reference is the only signal that distinguishes
    "reference used" from "no reference". Callers that must not silently fall
    back can pass require_reference=true to the API and get a 400 instead.
    """
    if ref is None:
        return None
    from PIL import Image
    try:
        if isinstance(ref, bytes):
            if not ref:
                raise ValueError("Reference image upload was empty.")
            import io as _io
            # img.load() forces the lazy decoder to actually read the bytes —
            # Image.open alone accepts truncated/corrupt data silently.
            img = Image.open(_io.BytesIO(ref))
            img.load()
            n_bytes = len(ref)
        elif isinstance(ref, str) and os.path.exists(ref):
            img = Image.open(ref)
            n_bytes = os.path.getsize(ref)
        else:
            return None
        img = img.convert("RGB")
        log.info(
            "consumed reference_image: %dx%d, %d bytes",
            img.width, img.height, n_bytes,
        )
        return img
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Reference image could not be decoded: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_LOADERS = {
    "flux2klein":    _load_flux2klein,
    "flux2klein9b":  _load_flux2klein9b,
    "flux2klein9b-nvfp4": _load_flux2klein9b_nvfp4,
    "ideogram4":     _load_ideogram4,
    "sana":          _load_sana,
    "boogu":         _load_boogu,
    "zimage":        _load_zimage,
    "qwenimage":     _load_qwenimage,
    "qwenimage-edit": _load_qwenimage_edit,
    "hidream":       _load_hidream,
    "ernie":         _load_ernie,
}

_GENERATORS = {
    "flux2klein":    _generate_flux2klein,
    "flux2klein9b":  _generate_flux2klein9b,
    "flux2klein9b-nvfp4": _generate_flux2klein9b_nvfp4,
    "ideogram4":     _generate_ideogram4,
    "sana":          _generate_sana,
    "boogu":         _generate_boogu,
    "zimage":        _generate_zimage,
    "qwenimage":     _generate_qwenimage,
    "qwenimage-edit": _generate_qwenimage_edit,
    "hidream":       _generate_hidream,
    "ernie":         _generate_ernie,
}
