"""
hidream_comfy_bridge.py — HTTP client for the headless ComfyUI sidecar that
runs the HiDream O1-Dev engine (image_lab engine key "hidream").

The lab (image_lab_engines.py) and the sidecar share the 16 GB card across
PROCESSES: the sidecar is its own service (arthur-comfy.service, own venv,
port 8188) with its own CUDA context. This module is the only seam between
them — the lab never imports comfy code and the sidecar never knows the lab
exists. VRAM hand-off:

  * before a generation, if the card looks short the bridge unloads the
    sidecar's own models, then asks the lab (via the `make_headroom`
    callback) to evict the TTS containers, and only then POSTs the prompt;
  * after a generation comfy keeps the checkpoint warm so back-to-back
    hidream requests are fast, until the lab pokes `comfy_free()` — it does
    so from `_unload_current` on every engine unload.

The API graph below was resolved 2026-09-07 from the official
Comfy-Org/workflow_templates `image_hidream_o1_dev.json` (LiteGraph →
API format; the template's textgen "Prompt Enhancement" subgraph is skipped
and the CLIPTextEncode nodes are fed the prompt directly):

  1 CheckpointLoaderSimple     ckpt hidream_o1_image_dev_fp8_scaled.safetensors
  2 CLIPTextEncode (positive)  prompt text
  3 CLIPTextEncode (negative)  "" (never used — the graph runs cfg 1.0)
  4 KSamplerSelect             sampler "lcm" (io-schema equivalent of the
                               template's SamplerLCM node)
  5 ModelNoiseScale            noise_scale 7.6 (HiDream's guidance-free
                               "scale" analog)
  6 BasicScheduler             scheduler "normal", steps 28, denoise 1.0
  7 EmptyHiDreamO1LatentImage  width/height/batch 1 (pixel-space UiT — the
                               checkpoint needs NO VAE for latents)
  8 SamplerCustom              model [5], add_noise true, noise_seed seed,
                               cfg 1.0, positive [2], negative [3],
                               sampler [4], sigmas [6], latent_image [7]
  9 VAEDecode                  samples [8], vae [1,2]
 10 SaveImage                  images [9], filename_prefix "hidream"
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("image_lab")

# Headless ComfyUI sidecar (arthur-comfy.service) — mirrors IMGLAB_COMFY_URL
# in image_lab_engines.py; the env var is the single override for both.
COMFY_URL    = os.environ.get("IMGLAB_COMFY_URL", "http://127.0.0.1:8188")
COMFY_TIMEOUT = 15.0

# Checkpoint installed into the sidecar's models/checkpoints by the deploy
# (Comfy-Org/HiDream-O1-Image, Dev fp8_scaled ~8.1 GB).
HIDREAM_CKPT = "hidream_o1_image_dev_fp8_scaled.safetensors"

_SAVE_NODE   = 10    # SaveImage node id in the graph below

# ── tiny HTTP helpers (stdlib urllib only — the bridge must import in any
#    venv the lab runs in) ───────────────────────────────────────────────────

def _request(method: str, path: str, payload=None, timeout: float = COMFY_TIMEOUT):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode()) if body else {}


def _get_json(path: str, timeout: float = COMFY_TIMEOUT):
    """GET a JSON endpoint; raises with a readable message on any failure."""
    try:
        return _request("GET", path, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI HTTP {exc.code} on {path}: "
                           f"{exc.read().decode()[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"ComfyUI unreachable at {COMFY_URL} "
                           f"({exc})") from exc


def _get_raw(path: str, timeout: float = COMFY_TIMEOUT) -> bytes:
    """GET a binary endpoint (e.g. /view PNG) — _request JSON-decodes, so
    binary bodies go through here (raw bytes, never json.loads)."""
    req = urllib.request.Request(COMFY_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ── public surface used by image_lab_engines.py ─────────────────────────────

def comfy_free() -> bool:
    """Unload the sidecar's models + free its memory (POST /free)."""
    try:
        _request("POST", "/free", {"unload_models": True, "free_memory": True})
        return True
    except Exception as exc:
        log.debug("ComfyUI /free failed (%s)", exc)
        return False


def ping() -> bool:
    try:
        _get_json("/system_stats", timeout=5.0)
        return True
    except Exception:
        return False


def probe() -> dict:
    """Availability probe: sidecar reachable AND the HiDream checkpoint
    installed. The checkpoint list comes from /object_info so a renamed /
    missing file shows up as a clear error instead of a mid-gen 404."""
    try:
        stats = _get_json("/system_stats", timeout=8.0)
        devices = stats.get("devices") or []
        if not devices or not devices[0].get("name"):
            return {"available": False,
                    "error": "ComfyUI /system_stats returned no GPU device"}
        info = _get_json("/object_info/CheckpointLoaderSimple", timeout=15.0)
        ckpt_list = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if HIDREAM_CKPT not in ckpt_list:
            return {"available": False,
                    "error": f"HiDream checkpoint {HIDREAM_CKPT!r} not found in "
                             "the ComfyUI sidecar (models/checkpoints)"}
        return {"available": True, "error": None}
    except Exception as exc:
        return {"available": False, "error": f"ComfyUI sidecar unreachable: {exc}"}


def vram_free_mb() -> int:
    """Free VRAM seen by the sidecar (MiB), or -1 when stats are unavailable."""
    try:
        stats = _get_json("/system_stats", timeout=8.0)
        dev = (stats.get("devices") or [{}])[0]
        free = dev.get("vram_free")
        return int(free // (1024 * 1024)) if free else -1
    except Exception:
        return -1


def _vram_need_mb(width: int, height: int) -> int:
    """Rough headroom for a hidream gen at this resolution.

    The fp8_scaled checkpoint is ~8.1-8.3 GiB resident; pixel-space
    activation buffers scale with the pixel area on top. Linear model from
    1024² (~10 GiB) capped at ~14.5 GiB by 2048². Calibrate live (Phase E)
    against the sidecar's measured device peak.
    """
    mb = 10000 * (width * height) / (1024 * 1024)
    return int(max(10000, min(mb, 14500)))


# ── graph builder + generation ──────────────────────────────────────────────

def build_api_graph(prompt: str, width: int, height: int, seed: int,
                    steps: int = 28) -> dict:
    """Return the API-format prompt graph (node-id → node dict)."""
    return {
        1:  {"class_type": "CheckpointLoaderSimple",
             "inputs": {"ckpt_name": HIDREAM_CKPT}},
        2:  {"class_type": "CLIPTextEncode",
             "inputs": {"text": prompt, "clip": ["1", 1]}},
        3:  {"class_type": "CLIPTextEncode",
             "inputs": {"text": "", "clip": ["1", 1]}},
        4:  {"class_type": "KSamplerSelect",
             "inputs": {"sampler_name": "lcm"}},
        5:  {"class_type": "ModelNoiseScale",
             "inputs": {"model": ["1", 0], "noise_scale": 7.6}},
        6:  {"class_type": "BasicScheduler",
             "inputs": {"model": ["5", 0], "scheduler": "normal",
                        "steps": steps, "denoise": 1.0}},
        7:  {"class_type": "EmptyHiDreamO1LatentImage",
             "inputs": {"width": width, "height": height, "batch_size": 1}},
        8:  {"class_type": "SamplerCustom",
             "inputs": {"model": ["5", 0], "add_noise": True,
                        "noise_seed": seed, "cfg": 1.0,
                        "positive": ["2", 0], "negative": ["3", 0],
                        "sampler": ["4", 0], "sigmas": ["6", 0],
                        "latent_image": ["7", 0]}},
        9:  {"class_type": "VAEDecode",
             "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        10: {"class_type": "SaveImage",
             "inputs": {"images": ["9", 0], "filename_prefix": "hidream"}},
    }


def _make_headroom(need_mb: int, make_headroom) -> int:
    """Apply the lab-side callback (TTS eviction) when free VRAM is short."""
    if make_headroom is None:
        raise RuntimeError(
            f"ComfyUI sidecar has < {need_mb} MiB free for a HiDream "
            f"generation and no headroom callback was provided.")
    free_mb = make_headroom(need_mb)
    log.info("ComfyUI sidecar VRAM after headroom: %d MiB", free_mb)
    return free_mb


def generate(prompt: str, width: int, height: int, seed: int,
             steps: int = 28, make_headroom=None,
             max_wait_s: float = 900.0):
    """Run one HiDream generation through the sidecar; return a PIL Image.

    Steps are the Dev checkpoint's fixed 28-step schedule (callers clamp).
    make_headroom: optional callable(need_mb) → free MiB after evicting the
    TTS containers (supplied by the lab); without it a short card raises.
    """
    import PIL.Image as _Image

    need_mb = _vram_need_mb(width, height)
    free_mb = vram_free_mb()
    if 0 <= free_mb < need_mb:
        log.info("ComfyUI sidecar VRAM %d MiB < %d MiB — unloading its "
                 "models first …", free_mb, need_mb)
        comfy_free()
        time.sleep(2)                     # let the driver settle
        free_mb = vram_free_mb()
        if 0 <= free_mb < need_mb:
            free_mb = _make_headroom(need_mb, make_headroom)
            if free_mb < need_mb:
                raise RuntimeError(
                    f"HiDream generation needs ~{need_mb} MiB free VRAM in "
                    f"the ComfyUI sidecar; only {free_mb} MiB after evicting "
                    f"the TTS containers.")
        log.info("ComfyUI sidecar VRAM: %d MiB free — proceeding", free_mb)

    graph = build_api_graph(prompt, width, height, seed, steps=steps)
    result = _request("POST", "/prompt", {"prompt": graph}, timeout=60.0)
    if "prompt_id" not in result:
        raise RuntimeError(f"ComfyUI /prompt rejected the HiDream graph: "
                           f"{result.get('error') or result}")
    prompt_id = result["prompt_id"]
    log.info("HiDream queued on the sidecar: prompt_id=%s (%dx%d, seed %d, "
             "%d steps) …", prompt_id, width, height, seed, steps)

    # Poll /history/{id} until the run completes (or errors out).
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            history = _get_json(f"/history/{prompt_id}", timeout=10.0)
        except RuntimeError:
            continue                      # transient — keep polling
        entry = history.get(prompt_id)
        if not entry:
            continue
        status = entry.get("status", {})
        if status.get("status_str") == "error" or not status.get("completed"):
            # Surface the first execution_error message for a clear UI error.
            for msg in status.get("messages", []):
                if msg and msg[0] == "execution_error":
                    raise RuntimeError(
                        f"HiDream failed inside ComfyUI: "
                        f"{msg[1].get('exception_message', msg[1])}")
            raise RuntimeError("HiDream failed inside ComfyUI "
                               f"(prompt {prompt_id})")
        images = []
        for out in (entry.get("outputs") or {}).values():
            images.extend(out.get("images", []))
        if images:
            # A comfy run writes every image before it marks success; take
            # the first (batch is 1 per call).
            img = images[0]
            path = (f"/view?filename={img['filename']}"
                    f"&subfolder={img.get('subfolder', '')}"
                    f"&type={img.get('type', 'output')}")
            png = _get_raw(path, timeout=60.0)
            image = _Image.open(io.BytesIO(png))
            return image.convert("RGB")
        # completed but no images yet — transient, keep polling
    raise RuntimeError(f"HiDream timed out on the ComfyUI sidecar after "
                       f"{max_wait_s:.0f} s (prompt {prompt_id})")
