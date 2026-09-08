#!/usr/bin/env python3
"""
qwenimage_edit_probe.py — VM spike harness for the proposed `qwenimage-edit`
engine (Qwen-Image-Edit-2511 via unsloth Q4_K_S GGUF, pipeline
QwenImageEditPlusPipeline). Part B of the native base-face plan
(plans/curious-sniffing-hejlsberg.md); gates Part C engine implementation.

Why this probe exists: the EditPlus pipeline conditions on a reference image
TWICE — vision tokens through the VL encoder (CONDITION_IMAGE_SIZE = 384² area)
and VAE-encoded ref latents concatenated to the noise latents
(VAE_IMAGE_SIZE = 1024² area → ~4096 extra tokens regardless of canvas). The
DiT sequence therefore DOUBLES vs the base 2512 txt2img engine at the same
canvas (measured peak there: 12.19 GiB torch-alloc / ~14.2 GiB driver at
1024²/20 st/CFG 4.0). Whether that fits the 15.48 GiB usable card is the whole
spike question.

Rhythm mirrored from production image_lab_engines.py (the qwenimage loader is
the template):
  1. Load the transformer from GGUF (+vae/scheduler/processor/tokenizer) —
     measure resident + load peak.
  2. Destroy it → encode phase: bnb-4bit Qwen2.5-VL encoder (the SAME
     OzzyGT mirror the qwenimage engine uses — 2512-encoder assumption, see
     note below) + a PARTIAL QwenImageEditPlusPipeline (no transformer/vae,
     real processor) running the pipeline-NATIVE `_get_qwen_prompt_embeds`
     with the reference image — measure encode peak.
  3. Release the encoder → reload the transformer pipe (measured).
  4. Generate with precomputed image-conditioned embeds + the uploaded ref
     image at each requested size; a watcher thread records the nvidia-smi
     driver peak for THIS pid throughout.
  5. Print a structured summary against the plan's kill criteria.

Encoder note: the official Edit-2511 repo vendors its own fp16 text encoder
(~13.6 GB, bnb-4bit-quantised at load if we ever need it). OzzyGT's bnb-4bit
mirror of the Qwen-Image-2512 encoder is used here because (a) it is already
cached on this VM and (b) the spike's verdict (VRAM) is encoder-INDEPENDENT —
the encoder never coexists with the transformer in either variant. Quality
differences would only affect the Part C encoder choice, not viability.

Usage (on the VM, in the service env so HF caches line up):
  set -a; . /opt/arthur-img/.env; set +a
  /opt/arthur-img-env/bin/python scripts/probe/qwenimage_edit_probe.py \
      --ref /tmp/channel-base-face.jpg --out /tmp/qwenimage_edit_spike
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, "/opt/arthur-img")          # the lab code the service runs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo scripts/

import torch                                    # noqa: E402

# Lab helpers — the exact code paths production will take (module head is
# torch-free; torch imports are function-local in the lab, so this is cheap).
from image_lab_engines import (                 # noqa: E402
    GGUF_ROOT, _component_config_dir, _ensure_gguf, _gguf_quant_config,
    _qwenimage_load_encoder,
)

EDIT_REPO  = "Qwen/Qwen-Image-Edit-2511"
GGUF_REPO  = "unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_FILE  = "qwen-image-edit-2511-Q4_K_S.gguf"
GGUF_DIR   = os.path.join(GGUF_ROOT, "qwenimage-edit")
ENCODER    = "OzzyGT/Qwen-Image-2512-bnb-4bit-text-encoder"  # shared w/ qwenimage

log = print


def _stage(name: str):
    """Mark a stage boundary with torch alloc + (live) driver used for this pid."""
    driver = _nvidia_pid_mb()
    log(f"\n── {name} ───────────────────────────────")
    log(f"torch alloc {torch.cuda.memory_allocated() / 1024**3:7.2f} GiB | "
        f"torch max   {torch.cuda.max_memory_allocated() / 1024**3:7.2f} GiB | "
        f"smi pid     {driver:7.1f} MiB")
    torch.cuda.reset_peak_memory_stats()
    return driver


def _nvidia_pid_mb() -> float:
    """Driver-side (nvidia-smi) VRAM in MiB currently used by THIS pid."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            pid, mem = line.split(",")
            if int(pid.strip()) == os.getpid():
                return float(mem.strip())
    except Exception:
        pass
    return 0.0


def _nvidia_total_mb() -> float:
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        used, total = out.strip().split(",")
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0.0, 0.0


class DriverPeakWatcher(threading.Thread):
    """Sample nvidia-smi every 150 ms; record the max total + our pid."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_total = 0.0
        self.peak_pid   = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                import subprocess
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                used_total = 0.0
                for line in out.splitlines():
                    pid_s, mem_s = line.split(",")
                    mem = float(mem_s.strip())
                    used_total += mem
                    if int(pid_s.strip()) == os.getpid():
                        self.peak_pid = max(self.peak_pid, mem)
                self.peak_total = max(self.peak_total, used_total)
            except Exception:
                pass
            self._stop.wait(0.15)

    def stop(self):
        self._stop.set()

    def reset(self):
        self.peak_total = 0.0
        self.peak_pid = 0.0


def _open_ref(path: str, max_side: int = 1024):
    from PIL import Image
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))          # ≤1024² — upload-capped side
    return img


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _try_evict_service_engine():
    """Best-effort: the live arthur-imglab service must not hold a resident
    engine while we measure (two processes would fight for the card)."""
    import subprocess
    try:
        out = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:8002/engines/unload",
             "--max-time", "10"],
            capture_output=True, text=True, timeout=15).stdout
        log(f"service /engines/unload → {out.strip() or '(no body)'}")
    except Exception as exc:
        log(f"service /engines/unload skipped: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/channel-base-face.jpg")
    ap.add_argument("--out", default="/tmp/qwenimage_edit_spike")
    ap.add_argument("--prompt", default="Make this person smile warmly and "
                    "laugh joyfully, bright soft studio lighting, sharp focus, "
                    "photorealistic portrait")
    ap.add_argument("--negative", default="")     # "" still enables CFG (>1)
    ap.add_argument("--sizes", default="1024x1024,720x1440",   # square + 9:16
                    help="comma-separated WxH runs at --steps/--cfg")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-gen", action="store_true",
                    help="encode phase only (no transformer reload/generation)")
    args = ap.parse_args()

    import subprocess as _sp
    os.makedirs(args.out, exist_ok=True)

    # ── Environment identity ───────────────────────────────────────────────
    import diffusers, transformers
    log("=" * 72)
    log("Qwen-Image-Edit-2511 spike probe")
    log(f"python      {sys.version.split()[0]}  torch {torch.__version__}  "
        f"diffusers {diffusers.__version__}  transformers {transformers.__version__}")
    log(f"cuda        {torch.version.cuda}  device {torch.cuda.get_device_name(0)}")
    log(f"usable      {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB of "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    log(f"GGUF dir    {GGUF_DIR}")
    log(f"ref image   {args.ref}")
    if not os.path.isfile(args.ref):
        log("FATAL: reference image not found")
        return 2

    from PIL import Image
    ref = _open_ref(args.ref)
    log(f"ref thumb   {ref.size} ({ref.size[0] * ref.size[1]} px)")
    import hashlib as _hl
    ref_sha = _hl.sha1(open(args.ref, "rb").read()).hexdigest()[:16]
    log(f"ref sha1    {ref_sha}")

    # ── Stage 0: baseline ──────────────────────────────────────────────────
    _try_evict_service_engine()
    torch.cuda.init()
    _free()
    used_t, total_t = _nvidia_total_mb()
    log(f"card idle   {used_t:.0f} / {total_t:.0f} MiB driver-used "
        f"(fleet contexts incl.)")
    _stage("stage0 idle baseline")

    # ── Stage 1: transformer + vae + scheduler load (no encoder) ───────────
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    t0 = time.time()
    gguf_path = _ensure_gguf(GGUF_REPO, GGUF_FILE, GGUF_DIR)
    log(f"GGUF: {gguf_path} ({os.path.getsize(gguf_path) / 1024**3:.2f} GiB)")
    watcher = DriverPeakWatcher()
    watcher.start()

    cfg_dir = _component_config_dir(EDIT_REPO, "transformer", "qwenimage-edit")
    log("loading transformer from GGUF …")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config              = cfg_dir,
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    ).to("cuda")
    log(f"transformer loaded in {time.time() - t0:.1f} s")
    _stage("stage1 transformer resident (encoder absent)")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        EDIT_REPO, subfolder="scheduler", torch_dtype=torch.bfloat16, token=None)
    vae = AutoencoderKLQwenImage.from_pretrained(
        EDIT_REPO, subfolder="vae", torch_dtype=torch.bfloat16, token=None)
    log("loading tokenizer / processor (CPU) …")
    tokenizer = Qwen2Tokenizer.from_pretrained(EDIT_REPO, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(EDIT_REPO, subfolder="processor")

    pipe = QwenImageEditPlusPipeline(
        scheduler    = scheduler,
        vae          = vae,
        text_encoder = None,
        tokenizer    = tokenizer,
        processor    = processor,
        transformer  = transformer,
    ).to("cuda")
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))
    for enable in ("enable_slicing", "enable_tiling"):
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()
    _stage("stage1b full gen pipe resident")
    log(f"stage1 wall {time.time() - t0:.1f} s")

    if args.skip_gen:
        log("--skip-gen: stopping after the transformer load.")
        return 0

    # ── Stage 2: encode phase (transformer parked, encoder + processor) ─────
    log("\n… destroying gen pipe to run the encode phase (park rhythm) …")
    del pipe, transformer, vae, scheduler
    _free()
    _stage("stage2 parked (encoder not yet loaded)")

    t0 = time.time()
    encoder = _qwenimage_load_encoder()          # bnb-4bit OzzyGT mirror
    log(f"encoder loaded in {time.time() - t0:.1f} s")

    enc_pipe = QwenImageEditPlusPipeline(
        scheduler    = None,
        vae          = None,
        text_encoder = encoder,
        tokenizer    = None,
        processor    = processor,
        transformer  = None,
    )
    enc_pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))

    # Mirror __call__'s preprocess: condition images resized to 384² area
    # BEFORE the processor sees them (CONDITION_IMAGE_SIZE = 384 * 384).
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions)
    cond_w, cond_h = calculate_dimensions(384 * 384, ref.size[0] / ref.size[1])
    cond_img = enc_pipe.image_processor.resize(ref, cond_h, cond_w)
    log(f"condition image {cond_img.size} (aspect kept, ~384² area)")

    embeds: dict[str, torch.Tensor] = {}
    t0 = time.time()
    with torch.no_grad():
        for label, text in (("prompt", args.prompt),
                            ("negative", args.negative)):
            pe, mask = enc_pipe._get_qwen_prompt_embeds(
                text, [cond_img], device="cuda", dtype=torch.bfloat16)
            embeds[label] = pe[0]                # single-row (drop batch dim)
            log(f"encode [{label:8s}] row {tuple(pe.shape)} mask="
                f"{'None' if mask is None else tuple(mask.shape)}")
    log(f"encode phase wall {time.time() - t0:.1f} s")
    _stage("stage2b encoder resident after encode")
    log(f"prompt_embeds seq length {embeds['prompt'].shape[0]} "
        f"(vs ~1024 text-only for the base engine — vision tokens inflate it)")

    # ── Stage 3: release encoder, reload gen pipe ──────────────────────────
    del encoder, enc_pipe
    _free()
    _stage("stage3 parked before transformer reload")

    t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        EDIT_REPO, subfolder="scheduler", torch_dtype=torch.bfloat16, token=None)
    vae = AutoencoderKLQwenImage.from_pretrained(
        EDIT_REPO, subfolder="vae", torch_dtype=torch.bfloat16, token=None)
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config              = cfg_dir,
        quantization_config = _gguf_quant_config(),
        torch_dtype         = torch.bfloat16,
    ).to("cuda")
    pipe = QwenImageEditPlusPipeline(
        scheduler    = scheduler,
        vae          = vae,
        text_encoder = None,
        tokenizer    = tokenizer,
        processor    = processor,
        transformer  = transformer,
    ).to("cuda")
    pipe.__class__._execution_device = property(lambda self: torch.device("cuda"))
    for enable in ("enable_slicing", "enable_tiling"):
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()
    log(f"transformer reload in {time.time() - t0:.1f} s")
    _stage("stage3b gen pipe resident (encode done, embeds in hand)")

    # ── Stage 4: generation runs ───────────────────────────────────────────
    summary = {"sizes": {}, "determinism": {}}
    embeds_cuda = {k: v.to("cuda").unsqueeze(0) for k, v in embeds.items()}

    for size_spec in args.sizes.split(","):
        W, H = (int(x) for x in size_spec.lower().split("x"))
        gen_tag = f"{W}x{H}"
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        watcher.reset()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            out = pipe(
                image            = [ref],
                prompt           = None,         # precomputed embeds — no re-encode
                prompt_embeds    = embeds_cuda["prompt"],
                negative_prompt  = None,
                negative_prompt_embeds = embeds_cuda["negative"],
                width            = W,
                height           = H,
                num_inference_steps = args.steps,
                true_cfg_scale   = args.cfg,
                generator        = generator,
            )
        img = out.images[0]
        peak_torch = torch.cuda.max_memory_allocated() / 1024**3
        wall = time.time() - t0
        out_p = os.path.join(args.out, f"edit_{gen_tag}_s{args.steps}_"
                               f"cfg{args.cfg}.png")
        img.save(out_p)
        sha = hashlib.sha1(open(out_p, "rb").read()).hexdigest()[:16]
        log(f"\nGEN {gen_tag} / {args.steps} st / CFG {args.cfg} → {wall:.1f} s "
            f"({wall / args.steps:.2f} s/step) — {out_p}")
        log(f"  torch peak {peak_torch:6.2f} GiB | driver peak total "
            f"{watcher.peak_total / 1024:6.2f} GiB | our pid "
            f"{watcher.peak_pid / 1024:6.2f} GiB | sha {sha}")
        summary["sizes"][gen_tag] = {
            "wall_s": round(wall, 1), "torch_peak_gib": round(peak_torch, 2),
            "driver_peak_total_gib": round(watcher.peak_total / 1024, 2),
            "driver_peak_pid_gib": round(watcher.peak_pid / 1024, 2),
            "sha1": sha,
        }

        if gen_tag == "1024x1024":               # determinism rerun, same seed
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            watcher.reset()
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            with torch.no_grad():
                out2 = pipe(
                    image            = [ref],
                    prompt           = None,
                    prompt_embeds    = embeds_cuda["prompt"],
                    negative_prompt  = None,
                    negative_prompt_embeds = embeds_cuda["negative"],
                    width            = W,
                    height           = H,
                    num_inference_steps = args.steps,
                    true_cfg_scale   = args.cfg,
                    generator        = generator,
                )
            out_p2 = os.path.join(args.out, f"edit_{gen_tag}_rerun.png")
            out2.images[0].save(out_p2)
            sha2 = hashlib.sha1(open(out_p2, "rb").read()).hexdigest()[:16]
            identical = sha == sha2
            log(f"RERUN {gen_tag} same seed → sha {sha2} — "
                f"{'PIXEL-IDENTICAL' if identical else 'DIFFERS'}")
            summary["determinism"]["1024x1024"] = {
                "pixel_identical": identical, "wall_s": round(time.time() - t0, 1)}

    watcher.stop()
    torch.cuda.reset_peak_memory_stats()

    # ── Stage 5: verdict against the plan's kill criteria ──────────────────
    log("\n" + "=" * 72)
    log("SPIKE SUMMARY")
    log(json.dumps(summary, indent=2))
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
