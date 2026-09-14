#!/usr/bin/env python3
"""
Probe: FLUX.2 klein 9B NVFP4 (nunchaku-lite) — A/B vs the lab's Q6_K GGUF route.

Answers (2026-09-08, RTX 5060 Ti 16 GB):
  1. Does the lite-infer nunchaku-lite checkpoint load + generate on sm_120?
  2. Load time + resident VRAM vs the Q6_K GGUF route (7.55 GiB transformer,
     14.2 GiB driver peak @ 1024² measured).
  3. Gen wall time @ 4 steps vs Q6's measured 25.3 s (09-05).
  4. Determinism: same-seed rerun pixel-identical?
  5. Reference path: does pipe(image=ref, ...) work through nunchaku-lite
     (klein native ref conditioning — the lab's flux2klein9b passes image=)?
     Same seed + prompt as a Q6 baseline draw on the live service → MAE.
  6. 9:16 (720×1440) fit with margin.

Mirrors the lab's klein9b engine conventions: ref thumbnailed ≤768² before the
pipe; 4 steps; no guidance (distilled, no CFG).

Usage:
  /opt/arthur-img-env/bin/python flux2klein9b_nvfp4_probe.py \
      --ref /tmp/channel-base-face.jpg --seed 20260909 \
      --out /tmp/klein9b_nvfp4_probe
"""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time

os.environ.setdefault("HF_HOME", "/opt/arthur-img-models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/opt/arthur-img-models/huggingface")
os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

REPO = "lite-infer/flux.2-klein-9b-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
PROMPT = (
    "The same person from the reference image smiling warmly with bright eyes, "
    "wearing a dark hoodie, bold white headline text across the top of the image "
    "reads PIZZA IS LIFE in clean modern lettering, bright studio lighting, "
    "photorealistic, sharp detail"
)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PeakWatcher:
    """Samples nvidia-smi driver memory.used until .stop(); .peak holds the max."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ref", default=None, help="reference image (channel base face)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", default="/tmp/klein9b_nvfp4_probe")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {"repo": args.repo, "prompt": args.prompt, "seed": args.seed, "stages": {}}

    def rpt(key, value):
        summary["stages"][key] = value
        print(f">> {key}: {value}", flush=True)

    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    rpt("diffusers", diffusers.__version__)
    rpt("cuda", torch.version.cuda)
    rpt("device", torch.cuda.get_device_name(0))

    # ---- stage 1: load -----------------------------------------------------
    w = PeakWatcher().start()
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16).to("cuda")
    load_s = time.time() - t0
    w.stop()
    rpt("load_s", round(load_s, 1))
    rpt("load_peak_driver_mib", w.peak)
    rpt("torch_reserved_giB", round(torch.cuda.memory_reserved() / 2**30, 2))
    rpt("torch_alloc_giB", round(torch.cuda.memory_allocated() / 2**30, 2))
    rpt("text_encoder", type(pipe.text_encoder).__name__)

    def gen(prompt, image, width, height, tag):
        w = PeakWatcher().start()
        t0 = time.time()
        try:
            kwargs = {}
            if image is not None:
                kwargs["image"] = image
            img = pipe(
                prompt=prompt, width=width, height=height,
                num_inference_steps=4,
                # CPU generator — matches the lab engine convention
                # (torch.Generator("cpu").manual_seed) so a same-seed Q6
                # baseline draw starts from the same noise latents.
                generator=torch.Generator("cpu").manual_seed(args.seed),
                **kwargs,
            ).images[0]
            ok = True
        except Exception as exc:  # noqa: BLE001 — probe must report, not die
            img, ok = None, False
            rpt(f"{tag}_error", f"{type(exc).__name__}: {exc}")
        finally:
            w.stop()
        rpt(f"{tag}_gen_s", round(time.time() - t0, 1) if ok else None)
        rpt(f"{tag}_peak_driver_mib", w.peak)
        if ok:
            path = os.path.join(args.out, f"{tag}_seed{args.seed}.png")
            img.save(path)
            rpt(f"{tag}_png", path)
            return path
        return None

    # ---- stage 2: T2I determinism pair (no ref) ---------------------------
    p1 = gen(args.prompt, None, 1024, 1024, "t2i_1024")
    p2 = gen(args.prompt, None, 1024, 1024, "t2i_1024_rerun")
    if p1 and p2:
        s1, s2 = sha1_file(p1), sha1_file(p2)
        rpt("t2i_determinism", "IDENTICAL " + s1 if s1 == s2 else f"MISMATCH {s1} vs {s2}")

    # ---- stage 3: reference path (klein native conditioning) --------------
    if args.ref:
        ref = Image.open(args.ref).convert("RGB")
        if ref.width * ref.height > 768 * 768:
            ref.thumbnail((768, 768))  # lab engine convention (KV budget)
        rpt("ref_thumb", f"{ref.width}x{ref.height}")
        p3 = gen(args.prompt, ref, 1024, 1024, "ref_1024")
        p4 = gen(args.prompt, ref, 720, 1440, "ref_720x1440")

    summary_path = os.path.join(args.out, "probe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)
    print("WROTE", summary_path, flush=True)


if __name__ == "__main__":
    main()
