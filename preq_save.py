#!/opt/arthur-img-env/bin/python
"""
preq_save.py — One-time script to pre-save quantized transformers to disk.

This script:
  1. Copies shared pipeline components (text encoders, VAE, configs) to
     /opt/arthur-img-models/quantized/<model>/shared/  via rsync.
  2. Loads each transformer with BitsAndBytes quantization and saves the
     quantized weights to  /opt/arthur-img-models/quantized/<model>/transformer[-2]-<quant>/
  3. After all saves succeed, deletes the original BF16 HF-cache directories.

Saving quantized weights reduces disk reads at load time:
  SD35 transformer  : BF16=15 GB  →  NF4=4 GB, INT8=8 GB
  Wan  transformer  : BF16=55 GB  →  NF4=14 GB  (INT8 not feasible: 28×2>32 GB RAM)

Usage (run as arthur, stop service first):
  sudo systemctl stop arthur-imglab
  nohup /opt/arthur-img-env/bin/python /opt/arthur-img/preq_save.py \
        > /var/log/preq_save.log 2>&1 &
  echo "PID=$!"
  # Monitor: tail -f /var/log/preq_save.log
  # When done: sudo systemctl start arthur-imglab
"""

from __future__ import annotations
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = "/opt/arthur-img-models/huggingface"
os.environ.setdefault("HF_TOKEN", "")

HF_CACHE   = "/opt/arthur-img-models/huggingface/hub"
PREQ_ROOT  = "/opt/arthur-img-models/quantized"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_sep(title: str) -> None:
    log("─" * 60)
    log(f"  {title}")
    log("─" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_vram() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_snapshot(org: str, model: str) -> Path:
    """Return the first (usually only) snapshot directory for an HF cache entry."""
    cache_key = f"models--{org}--{model}"
    snap_root = Path(HF_CACHE) / cache_key / "snapshots"
    if not snap_root.exists():
        raise FileNotFoundError(f"HF cache not found: {snap_root}")
    snapshots = sorted(snap_root.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {snap_root}")
    return snapshots[-1]   # latest


def hf_model_dir(org: str, model: str) -> Path:
    return Path(HF_CACHE) / f"models--{org}--{model}"


def copy_shared(src: Path, dst: Path, excludes: list[str]) -> None:
    """
    rsync -avL src/ dst/  excluding transformer weight subdirs.
    -L follows symlinks so real file content is copied (not HF cache symlinks).
    """
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avL", "--info=progress2", f"{src}/", f"{dst}/"]
    for exc in excludes:
        cmd += ["--exclude", exc]
    log(f"  rsync  {src}  →  {dst}")
    log(f"  excluding: {excludes}")
    result = subprocess.run(cmd, check=True, capture_output=False)
    log(f"  ✓ shared dir ready: {dst}")


def bnb4_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = __import__("torch").bfloat16,
        bnb_4bit_use_double_quant = True,
    )


def bnb8_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_8bit=True)


# ---------------------------------------------------------------------------
# SD 3.5 Large — NF4 and INT8
# ---------------------------------------------------------------------------

def save_sd35() -> None:
    log_sep("SD 3.5 Large")
    import torch
    from diffusers import SD3Transformer2DModel

    snap = find_snapshot("stabilityai", "stable-diffusion-3.5-large")
    out_root = Path(PREQ_ROOT) / "sd35"

    # 1. Shared components (text encoders + VAE + configs)
    copy_shared(snap, out_root / "shared", excludes=[
        "transformer/",
        "sd3.5_large.safetensors",   # monolithic ComfyUI file — unused by diffusers
        "text_encoders/",            # ComfyUI-format copies — not needed
    ])

    # 2. Quantized transformer variants
    for quant_name, qcfg in [("bnb4bit", bnb4_config()), ("bnb8bit", bnb8_config())]:
        out = out_root / f"transformer-{quant_name}"
        if out.exists():
            log(f"  SKIP sd35/transformer-{quant_name} (already exists)")
            continue

        log(f"  Loading SD35 transformer ({quant_name}) from {snap} …")
        t0 = time.time()
        transformer = SD3Transformer2DModel.from_pretrained(
            str(snap),
            subfolder           = "transformer",
            quantization_config = qcfg,
            torch_dtype         = torch.bfloat16,
        )
        elapsed = time.time() - t0
        log(f"  Loaded in {elapsed:.0f}s.  Saving to {out} …")
        transformer.save_pretrained(str(out))
        del transformer
        free_vram()
        log(f"  ✓ saved sd35/transformer-{quant_name}")

    # 3. Delete HF cache (original BF16)
    model_dir = hf_model_dir("stabilityai", "stable-diffusion-3.5-large")
    log(f"  Deleting HF cache: {model_dir}")
    shutil.rmtree(model_dir)
    log("  ✓ SD35 HF cache deleted")


# ---------------------------------------------------------------------------
# Wan2.2 — both T2V and I2V, both transformer and transformer_2
# ---------------------------------------------------------------------------

def save_wan_model(
    name: str,
    org: str,
    model: str,
    quants: list[tuple[str, object]],
) -> None:
    log_sep(f"Wan  {name}")
    import torch

    try:
        from diffusers import WanTransformer3DModel
    except ImportError:
        from diffusers.models import WanTransformer3DModel

    snap     = find_snapshot(org, model)
    out_root = Path(PREQ_ROOT) / name

    # 1. Shared components (text encoder + VAE + tokenizer + scheduler + configs)
    copy_shared(snap, out_root / "shared", excludes=[
        "transformer/",
        "transformer_2/",
    ])

    # 2. Quantize and save each transformer for each quant
    for quant_name, qcfg in quants:
        for subfolder in ["transformer", "transformer_2"]:
            out = out_root / f"{subfolder}-{quant_name}"
            if out.exists():
                log(f"  SKIP {name}/{subfolder}-{quant_name} (already exists)")
                continue

            log(f"  Loading {name}/{subfolder} ({quant_name}) …")
            t0 = time.time()
            t = WanTransformer3DModel.from_pretrained(
                str(snap),
                subfolder           = subfolder,
                quantization_config = qcfg,
                torch_dtype         = torch.bfloat16,
            )
            elapsed = time.time() - t0
            log(f"  Loaded in {elapsed:.0f}s.  Saving to {out} …")
            t.save_pretrained(str(out))
            del t
            free_vram()
            log(f"  ✓ saved {name}/{subfolder}-{quant_name}")

    # 3. Delete HF cache
    model_dir = hf_model_dir(org, model)
    log(f"  Deleting HF cache: {model_dir}")
    shutil.rmtree(model_dir)
    log(f"  ✓ {name} HF cache deleted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    log("=" * 60)
    log("  Pre-quantization save  —  Arthur Image Lab")
    log("=" * 60)
    log(f"  Output root : {PREQ_ROOT}")
    log(f"  HF cache    : {HF_CACHE}")
    log("")

    Path(PREQ_ROOT).mkdir(parents=True, exist_ok=True)

    try:
        # SD 3.5 Large: NF4 + INT8
        # Transformer: 15 GB BF16 → 4 GB NF4 / 8 GB INT8  (fits in 16 GB VRAM)
        save_sd35()

        # Wan T2V: NF4 only
        # Each transformer: 55 GB BF16 → 14 GB NF4
        # INT8 would be 28 GB × 2 = 56 GB — exceeds 32 GB system RAM
        save_wan_model(
            name   = "wan-t2v",
            org    = "Wan-AI",
            model  = "Wan2.2-T2V-A14B-Diffusers",
            quants = [("bnb4bit", bnb4_config())],
        )

        # Wan I2V: NF4 only (same VRAM/RAM constraint)
        save_wan_model(
            name   = "wan-i2v",
            org    = "Wan-AI",
            model  = "Wan2.2-I2V-A14B-Diffusers",
            quants = [("bnb4bit", bnb4_config())],
        )

    except Exception as exc:
        log(f"FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t_start
    log("")
    log("=" * 60)
    log(f"  ✓ All done in {elapsed/60:.1f} minutes")
    log("  Restart service: sudo systemctl start arthur-imglab")
    log("=" * 60)
