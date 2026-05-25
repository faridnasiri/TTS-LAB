#!/usr/bin/env python3
"""
nvfp4_save.py — Quantize all supported transformers to NVFP4 and save to disk.

For each transformer this script:
  1. Downloads the BF16 weights from HuggingFace into a temporary directory.
  2. Loads the transformer with NVFP4WeightOnlyConfig applied on the fly
     (quantization happens layer-by-layer during load, so peak RAM stays low).
  3. Saves the quantized transformer to /opt/arthur-img-models/nvfp4/<model>/<subfolder>/.
  4. Deletes the temporary BF16 cache to reclaim disk space immediately.

Supported models:
  - FLUX.2 [dev]     : black-forest-labs/FLUX.2-dev          → nvfp4/flux2/transformer/
  - SD 3.5 Large     : stabilityai/stable-diffusion-3.5-large → nvfp4/sd35/transformer/
  - Wan2.2 T2V       : Wan-AI/Wan2.2-T2V-A14B-Diffusers      → nvfp4/wan-t2v/transformer/
                                                                  nvfp4/wan-t2v/transformer_2/
  - Wan2.2 I2V       : Wan-AI/Wan2.2-I2V-A14B-Diffusers      → nvfp4/wan-i2v/transformer/
                                                                  nvfp4/wan-i2v/transformer_2/

Requirements:
  - torchao installed in the env (pip install torchao)
  - HF_TOKEN with access to gated models (FLUX.2-dev, stable-diffusion-3.5-large)

Run (after service is stopped or alongside it):
    /opt/arthur-img-env/bin/python /opt/arthur-img/nvfp4_save.py
    # or in a screen session:
    screen -S nvfp4 /opt/arthur-img-env/bin/python /opt/arthur-img/nvfp4_save.py
"""

import gc
import os
import shutil
import sys
import time

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
NVFP4_ROOT = "/opt/arthur-img-models/nvfp4"
TEMP_BF16  = "/opt/arthur-img-models/temp_bf16"

# ---------------------------------------------------------------------------
# Job list  (label, hf_repo, subfolder, out_dir)
# ---------------------------------------------------------------------------
JOBS = [
    # FLUX.2 [dev] — gated, needs HF_TOKEN with access to black-forest-labs/FLUX.2-dev
    (
        "flux2/transformer",
        "black-forest-labs/FLUX.2-dev",
        "transformer",
        f"{NVFP4_ROOT}/flux2/transformer",
    ),
    # SD 3.5 Large — gated, needs HF_TOKEN with access to stabilityai/stable-diffusion-3.5-large
    (
        "sd35/transformer",
        "stabilityai/stable-diffusion-3.5-large",
        "transformer",
        f"{NVFP4_ROOT}/sd35/transformer",
    ),
    # Wan2.2 T2V — two transformers (HighNoise + LowNoise)
    (
        "wan-t2v/transformer",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "transformer",
        f"{NVFP4_ROOT}/wan-t2v/transformer",
    ),
    (
        "wan-t2v/transformer_2",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "transformer_2",
        f"{NVFP4_ROOT}/wan-t2v/transformer_2",
    ),
    # Wan2.2 I2V — two transformers (HighNoise + LowNoise)
    (
        "wan-i2v/transformer",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "transformer",
        f"{NVFP4_ROOT}/wan-i2v/transformer",
    ),
    (
        "wan-i2v/transformer_2",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "transformer_2",
        f"{NVFP4_ROOT}/wan-i2v/transformer_2",
    ),
]


def _fmt_gb_path(path: str) -> str:
    """Return human-readable size of a file or directory."""
    if os.path.isfile(path):
        return f"{os.path.getsize(path) / 1e9:.2f} GB"
    if os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(path)
            for f in files
        )
        return f"{total / 1e9:.2f} GB"
    return "0 GB"


def _disk_free_gb(path: str = "/") -> float:
    st = shutil.disk_usage(path)
    return st.free / 1e9


def save_nvfp4_transformer(label: str, hf_repo: str, subfolder: str, out_dir: str):
    """Download one BF16 transformer, quantize to NVFP4, save, clean up."""
    import torch
    from diffusers import AutoModel, TorchAoConfig
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

    print(f"\n[{label}] ── Starting ──────────────────────────────────────────", flush=True)

    config_json = os.path.join(out_dir, "config.json")
    if os.path.isfile(config_json):
        print(f"[{label}] Already saved at {out_dir} — skipping.", flush=True)
        return True

    os.makedirs(out_dir, exist_ok=True)

    # Per-job temp cache dir so we can wipe it independently
    cache_dir = os.path.join(TEMP_BF16, label.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    free_before = _disk_free_gb("/opt/arthur-img-models")
    print(f"[{label}] Disk free: {free_before:.1f} GB", flush=True)
    print(f"[{label}] Downloading BF16: {hf_repo}/{subfolder} → {cache_dir}", flush=True)

    t0 = time.time()
    quant_config = TorchAoConfig(NVFP4WeightOnlyConfig())

    # device_map="auto" splits across GPU (16 GB) + CPU RAM (32 GB) = 48 GB addressable,
    # which is necessary for the 32 GB FLUX.2-dev BF16 transformer.
    # Smaller models (SD3.5, Wan) fit in CPU RAM alone so "auto" is still safe there.
    try:
        transformer = AutoModel.from_pretrained(
            hf_repo,
            subfolder           = subfolder,
            quantization_config = quant_config,
            torch_dtype         = torch.bfloat16,
            device_map          = "auto",
            token               = HF_TOKEN or True,
            cache_dir           = cache_dir,
        )
    except Exception as exc:
        print(f"[{label}] ERROR during load: {exc}", flush=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return False

    load_elapsed = time.time() - t0
    print(
        f"[{label}] Loaded + quantized in {load_elapsed:.0f}s. "
        f"Saving to {out_dir} …",
        flush=True,
    )

    try:
        transformer.save_pretrained(out_dir, safe_serialization=False)
    except Exception as exc:
        print(f"[{label}] ERROR during save: {exc}", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return False

    saved_size = _fmt_gb_path(out_dir)
    print(f"[{label}] Saved ({saved_size}). Cleaning up BF16 temp cache …", flush=True)

    # Free memory before next model
    del transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shutil.rmtree(cache_dir, ignore_errors=True)

    total_elapsed = time.time() - t0
    free_after = _disk_free_gb("/opt/arthur-img-models")
    print(
        f"[{label}] DONE in {total_elapsed:.0f}s. "
        f"Disk free: {free_after:.1f} GB",
        flush=True,
    )
    return True


def main():
    print("=" * 60, flush=True)
    print("  nvfp4_save.py", flush=True)
    print("=" * 60, flush=True)

    # ── Check torchao ─────────────────────────────────────────────────────────
    try:
        import torchao  # noqa: F401
        print(f"torchao {torchao.__version__} found.", flush=True)
    except ImportError:
        print(
            "ERROR: torchao is not installed.\n"
            "Install it with: /opt/arthur-img-env/bin/pip install torchao",
            flush=True,
        )
        sys.exit(1)

    try:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig  # noqa: F401
        print("NVFP4WeightOnlyConfig: available.", flush=True)
    except ImportError as exc:
        print(f"ERROR: NVFP4WeightOnlyConfig not available: {exc}", flush=True)
        sys.exit(1)

    # ── Check diffusers TorchAoConfig ────────────────────────────────────────
    try:
        from diffusers import TorchAoConfig  # noqa: F401
        print("diffusers TorchAoConfig: available.", flush=True)
    except ImportError as exc:
        print(f"ERROR: diffusers TorchAoConfig not available: {exc}", flush=True)
        sys.exit(1)

    os.makedirs(NVFP4_ROOT, exist_ok=True)
    os.makedirs(TEMP_BF16, exist_ok=True)

    print(f"\nNVFP4_ROOT : {NVFP4_ROOT}", flush=True)
    print(f"TEMP_BF16  : {TEMP_BF16}", flush=True)
    print(f"Disk free  : {_disk_free_gb('/opt/arthur-img-models'):.1f} GB\n", flush=True)

    # ── Run jobs sequentially ─────────────────────────────────────────────────
    results = {}
    for label, hf_repo, subfolder, out_dir in JOBS:
        ok = save_nvfp4_transformer(label, hf_repo, subfolder, out_dir)
        results[label] = ok

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  nvfp4_save.py — COMPLETE", flush=True)
    print("=" * 60, flush=True)
    ok_count   = sum(1 for v in results.values() if v)
    fail_count = sum(1 for v in results.values() if not v)
    for label, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {status}  {label}", flush=True)
    print(f"\nPassed: {ok_count}  Failed: {fail_count}", flush=True)

    if fail_count:
        print(
            "\nNOTE: Failed models will fall back to GGUF at runtime. "
            "Re-run this script after fixing the error to retry.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
