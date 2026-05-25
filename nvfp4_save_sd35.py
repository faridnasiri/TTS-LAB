#!/usr/bin/env python3
"""
nvfp4_save_sd35.py — Re-save ONLY the SD3.5 Large NVFP4 transformer.

Runs the same logic as nvfp4_save.py but for sd35 only.
Uses device_map={"":"cpu"} to prevent meta-tensor corruption.

Run in a screen session:
    screen -S nvfp4-sd35
    source /opt/arthur-img-env/bin/activate
    HF_TOKEN=<token> python /opt/arthur-img/nvfp4_save_sd35.py
"""
import sys
sys.path.insert(0, "/opt/arthur-img")
from nvfp4_save import save_nvfp4_transformer, NVFP4_ROOT

import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

label   = "sd35/transformer"
hf_repo = "stabilityai/stable-diffusion-3.5-large"
subfolder = "transformer"
out_dir = f"{NVFP4_ROOT}/sd35/transformer"

if os.path.isfile(os.path.join(out_dir, "config.json")):
    print(f"[{label}] Already saved at {out_dir}.")
    print("Delete it first if you want to re-save:")
    print(f"  rm -rf {out_dir}")
    sys.exit(0)

ok = save_nvfp4_transformer(label, hf_repo, subfolder, out_dir)
if ok:
    print("\nSD3.5 NVFP4 save complete!")
    print(f"Saved to: {out_dir}")
    print("Restart the service to pick it up:")
    print("  sudo systemctl restart arthur-imglab.service")
else:
    print("\nSD3.5 NVFP4 save FAILED — check output above.")
    sys.exit(1)
