#!/usr/bin/env python3
"""
bench_image_lab.py — Arthur Image Lab benchmark runner.

Runs each image-generating engine / quant combination with a fixed prompt,
records wall-clock timing per phase, saves generated images, and writes a
summary JSON that bench_report_gen.py uses to build the Word document.

Run on the VM (calls localhost:8002) so network RTT is zero:
  /opt/arthur-img-env/bin/python /opt/arthur-img/bench_image_lab.py

Results land in /tmp/bench/
"""

from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("IMGLAB_API_BASE", "http://localhost:8002")
REQUEST_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT_S", "900"))
DEFAULT_BENCH_DIR = r"C:\Temp\bench" if os.name == "nt" else "/tmp/bench"
OUT_DIR = Path(os.environ.get("BENCH_DIR", DEFAULT_BENCH_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many sequential inference runs to perform per-model (hot runs)
NUM_HOT_RUNS = int(os.environ.get("BENCH_NUM_HOT_RUNS", "3"))

PROMPT     = (
    "a photorealistic mountain lake at golden hour, reflections of snow-capped "
    "peaks, misty atmosphere, dramatic clouds, cinematic lighting, 8k"
)
NEG_PROMPT = "blurry, low quality, noise, watermark, text"
SEED       = 42
WIDTH      = 1024
HEIGHT     = 1024
NUM_IMAGES = 3

# (engine_key, quant, num_inference_steps, label)
# Use shorter 'turbo-style' runs for SD to measure steady-state inference quickly
# (steps reduced to 4 and 8 to emulate distilled / lightning checkpoints)
RUNS = [
    ("sd35",       "Q4_0",    4, "SD 3.5 Large · GGUF Q4_0 (turbo-style, 4 steps)"),
    ("sd35",       "nvfp4",   8, "SD 3.5 Large · NVFP4 (lightning, 8 steps)"),
    ("flux2klein", "",        4, "FLUX.2 Klein 4B · BF16 (distilled, 4 steps)"),
    ("flux2",      "Q4_K_M", 28, "FLUX.2 [dev] · GGUF Q4_K_M"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_mb() -> tuple[int, int]:
    """Return (used_MiB, free_MiB) from nvidia-smi, or (-1, -1) on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        used, free = map(int, out.split(", "))
        return used, free
    except Exception:
        return -1, -1


def journal_tail(since_epoch: float, n: int = 80) -> str:
    """Return last n lines of the imglab service journal since since_epoch."""
    since_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since_epoch - 2))
    try:
        return subprocess.check_output(
            ["journalctl", "-u", "arthur-imglab.service",
             "--since", since_str, "--no-pager", "-n", str(n)],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""


def parse_timing(journal: str) -> tuple[float | None, float | None]:
    """
    Extract (load_s, inference_s) from a journal snippet.
    Looks for patterns like:
      "SD 3.5 Large ready (quant=nvfp4) in 21.6 s"
      "10/10 [00:52<00:00,  5.24s/it]"
    """
    load_s = None
    inf_s  = None

    m = re.search(r"ready \([^)]+\) in ([\d.]+) s", journal)
    if m:
        load_s = float(m.group(1))

    # tqdm line: "10/10 [00:52<00:00"  → total = mm:ss
    m = re.search(r"\d+/\d+ \[(\d+):(\d+)<", journal)
    if m:
        inf_s = int(m.group(1)) * 60 + int(m.group(2))

    return load_s, inf_s


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

all_results: list[dict] = []
summary_path = OUT_DIR / "bench_results.json"

print(f"\nArthur Image Lab — Benchmark")
print(f"Prompt : {PROMPT[:80]}...")
print(f"Seed   : {SEED}   Resolution: {WIDTH}x{HEIGHT}")
print(f"Output : {OUT_DIR}\n")

for engine_key, quant, steps, label in RUNS:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  engine={engine_key}  quant={quant or '(default)'}  steps={steps}")
    print(sep)

    # Query VRAM before load
    try:
        pre_status = requests.get(f"{API_BASE}/status?format=json", timeout=10).json()
        v = pre_status.get("vram", {})
        vram_before = (int(v.get("allocated_gb", 0) * 1024), int(v.get("free_gb", 0) * 1024))
    except Exception:
        vram_before = (-1, -1)

    status = "ok"
    error = ""
    load_s = None
    inf_times: list[float] = []
    image_files = []

    # 1) Preload engine and measure load time
    print(f"  Preloading engine (quant={quant or '(default)'}) …")
    try:
        t0 = time.time()
        resp = requests.post(f"{API_BASE}/engines/{engine_key}/load", timeout=REQUEST_TIMEOUT)
        load_s = round(time.time() - t0, 1)
        if resp.status_code != 200:
            status = "error"
            error = f"Load failed: HTTP {resp.status_code}: {resp.text[:300]}"
            print(f"  ✗ Load failed: {error}")
    except Exception as exc:
        load_s = round(time.time() - t0, 1) if 't0' in locals() else None
        status = "exception"
        error = str(exc)
        print(f"  ✗ Load exception: {exc}")

    # 2) Run inference passes (only if load succeeded)
    if status == "ok":
        for run_idx in range(NUM_HOT_RUNS):
            run_seed = SEED + run_idx
            payload = {
                "prompt":               PROMPT,
                "negative_prompt":      NEG_PROMPT,
                "width":                str(WIDTH),
                "height":               str(HEIGHT),
                "num_inference_steps":  str(steps),
                "guidance_scale":       "4.5" if engine_key == "sd35" else "3.5",
                "seed":                 str(run_seed),
                "num_images":           "1",
            }
            if quant:
                payload["quant"] = quant

            print(f"  Run {run_idx+1}/{NUM_HOT_RUNS} (seed={run_seed}) …")
            t0 = time.time()
            try:
                resp = requests.post(f"{API_BASE}/generate/{engine_key}", data=payload, timeout=REQUEST_TIMEOUT)
                elapsed = round(time.time() - t0, 2)
                inf_times.append(elapsed)
                
                if resp.status_code == 200:
                    data = resp.json()
                    for i, item in enumerate(data.get("results", [])):
                        img_b64 = item.get("base64")
                        if img_b64:
                            img_data = base64.b64decode(img_b64)
                            fname = f"{engine_key}_{quant or 'default'}_{run_idx+1}_{i+1}.png"
                            img_path = OUT_DIR / fname
                            img_path.write_bytes(img_data)
                            image_files.append(str(img_path))
                            print(f"    ✓ Image {run_idx+1}/{NUM_HOT_RUNS} saved ({len(img_data)//1024} KB) in {elapsed}s")
                else:
                    status = "error"
                    error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    print(f"    ✗ Generation failed: {error}")
                    break

            except Exception as exc:
                elapsed = round(time.time() - t0, 2)
                inf_times.append(elapsed)
                status = "exception"
                error = str(exc)
                print(f"    ✗ Exception: {exc} (elapsed {elapsed}s)")
                break

            time.sleep(1)  # settle between runs

    # Query VRAM after
    try:
        post_status = requests.get(f"{API_BASE}/status?format=json", timeout=10).json()
        v = post_status.get("vram", {})
        vram_after = (int(v.get("allocated_gb", 0) * 1024), int(v.get("free_gb", 0) * 1024))
    except Exception:
        vram_after = (-1, -1)

    # Compute stats
    inf_s = round(sum(inf_times) / len(inf_times), 1) if inf_times else None
    sps = round(steps / inf_s, 2) if inf_s and inf_s > 0 else None
    total_s = round((load_s or 0) + (sum(inf_times) or 0), 1)

    result: dict = {
        "label":       label,
        "engine":      engine_key,
        "quant":       quant or "default",
        "steps":       steps,
        "status":      status,
        "total_s":     total_s,
        "load_s":      load_s,
        "inf_s":       inf_s,
        "inf_times":   inf_times,
        "steps_per_s": sps,
        "vram_used_before_mb": vram_before[0],
        "vram_used_after_mb":  vram_after[0],
        "image_files": image_files,
    }
    if error:
        result["error"] = error

    print(f"  ✓ Summary: load={load_s}s  avg_inf={inf_s}s ({len(inf_times)} runs)  sps={sps}  total={total_s}s")
    all_results.append(result)

    # Unload and pause
    try:
        requests.post(f"{API_BASE}/engines/unload", timeout=30)
        time.sleep(2)
    except Exception:
        pass

    time.sleep(3)  # pause between models

# ---------------------------------------------------------------------------
# Write summary JSON
# ---------------------------------------------------------------------------
summary_path.write_text(json.dumps(all_results, indent=2))
print(f"\n{'='*62}")
print(f"All done.  Results: {summary_path}")
print(f"Images  : {OUT_DIR}/*.png")
