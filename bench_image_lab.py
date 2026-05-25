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
API_BASE = "http://localhost:8002"
OUT_DIR  = Path("/tmp/bench")
OUT_DIR.mkdir(exist_ok=True)

PROMPT     = (
    "a photorealistic mountain lake at golden hour, reflections of snow-capped "
    "peaks, misty atmosphere, dramatic clouds, cinematic lighting, 8k"
)
NEG_PROMPT = "blurry, low quality, noise, watermark, text"
SEED       = 42
WIDTH      = 1024
HEIGHT     = 1024

# (engine_key, quant, num_inference_steps, label)
RUNS = [
    ("sd35",       "Q4_0",   28, "SD 3.5 Large · GGUF Q4_0"),
    ("sd35",       "nvfp4",  28, "SD 3.5 Large · NVFP4 ⚡"),
    ("flux2klein", "",        4, "FLUX.2 Klein 4B · BF16 (distilled)"),
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

    vram_before = vram_mb()
    t_start = time.time()

    payload = {
        "prompt":               PROMPT,
        "negative_prompt":      NEG_PROMPT,
        "width":                str(WIDTH),
        "height":               str(HEIGHT),
        "num_inference_steps":  str(steps),
        "guidance_scale":       "4.5" if engine_key == "sd35" else "3.5",
        "seed":                 str(SEED),
    }
    if quant:
        payload["quant"] = quant

    status   = "ok"
    error    = ""
    img_path = None

    try:
        resp = requests.post(
            f"{API_BASE}/generate/{engine_key}",
            data=payload,
            timeout=900,   # 15 min max (flux2 GGUF cold load can be slow)
        )
        t_end    = time.time()
        total_s  = t_end - t_start

        if resp.status_code == 200:
            data    = resp.json()
            item    = data["results"][0]
            img_b64 = item["base64"]
            img_data = base64.b64decode(img_b64)

            fname    = f"{engine_key}_{quant or 'default'}.png"
            img_path = OUT_DIR / fname
            img_path.write_bytes(img_data)
            print(f"  Image saved → {img_path}  ({len(img_data)//1024} KB)")
        else:
            status = "error"
            error  = f"HTTP {resp.status_code}: {resp.text[:300]}"
            print(f"  FAIL: {error}")

    except Exception as exc:
        t_end   = time.time()
        total_s = t_end - t_start
        status  = "exception"
        error   = str(exc)
        print(f"  EXCEPTION: {exc}")

    vram_after = vram_mb()

    # Parse timing from journal
    journal   = journal_tail(t_start)
    load_s, inf_s = parse_timing(journal)
    if load_s is None:
        print(f"  (journal parse: no 'ready' line found)")
    if inf_s is None:
        print(f"  (journal parse: no tqdm line found)")

    # Steps/sec
    sps = round(steps / inf_s, 2) if inf_s and inf_s > 0 else None

    result: dict = {
        "label":       label,
        "engine":      engine_key,
        "quant":       quant or "default",
        "steps":       steps,
        "status":      status,
        "total_s":     round(total_s, 1),
        "load_s":      load_s,
        "inf_s":       inf_s,
        "steps_per_s": sps,
        "vram_used_before_mb": vram_before[0],
        "vram_used_after_mb":  vram_after[0],
        "image_file":  str(img_path) if img_path else None,
    }
    if error:
        result["error"] = error

    print(
        f"  total={total_s:.1f}s  load={load_s}s  inf={inf_s}s  "
        f"sps={sps}  VRAM={vram_after[0]} MiB"
    )
    all_results.append(result)

    # Brief pause so the model is evicted and VRAM settles
    time.sleep(5)

# ---------------------------------------------------------------------------
# Write summary JSON
# ---------------------------------------------------------------------------
summary_path.write_text(json.dumps(all_results, indent=2))
print(f"\n{'='*62}")
print(f"All done.  Results: {summary_path}")
print(f"Images  : {OUT_DIR}/*.png")
