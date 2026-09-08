#!/usr/bin/env python3
"""
gguf_download.py — Pre-download ALL GGUF variants for every engine to local disk.

Run once on the VM to populate the GGUF cache before starting the service.
Files that already exist on disk are skipped without re-downloading.

Usage:
    /opt/arthur-img-env/bin/python /opt/arthur-img/gguf_download.py

Progress is printed to stdout so you can monitor it via:
    journalctl -fu arthur-imglab.service
  or in a screen / tmux session.
"""

import os
import sys
import time

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
GGUF_ROOT = "/opt/arthur-img-models/gguf"

# ---------------------------------------------------------------------------
# Download list  (repo_id, filename_in_repo, local_dir)
# ---------------------------------------------------------------------------
# sd35 + wan GGUF entries REMOVED 2026-09-07 (engines deleted — see
# docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md). Remaining GGUF
# engines (flux2klein9b + zimage) download on first use via _ensure_gguf(),
# so this pre-downloader has no entries left unless one is added explicitly.
DOWNLOADS = []


def _fmt_gb(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / 1e9:.2f} GB"
    except OSError:
        return "? GB"


def _disk_free_gb() -> float:
    import shutil
    st = shutil.disk_usage(GGUF_ROOT if os.path.isdir(GGUF_ROOT) else "/")
    return st.free / 1e9


def download_all():
    from huggingface_hub import hf_hub_download

    total = len(DOWNLOADS)
    print(f"=== GGUF pre-download: {total} files ===", flush=True)
    print(f"    Disk free: {_disk_free_gb():.1f} GB\n", flush=True)

    skipped = 0
    downloaded = 0
    errors = []

    for i, (repo_id, fname, local_dir) in enumerate(DOWNLOADS, 1):
        local_path = os.path.join(local_dir, fname)
        label = f"[{i:2d}/{total}]"

        if os.path.isfile(local_path):
            print(f"{label} SKIP  (exists, {_fmt_gb(local_path)}): {fname}", flush=True)
            skipped += 1
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"{label} START {repo_id}/{fname}", flush=True)
        t0 = time.time()
        try:
            hf_hub_download(
                repo_id               = repo_id,
                filename              = fname,
                local_dir             = local_dir,
                local_dir_use_symlinks= False,
                token                 = HF_TOKEN or None,
            )
            elapsed = time.time() - t0
            size    = _fmt_gb(local_path)
            speed   = os.path.getsize(local_path) / elapsed / 1e6
            print(f"{label} DONE  ({size}, {elapsed:.0f}s, {speed:.0f} MB/s): {fname}", flush=True)
            print(f"         Disk free: {_disk_free_gb():.1f} GB", flush=True)
            downloaded += 1
        except Exception as exc:
            print(f"{label} ERROR {fname}: {exc}", flush=True)
            errors.append((fname, str(exc)))

    print(f"\n=== GGUF pre-download finished ===", flush=True)
    print(f"    Downloaded : {downloaded}", flush=True)
    print(f"    Skipped    : {skipped}", flush=True)
    print(f"    Errors     : {len(errors)}", flush=True)
    if errors:
        for fname, err in errors:
            print(f"    ✗ {fname}: {err}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    download_all()
