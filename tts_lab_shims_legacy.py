"""
tts_lab_shims_legacy.py — minimal shims for the legacy engine container.

These 3 engines (indextts, parler, qwen3tts) run on torch 1.13 + tf 4.46 —
the versions they were BUILT for. Framework patches are NOT needed:
  - masking_utils EXISTS in tf 4.46
  - isin_mps_friendly EXISTS in tf 4.46
  - ExtensionsTrie, AddedToken EXIST in tf 4.46
  - download_url, find_pruneable_heads_and_indices EXIST in tf 4.46
  - PretrainedConfig.pad_token_id EXISTS in tf 4.46
  - _trace_wrapped_higher_order_op does NOT exist in torch 1.13
  - inspect.getsourcefile crash does NOT happen on torch 1.13

Only system configuration and engine-specific fixes remain.
"""
from __future__ import annotations
import os

# ── Thread-pool pinning (before any ML import) ──────────────────
_N_CORES = os.cpu_count() or 6
os.environ.setdefault("OMP_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("MKL_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(_N_CORES))
os.environ.setdefault("ORT_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("XDG_CACHE_HOME",       "/opt/models/cache")
os.environ.setdefault("COQUI_TOS_AGREED",     "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Device detection ────────────────────────────────────────────
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
DEVICE_NAME = _torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
try:
    _free, _total = _torch.cuda.mem_get_info()
    VRAM_TOTAL_MB = int(_total // 1048576)
except Exception:
    VRAM_TOTAL_MB = 0
_N_CORES = os.cpu_count() or 6

# ── torch.isin wrapper (parler may need this on torch 1.13) ─────
# torlier TTS references torch.isin with an older API.
# On torch 1.13 this should work natively, but include the shim
# just in case parler-tts was written for an even older API.
_orig_isin = _torch.isin
def _compat_isin(*args, **kwargs):
    if "elements" in kwargs:
        kwargs["input"] = kwargs.pop("elements")
    return _orig_isin(*args, **kwargs)
_torch.isin = _compat_isin
