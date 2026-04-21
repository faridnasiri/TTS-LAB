#!/usr/bin/env bash
# fix_tts_env.sh — Fix all known TTS Lab dependency conflicts
#
# Run on the VM after any pip install that may have broken the env:
#   bash /opt/arthur/fix_tts_env.sh
#
# Fixes applied:
#   1. NLTK data for root user (/usr/share/nltk_data)
#   2. protobuf 3.x → 5.x (removes descript-audiotools 0.7.2 pin)
#   3. torchvision CUDA version matched to installed torch
#   4. numpy pinned <2.3 (numba/OuteTTS require)
#   5. opencv-python-headless ABI matched to numpy
#   6. vllm upgraded to 0.19.1+ (for Orpheus; torch 2.10 compatible)
#
# Safe to re-run — each step checks current state first.

set -uo pipefail
PIP=/opt/arthur-bench-env/bin/pip
PY=/opt/arthur-bench-env/bin/python3

ok()   { echo "  ✅  $*"; }
warn() { echo "  ⚠️   $*"; }
step() { echo ""; echo "━━━  $*  ━━━"; }

# ── 1. NLTK for root ──────────────────────────────────────────────────────────
step "1 — NLTK data → /usr/share/nltk_data"
if [ -d /usr/share/nltk_data/taggers/averaged_perceptron_tagger_eng ]; then
    ok "NLTK tagger already present"
else
    $PY - << 'PYEOF'
import nltk
for corpus in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict']:
    nltk.download(corpus, download_dir='/usr/share/nltk_data', quiet=True)
print("NLTK corpora installed to /usr/share/nltk_data")
PYEOF
fi

# ── 2. protobuf ───────────────────────────────────────────────────────────────
step "2 — protobuf 5.x (remove descript-audiotools 0.7.2 pin)"
PROTO_VER=$($PY -c "import google.protobuf; print(google.protobuf.__version__)" 2>/dev/null || echo "0")
MAJOR=${PROTO_VER%%.*}
if [ "${MAJOR:-0}" -ge 5 ]; then
    ok "protobuf $PROTO_VER — already 5.x"
else
    warn "protobuf $PROTO_VER — upgrading..."
    $PIP install 'tensorboard>=2.17' --upgrade -q
    $PIP uninstall descript-audiotools -y 2>/dev/null || true
    $PIP install 'descript-audiotools-unofficial' --force-reinstall -q
    $PIP install 'protobuf>=5.29.6' --upgrade -q
    ok "protobuf upgraded"
fi

# ── 3. torchvision CUDA match ─────────────────────────────────────────────────
step "3 — torchvision CUDA version match"
TORCH_VER=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_TAG=$(echo "$TORCH_VER" | grep -oP 'cu\d+' || echo "")
if [ -z "$CUDA_TAG" ]; then
    warn "Could not detect CUDA tag from torch=$TORCH_VER — skipping torchvision fix"
else
    TV_OK=$($PY -c "import torchvision; print('ok')" 2>/dev/null || echo "fail")
    if [ "$TV_OK" = "ok" ]; then
        ok "torchvision import OK (torch=$TORCH_VER)"
    else
        warn "torchvision import failed — reinstalling for $CUDA_TAG"
        $PIP install torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --upgrade -q
        ok "torchvision reinstalled"
    fi
fi

# ── 4. numpy <2.3 ─────────────────────────────────────────────────────────────
step "4 — numpy pinned <2.3"
NP_VER=$($PY -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0")
NP_MINOR=$(echo "$NP_VER" | cut -d. -f2)
if [ "${NP_MINOR:-99}" -ge 3 ]; then
    warn "numpy $NP_VER >= 2.3 — pinning back (numba/OuteTTS require <2.3)"
    $PIP install 'numpy<2.3' --upgrade -q
    ok "numpy pinned"
else
    ok "numpy $NP_VER — OK"
fi

# ── 5. opencv-python-headless ABI ────────────────────────────────────────────
step "5 — opencv-python-headless ABI"
CV_OK=$($PY -c "import cv2; print('ok')" 2>/dev/null || echo "fail")
if [ "$CV_OK" = "fail" ]; then
    warn "cv2 import failed — reinstalling"
    $PIP install opencv-python-headless --force-reinstall -q
    ok "cv2 reinstalled"
else
    ok "cv2 import OK"
fi

# ── 6. vllm for Orpheus ──────────────────────────────────────────────────────
step "6 — vllm >=0.9.0 (Orpheus/torch 2.10 compatible)"
VLLM_VER=$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "0")
VLLM_MAJOR=$(echo "$VLLM_VER" | cut -d. -f1)
VLLM_MINOR=$(echo "$VLLM_VER" | cut -d. -f2)
# Need >= 0.9
if [ "${VLLM_MAJOR:-0}" -gt 0 ] || [ "${VLLM_MINOR:-0}" -ge 9 ]; then
    ok "vllm $VLLM_VER — OK"
else
    warn "vllm $VLLM_VER — upgrading to >=0.9.0"
    $PIP install 'vllm>=0.9.0' --upgrade -q
    ok "vllm upgraded"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
step "Verification"
$PY - << 'PYEOF'
import sys
checks = []
def chk(label, fn):
    try: fn(); checks.append(f"OK  {label}")
    except Exception as e: checks.append(f"NO  {label}: {str(e)[:60]}")

chk("protobuf builder",  lambda: __import__("google.protobuf.internal", fromlist=["builder"]))
chk("audiotools",        lambda: __import__("audiotools"))
chk("numpy <2.3",        lambda: (_ for _ in ()).throw(Exception(f"numpy {__import__('numpy').__version__}")) if int(__import__('numpy').__version__.split('.')[1]) >= 3 else None)
chk("torch cuda",        lambda: (_ for _ in ()).throw(Exception("no cuda")) if not __import__("torch").cuda.is_available() else None)
chk("torchvision",       lambda: __import__("torchvision"))
chk("cv2",               lambda: __import__("cv2"))
chk("vllm",              lambda: __import__("vllm"))
chk("orpheus_tts",       lambda: __import__("orpheus_tts"))
chk("nltk tagger",       lambda: __import__("nltk").data.find("taggers/averaged_perceptron_tagger_eng"))

import numpy, torch
print(f"  numpy={numpy.__version__}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
for c in checks: print(f"  {c}")
PYEOF

echo ""
echo "Done. Restart service if any changes were made:"
echo "  sudo systemctl restart arthur-lab"
