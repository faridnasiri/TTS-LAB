#!/usr/bin/env bash
# Download TTS model files for the Arthur benchmark
# Run from tools/arthur_server/ directory
#
# Downloads to: tools/arthur_server/models/
# Total size: ~600 MB for Piper + Kokoro
# XTTS-v2 / Parler / Chatterbox download their weights via HuggingFace on first run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

echo "Downloading model files to ${MODELS_DIR}"
echo ""

# ── Piper TTS — en_US-ryan-high (~65 MB) ─────────────────────────────────────
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
if [ ! -f "${MODELS_DIR}/en_US-ryan-high.onnx" ]; then
    echo "▶ Piper: en_US-ryan-high (~65 MB)"
    wget -q --show-progress \
        "${PIPER_BASE}/en_US-ryan-high.onnx" \
        -O "${MODELS_DIR}/en_US-ryan-high.onnx"
    wget -q --show-progress \
        "${PIPER_BASE}/en_US-ryan-high.onnx.json" \
        -O "${MODELS_DIR}/en_US-ryan-high.onnx.json"
    echo "   ✅ Piper model saved"
else
    echo "✅ Piper model already present"
fi

# ── Kokoro-82M (~310 MB total) ────────────────────────────────────────────────
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files"
if [ ! -f "${MODELS_DIR}/kokoro-v1.0.onnx" ]; then
    echo "▶ Kokoro: kokoro-v1.0.onnx (~310 MB)"
    wget -q --show-progress \
        "${KOKORO_BASE}/kokoro-v1.0.onnx" \
        -O "${MODELS_DIR}/kokoro-v1.0.onnx"
    echo "   ✅ Kokoro model saved"
else
    echo "✅ Kokoro model already present"
fi

if [ ! -f "${MODELS_DIR}/voices-v1.0.bin" ]; then
    echo "▶ Kokoro: voices-v1.0.bin (~14 MB)"
    wget -q --show-progress \
        "${KOKORO_BASE}/voices-v1.0.bin" \
        -O "${MODELS_DIR}/voices-v1.0.bin"
    echo "   ✅ Kokoro voices saved"
else
    echo "✅ Kokoro voices already present"
fi

# ── Notes on auto-downloading models ─────────────────────────────────────────
echo ""
echo "ℹ  MeloTTS, XTTS-v2, Parler-TTS, and Chatterbox download their model"
echo "   weights automatically from HuggingFace on the first benchmark run."
echo "   (~200 MB MeloTTS, ~1.8 GB XTTS-v2, ~880 MB Parler-mini, ~1.2 GB Chatterbox)"
echo "   Ensure the VM has internet access during first run."
echo ""
echo "   HuggingFace cache: ~/.cache/huggingface/"
echo ""

echo "download_models.sh complete."
