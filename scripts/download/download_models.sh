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
echo "ℹ  The following models auto-download from HuggingFace on first Synthesise click:"
echo ""
echo "   Engine           Size           Note"
echo "   MeloTTS          ~200 MB        auto (HF)"
echo "   ChatTTS          ~1.2-2.3 GB    auto (HF)"
echo "   OuteTTS          ~1.0 GB        auto (HF)"
echo "   Bark             ~1.3 GB        auto (XDG_CACHE_HOME or ~/.cache/suno)"
echo "   StyleTTS 2       ~700 MB        auto (HF)"
echo "   F5-TTS           ~1.2 GB        auto (HF)"
echo "   Dia-1.6B         ~3.0 GB        auto (HF)"
echo "   XTTS-v2          ~1.8 GB        auto (Coqui ~/.cache/tts)"
echo "   Parler-TTS       ~880 MB        auto (HF)"
echo "   Chatterbox       ~1.2 GB        auto (HF)"
echo "   Fish Speech      ~1.1 GB        auto (HF) on first load"
echo "   Sesame CSM 1B    ~2.0 GB        auto (HF, GATED — huggingface-cli login)"
echo "   Qwen3-TTS        ~1-3 GB        auto (HF) via transformers"
echo "   Orpheus 3B       ~3.0 GB        auto (HF) canopylabs/orpheus-3b-0.1-ft"
echo "   IndexTTS-2       ~1.5 GB        auto (HF) IndexTeam/IndexTTS"
echo "   Zonos v0.1       ~1.2 GB        pre-fetched by setup_tts_lab.sh step 20"
echo "   OpenVoice v2     ~200 MB        pre-fetched by setup_tts_lab.sh step 21"
echo "   CosyVoice2       ~2.0 GB        MANUAL: see setup_tts_lab.sh Step 11"
echo ""
echo "   Total (all except CosyVoice2): ~22-27 GB on first run"
echo "   Ensure /opt/models has 30+ GB free before running all engines."
echo ""
echo "   HuggingFace cache: \${HF_HOME:-~/.cache/huggingface/}"
echo ""

echo "download_models.sh complete."
