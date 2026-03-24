#!/usr/bin/env bash
# Arthur TTS Benchmark — VM setup + run script
#
# Run on the Ubuntu VM (192.168.0.87):
#   chmod +x run_benchmark.sh
#   sudo bash run_benchmark.sh
#
# Then listen to /tmp/tts_bench/*.wav to evaluate voice quality.
# Numbers alone don't pick the winner — Arthur must SOUND like a confused 78-year-old.

set -euo pipefail

BENCH_ENV="/opt/arthur-bench-env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Arthur TTS Benchmark — Setup & Run             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Swap check ────────────────────────────────────────────────────────
echo "── Step 0: Checking swap ────────────────────────────────────────────────"
SWAP_MB=$(free -m | awk '/^Swap:/ { print $2 }')
RAM_MB=$(free -m  | awk '/^Mem:/  { print $2 }')
echo "   RAM: ${RAM_MB} MB    Swap: ${SWAP_MB} MB"

if [ "${SWAP_MB}" -lt 2000 ]; then
    echo ""
    echo "   ⚠  Swap is < 2 GB.  XTTS-v2 (~3.2 GB) and CosyVoice2 (~2.5 GB) will OOM."
    echo "   Adding 4 GB swap now..."
    if [ ! -f /swapfile ]; then
        fallocate -l 4G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
    fi
    swapon /swapfile 2>/dev/null || true
    # Persist across reboots
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "   ✅ Swap added: $(free -m | awk '/^Swap:/ { print $2 }') MB now available"
else
    echo "   ✅ Swap OK"
fi
echo ""

# ── Step 1: System packages ───────────────────────────────────────────────────
echo "── Step 1: System packages ──────────────────────────────────────────────"
apt-get install -y --no-install-recommends \
    python3.11-venv python3.11-dev \
    ffmpeg espeak-ng \
    build-essential libsndfile1 \
    git wget curl \
    > /dev/null 2>&1
echo "   ✅ Done"
echo ""

# ── Step 2: Benchmark venv ────────────────────────────────────────────────────
echo "── Step 2: Creating benchmark venv at ${BENCH_ENV} ─────────────────────"
if [ ! -d "${BENCH_ENV}" ]; then
    python3.11 -m venv "${BENCH_ENV}"
    echo "   ✅ venv created"
else
    echo "   ✅ venv already exists"
fi
# shellcheck source=/dev/null
source "${BENCH_ENV}/bin/activate"
echo ""

# ── Step 3: pip install ───────────────────────────────────────────────────────
echo "── Step 3: Installing benchmark packages (this takes a few minutes) ────"
pip install --quiet --upgrade pip setuptools wheel

# Install PyTorch first (CPU-only, smaller download)
pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install --quiet -r "${SCRIPT_DIR}/requirements_benchmark.txt"
echo "   ✅ Packages installed"
echo ""

# ── Step 4: Download model files ─────────────────────────────────────────────
echo "── Step 4: Downloading model files ─────────────────────────────────────"
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

# ── Step 5: CosyVoice2 manual install (optional) ─────────────────────────────
echo "── Step 5: CosyVoice2 (optional, ~5 GB download) ───────────────────────"
if [ ! -d "/opt/CosyVoice" ]; then
    echo "   To include CosyVoice2 in the benchmark, run:"
    echo "     git clone https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice"
    echo "     pip install -r /opt/CosyVoice/requirements.txt"
    echo "     cd /opt/CosyVoice && python tools/download_model.py CosyVoice2-0.5B"
    echo "   Skipping for now — benchmark will auto-skip this engine."
else
    echo "   ✅ CosyVoice found at /opt/CosyVoice"
fi
echo ""

# ── Step 6: Run benchmark ─────────────────────────────────────────────────────
echo "── Step 6: Running benchmark ────────────────────────────────────────────"
echo "   (each model tested sequentially, one in RAM at a time)"
echo ""

cd "${SCRIPT_DIR}"

# Pass --no-cosyvoice if /opt/CosyVoice doesn't exist
EXTRA_ARGS=""
[ ! -d "/opt/CosyVoice" ] && EXTRA_ARGS="${EXTRA_ARGS} --no-cosyvoice"

python tts_benchmark.py ${EXTRA_ARGS}

echo ""
echo "── Done ──────────────────────────────────────────────────────────────────"
echo "  WAV files: /tmp/tts_bench/*.wav"
echo "  Results  : ${SCRIPT_DIR}/benchmark_results.json"
echo ""
echo "  Listen to each WAV. The winner must sound like a confused 78-year-old,"
echo "  not just have the lowest RTF number."
echo ""
echo "  Recommended listening order (highest Arthur potential first):"
echo "    1. chatterbox.wav  — has exaggeration control for confusion"
echo "    2. kokoro.wav      — bm_lewis British male, warmth built-in"
echo "    3. xtts.wav        — highest quality if RAM allows"
echo "    4. parler.wav      — text-described voice, tune the prompt"
echo "    5. melo.wav        — clear American male, sounds younger"
echo "    6. piper.wav       — fastest but most robotic"
echo "    7. cosyvoice.wav   — great for Chinese; English accent varies"
echo ""
echo "  To deploy winner into arthur_server.py:"
echo "    1. Edit arthur_server.py — replace _speak() with the winning engine"
echo "    2. cd ${SCRIPT_DIR} && ./deploy.ps1  (from Windows host)"
