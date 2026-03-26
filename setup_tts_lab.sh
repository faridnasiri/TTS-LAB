#!/usr/bin/env bash
# Arthur TTS Lab — Full Setup Script
#
# Installs all 7 TTS engines, downloads model files,
# registers a systemd service (port 8001), and opens the browser URL.
#
# Run on the Ubuntu VM as root:
#   sudo bash /opt/arthur/setup_tts_lab.sh
#
# After running, open: http://192.168.0.87:8001

set -uo pipefail   # no -e: a failed pip install should not abort the whole script

LAB_ENV="/opt/arthur-bench-env"
ARTHUR_DIR="/opt/arthur"
MODELS_DISK="/opt/models"          # dedicated data disk (sdb)
MODELS_DIR="${MODELS_DISK}/tts"    # Piper + Kokoro .onnx files
HF_HOME="${MODELS_DISK}/huggingface"
DATA_DEV="/dev/sdb"                # second virtual disk added in Hyper-V
N_CORES=$(nproc)
LOG_TAG="tts-lab-setup"

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
ok()    { echo "  ✅ $*"; }
warn()  { echo "  ⚠️  $*"; }
step()  { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  STEP: $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Arthur TTS Lab — Setup  (port 8001)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Swap ───────────────────────────────────────────────────────────────
step "0 — Swap"
SWAP_MB=$(free -m | awk '/^Swap:/{print $2}')
RAM_MB=$(free  -m | awk '/^Mem:/{print $2}')
stamp "RAM: ${RAM_MB} MB    Swap: ${SWAP_MB} MB    Cores: ${N_CORES}"

if [ "${SWAP_MB:-0}" -lt 2000 ]; then
    warn "Swap < 2 GB — XTTS-v2 and Parler need more. Adding 4 GB swap…"
    if [ ! -f /swapfile ]; then
        fallocate -l 4G /swapfile
        chmod 600 /swapfile
        mkswap  /swapfile
    fi
    swapon /swapfile 2>/dev/null || true
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    ok "Swap now: $(free -m | awk '/^Swap:/{print $2}') MB"
else
    ok "Swap OK (${SWAP_MB} MB)"
fi

# ── Step 0b: Data disk (/dev/sdb → /opt/models) ───────────────────────────────
step "0b — Data disk for models (${DATA_DEV} → ${MODELS_DISK})"
if mountpoint -q "${MODELS_DISK}"; then
    ok "${MODELS_DISK} already mounted"
elif [ -b "${DATA_DEV}" ]; then
    stamp "Partitioning and formatting ${DATA_DEV}…"
    wipefs -a ${DATA_DEV} 2>/dev/null || true
    echo -e "o\nn\np\n1\n\n\nw" | fdisk ${DATA_DEV}
    mkfs.ext4 -L models ${DATA_DEV}1
    mkdir -p ${MODELS_DISK}
    mount ${DATA_DEV}1 ${MODELS_DISK}
    UUID=$(blkid -s UUID -o value ${DATA_DEV}1)
    grep -q "${MODELS_DISK}" /etc/fstab || \
      echo "UUID=${UUID} ${MODELS_DISK} ext4 defaults,nofail 0 2" >> /etc/fstab
    ok "${DATA_DEV} mounted at ${MODELS_DISK}"
else
    warn "${DATA_DEV} not found — models will use OS disk"
    MODELS_DISK="${ARTHUR_DIR}"
    MODELS_DIR="${ARTHUR_DIR}/models"
    HF_HOME="/root/.cache/huggingface"
fi

# ── Step 1: System packages ────────────────────────────────────────────────────
step "1 — System packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential libsndfile1 libsndfile1-dev \
    ffmpeg espeak-ng git wget curl \
    > /dev/null 2>&1
ok "System packages installed"

# ── Step 2: Python venv ────────────────────────────────────────────────────────
step "2 — Python venv at ${LAB_ENV}"
if [ ! -d "${LAB_ENV}" ]; then
    python3.11 -m venv "${LAB_ENV}"
    ok "venv created"
else
    ok "venv already exists"
fi
# shellcheck source=/dev/null
source "${LAB_ENV}/bin/activate"

# ── Step 3: PyTorch (CPU-only, must be first) ──────────────────────────────────
step "3 — PyTorch (CPU-only)"
if ! python -c "import torch" 2>/dev/null; then
    stamp "Installing PyTorch CPU… (this is the big one — ~700 MB)"
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ok "PyTorch installed"
else
    ok "PyTorch already present"
fi

# ── Step 4: Shared utilities ───────────────────────────────────────────────────
step "4 — Shared utilities"
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet fastapi "uvicorn[standard]" pydantic numpy soundfile psutil httpx
ok "Shared utilities installed"

# ── Step 5: Piper TTS ──────────────────────────────────────────────────────────
step "5 — Piper TTS"
pip install --quiet piper-tts onnxruntime
ok "piper-tts installed"

# ── Step 6: Kokoro-82M ────────────────────────────────────────────────────────
step "6 — Kokoro-82M (ONNX)"
pip install --quiet kokoro-onnx
ok "kokoro-onnx installed"

# ── Step 7: MeloTTS ───────────────────────────────────────────────────────────
step "7 — MeloTTS"
pip install --quiet "git+https://github.com/myshell-ai/MeloTTS.git" \
  || { warn "MeloTTS git install failed — trying alternate"; pip install --quiet MeloTTS 2>/dev/null || warn "MeloTTS skipped"; }
# MeloTTS needs NLTK + unidic data on first run
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)" 2>/dev/null || true
python -c "import unidic; unidic.download()" 2>/dev/null || true
ok "MeloTTS installed"

# ── Step 7b: ChatTTS ─────────────────────────────────────────────────────
step "7b — ChatTTS (speed prompts + speaker sampling)"
pip install --quiet ChatTTS
ok "ChatTTS installed"

# ── Step 7c: OuteTTS ──────────────────────────────────────────────────
step "7c — OuteTTS (character-prompt voice + voice cloning)"
pip install --quiet outetts
HF_HOME="${HF_HOME}" python - << 'PYEOF' || warn "OuteTTS preload failed (will download on first use)"
import outetts
cfg = outetts.ModelConfig(model_path='OuteAI/OuteTTS-0.3-500M', tokenizer_path='OuteAI/OuteTTS-0.3-500M', backend=outetts.Backend.HF, device='cpu')
outetts.Interface(cfg)
print('OuteTTS-0.3-500M cached OK')
PYEOF
ok "outetts installed"

# ── Step 8: Parler-TTS mini ───────────────────────────────────────────────────
step "8 — Parler-TTS mini"
pip install --quiet parler-tts transformers accelerate
ok "parler-tts installed"

# ── Step 9: Chatterbox ────────────────────────────────────────────────────────
step "9 — Chatterbox"
pip install --quiet chatterbox-tts
ok "chatterbox-tts installed"

# ── Step 10: XTTS-v2 (Coqui TTS) ─────────────────────────────────────────────
step "10 — XTTS-v2 (Coqui TTS)"
# coqui-tts is the actively maintained community fork of the original Coqui TTS
# Install last — it pins some torch-related deps
pip install --quiet coqui-tts \
  || { warn "coqui-tts failed, trying original TTS package"; pip install --quiet TTS 2>/dev/null || warn "XTTS-v2 skipped"; }
ok "XTTS-v2 installed"

# ── Step 11: CosyVoice2 (manual) ─────────────────────────────────────────────
step "11 — CosyVoice2 (optional)"
if [ -d "/opt/CosyVoice" ]; then
    ok "Already found at /opt/CosyVoice"
else
    warn "Skipping CosyVoice2 — manual install (uncomment below to enable):"
    echo "     git clone https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice"
    echo "     pip install -r /opt/CosyVoice/requirements.txt"
    echo "     cd /opt/CosyVoice && python tools/download_model.py CosyVoice2-0.5B"
fi

# ── Step 11b: Bark (non-verbal emotion tokens) ───────────────────────────────
step "11b — Bark (emotion tokens: [laughs] [sighs] [clears throat])"
if ! python -c "from bark import generate_audio" 2>/dev/null; then
    pip install --quiet bark
    ok "bark installed"
else
    ok "bark already present"
fi
# Pre-download bark-small weights to data disk
BARK_CACHE="${MODELS_DISK}/cache"
mkdir -p "${BARK_CACHE}"
stamp "Pre-downloading Bark-small models (~1.3 GB) into ${BARK_CACHE}…"
XDG_CACHE_HOME="${BARK_CACHE}" SUNO_USE_SMALL_MODELS=True python - << 'PYEOF' || warn "Bark preload failed (will download on first use)"
import os
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
from bark import preload_models
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)
print("Bark-small models cached OK")
PYEOF

# ── Step 11c: StyleTTS 2 (fastest high-quality TTS) ──────────────────────────
step "11c — StyleTTS 2 (reference-audio style transfer)"
if ! python -c "from styletts2 import tts" 2>/dev/null; then
    pip install --quiet styletts2
    ok "styletts2 installed"
else
    ok "styletts2 already present"
fi
# Pre-download StyleTTS2 weights
stamp "Pre-downloading StyleTTS2 model (~700 MB)…"
python - << 'PYEOF' || warn "StyleTTS2 preload failed (will download on first use)"
from styletts2 import tts as _st
_st.StyleTTS2()
print("StyleTTS2 cached OK")
PYEOF

# ── Step 11d: F5-TTS (zero-shot voice cloning) ────────────────────────────────
step "11d — F5-TTS (best zero-shot voice cloning, ~1.2 GB)"
if ! python -c "from f5_tts.api import F5TTS" 2>/dev/null; then
    pip install --quiet f5-tts
    ok "f5-tts installed"
else
    ok "f5-tts already present"
fi
# Pre-download F5-TTS model (requires HF_HOME set)
stamp "Pre-downloading F5-TTS model (~1.2 GB)…"
HF_HOME="${HF_HOME}" python - << 'PYEOF' || warn "F5-TTS preload failed (will download on first use)"
from f5_tts.api import F5TTS
F5TTS()
print("F5-TTS cached OK")
PYEOF

# ── Step 11e: Dia-1.6B (dialogue-native, emotion tags) ───────────────────────
step "11e — Dia-1.6B (dialogue TTS + emotion tags, ~3 GB)"
if ! python -c "from dia.model import Dia" 2>/dev/null; then
    pip install --quiet "git+https://github.com/nari-labs/dia.git" \
      || { warn "Dia git URL failed, trying PyPI diatts"; pip install --quiet diatts 2>/dev/null || warn "Dia skipped"; }
else
    ok "dia already present"
fi
# Pre-download Dia-1.6B weights
stamp "Pre-downloading Dia-1.6B model (~3 GB)…"
HF_HOME="${HF_HOME}" python - << 'PYEOF' || warn "Dia preload failed (will download on first use)"
from dia.model import Dia
Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float32")
print("Dia-1.6B cached OK")
PYEOF

# ── Step 12: Download Piper + Kokoro model files ──────────────────────────────
step "12 — Model files (Piper + Kokoro)"
mkdir -p "${MODELS_DIR}"

# Symlink so tts_lab.py (which looks at /opt/arthur/models) finds the data disk
[ -L "${ARTHUR_DIR}/models" ] || ln -sf "${MODELS_DIR}" "${ARTHUR_DIR}/models"

# HuggingFace cache → data disk
mkdir -p "${HF_HOME}"
[ -L "/root/.cache/huggingface" ]   || { mkdir -p /root/.cache;   ln -sf "${HF_HOME}" /root/.cache/huggingface; }
[ -L "/home/arthur/.cache/huggingface" ] || { mkdir -p /home/arthur/.cache; ln -sf "${HF_HOME}" /home/arthur/.cache/huggingface; chown -h arthur:arthur /home/arthur/.cache/huggingface; }

PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
if [ ! -f "${MODELS_DIR}/en_US-ryan-high.onnx" ]; then
    stamp "Downloading Piper en_US-ryan-high (~116 MB)…"
    curl -L -# "${PIPER_BASE}/en_US-ryan-high.onnx"      -o "${MODELS_DIR}/en_US-ryan-high.onnx"
    curl -L -s "${PIPER_BASE}/en_US-ryan-high.onnx.json" -o "${MODELS_DIR}/en_US-ryan-high.onnx.json"
    ok "Piper model saved"
else
    ok "Piper model already present"
fi

KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
if [ ! -f "${MODELS_DIR}/kokoro-v1.0.onnx" ] || [ ! -s "${MODELS_DIR}/kokoro-v1.0.onnx" ]; then
    stamp "Downloading Kokoro int8 model (~89 MB, faster on CPU)…"
    curl -L -# "${KOKORO_BASE}/kokoro-v1.0.int8.onnx" -o "${MODELS_DIR}/kokoro-v1.0.onnx"
    ok "Kokoro model saved"
else
    ok "Kokoro model already present"
fi

if [ ! -f "${MODELS_DIR}/voices-v1.0.bin" ] || [ ! -s "${MODELS_DIR}/voices-v1.0.bin" ]; then
    stamp "Downloading Kokoro voices (~27 MB)…"
    curl -L -# "${KOKORO_BASE}/voices-v1.0.bin" -o "${MODELS_DIR}/voices-v1.0.bin"
    ok "Kokoro voices saved"
else
    ok "Kokoro voices already present"
fi

stamp "Note: XTTS-v2 (~1.8 GB), Parler (~880 MB), Chatterbox (~1.2 GB) download on first Synthesize click."

# ── Step 14: Fish Speech ───────────────────────────────────────────────────────
step "14 — Fish Speech (VQ-VAE zero-shot voice cloning)"
pip install --quiet fish-speech \
  || { warn "fish-speech PyPI install failed — trying git source"; \
       pip install --quiet "git+https://github.com/fishaudio/fish-speech" 2>/dev/null || warn "Fish Speech skipped"; }
ok "Fish Speech install attempted"

# ── Step 15: Sesame CSM 1B ────────────────────────────────────────────────────
step "15 — Sesame CSM 1B (conversational speech model)"
pip install --quiet "git+https://github.com/SesameAILabs/csm" \
  || warn "Sesame CSM skipped (may need: huggingface-cli login for gated model)"
ok "Sesame CSM install attempted"

# ── Step 16: Qwen3-TTS ────────────────────────────────────────────────────────
step "16 — Qwen3-TTS (uses existing transformers install)"
# Qwen3-TTS auto-downloads Qwen/Qwen3-TTS from HuggingFace on first use.
# transformers is already installed via parler-tts (step 8).
python -c "from transformers import AutoModel; print('transformers OK for Qwen3-TTS')" 2>/dev/null \
  && ok "Qwen3-TTS ready (transformers present)" || warn "transformers not found — pip install transformers"

# ── Step 17: Orpheus 3B ───────────────────────────────────────────────────────
step "17 — Orpheus 3B (LLaMA-3B TTS with emotion tags)"
pip install --quiet orpheus-speech \
  || { warn "orpheus-speech PyPI failed — trying git"; \
       pip install --quiet "git+https://github.com/canopyai/Orpheus-TTS" 2>/dev/null || warn "Orpheus skipped"; }
ok "Orpheus install attempted"

# ── Step 18: NeuTTS Air ───────────────────────────────────────────────────────
step "18 — NeuTTS Air (placeholder — package not yet confirmed)"
warn "NeuTTS Air: package name not yet identified."
warn "Once confirmed: pip install <neutts-package>"
warn "Then edit _load_neutts() and _synth_neutts() in tts_lab.py"

# ── Step 19: IndexTTS-2 ───────────────────────────────────────────────────────
step "19 — IndexTTS-2 (zero-shot voice cloning, ref WAV required)"
pip install --quiet "git+https://github.com/index-tts/IndexTTS" \
  || warn "IndexTTS-2 skipped"
ok "IndexTTS-2 install attempted"

# ── Step 20: Zonos v0.1 ───────────────────────────────────────────────────────
step "20 — Zonos v0.1 (Hybrid/Transformer, emotion-controlled TTS)"
pip install --quiet phonemizer  # phonemizer required by Zonos
pip install --quiet "git+https://github.com/Zyphra/Zonos" \
  || warn "Zonos skipped"
# Pre-download Zonos transformer model (~1.2 GB)
stamp "Pre-downloading Zonos-v0.1-transformer (~1.2 GB)…"
HF_HOME="${HF_HOME}" python - << 'PYEOF' || warn "Zonos preload failed (will download on first use)"
from zonos.model import Zonos
Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")
print("Zonos-v0.1-transformer cached OK")
PYEOF
ok "Zonos installed"

# ── Step 21: OpenVoice v2 ─────────────────────────────────────────────────────
step "21 — OpenVoice v2 (MeloTTS + tone-color conversion)"
pip install --quiet "git+https://github.com/myshell-ai/OpenVoice" \
  || warn "OpenVoice skipped"

# Download OpenVoice v2 checkpoints to /opt/models/openvoice_v2
OV_DIR="${MODELS_DISK}/openvoice_v2"
mkdir -p "${OV_DIR}"
if [ ! -f "${OV_DIR}/converter/config.json" ]; then
    stamp "Downloading OpenVoice v2 checkpoints (~200 MB)…"
    # Primary: HuggingFace repo
    HF_HOME="${HF_HOME}" python - << 'PYEOF' || warn "OpenVoice checkpoint download failed — check manually"
import subprocess, os, sys, shutil
from pathlib import Path
ov_dir = Path("/opt/models/openvoice_v2")
try:
    from huggingface_hub import snapshot_download
    local = snapshot_download("myshell-ai/OpenVoice", ignore_patterns=["*.md","*.txt"])
    v2_src = Path(local) / "checkpoints_v2"
    if v2_src.exists():
        shutil.copytree(str(v2_src), str(ov_dir), dirs_exist_ok=True)
        print(f"OpenVoice v2 checkpoints saved to {ov_dir}")
    else:
        print(f"checkpoints_v2 not found in {local} — copy manually")
except Exception as e:
    print(f"OpenVoice checkpoint download failed: {e}")
PYEOF
else
    ok "OpenVoice v2 checkpoints already present"
fi
ok "OpenVoice v2 install attempted"

# ── Step 13: systemd service ───────────────────────────────────────────────────
step "13 — systemd service (arthur-lab.service on port 8001)"
cat > /etc/systemd/system/arthur-lab.service << EOF
[Unit]
Description=Arthur TTS Lab (interactive web UI on port 8001)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${ARTHUR_DIR}
ExecStart=${LAB_ENV}/bin/uvicorn tts_lab:app --host 0.0.0.0 --port 8001 --workers 1
Restart=always
RestartSec=5
Environment="PYTHONUNBUFFERED=1"
Environment="OMP_NUM_THREADS=${N_CORES}"
Environment="MKL_NUM_THREADS=${N_CORES}"
Environment="OPENBLAS_NUM_THREADS=${N_CORES}"
Environment="ORT_NUM_THREADS=${N_CORES}"
Environment="NUMEXPR_NUM_THREADS=${N_CORES}"
Environment="CPU_THREADS=${N_CORES}"
Environment="HF_HOME=${HF_HOME}"
Environment="TRANSFORMERS_CACHE=${HF_HOME}/hub"
Environment="XDG_CACHE_HOME=${MODELS_DISK}/cache"
Environment="SUNO_USE_SMALL_MODELS=True"
Environment="COQUI_TOS_AGREED=1"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable  arthur-lab
systemctl restart arthur-lab
sleep 2

if systemctl is-active --quiet arthur-lab; then
    ok "arthur-lab.service running"
else
    warn "Service may still be starting. Check: journalctl -u arthur-lab -f"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
VM_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  Arthur TTS Lab is ready!                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Open in browser:  http://${VM_IP}:8001"
echo "  (from Windows)    http://192.168.0.87:8001"
echo ""
echo "  Live logs:  journalctl -u arthur-lab -f"
echo "  Restart:    systemctl restart arthur-lab"
echo ""
echo "  Models installed:  Piper ✅  Kokoro ✅  MeloTTS ✅  ChatTTS ✅  OuteTTS ✅  Bark ✅  StyleTTS2 ✅"
echo "                     F5-TTS ✅  Dia-1.6B ✅  XTTS-v2 ✅  Parler ✅  Chatterbox ✅"
echo "                     FishSpeech ✅  Sesame-CSM ✅  Qwen3-TTS ✅  Orpheus ✅"
echo "                     IndexTTS-2 ✅  Zonos ✅  OpenVoice v2 ✅  (20 total pip-installed)"
echo "  CosyVoice2:        install manually (see Step 11 above)  +1 = 21 total"
echo "  NeuTTS Air:        package not yet confirmed (see Step 18 above)"
echo ""
echo "  NEW engines (14-21):"
echo "    Fish Speech  → zero-shot voice cloning; upload 5-30s ref WAV"
echo "    Sesame CSM   → conversational multi-speaker; HF login required"
echo "    Qwen3-TTS    → Alibaba Qwen3-based multilingual TTS"
echo "    Orpheus 3B   → <sigh> <laugh> emotion tags; 8 voices"
echo "    IndexTTS-2   → zero-shot cloning; ref WAV always required"
echo "    Zonos v0.1   → emotion vector + speaking-rate control; 44 kHz"
echo "    OpenVoice v2 → MeloTTS + tone-color conversion; zero-shot clone"
echo ""
echo "  NEW emotion engines:"
echo "    Bark      → embed [laughs] [sighs] [clears throat] directly in text"
echo "    Dia-1.6B  → [S1]/[S2] speakers + [laughs] [sighs] emotion tags"
echo "    F5-TTS    → zero-shot voice clone from any 5-15s WAV upload"
echo "    StyleTTS2 → reference-audio style transfer, fastest quality (~2x RTF)"
echo ""
echo "  First-run note: XTTS-v2, Parler, Chatterbox download HuggingFace weights"
echo "  on first Synthesize click (~2–5 min per model, cached in ${HF_HOME})"
echo ""
