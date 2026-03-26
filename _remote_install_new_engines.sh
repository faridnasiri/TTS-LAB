#!/usr/bin/env bash
# Runs on the Ubuntu VM — called by deploy_tts_lab.ps1
# Installs pip packages for engines 14-21, syntax-checks tts_lab.py,
# then prints a final availability table.
# All installs are best-effort (|| true) so one failure can't break the rest.

LAB_ENV="/opt/arthur-bench-env"
ARTHUR_DIR="/opt/arthur"
HF_HOME="${HF_HOME:-/opt/models/huggingface}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Arthur TTS Lab — Installing new engines 14-21"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

source "${LAB_ENV}/bin/activate"
PY="${LAB_ENV}/bin/python"
PIP="${LAB_ENV}/bin/pip"

ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }
skip() { echo "  ⏭  $*"; }

# ── 14. Fish Speech ───────────────────────────────────────────────────────────
echo "▶ [14] Fish Speech (VQ-VAE voice cloning)..."
${PIP} install --quiet fish-speech \
  && ok "fish-speech installed" \
  || { warn "fish-speech PyPI failed — trying git source..."
       ${PIP} install --quiet "git+https://github.com/fishaudio/fish-speech" \
         && ok "fish-speech installed from git" \
         || warn "Fish Speech skipped — check: pip install fish-speech"; }

# ── 15. Sesame CSM 1B ─────────────────────────────────────────────────────────
echo "▶ [15] Sesame CSM 1B..."
${PIP} install --quiet "git+https://github.com/SesameAILabs/csm" \
  && ok "Sesame CSM installed" \
  || warn "Sesame CSM skipped — may also need: huggingface-cli login (gated model)"

# ── 16. Qwen3-TTS ─────────────────────────────────────────────────────────────
echo "▶ [16] Qwen3-TTS (uses existing transformers)..."
${PY} -c "from transformers import AutoModel; print('  ✅ transformers present — Qwen3-TTS ready')" \
  || warn "transformers not found — pip install transformers"

# ── 17. Orpheus 3B ────────────────────────────────────────────────────────────
echo "▶ [17] Orpheus 3B (LLaMA-3B TTS)..."
${PIP} install --quiet orpheus-speech \
  && ok "orpheus-speech installed" \
  || { warn "orpheus-speech PyPI failed — trying git..."
       ${PIP} install --quiet "git+https://github.com/canopyai/Orpheus-TTS" \
         && ok "orpheus installed from git" \
         || warn "Orpheus skipped — check: pip install orpheus-speech"; }

# Patch hardcoded snac_device="cuda" → "cpu" in decoder.py (CPU-only VM fix)
DECODER=$(${PY} -c "import orpheus_tts.decoder as d; import inspect; print(inspect.getfile(d))" 2>/dev/null)
if [ -n "$DECODER" ]; then
    ${PY} -c "
p='$DECODER'
t=open(p).read().replace('snac_device = \"cuda\"','snac_device = \"cpu\"')
open(p,'w').write(t)
" && ok "decoder.py patched: snac_device=cpu" || warn "decoder.py patch failed"
fi

# ── 18. NeuTTS Air ────────────────────────────────────────────────────────────
skip "[18] NeuTTS Air — package unconfirmed; edit _load_neutts() in tts_lab.py once identified"

# ── 19. IndexTTS-2 ────────────────────────────────────────────────────────────
echo "▶ [19] IndexTTS-2 (zero-shot cloning)..."
${PIP} install --quiet "git+https://github.com/index-tts/IndexTTS" \
  && ok "IndexTTS-2 installed" \
  || warn "IndexTTS-2 skipped — check: pip install git+https://github.com/index-tts/IndexTTS"

# ── 20. Zonos v0.1 ────────────────────────────────────────────────────────────
echo "▶ [20] Zonos v0.1 (emotion-controlled TTS)..."
${PIP} install --quiet phonemizer \
  && ok "phonemizer installed"
${PIP} install --quiet "git+https://github.com/Zyphra/Zonos" \
  && ok "Zonos installed" \
  || warn "Zonos skipped — check: pip install git+https://github.com/Zyphra/Zonos"

# ── 21. OpenVoice v2 ─────────────────────────────────────────────────────────
echo "▶ [21] OpenVoice v2 (MeloTTS + tone-color)..."
${PIP} install --quiet "git+https://github.com/myshell-ai/OpenVoice" \
  && ok "OpenVoice v2 installed" \
  || warn "OpenVoice skipped — check: pip install git+https://github.com/myshell-ai/OpenVoice"

# Download OpenVoice v2 checkpoints if not present
OV_DIR="/opt/models/openvoice_v2"
if [ ! -f "${OV_DIR}/converter/config.json" ]; then
    echo "  ↓ Downloading OpenVoice v2 checkpoints..."
    mkdir -p "${OV_DIR}"
    HF_HOME="${HF_HOME}" ${PY} - << 'PYEOF' || warn "OpenVoice checkpoint download failed — run manually"
from pathlib import Path
import shutil
ov_dir = Path("/opt/models/openvoice_v2")
try:
    from huggingface_hub import snapshot_download
    local = snapshot_download("myshell-ai/OpenVoice", ignore_patterns=["*.md","*.txt","*.gitignore"])
    v2_src = Path(local) / "checkpoints_v2"
    if v2_src.exists():
        shutil.copytree(str(v2_src), str(ov_dir), dirs_exist_ok=True)
        print(f"  ✅ OpenVoice v2 checkpoints → {ov_dir}")
    else:
        print(f"  ⚠  checkpoints_v2 not found in snapshot — copy from {local} manually")
except Exception as e:
    print(f"  ⚠  {e}")
PYEOF
else
    ok "OpenVoice v2 checkpoints already present"
fi

# ── Syntax check ──────────────────────────────────────────────────────────────
echo ""
echo "━━━━  Syntax check  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
${PY} -c "
import ast, sys
src = open('${ARTHUR_DIR}/tts_lab.py').read()
try:
    ast.parse(src)
    print('  ✅ tts_lab.py  syntax OK')
except SyntaxError as e:
    print(f'  ❌ tts_lab.py  SYNTAX ERROR at line {e.lineno}: {e.msg}')
    sys.exit(1)
"

echo ""
echo "━━━━  Install complete  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
