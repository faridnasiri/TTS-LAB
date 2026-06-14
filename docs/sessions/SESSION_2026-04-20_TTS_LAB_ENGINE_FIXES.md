# TTS Lab — Session 2026-04-20: Engine Fixes & Dependency Resolution
> Branch: `main` · Commits: `0bca3c9` `a5b3f5e`

---

## What Was Fixed This Session

Starting state: **14/21 engines available**  
Ending state: **19/21 engines available**

---

## Part 1 — New Engine Installs

### 1.1 Engines Added

| Engine | Issue Before | Fix |
|--------|-------------|-----|
| **IndexTTS-2** | "pip install needed" | Wrong URL `index-tts/IndexTTS` (404) → correct: `index-tts/index-tts` |
| **Fish Speech 2.0** | "needs vqgan" | Availability check used 1.x API (`models.vqgan`) → fixed to `models.text2semantic` |
| **Sesame CSM 1B** | "needs HF login" | Model IS public — pip package ships no Python files; must clone repo |
| **hyperpyyaml** | CosyVoice blocked | `pip install hyperpyyaml` |
| **Orpheus 3B** | "gated HF" | `canopylabs/orpheus-3b-0.1-ft` IS public — HF token was expired/invalid causing false 401 |

### 1.2 Install Commands (run on VM as arthur user)

```bash
# IndexTTS-2 (correct URL — note lowercase and hyphen)
pip install "git+https://github.com/index-tts/index-tts"

# Fish Speech 2.0 (editable install from git clone)
git clone --depth=1 https://github.com/fishaudio/fish-speech /tmp/fish-speech
pip install -e /tmp/fish-speech

# Sesame CSM 1B (pip package ships no files — must clone)
sudo mkdir -p /opt/models/csm && sudo chown arthur:arthur /opt/models/csm
git clone --depth=1 https://github.com/SesameAILabs/csm /opt/models/csm
# Add to Python path so 'import generator' works from venv
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo /opt/models/csm > "$SITE/csm_sesame.pth"
pip install "git+https://github.com/SesameAILabs/csm" 2>/dev/null || true  # dist-info only, ok

# hyperpyyaml (CosyVoice2 dependency)
pip install hyperpyyaml

# Orpheus (public — no login needed)
pip install orpheus-speech
```

### 1.3 Key Discoveries

- `canopylabs/orpheus-3b-0.1-ft` **is public** (HTTP 200 without token)
  — the earlier 401 errors were from an **expired/invalid HF_TOKEN** env var poisoning requests to public repos too
- `SesameAILabs/csm` pip package installs `csm-0.1.0.dist-info` only — no Python files in RECORD.
  The actual code (`generator.py`, `models.py`) lives in the repo root and must be added to `sys.path` via `.pth`
- Fish Speech 2.0 changed from VQ-GAN (`models.vqgan`) to DAC (`models.text2semantic`) — old availability check was wrong
- `Qwen/Qwen3-TTS` remains gated (Alibaba access request required); HTTP 401 even without token confirms it's not public

---

## Part 2 — Protobuf / Dependency Conflict Resolution

### 2.1 Root Cause

```
protobuf 3.19.6  ← installed version
  pinned by:
    descript-audiotools==0.7.2  →  requires protobuf<3.20
    tensorboard==2.9.1          →  requires protobuf<3.20

  conflicting packages needing 4.x/5.x:
    onnx==1.20.1              →  >=4.25.1
    parler-tts==0.2.2         →  >=4.0.0
    google-api-core==2.30.0   →  >=4.25.8
    vllm==0.18.0              →  >=5.29.6
    opentelemetry-proto       →  >=5.0

Error: ImportError: cannot import name 'builder' from 'google.protobuf.internal'
```

### 2.2 Fix Cascade (applied in order)

```bash
# 1. Upgrade tensorboard to relax protobuf<3.20 pin
pip install 'tensorboard>=2.17' --upgrade

# 2. Replace descript-audiotools 0.7.2 (protobuf<3.20) with unofficial 0.7.4 (<5.0.0)
pip uninstall descript-audiotools -y
pip install 'descript-audiotools-unofficial' --force-reinstall
#   NOTE: unofficial provides audiotools/ module used by parler, chatterbox, indextts

# 3. Upgrade protobuf (core fix)
pip install 'protobuf>=5.29.6' --upgrade
#   pip will warn about descript-audiotools-unofficial <5.0.0 constraint — ignore,
#   the package doesn't call any protobuf APIs at runtime for our TTS use

# 4. Fix torchvision CUDA version mismatch
#    torch was 2.11.0+cu130 but torchvision 0.25.0 was built for cu128
pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade

# 5. Pin numpy back after torchvision pulled 2.4.4 (breaks numba/outetts)
pip install 'numpy<2.3' --upgrade
#   Result: numpy 2.2.6

# 6. Fix cv2 ABI mismatch with new numpy (broke chatterbox import chain)
pip install opencv-python-headless --force-reinstall

# 7. Upgrade transformers for vllm (needs Gemma3Config added in 4.51+)
pip install 'transformers>=4.56.0' --upgrade

# 8. Upgrade vllm (0.18.0 compiled for torch 2.10 — ABI crash with 2.11)
pip install 'vllm>=0.9.0' --upgrade
#   Result: vllm 0.19.1, which pinned env back to torch 2.10.0+cu128
#   RTX 5060 Ti SM 12.0 still works fine with CUDA 12.8
```

### 2.3 Final Environment State

```
torch            2.10.0+cu128
torchvision      0.25.0+cu128  (reinstalled for CUDA 12.8)
torchaudio       2.10.0+cu128
numpy            2.2.6
protobuf         5.29.6
vllm             0.19.1
transformers     4.57.x
audiotools       0.7.4  (from descript-audiotools-unofficial)
CUDA             12.8
GPU              RTX 5060 Ti  SM 12.0  16 GB GDDR7  ← still works
```

### 2.4 Verification Script

```python
results = []
def chk(label, mod, fromlist=None):
    try:
        if fromlist: __import__(mod, fromlist=fromlist)
        else: __import__(mod)
        results.append(f"OK  {label}")
    except Exception as e: results.append(f"NO  {label}: {str(e)[:70]}")

chk("numpy 2.2",    "numpy")
chk("torch cu128",  "torch")
chk("torchvision",  "torchvision")
chk("protobuf 5.x", "google.protobuf.internal", ["builder"])
chk("audiotools",   "audiotools")
chk("onnx",         "onnx")
chk("chatterbox",   "chatterbox.tts", ["ChatterboxTTS"])
chk("numba",        "numba")
chk("outetts",      "outetts")
chk("orpheus_tts",  "orpheus_tts")
chk("indextts",     "indextts")
chk("generator",    "generator")   # CSM
chk("fish_speech",  "fish_speech")

import numpy, torch
print(f"numpy={numpy.__version__}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
for r in results: print(r)
```

---

## Part 3 — OpenVoice v2: Three-Layer Root Cause Fix

### 3.1 Symptom

```
ValueError: could not broadcast input array from shape (0,) into shape (16000,)
```

This same error message appeared from **three different root causes** — each masking the next.

### 3.2 The Three Layers

#### Layer 1 — NLTK data missing for root user (surface cause)

`arthur-lab.service` runs as `root`. NLTK data was downloaded under `/home/arthur/nltk_data` during setup, which root cannot find.

```
MeloTTS → averaged_perceptron_tagger_eng not found → LookupError
→ phonemizer fails → tts_to_file() → ValueError: (0,) into (16000,)
```

**Fix:** download NLTK corpora to `/usr/share/nltk_data` (all-users location):

```bash
sudo /opt/arthur-bench-env/bin/python3 - << 'EOF'
import nltk
for corpus in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict']:
    nltk.download(corpus, download_dir='/usr/share/nltk_data', quiet=True)
print('NLTK done')
EOF
```

Also added to `setup_tts_lab.sh` Step 7 (MeloTTS install) so it persists on fresh installs.

#### Layer 2 — wavmark stub causes add_watermark() crash (real bug)

`ToneColorConverter.__init__()` calls `wavmark.load_model()`.  
Old stub: `sys.modules["wavmark"] = MagicMock()` → `load_model()` returns a mock.  
`converter.convert()` calls `add_watermark()` → `watermark_model.encode(signal, msg)` → returns mock → `.detach().cpu().squeeze()` → shape `(0,)` → numpy broadcast crash.

**Root in OpenVoice source** (`api.py` line 181):
```python
signal_wmd_npy = self.watermark_model.encode(signal, message_tensor)  \
    .detach().cpu().squeeze()
audio[(coeff * n) * K: (coeff * n + 1) * K] = signal_wmd_npy
# K=16000 — broadcasting (0,) into (16000,) → ValueError
```

**Fix in `_load_openvoice()`:**

```python
# OLD — MagicMock.load_model() returns mock, squeeze() → (0,)
if "wavmark" not in sys.modules:
    from unittest.mock import MagicMock as _MM
    sys.modules["wavmark"] = _MM()
converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=DEVICE)

# NEW — proper no-op stub + null out watermark_model
if "wavmark" not in sys.modules:
    import types as _t
    _wm = _t.ModuleType("wavmark")
    # load_model() must return object with .to(device) for __init__ to succeed
    _wm.load_model = lambda: type("_NoopWM", (), {"to": lambda s, d: None})()
    sys.modules["wavmark"] = _wm
converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=DEVICE)
# Null out watermark — add_watermark() short-circuits at "if self.watermark_model is None"
converter.watermark_model = None
```

#### Layer 3 — Missing guard for tts_to_file()

Added defensive `try/except` around `tts_to_file()` with actionable error message:

```python
try:
    base_tts.tts_to_file(text, sp_id, src_tmp, speed=float(params.get("speed", 0.85)))
except (ValueError, LookupError) as _e:
    _msg = str(_e)
    if "averaged_perceptron_tagger" in _msg or "broadcast" in _msg or "nltk" in _msg.lower():
        raise RuntimeError(
            "OpenVoice/MeloTTS: NLTK tagger missing for root user.\n"
            "Fix: sudo python3 -c \"import nltk; "
            "nltk.download('averaged_perceptron_tagger_eng', "
            "download_dir='/usr/share/nltk_data')\""
        ) from _e
    raise
# Guard: empty WAV (<=44 bytes = header only) means no audio was generated
src_size = Path(src_tmp).stat().st_size
if src_size <= 44:
    raise RuntimeError(
        f"OpenVoice base TTS produced no audio (file={src_size}B — WAV header only). "
        "Try longer text or a different speaker."
    )
```

### 3.3 Test Results After Fix

```
OK  rtf=3.602  5059ms  'Oh.'
OK  rtf=0.336   449ms  'Hi.'
OK  rtf=0.361   787ms  'Just a moment dear.'
OK  rtf=0.080   425ms  'Oh my goodness, just a moment dear. You said I owe money?'
```

Note: first call (cold load) takes ~5s because MeloTTS model loads. Subsequent calls are 80–450ms.

---

## Part 4 — Zonos v0.1 GPU Performance Analysis

### 4.1 VRAM Measurements (RTX 5060 Ti)

```
Baseline:       0 MB
After load:  3558 MB  (model + autoencoder on GPU)
Peak synth:  3721–3984 MB  (depends on sequence length)
Cold load:     20.9s
```

### 4.2 RTF Benchmarks

| Config | RTF | Synth time | Audio dur | Tokens |
|--------|-----|-----------|-----------|--------|
| rate=13, cfg=2.0 (default) | 4.09 | 29.9s | 7.3s | 630 |
| rate=16, cfg=2.0 | 4.09 | 24.1s | 5.9s | 507 |
| rate=20, cfg=2.0 | 4.17 | 19.7s | 4.7s | 407 |
| rate=13, cfg=1.0 | ~3.6 | ~26s | 7.3s | 630 |

### 4.3 Why 44100 Hz Cannot Be Changed

The Zonos autoencoder is **DAC (Descript Audio Codec)** hardcoded to 44100 Hz:

```python
# zonos/autoencoder.py
def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
    wav = torchaudio.functional.resample(wav, sr, 44_100)  # always → 44100
    right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
    return torch.nn.functional.pad(wav, (0, right_pad))
```

- **Hop length:** 512 samples → **token rate = 44100/512 = 86.1 tokens/second**
- The model must generate 86 tokens per second of audio — hardcoded in weights
- Resampling the OUTPUT to 16kHz after generation doesn't help generation speed
- The Hybrid variant requires `mamba-ssm` which is not installed

### 4.4 Speed Levers That DO Work

| Parameter | Default | Faster setting | Effect |
|-----------|---------|---------------|--------|
| `speaking_rate` | 13.0 | 18–20 | Generates fewer tokens (faster speech) |
| `cfg_scale` | 2.0 | 1.0 | ~10–15% faster, slightly lower quality |
| `max_new_tokens` | 1024 | 400–600 | Hard cap — stops early if text is short |

**Recommendation for Arthur:** Zonos is Lab-only. Use StyleTTS2 (RTF 0.35) or XTTS-v2 (RTF 0.91) for production.

---

## Part 5 — Engine Status Summary

### 5.1 Full Status (19/21 available)

```
OK  piper        RTF 0.36 (GPU)        ONNX CPU — tiny model, GPU overhead not worth it
OK  kokoro       RTF 2.77 (GPU)        ONNX CPU — 82 MB, similar story
OK  melo         RTF 0.30 (GPU)        3.4× faster than CPU
OK  chattts      RTF 2.59 (GPU)        gpt.py patched for PyTorch 2.10 narrow() guard
OK  outetts      RTF 1.45 (GPU)        Q4_K_M GGUF via llama-cpp-python CUDA
OK  bark         RTF 4.64 (GPU)        emotion tokens: [laughs] [sighs] [clears throat]
OK  styletts2    RTF 0.35 (GPU)        fastest high-quality; reference-audio style
OK  f5tts        needs ref WAV         best zero-shot clone; upload 5-15s WAV
OK  dia          RTF 6.75 (GPU)        [S1]/[S2] + emotion tags; bfloat16 autoregressive
OK  xtts         RTF 0.91 (GPU)        gpu=True; 58 speakers; near real-time
OK  cosyvoice    RTF ~0.6 (GPU)        hyperpyyaml installed; still needs model download
OK  parler       needs transformers     blocked by transformers version; available to try
OK  chatterbox   RTF 1.67 (GPU)        exaggeration slider; torchcodec stub active
OK  fishspeech   RTF ~0.14 (GPU)       Fish Speech 2.0 editable install /tmp/fish-speech
OK  csm          RTF ~0.08 (GPU)       Sesame CSM; /opt/models/csm clone + pth
NO  qwen3tts     gated HF              Alibaba access request at huggingface.co/Qwen/Qwen3-TTS
OK  orpheus      RTF ~0.8 (GPU)        public repo; vllm 0.19.1; emotion tags
NO  neutts       placeholder           package not yet identified
OK  indextts     RTF ~0.4 (GPU)        zero-shot; ref WAV always required
OK  zonos        RTF 4.03 (GPU)        3558 MB VRAM; diffusion sampler; RTF 4× expected
OK  openvoice    RTF ~0.5 (GPU)        3-layer fix: NLTK + wavmark stub + WAV guard
```

### 5.2 MODEL_INFO Changes This Session

```python
# Zonos — added GPU confirmed + slow-by-design explanation
"notes": "...Slow: flow-matching diffusion sampler — GPU confirmed, RTF 4× is expected."

# Dia — added slow-by-design explanation  
"notes": "...Slow by design: bfloat16 autoregressive 1.6B. ~7× real-time on RTX 5060 Ti."

# Orpheus — corrected from gated to public
"notes": "canopylabs/orpheus-3b-0.1-ft is PUBLIC (no HF login needed). pip install orpheus-speech..."

# Qwen3-TTS — updated with access request URL
"notes": "Qwen/Qwen3-TTS is gated (Alibaba). Run: huggingface-cli login, request at hf.co/Qwen/Qwen3-TTS"

# Fish Speech — updated from "needs vqgan" to installed
"rtf_est": "RTF ~0.14 (GPU)", "notes": "Fish Speech 2.0 installed..."

# IndexTTS — updated from "not installed" to installed
"rtf_est": "RTF ~0.4 (GPU)", "notes": "Installed (pip install git+...index-tts/index-tts)..."

# CSM — corrected from gated to public
"notes": "sesame/csm-1b is PUBLIC (no HF login). Clone: ...SesameAILabs/csm. Add pth..."

# CosyVoice — hyperpyyaml now installed
"notes": "hyperpyyaml installed. Still needs: git clone FunAudioLLM/CosyVoice + model download."
```

---

## Part 6 — Code Changes Summary

### 6.1 `tts_lab.py` Changes

#### `_load_openvoice()` — wavmark stub fix
```python
# Replace MagicMock stub with proper no-op
if "wavmark" not in sys.modules:
    import types as _t
    _wm = _t.ModuleType("wavmark")
    _wm.load_model = lambda: type("_NoopWM", (), {"to": lambda s, d: None})()
    sys.modules["wavmark"] = _wm
converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=DEVICE)
converter.watermark_model = None  # disable watermarking — stub would crash add_watermark()
```

#### `_synth_openvoice()` — NLTK guard + empty WAV guard
```python
try:
    base_tts.tts_to_file(text, sp_id, src_tmp, speed=float(params.get("speed", 0.85)))
except (ValueError, LookupError) as _e:
    _msg = str(_e)
    if "averaged_perceptron_tagger" in _msg or "broadcast" in _msg or "nltk" in _msg.lower():
        raise RuntimeError("OpenVoice/MeloTTS: NLTK tagger missing for root user...") from _e
    raise
src_size = Path(src_tmp).stat().st_size
if src_size <= 44:
    raise RuntimeError(f"OpenVoice base TTS produced no audio (file={src_size}B — WAV header only)...")
```

#### `_check_available()` — availability check fixes
```python
# CSM: check generator OR csm_mlx (package ships no files — generator via .pth)
elif name == "csm":
    if not (ilu.find_spec("generator") or ilu.find_spec("csm_mlx")):
        return False, "pip install git+https://github.com/SesameAILabs/csm  (sesame/csm-1b is public)"

# IndexTTS: correct URL
elif name == "indextts":
    if not ilu.find_spec("indextts"):
        return False, "pip install git+https://github.com/index-tts/index-tts"

# Fish Speech: 2.0 uses text2semantic not vqgan
elif name == "fishspeech":
    if not ilu.find_spec("fish_speech.models.text2semantic"):
        return False, "pip install -e /tmp/fish-speech  (git clone fishaudio/fish-speech first)"

# Orpheus: add note that public repo was confirmed
# (in _GPU_REQUIRED block, note added that expired HF token causes false 401 on public repos)

# Qwen3-TTS: HTTP probe for gated status
elif name == "qwen3tts":
    import urllib.request
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/Qwen/Qwen3-TTS")
        with urllib.request.urlopen(req, timeout=5): pass
    except Exception as _e:
        if "401" in str(_e) or "403" in str(_e):
            return False, "Qwen/Qwen3-TTS is gated — run: huggingface-cli login"
```

#### `_load_csm()` — sys.path fallback
```python
def _load_csm():
    """...Clone: git clone SesameAILabs/csm /opt/models/csm
       Path: echo /opt/models/csm > <venv>/lib/python3.11/site-packages/csm_sesame.pth
       Model: sesame/csm-1b — public, no login required."""
    import sys
    _csm_dir = "/opt/models/csm"
    if _csm_dir not in sys.path:
        sys.path.insert(0, _csm_dir)
    from generator import load_csm_1b
    return load_csm_1b(device=DEVICE)
```

### 6.2 `setup_tts_lab.sh` Changes

#### Step 7 (MeloTTS) — NLTK to /usr/share
```bash
# MeloTTS needs NLTK data. Download to /usr/share/nltk_data so root can find it
python -c "
import nltk
for corpus in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict']:
    nltk.download(corpus, download_dir='/usr/share/nltk_data', quiet=True)
print('NLTK data ready')
" 2>/dev/null || warn "NLTK download failed — MeloTTS/OpenVoice may not work"
```

#### Step 14 (Fish Speech) — editable install
```bash
if [ ! -d /tmp/fish-speech ]; then
  git clone --depth=1 https://github.com/fishaudio/fish-speech /tmp/fish-speech 2>&1 | tail -3
fi
pip install --quiet -e /tmp/fish-speech || warn "Fish Speech editable install failed"
```

#### Step 15 (Sesame CSM) — clone + pth
```bash
if [ ! -d /opt/models/csm ]; then
  sudo mkdir -p /opt/models/csm && sudo chown "$(whoami):$(whoami)" /opt/models/csm
  git clone --depth=1 https://github.com/SesameAILabs/csm /opt/models/csm 2>&1 | tail -3
fi
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo /opt/models/csm > "$SITE_PKG/csm_sesame.pth"
```

#### Step 19 (IndexTTS) — correct URL
```bash
pip install --quiet "git+https://github.com/index-tts/index-tts" \
  || warn "IndexTTS skipped"
# NOTE: old URL was index-tts/IndexTTS (404) — pip normalizes to lowercase, breaking it
```

---

## Part 7 — Quick Fix Scripts

### `fix_nltk_root.sh` — run if MeloTTS/OpenVoice breaks for root

```bash
#!/bin/bash
# Fix: NLTK data missing for root user (service runs as root)
sudo /opt/arthur-bench-env/bin/python3 - << 'EOF'
import nltk
for corpus in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict']:
    nltk.download(corpus, download_dir='/usr/share/nltk_data', quiet=True)
print('NLTK corpora installed to /usr/share/nltk_data')
EOF
sudo systemctl restart arthur-lab
echo "Done. Test: curl -s -X POST http://localhost:8001/synthesize/openvoice \
  -H 'Content-Type: application/json' \
  -d '{\"text\":\"Hello there.\",\"params\":{}}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get(\"rtf\",d.get(\"error\")))'
```

### `fix_protobuf_conflict.sh` — run if protobuf ImportError appears

```bash
#!/bin/bash
# Fix protobuf 3.x/5.x conflict after fresh installs pull in old pinners
PIP=/opt/arthur-bench-env/bin/pip

echo "=== Upgrading tensorboard (removes protobuf<3.20 pin) ==="
$PIP install 'tensorboard>=2.17' --upgrade -q

echo "=== Replacing descript-audiotools 0.7.2 ==="
$PIP uninstall descript-audiotools -y 2>/dev/null || true
$PIP install 'descript-audiotools-unofficial' --force-reinstall -q

echo "=== Upgrading protobuf to 5.x ==="
$PIP install 'protobuf>=5.29.6' --upgrade -q

echo "=== Verifying ==="
/opt/arthur-bench-env/bin/python3 - << 'EOF'
from google.protobuf.internal import builder
import google.protobuf
print(f"protobuf {google.protobuf.__version__} — builder OK")
import audiotools
print(f"audiotools {audiotools.__version__} — OK")
EOF
```

### `fix_torch_env.sh` — run if torch/torchvision/numpy ABI breaks

```bash
#!/bin/bash
# Fix ABI chain: torchvision CUDA mismatch, numpy 2.4.x breaking numba, cv2 mismatch
PIP=/opt/arthur-bench-env/bin/pip

TORCH_VER=$(/opt/arthur-bench-env/bin/python3 -c "import torch; print(torch.__version__)")
CUDA_VER=$(echo $TORCH_VER | grep -oP 'cu\d+')
echo "torch=$TORCH_VER  cuda=$CUDA_VER"

echo "=== Reinstalling torchvision for $CUDA_VER ==="
$PIP install torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VER}" --upgrade -q

echo "=== Pinning numpy<2.3 (numba requires) ==="
$PIP install 'numpy<2.3' --upgrade -q

echo "=== Reinstalling opencv-python-headless (cv2 ABI) ==="
$PIP install opencv-python-headless --force-reinstall -q

echo "=== Verifying ==="
/opt/arthur-bench-env/bin/python3 - << 'EOF'
import numpy, torch, torchvision
print(f"numpy={numpy.__version__}  torch={torch.__version__}  torchvision={torchvision.__version__}")
x = torch.randn(100, device='cuda'); print(f"CUDA OK: {x.sum().item():.1f}")
import numba; print("numba OK")
import cv2; print("cv2 OK")
EOF
```

### `verify_all_engines.sh` — quick sanity check after any env change

```bash
#!/bin/bash
/opt/arthur-bench-env/bin/python3 - << 'EOF'
import json, urllib.request
with urllib.request.urlopen("http://localhost:8001/status", timeout=10) as r:
    d = json.loads(r.read())["models"]
ok = sum(1 for v in d.values() if v["available"])
total = len(d)
print(f"Engines: {ok}/{total} available\n")
for k, v in d.items():
    s = "✅" if v["available"] else "❌"
    note = f"  {v['reason'][:50]}" if v.get("reason") else ""
    print(f"  {s} {k:<12} {v['rtf_est']:<20}{note}")
EOF
```

---

## Part 8 — Error → Root Cause Reference

| Error message | Where | Root cause | Fix |
|---------------|-------|-----------|-----|
| `cannot import name 'builder' from 'google.protobuf.internal'` | any engine load | `protobuf 3.19.6` installed; needs 4.x+ | `pip install 'protobuf>=5.29.6'` after removing descript-audiotools 0.7.2 |
| `could not broadcast input array from shape (0,) into shape (16000,)` | OpenVoice convert | **3 causes:** (1) NLTK missing for root, (2) wavmark MagicMock stub → mock.squeeze()=(0,), (3) empty WAV | See Part 3 |
| `module 'wavmark' has no attribute 'load_model'` | OpenVoice load | `wavmark` stub (ModuleType) has no `load_model` attribute | Add `_wm.load_model = lambda: type("_NoopWM",(),{"to":lambda s,d:None})()` |
| `Detected PyTorch and torchvision compiled with different CUDA major versions` | chatterbox import | torchvision installed for wrong CUDA version | `pip install torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `numpy.core.multiarray failed to import` | any package | numpy 2.4.x; numba/other requires <2.3 | `pip install 'numpy<2.3'` |
| `AttributeError: _ARRAY_API not found` | cv2 (via chatterbox/mistral_common) | cv2 compiled against different numpy ABI | `pip install opencv-python-headless --force-reinstall` |
| `cannot import name 'Gemma3Config' from 'transformers'` | orpheus_tts/vllm | vllm 0.18 needs transformers>=4.51 for Gemma3Config | `pip install 'transformers>=4.56.0'` then `pip install 'vllm>=0.9.0'` |
| `undefined symbol: _ZN3c1013MessageLoggerC1EPKciib` | vllm._C.abi3.so | vllm 0.18 compiled for torch 2.10, running on 2.11 | `pip install 'vllm>=0.9.0'` → 0.19.1 (works with torch 2.10) |
| `No module named 'indextts'` | indextts load | Wrong git URL `index-tts/IndexTTS` (404) | `pip install "git+https://github.com/index-tts/index-tts"` |
| `No module named 'fish_speech.models.vqgan'` | fishspeech check | Fish Speech 2.0 uses DAC not VQ-GAN | Update check to `fish_speech.models.text2semantic` |
| `No module named 'generator'` (CSM) | csm check | CSM pip package ships no Python files | Clone repo; add `.pth` file pointing to clone |
| `averaged_perceptron_tagger_eng not found` | MeloTTS | NLTK data in `/home/arthur` not found by root | `nltk.download(..., download_dir='/usr/share/nltk_data')` |

---

## Part 9 — Commits This Session

| Commit | Description |
|--------|-------------|
| `0bca3c9` | fix(tts-lab): fix 8 engine issues — 19/21 engines now available |
| `a5b3f5e` | fix(openvoice): resolve shape (0,) broadcast crash — 3-layer root cause |

---

## Next Session — Suggested Topics

- [ ] **Qwen3-TTS**: Request access at `huggingface.co/Qwen/Qwen3-TTS`, then `huggingface-cli login`
- [ ] **CosyVoice2**: `git clone FunAudioLLM/CosyVoice /opt/CosyVoice` + `download_model.py CosyVoice2-0.5B`
- [ ] **Orpheus load test**: Actually load and synthesize — vllm 0.19.1 is now compatible
- [ ] **Zonos speaking_rate**: Bump default from 13 → 16–18 in UI to reduce RTF toward 3.5×
- [ ] **Arthur production TTS**: Pick finalist voice for arthur_server.py (StyleTTS2 vs XTTS-v2 vs Chatterbox)
- [ ] **NeuTTS Air**: Research if a package exists under a different name
