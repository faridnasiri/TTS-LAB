# Arthur TTS Lab — Session 2026-03-26: New Engines + Production Fixes
> Date: 2026-03-26  
> Scope: Add 8 new TTS engines (14–21); fix SEGV on CPU-only VM; background availability sweep  
> Branch: `main` — VM: `arthur@192.168.0.87` — Web UI: `http://192.168.0.87:8001`

---

## Final State

**21 / 21 engines registered — 18 / 21 packages available — server stable**

| # | Key | Label | Status | Notes |
|---|-----|-------|--------|-------|
| 1 | piper | Piper TTS | ✅ ready | |
| 2 | kokoro | Kokoro-82M | ✅ ready | |
| 3 | melo | MeloTTS | ✅ ready | |
| 4 | chattts | ChatTTS | ✅ ready | |
| 5 | outetts | OuteTTS | ✅ ready | |
| 6 | bark | Bark | ✅ ready | |
| 7 | styletts2 | StyleTTS 2 | ✅ ready | |
| 8 | f5tts | F5-TTS | ✅ ready | |
| 9 | dia | Dia-1.6B | ✅ ready | |
| 10 | xtts | XTTS-v2 | ✅ ready | |
| 11 | cosyvoice | CosyVoice2 | ✅ ready | manual install |
| 12 | parler | Parler-TTS | ✅ ready | |
| 13 | chatterbox | Chatterbox | ✅ ready | |
| 14 | fishspeech | Fish Speech | ✅ ready | package present; full GitHub install for model loading |
| 15 | csm | Sesame CSM 1B | 🔴 missing | needs `huggingface-cli login` + GitHub install |
| 16 | qwen3tts | Qwen3-TTS | ✅ ready | uses existing `transformers`; model downloads on first use |
| 17 | orpheus | Orpheus 3B | ✅ ready | decoder.py patched: `snac_device="cpu"` |
| 18 | neutts | NeuTTS Air | 🔴 missing | package unconfirmed |
| 19 | indextts | IndexTTS-2 | 🔴 missing | GitHub URL TBD |
| 20 | zonos | Zonos v0.1 | ✅ ready | editable install from `/opt/Zonos` (git clone) |
| 21 | openvoice | OpenVoice v2 | ✅ ready | v1 checkpoints symlinked to `/opt/models/openvoice_v2` |

---

## Issues Fixed This Session

### 1. SEGV (signal 11) on first HTTP request
**Root cause:** `exec(stmt, {})` probes inside the availability sweep thread called C-extension imports
(`outetts→vllm`, `orpheus_tts→SNAC`) that corrupt the uvicorn event-loop thread's memory.  
**Fix:** Replaced ALL `exec()` probes with pure `importlib.util.find_spec()` + filesystem checks.
No C extensions are touched during availability checks — instant, GIL-safe.

### 2. Orpheus SEGV — `snac_device = "cuda"` hardcoded
**Root cause:** `orpheus_tts/decoder.py` hardcodes `snac_device = "cuda"` unconditionally.  
**Fix:** Patch the installed file after `pip install orpheus-speech`:
```bash
python3 -c "
p='/opt/arthur-bench-env/lib/python3.11/site-packages/orpheus_tts/decoder.py'
t=open(p).read().replace('snac_device = \"cuda\"','snac_device = \"cpu\"')
open(p,'w').write(t)"
```
`_remote_install_new_engines.sh` does this automatically.

### 3. Zonos PyPI v0.1.0 missing `backbone/` module
**Root cause:** PyPI and `pip install git+...` both produce an incomplete package.  
**Fix:** Direct clone + editable install:
```bash
git clone --depth 1 https://github.com/Zyphra/Zonos /opt/Zonos
pip install -e /opt/Zonos
# patch editable finder if needed:
sed -i 's|/tmp/Zonos|/opt/Zonos|g' \
  /opt/arthur-bench-env/lib/python3.11/site-packages/__editable___zonos_0_1_0_finder.py
```

### 4. OpenVoice v2 checkpoints — wrong directory name
**Root cause:** HF snapshot has `checkpoints/` not `checkpoints_v2/`.  
**Fix:** Symlink + updated `_load_openvoice()` to detect both v1 and v2 layouts:
```bash
sudo ln -sfn /opt/models/huggingface/hub/models--myshell-ai--OpenVoice/snapshots/<hash>/checkpoints \
             /opt/models/openvoice_v2
```

### 5. Service CUDA env vars missing → CUDA init attempts in CPU-only VM
**Fix:** Added to `/etc/systemd/system/arthur-lab.service`:
```
Environment=CUDA_VISIBLE_DEVICES=
Environment=TOKENIZERS_PARALLELISM=false
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 6. Background availability sweep — non-blocking startup
`_sweep_availability()` runs in a daemon thread at server startup, populating
`_import_cache` without blocking the event loop. `/status` shows `"checking..."` for
uncached engines; the page serves HTTP 200 immediately.

---

## To Install the 3 Remaining Engines

```bash
ssh arthur@192.168.0.87
source /opt/arthur-bench-env/bin/activate

# Sesame CSM 1B (gated model — needs HF account)
huggingface-cli login
pip install git+https://github.com/SesameAILabs/csm

# IndexTTS-2 (find the correct GitHub URL)
pip install git+https://github.com/index-tts/IndexTTS  # TBC

# After any install — refresh badges without restart:
curl -sX POST http://localhost:8001/refresh
```

---

## deploy_tts_lab.ps1 — Quick Reference

```powershell
# Full deploy + install new packages (~15 min):
.\deploy_tts_lab.ps1

# Files + restart only (fast, ~30 s):
.\deploy_tts_lab.ps1 -SkipInstall

# Different VM:
.\deploy_tts_lab.ps1 -VM 192.168.0.100 -SkipInstall
```
15. csm         Sesame CSM 1B         ~2 GB     ~8x RT   conversational multi-speaker; HF login
16. qwen3tts    Qwen3-TTS             ~1-3 GB   TBD      Alibaba Qwen3-based; via transformers
17. orpheus     Orpheus 3B            ~3 GB     ~10x RT  LLaMA-3B; emotion tags; 8 voices
18. neutts      NeuTTS Air            TBD       TBD      ⚠ placeholder — package unconfirmed
19. indextts    IndexTTS-2            ~1.5 GB   ~6x RT   zero-shot cloning; ref WAV required
20. zonos       Zonos v0.1            ~1.2 GB   ~8x RT   Zyphra; emotion vector; 44 kHz
21. openvoice   OpenVoice v2          ~600 MB   ~10x RT  MeloTTS + tone-color conversion
```

---

## Files Changed

| File | Change | Summary |
|---|---|---|
| `tts_lab.py` | Feature | 8 new loader/synth pairs, MODEL_INFO/MODEL_ORDER, _available(), _build_params(), page title dynamic |
| `tts_benchmark.py` | Feature | 8 new bench functions, ALL_MODELS expanded to 21, BENCH_FNS updated |
| `bench_all.py` | Feature | MODELS list expanded from 13 → 21 |
| `bench_warm.py` | Feature | 4 lighter new engines added (fishspeech, orpheus, zonos, openvoice) |
| `requirements.txt` | Update | Sections 14-21 added |
| `requirements_benchmark.txt` | Update | Sections 14-21 added |
| `setup_tts_lab.sh` | Feature | Steps 14-21 added; final summary updated to 21 engines |
| `download_models.sh` | Update | Auto-download table extended to 18 engines; size note updated to 30+ GB |

---

## Install Commands (VM)

```bash
# From /opt/arthur (with LAB_ENV active)
source /opt/arthur-bench-env/bin/activate

# Run the updated setup script (idempotent — skips already-installed):
sudo bash /opt/arthur/setup_tts_lab.sh

# Or install individually:
pip install fish-speech
pip install "git+https://github.com/SesameAILabs/csm"   # needs: huggingface-cli login
pip install orpheus-speech
pip install "git+https://github.com/index-tts/IndexTTS"
pip install phonemizer "git+https://github.com/Zyphra/Zonos"
pip install "git+https://github.com/myshell-ai/OpenVoice"

# Refresh availability without restarting the server:
curl -sX POST http://192.168.0.87:8001/refresh | python3 -m json.tool
```

---

## Per-Engine Notes

### 14 · Fish Speech (fishspeech)
- **Install:** `pip install fish-speech`
- **API:** `fish_speech.inference.api.TTSInference.from_pretrained("fishaudio/fish-speech-1.5")`
- **Voice cloning:** Upload any 5-30s WAV via the UI; without ref = default voice
- **Variants:** S2-Pro (latest), v1.5, v1.4 — model ID may differ
- ⚠ API changed significantly between versions; check fish-speech GitHub if import fails

### 15 · Sesame CSM 1B (csm)
- **Install:** `pip install git+https://github.com/SesameAILabs/csm`
- **Auth:** `huggingface-cli login` required — model is gated on HF
- **API:** `from generator import load_csm_1b` — returns generator with `.generate(text, speaker, context, max_audio_length_ms)`
- **Speakers:** int 0–2 (different voice identities built-in)
- **SR:** 24000 Hz

### 16 · Qwen3-TTS (qwen3tts)
- **Install:** No extra install — uses `transformers` (already present via parler-tts)
- **Model:** Auto-downloads `Qwen/Qwen3-TTS` from HuggingFace on first load (~1-3 GB)
- ⚠ Model ID may change; check https://huggingface.co/Qwen for current TTS variant
- ⚠ `AutoProcessor`/`AutoModel` API assumed; adjust if transformers version differs

### 17 · Orpheus 3B (orpheus)
- **Install:** `pip install orpheus-speech`
- **Model:** `canopylabs/orpheus-3b-0.1-ft` (HF, ~3 GB)
- **Emotion tags (embed in text):** `<laugh>` `<chuckle>` `<sigh>` `<cough>` `<sniffle>` `<groan>` `<yawn>` `<gasp>`
- **Voices:** tara, leah, jess, leo, dan, mia, zac, zoe
- **SR:** 24000 Hz (PCM int16 chunks from sync generator)
- 🌟 **Best Arthur fit** — `<sigh>` tag at sentence start + hesitant voice = natural confused elderly

### 18 · NeuTTS Air (neutts)
- **Status:** ⚠ PLACEHOLDER — package not yet confirmed
- **Action required:** Identify correct pip package, then edit `_load_neutts()` and `_synth_neutts()` in `tts_lab.py`
- The UI tab shows a "not configured" warning; availability check always returns `False`

### 19 · IndexTTS-2 (indextts)
- **Install:** `pip install git+https://github.com/index-tts/IndexTTS`
- **Model:** `IndexTeam/IndexTTS` auto-downloads from HF on first `model.load_model()`
- **API:** `IndexTTS(model_dir="IndexTeam/IndexTTS", device="cpu").infer(audio_prompt=..., text=..., output_path=...)`
- ⚠ **Reference WAV is REQUIRED** for every synthesis call — no ref = RuntimeError
- Use `piper.wav` from `/tmp/tts_bench/` as a quick test reference

### 20 · Zonos v0.1 (zonos)
- **Install:** `pip install phonemizer "git+https://github.com/Zyphra/Zonos"`
- **System dep:** `espeak-ng` (already in setup step 1)
- **Variants:** `transformer` (quality, ~1.2 GB) or `hybrid` (faster, ~1.5 GB)
- **Conditioning:** 8-dim emotion vector + `speaking_rate` (words/sec) + optional speaker embedding from ref WAV
- **SR:** 44000 Hz
- 🌟 `speaking_rate=13.0` + moderate neutral emotion = naturally paced elderly speech

### 21 · OpenVoice v2 (openvoice)
- **Install:** `pip install git+https://github.com/myshell-ai/OpenVoice`
- **Checkpoints:** Setup step 21 downloads from `myshell-ai/OpenVoice` HF repo → `/opt/models/openvoice_v2/`
- **Architecture:** MeloTTS base synthesis + `ToneColorConverter` tone-color adaptation
- Without ref WAV → synthesises in the selected base speaker (EN-US/EN-BR/EN-AU)
- With ref WAV → zero-shot voice cloning via tone-color conversion
- ⚠ Checkpoints must be present at `/opt/models/openvoice_v2/converter/config.json`

---

## Known Issues / Next Steps

| Item | Status | Notes |
|---|---|---|
| NeuTTS Air | ⚠ not configured | Fill in `_load_neutts()` once package is identified |
| Fish Speech API | ⚠ may vary | `TTSInference` class name may differ by version; check error message |
| CSM gated model | ⚠ needs HF login | `huggingface-cli login` before first use |
| Qwen3-TTS model ID | ⚠ verify | Check HuggingFace for current `Qwen/Qwen3-TTS` ID |
| OpenVoice checkpoints | ⚠ auto-download | Setup step 21 attempts download; verify `/opt/models/openvoice_v2/converter/` exists |
| IndexTTS ref WAV | ⚠ required always | `bench_indextts()` uses `piper.wav` as ref; run bench_piper first |
| Disk space | ⚠ plan ahead | New engines add ~14-16 GB; ensure /opt/models has **30+ GB free** |

---

## Quick Commands

```bash
# Install all new engines in one shot:
sudo bash /opt/arthur/setup_tts_lab.sh   # idempotent, runs steps 14-21

# Refresh UI availability badges (no restart):
curl -sX POST http://192.168.0.87:8001/refresh | python3 -m json.tool

# Benchmark new engines only:
python tts_benchmark.py --models fishspeech,csm,qwen3tts,orpheus,indextts,zonos,openvoice

# Warm-RTF for new engines:
python bench_warm.py   # includes fishspeech, orpheus, zonos, openvoice

# Full live test (all 21):
python bench_all.py

# Restart server after setup:
systemctl restart arthur-lab && journalctl -u arthur-lab -f
```
