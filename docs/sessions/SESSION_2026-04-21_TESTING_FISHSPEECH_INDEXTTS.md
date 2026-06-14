# TTS Lab — Session 2026-04-21: Automated Testing, Fish Speech Fix, IndexTTS-2 Fix
> Branch: `main` · Commits: `279be14` `6203cb0`

---

## What Was Done This Session

Starting state: **19/21 engines available, 12/13 testable engines synthesising (fishspeech broken)**
Ending state: **16/21 engines available (correctly reflects gated HF models), 13/13 testable PASS**

---

## Part 1 — Automated TTS Test Harness

### 1.1 Problem
Every fix required manually opening the web UI, synthesising, copy-pasting the error here. Slow and error-prone.

### 1.2 Solution: `_tts_test.py` + `tts_test.ps1`

**`tools/arthur_server/_tts_test.py`** — runs on the VM, tests all engines via HTTP API:
- Synthesises real audio from each engine with correct params
- Measures duration, RTF, sample rate, load time
- Decodes WAV and validates it has > 50ms of audio
- `--engine ENGINE` to test one engine
- `--unload` to free VRAM between tests (prevents OOM in full suite)
- `--timeout N` to override per-engine timeout
- Correct skip list: `None` params = skip with reason printed

**`tools/arthur_server/tts_test.ps1`** — runs on dev machine:
```powershell
.\tts_test.ps1              # deploy tts_lab.py → restart service → test all
.\tts_test.ps1 -Engine xtts # test one engine only (no unload, stays hot)
.\tts_test.ps1 -NoDeploy    # skip deploy/restart, just run tests
```

Waits for service to be ready (polls `/status` up to 40s) before running tests.
Passes `--unload` automatically for full suite, skips it for single-engine so model stays cached.

---

## Part 2 — Transformers 5.x Compatibility Shims

### 2.1 Problem
`indextts` and `coqui TTS` import removed transformers symbols at **module load time**.
By the time `_load_indextts()` ran its local stubs, the import had already failed.

### 2.2 Root Cause
Our existing local stubs inside `_load_indextts()` were **too late** — they ran after the module-level imports had already failed.

### 2.3 Symbols Added to Global Startup Block

The shims now run at server startup **before any TTS package is imported**:

| Symbol | Module | Why removed |
|--------|--------|-------------|
| `NEED_SETUP_CACHE_CLASSES_MAPPING` | `transformers.generation.configuration_utils` | Merged into GenerationConfig in 5.x |
| `QUANT_BACKEND_CLASSES_MAPPING` | `transformers.generation.configuration_utils` | Same — quantization refactor |
| `ALL_CACHE_IMPLEMENTATIONS` | `transformers.generation.configuration_utils` | Same |
| `_crop_past_key_values` | `transformers.generation.candidate_generator` | KV-cache utility removed |
| `GenerateOutput` | `transformers.generation.utils` | Split into typed subclasses |
| `SequenceSummary` | `transformers.modeling_utils` | Removed in 5.x |
| `QuantizedCacheConfig` | `transformers.cache_utils` | Cache refactor |
| `QuantizedCache` | `transformers.cache_utils` | Cache refactor |
| `QuantoQuantizedCache` | `transformers.cache_utils` | Cache refactor |
| `HQQQuantizedCache` | `transformers.cache_utils` | Cache refactor |
| `OffloadedCache` | `transformers.cache_utils` | Cache refactor |
| `SlidingWindowCache` | `transformers.cache_utils` | Cache refactor |
| `StaticCacheConfig` | `transformers.cache_utils` | Cache refactor |

### 2.4 Global Pin: `transformers<5.0`
```bash
pip install 'transformers>=4.51,<5.0'   # pinned at 4.57.6
```
Prevents any future `pip upgrade` from pulling transformers 5.x again.

---

## Part 3 — Fish Speech 1.5.1 Full Rewrite

### 3.1 Problem Chain (5 bugs in sequence)

The fish-speech install was at `/tmp/fish-speech` — **wiped on every VM reboot**.

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `Project root directory not found` | `/tmp/fish-speech` gone after reboot | Move to `/opt/models/fish-speech` (permanent) |
| 2 | `module 'torchaudio' has no attribute 'list_audio_backends'` | torchaudio 2.x removed this function | Stub it at startup: `_ta.list_audio_backends = lambda: ["soundfile"]` |
| 3 | `'utf-8' codec can't decode byte 0x80` | `load_model()` needs a **directory** not a `.pth` file | Pass `str(model_dir)` not `str(llama_pth)` |
| 4 | `The expanded size of tensor (24) must match existing size (8191)` | KV-cache not initialised before `generate_long` | Call `model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, ...)` after load |
| 5 | `tuple indices must be integers or slices, not tuple` | `decoder.decode()` returns `(audio_tensor, audio_lengths)` tuple | Unpack: `audio_tensor, _ = decoder.decode(...)` |

### 3.2 Additional: `feature_lengths` Required
`FireflyArchitecture.decode(indices, feature_lengths)` requires a lengths tensor:
```python
feature_lengths = torch.tensor([indices.shape[2]], device=DEVICE, dtype=torch.long)
audio_tensor, _ = decoder.decode(indices=indices, feature_lengths=feature_lengths)
audio = audio_tensor[0, 0].cpu().float().numpy()
```

### 3.3 Permanent Install
```bash
sudo git clone --depth=1 --branch v1.5.1 \
    https://github.com/fishaudio/fish-speech /opt/models/fish-speech
echo '/opt/models/fish-speech' | sudo tee \
    /opt/arthur-bench-env/lib/python3.11/site-packages/fish_speech_src.pth
```
No editable install needed — `.pth` file adds repo root to `sys.path`.

### 3.4 Final RTF
```
Fish Speech 1.5.1: dur=325ms  rtf=4.12×  sr=44100Hz  load=14.82s  synth=1340ms
```
First cold load takes ~150s (no queue worker). Warm calls are 1-2s.

---

## Part 4 — CSM & Orpheus: Correctly Marked as Gated

### 4.1 Discovery
Both `sesame/csm-1b` and `canopylabs/orpheus-3b-0.1-ft` have `gated=auto` on HuggingFace.
The HF metadata API returns HTTP 200 (public metadata) but file downloads return HTTP 401 (needs token + access agreement).

The previous session notes incorrectly stated these were public — they're not. The earlier 200 responses were from the API endpoint, not file downloads.

### 4.2 Fix: HTTP Probe in `_check_available()`
```python
# Probe actual file access, not just metadata API
req = urllib.request.Request(
    "https://huggingface.co/sesame/csm-1b/resolve/main/config.json")
with urllib.request.urlopen(req, timeout=5): pass
# → HTTPError 401 → return False, "sesame/csm-1b is gated — run: huggingface-cli login"
```

### 4.3 Result
Both engines show clean "gated" status in UI instead of mysterious load failures.

---

## Part 5 — IndexTTS-2 Fix (in progress at session end)

### 5.1 Error Chain

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `cannot import name 'QUANT_BACKEND_CLASSES_MAPPING'` | transformers 5.x removed it | Added to global stubs dict |
| 2 | `cannot import name 'SequenceSummary'` | transformers 5.x removed it | Added to `transformers.modeling_utils` stubs |
| 3 | `QuantizedCacheConfig` import error | cache_utils stubs ran too late (inside function) | Moved to global startup block |
| 4 | `No such file or directory: '/opt/arthur/checkpoints/config.yaml'` | IndexTTS treats HF repo ID as local path | Use `snapshot_download()`, pass `cfg_path` and `model_dir` explicitly |
| 5 | `UnifiedVoice.__init__() got unexpected keyword 'emo_condition_module'` | Installed package uses old `model.py`; IndexTTS-2 model needs `model_v2.py` | Patched `infer.py`: `from indextts.gpt.model_v2 import UnifiedVoice` |
| 6 | `operator torchvision::nms does not exist` | torchvision CUDA version mismatch after reinstall | `pip install --force-reinstall torchvision==0.26.0+cu128` |
| 7 | `libtorchaudio.so` load failure | torchaudio mismatched after torch upgraded to 2.11 | `pip install torch==2.11.0 torchaudio==2.11.0 --index-url .../cu128` |
| 8 | `numpy.core.multiarray failed to import` | matplotlib compiled against numpy 1.x, broken with 2.x | **IN PROGRESS** — `pip install matplotlib --upgrade` needed |

### 5.2 Status at Session End
**IndexTTS-2 not yet working** — blocked on matplotlib/numpy ABI mismatch.
Fix to apply next session:
```bash
/opt/arthur-bench-env/bin/pip install matplotlib --upgrade --force-reinstall
```

### 5.3 What Does Work After These Fixes
- All import errors cleared
- Model loads correctly from `/opt/models/huggingface/hub/models--IndexTeam--IndexTTS-2/`
- `infer.py` patched to use `model_v2.UnifiedVoice`
- torch/torchvision/torchaudio all at 2.11.0+cu128

---

## Part 6 — Full Test Results (End of Session)

```
TTS Lab — synthesis test   http://localhost:8001
────────────────────────────────────────────────────────────────────────
  ▶ piper          PASS  dur=2867ms  rtf=0.69×  sr=22050Hz
  ▶ kokoro         PASS  dur=4032ms  rtf=3.56×  sr=24000Hz
  ▶ melo           PASS  dur=4420ms  rtf=0.72×  sr=44100Hz
  ▶ chattts        PASS  dur=3468ms  rtf=4.02×  sr=24000Hz
  ▶ outetts        PASS  dur=4119ms  rtf=1.72×  sr=44100Hz
  ▶ bark           PASS  dur=12280ms rtf=4.70×  sr=24000Hz
  ▶ styletts2      PASS  dur=4922ms  rtf=0.42×  sr=24000Hz
  ↷ f5tts          skipped (needs ref WAV)
  ▶ dia            PASS  dur=2786ms  rtf=6.19×  sr=44100Hz
  ▶ xtts           PASS  dur=4673ms  rtf=0.85×  sr=24000Hz
  ↷ cosyvoice      skipped (model not downloaded)
  ↷ parler         skipped (needs transformers<=4.46)
  ▶ chatterbox     PASS  dur=2760ms  rtf=2.06×  sr=24000Hz
  ▶ fishspeech     PASS  dur=325ms   rtf=4.12×  sr=44100Hz  ← FIXED this session
  ↷ csm            skipped (gated HF)
  ↷ qwen3tts       skipped (gated HF)
  ↷ orpheus        skipped (gated HF)
  ↷ neutts         skipped (placeholder)
  ↷ indextts       skipped (needs ref WAV)
  ▶ zonos          PASS  dur=3088ms  rtf=4.13×  sr=44100Hz
  ▶ openvoice      PASS  dur=4365ms  rtf=0.12×  sr=22050Hz
────────────────────────────────────────────────────────────────────────
Results: 13 passed  0 failed  8 skipped   ← 100% of testable engines
```

---

## Part 7 — Environment State

```
torch            2.11.0+cu128
torchvision      0.26.0+cu128
torchaudio       2.11.0+cu128
numpy            2.2.6
transformers     4.57.6  (pinned <5.0)
CUDA             12.8
GPU              RTX 5060 Ti  16 GB GDDR7
```

---

## Part 8 — Error → Root Cause Reference

| Error | Module | Root Cause | Fix |
|-------|--------|-----------|-----|
| `cannot import name 'QUANT_BACKEND_CLASSES_MAPPING'` | `transformers.generation.configuration_utils` | Removed in transformers 5.x | Add to `_GENERATION_MODULE_STUBS` dict |
| `cannot import name 'SequenceSummary'` | `transformers.modeling_utils` | Removed in 5.x | Add to `_GENERATION_MODULE_STUBS` for `modeling_utils` |
| `cannot import name 'QuantizedCacheConfig'` | `transformers.cache_utils` | Removed in 5.x; stubs were inside function (too late) | Move to global startup block |
| `Project root directory not found` | fish_speech | `/tmp/fish-speech` wiped on reboot | Move to `/opt/models/fish-speech` |
| `'utf-8' codec can't decode byte 0x80'` | fish_speech | `load_model` needs directory not `.pth` file | Pass `str(model_dir)` |
| `tensor size (24) must match (8191)` | fish_speech | KV-cache not initialised | Call `model.setup_caches()` after load |
| `tuple indices must be integers` | fish_speech | `decode()` returns `(tensor, lengths)` tuple | Unpack properly |
| `emo_condition_module unexpected kwarg` | indextts | `infer.py` uses old `model.py` not `model_v2.py` | Patch `infer.py` import |
| `torchvision::nms does not exist` | torchvision | CUDA version mismatch | `pip install --force-reinstall torchvision==0.26.0+cu128` |
| `libtorchaudio.so` load failure | torchaudio | torch version jumped, torchaudio mismatched | Reinstall full torch+audio+vision trio at same version |
| `numpy.core.multiarray failed to import` | matplotlib (via indextts) | matplotlib `.so` compiled for numpy 1.x | `pip install matplotlib --upgrade --force-reinstall` |

---

## Part 9 — Commits This Session

| Commit | Description |
|--------|-------------|
| `279be14` | fix(tts-lab): transformers 5.x shims, transformers<5 pin, fishspeech 1.5.1 rewrite |
| `6203cb0` | feat(tts-lab): add automated TTS synthesis test script |

---

## Next Session — Action Items

- [ ] **IndexTTS-2**: `pip install matplotlib --upgrade --force-reinstall` → test with ref WAV
- [ ] **Web UI redesign**: left sidebar with engine list stacked vertically, click → engine page
- [ ] **CosyVoice2**: `git clone FunAudioLLM/CosyVoice /opt/CosyVoice` + download model
- [ ] **Orpheus / CSM**: `huggingface-cli login` + request access at HF → test
- [ ] **Qwen3-TTS**: Request Alibaba access at `huggingface.co/Qwen/Qwen3-TTS`
- [ ] **Fish Speech RTF**: Consider running `launch_thread_safe_queue` worker for faster warm inference
- [ ] **Production TTS pick**: Choose between StyleTTS2 (RTF 0.42) / XTTS (0.85) / Chatterbox (2.06) for `arthur_server.py`
