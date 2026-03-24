# Arthur Server VM — Complete Setup Reference
> Last updated: 2026-03-24  
> VM: `arthur@192.168.0.87`  SSH key: `%USERPROFILE%\.ssh\id_arthur_vm`

---

## 1. Hardware

| Component | Spec |
|---|---|
| **CPU** | Intel Xeon D-1528 @ 1.90 GHz base (Broadwell-DE) |
| **vCores** | 12 (6 physical + HT) |
| **SIMD** | AVX, AVX2, SSE4.1, SSE4.2 — no AVX-512 |
| **RAM** | Hyper-V Dynamic Memory — 4 GB min / **16 GB max** |
| **RAM (current idle)** | ~1.2 GB allocated (Hyper-V deflates when idle) |
| **RAM (under full load)** | ~10–16 GB (Hyper-V balloons as models load) |
| **OS disk** | `/dev/sda1` — 78 GB ext4, ~23 GB used, ~56 GB free |
| **Model disk** | `/dev/sdb1` — 20 GB ext4, mounted at `/opt/models` |
| **Swap** | `/swapfile` — 4 GB file, persists across reboots |
| **OS** | Ubuntu 22.04 LTS (Jammy) |

### ⚠️ Recommended Hyper-V Memory Change (not yet applied)
In Hyper-V Manager → VM Settings → Memory, set **Minimum RAM = 4 GB**.  
Current default (~1 GB) forces XTTS-v2 into swap on cold start because  
Hyper-V hasn't ballooned yet when the 3.2 GB model begins loading.

---

## 2. Disk Layout

```
/dev/sda1  78 GB   /                 OS, Python venvs, swap
/dev/sdb1  20 GB   /opt/models       All model weights (symlinked into /opt/arthur/models)

/opt/models/
├── tts/                            ONNX models (pre-downloaded, no internet needed)
│   ├── en_US-ryan-high.onnx        116 MB   Piper voice
│   ├── en_US-ryan-high.onnx.json    4 KB    Piper voice config
│   ├── kokoro-v1.0.onnx             89 MB   Kokoro int8 quantised
│   └── voices-v1.0.bin              27 MB   Kokoro voice pack
│
├── huggingface/                    HuggingFace model cache (HF_HOME)
│   └── hub/
│       ├── models--Systran--faster-whisper-base.en      141 MB  ← ACTIVE
│       ├── models--Systran--faster-whisper-small.en     464 MB  (cached, not active)
│       ├── models--Systran--faster-whisper-medium.en    1.5 GB  (cached, not active)
│       ├── models--myshell-ai--MeloTTS-English          199 MB  ← ACTIVE
│       ├── models--bert-base-uncased                    421 MB  (MeloTTS dependency)
│       ├── models--bert-base-multilingual-uncased         3 MB  (MeloTTS dependency)
│       ├── models--ResembleAI--chatterbox               3.0 GB  ← ACTIVE
│       └── models--parler-tts--parler-tts-mini-v1       3.3 GB  ← ACTIVE
│
└── /root/.local/share/tts/         Coqui-TTS cache (NOT in HF_HOME — coqui uses own path)
    └── tts_models--multilingual--multi-dataset--xtts_v2/
        ├── model.pth                ~1.8 GB  ← ACTIVE
        ├── speakers_xtts.pth        small    voice embeddings
        └── config.json              config

/opt/arthur/
├── models -> /opt/models/tts        symlink
├── arthur_server.py                production AI bridge
├── tts_lab.py                       TTS evaluation web UI
├── setup_tts_lab.sh                 full re-install script
├── setup_vm.sh                      production server setup
├── bench_all.py                     6-model sequential benchmark
├── bench_warm.py                    isolated cold+warm RTF benchmark
├── download_models.sh               ONNX model downloader
├── run_benchmark.sh                 legacy benchmark runner
├── tts_benchmark.py                 legacy benchmark code
└── requirements*.txt                pip requirements files
```

### Disk totals (current)
| Path | Used |
|---|---|
| `/opt/models/tts/` | 231 MB |
| `/opt/models/huggingface/` | 7.2 GB |
| `/root/.local/share/tts/` | ~1.9 GB |
| `/opt/arthur-bench-env/` | 8.2 GB |
| `/opt/arthur-env/` | ~200 MB |
| **Total model+env** | **~18 GB** |

---

## 3. Python Environments

### `/opt/arthur-env` — Production server (port 8000)
Python 3.11, minimal. Used only by `arthur.service`.

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.135.2 | HTTP server framework |
| `uvicorn` | 0.42.0 | ASGI server |
| `faster-whisper` | 1.2.1 | STT (CTranslate2 backend) |

### `/opt/arthur-bench-env` — TTS Lab (port 8001)
Python 3.11, full AI stack.

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0+cpu | PyTorch CPU build |
| `torchaudio` | 2.6.0 | Audio processing |
| `onnxruntime` | 1.24.4 | ONNX inference (Piper, Kokoro) |
| `transformers` | 4.46.1 | HuggingFace model loader |
| `piper-tts` | 1.4.1 | Piper TTS engine |
| `kokoro-onnx` | 0.5.0 | Kokoro ONNX wrapper |
| `melotts` | 0.1.2 | MeloTTS (installed from GitHub) |
| `parler_tts` | 0.2.3 | Parler-TTS |
| `chatterbox-tts` | 0.1.6 | Chatterbox (Resemble AI) |
| `coqui-tts` | 0.27.5 | XTTS-v2 (community Coqui fork) |
| `soundfile` | 0.13.1 | WAV I/O |
| `numpy` | 2.4.3 | Array ops |
| `psutil` | 7.2.2 | RAM monitoring in /status |
| `fastapi` | 0.135.2 | HTTP server |
| `uvicorn` | 0.42.0 | ASGI server |

---

## 4. Systemd Services

### `arthur.service` — Production AI Bridge (port 8000)
```ini
ExecStart=/opt/arthur-env/bin/uvicorn arthur_server:app --host 0.0.0.0 --port 8000 --workers 1
Environment="GEMINI_API_KEY=<key>"
Environment="OMP_NUM_THREADS=12"
Environment="MKL_NUM_THREADS=12"
Environment="ORT_NUM_THREADS=12"
Environment="CPU_THREADS=12"
Environment="HF_HOME=/opt/models/huggingface"
```
- Starts automatically on boot (`systemctl enable arthur`)
- Loads Whisper `base.en` on startup (141 MB, ~0.8s load)
- Whisper uses `cpu_threads=12` via `CPU_THREADS` env var

### `arthur-lab.service` — TTS Evaluation Lab (port 8001)
```ini
ExecStart=/opt/arthur-bench-env/bin/uvicorn tts_lab:app --host 0.0.0.0 --port 8001 --workers 1
Environment="OMP_NUM_THREADS=12"
Environment="MKL_NUM_THREADS=12"
Environment="OPENBLAS_NUM_THREADS=12"
Environment="ORT_NUM_THREADS=12"
Environment="NUMEXPR_NUM_THREADS=12"
Environment="CPU_THREADS=12"
Environment="HF_HOME=/opt/models/huggingface"
Environment="TRANSFORMERS_CACHE=/opt/models/huggingface/hub"
Environment="COQUI_TOS_AGREED=1"
```
- `tts_lab.py` also calls `torch.set_num_threads(12)` at import (overrides PyTorch's default of `nproc/2`)
- Piper and Kokoro ONNX sessions: `intra_op_num_threads=12` via `SessionOptions`

### Common commands
```bash
sudo systemctl status arthur arthur-lab
sudo systemctl restart arthur arthur-lab
sudo journalctl -u arthur-lab -f            # live logs
sudo journalctl -u arthur -f
```

---

## 5. TTS Models — Installed vs Available

### Model 1: Piper TTS
| | |
|---|---|
| **Installed** | `en_US-ryan-high.onnx` — 116 MB |
| **Runtime RAM** | ~200 MB |
| **Warm RTF** | **0.75×** ✅ real-time on Xeon D |
| **API change fixed** | 1.4.x changed `synthesize()` to yield `AudioChunk` objects — fixed |

**Other available voices** (swap by downloading a different file):
| Voice | Quality | Size | Notes |
|---|---|---|---|
| `en_US-ryan-medium` | Medium | 63 MB | Faster, slightly less natural |
| `en_US-ryan-high` | High | 116 MB | **← installed** |
| `en_US-amy-medium` | Medium | 63 MB | Female |
| `en_US-amy-high` | High | 116 MB | Female |
| `en_GB-jenny_dioco-medium` | Medium | 80 MB | British female |
| `en_GB-alan-medium` | Medium | 63 MB | British male — good Arthur fit |
| All voices at: `https://huggingface.co/rhasspy/piper-voices` | | | |

---

### Model 2: Kokoro-82M
| | |
|---|---|
| **Installed** | `kokoro-v1.0.int8.onnx` — 89 MB (saved as `kokoro-v1.0.onnx`) |
| **Runtime RAM** | ~500 MB |
| **Warm RTF** | **3.85×** ❌ too slow (clock-speed limited) |
| **URL fixed** | Release tag moved from `model-files` → `model-files-v1.0` |

**Other available model sizes** (same `voices-v1.0.bin` works for all):
| File | Size | Notes |
|---|---|---|
| `kokoro-v1.0.int8.onnx` | 89 MB | **← installed** Faster on CPU but still slow on Xeon D |
| `kokoro-v1.0.fp16.onnx` | ~165 MB | Half-precision — needs AVX2 (we have it), may be faster |
| `kokoro-v1.0.onnx` (fp32) | ~330 MB | Full precision — sometimes faster than int8 on AVX2 CPUs |
| `kokoro-v1.0.fp16-gpu.onnx` | ~165 MB | GPU only |
| `kokoro-v1.1-zh.onnx` | N/A | Chinese language model |

**Worth trying:** `kokoro-v1.0.onnx` (fp32, 330 MB). AVX2 FP32 matrix multiply can outperform int8 on Broadwell-DE.

---

### Model 3: MeloTTS
| | |
|---|---|
| **Installed** | `melotts 0.1.2` from PyPI (GitHub: myshell-ai/MeloTTS) |
| **Model cache** | 199 MB at `/opt/models/huggingface/hub/models--myshell-ai--MeloTTS-English` |
| **Runtime RAM** | ~1,200 MB |
| **Warm RTF** | **0.75–1.8×** ⚠️ varies with RAM pressure |
| **Bugs fixed** | `spk2id` is `HParams` not `dict` — converted; `EN_INDIA` key normalised |
| **unidic fixed** | 526 MB dictionary downloaded to venv for MeCab Japanese tokeniser |

**Available accents** (no re-download needed):
| Speaker key | Accent |
|---|---|
| `EN-Default` | Default US |
| `EN-US` | American |
| `EN-BR` | British |
| `EN-AU` | Australian |
| `EN_INDIA` | Indian |

---

### Model 4: XTTS-v2 (Coqui TTS)
| | |
|---|---|
| **Installed** | `coqui-tts 0.27.5` — model `tts_models/multilingual/multi-dataset/xtts_v2` |
| **Model files** | `model.pth` ~1.8 GB + `speakers_xtts.pth` at `/root/.local/share/tts/` |
| **Runtime RAM** | ~3,200 MB |
| **Warm RTF** | **4.7×** ❌ too slow |
| **Load time (from cache)** | ~31s |
| **Fixes applied** | `COQUI_TOS_AGREED=1`; `_patch_transformers_for_coqui()` patches 3 missing symbols: `isin_mps_friendly`, `is_torch_greater_or_equal`, `is_torchcodec_available` |

**Note on cache path:** XTTS is stored in `/root/.local/share/tts/`, NOT in `/opt/models/huggingface/`.  
To move it: `mv /root/.local/share/tts /opt/models/tts_coqui && ln -s /opt/models/tts_coqui /root/.local/share/tts`

**Available speakers** (built-in, no extra download):  
Torcull Diarmuid, Brenda Stern, Ana Florence, Claribel Dervla, Gracie Wise, + 13 others.

---

### Model 5: Parler-TTS
| | |
|---|---|
| **Installed** | `parler_tts 0.2.3` — model `parler-tts/parler-tts-mini-v1` |
| **Model cache** | 3.3 GB at `/opt/models/huggingface/hub/models--parler-tts--parler-tts-mini-v1` |
| **Runtime RAM** | ~1,500 MB |
| **Warm RTF** | **23×** ❌ not suitable for real-time |
| **Load time** | ~33s |
| **Dependency fix** | `transformers 5.x → 4.46.1` (5.x removed `SlidingWindowCache`) |

**Voice is controlled entirely by text description** — no separate model download needed:
```python
"An elderly man with a slow warm slightly confused voice speaks gently."
```

**Other available models:**
| Model | Size | Notes |
|---|---|---|
| `parler-tts/parler-tts-mini-v1` | 3.3 GB | **← installed** |
| `parler-tts/parler-tts-large-v1` | ~5 GB | Higher quality, slower |
| `parler-tts/parler-tts-mini-expresso` | 3.3 GB | More expressive styles |

---

### Model 6: Chatterbox (Resemble AI)
| | |
|---|---|
| **Installed** | `chatterbox-tts 0.1.6` — model `ResembleAI/chatterbox` |
| **Model cache** | 3.0 GB at `/opt/models/huggingface/hub/models--ResembleAI--chatterbox` |
| **Runtime RAM** | ~1,800 MB |
| **Warm RTF** | **13.5×** ❌ not suitable for real-time |
| **Load time** | ~25s |
| **Key parameter** | `exaggeration` (0.0–1.0): higher = more emotional hesitation/confusion |

**No alternative model sizes** — single model weights, single version.

---

### Model 7: CosyVoice2 — NOT INSTALLED
Requires manual `git clone` + model download (~2 GB). Primarily Chinese TTS with zero-shot English.  
Install when needed:
```bash
git clone https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice
cd /opt/CosyVoice && pip install -r requirements.txt
python tools/download_model.py CosyVoice2-0.5B
```

---

## 6. Whisper STT Models

Used by `arthur_server.py` on port 8000.  
All three are cached at `/opt/models/huggingface/hub/` (downloaded during earlier sessions).

| Model | Size | WER | CPU RTF | Status |
|---|---|---|---|---|
| `tiny.en` | 74 MB | ~8% | ~200× RT | not cached |
| `base.en` | 141 MB | ~5% | ~100× RT | **← ACTIVE** |
| `small.en` | 464 MB | ~3% | ~30× RT | cached — try this |
| `medium.en` | 1.5 GB | ~2% | ~8× RT | cached |
| `large-v3` | 3.0 GB | ~1% | ~3× RT | not cached |

**To switch to small.en** (better accuracy, already cached, still real-time on 12 cores):
```python
# arthur_server.py line ~50:
WHISPER_MODEL = "small.en"   # was "base.en"
```
Then: `scp arthur_server.py arthur@192.168.0.87:/opt/arthur/ && sudo systemctl restart arthur`

---

## 7. Thread / CPU Configuration

Applied everywhere so every library uses all 12 cores:

| Layer | How | Set to |
|---|---|---|
| OS environment vars | systemd `Environment=` in both `.service` files | `OMP=MKL=ORT=OPENBLAS=NUMEXPR=CPU_THREADS=12` |
| PyTorch intra-op | `torch.set_num_threads(12)` at `tts_lab.py` import | 12 |
| PyTorch inter-op | `torch.set_num_interop_threads(6)` | 6 |
| OnnxRuntime (Piper) | `SessionOptions.intra_op_num_threads=12` in `_load_piper()` | 12 |
| OnnxRuntime (Kokoro) | `SessionOptions.intra_op_num_threads=12` in `_load_kokoro()` | 12 |
| CTranslate2 (Whisper) | `WhisperModel(..., cpu_threads=CPU_THREADS)` | 12 |

**Important:** `torch.set_num_threads()` must be called **after** `import torch` because PyTorch defaults to `nproc/2 = 6` even when `OMP_NUM_THREADS=12` is set. The explicit call in `tts_lab.py` overrides this.

---

## 8. RTF Benchmark Results (12 cores, warm calls)

Same text for all models: *"Oh my goodness, just a moment dear, let me find my reading glasses. Now you said I owe money to the IRS? Can you give me that case number again, nice and slow?"* (~12s audio)

| Model | Cold RTF | **Warm RTF** | Real-time? | Load time | RAM |
|---|---|---|---|---|---|
| **Piper** | 1.60× | **0.75×** | ✅ YES | 3.3s | 200 MB |
| **MeloTTS** | 1.11× | **0.75–1.8×** | ✅ YES* | 4.3s | 1,200 MB |
| Kokoro | 5.05× | 3.85× | ❌ | 2.1s | 500 MB |
| XTTS-v2 | 5.0× | 4.7× | ❌ | 31s | 3,200 MB |
| Chatterbox | 11.8× | 13.5× | ❌ | 25s | 1,800 MB |
| Parler-TTS | 27× | 23× | ❌ | 33s | 1,500 MB |

*MeloTTS warm RTF degrades to ~1.8× when other heavy models are resident (RAM pressure).  
RTF < 1.0 = synthesises faster than audio plays. RTF > 1.0 = too slow for live calls.

**Root cause of slow RTF:** Xeon D-1528 base clock 1.90 GHz + no AVX-512.  
All neural models are compute-bound on single-utterance inference. More cores help  
torch/MKL operations but not the sequential ONNX forward pass.

---

## 9. Configurations That Can Still Be Applied

### 🔴 High impact — do these first

| Config | What | Command / Change |
|---|---|---|
| **Hyper-V min RAM = 4 GB** | Stops XTTS spilling 686 MB into swap on cold start | Hyper-V Manager → VM Settings → Memory → Minimum: 4096 MB |
| **Switch Whisper to small.en** | Better STT accuracy, already cached, still real-time | `WHISPER_MODEL = "small.en"` in `arthur_server.py` |
| **Move XTTS cache to `/opt/models`** | All model data on the data disk instead of OS disk | `mv /root/.local/share/tts /opt/models/tts_coqui && ln -s /opt/models/tts_coqui /root/.local/share/tts` |

### 🟡 Performance — worth trying

| Config | Expected gain | How |
|---|---|---|
| **Kokoro fp32 model** | Possibly faster than int8 on AVX2 | Replace `kokoro-v1.0.onnx` with `model-files-v1.0/kokoro-v1.0.onnx` (330 MB) |
| **OnnxRuntime + OpenVINO EP** | 1.5–2× for Piper and Kokoro on Intel CPU | `pip install onnxruntime-openvino` and pass `providers=["OpenVINOExecutionProvider"]` |
| **Piper British voice (alan-medium)** | Better Arthur accent fit | Download `en_GB-alan-medium.onnx` from piper-voices HuggingFace repo |

### 🟢 Quality — evaluation use

| Config | What | Notes |
|---|---|---|
| **Parler prompt tuning** | Adjust voice character via text description | Longer, more specific descriptions improve consistency |
| **Chatterbox exaggeration** | 0.5–0.7 sounds most confused-elderly | 0.0 = flat, 1.0 = theatrical |
| **XTTS speaker** | Try all 16 built-in speakers | `Torcull Diarmuid` sounds most elderly/male |
| **Piper voice pack swap** | Try `en_GB-alan-medium` for British Arthur | 63 MB download |

### 🔵 Future / GPU

| Config | Impact | Cost |
|---|---|---|
| **GPU passthrough (GTX 1060)** | All models become real-time (XTTS: 4.7× → 0.05×) | ~$80 used GPU |
| **Upgrade VM max CPU clock** | Kokoro RTF scales linearly with GHz | Host CPU upgrade |
| **Use Arthur server TTS locally** | Replace cloud Gemini TTS with Piper (0.75× RTF) | Change `arthur_server.py` TTS provider |

---

## 10. Bugs Discovered & Fixed During This Session

| # | Bug | Fix |
|---|---|---|
| 1 | `deploy.ps1` pointed to Hyper-V host IP `.153` instead of VM `.87` | Changed default `$VM` to `192.168.0.87` |
| 2 | `deploy.ps1` used password auth — SSH key existed | Replaced `$Pass`/`plink` with `-i $Key` / `scp -i $Key` |
| 3 | All `.sh` files had Windows CRLF line endings → `\r: command not found` | PowerShell CRLF→LF conversion at deploy time |
| 4 | All `.sh` files had UTF-8 BOM → `#!/usr/bin/env: No such file or directory` | `WriteAllText` with `new UTF8Encoding(false)` (no-BOM) |
| 5 | `setup_tts_lab.sh` had `set -e` — `pip install melo-tts` failed (wrong name) and killed the whole script | Removed `-e`; corrected package names |
| 6 | `melo-tts` does not exist on PyPI | Installed via GitHub URL; actual PyPI name is `melotts` |
| 7 | `TTS` package abandoned → use `coqui-tts` | Changed install to `coqui-tts 0.27.5` |
| 8 | Kokoro download URL returned 404 | Release tag changed `model-files` → `model-files-v1.0`; switched `wget` → `curl -L` |
| 9 | Kokoro files were 0 bytes (no redirect follow) | Used `curl -L` instead of `wget -q` |
| 10 | `unidic` dictionary not downloaded → MeloTTS import fails with MeCab error | `sudo python -m unidic download` (526 MB) |
| 11 | Piper `synthesize()` API changed in 1.4.x | Removed `sentence_silence` kwarg; rewrote to iterate `AudioChunk` objects |
| 12 | `inst.config.audio.sample_rate` doesn't exist in `PiperConfig` 1.4.x | Changed to `inst.config.sample_rate` (flat attribute) |
| 13 | MeloTTS `spk2id` is `HParams` object, not `dict` — `.get()` fails | Wrapped in `dict(...)` before lookup |
| 14 | `EN-INDIA` key doesn't exist — it's `EN_INDIA` | Fixed normalisation in `_synth_melo()` |
| 15 | `transformers 5.3.0` breaks `parler_tts 0.2.3` (`SlidingWindowCache` missing) | Downgraded to `transformers 4.46.1` |
| 16 | `coqui-tts` fails with `isin_mps_friendly` / `is_torch_greater_or_equal` / `is_torchcodec_available` missing | `_patch_transformers_for_coqui()` monkey-patches all 3 at load time |
| 17 | `COQUI_TOS_AGREED` not set → XTTS-v2 prompts for license interactively → `EOFError: EOF when reading a line` in systemd | Added `os.environ["COQUI_TOS_AGREED"] = "1"` in `_load_xtts()` and service unit |
| 18 | PyTorch defaults to `nproc/2 = 6` threads even with `OMP_NUM_THREADS=12` set | Explicit `torch.set_num_threads(12)` after `import torch` in `tts_lab.py` |
| 19 | `WhisperModel` on `arthur_server.py` used 0 explicit threads | Added `cpu_threads=CPU_THREADS` and `num_workers=1` |
| 20 | `_available()` used `importlib.find_spec()` — passes even when imports fail (dependency conflicts) | Replaced with live `exec(import_stmt)` test, results cached in `_import_cache` |
| 21 | `/unload/<model>` endpoint called in `bench_warm.py` but doesn't exist | Noted as 404; eviction handled automatically by `_evict_heavy()` |

---

## 11. File Reference

| File | Location | Purpose |
|---|---|---|
| `arthur_server.py` | `/opt/arthur/` | Production: Whisper STT + Gemini LLM + Gemini TTS bridge |
| `tts_lab.py` | `/opt/arthur/` | TTS evaluation web UI (port 8001) |
| `setup_tts_lab.sh` | `/opt/arthur/` | Full TTS lab install (idempotent, re-runnable) |
| `setup_vm.sh` | `/opt/arthur/` | Production server install (Whisper + arthur.service) |
| `download_models.sh` | `/opt/arthur/` | Download Piper + Kokoro ONNX files only |
| `bench_all.py` | `/opt/arthur/` + local | Sequential 6-model benchmark (cold start) |
| `bench_warm.py` | `/opt/arthur/` + local | Isolated cold+warm RTF per model |
| `deploy.ps1` | `tools/arthur_server/` | Deploy all files from Windows to VM via SSH key |
| `requirements.txt` | `tools/arthur_server/` | Production venv requirements |
| `requirements_benchmark.txt` | `tools/arthur_server/` | TTS lab venv requirements |

---

## 12. Quick-Reference Commands

```bash
# SSH into VM
ssh -i %USERPROFILE%\.ssh\id_arthur_vm arthur@192.168.0.87

# Deploy all files from Windows
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy.ps1

# Service management
sudo systemctl status arthur arthur-lab
sudo systemctl restart arthur arthur-lab
sudo journalctl -u arthur-lab -f

# Run benchmark
/opt/arthur-bench-env/bin/python3 /opt/arthur/bench_warm.py

# RAM snapshot
free -h && df -h /opt/models

# Check what models are loaded
curl -s http://localhost:8001/status | python3 -m json.tool | grep -E '"label"|"status"|"available"'

# Synthesise test (Piper)
curl -s -X POST http://localhost:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello there, this is Arthur.","params":{"voice":"en_US-ryan-high"}}' | \
  python3 -c "import sys,json,base64; open('/tmp/t.wav','wb').write(base64.b64decode(json.load(sys.stdin)['audio_b64']))"

# Extend OS partition after disk resize
sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1
```

---

## 13. Architecture Reminder

```
Android (Pixel 5)
    │
    │ scam call detected
    ▼
BaiterInCallService
    │
    ├─── STT: Android SpeechRecognizer (on-device)
    │
    └─── AI Bridge Mode (optional):
              │
              ▼  HTTP / WebSocket
         arthur_server.py  (port 8000)
              │
              ├── STT:  faster-whisper base.en  (local, 12 threads)
              ├── LLM:  Gemini 2.0 Flash        (cloud API)
              └── TTS:  Gemini 2.5 Flash TTS    (cloud API, voice "Gacrux")
                        ← swap this for Piper when going fully local

TTS Lab  (port 8001)  ← for voice selection only, not used in production
    ├── Piper TTS      RTF 0.75  ✅ real-time
    ├── MeloTTS        RTF 0.75  ✅ real-time (stable RAM conditions)
    ├── Kokoro-82M     RTF 3.85  ❌ evaluation only
    ├── XTTS-v2        RTF 4.70  ❌ evaluation only
    ├── Chatterbox     RTF 13.5  ❌ evaluation only
    ├── Parler-TTS     RTF 23×   ❌ evaluation only
    └── CosyVoice2     N/A       ❌ not installed
```
