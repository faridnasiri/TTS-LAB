# Arthur Server VM — Complete Setup Reference  v2
> Last updated: 2026-03-25 (expanded from v1 — 11 TTS engines, all bugs, live disk/RAM numbers)
> VM: `arthur@192.168.0.87`  SSH key: `%USERPROFILE%\.ssh\id_arthur_vm`

---

## 1. Hardware (live)

| Component | Spec |
|---|---|
| **CPU** | Intel Xeon D-1528 @ 1.90 GHz base (Broadwell-DE) |
| **vCores** | 12 (6 physical × 2 HT) |
| **SIMD** | AVX, AVX2, SSE4.1, SSE4.2 — **no AVX-512** |
| **RAM** | Hyper-V Dynamic Memory — **4 GB min / 32 GB max** _(min raised from 1 GB this session)_ |
| **RAM idle** | ~800 MB allocated |
| **RAM full load** | ~9–11 GB (Bark + Dia + StyleTTS2 all resident) |
| **OS disk** | `/dev/sdb1` — 78 GB ext4, ~25 GB used, **53 GB free** |
| **Model disk** | `/dev/sda1` — **40 GB** ext4 _(expanded from 20 GB this session)_, 23 GB used, **15 GB free** |
| **Swap** | `/swapfile` — 4 GB, persists across reboots |
| **OS** | Ubuntu 22.04 LTS (Jammy) |

> **Note — disks are swapped vs. the old reference doc.**
> After Hyper-V disk expansion the controller mapping changed:
> `/dev/sda1` = model data disk (40 GB, `/opt/models`)
> `/dev/sdb1` = OS disk (90 GB, `/`)

### Changes made to hardware this session
| Change | Before | After |
|---|---|---|
| Hyper-V min RAM | ~1 GB | **4 GB** |
| Hyper-V max RAM | 16 GB | **32 GB** |
| Model disk size | 20 GB | **40 GB** (growpart + resize2fs live, no downtime) |

---

## 2. Disk Layout (live — 2026-03-25)

```
/dev/sda1  40 GB   /opt/models      model data disk  (23 GB used / 15 GB free)
/dev/sdb1  78 GB   /                OS disk          (25 GB used / 53 GB free)

/opt/models/
│
├── tts/                            ONNX + Piper voice files   (581 MB)
│   ├── en_US-ryan-high.onnx        116 MB  ← Piper default voice
│   ├── en_US-ryan-high.onnx.json     4 KB
│   ├── en_GB-alan-medium.onnx       63 MB  ← British male (Arthur fit)
│   ├── en_GB-alan-medium.onnx.json   4 KB
│   ├── en_GB-cori-medium.onnx       80 MB  ← British female
│   ├── en_GB-cori-medium.onnx.json   4 KB
│   ├── en_US-danny-low.onnx         25 MB  ← US male low quality
│   ├── en_US-joe-medium.onnx        63 MB  ← US male medium
│   ├── en_US-lessac-high.onnx      116 MB  ← US female high
│   ├── kokoro-v1.0.onnx             89 MB  ← Kokoro int8 quantised
│   └── voices-v1.0.bin              27 MB  ← Kokoro voice pack (54 voices)
│
├── cache/                          XDG_CACHE_HOME redirect
│   └── suno/                       Bark model weights        (4.4 GB)
│       └── bark_v0/
│           ├── text_2.pt            (small text model)
│           ├── coarse_2.pt          (small coarse model)
│           ├── fine_2.pt            (small fine model)
│           └── encodec.th           (audio codec)
│
├── tts_coqui/                      XTTS-v2 cache             (1.8 GB)
│   └── tts_models--multilingual--multi-dataset--xtts_v2/
│       ├── model.pth                ~1.8 GB ← ACTIVE
│       ├── speakers_xtts.pth        small
│       └── config.json
│   (symlinked from /root/.local/share/tts → moved here this session)
│
└── huggingface/                    HF_HOME                   (17 GB total)
    └── hub/
        ├── models--Systran--faster-whisper-base.en    141 MB  ← ACTIVE (production)
        ├── models--Systran--faster-whisper-small.en   464 MB  (cached, not active)
        ├── models--Systran--faster-whisper-medium.en  1.5 GB  (cached, not active)
        ├── models--myshell-ai--MeloTTS-English        199 MB  ← ACTIVE
        ├── models--bert-base-uncased                  421 MB  (MeloTTS BERT dep)
        ├── models--bert-base-multilingual-uncased       3 MB  (MeloTTS dep)
        ├── models--bert-base-multilingual-cased         3 MB  (dep)
        ├── models--charactr--vocos-mel-24khz           52 MB  (F5-TTS vocoder dep)
        ├── models--SWivid--F5-TTS                     1.3 GB  ← ACTIVE
        ├── models--nari-labs--Dia-1.6B-0626           6.1 GB  ← ACTIVE (v2 config)
        ├── models--nari-labs--Dia-1.6B                 28 KB  (stub — config mismatch)
        ├── models--ResembleAI--chatterbox             3.0 GB  ← ACTIVE
        ├── models--parler-tts--parler-tts-mini-v1     3.3 GB  ← ACTIVE
        └── (bert multilingual deps, MeloTTS lang BERTs)

/opt/arthur/
├── models -> /opt/models/tts         symlink
├── arthur_server.py                  production AI bridge
├── tts_lab.py                        TTS evaluation web UI   ← UPDATED (11 engines)
├── setup_tts_lab.sh                  full re-install script  ← UPDATED
├── setup_vm.sh                       production server setup
├── bench_all.py                      cold-start sequential benchmark
├── bench_warm.py                     isolated warm RTF benchmark
├── download_models.sh                Piper + Kokoro downloader
├── run_benchmark.sh                  legacy benchmark runner
├── tts_benchmark.py                  legacy benchmark code
└── requirements*.txt
```

### Live disk totals
| Path | Used | Notes |
|---|---|---|
| `/opt/models/tts/` | 581 MB | 6 Piper voices + Kokoro ONNX + voices |
| `/opt/models/cache/suno/` | 4.4 GB | Bark-small weights |
| `/opt/models/tts_coqui/` | 1.8 GB | XTTS-v2 (moved from OS disk) |
| `/opt/models/huggingface/` | 17 GB | All HF-cached models |
| `/opt/arthur-bench-env/` | 8.8 GB | TTS lab Python venv |
| `/opt/arthur-env/` | 544 MB | Production Python venv |
| **Total** | **~33 GB** | Fits on 40 GB model disk |

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
Python 3.11, full AI stack. **All versions confirmed live this session.**

| Package | Version | Purpose |
|---|---|---|
| `torch` | **2.6.0+cpu** | PyTorch CPU build |
| `torchaudio` | 2.6.0 | Audio I/O |
| `transformers` | **4.46.1** _(pinned — 5.x breaks parler/coqui)_ | HF model loader |
| `onnxruntime` | 1.24.4 | ONNX inference (Piper, Kokoro) |
| `protobuf` | **6.33.6** _(upgraded 3.19→6.x this session)_ | Serialisation |
| `numpy` | 2.4.3 | Array ops |
| `soundfile` | 0.13.1 | WAV I/O |
| `psutil` | 7.2.2 | RAM monitoring |
| `fastapi` | 0.135.2 | HTTP server |
| `uvicorn` | 0.42.0 | ASGI server |
| `piper-tts` | 1.4.1 | Piper TTS engine |
| `kokoro-onnx` | 0.5.0 | Kokoro ONNX wrapper |
| `melotts` | 0.1.2 | MeloTTS (from GitHub) |
| `bark` | **0.1.5** _(new)_ | Bark emotion-token TTS |
| `styletts2` | **0.1.6** _(new)_ | StyleTTS 2 |
| `f5-tts` | **1.1.18** _(new)_ | F5-TTS zero-shot cloning |
| `dia` | git main _(new)_ | Dia-1.6B dialogue TTS |
| `parler_tts` | 0.2.3 | Parler-TTS |
| `chatterbox-tts` | 0.1.6 | Chatterbox |
| `coqui-tts` | 0.27.5 | XTTS-v2 |

---

## 4. Systemd Services

### `arthur.service` — Production AI Bridge (port 8000)
```ini
[Service]
ExecStart=/opt/arthur-env/bin/uvicorn arthur_server:app --host 0.0.0.0 --port 8000 --workers 1
Environment="GEMINI_API_KEY=<redacted>"
Environment="OMP_NUM_THREADS=12"
Environment="MKL_NUM_THREADS=12"
Environment="ORT_NUM_THREADS=12"
Environment="CPU_THREADS=12"
Environment="HF_HOME=/opt/models/huggingface"
```

### `arthur-lab.service` — TTS Evaluation Lab (port 8001)
```ini
[Service]
ExecStart=/opt/arthur-bench-env/bin/uvicorn tts_lab:app --host 0.0.0.0 --port 8001 --workers 1
Environment="OMP_NUM_THREADS=12"
Environment="MKL_NUM_THREADS=12"
Environment="OPENBLAS_NUM_THREADS=12"
Environment="ORT_NUM_THREADS=12"
Environment="NUMEXPR_NUM_THREADS=12"
Environment="CPU_THREADS=12"
Environment="HF_HOME=/opt/models/huggingface"
Environment="TRANSFORMERS_CACHE=/opt/models/huggingface/hub"
Environment="XDG_CACHE_HOME=/opt/models/cache"        # ← NEW: redirects Bark cache to data disk
Environment="SUNO_USE_SMALL_MODELS=True"               # ← NEW: Bark small (1.3 GB vs 5.5 GB)
Environment="COQUI_TOS_AGREED=1"
```

**Why each var is needed:**
| Var | Reason |
|---|---|
| `OMP/MKL/ORT/OPENBLAS/NUMEXPR_NUM_THREADS=12` | Each library defaults to its own thread count; explicit set uses all 12 vCores |
| `CPU_THREADS=12` | Read by `arthur_server.py` → passed to `WhisperModel(cpu_threads=...)` |
| `HF_HOME` | All HuggingFace downloads go to model data disk, not OS disk |
| `XDG_CACHE_HOME` | Bark uses XDG cache path; redirected to model disk instead of `/root/.cache` |
| `SUNO_USE_SMALL_MODELS=True` | Bark downloads ~1.3 GB small models vs ~5.5 GB full |
| `COQUI_TOS_AGREED=1` | Without this XTTS-v2 prompts for license interactively → `EOFError` in systemd |

### Common service commands
```bash
sudo systemctl status arthur arthur-lab
sudo systemctl restart arthur arthur-lab
sudo journalctl -u arthur-lab -f     # live logs
sudo journalctl -u arthur -f
```

---

## 5. TTS Engine Comparison — All 11 Models

### 5.1 Quick-glance table

| # | Engine | Installed | Warm RTF | Real-time | RAM | Load time | Arthur fit | Unique capability |
|---|---|---|---|---|---|---|---|---|
| 1 | **Piper TTS** | ✅ | **0.75×** | ✅ | 200 MB | 3s | ⭐⭐ | Fastest. 6 voices. ONNX, no PyTorch |
| 2 | **MeloTTS** | ✅ | **0.75–1.8×** | ✅* | 1.2 GB | 4s | ⭐⭐⭐ | 5 accents. British accent available |
| 3 | **StyleTTS 2** | ✅ | **~3×** | ❌ | ~1.5 GB | 14s | ⭐⭐⭐⭐ | Fastest neural. Style from reference WAV |
| 4 | **Kokoro-82M** | ✅ | **3.85×** | ❌ | 500 MB | 2s | ⭐⭐⭐⭐⭐ | 54 voices, 9 languages |
| 5 | **XTTS-v2** | ✅ | **4.7×** | ❌ | 3.2 GB | 31s | ⭐⭐⭐⭐⭐ | 58 speakers, 17 languages, voice clone |
| 6 | **F5-TTS** | ✅ | **~4–6×** | ❌ | 2 GB | 15s | ⭐⭐⭐⭐ | Best zero-shot voice cloning |
| 7 | **Bark** | ✅ | **~17×** | ❌ | 1.5 GB | 66s | ⭐⭐⭐⭐⭐ | `[laughs]` `[sighs]` `[clears throat]` in text |
| 8 | **Chatterbox** | ✅ | **13.5×** | ❌ | 1.8 GB | 25s | ⭐⭐⭐⭐⭐ | Exaggeration slider (0–1). Voice clone |
| 9 | **Parler-TTS** | ✅ | **23×** | ❌ | 1.5 GB | 33s | ⭐⭐⭐⭐ | Text description controls voice entirely |
| 10 | **Dia-1.6B** | ✅ | **~55×** | ❌ | 3+ GB | 22s | ⭐⭐⭐⭐⭐ | Dialogue-native. `[S1]`/`[S2]` + emotion tags |
| 11 | **CosyVoice2** | ❌ | N/A | ❌ | ~2.5 GB | N/A | ⭐⭐⭐ | Chinese-first. Zero-shot English |

*MeloTTS degrades to ~1.8× RTF when heavy models are resident in RAM.

**Arthur fit** = how well the voice/controls match a confused elderly man persona.

### 5.2 Model detail — sorted by usefulness for Arthur

---

#### Model 1 — Piper TTS ✅ INSTALLED
| | |
|---|---|
| **Package** | `piper-tts 1.4.1` |
| **Engine type** | ONNX (no PyTorch at all) |
| **Voices installed** | 6 (see table below) |
| **Runtime RAM** | ~200 MB |
| **Cold RTF** | 1.60× |
| **Warm RTF** | **0.75×** ✅ real-time |
| **Load time** | ~3s |
| **Sample rate** | 22 050 Hz |

**Voices on disk (`/opt/models/tts/`):**
| File | Size | Lang | Gender | Notes |
|---|---|---|---|---|
| `en_US-ryan-high.onnx` | 116 MB | US | Male | ← default |
| `en_US-lessac-high.onnx` | 116 MB | US | Female | High quality |
| `en_US-joe-medium.onnx` | 63 MB | US | Male | Slightly deeper |
| `en_US-danny-low.onnx` | 25 MB | US | Male | Fastest, lowest quality |
| `en_GB-alan-medium.onnx` | 63 MB | **British** | Male | **Best Arthur fit** |
| `en_GB-cori-medium.onnx` | 80 MB | British | Female | |

**More voices available** at `https://huggingface.co/rhasspy/piper-voices` — download `.onnx` + `.onnx.json` pair.

**API bugs fixed (v1.4.x):**
- `synthesize()` now yields `AudioChunk` objects, not raw bytes
- `SynthesisConfig` replaces removed `sentence_silence` kwarg
- `config.sample_rate` (flat) replaced removed `config.audio.sample_rate`

---

#### Model 2 — MeloTTS ✅ INSTALLED
| | |
|---|---|
| **Package** | `melotts 0.1.2` (GitHub: myshell-ai/MeloTTS) |
| **Model cache** | 199 MB at `huggingface/hub/models--myshell-ai--MeloTTS-English` |
| **Runtime RAM** | ~1 200 MB |
| **Cold RTF** | 1.11× |
| **Warm RTF** | **0.75–1.8×** ✅* real-time |
| **Load time** | ~4s |
| **Sample rate** | 44 100 Hz |

**Accents:**
| Key | Accent |
|---|---|
| `EN-Default` | American default |
| `EN-US` | American |
| `EN-BR` | **British** ← Arthur pick |
| `EN-AU` | Australian |
| `EN_INDIA` | Indian |

**Bugs fixed:**
- `spk2id` attribute is `HParams` object — wrapped in `dict(...)` before `.get()`
- Indian English key is `EN_INDIA` (underscore), not `EN-INDIA`

---

#### Model 3 — StyleTTS 2 ✅ INSTALLED (new this session)
| | |
|---|---|
| **Package** | `styletts2 0.1.6` |
| **Model cache** | ~700 MB in `/opt/models/cache/` |
| **Runtime RAM** | ~1 500 MB |
| **Cold RTF** | ~3× |
| **Warm RTF** | **~3×** — fastest neural model |
| **Load time** | ~14s |
| **Sample rate** | 24 000 Hz |

**Parameters exposed in UI:**
| Param | Range | Default | Effect |
|---|---|---|---|
| `alpha` | 0–1 | 0.3 | 0 = copy reference style exactly |
| `beta` | 0–1 | 0.7 | 0 = copy reference prosody exactly |
| `diffusion_steps` | 3–15 | 5 | More = better quality + slower |
| `embedding_scale` | 0.5–3 | 1.0 | Voice embedding strength |
| Reference WAV | upload | optional | Sets voice timbre |

**API bug fixed:** `ref_audio=` parameter was renamed to `target_voice_path=` in current package.

**PyTorch 2.6 bug fixed:** Legacy checkpoints contain `builtins.getattr` globals blocked by `weights_only=True`. Fixed by patching `torch.load` to `weights_only=False` during model load only, then restoring.

---

#### Model 4 — Kokoro-82M ✅ INSTALLED
| | |
|---|---|
| **Package** | `kokoro-onnx 0.5.0` |
| **Model file** | `kokoro-v1.0.onnx` (int8, 89 MB) + `voices-v1.0.bin` (27 MB) |
| **Runtime RAM** | ~500 MB |
| **Cold RTF** | 5.05× |
| **Warm RTF** | **3.85×** ❌ too slow |
| **Load time** | ~2s |
| **Sample rate** | 24 000 Hz |

**54 voices across 9 languages:**
| Prefix | Lang | Example voices |
|---|---|---|
| `bm_` | British Male | `bm_lewis` ← **Arthur pick**, `bm_daniel`, `bm_george` |
| `bf_` | British Female | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily` |
| `am_` | American Male | `am_adam`, `am_echo`, `am_michael`, `am_puck` |
| `af_` | American Female | `af_bella`, `af_heart`, `af_nova`, `af_sarah` |
| `ef_/em_` | Spanish | `ef_dora`, `em_alex` |
| `ff_` | French | `ff_siwis` |
| `hf_/hm_` | Hindi | `hf_alpha`, `hm_omega` |
| `jf_/jm_` | Japanese | `jf_alpha`, `jm_kumo` |
| `zf_/zm_` | Chinese | `zf_xiaobei`, `zm_yunxi` |

**Worth trying:** `kokoro-v1.0.onnx` (fp32 330 MB). AVX2 FP32 matrix multiply can outperform int8 on Broadwell-DE.

---

#### Model 5 — XTTS-v2 ✅ INSTALLED
| | |
|---|---|
| **Package** | `coqui-tts 0.27.5` |
| **Model files** | `model.pth` ~1.8 GB at `/opt/models/tts_coqui/` |
| **Runtime RAM** | ~3 200 MB |
| **Cold RTF** | 5.0× |
| **Warm RTF** | **4.7×** ❌ too slow |
| **Load time** | ~31s |
| **Sample rate** | 24 000 Hz |

**58 built-in speakers (selection):**
`Torcull Diarmuid` (Arthur pick), `Brenda Stern`, `Craig Gutsy`, `Damien Black`,
`Dionisio Schuyler`, `Gitta Nikolina`, `Viktor Menelaos`, `Wulf Carlevaro` + 50 more.

**17 languages:** English, French, German, Spanish, Italian, Portuguese, Polish, Czech,
Arabic, Hindi, Hungarian, Japanese, Korean, Dutch, Russian, Turkish, Chinese.

**Supports voice cloning** — pass a reference WAV via `/upload` then reference audio param.

**Bugs fixed:**
- `COQUI_TOS_AGREED=1` required (interactive license prompt kills systemd service)
- `_patch_transformers_for_coqui()` monkey-patches 3 symbols removed from transformers 4.46: `isin_mps_friendly`, `is_torch_greater_or_equal`, `is_torchcodec_available`

**Cache moved** from `/root/.local/share/tts` → `/opt/models/tts_coqui` (symlinked) to keep all model data on the data disk.

---

#### Model 6 — F5-TTS ✅ INSTALLED (new this session)
| | |
|---|---|
| **Package** | `f5-tts 1.1.18` |
| **Model cache** | 1.3 GB at `huggingface/hub/models--SWivid--F5-TTS` |
| **Runtime RAM** | ~2 000 MB |
| **Warm RTF** | **~4–6×** (estimate — not benchmarked cold) |
| **Load time** | ~15s |
| **Sample rate** | 24 000 Hz |

**Parameters:**
| Param | Range | Default | Effect |
|---|---|---|---|
| Reference WAV | upload | **REQUIRED** | Target voice to clone |
| Reference text | string | "" | Exact words spoken in reference clip |
| `speed` | 0.5–2.0 | 1.0 | Speaking rate |
| `nfe_step` | 8–64 | 32 | Diffusion steps (more = better + slower) |

> **F5-TTS requires a reference WAV** — without one synthesis fails. Upload a 5–15s clip of any voice you want to clone. The model then synthesises the new text in that voice.

---

#### Model 7 — Bark ✅ INSTALLED (new this session)
| | |
|---|---|
| **Package** | `bark 0.1.5` |
| **Model cache** | 4.4 GB at `/opt/models/cache/suno/` (small models) |
| **Runtime RAM** | ~1 500 MB |
| **Cold RTF** | ~17× (measured) — load 66s first time |
| **Warm RTF** | **~17×** ❌ not real-time, but uniquely expressive |
| **Load time** | ~66s cold (model download + load); ~25s warm |
| **Sample rate** | 24 000 Hz |

**Unique feature: Non-verbal audio tokens embedded directly in text:**
| Token | Effect |
|---|---|
| `[laughs]` | Inserts a laugh |
| `[sighs]` | Inserts a sigh |
| `[clears throat]` | Throat clearing |
| `[hesitantly]` | Hesitant tone |
| `[gasps]` | Gasp |
| `[long pause]` | Extended pause |
| `[nervously]` | Nervous intonation |
| `[quietly]` | Quieter delivery |
| `[MAN]` / `[WOMAN]` | Force speaker gender |

**Arthur example:**
```
"Hello? [sighs] Oh my goodness, who is this? [clears throat] Just a moment dear, let me find my glasses."
```

**Voice presets (10 English, selectable in UI):**
| Preset | Character |
|---|---|
| `v2/en_speaker_6` | Male, measured ← **Arthur pick** |
| `v2/en_speaker_7` | Male, elderly |
| `v2/en_speaker_9` | Male, older |
| `v2/en_speaker_0` | Male, deep |
| `v2/en_speaker_3` | Male, gravelly |
| `v2/en_speaker_1` | Male, warm |
| `v2/en_speaker_2` | Female |
| `v2/en_speaker_5` | Female, soft |
| `v2/en_speaker_4` | Male, neutral |
| `v2/en_speaker_8` | Female |

**PyTorch 2.6 bug fixed:** Legacy Bark checkpoints contain `numpy.core.multiarray.scalar`
globals blocked by `weights_only=True`. In NumPy 2.x `np.core` was removed, so
`add_safe_globals` cannot fix it. Solution: patch `torch.load` → `weights_only=False`
during `preload_models()` only, then restore the original `torch.load`.

---

#### Model 8 — Chatterbox ✅ INSTALLED
| | |
|---|---|
| **Package** | `chatterbox-tts 0.1.6` |
| **Model cache** | 3.0 GB at `huggingface/hub/models--ResembleAI--chatterbox` |
| **Runtime RAM** | ~1 800 MB |
| **Cold RTF** | 11.8× |
| **Warm RTF** | **13.5×** ❌ |
| **Load time** | ~25s |
| **Sample rate** | 24 000 Hz |

**Parameters:**
| Param | Range | Default | Sweet spot for Arthur |
|---|---|---|---|
| `exaggeration` | 0.0–1.0 | 0.65 | **0.5–0.7** — confused elderly |
| `cfg_weight` | 0.1–1.0 | 0.5 | Lower = more natural |
| `seed` | 0–9999 | 0 (random) | Fix for reproducibility |
| Reference WAV | upload | optional | Voice cloning |

---

#### Model 9 — Parler-TTS ✅ INSTALLED
| | |
|---|---|
| **Package** | `parler_tts 0.2.3` |
| **Model** | `parler-tts/parler-tts-mini-v1` |
| **Model cache** | 3.3 GB at `huggingface/hub/models--parler-tts--parler-tts-mini-v1` |
| **Runtime RAM** | ~1 500 MB |
| **Cold RTF** | 27× |
| **Warm RTF** | **23×** ❌ |
| **Load time** | ~33s |
| **Sample rate** | 44 100 Hz |

**Unique:** Voice is controlled entirely by a **text description**. No voice file needed.
```
"An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly,
with natural pauses and a soft Southern American accent."
```
Longer, more specific descriptions improve consistency.

**Dependency fix:** `transformers 5.x` removed `SlidingWindowCache` used by parler_tts 0.2.3. Pinned to `transformers==4.46.1`.

**Other available models:**
| Model | Size | Notes |
|---|---|---|
| `parler-tts-mini-v1` | 3.3 GB | ← installed |
| `parler-tts-large-v1` | ~5 GB | Higher quality, slower |
| `parler-tts-mini-expresso` | 3.3 GB | More expressive styles |

---

#### Model 10 — Dia-1.6B ✅ INSTALLED (new this session)
| | |
|---|---|
| **Package** | `dia` (git: nari-labs/dia) |
| **Model used** | `nari-labs/Dia-1.6B-0626` (v2 config) |
| **Model cache** | 6.1 GB at `huggingface/hub/models--nari-labs--Dia-1.6B-0626` |
| **Runtime RAM** | ~3 000+ MB |
| **Warm RTF** | **~55×** ❌ very slow on CPU |
| **Load time** | ~22s |
| **Sample rate** | 44 100 Hz |

**Unique feature: Native dialogue with speaker turns AND emotion tags:**
```
[S1] Hello? Who is this?
[S2] Hello, this is the IRS.
[S1] Oh my goodness. [sighs] I've been waiting for your call. [laughs nervously]
[S2] We need your social security number immediately.
[S1] Just a moment dear. [clears throat]
```

**Emotion tags:** `[laughs]` `[sighs]` `[coughs]` `[groans]` `[gasps]` `[sobs]` `[clears throat]`

**Parameters:**
| Param | Range | Default | Effect |
|---|---|---|---|
| `cfg_scale` | 1.0–5.0 | 3.0 | Guidance strength |
| `temperature` | 0.5–2.0 | 1.2 | Randomness |
| `top_p` | 0.5–1.0 | 0.95 | Nucleus sampling cutoff |
| `max_tokens` | 0=auto | auto | 0 = calc from text length |

**Critical bug fixed — infinite generation hang:**
Default `max_tokens=3072` caused 35+ minute hangs on short text because Dia
generated all 3072 tokens without hitting EOS. Fixed with auto-estimate:
```python
auto_tokens = min(1024, max(256, len(text) * 6))
```
UI slider default is 0 (auto). Override for long texts.

**Config mismatch fixed:** `nari-labs/Dia-1.6B` has outdated config missing `encoder_config`
field needed by current package. Code now tries `Dia-1.6B-0626` first (v2 config,
compatible), falls back to `Dia-1.6B` if not available.

---

#### Model 11 — CosyVoice2 ❌ NOT INSTALLED
| | |
|---|---|
| **Package** | Not on PyPI — requires manual git clone |
| **Error** | `No module named 'hyperpyyaml'` (missing dependency) |
| **Why skipped** | Chinese-first model; complex manual install; lower value for English Arthur |

**Install when needed:**
```bash
git clone https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice
cd /opt/CosyVoice && pip install -r requirements.txt
python tools/download_model.py CosyVoice2-0.5B
# Then re-run: sudo systemctl restart arthur-lab
```

---

## 6. Whisper STT Models

Used by `arthur_server.py` on port 8000.

| Model | Size | WER | CPU RTF | Cache | Status |
|---|---|---|---|---|---|
| `tiny.en` | 74 MB | ~8% | ~200× RT | ❌ | not downloaded |
| `base.en` | 141 MB | ~5% | ~100× RT | ✅ | **← ACTIVE** |
| `small.en` | 464 MB | ~3% | ~30× RT | ✅ | cached, try this next |
| `medium.en` | 1.5 GB | ~2% | ~8× RT | ✅ | cached |
| `large-v3` | 3.0 GB | ~1% | ~3× RT | ❌ | not downloaded |

**Recommended upgrade** (switch to `small.en`):
```python
# arthur_server.py line ~50:
WHISPER_MODEL = "small.en"   # was "base.en"
```
Then: `scp -i %USERPROFILE%\.ssh\id_arthur_vm arthur_server.py arthur@192.168.0.87:/opt/arthur/ && ssh ... sudo systemctl restart arthur`

---

## 7. Thread / CPU Configuration

All 12 vCores used by every library:

| Layer | How | Value |
|---|---|---|
| OS env vars | systemd `Environment=` in both service units | `OMP=MKL=ORT=OPENBLAS=NUMEXPR=CPU_THREADS=12` |
| PyTorch intra-op | `torch.set_num_threads(12)` at `tts_lab.py` top | 12 |
| PyTorch inter-op | `torch.set_num_interop_threads(6)` | 6 |
| OnnxRuntime (Piper) | `SessionOptions.intra_op_num_threads=12` | 12 |
| OnnxRuntime (Kokoro) | `SessionOptions.intra_op_num_threads=12` | 12 |
| CTranslate2 (Whisper) | `WhisperModel(cpu_threads=CPU_THREADS, num_workers=1)` | 12 |

> **Why `torch.set_num_threads()` must be explicit:** PyTorch defaults to `nproc/2 = 6`
> even when `OMP_NUM_THREADS=12` is set. The explicit call overrides this.
> Must be called **after** `import torch`.

---

## 8. RTF Benchmark Results

### 8.1 Original 6 models (from earlier session, warm calls)
Same test phrase (~12s audio): *"Oh my goodness, just a moment dear, let me find my reading glasses. Now you said I owe money to the IRS? Can you give me that case number again, nice and slow?"*

| Model | Cold RTF | **Warm RTF** | Real-time? | Load time | RAM |
|---|---|---|---|---|---|
| **Piper** | 1.60× | **0.75×** | ✅ | 3.3s | 200 MB |
| **MeloTTS** | 1.11× | **0.75–1.8×** | ✅* | 4.3s | 1 200 MB |
| Kokoro | 5.05× | 3.85× | ❌ | 2.1s | 500 MB |
| XTTS-v2 | 5.0× | 4.7× | ❌ | 31s | 3 200 MB |
| Chatterbox | 11.8× | 13.5× | ❌ | 25s | 1 800 MB |
| Parler-TTS | 27× | 23× | ❌ | 33s | 1 500 MB |

*MeloTTS degrades to 1.8× when other heavy models are resident.

### 8.2 New 4 models (measured this session)
Test phrase (~3s audio): *"Oh my goodness, just a moment dear."*

| Model | Warm RTF (measured) | Load time | RAM | Notes |
|---|---|---|---|---|
| **StyleTTS 2** | **2.96×** | 13.5s | ~1 500 MB | Best neural quality/speed ratio |
| **F5-TTS** | not benchmarked yet | ~15s | ~2 000 MB | Requires reference WAV |
| **Bark** | **17.4×** | 66s cold | ~1 500 MB | Unique emotion tokens |
| **Dia-1.6B** | **55×** | 21.9s | ~3 000 MB | Dialogue-native; very slow on CPU |

> **Xeon D-1528 root cause:** 1.90 GHz base clock + no AVX-512.
> All neural models are compute-bound on single-utterance inference.
> More cores help torch/MKL operations but not the sequential autoregressive loop.

---

## 9. Complete Bug Log (all 29)

| # | Bug | Fix |
|---|---|---|
| 1 | `deploy.ps1` default `$VM` pointed at Hyper-V host IP `.153` not VM `.87` | Changed `$VM` default |
| 2 | `deploy.ps1` used plink/password auth; SSH key already existed but was unused | Replaced plink with `ssh -i $Key` / `scp -i $Key` |
| 3 | All `.sh` files had Windows CRLF → `\r: command not found` on Linux | PowerShell CRLF→LF conversion in deploy step |
| 4 | All `.sh` files had UTF-8 BOM → `#!/usr/bin/env: No such file or directory` | `WriteAllText` with `new UTF8Encoding(false)` (no-BOM) |
| 5 | `setup_tts_lab.sh` had `set -e` — first `pip install` failure killed whole script | Removed `set -e`; correct package names used instead |
| 6 | `melo-tts` does not exist on PyPI | Correct name is `melotts`; also install via GitHub URL |
| 7 | `TTS` package (original Coqui) is abandoned | Changed to `coqui-tts 0.27.5` (community fork) |
| 8 | Kokoro `wget` URL returned 404 | Release tag changed `model-files` → `model-files-v1.0` |
| 9 | Kokoro downloaded files were 0 bytes | `wget` didn't follow HF redirects; changed to `curl -L` |
| 10 | MeloTTS import failed with MeCab error | `sudo python -m unidic download` (526 MB dictionary) |
| 11 | Piper `synthesize()` API changed in 1.4.x | Rewrote to iterate `AudioChunk` objects, use `SynthesisConfig` |
| 12 | `inst.config.audio.sample_rate` missing in Piper 1.4.x | Changed to `inst.config.sample_rate` (flat attribute) |
| 13 | MeloTTS `spk2id` is `HParams` not `dict` — `.get()` fails | Wrapped in `dict(...)` before lookup |
| 14 | MeloTTS Indian English key is `EN_INDIA` not `EN-INDIA` | Fixed key normalisation in `_synth_melo()` |
| 15 | `transformers 5.3.0` removed `SlidingWindowCache` — parler_tts 0.2.3 breaks | Pinned `transformers==4.46.1` |
| 16 | `coqui-tts` fails: `isin_mps_friendly`, `is_torch_greater_or_equal`, `is_torchcodec_available` missing | `_patch_transformers_for_coqui()` monkey-patches all 3 |
| 17 | XTTS-v2 prompts for licence interactively in systemd → `EOFError` | `COQUI_TOS_AGREED=1` in env + `_load_xtts()` |
| 18 | PyTorch defaults to `nproc/2 = 6` threads despite `OMP_NUM_THREADS=12` | Explicit `torch.set_num_threads(12)` after `import torch` |
| 19 | `WhisperModel` used no explicit threads (defaulted to whatever CTranslate2 chose) | Added `cpu_threads=CPU_THREADS, num_workers=1` |
| 20 | `_available()` used `importlib.find_spec()` — passes even when imports fail | Replaced with live `exec(import_stmt)` test, cached in `_import_cache` |
| 21 | `protobuf 3.19.6` missing `builder` module (added in 3.20) | Upgraded to `protobuf>=4.25,<7` |
| 22 | `transformers 4.40.2` (pip default) too old — no `MinPLogitsWarper` for Chatterbox | Pinned to `transformers==4.46.1` |
| 23 | Corrupt partial installs (`~rotobuf`, `~ransformers`, `~okenizers` dirs) left by failed upgrade | `sudo rm -rf` the tilde-prefixed dirs from site-packages |
| 24 | PyTorch 2.6 `weights_only=True` blocks numpy globals in Bark pickled checkpoints | Patch `torch.load` → `weights_only=False` during `preload_models()` only |
| 25 | PyTorch 2.6 `weights_only=True` blocks `builtins.getattr` in StyleTTS2 checkpoints | Same `torch.load` patch during `StyleTTS2()` constructor |
| 26 | StyleTTS2 API: `ref_audio=` parameter renamed to `target_voice_path=` | Fixed `_synth_styletts2()` parameter name |
| 27 | Dia API: `speed_factor` parameter removed; `max_tokens` default 3072 hangs for 35+ min | Removed `speed_factor`; added auto-estimate `min(1024, len(text)*6)` |
| 28 | `nari-labs/Dia-1.6B` config missing `encoder_config` field — pydantic validation fails | Code tries `Dia-1.6B-0626` first (v2 config, 6.1 GB); falls back gracefully |
| 29 | NLTK `punkt_tab` downloaded for `arthur` user; service runs as `root` | Re-ran `nltk.download` under `sudo` |

---

## 10. What To Try Next

### 🔴 High impact
| Config | What | Command |
|---|---|---|
| **Switch Whisper to small.en** | Better STT accuracy, still real-time on 12 cores | `WHISPER_MODEL = "small.en"` in `arthur_server.py` |
| **Piper en_GB-alan-medium** | Best Arthur accent | Already on disk — change `voice` param to `en_GB-alan-medium` in UI |

### 🟡 Performance
| Config | Expected gain | How |
|---|---|---|
| **Kokoro fp32 model** | May beat int8 on AVX2 | Download `kokoro-v1.0.onnx` (fp32 330 MB), replace current file |
| **OpenVINO EP for Piper** | 1.5–2× on Intel CPU | `pip install onnxruntime-openvino` + `providers=["OpenVINOExecutionProvider"]` |
| **MeloTTS with no resident models** | RTF returns to 0.75× | Keep only Piper + Melo loaded |

### 🟢 Voice quality evaluation
| Config | What | Notes |
|---|---|---|
| **Bark + emotion tokens** | `[sighs]` `[laughs]` `[clears throat]` | Best for naturalness despite slow RTF |
| **Chatterbox exaggeration 0.6** | Confused elderly hesitation | Use with voice clone reference |
| **Kokoro `bm_lewis`** | British male, best Kokoro voice for Arthur | Change voice dropdown in UI |
| **Parler-TTS prompt tuning** | Text description controls everything | Try longer, more specific descriptions |
| **XTTS `Torcull Diarmuid`** | Best elderly male speaker | Built-in, no download |

### 🔵 Future (GPU)
| Config | Impact | Notes |
|---|---|---|
| **GPU passthrough** | All models real-time | GTX 1060 or better. XTTS: 4.7× → 0.05× |
| **Install CosyVoice2** | Zero-shot English cloning | Manual git clone (~2 GB download) |
| **StyleTTS2 reference voice** | Arthur-specific timbre | Record 30s of target voice, upload as reference |

---

## 11. File Reference

| File | Location | Purpose |
|---|---|---|
| `arthur_server.py` | `/opt/arthur/` | Production: Whisper STT + Gemini LLM + Gemini TTS |
| `tts_lab.py` | `/opt/arthur/` | TTS evaluation web UI (port 8001) — 11 engines |
| `setup_tts_lab.sh` | `/opt/arthur/` | Full TTS lab install (idempotent, re-runnable) |
| `setup_vm.sh` | `/opt/arthur/` | Production server install (arthur.service only) |
| `download_models.sh` | `/opt/arthur/` | Download Piper + Kokoro ONNX models only |
| `bench_all.py` | `/opt/arthur/` | Sequential cold-start benchmark all models |
| `bench_warm.py` | `/opt/arthur/` | Isolated warm RTF benchmark per model |
| `tts_benchmark.py` | `/opt/arthur/` | Legacy sequential 6-model benchmark |
| `run_benchmark.sh` | `/opt/arthur/` | Shell wrapper for tts_benchmark.py |
| `requirements.txt` | `tools/arthur_server/` | Production venv (arthur-env) |
| `requirements_benchmark.txt` | `tools/arthur_server/` | TTS lab venv (arthur-bench-env) |
| `deploy.ps1` | `tools/arthur_server/` | Deploy all files Windows → VM via SSH key |
| `VM_SETUP_REFERENCE.md` | `tools/arthur_server/` | This document |

---

## 12. Quick-Reference Commands

```bash
# SSH into VM
ssh -i %USERPROFILE%\.ssh\id_arthur_vm arthur@192.168.0.87

# Deploy all files from Windows
cd C:\repos\Spamblocker\tools\arthur_server && .\deploy.ps1

# Service management
sudo systemctl status arthur arthur-lab
sudo systemctl restart arthur arthur-lab
sudo journalctl -u arthur-lab -f
sudo journalctl -u arthur -f

# Disk + RAM snapshot
free -h && df -h / /opt/models

# Check which TTS models are available/loaded
curl -s http://localhost:8001/status | python3 -m json.tool

# Synthesise test — Piper (replace voice for other installed voices)
curl -s -X POST http://localhost:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello there, this is Arthur.","params":{"voice":"en_GB-alan-medium"}}' | \
  python3 -c "import sys,json,base64; open('/tmp/t.wav','wb').write(base64.b64decode(json.load(sys.stdin)['audio_b64']))"
aplay /tmp/t.wav

# Synthesise test — Bark with emotion tokens
curl -s -X POST http://localhost:8001/synthesize/bark \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello? [sighs] Oh my goodness, just a moment dear. [clears throat]","params":{"voice_preset":"v2/en_speaker_6"}}' | \
  python3 -c "import sys,json,base64; open('/tmp/t.wav','wb').write(base64.b64decode(json.load(sys.stdin)['audio_b64']))"

# Expand model disk partition (run after Hyper-V disk resize)
sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1

# Download NLTK data for root (run if StyleTTS2 fails with punkt_tab error)
sudo /opt/arthur-bench-env/bin/python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('cmudict')"
```

---

## 13. Architecture Reminder

```
Android (Pixel 5)
    │
    │  scam call detected
    ▼
BaiterInCallService
    │
    ├── STT: Android SpeechRecognizer (on-device)
    │
    └── AI Bridge Mode (optional):
              │
              ▼  HTTP
         arthur_server.py  (port 8000)
              │
              ├── STT:  faster-whisper base.en  (local, 12 threads)
              ├── LLM:  Gemini 2.0 Flash        (cloud API)
              └── TTS:  Gemini 2.5 Flash TTS    (cloud API, voice "Gacrux")
                        ← swap with Piper (en_GB-alan-medium) to go fully local

TTS Lab  (port 8001)  ← voice evaluation only, not used in production
    │
    ├── Real-time capable (RTF < 1.0):
    │     Piper TTS      0.75×  ✅  6 voices, ultra-light
    │     MeloTTS        0.75×  ✅* 5 accents (degrades under RAM pressure)
    │
    ├── Evaluation / voice selection (too slow for live calls):
    │     StyleTTS 2     ~3×    ⚡ fastest neural, reference-audio style
    │     Kokoro-82M     3.85×  ⚡ 54 voices, 9 languages
    │     XTTS-v2        4.7×   58 speakers, voice cloning, 17 languages
    │     F5-TTS         ~5×    best zero-shot voice cloning
    │     Chatterbox     13.5×  exaggeration + voice cloning
    │     Bark           17×    emotion tokens [laughs] [sighs] [clears throat]
    │     Parler-TTS     23×    text-description voice control
    │     Dia-1.6B       55×    dialogue-native [S1]/[S2] + emotion tags
    │
    └── Not installed:
          CosyVoice2     N/A    manual install, Chinese-first
```
