# Arthur TTS Lab

A self-hosted, 21-engine Text-to-Speech benchmark and evaluation lab. Compare every major open-source TTS model side-by-side through a single web UI.

---

## Features

- **21 TTS engines** — every major open-source TTS model in one web UI
- **5 Image & Video engines** — FLUX.2, SD 3.5, Ideogram4, Wan2.2, FLUX.2 Klein
- **Side-by-side comparison** — switch engines instantly, compare voices, measure quality
- **Voice Library** — browse, play, download Persian reference voices from Common Voice
- **Reference WAV cloning** — upload or select reference audio for zero-shot voice cloning engines
- **RTF benchmarking** — automated Real-Time Factor measurement across all TTS engines
- **Persian text processing** — G2P, hazm, parsivar providers with live preview
- **NVFP4 native quantization** — Blackwell-optimized weights for FLUX.2 and Wan2.2
- **Single-click deploy** — PowerShell scripts deploy everything to an Ubuntu VM via SSH

---

## Quick Start

```powershell
# Deploy to existing VM (~30 sec):
.\scripts\deploy\deploy_lab.ps1 -Phase 5

# Full deploy to fresh VM (~30-60 min):
.\scripts\deploy\deploy_lab.ps1

# Open in browser:
# http://192.168.0.87:8001
```

---

## Architecture

```
Windows dev machine  ──SCP──►  Ubuntu VM (192.168.0.87)
  scripts/deploy/deploy_lab.ps1                  /opt/arthur/
  tts_lab*.py    ──────────────►    tts_lab.py           FastAPI entry-point
  patch_*.py                        tts_lab_shims.py     startup compat patches
                                    tts_lab_config.py    model catalogue + state
                                    tts_lab_utils.py     shared audio helpers
                                    tts_lab_engines.py   21 load/synth pairs
                                    tts_lab_dispatch.py  HTTP handlers
                                    tts_lab_ui.py        web UI (HTML/JS)
                                        |
                                   FastAPI (uvicorn, port 8001)
                                        |
                              arthur-lab.service (systemd)
```

### Component Layout

| File | Role |
|---|---|
| `tts_lab.py` | FastAPI app, lifespan, top-level route wiring |
| `tts_lab_shims.py` | **Imported FIRST** — `sys.modules` stubs for `transformers` compat |
| `tts_lab_config.py` | `MODEL_INFO` catalogue, all voice lists, per-engine state, paths |
| `tts_lab_utils.py` | Shared helpers: `wav_to_bytes()`, `resample()`, temp file helpers |
| `tts_lab_engines.py` | All 21 `_load_X()` + `_synth_X()` pairs |
| `tts_lab_dispatch.py` | `_ensure_loaded()`, `_do_synth()`, HTTP handler implementations |
| `tts_lab_ui.py` | Full HTML/JS web UI — sidebar, waveform player, engine tabs, debug drawer |
| `voice_library.py` | Voice Library — browse, import, manage reference voices |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/status` | JSON: all engines, availability, RAM estimates |
| `POST` | `/synthesize/{engine}` | Synthesise audio — returns WAV binary |
| `POST` | `/synthesize/{engine}` (multipart) | With reference WAV upload |
| `GET` | `/logs` | Last 200 server-side log entries (ring buffer) |
| `POST` | `/refresh` | Re-probe all engine availability without restart |
| `GET` | `/models/{engine}` | Engine metadata |
| `POST` | `/models/{engine}/load` | Force-load an engine into memory |

### Synthesise Request

```json
{
  "text": "Hello world.",
  "params": {
    "voice":        "bm_lewis",
    "speed":        1.0,
    "speaker":      "Baldur Sanjin",
    "language":     "en",
    "description":  "A warm elderly male voice speaking slowly and clearly.",
    "exaggeration": 0.5
  }
}
```

Parameters are engine-specific — unused ones are silently ignored.

---

## 21 Engines

| # | Key | Label | Model Size | Voice Cloning | Notes |
|---|---|---|---|---|---|
| 1 | `piper` | Piper TTS | 61-116 MB | — | ONNX CPU-only. Real-time on any hardware. 6 voices. |
| 2 | `kokoro` | Kokoro-82M | 89 MB | — | 54 voices, 9 languages. |
| 3 | `melo` | MeloTTS | 200 MB | — | 5 English accents. |
| 4 | `chattts` | ChatTTS | 1.2-2.3 GB | — | Speed prompts `[speed_N]`, speaker sampling. |
| 5 | `outetts` | OuteTTS 1.0 | 384 MB (Q4) | — | GGUF via llama.cpp. |
| 6 | `bark` | Bark | 2.5 GB | — | Emotion tokens: `[laughs]` `[sighs]` `[clears throat]`. |
| 7 | `styletts2` | StyleTTS 2 | 0.7 GB | ✅ | Style transfer from reference WAV. |
| 8 | `f5tts` | F5-TTS | 1.2 GB | ✅ | Best zero-shot voice cloning. Needs 5-15 s reference. |
| 9 | `dia` | Dia-1.6B | 3 GB | ✅ | Dialogue-native. `[S1]`/`[S2]` speakers + emotion tags. |
| 10 | `xtts` | XTTS-v2 | 1.8 GB | ✅ | 58 speakers, 17 languages. |
| 11 | `cosyvoice` | CosyVoice2 | 2 GB | ✅ | Zero-shot + cross-lingual. |
| 12 | `parler` | Parler-TTS | 2.5-3.3 GB | — | Natural-language voice description prompts. |
| 13 | `chatterbox` | Chatterbox | 3.0 GB | ✅ | Exaggeration slider + voice cloning. |
| 14 | `fishspeech` | Fish Speech | ~1.1 GB | ✅ | Zero-shot cloning, reference WAV optional. |
| 15 | `csm` | Sesame CSM 1B | ~2 GB | — | Multi-speaker, context-aware. |
| 16 | `qwen3tts` | Qwen3-TTS | ~3 GB | — | 9 built-in speakers. |
| 17 | `orpheus` | Orpheus 3B | ~3 GB | — | Emotion: `<laugh>` `<sigh>` `<chuckle>` `<gasp>`. |
| 18 | `neutts` | NeuTTS Air | TBD | — | Not yet configured. |
| 19 | `indextts` | IndexTTS-2 | ~1.5 GB | ✅ | Zero-shot cloning. Reference WAV required. |
| 20 | `zonos` | Zonos v0.1 | ~1.2 GB | ✅ | Emotion vector + speaking-rate. 44 kHz output. |
| 21 | `openvoice` | OpenVoice v2 | ~600 MB | ✅ | MeloTTS base + tone-color conversion. |

---

## Image Lab — 5 Image & Video Engines

The project also includes a separate Image & Video generation lab on port 8002:

| # | Key | Label | Type | VRAM | Notes |
|---|---|---|---|---|---|
| 1 | `flux2` | FLUX.2 [dev] | Image | ~16 GB | 32B rectified flow transformer. GGUF quantised. I2I editing. |
| 2 | `flux2klein` | FLUX.2 Klein 4B | Image | ~13 GB | Compact 4B model. Apache 2.0. Step-distilled. Runs on RTX 5060 Ti at BF16. |
| 3 | `sd35` | SD 3.5 Large | Image | ~12 GB | 8B MMDiT. GGUF quantised. Turbo/Lightning speed presets. |
| 4 | `wan` | Wan2.2 | Video | ~14 GB | T2V + I2V. Up to 5s cinematic video. Dual-transformer GGUF. |
| 5 | `ideogram4` | Ideogram 4 | Image | ~6-10 GB | 9.3B DiT + Qwen3-VL text encoder. Native text rendering. NF4/FP8 quants. Magic-prompt expansion via OpenRouter. |

> **Optional:** ComfyUI integration toggle via `IMGLAB_USE_COMFYUI=1` env var.

### Image Lab API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Image Lab Web UI (port 8002) |
| `POST` | `/generate/{engine}` | Generate image or video |
| `GET` | `/gallery` | Browse generated images/videos |
| `DELETE` | `/gallery/{id}` | Delete gallery entry |
| `POST` | `/generate/ideogram4/caption` | Expand prompt via Ideogram4 caption endpoint |

Image Lab deploys separately: `.\scripts\deploy\deploy_image_lab.ps1`

See [`docs/image-lab/`](docs/image-lab/) for detailed documentation.

---

## Deploy Script

Single PowerShell script, 8 idempotent phases:

```powershell
.\scripts\deploy\deploy_lab.ps1 [-VM <ip>] [-User <user>] [-Phase <1-8>] [-SkipPhases "n,n"] [-GPU]
```

| Phase | What it does |
|---|---|
| 1 | apt packages, 8 GB swap, data disk mount, Python 3.11 venv |
| 2 | PyTorch (CPU or CUDA) + onnxruntime + soundfile |
| 3 | All 21 engine pip installs (best-effort, logged per engine) |
| 4 | Model downloads — Piper ONNX, Kokoro, Parler, IndexTTS-2 |
| 5 | SCP 7 `tts_lab_*.py` modules + patch scripts to VM |
| 6 | Re-apply transformers/parler_tts compat patches |
| 7 | Write `arthur-lab.service`, `systemctl enable`, restart |
| 8 | HTTP 200 check, `/status` table, Piper smoke-test synthesis |

```powershell
.\scripts\deploy\deploy_lab.ps1              # fresh VM: phases 1-8
.\scripts\deploy\deploy_lab.ps1 -Phase 5    # redeploy code only (most common)
.\scripts\deploy\deploy_lab.ps1 -Phase 6    # re-patch + restart
.\scripts\deploy\deploy_lab.ps1 -Phase 7    # restart service only
.\scripts\deploy\deploy_lab.ps1 -GPU        # use CUDA PyTorch wheels
.\scripts\deploy\deploy_lab.ps1 -SkipPhases "4"   # skip model downloads
```

---

## VM Details

| Property | Value |
|---|---|
| Host | Proxmox node, VM 104 |
| OS | Ubuntu 22.04 |
| IP | 192.168.0.87 |
| Port | 8001 |
| SSH key | `~/.ssh/id_arthur_vm` |
| venv | `/opt/arthur-bench-env/` (Python 3.11) |
| Lab code | `/opt/arthur/` |
| Models | `/opt/models/` (650 GB data disk) |
| HF cache | `/opt/models/huggingface/` |
| Service | `arthur-lab.service` (systemd) |
| PyTorch | CPU-only |
| RAM | 32 GB |

---

## Compatibility Patches

The project targets `transformers 4.53.2` and applies 5 patch scripts on every deploy to bridge API gaps with newer transformers versions and engine-specific quirks. All patches are idempotent.

| Patch | Purpose |
|---|---|
| [`patches/patch_parler_tts.py`](patches/patch_parler_tts.py) | `parler_tts` 0.2.3 → `transformers` 4.51+ compat (6 fixes) |
| [`patches/patch_transformers_stubs.py`](patches/patch_transformers_stubs.py) | Missing `transformers` 4.54+ modules (masking_utils, modeling_layers, SequenceSummary) |
| [`patches/fix_transformers_shims.py`](patches/fix_transformers_shims.py) | Decorator shims (`auto_docstring`, `check_model_inputs`) + `GeneralInterface` |
| [`patches/patch_torchaudio.py`](patches/patch_torchaudio.py) | `torchaudio` backend compat for CosyVoice |
| [`patches/patch_torchaudio_init.py`](patches/patch_torchaudio_init.py) | Additional torchaudio import guards |

See the [patches/](patches/) directory for details.

---

## Testing

| Script | Description |
|---|---|
| [`scripts/test/e2e_test.ps1`](scripts/test/e2e_test.ps1) | Full 10-section E2E test suite (SSH, health, synthesis, ref-WAV, crash scan) |
| [`scripts/test/quick_test.sh`](scripts/test/quick_test.sh) | Fast synthesis smoke test (10 engines, 180 s timeout) |
| [`scripts/test/test_slow_engines.sh`](scripts/test/test_slow_engines.sh) | 5-min timeout test for indextts/qwen3tts/openvoice |

---

## Benchmarks

Automated RTF measurement across all engines. Results are in [`docs/benchmarks/`](docs/benchmarks/).

| Script | Description |
|---|---|
| [`scripts/benchmark/tts_benchmark.py`](scripts/benchmark/tts_benchmark.py) | Automated RTF benchmark across all engines |
| [`scripts/benchmark/bench_all.py`](scripts/benchmark/bench_all.py) | Batch benchmark runner (calls running server) |
| [`scripts/benchmark/bench_warm.py`](scripts/benchmark/bench_warm.py) | Warm-cache benchmark (excludes load time) |

---

## Adding a New Engine

1. Add entry to `MODEL_INFO` dict in `tts_lab_config.py`
2. Add key to `MODEL_ORDER` list in `tts_lab_config.py`
3. Add `_load_xxx()` and `_synth_xxx()` in `tts_lab_engines.py`
4. Register both in `LOADERS` and `SYNTHERS` dicts at bottom of `tts_lab_engines.py`
5. Add package name to `pkg_map` in `_check_available()` in `tts_lab_dispatch.py`
6. Deploy: `.\scripts\deploy\deploy_lab.ps1 -Phase 5`

---

## Documentation

| Document | Topic |
|---|---|
| [`docs/sessions/SESSION_SUMMARY.md`](docs/sessions/SESSION_SUMMARY.md) | Rolling cross-session summary |
| [`docs/reference/TTS_MODEL_COMPARISON.md`](docs/reference/TTS_MODEL_COMPARISON.md) | Side-by-side quality comparison notes v1 |
| [`docs/reference/TTS_MODEL_COMPARISON2.md`](docs/reference/TTS_MODEL_COMPARISON2.md) | Side-by-side quality comparison notes v2 |
| [`docs/reference/PERSIAN_TTS_MODELS.md`](docs/reference/PERSIAN_TTS_MODELS.md) | Comprehensive Persian/Farsi TTS reference |
| [`docs/reference/KNOWN_ISSUES.md`](docs/reference/KNOWN_ISSUES.md) | Current bugs and planned fixes |
| [`docs/reference/VM_SETUP_REFERENCE.md`](docs/reference/VM_SETUP_REFERENCE.md) | Proxmox VM setup, disk expansion, network |
| [`docs/reference/GPU_QA_REFERENCE.md`](docs/reference/GPU_QA_REFERENCE.md) | SM 12.0 (Blackwell) library compatibility |
| [`docs/reference/GPU_UPGRADE_ANALYSIS.md`](docs/reference/GPU_UPGRADE_ANALYSIS.md) | GPU upgrade analysis and flash-attn verdict |
| [`docs/image-lab/`](docs/image-lab/) | Image Lab subsystem documentation |

Full session notes are in [`docs/sessions/`](docs/sessions/).

---

## Environment Variables (systemd service)

| Variable | Value | Purpose |
|---|---|---|
| `COQUI_TOS_AGREED` | `1` | Suppress XTTS ToS prompt |
| `HF_HOME` | `/opt/models/huggingface` | HF model cache on data disk |
| `XDG_CACHE_HOME` | `/opt/models/cache` | General cache on data disk |
| `SUNO_USE_SMALL_MODELS` | `False` | Use full Bark models |
| `CUDA_VISIBLE_DEVICES` | *(empty)* | Hide GPU — force CPU on this VM |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer fork warning |
| `HF_TOKEN` | *(from VM keychain)* | Access gated HF models |

---

## Useful Commands

```powershell
# Deploy from Windows
.\scripts\deploy\deploy_lab.ps1 -Phase 5

# SSH to VM
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87
```

```sh
# Service logs
sudo journalctl -u arthur-lab -f
sudo journalctl -u arthur-lab -n 50 --no-pager

# Restart service
sudo systemctl restart arthur-lab

# Re-apply patches manually (after pip upgrade)
source /opt/arthur-bench-env/bin/activate
python3 /opt/arthur/patches/patch_transformers_stubs.py
python3 /opt/arthur/patches/fix_transformers_shims.py
python3 /opt/arthur/patches/patch_parler_tts.py

# Check all 21 engines
curl -s http://localhost:8001/status | python3 -m json.tool

# Quick synthesis test
bash /tmp/quick_test.sh
```

---

## License

MIT — see [LICENSE](LICENSE).
