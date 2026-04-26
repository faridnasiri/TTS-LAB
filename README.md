# Arthur TTS Lab

A self-hosted, 21-engine Text-to-Speech benchmark and evaluation lab running on an Ubuntu VM.
Compare every major open-source TTS model side-by-side through a single web UI on port 8001.

---

## Quick Start

```powershell
# From Windows dev machine — existing VM (code + patches + restart, ~30 sec):
cd C:\repos\Spamblocker\tools\tts-lab
.\deploy_lab.ps1 -Phase 5

# Brand-new blank VM (everything from zero, ~30-60 min):
.\deploy_lab.ps1

# Open in browser:
# http://192.168.0.87:8001
```

---

## Architecture

```
Windows dev machine  ──SCP──►  Ubuntu VM (192.168.0.87)
  deploy_lab.ps1                  /opt/arthur/
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

---

## VM Details

| Property | Value |
|---|---|
| Host | Proxmox node, VM 104 |
| OS | Ubuntu 22.04 |
| IP | 192.168.0.87 |
| Port | 8001 |
| SSH key | `~/.ssh/id_arthur_vm` |
| SSH user | `arthur` |
| venv | `/opt/arthur-bench-env/` (Python 3.11) |
| Lab code | `/opt/arthur/` |
| Models | `/opt/models/` (650 GB data disk, ext4) |
| HF cache | `/opt/models/huggingface/` |
| Service | `arthur-lab.service` (systemd, runs as root) |
| PyTorch | CPU-only (no GPU passthrough on this VM) |
| RAM | 32 GB |
| Disk | 650 GB (nvme-lvm expanded) |

---

## 21 Engines

| # | Key | Label | Model Size | RTF (GPU) | Arthur Fit | Notes |
|---|---|---|---|---|---|---|
| 1 | `piper` | Piper TTS | 61-116 MB | 0.36x | ** | ONNX CPU-only. Real-time on any hardware. 6 voices. |
| 2 | `kokoro` | Kokoro-82M | 89 MB | 2.77x | ***** | 54 voices, 9 languages. `bm_lewis` = best Arthur voice. |
| 3 | `melo` | MeloTTS | 200 MB | 0.30x | *** | 5 English accents. `EN-BR` sounds slightly older. |
| 4 | `chattts` | ChatTTS | 1.2-2.3 GB | 2.59x | **** | Speed prompts `[speed_N]`, speaker sampling. |
| 5 | `outetts` | OuteTTS 1.0 | 384 MB (Q4) | 1.45x | **** | GGUF via llama.cpp. Default: `OuteTTS-1.0-0.6B-Q4_K_M.gguf` |
| 6 | `bark` | Bark | 2.5 GB | 4.64x | ***** | Emotion tokens: `[laughs]` `[sighs]` `[clears throat]` |
| 7 | `styletts2` | StyleTTS 2 | 0.7 GB | 0.35x | **** | Fast + high quality. Style transfer from reference WAV. |
| 8 | `f5tts` | F5-TTS | 1.2 GB | needs ref WAV | **** | Best zero-shot voice cloning. Needs 5-15 s reference WAV. |
| 9 | `dia` | Dia-1.6B | 3 GB | 6.75x | ***** | Dialogue-native. `[S1]`/`[S2]` speakers + emotion tags. |
| 10 | `xtts` | XTTS-v2 | 1.8 GB | 0.91x | ***** | 58 speakers, 17 languages. Voice cloning. |
| 11 | `cosyvoice` | CosyVoice2 | 2 GB | ~0.6x | *** | Needs manual install. Zero-shot + cross-lingual. |
| 12 | `parler` | Parler-TTS | 2.5-3.3 GB | ~4.9x | **** | Natural-language voice description prompts. |
| 13 | `chatterbox` | Chatterbox | 3.0 GB | 1.67x | ***** | Exaggeration slider + voice cloning. |
| 14 | `fishspeech` | Fish Speech | ~1.1 GB | ~0.14x | **** | Zero-shot voice cloning, reference WAV optional. |
| 15 | `csm` | Sesame CSM 1B | ~2 GB | ~0.08x | **** | Multi-speaker, context-aware. Gated HF model. |
| 16 | `qwen3tts` | Qwen3-TTS | ~3 GB | ~4.4x | *** | 9 built-in speakers. `Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| 17 | `orpheus` | Orpheus 3B | ~3 GB | ~0.8x | ***** | Emotion: `<laugh>` `<sigh>` `<chuckle>` `<gasp>`. Needs CUDA. |
| 18 | `neutts` | NeuTTS Air | TBD | TBD | *** | Not yet configured - edit `_load_neutts()`. |
| 19 | `indextts` | IndexTTS-2 | ~1.5 GB | ~0.4x | **** | Zero-shot cloning. Reference WAV required. |
| 20 | `zonos` | Zonos v0.1 | ~1.2 GB | 4.03x | **** | Emotion vector + speaking-rate. 44 kHz output. |
| 21 | `openvoice` | OpenVoice v2 | ~600 MB | ~0.5x | *** | MeloTTS base + tone-color conversion. |

> **RTF** = Real-Time Factor. RTF 1.0 = generates 1 s of audio per 1 s of compute.
> RTF < 1 = faster than real-time. All RTF figures measured on RTX 5060 Ti 16 GB unless noted.

---

## Deploy Script — deploy_lab.ps1

Single PowerShell script, 8 idempotent phases. Run from the Windows dev machine.

```powershell
.\deploy_lab.ps1 [-VM <ip>] [-User <user>] [-Phase <1-8>] [-SkipPhases "n,n"] [-GPU]
```

| Phase | What it does | Skip when |
|---|---|---|
| 1 | apt packages, 8 GB swap, data disk mount, Python 3.11 venv | VM already bootstrapped |
| 2 | PyTorch (CPU or CUDA) + onnxruntime + soundfile | Already installed |
| 3 | All 21 engine pip installs (best-effort, logged per engine) | Already installed |
| 4 | Model downloads - Piper ONNX, Kokoro, Parler, IndexTTS-2 | Models already on disk |
| 5 | SCP 7 tts_lab_*.py modules + 3 patch scripts to VM | — (always fast, ~30 s) |
| 6 | Re-apply transformers/parler_tts compat patches | — (idempotent, always safe) |
| 7 | Write arthur-lab.service, systemctl enable, restart | — (always needed) |
| 8 | HTTP 200 check, /status table, Piper smoke-test synthesis | — |

```powershell
.\deploy_lab.ps1              # fresh VM: phases 1-8
.\deploy_lab.ps1 -Phase 5    # redeploy code only (most common)
.\deploy_lab.ps1 -Phase 6    # re-patch + restart
.\deploy_lab.ps1 -Phase 7    # restart service only
.\deploy_lab.ps1 -GPU        # use CUDA PyTorch wheels
.\deploy_lab.ps1 -SkipPhases "4"   # skip model downloads
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/status` | JSON: all engines, availability, RAM estimates |
| `POST` | `/synthesize/{engine}` | Synthesise audio - returns WAV binary |
| `POST` | `/synthesize/{engine}` (multipart) | With reference WAV upload |
| `GET` | `/logs` | Last 200 server-side log entries (ring buffer) |
| `POST` | `/refresh` | Re-probe all engine availability without restart |
| `GET` | `/models/{engine}` | Engine metadata |
| `POST` | `/models/{engine}/load` | Force-load an engine into memory |

### Synthesise request body

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

Parameters are engine-specific - unused ones are silently ignored.

---

## Source Code Map

```
tts-lab/
|
+-- tts_lab.py              FastAPI app, lifespan, top-level route wiring
+-- tts_lab_shims.py        Imported FIRST - sys.modules stubs for
|                              transformers.masking_utils, modeling_layers,
|                              indextts alias, SequenceSummary stub
+-- tts_lab_config.py       Model catalogue (MODEL_INFO), all voice lists,
|                              Arthur presets, per-engine _state dict, paths
+-- tts_lab_utils.py        Shared helpers: wav_to_bytes(), resample(),
|                              bytes_to_numpy(), temp file helpers
+-- tts_lab_engines.py      All 21 engine pairs: _load_X() + _synth_X()
|                              LOADERS dict, SYNTHERS dict
+-- tts_lab_dispatch.py     _ensure_loaded(), _do_synth(), _check_available()
|                              HTTP handler implementations
+-- tts_lab_ui.py           Full HTML/JS web UI - sidebar, waveform player,
                               engine tabs, debug log drawer, ref-WAV upload
```

---

## Compatibility Patch Scripts

Three scripts re-applied on every deploy (Phase 6). All idempotent - guarded by marker strings.

### patch_parler_tts.py
Patches `parler_tts` 0.2.3 for `transformers` 4.51+ API breakages:

| Patch | Problem | Fix |
|---|---|---|
| `_pad/bos/eos_token_tensor` | Removed from `GenerationConfig` in 4.51 | Replaced with `torch.tensor(generation_config.pad_token_id)` |
| `generation_config.update()` | Now returns `None` instead of `model_kwargs` | `generation_config.update(**kw); model_kwargs = kw` |
| `_prepare_attention_mask_for_generation` | Signature changed `(inputs, pad_t, eos_t)` | Replaced with inline `torch.ones(...)` mask |
| `_get_initial_cache_position` | Signature changed to `(seq_len, device, model_kwargs)` | Shimmed via `(a, b=None, c=None)` |
| `GenerationMixin` | `PreTrainedModel` no longer inherits it in 4.50 | Added `_ParlerGenMixin` to class MRO |
| `ParlerTTSConfig.__init__` | Raises if called with no args | Early return - 4.53 calls it empty in `to_diff_dict()` |

### patch_transformers_stubs.py
Creates missing modules present in `transformers 4.54+` but absent in `4.53.2`:

- `transformers/masking_utils.py` - stub with `create_masks_for_generate()`
- `transformers/modeling_layers.py` - stub with `GradientCheckpointingLayer`
- `transformers/modeling_utils.py` - appends `SequenceSummary` + `ALL_ATTENTION_FUNCTIONS`

### fix_transformers_shims.py
Patches `transformers/utils/__init__.py` and `transformers/utils/generic.py`:

- `auto_docstring()` decorator shim (added in 4.54)
- `check_model_inputs()` decorator shim
- `GeneralInterface` base class (used by `masking_utils.AttentionMaskInterface`)

---

## Testing

### Quick synthesis test (fast engines, ~2 min)
```sh
scp -i ~/.ssh/id_arthur_vm quick_test.sh arthur@192.168.0.87:/tmp/
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87 "bash /tmp/quick_test.sh"
```

### Slow-loading engines (5-min timeout each)
```sh
scp -i ~/.ssh/id_arthur_vm test_slow_engines.sh arthur@192.168.0.87:/tmp/
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87 "bash /tmp/test_slow_engines.sh"
```

### Full E2E test suite (10 sections, from Windows)
```powershell
.\e2e_test.ps1
.\e2e_test.ps1 -VM 10.0.0.5
```

E2E sections: SSH, service health, HTTP 200, /status, fix verifications (2),
synthesis tests, ref-WAV engine checks, GPU engine graceful-fail, journal crash scan.

---

## Known Issues

See `KNOWN_ISSUES.md` for full detail.

| Engine | Issue | Fix needed |
|---|---|---|
| `csm` | Package not installed - gated HF model | `huggingface-cli login` then install |
| `orpheus` | Needs CUDA - not available on this CPU VM | Install on GPU VM |
| `neutts` | `_load_neutts()` raises NotImplementedError | Identify package + implement |
| `qwen3tts` | `_attn_implementation_autoset` intermittent | Shim in `tts_lab_shims.py` |

---

## Benchmark Results

| Date | Hardware | File |
|---|---|---|
| 2026-03-26 | CPU baseline | `BENCHMARK_RESULTS_2026-03-26.md` |
| 2026-04-20 | RTX 5060 Ti 16 GB | `BENCHMARK_RESULTS_2026-04-20_RTX5060Ti.md` |
| 2026-04-23 | RTX 5060 Ti - Qwen3 SDPA | `BENCHMARK_RESULTS_2026-04-23.md` |

---

## Session Notes

| Date | File | Topic |
|---|---|---|
| 2026-03-25 | `SESSION_2026-03-25_CODEBASE_COMPLETION.md` | Initial 13-engine build |
| 2026-03-25 | `SESSION_2026-03-25_MODEL_EXPANSION_REFERENCE.md` | Model expansion reference |
| 2026-03-26 | `SESSION_2026-03-26_DEPLOY_AND_STABILITY.md` | Deploy stability, GPU passthrough |
| 2026-03-26 | `SESSION_2026-03-26_NEW_ENGINES.md` | Engines 14-21 added |
| 2026-04-21 | `SESSION_2026-04-21_TESTING_FISHSPEECH_INDEXTTS.md` | Fish Speech + IndexTTS |
| 2026-04-22 | `SESSION_2026-04-22_GPU_FLASH_ATTN.md` | Flash-attn verdict, SM 12.0 |
| 2026-04-22 | `SESSION_2026-04-22_QWEN3TTS.md` | Qwen3-TTS first integration |
| 2026-04-23 | `SESSION_2026-04-23_QWEN3TTS_UI_UPGRADE.md` | Qwen3 params + UI sidebar |
| 2026-04-25 | `SESSION_2026-04-25_ENGINE_FIXES.md` | 7-fix transformers 4.53 marathon |
| 2026-04-25 | `SESSION_2026-04-25_FIX2_INDEXTTS_QWEN3TTS.md` | IndexTTS v2 + Qwen3 config shim |

---

## File Reference

| File | Purpose |
|---|---|
| `deploy_lab.ps1` | PRIMARY - zero-to-hero 8-phase deploy |
| `e2e_test.ps1` | Full 10-section end-to-end test suite |
| `quick_test.sh` | Fast synthesis smoke test (10 engines, 180 s timeout) |
| `test_slow_engines.sh` | 5-min timeout test for indextts/qwen3tts/openvoice |
| `tts_benchmark.py` | Automated RTF benchmark across all engines |
| `bench_all.py` | Batch benchmark runner |
| `bench_warm.py` | Warm-cache benchmark (excludes load time) |
| `download_models.sh` | Downloads all model files to /opt/models/ |
| `requirements.txt` | Core pip dependencies for the lab service |
| `requirements_benchmark.txt` | Additional benchmark-only dependencies |
| `fix_tts_env.sh` | Emergency venv repair script |
| `patch_parler_tts.py` | parler_tts -> transformers 4.51+ compat patches |
| `patch_transformers_stubs.py` | Creates missing transformers 4.54+ module stubs |
| `fix_transformers_shims.py` | Patches decorators + GeneralInterface into transformers.utils |
| `patch_torchaudio.py` | torchaudio backend compat for CosyVoice |
| `_arthur-lab.service` | Reference systemd unit file |
| `VM_SETUP_REFERENCE.md` | Proxmox VM setup, disk expansion, network |
| `GPU_QA_REFERENCE.md` | SM 12.0 (Blackwell) library compatibility table |
| `GPU_UPGRADE_ANALYSIS.md` | GPU upgrade analysis and flash-attn verdict |
| `TTS_MODEL_COMPARISON.md` | Side-by-side quality comparison notes |
| `KNOWN_ISSUES.md` | Current bugs and planned fixes |
| `archive/` | Old one-off debug scripts (preserved, not in deploy) |

---

## Environment Variables (systemd service)

| Variable | Value | Purpose |
|---|---|---|
| `COQUI_TOS_AGREED` | `1` | Suppress XTTS ToS prompt |
| `HF_HOME` | `/opt/models/huggingface` | HF model cache on data disk |
| `XDG_CACHE_HOME` | `/opt/models/cache` | General cache on data disk |
| `SUNO_USE_SMALL_MODELS` | `False` | Use full Bark models |
| `CUDA_VISIBLE_DEVICES` | (empty) | Hide GPU - force CPU on this VM |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer fork warning |
| `HF_TOKEN` | (injected from VM keychain) | Access gated HF models |

---

## Adding a New Engine

1. Add entry to `MODEL_INFO` dict in `tts_lab_config.py`
2. Add key to `MODEL_ORDER` list in `tts_lab_config.py`
3. Add `_load_xxx()` and `_synth_xxx()` in `tts_lab_engines.py`
4. Register both in `LOADERS` and `SYNTHERS` dicts at the bottom of `tts_lab_engines.py`
5. Add package name to `pkg_map` in `_check_available()` in `tts_lab_dispatch.py`
6. Deploy: `.\deploy_lab.ps1 -Phase 5`

---

## Useful Commands

```powershell
# Deploy from Windows
.\deploy_lab.ps1 -Phase 5

# SSH to VM
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87
```

```sh
# Service logs
sudo journalctl -u arthur-lab -f
sudo journalctl -u arthur-lab -n 50 --no-pager

# Restart
sudo systemctl restart arthur-lab

# Re-apply patches manually (after pip upgrade)
source /opt/arthur-bench-env/bin/activate
python3 /opt/arthur/patch_transformers_stubs.py
python3 /opt/arthur/fix_transformers_shims.py
python3 /opt/arthur/patch_parler_tts.py

# Check all 21 engines
curl -s http://localhost:8001/status | python3 -m json.tool

# Refresh availability badges (no restart)
curl -sX POST http://localhost:8001/refresh | python3 -m json.tool

# Quick synthesis test
bash /tmp/quick_test.sh
```
