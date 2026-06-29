# Arthur TTS Lab — Known Issues & Next Steps

## Engines fixed (session 2026-06-27/29)

### omnivoice — remote routing ✅ FIXED
- **Was:** `"Not available: pip install omnivoice needed"` when called via orchestrator
- **Root cause:** `OMNIVOICE_URL` env var was not set in the orchestrator container — only 7 of 28 `_URL` vars were configured. The orchestrator tried to load omnivoice locally (no ML libs).
- **Fix applied:** Added all 28 `{ENGINE}_URL` env vars to Makefile `deploy-orchestrator` target. Recreated orchestrator container.
- **Files:** `Makefile`, `docker-compose.yml`

### omnivoice — voice cloning (torchcodec) ✅ FIXED
- **Was:** `AttributeError: module 'torchcodec' has no attribute 'decoders'` when using `audio_prompt_id`
- **Root cause:** `torchcodec` v99.0.0 is a **dummy stub** (`class AudioDecoder: pass`) installed to satisfy `f5-tts`'s pip dependency. When OmniVoice transcribes reference audio via transformers' ASR pipeline, `isinstance(inputs, torchcodec.decoders.AudioDecoder)` fails at runtime even though the attribute exists when checked directly.
- **Fix applied:** Monkey-patched `is_torchcodec_available` → `False` in `tts_lab_shims.py`. This forces the ASR pipeline to use its default preprocessing path.
- **Files:** `tts_lab_shims.py`
- **Note:** This is a monkey patch. Proper fix: isolate f5-tts and omnivoice into separate containers so the dummy torchcodec isn't needed, or install a real torchcodec version.

### LLM-TTS VRAM coordination ✅ FIXED
- **Was:** Heavy TTS engines (omnivoice, etc.) hit CUDA OOM because the LLM was using ~13.2 GB VRAM. No mechanism existed to evict the LLM before heavy TTS synthesis.
- **Root cause:** The LLM→TTS eviction protocol (`_evict_all_tts_engines()` before LLM inference) existed, but the reverse (TTS→LLM) was missing.
- **Fix applied:** Mounted Docker socket in orchestrator. Added `_stop_llm_container()` / `_start_llm_container()` in `tts_lab_dispatch.py`. Heavy TTS engines auto-stop the LLM container; LLM requests auto-restart it.
- **Files:** `tts_lab_dispatch.py`, `Makefile`, `docker-compose.yml`

---

## Engines fixed (as of 2026-04-25 session 2)

### indextts — ✅ FIXED
- **Was:** `AttributeError: 'IndexTTS2' object has no attribute 'load_model'`
- **Fix applied:** Removed `model.load_model()` from `_load_indextts()` in `tts_lab_engines.py`
- **Verified:** `IndexTTS2` has no `load_model` method; `__init__` loads all weights

### qwen3tts — ✅ FIXED (shim hardened)
- **Was:** `'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'`
- **Fix applied:** Added class-level shim in `tts_lab_shims.py` — sets `_attn_implementation_autoset = False` on `Qwen3TTSSpeakerEncoderConfig` at startup
- **Root cause:** Config `__init__` never calls `super().__init__()`, so `PretrainedConfig` never sets the attribute

---

## Open Issues

| Engine | Symptom | Notes |
|---|---|---|
| dia | Hangs >180s | Pre-existing. Engine loads but synthesis never completes. Not caused by 2026-06-27 changes. |
| styletts2 | Hangs >180s | Pre-existing. Same symptom as dia. Not caused by 2026-06-27 changes. |
| cosyvoice | Not installed | Needs `git clone FunAudioLLM/CosyVoice /opt/CosyVoice` |
| manatts | Not installed | Needs `pip install parallel-wavegan` |
| openvoice | Not installed | Needs `pip install openvoice` |
| neutts | Not configured | Needs `_load_neutts()` implementation in `tts_lab_engines.py` |
| orpheus | Gated HF model | Needs `huggingface-cli login` |
| csm | Gated HF model | Needs `huggingface-cli login` |
| parler | Version-gated | Requires `transformers==4.46.1` in engine-legacy container |
| higgs/vibevoice/s2pro | Containers not running | SGLang-based engines — `docker compose --profile sglang up -d` |
| indextts/parler | engine-legacy not running | `docker compose --profile legacy up -d` |

## Infra notes
- transformers pinning: pinned to `4.53.2` in requirements.txt
- All site-packages patches re-applied on every deploy via step 4.5 in `deploy_tts_lab.ps1`
- VM root disk expanded to 650 GB — no storage pressure
- E2E test script: `e2e_test.ps1` — run after every deploy to verify all fixes hold
- **Docker socket mounted in orchestrator** — enables LLM container start/stop for VRAM coordination
- **torchcodec v99.0.0 is a dummy stub** — do NOT upgrade it; if removed, f5-tts breaks. If kept as-is, ASR pipelines crash. The shim (`is_torchcodec_available → False`) bridges the gap.
