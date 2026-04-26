# Arthur TTS Lab — Known Issues & Next Steps

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

## Engines not yet installed
| Engine | Package | Notes |
|---|---|---|
| csm | sesame/csm-1b | Gated HF model — needs `huggingface-cli login` |
| orpheus | `pip install orpheus-speech` ✅ installed | Requires CUDA GPU — vllm crashes on CPU-only VM (guarded by `_require_gpu`) |
| neutts | unknown | Need to identify correct package |

## Infra notes
- transformers pinning: pinned to `4.53.2` in requirements.txt
- All site-packages patches re-applied on every deploy via step 4.5 in `deploy_tts_lab.ps1`
- VM root disk expanded to 650 GB — no storage pressure
- E2E test script: `e2e_test.ps1` — run after every deploy to verify all fixes hold
