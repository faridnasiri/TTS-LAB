# Arthur TTS Lab — Known Issues & Next Steps

## Engines requiring fixes (as of 2026-04-25)

### indextts — `IndexTTS2` has no `load_model()`
- **Error:** `AttributeError: 'IndexTTS2' object has no attribute 'load_model'`
- **Cause:** IndexTTS v2 renamed/removed the `load_model()` method; initialisation now happens in `__init__`
- **Fix needed in:** `tts_lab_engines.py` → `_load_indextts()` — remove the `model.load_model()` call
- **Investigate:** `python3 -c "from indextts.infer_v2 import IndexTTS2; help(IndexTTS2.__init__)"`

### qwen3tts — `_attn_implementation_autoset` missing on config
- **Error:** `'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'`
- **Cause:** transformers 4.53 sets this in `PretrainedConfig.__init__` at line 302, but qwen_tts's
  `Qwen3TTSSpeakerEncoderConfig` may be constructing its config without calling `super().__init__()` first.
- **Fix options:**
  1. Shim: add `_attn_implementation_autoset = False` as class attribute on the config class at startup
  2. Patch `qwen_tts/core/models/configuration_qwen3_tts.py` to set the attribute in `__init__`

## Engines not yet installed
| Engine | Package | Notes |
|---|---|---|
| csm | sesame/csm-1b | Gated HF model — needs `huggingface-cli login` |
| orpheus | `pip install orpheus-speech` | Should be straightforward |
| neutts | unknown | Need to identify correct package |

## Infra notes
- transformers pinning: consider pinning to `4.53.2` in requirements.txt to prevent future breakage
- All site-packages patches are re-applied on every deploy via step 4.5 in `deploy_tts_lab.ps1`
- VM root disk expanded to 650 GB — no storage pressure
