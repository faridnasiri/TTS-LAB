# Persian/Farsi TTS in Arthur TTS Lab — Comprehensive Reference

> Date: 2026-06-12  
> Summary of all Persian TTS engines evaluated, integrated, and debugged.

---

## 1. Engine Summary

| Engine | Quality (Persian) | Speed (RTF) | Voices | Status |
|---|---|---|---|---|
| **Chatterbox (Persian fine-tune)** | Best | 3.0× | Cloning via ref WAV | ✅ Working |
| **Matcha-TTS** | Good | 0.27× | Khadijah (F), Musa (M) | ✅ Working |
| **ManaTTS (Tacotron2)** | Decent | 1.6× (GPU) | Single female (ref WAV) | ✅ Working |
| **Mana-Persian-Piper** | Light | 0.23× | Single female | ✅ Working |
| **Chatterbox (base)** | Poor for FA | 3.0× | Cloning via ref WAV | ⚠️ English-focused |
| **Chatterbox (23lang v2)** | Urdu-like | 3.6× | — | ❌ Removed |
| **Chatterbox (mtl23ls v2)** | Chinese-like | — | — | ❌ Removed |
| **Fish Speech** | Works | 152s (CPU) | — | ❌ Too slow |

---

## 2. Chatterbox — Persian Fine-Tune

### Model
- **Source:** `hootan09/ChatterBox` on HuggingFace (duplicate of `ResembleAI/chatterbox` + Persian additions)
- **Fine-tune:** `Thomcles/Chatterbox-TTS-Persian-Farsi` — gated repo, requires access request
- **Architecture:** 0.5B Llama backbone + T3 text-to-speech module + S3Gen vocoder + Perth watermarker
- **Persian T3 weights:** `t3_fa.safetensors` — 2454-token vocabulary, fine-tuned on Persian data
- **Tokenizer:** `mtl_tokenizer.json` (2352-token subword tokenizer)
- **Sample rate:** 24000 Hz

### Loading Approach
The base Chatterbox model creates T3 with `T3Config.english_only()` (704 tokens). The Persian fine-tune has 2454 tokens. We:

1. Load base model on **CPU** (saves VRAM)
2. Create a **new T3** with `T3Config.multilingual()` (2454 tokens)
3. Load `t3_fa.safetensors` weights with `strict=True`
4. Replace `inst.t3`, `inst.tokenizer`
5. Move components to GPU

```python
# Key: must create T3 with matching vocab size before loading weights
t3_new = T3(hp=T3Config.multilingual())  # 2454 tokens
t3_state = load_file("t3_fa.safetensors")
t3_new.load_state_dict(t3_state, strict=True)  # full match
inst.t3 = t3_new
inst.tokenizer = EnTokenizer("mtl_tokenizer.json")  # subword, 2352 tokens
```

### Issues Solved
1. **Gated repo (401)** — HF token on VM was invalid. Fixed by uploading a fresh token to `/tmp/hf_token.txt`.
2. **Vocab size mismatch** — T3 created with 704 tokens, Persian weights have 2454. `strict=False` silently skipped embedding layers → tokenizer produced OOB IDs → CUDA assert. Fixed by creating T3 with `T3Config.multilingual()`.
3. **Tokenizer mismatch** — Initially used `grapheme_mtl_merged_expanded_v1.json` (character-level, 2454 tokens). Persian fine-tune was trained with subword tokenizer `mtl_tokenizer.json` (2352 tokens). Using wrong tokenizer produced garbled output.
4. **CUDA OOM** — Loading base model on GPU + Persian T3 on GPU simultaneously exhausted VRAM (another process held 10.35GB). Fixed by loading base on CPU first, then moving only needed components to GPU.
5. **SDPA attention + output_attentions** — `AlignmentStreamAnalyzer` requires `output_attentions=True` which SDPA doesn't support. Fixed via shim in `tts_lab_shims.py` that forces `_attn_implementation = "eager"` before attention spy creation.
	   - **Regression (2026-06-12):** An "optimization" added a `finally` block that restored `_attn_implementation` to its previous value (SDPA) after hook *registration*. But the spy hooks fire during the model *forward pass*, not during registration. So the config was back to `"sdpa"` by the time the model ran → SDPA returned `None` for attention weights → spy hooks captured `None` → `torch.stack(self.last_aligned_attns)` crashed with `TypeError: expected Tensor as element 0 in argument 0, but got NoneType`. Fix: removed the `finally` restore — eager attention must persist for the entire lifetime of the spy-hooked layers. See `tts_lab_shims.py` lines 366-385.
6. **`scipy.signal.kaiser` removed** — scipy 1.16 removed this function, breaking `parallel_wavegan`. Fixed by aliasing `scipy.signal.windows.kaiser` back to `scipy.signal.kaiser` in shims.

### Multilingual Variants (Failed)
- **`t3_23lang.safetensors`** (2352 tokens): Sounded like Urdu for Persian
- **`t3_mtl23ls_v2.safetensors`** (2454 tokens): Sounded Chinese — needs grapheme tokenizer, trained on different languages

**Conclusion:** Only the dedicated Persian fine-tune (`t3_fa.safetensors`) produces acceptable Persian speech.

### Diacritics
The tokenizer supports all Persian vowel marks (ًٌٍَُِّْ). The model *can* accept diacritized input (e.g., `سَلام` instead of `سلام`), but whether it helps depends on if the training data included diacritized text. Needs empirical testing.

### Text Processing Providers (2026-06-12)

A `use_g2p` dropdown in the Chatterbox and ManaTTS panels offers four text processing providers. The selected provider runs before synthesis; a live preview below the text input shows the processed output.

| Provider | Adds vowel marks? | What it does |
|---|---|---|
| **persian-phonemizer** | ✅ Yes | G2P via Moin dictionary + seq2seq neural. Normalizes Arabic→Persian chars internally. |
| **hazm** | ❌ No | Normalizes Arabic→Persian chars, fixes ZWNJ spacing, preserves Persian digits |
| **parsivar** | ❌ No | Same as hazm + converts Persian digits to ASCII (۱۲۳۴→1234), date normalization |
| **none** | ❌ No | Raw text, no processing |

Only `persian-phonemizer` adds vowel marks — it is the only Persian G2P library available. Hazm and Parsivar are text normalizers, not G2P engines; they clean text structure but don't add diacritics. They are useful for comparing normalized vs raw text output.

**Chaining bug (2026-06-12):** An attempt to chain normalization + G2P (`hazm → G2P`, `parsivar → G2P`) made all three providers produce identical output because `persian-phonemizer` already uses hazm internally for normalization. The aliasing `hazm→hazm_g2p` routed through the same phonemizer, producing the same diacritized text. Reverted to standalone behavior — each provider now produces distinctly different output.

**Text preview:** A `/preview-text` endpoint (`tts_lab.py`) returns the processed text for the selected provider. JavaScript in the UI calls it on text input change (400ms debounce) and dropdown change. The preview div (`#text-preview`) shows the processed text in monospace with `color:#d4d4d4` on dark background.

**Key files:**
- `tts_lab_engines.py:1031-1100` — `_process_persian_text()` dispatch function with lazy-init cache
- `tts_lab.py:210-223` — `/preview-text` endpoint
- `tts_lab_ui.py:1238-1247` — preview provider dropdown + preview div
- `tts_lab_ui.py:768-800` — JavaScript preview logic (debounce, fetch, sync dropdowns)

### Length Limitations
- **`max_text_tokens = 2048`** — baked into checkpoint weights (position embeddings shape `[2050, 1024]`). Cannot be increased without retraining.
- With diacritics, each Persian character becomes 2+ tokens (`بِ` → `ب` + `ِ`), so effective limit is **~1000-1500 characters**.
- Without diacritics, limit is **~2000 characters** (roughly 5-10 seconds of audio).
- **`max_new_tokens`** — speech output tokens, overridable via the Max Length slider (default 20,000). At 25 speech tokens/second, this allows up to 800s of audio, but the model usually emits EOS well before that.
- **Workaround:** Split long text into sentences and synthesize separately.
- **Crash guard:** Text exceeding 2048 tokens is auto-truncated instead of failing.

---

## 3. Matcha-TTS

### Model
- **Architecture:** Flow-matching ONNX via `sherpa-onnx`
- **Persian models:** `csukuangfj/matcha-tts-fa_en-khadijah` (female) and `csukuangfj/matcha-tts-fa_en-musa` (male)
- **Vocoder:** `csukuangfj/sherpa-onnx-hifigan` → `hifigan_v2.onnx` (3.75 MB)
- **Sample rate:** 22050 Hz
- **Parameters:** `voice` (khadijah/musa), `speed` (0.5-2.0), `temperature` (0-2.0)

### Loading
```python
import sherpa_onnx
matcha_cfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
    acoustic_model=model_path,
    vocoder=vocoder_path,
    tokens=tokens_path,
    data_dir=data_dir,      # bundled espeak-ng-data
    noise_scale=0.333,
    length_scale=1.0,
)
tts = sherpa_onnx.OfflineTts(tts_config)
result = tts.generate(text, sid=0, speed=speed)
# result.samples, result.sample_rate
```

### Key Points
- Handles phonemization internally via espeak-ng (voice="fa")
- Fastest Persian engine (RTF 0.27×)
- Temperature change forces model reload (noise_scale is fixed at construction)
- Voice switching triggers reload (detected in `_ensure_loaded`)

---

## 4. ManaTTS — Tacotron2

### Model
- **Architecture:** Tacotron v1 (SV2TTS pipeline) + HiFi-GAN VCTK v1 vocoder
- **Source:** `MahtaFetrat/Persian-Tacotron2-on-ManaTTS` (synthesizer.pt, 371 MB)
- **Implementation repo:** `MahtaFetrat/Persian-MultiSpeaker-Tacotron2` (cloned to `/opt/models/`)
- **Encoder:** GE2E speaker encoder (`encoder.pt`, from cloned repo)
- **Vocoder:** HiFi-GAN VCTK v1, downloaded separately (916 MB)
- **Sample rate:** 24000 Hz
- **MOS:** 3.76 (from paper)

### Loading
```python
from encoder import inference as encoder_mod
from synthesizer.inference import Synthesizer

encoder_mod.load_model("encoder.pt")
synthesizer = Synthesizer("synthesizer.pt")
vocoder = load_pwg("vocoder_HiFiGAN.pkl")  # parallel_wavegan
```

### Issues Solved
1. **Missing HiFi-GAN vocoder** — Only `synthesizer.pt` in HF snapshot. Downloaded 916MB vocoder separately via `parallel_wavegan.utils.download_pretrained_model("vctk_hifigan.v1")`.
2. **Missing `config.yml`** — `parallel_wavegan.load_model()` requires config alongside checkpoint. Copied from downloaded vocoder archive.
3. **`scipy.signal.kaiser` removed** — Same fix as Chatterbox (shim).
4. **Character vocabulary KeyError** — Persian digits (۰۱۲۳۴۵۶۷۸۹) not in Tacotron's symbol set. Added `_normalize_persian_text()` to convert digits and Arabic characters before synthesis.
5. **Robotic output** — Caused by Griffin-Lim fallback when HiFi-GAN failed to load silently. Fixed by ensuring vocoder loads correctly.

### Text Normalization
```python
def _normalize_persian_text(text):
    persian_digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    text = text.translate(persian_digits)
    arabic_to_persian = str.maketrans({"ك":"ک", "ي":"ی", "ة":"ه", "ؤ":"و", "أ":"ا", "إ":"ا", "ئ":"ی", "ء":""})
    text = text.translate(arabic_to_persian)
    # Strip chars outside model's symbol set
    return "".join(c for c in text if c in allowed_symbols or c.isspace())
```

---

## 5. Mana-Persian-Piper

### Model
- **Architecture:** Piper ONNX (medium)
- **Source:** `MahtaFetrat/Mana-Persian-Piper` on HuggingFace
- **Files:** `fa_IR-mana-medium.onnx` (60 MB) + `.onnx.json`
- **Sample rate:** 22050 Hz
- **Installation:** Download ONNX + JSON to `/opt/arthur/models/`, auto-discovered by existing Piper engine

### Key Point
- Leverages the existing Piper engine — no new loader/synth needed
- Just drop the ONNX files in the models directory and the voice appears in the Piper dropdown
- Lightweight, CPU-friendly, but pronunciation limited by Piper architecture

---

## 6. Infrastructure Fixes

### Transformers 4.57.6 Compatibility

| Issue | Fix | Where |
|---|---|---|
| `GeneralInterface(MutableMapping)` truncated to empty stub | Rewrote full class with 9 methods | `fix_transformers_shims.py` |
| `LlamaModel.forward()` no longer collects `hidden_states` | Monkey-patch to populate when None | `tts_lab_shims.py` |
| `AttentionInterface` missing `valid_keys`, `__getitem__`, `register` | Restored on `GeneralInterface` base class | `tts_lab_shims.py` |
| `AttentionMaskInterface` not subscriptable | Same `GeneralInterface` fix covers it | `tts_lab_shims.py` |
| `scipy.signal.kaiser` removed in scipy 1.16 | Alias from `scipy.signal.windows` | `tts_lab_shims.py` |
| `AlignmentStreamAnalyzer` + SDPA incompatibility | Force eager attention before spy | `tts_lab_shims.py` |

### Disk Space
- Deleted stale 32GB swap file (`/opt/models/swap.img`) and 4.4GB pip cache
- Freed 36GB on `/opt/models` partition

---

## 7. Files Modified

| File | Changes |
|---|---|
| `tts_lab_config.py` | Added MATCHA_VOICES, MATCHA_MODEL_REPOS, MANATTS_REPO_DIR, MODEL_INFO entries, MODEL_ORDER, SYNTH_TIMEOUT |
| `tts_lab_dispatch.py` | Added pkg_map entries, availability checks, matcha voice/temp reload, chatterbox model reload |
| `tts_lab_engines.py` | Added `_load_matcha`, `_synth_matcha`, `_load_manatts`, `_synth_manatts`, updated `_load_chatterbox`, `_split_persian_text`, `_normalize_persian_text`, LOADERS/SYNTHERS |
| `tts_lab_shims.py` | Added `GeneralInterface` MutableMapping shim, `LlamaModel.forward` hidden_states fix, scipy.kaiser alias, Chatterbox SDPA attention fix |
| `tts_lab_ui.py` | Added matcha voice/speed/temp controls, manatts ref WAV dropdown + upload, chatterbox model selector |
| `tts_lab.py` | Added `/refs` endpoint, MATCHA_VOICES import, voices entries, manatts SYNTH_TIMEOUT |
| `fix_transformers_shims.py` | Rewrote `GeneralInterface` stub as full `MutableMapping` implementation |

---

## 8. Model File Locations on VM

```
/opt/arthur/models/
├── fa_IR-mana-medium.onnx          # Mana-Persian-Piper
├── fa_IR-mana-medium.onnx.json

/opt/models/
├── Persian-MultiSpeaker-Tacotron2/  # ManaTTS implementation repo
│   └── saved_models/final_models/
│       ├── encoder.pt
│       ├── synthesizer.pt           # 371 MB
│       ├── vocoder_HiFiGAN.pkl     # 916 MB
│       └── config.yml
├── manatts-vocoder/                 # HiFi-GAN download
│   └── vctk_hifigan.v1/

~/.cache/huggingface/hub/
├── models--csukuangfj--matcha-tts-fa_en-khadijah/  # 74 MB ONNX
├── models--csukuangfj--matcha-tts-fa_en-musa/      # 74 MB ONNX
├── models--csukuangfj--sherpa-onnx-hifigan/        # 3.75 MB
├── models--hootan09--ChatterBox/     # t3_fa.safetensors + tokenizers
├── models--MahtaFetrat--Persian-Tacotron2-on-ManaTTS/
└── models--MahtaFetrat--Mana-Persian-Piper/
```

---

## 9. Voice Cloning (Chatterbox)

The Persian fine-tune supports zero-shot voice cloning via reference WAV. To add a new voice:

1. Record a **clean 5-15 second** WAV of the target speaker (minimal background noise, 16-24 kHz mono)
2. In the Chatterbox panel, use the "Voice cloning reference WAV" upload button
3. Select the uploaded WAV — `audio_prompt_id` auto-populates
4. Generate — the output will clone the reference speaker's voice

Uploaded reference WAVs persist in `/tmp/tts_uploads/` and appear in the Reference WAV dropdown on subsequent visits.

---

## 10. Recommended Usage

For best Persian pronunciation, use in this order:

1. **Chatterbox (Persian fine-tune)** — best quality, supports voice cloning via ref WAV, ~3s synthesis
   - For long texts: split into sentences, synthesize each separately
   - For mispronounced words: try adding vowel marks (diacritics)
   - For custom voices: upload a reference WAV
2. **Matcha-TTS (Khadijah/Musa)** — fast, decent quality, proper phonemization, ~0.3s synthesis
3. **ManaTTS** — natural but slower, needs ref WAV, ~7s synthesis on GPU

If pronunciation is consistently wrong for specific words, try adding vowel marks (diacritics) in Chatterbox Persian mode.
