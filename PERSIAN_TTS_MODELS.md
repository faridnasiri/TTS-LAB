# Persian TTS Models — Findings & Fixes

## Overview
This document records technical findings, issues, and solutions related to Persian (Farsi) text-to-speech synthesis with the Chatterbox T3 model in TTS Lab.

---

## 1. آ Character Fix — Tokenizer Gap

**Symptom:** Chatterbox Persian fine-tune could not pronounce "آ" (ALEF MADDA — the long /ɒː/ vowel). It either produced silence, garbled output, or the wrong vowel.

**Root Cause:** The model's `text_emb` is [2454, 1024] (from `T3Config.multilingual()`), but the Persian fine-tune ships with an mTL tokenizer containing only 2352 BPE tokens. The character آ (U+0622) is missing from this tokenizer. Furthermore, the v3/multilingual tokenizer (2454 tokens) has آ at index 2356, but the model's embedding at that position corresponds to `€` (Euro sign) — a different character entirely.

**Failed Approaches:**
1. **Token injection** — Adding آ to the tokenizer and expanding the embedding matrix to [2454, 1024]. Failed because BPE merge rules fragment injected tokens without corresponding merge table updates.
2. **Simple replacement** — Mapping آ → ا (plain ALEF). Failed because plain alef defaults to the short /æ/ vowel (like Fatha), not the long /ɒː/.
3. **V3 tokenizer swap** — Using the v3 tokenizer directly. Failed because the Persian fine-tune was trained with a different token distribution; mismatched embedding positions produce garbage.

**Working Solution — Unicode Decomposition:**
```python
text = text.replace("آ", "آ")  # ALEF MADDA → ALEF + MADDAH ABOVE
text = text.replace("أ", "ا")   # ALEF HAMZA ABOVE → ALEF
text = text.replace("إ", "ا")   # ALEF HAMZA BELOW → ALEF
```

آ (U+0622) decomposes to ا (U+0627 ALEF) + ٓ (U+0653 MADDAH ABOVE). Both characters exist in the 2352-token mTL tokenizer. The MADDAH ABOVE combining diacritic explicitly marks the long /ɒː/ vowel, which the model's G2P (persian_phonemizer) correctly interprets.

Applied in: `tts_lab_engines.py:_synth_chatterbox()` — guarded by `inst._needs_persian_char_map = True` (set during Persian model load).

---

## 2. Long-Form Synthesis — Chunking & Early EOS

**Symptom:** Same Persian text produced wildly different output lengths across runs (sometimes ~3s, sometimes ~10s, rarely full coverage). Long texts (3+ minute) would silently truncate — second and third sentences ignored.

### Root Cause Analysis

#### A. Alignment Stream Analyzer Forces Premature EOS
Chatterbox has an internal "hallucination checker" in `alignment_stream_analyzer.py` (lines 166-175). It monitors generated speech tokens and forces an end-of-speech (EOS) token when it detects:
- `token_repetition`: The **same token generated 2× in a row** (very aggressive)
- `alignment_repetition`: Alignment pattern repeating
- `long_tail`: Generation running too long

The model was trained on **short utterances (5-8 seconds)**. For longer texts, the model eventually generates a repeated token → EOS forced → chunk silently truncated → part of the text never heard.

```python
# alignment_stream_analyzer.py lines 166-175
if cur_text_posn < S - 3 and S > 5:
    logits[..., self.eos_idx] = -2**15      # Suppress EOS while text remains

if long_tail or alignment_repetition or token_repetition:
    logits = -(2**15) * torch.ones_like(logits)  # Force EOS — kill generation
    logits[..., self.eos_idx] = 2**15
```

#### B. Non-Deterministic Sampling
The T3 inference loop always uses `torch.multinomial` (sampling), never greedy argmax — even when `do_sample=False` is passed. Key parameters:
- Default `temperature=0.8` → sampling from softened distribution
- `CUDNN.deterministic = False` → CUDA operations produce slightly different results each run
- `torch.manual_seed()` only sets CPU seed → GPU ops remain non-deterministic

This means the model takes a different path through the token space each run, sometimes hitting repetition early and sometimes later.

#### C. Chunks Too Large for Training Distribution
Old chunk size of 180 chars (pre-G2P) became ~270-350 chars after `persian_phonemizer` added combining diacritics. The model, trained on 5-8s clips (~40-80 pre-G2P chars), couldn't handle sequences this long — it would start repeating tokens.

### Fix Applied (2026-06-15)

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Persian chunk size | 180 chars | **80 chars** | Matches training distribution (~5-8s utterances) |
| Non-Persian chunk size | 180 chars | **150 chars** | English models handle longer sequences |
| Force-split threshold | Only 1 chunk > 400 | **Any chunk > 100** | Catches comma-split sub-parts still too long |
| `max_chunk` ceiling | 150% of chunk_size | **125%** of chunk_size | More aggressive splitting |
| Repetition penalty | 1.2 (model default) | **1.5** | Discourages token repetition → fewer EOS triggers |
| CUDA determinism | Not set | `cudnn.deterministic=True` | Deterministic forward pass when seed is set |
| CUDA seed | CPU only | `cuda.manual_seed_all()` | GPU RNG state set alongside CPU |
| Per-chunk seed | `manual_seed(seed+i)` | `manual_seed + cuda.manual_seed_all(seed+i)` | Full determinism for multi-chunk |

### Force-Split Safety Net
The `_split_for_tts()` function splits at sentence boundaries (`.!?۔！？`) and then at commas (`،,؛;`) for long sentences. However, comma-split sub-parts can still exceed the chunk limit. A post-processing pass now force-splits **any** chunk exceeding `_CHUNK_CHARS × 1.25`:

```python
_max_chunk = int(_CHUNK_CHARS * 1.25)
_clean: list[str] = []
for _ch in chunks:
    if len(_ch) > _max_chunk:
        for _j in range(0, len(_ch), _CHUNK_CHARS):
            _clean.append(_ch[_j:_j + _CHUNK_CHARS])
    else:
        _clean.append(_ch)
chunks = _clean
```

### Test Results (264-char Persian text, 5 runs each)

| Configuration | Seed | Sizes | Coverage |
|---------------|------|-------|----------|
| Old code (180 char chunks) | `1234` | 154KB / 3.2s | ~20% — most text dropped |
| Old code (180 char chunks) | Random | 228-525KB / 4.8-10.9s | Variable, never full |
| Intermediate (100 char chunks) | `1234` | 512KB / 10.7s | ~60% — still losing chunks |
| **Final (80 char chunks + fixes)** | **`1234`** | **2,298KB / 47.9s** | **100% — full text, consistent** |
| Final (80 char chunks + fixes) | Random | 1,560-2,889KB / 32-60s | 100% — always covers text |

**Key finding:** With a seed, output sizes are byte-identical across runs (same duration, same structure). MD5 hashes differ due to irreducible CUDA non-determinism in attention operations, but this is perceptually imperceptible.

### Recommendation
**Always use a seed parameter** for consistent output. In the UI, set seed to any non-zero number. Via API, pass `"seed": "1234"` in params.

---

## 3. Model Architecture Notes

### Chatterbox T3
- **Backbone:** 0.5B/1.0B Llama-style transformer
- **T3 module:** Text-to-speech token generation (autoregressive)
- **S3Gen vocoder:** Speech token → waveform
- **Persian fine-tune:** [`hootan09/ChatterBox`](https://huggingface.co/hootan09/ChatterBox) — `t3_fa.safetensors` + `mtl_tokenizer.json` (2352 BPE tokens)
- **max_speech_tokens:** 4096 (config default), but generate() hardcodes `max_new_tokens=1000`
- **start_speech_token:** 6561, **stop_speech_token:** 6562
- **Sample rate:** 24000 Hz (S3GEN_SR)

### GPU Utilization (15-20%)
Autoregressive generation is **memory-bandwidth-bound** — each step generates one token, requiring a full forward pass through the model. The GPU spends most time waiting for weights to transfer from VRAM to compute units. 15-20% utilization is **normal** for this architecture and cannot be improved without model-level changes (speculative decoding, batching, etc.).

### Reference Voice Training Data
- Model was trained on ~5-8 second utterances
- Reference voices should be **5-10 seconds** for best speaker embedding extraction
- MP3 voices work after conversion to WAV (import script handles this)
- Voice Library path: `/opt/arthur/voice_library/voices/{id}/sample.wav`

---

## 4. Persian G2P (Grapheme-to-Phoneme)

### persian_phonemizer
- Adds combining diacritics as separate tokens (~1.5-2× token expansion)
- Enables correct short vowel pronunciation (a/e/o)
- Must be applied BEFORE chunking to ensure accurate token counts
- Configurable via `use_g2p` parameter (default: `"persian_phonemizer"`)

### Character Mapping
The Persian fine-tune's mTL tokenizer is missing several Arabic-script characters. These are handled at the text level:

| Character | Name | Mapping | Unicode |
|-----------|------|---------|---------|
| آ | ALEF MADDA | ا + ٓ | U+0622 → U+0627 + U+0653 |
| أ | ALEF HAMZA ABOVE | ا | U+0623 → U+0627 |
| إ | ALEF HAMZA BELOW | ا | U+0625 → U+0627 |

---

## 5. Issues Solved

1. ✅ **آ pronunciation** — Unicode decomposition to ا + MADDAH ABOVE (2026-06-10)
2. ✅ **Deterministic output** — CUDA determinism flags + seed propagation (2026-06-15)
3. ✅ **Early EOS / truncated output** — Reduced chunk size + force-split safety net + increased repetition_penalty (2026-06-15)
4. ✅ **Long text chunking** — Sentence-boundary splitting with comma fallback, transparent to UI and API (2026-06-10)
5. ✅ **Chunk error resilience** — Failed chunks reported, survivors stitched; only raises if ALL chunks fail (2026-06-10)
6. ✅ **Fifteen ElevenLabs voices imported** — Voice Library + UPLOAD_DIR, playable via ▶ button in dropdowns (2026-06-14)
7. ✅ **▶ Play button** on Chatterbox and ManaTTS reference WAV dropdowns (2026-06-14)

---

## 6. Known Limitations

1. **Byte-identical output impossible** — CUDA attention operations are fundamentally non-deterministic even with all determinism flags set. Sizes match; hashes differ. Perceptually imperceptible.
2. **GPU at 15-20%** — Normal for autoregressive generation (memory-bandwidth-bound).
3. **Random mode variance** — Without a seed, output duration varies 32-60s for the same text. Use a seed for consistency.
4. **Single-threaded generation** — Chunks are synthesized sequentially (not parallel). Parallelism wouldn't help due to the memory-bandwidth bottleneck.
5. **HuggingFace bug report** — Drafted for hootan09 about missing characters in the Persian tokenizer; not yet submitted.

---

*Last updated: 2026-06-15*
