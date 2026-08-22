# Chatterbox-Turbo: "en-leo2" IS the built-in voice + expressiveness knobs are dead

**Date:** 2026-08-22
**Reported by:** Remote caller (SpamBlocker `TTSProvider.ps1` → `Invoke-ChatterBoxTurboTTS`) — "voice is fine but robotic"
**Engine:** `chatterboxturbo` on engine-current (8101) · `chatterbox-tts==0.1.7`

## Symptom

Caller synthesizes with `audio_prompt_id: "en-leo2"`, `exaggeration: 0.5`, `cfg_weight: 0.5`
and gets flat, robotic output that never changes character no matter how the params are tuned.

## Root causes (two independent)

### 1. `exaggeration` / `cfg_weight` / `min_p` are ignored by the Turbo decoder

The distilled one-step `ChatterboxTurboTTS` has **no CFG and no emotion control at inference**.
Upstream `chatterbox-tts` 0.1.7 (`tts_turbo.py`) logs
`CFG, min_p and exaggeration are not supported by Turbo version and will be ignored`
for any non-zero value. The lab forwarded these params to `generate()` without surfacing
the warning — callers were tuning dead knobs. These knobs only work on the base
`chatterbox` engine (`exaggeration` → `emotion_adv`, real CFG).

**Fix (lab):** `_synth_chatterboxturbo` now emits a visible `PARAMS` ring-log warning when any
of the three knobs is passed non-zero (commit `833d15a`), and the API reference documents
which levers actually work on Turbo.

### 2. `en-leo2.wav` produces the model's BUILT-IN voice, byte-for-byte

A/B synthesis (fixed text, seed 42, long + short variants) showed:

| reference | md5 | verdict |
|---|---|---|
| (none) | `9b1448f3…` | builtin conds |
| `en-zzz` (nonexistent) | `9b1448f3…` | builtin conds |
| **`en-leo2`** | **`9b1448f3…`** | **== builtin conds** |
| `en-brian` | `68117c3e…` | genuine clone |
| `en-chris` / `en-kebin` / `en-leo` / `en-william` | distinct | genuine clones |

Verified in-process: the `Reference mel length…` warning fires for `en-leo2` (the ref IS
loaded), yet output is byte-identical to no-ref. The turbo checkpoint ships `conds.pt`
(`t3.speaker_emb (1,256)`, `gen.prompt_feat (1,500,80)`, …) and `en-leo2.wav` derives
exactly those conditionals — i.e. the curated `en-leo2` clip **is** the stock voice's source.
Cloning it is a no-op; the caller has been hearing the untuned default voice all along.

## What actually controls Turbo naturalness (in order)

1. **Reference clip delivery** — Turbo transfers prosody/emotion from the prompt WAV
   (speaker emb + speech-token prompt + S3Gen ref). A flat clip → flat output.
   Avoid `en-leo2` (stock voice); pick `en-brian`/`en-william`/… or a new curated clip.
2. **Punctuation in the text** (`.?!,`) — drives pause/contour; the caller's cleanup must
   not flatten sentence boundaries. Model's `punc_norm` appends a trailing period.
3. **Digits** — caller converts digits to Persian ۰–۹; the English-only 704-token BPE vocab
   can't pronounce them → glitches. Word-out numbers for English voices.
4. **Chunk seams** — >150 chars splits with 350 ms gaps (default `chunk_silence_ms`);
   long narration gets audible pauses.
5. **`temperature`/`top_p`/`repetition_penalty`** — honored (defaults 0.8 / 0.95 / 1.2).

## Options for the caller

- **Stay on Turbo:** switch `audio_prompt_id` to a genuine clone (e.g. `en-brian`,
  `en-william`), keep punctuation, word-out digits, optionally lower `chunk_silence_ms`.
  Fast path preserved (RTF ~1.1×).
- **Switch to base `chatterbox`:** `exaggeration`/`cfg_weight` become real
  (`exaggeration` 0.65–0.8 is the lever). Costs ~3 GB VRAM, RTF ~2.4×.

## Sample audio

A/B WAVs generated 2026-08-22 (orchestrator path + direct engine path) were downloaded
to the dev machine for listening comparison — `en-leo2` vs `en-brian` etc. on identical text.
