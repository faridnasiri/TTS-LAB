# Chatterbox-Turbo: dead expressiveness knobs + sticky-conditionals bug (en-leo2 IS a genuine clone)

**Date:** 2026-08-22
**Reported by:** Remote caller (SpamBlocker `TTSProvider.ps1` → `Invoke-ChatterBoxTurboTTS`) — "voice is fine but robotic"
**Engine:** `chatterboxturbo` on engine-current (8101) · `chatterbox-tts==0.1.7`

> **Correction (2026-08-22, same day):** the first version of this doc claimed
> `en-leo2.wav` *is* the model's built-in voice. That was wrong — an artifact of
> the sticky-conditionals bug below. `en-leo2` is a genuine, distinct clone voice.
> Root cause 2 was rewritten after tensor-level and byte-level proof.

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

**Fix (lab):** `_synth_chatterboxturbo` emits a visible `PARAMS` ring-log warning when any
of the three knobs is passed non-zero (commit `833d15a`), and the API reference documents
which levers actually work on Turbo.

### 2. Sticky conditionals — "no reference" inherited the previous caller's voice (FIXED)

`generate()` has no path for "use the built-in voice": when `audio_prompt_path` is absent it
silently **reuses `self.conds`** — the conditionals of the *previous* request. The lab resolved
`audio_prompt_id` but never reset state, so:

- a request with **no** `audio_prompt_id` (or an unresolvable one) spoke in the **previous caller's voice**;
- the built-in voice was only ever heard on a fresh instance.

This bug contaminated A/B testing: "no-ref" runs executed after a ref call produced that ref's
audio, so `en-leo2` (run early) appeared byte-identical to "no-ref" — the false "en-leo2 == builtin"
conclusion. Reproduced with the fix in place: adversarial order (`en-brian` → no-ref → `en-zzz`)
now yields `b6292dfb` → `b8a24cf7` → `b8a24cf7`; before the fix the last two were both `b6292dfb`.

**Fix (lab, commit `b24d3c1`):** loaders stash the checkpoint's pristine `conds` as
`inst._builtin_conds` (`_stash_builtin_conds`); synths call `_reset_engine_conds` whenever a
request has no resolvable reference, and log a `REF` line with the resolved path (or the fallback).
Applies to `chatterboxturbo` **and** base `chatterbox`.

### `en-leo2` is a genuine clone — proof

| evidence | result |
|---|---|
| Fresh `prepare_conditionals(en-leo2.wav)` vs checkpoint `conds.pt` (tensor-by-tensor) | **differs in every field** — `speaker_emb` maxdiff 0.205, `cond_prompt_speech_tokens` (1,226) vs (1,375), `prompt_feat` (1,450,80) vs (1,500,80), `embedding` maxdiff 3.65 |
| In-process + live A/B, seed 42, same text | `en-leo2` = `2bfe7497…`, no-ref = `b8a24cf7…`, `en-brian` = `b6292dfb…` — three distinct outputs, all deterministic and order-independent post-fix |
| File integrity | deployed `/opt/arthur/reference_voices/en-leo2.wav` md5 `fcbfff16…` == locally staged copy; 9.0 s healthy speech (RMS 2325) |

The `en-leo2` clip is a legitimate ElevenLabs-preview voice (`en-leo2.json` sidecar). Its
flat, low-energy delivery is what transfers to the caller's output — the robotic feel is the
*reference clip's prosody*, not a fallback to a stock voice.

## What actually controls Turbo naturalness (in order)

1. **Reference clip delivery** — Turbo transfers prosody/emotion from the prompt WAV
   (speaker emb + speech-token prompt + S3Gen ref). A flat clip → flat output.
   `en-leo2` is the flattest of the six; `en-brian`/`en-william`/`en-kebin` have more energy.
2. **Punctuation in the text** (`.?!,`) — drives pause/contour; the caller's cleanup must
   not flatten sentence boundaries. Model's `punc_norm` appends a trailing period.
3. **Digits** — caller converts digits to Persian ۰–۹; the English-only 704-token BPE vocab
   can't pronounce them → glitches. Word-out numbers for English voices.
4. **Chunk seams** — >150 chars splits with 350 ms gaps (default `chunk_silence_ms`);
   long narration gets audible pauses.
5. **`temperature`/`top_p`/`repetition_penalty`** — honored (defaults 0.8 / 0.95 / 1.2).

## Options for the caller

- **Stay on Turbo:** swap `en-leo2` → `en-brian` or `en-william` (more energetic delivery),
  keep punctuation, word-out digits, optionally lower `chunk_silence_ms`.
  Fast path preserved (RTF ~1.1×).
- **Switch to base `chatterbox`:** `exaggeration`/`cfg_weight` become real
  (`exaggeration` 0.65–0.8 is the lever). Costs ~3 GB VRAM, RTF ~2.4×.

## Sample audio

Clean A/B WAVs (post-fix, same text, seed 42) at `C:\Users\farid\AppData\Local\Temp\tts_abtest2\`
on the dev machine: `builtin.wav`, `en-leo2.wav`, `en-brian.wav`, `en-william.wav`,
`en-chris.wav`, `en-kebin.wav`, `en-leo.wav`. (The earlier `tts_abtest\` batch is stale —
generated under the sticky-conds bug — and should not be used for voice comparisons.)
