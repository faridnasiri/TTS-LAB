# Handoff — S1-Mini engine + quantization feasibility (2026-08-23)

> **Status: REMOVED 2026-08-23** — the `s1mini` engine was **deleted** by user
> decision ("can we get rid of s1 mini? i do not want it, it does not worth
> efforts"). This doc is kept as the record of what was tried and what was
> learned. The sections below describe the *implemented* state; see
> [Removal](#-removed-2026-08-23) at the bottom for what changed.

## TL;DR — verdicts on the four ideas

| Idea from request | Verdict | Why |
|---|---|---|
| **OpenAudio S1-Mini (0.5B)** — light Fish model | ❌ **REMOVED** (was briefly implemented as `s1mini` engine) | Produced noise-only output — root cause found (missing `<|begin_of_text|>`), fix was one line, but the user chose removal over fixing. |
| **Fish Speech 1.5 (~500M)** | ✅ Already in the lab | The `fishspeech` engine (fishaudio/fish-speech-1.5, ~1.1 GB). |
| **S2-Pro INT4/AWQ/GGUF** (14 GB → 5-6 GB) | ❌ **Not feasible** | sgl-omni 0.1.3's s2-pro pipeline has no quantization support — no `--quantization` flag, no quantized request param, no official quantized checkpoints; GGUF is only community llama.cpp ports sgl-omni cannot serve. |
| **EditX INT4/GPTQ/AWQ** (8 GB → 3-4 GB) | ✅ Already done | The lab runs Step-Audio-EditX **AWQ-4bit** already (that IS the quantization; ~12.8 GiB resident incl. vLLM + CosyVoice). |

## Root-cause record (why the output was noise)

**Diagnosis (2026-08-23, complete):** the s1-mini checkpoint (created
2025-05-31 on HF) was trained with a leading `<|begin_of_text|>` (id 151643)
token at sequence start, but the fish-speech **main**-branch conversation
builder (post-June-2025 refactor) omits it. Without BOS every position sits
off-by-one vs training → the model echoes the current token
(`pred(k) ≈ t_k`) → degenerate semantic-0 fixed point → all-1023 VQ codes →
noise.

Evidence: (1) teacher-forced next-token after the correct prompt = confident
semantic-0 (24.62 vs 21.62); (2) model cannot predict its own template text
(5–9.5% next-token acc); (3) per-position table shows the ECHO pattern
(off-by-one position shift); (4) **BOS prepend fixes it**: next-token acc
5.6%→42.1%, teacher-forced semantic argmax 659 (healthy spread) vs 0;
(5) fp32 vs bf16 identical → not precision; (6) tokenizer fully verified
(vocab 155776, rank-ordered tiktoken, roundtrip OK).

**The fix (never applied end-to-end, now moot):** prepend
`TextPart(text='<|begin_of_text|>', cal_loss=False)` at `parts[0]` of the
content sequence — BEFORE the auto-generated `im_start` part
(variant [BOS, im_start, ...] was the diagnostically-validated one).

**Why this checkpoint is the only one affected:** s2-pro (daa9b4f, 2026-03-10)
removed the BOS-era tokenizer constant; the refactor that dropped BOS from the
conversation came in June 2025 (89474bb) — after s1-mini's training data era.
The fishspeech v1.5.1 engine is a different pipeline and is unaffected.

## What was removed

- `tts_lab_engines.py` — `_load_s1mini()`/`_synth_s1mini()` + LOADERS/SYNTHERS entries.
- `tts_lab_config.py` — `S1MINI_REPO_DIR`/`S1MINI_MODEL_ID` consts, `MODEL_INFO["s1mini"]`, `MODEL_ORDER` entry, `SYNTH_TIMEOUT["s1mini"]`.
- `tts_lab_dispatch.py` — availability probe (checkout dir + gated-repo token check).
- `tts_lab_ui.py` — params panel + license/quality warnings.
- `docker-compose.yml` — `S1MINI_URL` env line.
- `patches/patch_fish_speech_s1.py` — deleted (the two checkout patches; the
  checkout itself was deleted so they are moot).
- VM: `/opt/models/fish-speech-s1` checkout deleted, `models--fishaudio--s1-mini`
  HF cache purged.
- `docs/engine_compatibility.yaml` — entry → `status: removed` with the BOS
  finding in notes; summary counts corrected (supported 19, experimental 6).

**Unaffected:** the `fishspeech` v1.5.1 engine (its own checkout at
`/opt/models/fish-speech`), s2pro, and `_purge_fish_speech_modules()` (kept —
still used defensively by the fishspeech loader).
