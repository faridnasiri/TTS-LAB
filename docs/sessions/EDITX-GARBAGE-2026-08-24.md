# EditX Garbage Voices — Diagnosis & Fix (2026-08-24)

> User report: **"editx tts produces garbage voices only"** — GPU row showed
> engine-editx 4.0 GiB (server) + 8.0 GiB (vLLM EngineCore), i.e. the model was
> loaded and serving; outputs were nonsense.

## TL;DR

Three independent causes, all verified with full token-stream dumps + whisper
round-trip:

| # | Cause | Fix |
|---|---|---|
| 1 | **Interleave rotation**: model prepends spurious vq06 token(s), shifting the `[02,02,06,06,06]` frame the vocoder parses positionally → every frame garbles | `tts.py _generate` de-rotation: drop the leading offset with best 5-chunk alignment + truncate to whole chunks (Dockerfile.engine-editx patch #3) |
| 2 | **No reliable EOS**: argmax loops forever on a near-silent 8-token attractor; model only stops when sampling randomly hits `<|EOT|>` | Defaults in `_synth_editx`: temp **0.5**, rep-pen **1.1**, text-scaled **max_tokens cap**, **always a seed** (crc32 of ref+text; UI 🎲 re-rolls) |
| 3 | **Persian unsupported**: model trained on EN/ZH/JA/KO only — tokenizer has the chars, the model can't speak them | Documented in UI + MODEL_INFO; Persian path stays with the other engines |

Verified: fox round-trip clean at ~90%+ per draw across two container restarts.

## The three failure modes (all real)

1. **Degenerate loop** — greedy (temp 0): 2791 tokens, ZERO text/EOS, only
   **8 distinct ids** (6 vq02 + 1 vq06), perfect `(2,2,6,6,6)` chunking → 67 s of
   near-silence/hum. The argmax attractor. Whisper: zero segments.
2. **Interleave rotation** — sampled runs with `(6,6,6,2,2)` chunks (stream
   shifted by 3, or by 1: `(6,2,2,6,6)`): real content, vocoder misparse →
   garbled speech. Whisper heard scrambled pseudo-Chinese even on English prompts.
3. **Content lottery + tail leak** — outputs randomly choose which prompt text to
   speak: user text (fox ✓), the system-template text, the **ref transcript**
   (echoed the zh ref once), or the role marker "assistant" (the dominant leak —
   the model keeps generating after the content and drifts to the last text
   token). Root cause: in training the system text, user text AND target audio all
   carry the SAME source_text — inference with differing texts is OOD. Same-text
   prompts did NOT fix it (model sometimes echoes the ref audio instead).

Sampler notes (this vLLM 0.26 nightly, verified 2026-08-24):
- Same seed + same process + interleaved requests → **bit-identical** (seed IS honored).
- Same seed + fresh process → **different** output (startup-time lottery).
- No seed (vLLM seed-0 path) → **near-silence every time** (4/4 measurements).
- Some seeds are deterministically cursed across processes (s100 → "Assistant
  Department…" in both processes tested); most are process-lottery.

## The fix (shipped)

1. **`docker/Dockerfile.engine-editx`** — third inline tts.py patch (after the
   BatchEncoding + sampling patches): de-rotation + whole-chunk truncation at the
   `_generate` choke point (covers clone AND edit paths). Anchor:
   `output_ids = torch.tensor(output_token_ids, dtype=torch.long)` (count-guarded).
2. **`tts_lab_engines.py` `_synth_editx`** — defaults:
   - `temperature` → 0.5 (0.7 drifts: 777 was garbage at 0.7, clean at 0.5)
   - `repetition_penalty` → 1.1 (breaks the loop attractor)
   - `max_tokens` → `max(80, int(len(text)*3.5)+20)` when unset/0 (leak-killer)
   - `seed` → `crc32(ref_path + ":" + text)` when unset/0 (never unseeded;
     reproducible; 🎲 re-rolls a bad draw)
3. **`tts_lab_ui.py`** — temp slider default 0.5; seed default 0 (= auto);
   max_tokens default 0 (= auto, range 0-2048); Persian-unsupported warning.
4. **`tts_lab_config.py`** — MODEL_INFO note: "NO Persian".

## max_tokens semantics (corrected)

`clone()` computes `max_tokens=max(1, max_tokens - len(token_ids))` — but
`token_ids` there is still the **BatchEncoding** (dict-like, `len()` = key count,
not token count), so the subtraction is a no-op: the request value lands in
`SamplingParams.max_tokens` **verbatim** (verified: 50→48, 100→98, 150→148
generated tokens). The old "total budget" comment was wrong — it caps *generated*
audio tokens.

## Verification

- zh-ref + fox, temp 0.5 + rep 1.1 + cap 150: **9/11 seeds clean** ("The quick
  brown fox jumps over the lazy dog.") incl. previously-cursed 777/7; s100 cursed
  in both processes; no-seed = silence.
- Fresh restart (process #4): 777/13/42 clean again, s100 cursed again.
- Long text: correct content, minor tail leak cut by the cap.
- Persian: 8/8 combos fail (en/zh/fa refs × fa/en targets × fa/en lang tags) →
  model limitation, not a bug.
- Engine log shows `INTERLEAVE FIX: dropping 3 leading token(s) (align 0.00 -> 1.00)`
  on the runs that needed it.

## Follow-up incident — 21:59 crash (post-deploy) ✅ HARDENED

The de-rotation fix shipped, then ~1 h later the container CRASHED mid-request:

```
ScatterGatherKernel.cu:203: Assertion `idx_dim >= 0 && idx_dim < index_size`
CUDA error: device-side assert triggered  (×5, cascading 500s)
onnxruntime::OnnxRuntimeException → terminate → process death → compose restart
```

**Chain:** a fully garbage draw (487 tokens with **text ids min=975 and ids up
to 74600** mixed into audio positions; align 0.10 → 0.14 even after the 4-token
drop) → flow decoder `upsample_encoder_v2.pre_lookahead_layer` indexes the
codebook with text ids → negative index → **device-side assert** → the assert
poisons the ENTIRE CUDA context → every later CUDA call (FunASR load, retries)
fails with the stale error → onnxruntime throws → abort → "Server disconnected".

**Fix (patch #3 v2, live + Dockerfile, byte-verified identical):** the block now
*validate-and-strips* instead of trusting the stream:

- after the best-offset drop, walk chunks from the start and keep the **longest
  fully-valid `[2,2,6,6,6]` prefix** (also strips the "assistant" tail leak for
  free — it was previously only capped);
- if the valid prefix is < 2 chunks → `raise RuntimeError("EditX bad draw: no
  valid interleave prefix … — retry with a different seed")` — raised BEFORE the
  tensor conversion/vocoder, so no CUDA is touched and the context stays clean;
  the orchestrator relays the message to the UI (🎲 retry).

A strict `align >= 1.0` gate would have rejected good draws with tail leaks; the
prefix-strip accepts them and decodes only the valid part.

## Remaining caveats

- ~8-10% of draws still land on wrong content (seed lottery). The UI 🎲 button
  (random seed) is the retry path — same request, new draw. No token-level
  classifier can detect the garbage cheaply (garbage and clean look statistically
  identical: distinct-id ratio, repetition, length all overlap).
- A genuinely garbage draw no longer crashes anything — it 500s with a clear
  message. Any future CUDA error still poisons the context until the container
  restarts (compose restarts it automatically).
- Persian remains unavailable in EditX by model design.

## Files

| File | Change |
|---|---|
| `docker/Dockerfile.engine-editx` | patch #3 — `_generate` interleave fix v2: de-rotation + validate-and-strip + bad-draw gate (post-crash hardening) |
| `tts_lab_engines.py` | `_synth_editx` sampling defaults + seed policy |
| `tts_lab_ui.py` | editx panel defaults + Persian warning |
| `tts_lab_config.py` | MODEL_INFO note |
| `docs/reference/KNOWN_ISSUES.md` | incident entry |

## Diagnostic artifacts (VM, /tmp)

`editx_patch_derotate.py` (clone-level patch — superseded by the _generate-level
image patch), `editx_patch_dump.py` (token dump), `editx_analyze.py`,
`editx_struct.py` (chunk classifier), `batt_*/st_*/t5_*/fix_*/mt_*/fc_*/fa_*/rng_*`
(batteries + dumps + wavs), `det_*.json` (determinism checks).
