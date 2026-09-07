# Arthur Server — Session Summary
> Chat sessions: 2026-03-23 → 2026-09-07
> Branch: `main`

---

## Session 2026-09-07 — Image Lab: SANA 1.6B + Boogu-Image 0.1 Turbo added

Full write-up: [SESSION_2026-09-07_IMGLAB_SANA_BOOGU.md](SESSION_2026-09-07_IMGLAB_SANA_BOOGU.md).
Landscape survey: [MODEL_LANDSCAPE_2026-09-06.md](../image-lab/MODEL_LANDSCAPE_2026-09-06.md).
Two new engines on port 8002 (`sana`: Sprint 1–4-step + 1.5 ~20-step variants on the
`quant` field; `boogu`: fp8, the sole CPU-offload exception), all verified live.
Found + fixed a diffusers SCM blocker (SanaSprintPipeline's `intermediate_timesteps=1.3`
default is only legal at 2 steps → pass `None` for steps ≠ 2). Disk incident: the root
disk is ONE 630 GB volume (not separate mounts) and hit 100% during the Boogu download —
freed by removing a stale 32 GB unreferenced cache entry + `docker builder prune -af`
(87 GB), then deleting the 69 GB legacy `hub/` orphan cache → 133 GB free. Gates
recalibrated to measured peaks (sana 11,500 / boogu 13,200 MiB); VRAM/RAM verdicts in
the session doc. Commits: (this change).

## Session 2026-08-24 (late) — EditX garbage voices FIXED + two container-crash hardenings

Full write-up: [EDITX-GARBAGE-2026-08-24.md](EDITX-GARBAGE-2026-08-24.md). Commits `622a039`, `436bddc`, `23ca9e4` (all local, unpushed).

### Garbage voices — three independent root causes (verified with token dumps + whisper round-trip)

| # | Cause | Fix |
|---|---|---|
| 1 | **Interleave rotation** — the model prepends spurious vq06 token(s), shifting the `[02,02,06,06,06]` frame the CosyVoice vocoder parses positionally → every frame garbles | `tts.py _generate` de-rotation (Dockerfile.engine-editx patch #3): drop the leading offset with best whole-chunk alignment, then decode only the longest fully-valid chunk prefix |
| 2 | **No reliable EOS** — greedy argmax loops forever on a near-silent 8-token attractor (67 s ramble); unseeded (vLLM seed-0) requests always land on it | `_synth_editx` defaults: temp **0.5**, rep-pen **1.1**, text-scaled **max_tokens cap**, seed = **crc32(ref+text)** (🎲 re-rolls a bad draw) |
| 3 | **Persian unsupported** — trained EN/ZH/JA/KO only | Documented in UI + MODEL_INFO; Persian path stays with the other engines |

### Two crashes same day → two hardenings

1. **21:59 death ("Server disconnected")** — a fully garbage draw (text ids in audio slots) hit the flow decoder's `ScatterGatherKernel.cu:203` **device-side assert** → poisoned the entire CUDA context → cascading 500s → onnxruntime terminate → restart. Fixed: the gate **validate-and-strips** before any CUDA — <2 valid chunks raises a clean "bad draw — retry with a different seed" error; otherwise only the valid prefix is decoded (also strips the "assistant" tail leak). Commit `436bddc`.
2. **22:19 death (user raised temperature)** — the gate 500'd cleanly, but the engine server's generic-error path **auto-evicted + reloaded**; the old vLLM EngineCore still pinned its 12 GiB → new core init failed (0.87/15.48 GiB free) → cascade → container recycled. Fixed: `_synth_editx` converts the bad-draw error into **`SynthParamError`** → 400 **without evict/reload** (the server's own comment at `tts_lab_engine_server.py:338` already warned editx reload fails outright). Bad draws now cost a 🎲 click, never a container death. Commit `23ca9e4`.

### Verified live

- Cold start (first synth after container start) = engine init **196–277 s** (CUDA graph capture + JIT) — first-request curl timeouts must be ≥ 600 s.
- Determinism: same seed + same process → bit-identical WAVs (md5 `2494a67f4bce` verified across runs).
- ~90%+ clean draws at temp 0.5; at temp 1.2 provocation (seeds 1–5): 3× 200 (tail-stripped) + 2× HTTP 400 "EditX bad draw: no valid interleave prefix" — container stayed `Up (healthy)` at 12.9 GiB resident; seed-777 fox whispered clean afterwards.
- Remaining caveat: ~8–10% of draws still land on wrong content (seed lottery) — 🎲 is the retry path; non-draw synth errors (CUDA OOM) still take evict+reload → orchestrator recycling (31c4b3e) self-heals with a visible disconnect.

### Files

| File | Change |
|---|---|
| `docker/Dockerfile.engine-editx` | patch #3 v2 — `_generate` de-rotation + validate-and-strip + bad-draw gate (byte-verified identical to live container) |
| `tts_lab_engines.py` | `_synth_editx` sampling defaults + seed policy + bad-draw → SynthParamError |
| `tts_lab_ui.py` / `tts_lab_config.py` | editx panel defaults (temp 0.5, seed 0=auto, max_tokens 0=auto) + Persian-unsupported warning |

**Upstream faults drafted for a GitHub issue** (stepfun-ai/Step-Audio-EditX — user to post): `BatchEncoding` treated as list → TypeError in `<audio_N>` filter + silently broken max_tokens cap; no chunk-alignment guarantee; greedy never terminates on the 8-token attractor. NOT upstream: vLLM nightly seed behavior, Persian support.

---

## Session 2026-08-24 (late 2) — OmniVoice "my voice is…" repetition FIXED (stale ref transcript)

### Symptom

OmniVoice cloning output recited the reference **transcript** ("*…My voice is engineered for clarity…*") instead of only the target text, then degraded into rambling garbage. Longer outputs (speed 0.5) ran 56 s for a ~230-char news paragraph.

### Root cause — UI stale ref_text, verified end-to-end

1. **UI auto-fill only ran on an EMPTY `ref_text` field** (`tts_lab_ui.py` ref-select handler, added 2026-08-23 for editx/s2pro/qwen3tts). Pick en-leo2 → its sidecar transcript auto-fills the field.
2. Switch the ref voice to **en-brian / en-chris / en-william** → the field is non-empty → **leo2's transcript stayed in `ref_text`**, and every request paired the NEW audio with the OLD transcript.
3. OmniVoice's prompt is `ref_text + " " + text` (`_combine_text`); with mismatched prompt text vs ref audio the diffusion LM **re-reads the prompt** — confirmed by transcribing the user's saved outputs in `/opt/arthur/generation_history/` (`b6e3e57f` etc. via `docker cp` + whisper round-trip: *"Favois is engineered for clarity, precision, and online crossover. Stocks were little changed…"* + garbage tail).

A/B tests on the VM (omnivoice 0.2.1 in engine-current) proved the same ref + **matching** transcript is clean for short, long-chunked, and Persian text — the bug is purely the mismatched pairing, not the model or the sidecar transcripts.

### Fix

`tts_lab_ui.py` ref-select handler: track the last **auto-filled** transcript per select (`sel.dataset.autoRefText`). On ref switch, replace it with the new voice's sidecar transcript (or clear it when the new voice has none); never clobber text the user typed/edited. Rebuilt + recreated `tts-lab-orchestrator` (verified serving `autoRefText` JS).

### Files

| File | Change |
|---|---|
| `tts_lab_ui.py` | ref-select handler — stale auto-fill replacement on voice switch |

---

## Session 2026-08-22/23 — S2-Pro Unblocked (sglang-omni) + Step Audio EditX Deployed

Deployed and validated the two newest engines on the VM (container-only mode, ports 8009/8105/8005). Working tree at `2f44718` (all pushes on `main`).

### S2-Pro (Fish Speech) — UNBLOCKED after 2 months

`docs/issues/s2pro-investigation.md` verdict flipped BLOCKED → UNBLOCKED 2026-08-22. The old pip-built SGLang image (cu128 world, nvjitlink pin conflict) was replaced by an **SGLang-Omni 0.1.3** image (`docker/Dockerfile.sglang` rewritten; new tag `tts-lab-sglang-omni`; served via `sgl-omni serve`).

VRAM fit on the 16 GB card took four rounds (all committed): decode CUDA graphs off (capture OOM), `max_running_requests 2`, `mem_fraction_static 0.85`, and finally **vocoder codec moved to CPU** (`gpu: null` in `s2pro_tts.yaml` — frees ~3.4 GiB; codec is descript-audio-codec, output stays 44.1 kHz). Boot now healthy: one ~11.75 GB pipeline process, 15,540 MiB peak.

| Fact | Detail |
|---|---|
| First synth | 182 s — flashinfer sm_120 decode kernels JIT-compiled on first request (nvcc; no prebuilt kernels) |
| Steady state | 4.6 s for 3.3 s audio via orchestrator (RTF ~11.5; per-step latency from CPU-vocoder IPC) |
| API quirk | `POST /v1/audio/speech` returns **raw WAV bytes**, not base64 JSON; `/health` lives at server ROOT, not under /v1/ |
| Cache | 11 G s2-pro cache moved onto `/opt/models` host mount (was in container writable layer — would re-download); flashinfer JIT persisted at `/opt/models/flashinfer-jit` |
| Orchestrator | s2pro is always-resident ~11 GB → orchestrator stops its container before EditX/LLM loads (`HEAVY` branch), restarts on next s2pro call |

Dispatch fixes: binary-body handling with stdlib wave parse (orchestrator has no numpy), health URL stripping `/v1/`, `SYNTH_TIMEOUT["s2pro"] = 600` (JIT allowance).

### Step Audio EditX — deployed + fully verified (AWQ-4bit)

- `docker/Dockerfile.engine-editx`: ubuntu24.04 base (python 3.12) + **torch 2.13.0+cu130 stable** + **vLLM 0.26 nightly wheel** (wheels.vllm.ai commit 30b34171b) via `uv sync` into `/opt/arthur/Step-Audio-EditX/.venv`. The repo's pinned wheel (torch-2.9.1-era) can never run here — its `_C` links `c10_cuda_check_implementation`, removed from every sm_120-capable torch.
- Crash loop fixed: image was missing `tts_lab_utils.py` (`COPY` added, 8d432d1).
- **Weights are NOT downloaded at load** — `_load_editx` requires them pre-present under `/opt/models/editx/{Step-Audio-EditX-AWQ-4bit, Step-Audio-Tokenizer}`. The engine-server's startup availability probe 503s `/synthesize` when absent ("EditX weights missing at /opt/models/editx") and the lazy-load never runs — restart the container after adding weights so the probe re-runs.
- Weights download (~9 GB AWQ + ~1.5 GB tokenizer, public repos) was kicked off inside the container venv to `/opt/models/editx/` (host mount persists).
- Orchestrator: `EDITX_URL=http://engine-editx:8105`, `SYNTH_TIMEOUT["editx"] = 600` (vLLM warmup), `HEAVY` branch stops LLM + s2pro containers first; engine-server POST timeout now follows `SYNTH_TIMEOUT` (2f44718).
- 503 saga: first synth via orchestrator refused with "EditX weights missing" — the availability probe gates `/synthesize` before the lazy-load path (see above).
- **Four fix layers** baked into the Dockerfile (2026-08-23): `step1.py` `intermediate_tensors = None` default (vllm 0.26 CUDA-graph capture), CCCL version-gate `#if 0` (torch-pinned cudart 13.0 vs pip nvcc 13.3 skew), pip cu13 layout symlinks (`lib64→lib`, `libcudart.so`, stubs `libcuda.so`), and `tts.py` `_generate` BatchEncoding coercion (`apply_chat_template` returns dict-like). Engine-server wrapper injects `CUDA_HOME`/`PATH`/`FLASHINFER_WORKSPACE_BASE` before any ML import. ⚠️ Module-caching trap: patching `tts.py` on disk needs a `docker restart` — the running server keeps `sys.modules` copies.

### Verification (2026-08-22/23)

- **s2pro:** synth verified end-to-end; **voice clone verified 2026-08-23** — `audio_prompt_id=en-leo` + ref_text → 3.8 s / 44.1 kHz WAV, error=None, full orchestrator stop/start round-trip (116 s incl. container start + model load).
- **editx first synth** (03:32): 7.2 s / 24 kHz WAV, generate alone ~5 s, resident ~12.8 GiB.
- **editx edit-type tests** (03:40): emotion (8.0 s, RTF 2.26), style (6.2 s, RTF 1.93), speed (10.4 s, RTF 1.70) — all 200 OK, error=None, engine resident (no reload).
- **editx clone tests** (03:42): clone + ref_text → 6.5 s sane output (RTF 1.66); clone WITHOUT ref_text falls back to target text as prompt transcript → **62 s degenerate ramble** — faithful clones need a real transcript.
- **2026-08-23 continuation session:** editx image rebuild on VM in progress (`make build-engine ENGINE=editx` — bakes all four fix layers; container recreate + re-verify follow).

### State at close

- s2pro: SUPPORTED, available in orchestrator, synth + clone verified.
- editx: SUPPORTED, available in orchestrator, first synth + edit types + clone verified; image rebuilt with baked patches.
- LLM qwen36: stopped while editx resident (eviction protocol) — restart via orchestrator on next LLM call.
- Counts updated in `docs/engine_compatibility.yaml`: 30 engines — 18 supported, 8 experimental, 3 blocked (+1 planned LLM).

---

## Session 2026-08-23 (late) — Persian clones fixed: sidecar transcripts + sglang language enum

User: "can farsi be fixed? i need it". Three layers:

1. **fa-* transcripts.** The curated fa-* voices are ElevenLabs Farsi TTS output (`source: "elevenlabs-farsi"` — no dataset transcript exists; whisper base mangles Persian). The user provided the authoritative script — all 15 fa-* voices say «قشنگ ترین خاطره زندگی ام روزی بود که با ژاله و چنگیز رفتیم شمال» — written into every sidecar (`source: user-provided-2026-08-23`). A faster-whisper large-v3 cross-check (installed in /opt/arthur-extra-env; int8 CPU, 15 clips) confirmed the sentence for 13/15 (orthographic drift only); fa-charles + fa-liam were heard reading a different news sentence («عباس عراقچی وزیر امور خارجه ایران نیز… مذاکرات به خوبی پیش رفته است») by BOTH ASR models — flagged to the user, sidecars reverted to the user's sentence per their call ("these are 100% wrong scripts"). Note: the earlier `fa_diagnose.sh` had briefly rewritten charles/liam to the ASR text; `deploy_fa_fix.sh` reverted them.
2. **s2pro 400 on every Persian clone** — the real blocker. sgl-omni's `/v1/audio/speech` validates `language` against a fixed 11-value enum (Auto|Chinese|English|French|German|Italian|Japanese|Korean|Portuguese|Russian|Spanish); the UI sent `"fa"` → `400 BadRequestError`. Fixed in `_do_synth_sglang` (tts_lab_dispatch.py): `_sglang_lang()` maps lab codes → enum names, unknown → `"Auto"` (model auto-detects fa). Deployed, verified: s2pro + fa-ryan + sidecar transcript → **6.4 s / 44.1 kHz, error=None**. This also means en-* clones via s2pro now send `"English"` instead of `"en"`.
3. **editx fa-ryan ramble (59 s) — model limitation, documented.** With the same correct transcript, editx clones fa-alex (9.0 s), fa-mike (13.9 s), fa-ash (21.1 s borderline) but degenerates on very deep male refs: fa-ryan (f0 ≈ 84 Hz → 59 s), fa-adam (76 Hz → 55.6 s). Pitch-shifting fa-ryan up 30% collapsed instead (0.1 s — resample artifacts), so the f0 hypothesis is unproven but the trend is consistent. Verdict: EditX deep-voice OOD — use s2pro for those voices (works for all fa). Added to `docs/engine_compatibility.yaml` notes.

State: all 21 ref voices (6 en + 15 fa) carry transcripts → 📝 in the dropdown, menu-clone works without typing ref_text. **Use s2pro for Persian clones** (44.1 kHz, all voices); editx for most voices except the deep-male pair. Memory: `sglang-language-enum.md`.

---

## Session 2026-08-23 (evening) — UI fixes: qwen36 DNS, flat menu-clone, dark text; neutts removed

Three UI issues reported by the user, all fixed and deployed (commits `d503619`, `9a0a3e7`):

1. **`⚠ [Errno -3] Temporary failure in name resolution` in every engine panel.** The qwen36 LLM (retired earlier that day) was still in `MODEL_ORDER` + compose `QWEN36_URL`, so `/status` probed `http://llm-qwen36:8006`, which no longer resolves inside the compose network. Removed the qwen36 entry (catalog + order), deleted the compose service/env, and hardened `_check_available_remote` — any dead container now collapses to a clean `offline — engine container not running` instead of leaking the raw errno.
2. **Clone from the menu generated a flat robotic voice.** The curated ref voices' sidecars carry NO transcript, and `_synth_editx` fell back to the *target* text as the prompt transcript when `ref_text` was empty (measured earlier: 62 s degenerate ramble). Fixes: new `_ref_transcript()` helper (stdlib, orchestrator-safe) reads the sidecar `transcription`; `_synth_editx`, `_synth_s2pro`, and the dispatch SGLang path all use it before the target-text fallback. The 6 curated en-* voices were **transcribed with whisper** (base model, `/opt/arthur-extra-env/bin/whisper` — CPU) and their sidecars now carry real transcripts (e.g. en-leo: "Hey there, I'm Leo. Bring your stories to life…"). fa-* transcripts were deliberately NOT written — whisper base mangles Persian; those keep the ⚠ no-transcript hint in the dropdown. Voice-library `use-ref` also writes a transcription sidecar.
3. **Dark text on dark UI.** `.desc-input` referenced the never-defined `var(--fg)` → browser-default (dark) text on the dark textarea. → `var(--text)`; added `.form-select option` rule (native options don't inherit); fixed `var(--ok)` → `--accent2`.

Also: editx panel + notes said "41.6 kHz" — actual output is 24 kHz (fixed). **neutts (NeuTTS Air) removed completely** per user request — it was an unimplemented stub surfacing a red "not configured" warning; all code, catalog, UI, dispatch entries and doc counts removed (28 total / 7 experimental).

The voice-library endpoints (`/voice-library/*`) are still orchestrator-mode-dead (`voice_library_mod = None` by design, `tts_lab.py:68`) — the Browse Voices tab 404s in container mode. The refs dropdown (en-*/fa-* curated voices + uploads) is the working path; the sidebar "Browse Voices" item is cosmetic until the orchestrator ships `voice_library.py` + a `VOICES_DIR` mount.

---

## Session 2026-07-01 — Engine-Current Rebuild, Missing Deps, Final Verification

### Engine-Current Image Rebuild

Docker build on VM repeatedly hung (BuildKit, legacy builder). See ADHOC-LOG §12 for full details. Ended up building from intermediate image + manual pip installs + `docker commit`.

### Missing Dependencies Discovered

The intermediate image (step 14/41) was missing these packages that the full Dockerfile would have installed:

- **`resemble-perth`** (not `perth`!) — chatterbox needs `PerthImplicitWatermarker`
- **`s3tokenizer`**, `conformer`, `diffusers`, `pyloudnorm` — chatterbox deps
- **`transformers` from git main** — `HiggsAudioV2TokenizerModel` not in PyPI 5.12.1
- **`soxr`** — needed by transformers git for audio resampling
- **`python3-dev`, `gcc`** — triton JIT needs gcc to compile CUDA stubs

### Final Verified Engines (2026-07-01)

| Engine | Status |
|---|---|
| omnivoice + voice clone | ✅ sr=24000, rtf=7.9× |
| omnivoice basic | ✅ sr=24000, rtf=4.9× |
| chatterbox (persian) | ✅ sr=24000, rtf=37.7× |
| chatterboxturbo | ✅ sr=24000, rtf=13.9× |
| piper | ✅ sr=22050, rtf=1.1× |

### Documentation Updated

- `04-ADHOC-LOG.md` — §12: rebuild saga + missing deps table
- `06-STATE` — Updated to 2026-07-01 with test results + deps table
- `SESSION_SUMMARY.md` — This entry

---

## Session 2026-06-27/29 — Remote Engine Routing, OmniVoice Fix, LLM VRAM Coordination

### What Was Fixed

Three bugs were preventing remote TTS engines (particularly OmniVoice) from working through the orchestrator.

#### Bug 1: Missing Engine URL Env Vars
- **Symptom:** `POST /synthesize/omnivoice` → `"Not available: pip install omnivoice needed"`
- **Root cause:** Only 7 of 28 `{ENGINE}_URL` env vars were set in the orchestrator container. Missing engines fell through to local dispatch (no ML libs).
- **Fix:** Added 21 missing `-e {ENGINE}_URL=...` lines to Makefile `deploy-orchestrator`. Recreated container with all 28 URLs.
- **Files:** `Makefile`, `docker-compose.yml`

#### Bug 2: OmniVoice Voice Cloning — torchcodec Stub Conflict
- **Symptom:** `audio_prompt_id` → 500 error: `AttributeError: module 'torchcodec' has no attribute 'decoders'`
- **Root cause:** `torchcodec` v99.0.0 dummy stub installed for f5-tts compatibility. When ASR pipeline actually uses it (not just imports), `torchcodec.decoders` is inaccessible at runtime.
- **Fix:** Monkey-patched `is_torchcodec_available` → `False` in `tts_lab_shims.py`.
- **Files:** `tts_lab_shims.py`
- **Note:** Monkey patch. Proper fix requires container isolation (f5-tts vs omnivoice) or real torchcodec.

#### Bug 3: LLM VRAM Blocking Heavy TTS
- **Symptom:** CUDA OOM when loading heavy TTS engines (LLM using ~13.2 GB VRAM)
- **Root cause:** LLM→TTS eviction existed, but TTS→LLM eviction was missing.
- **Fix:** Mounted Docker socket in orchestrator. Added `_stop_llm_container()` / `_start_llm_container()` in dispatch. Heavy TTS stops LLM; LLM inference restarts it.
- **Files:** `tts_lab_dispatch.py`, `Makefile`, `docker-compose.yml`

### Verification

- 13 of 15 tested engines working (piper, kokoro, melo, chattts, f5tts, bark, outetts, chatterbox, fishspeech, zonos, qwen3tts, omnivoice, omnivoice+clone)
- 2 pre-existing failures: dia, styletts2 (hang >180s, not caused by these changes)
- 11 unavailable (expected — optional containers not running or blocked engines)

### Commit

`eb48b67` — fix: remote engine routing, omnivoice voice cloning, LLM VRAM eviction (4 files)

### Documentation Updated

- `docs/containerization/04-ADHOC-LOG.md` — Section 11: full incident report
- `docs/containerization/06-STATE-2026-06-29.md` — New state snapshot
- `docs/reference/KNOWN_ISSUES.md` — Rewritten with all fixed + open issues
- `docs/engine_compatibility.yaml` — Updated omnivoice notes
- `docs/sessions/SESSION_SUMMARY.md` — This entry

---

## Session 2026-04-20 — GPU Engine Fixes (RTX 5060 Ti)

### What Was Fixed

The VM received an RTX 5060 Ti (16 GB GDDR7). Several TTS engines were broken or not using GPU. This session fixed them all one by one with a full benchmark confirming results.

#### VM Infrastructure Changes
| Change | Detail |
|--------|--------|
| Pip cache | Moved to `/opt/models/pip-cache` (180 GB disk) — prevents root disk filling up |
| `onnxruntime-gpu` | Installed — CUDA EP + TensorRT EP now available |
| `libnvrtc.so.13` | Symlinked → `libnvrtc.so.12` for torchcodec ABI compat |
| ChatTTS `gpt.py` | Patched on VM: `narrow(1,-n,n)` guarded for `n=0` (PyTorch 2.10 strict validation) |
| `restart_server()` | Added `sudo` — bench restarts now actually work between engines |

#### Engine Fixes (bench4 — final confirmed results)

| Engine | Before | After | Root cause / Fix |
|--------|--------|-------|-----------------|
| **XTTS-v2** | RTF 3.85 (CPU) | **RTF 0.91** | `gpu=True` in coqui TTS constructor |
| **Chatterbox** | 500 error | **RTF 1.67** | `types.ModuleType` stub (not MagicMock); bypass `torchaudio.save()` → `_to_wav()` |
| **Zonos** | 500 error | **RTF 4.03** | `generate(prefix_conditioning=…)` keyword + `autoencoder.decode()` |
| **ChatTTS** | 500 error | **RTF 2.59** | Patched `gpt.py` narrow() + `empty_cache()` + fixed seed=2024 |
| **MeloTTS** | RTF 1.01 | **RTF 0.30** | Already GPU — confirmed 3.4× faster |
| **StyleTTS2** | RTF 1.52 | **RTF 0.35** | Already GPU — confirmed 4.3× faster |
| **Piper** | RTF 0.37 | RTF 0.36 | GPU EP was 40× slower — kept CPU ONNX (tiny model) |
| **Kokoro** | RTF 2.83 | RTF 2.77 | GPU EP no speedup for 82 MB model — kept CPU ONNX |
| **Parler** | 500 error | ⚠️ version-gated | Requires `transformers==4.46.1`; bench env has 4.57.6. Needs own venv. |
| **OpenVoice** | device mismatch | ⚠️ VAD edge case | Speaker embeddings moved to `DEVICE`; SE extractor null-guarded |
| **OuteTTS** | 500 error | ⚠️ needs GGUF | HF backend pre-encodes any text as ~15K tokens. Use LLAMACPP + `.gguf` file |
| **Orpheus** | 500 error | ⚠️ gated HF | Needs `huggingface-cli login` (canopylabs/orpheus-3b-0.1-ft) |

#### Root Causes Reference

| Error | Cause |
|-------|-------|
| `torchcodec.__spec__ is not set` | MagicMock stub — use `types.ModuleType` with proper `__spec__` |
| `torchaudio.save() TorchCodec required` | torchaudio 2.10 routes save through torchcodec — use `_to_wav()` directly |
| `Zonos: no .decode() found` | API changed: `generate(prefix_conditioning=)` + `autoencoder.decode()` |
| `ChatTTS narrow() length must be non-negative` | PyTorch 2.10 strict: `narrow(1,-n,n)` with `n=0` raises. Patch `gpt.py` line 215, 230, 239, 251 |
| `Parler Config has to be initialized…` | `transformers>=4.51` calls `__init__()` with no args; parler raises ValueError |
| `Parler 'NoneType' has no .update()` | `generation_config` is `None` in transformers 4.57 when no JSON exists |
| `OpenVoice tensor device mismatch` | Speaker SE tensors loaded on CPU, converter on CUDA — use `map_location=DEVICE` |
| `OuteTTS max_length < input_ids` | HF backend encodes any text as ~15K tokens — needs GGUF + LLAMACPP backend |
| `piper/kokoro GPU EP slower` | Tiny ONNX models: GPU memory transfer overhead exceeds GPU compute speedup |

#### Commits This Session

| Commit | Description |
|--------|-------------|
| `7e60c7c` | XTTS `gpu=True`, Zonos API, Chatterbox stub+synth, Piper/Kokoro CPU revert |
| `010fc48` | Chatterbox `ModuleType` stub, Parler version gate, OpenVoice GPU tensors |
| `a9c05b2` | Benchmark results table updated (final bench4 numbers) |
| `93847ae` | ChatTTS `gpt.py` patch, OuteTTS GGUF directive, OuteTTS CHUNKED mode |

---

## Project Goal

**SpamBlocker** is an Android scam-baiting app for Google Pixel running on the user's personal phone number.

When a scam call comes in, instead of hanging up, the app **wastes scammers' time** by deploying Arthur Henderson — a convincingly confused 78-year-old retired postal worker from Phoenix, Arizona — as an AI decoy. Arthur keeps scammers on the line as long as possible while extracting real operational intelligence that can be reported to law enforcement:

| Intelligence target | Why it matters |
|---|---|
| Callback number | VoIP provider can be subpoenaed for account owner |
| Website / URL | WHOIS + hosting company have registrant identity |
| AnyDesk / TeamViewer ID | Both companies cooperate with law enforcement |
| Crypto wallet addresses | Traceable via blockchain + exchange KYC |
| Badge / case numbers | Links to scam operation structure |

**Arthur's character:**  
Lives alone with his cat Mr. Whiskers since wife Martha passed. Son in Tucson calls Sundays. Slow, polite, always "about to comply", never hostile. Progresses through 4 frustration stages across the call to stay believable.

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ANDROID (Pixel 5 — user's personal number)                     │
│                                                                 │
│  BaiterScreeningService                                         │
│    ↓ unknown caller + high spam score                           │
│    → silence ringer, flag PendingAutoBait                       │
│                                                                 │
│  IncomingCallActivity  →  user taps [BAIT]                      │
│                                                                 │
│  BaiterInCallService  (InCallService)                           │
│    ├─ STT: Android SpeechRecognizer (on-device)                 │
│    ├─ ScamKeywordDetector  → scam alert overlay                 │
│    ├─ ScammerIntelExtractor → callback#, URLs, AnyDesk IDs      │
│    │                                                            │
│    └─ ConversationMode ─────────────────────────────────────┐  │
│         ├─ LocalGemini  → GeminiConversationEngine           │  │
│         │                  ↓ Gemini Flash API (LLM)          │  │
│         │                  ↓ Gemini TTS / ElevenLabs         │  │
│         │                  ↓ HardwareLoopbackPlayer          │  │
│         │                    (speaker → mic acoustic inject)  │  │
│         │                                                    │  │
│         ├─ VapiBridge  → PlaceCall(AiBridgeNumber)           │  │
│         │                  ↓ conference with scammer         │  │
│         │                  ↓ VAPI handles AI + voice         │  │
│         │                  ↓ VapiCallMonitor polls transcript │  │
│         │                                                    │  │
│         └─ HomeBridge  → PlaceCall(HomeBridgeNumber) ────────┘  │
│                            ↓ conference with scammer            │
└────────────────────────────┼────────────────────────────────────┘
                             │ PSTN / carrier
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  TWILIO  (+1 425-675-6272)                                     │
│    TwiML → MediaStream WebSocket                               │
│    ↓ 8kHz μ-law audio (bidirectional)                          │
└────────────────────────────┼───────────────────────────────────┘
                             │ WSS
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  CLOUDFLARE TUNNEL  (arthur.sys.tips)                          │
│    ↓ forwards to localhost:8000                                │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  UBUNTU VM  192.168.0.87  (Hyper-V, 6 vCPU, Xeon D-1528)     │
│                                                                │
│  nginx → uvicorn → arthur_server.py                           │
│                                                                │
│  Per utterance:                                               │
│    μ-law chunks → PCM 16kHz                                   │
│    → faster-whisper base.en (local, RTF 0.35x)  → text        │
│    → Gemini Flash 2.0 (Arthur persona + stage)  → response    │
│    → Gemini 2.5 Flash TTS, voice=Gacrux         → PCM 24kHz  │
│    → resample → μ-law 8kHz → Twilio stream                   │
└────────────────────────────────────────────────────────────────┘
```

---

## What Was Built

### Home Bridge Mode (new conversation path)
Android app now has **3 conversation modes**:

| Mode | How it works |
|---|---|
| `LocalGemini` | Android on-device Gemini + Gemini TTS via TRRS loopback |
| `VapiBridge` | Outbound call to VAPI number → VAPI handles AI |
| **`HomeBridge`** ← **new** | Outbound call to Twilio DID → arthur.sys.tips → Ubuntu VM |

The Home Bridge flow:
```
Scammer calls Pixel 5
       ↓
Android conferences in a 2nd outbound call to Twilio DID
       ↓
Twilio TwiML → MediaStream WebSocket → arthur.sys.tips (Cloudflare Tunnel)
       ↓
Ubuntu VM (192.168.0.87): faster-whisper STT → Gemini Flash LLM → Gemini TTS
       ↓
PCM audio → μ-law → Twilio → back into the conference → scammer hears Arthur
```

---

## Ubuntu VM — arthur-server

**Host:** Hyper-V on WIN-ER1U9A7NKMI (192.168.0.153)  
**VM IP:** 192.168.0.87  
**SSH key:** `%USERPROFILE%\.ssh\id_arthur_vm`  
**User:** `arthur`

### Hardware
| Resource | Before | After |
|---|---|---|
| vCPUs | 2 | **6** (upgraded this session) |
| RAM | 3.8 GB | 3.8 GB |
| CPU | Intel Xeon D-1528 @ 1.90GHz | same |
| Swap | 0 | 0 ⚠️ add 2GB recommended |

### Project Services Running
| Service | Port | Purpose |
|---|---|---|
| `arthur.service` | 8000 | Python uvicorn server (arthur_server.py) |
| `cloudflared-arthur.service` | — | Exposes port 8000 as `arthur.sys.tips` |
| `nginx.service` | 443 | Reverse proxy / TLS termination |

### Files on VM
```
/opt/arthur/
  arthur_server.py     ← main server
/opt/arthur-env/       ← Python 3.11 venv
/root/.cache/huggingface/   ← faster-whisper model weights
```

---

## STT: faster-whisper

- **Package:** `faster-whisper 1.2.1` by SYSTRAN (NOT OpenAI's original)  
- **Same weights as OpenAI Whisper** — just 4x faster via CTranslate2 engine  
- **Runs 100% locally** — no API calls, no internet needed after model download  
- **Current model:** `base.en` (150MB, int8 quantized)

### Benchmark Results (real speech, espeak phrases)

| Model | vCPUs | Avg STT | Avg RTF | Verdict |
|---|---|---|---|---|
| `base.en` | 2 | 4.84s | 0.51x | ✅ Real-time |
| `small.en` | 2 | 15.26s | 1.59x | ❌ Too slow |
| `base.en` | **6** | **2.75s** | **0.35x** | ✅ Real-time (1.8x faster) |
| `small.en` | **6** | **7.76s** | **0.99x** | ✅ Borderline real-time |

**Current config:** `base.en` — 3x headroom, safe choice.  
**To switch to `small.en`:** Edit `WHISPER_MODEL` in `arthur_server.py` and run `deploy.ps1`. `small.en` has better accent accuracy (Indian/Asian scammer voices) and just fits at 6 vCPUs.

### Why RTF matters
- RTF < 1.0 = Whisper transcribes faster than audio was spoken ✅
- RTF > 1.0 = transcription lags behind real-time, Arthur's response is delayed ❌
- Real bottleneck in practice is **Gemini API (~500ms–2s)**, not Whisper

---

## arthur_server.py Architecture

```python
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
GEMINI_FLASH     = "gemini-2.0-flash"          # LLM
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS
GEMINI_TTS_VOICE = "Gacrux"                    # Mature male voice
WHISPER_MODEL    = "base.en"                   # STT
TWILIO_RATE      = 8000                        # μ-law 8kHz from Twilio
```

**Call flow per utterance:**
1. Twilio streams 8kHz μ-law audio chunks via WebSocket
2. Server accumulates chunks, detects silence via energy threshold
3. Converts μ-law → 16kHz PCM → faster-whisper → text
4. Text → Gemini Flash (with Arthur persona + stage prompts) → response text
5. Response text → Gemini TTS → 24kHz PCM → resample to 8kHz → μ-law → Twilio stream

**Arthur persona stages** (time-based, gets more confused):
- Stage 0 (0–3 min): Politely confused, asks clarifying questions
- Stage 1 (3–6 min): Slightly anxious, asks for repetition
- Stage 2 (6–9 min): Increasingly confused, circles back to earlier topics
- Stage 3 (9+ min): Maximum confusion, seemingly unable to proceed

---

## Local TTS Options (not yet integrated)

Discussed as a future improvement to replace Gemini TTS with fully local voice:

| Engine | Size | CPU Speed | Quality | Best Arthur voice |
|---|---|---|---|---|
| **Kokoro** | 82MB | ~35x RT | ⭐⭐⭐⭐⭐ | `bm_lewis` (British male) |
| **Piper** | 50–200MB | ~100x RT | ⭐⭐⭐ | `en_US-ryan-high` |
| **XTTS v2** | 1.8GB | ~3x RT | ⭐⭐⭐⭐⭐ | Voice cloning |
| **Parler TTS** | 880MB | ~5x RT | ⭐⭐⭐⭐ | Text-described voice |

**Recommended next step:** Integrate Kokoro `bm_lewis` as primary TTS in `arthur_server.py`, with Gemini TTS as fallback. Would reduce TTS latency from ~1–2s (cloud) to ~0.1s (local).

---

## Deploy Workflow

```powershell
# Deploy updated arthur_server.py to VM
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy.ps1
```

Or manually:
```powershell
$key = "$env:USERPROFILE\.ssh\id_arthur_vm"
scp -i $key arthur_server.py arthur@192.168.0.87:/tmp/
ssh -i $key arthur@192.168.0.87 "sudo cp /tmp/arthur_server.py /opt/arthur/ && sudo systemctl restart arthur"
```

Check live logs:
```powershell
ssh -i $key arthur@192.168.0.87 "sudo journalctl -u arthur -f"
```

---

## Cost Summary

| Component | Provider | Cost |
|---|---|---|
| STT (faster-whisper) | Local VM | $0 |
| LLM (Gemini Flash) | Google free tier | $0 |
| TTS (Gemini TTS) | Google free tier | $0 |
| Inbound call | Carrier | $0 (personal number) |
| Outbound to Twilio DID | Twilio | ~$0.008/min |
| **Total per call** | | **~$0.008/min** |

---

## Key Files Changed This Session

| File | Change |
|---|---|
| `Spamblocker/Services/BaiterInCallService.cs` | Added `InitiateHomeBridge()`, `_pendingHomeBridge`, `isHomeBridge` param to `MonitorBridgeCall()` |
| `Spamblocker/IncomingCallActivity.cs` | Added Home Bridge button wiring |
| `Spamblocker/Resources/layout/activity_bait_call.xml` | Added `btnBaitHomeBridge` button |
| `tools/arthur_server/arthur_server.py` | Full Python home bridge server (Twilio MediaStream) |
| `tools/arthur_server/deploy.ps1` | Windows deploy script |
| `tools/arthur_server/setup_vm.sh` | Ubuntu VM one-shot setup script |

---

## Next Chat — Suggested Topics

- [ ] Integrate Kokoro local TTS into `arthur_server.py`
- [ ] Add 2GB swap to Ubuntu VM (`fallocate -l 2G /swapfile`)
- [ ] Switch STT to `small.en` (better accent handling, safe now at 6 vCPUs)
- [ ] Add `small.en` model upgrade to `setup_vm.sh`
- [ ] Test end-to-end with a real scam call

---

## Session 2026-08-13/14 — Ideogram 4: Blank Images (Caption Starvation) + Seed Randomization

Full write-up: [docs/image-lab/IDEOGRAM4_FIX_2026-08-14.md](../image-lab/IDEOGRAM4_FIX_2026-08-14.md). Commit `ea24d9c`.

### What happened

Ideogram 4 generations from **short plain-text prompts** came back uniform
gray at 1024², with a "caption verifier" warning that looked like a safety
filter. Root cause: Ideogram 4 conditions the image only through attention to
text tokens; below ~1% text-token ratio the CFG branches collapse (measured:
blank at 0.24–0.41%, real at 0.73–2.85%). "Safety filter" = the caption
verifier's JSON warning — no safety filter exists.

### What was done

1. **Auto-expansion (default ON):** short plain-text prompts (<1% text-ratio)
   are expanded to a JSON caption via **Ideogram's hosted magic-prompt API**
   (`api.ideogram.ai/v1/ideogram-v4/magic-prompt`, free, `IDEOGRAM_API_KEY`).
   JSON captions and long prompts pass through **byte-identical**; expansion
   failure never blocks the request. **No local LLM** — an on-device Qwen3-VL
   expansion is impossible because the pipe's text_encoder is a headless
   `Qwen3VLModel` (no lm_head, no generate()); Qwen3-8B-Instruct is HF-gated
   and the lab token has no access. User chose cloud over local.
2. **Seed randomization:** `seed=-1` now draws a random seed server-side
   (was the pipeline's deterministic default 67280421310721 → byte-identical
   images). Actual seed recorded in API response + gallery.

### Verified

Engine-level + live API (service restarted): short prompt → real image +
expanded caption; production JSON caption → byte-identical + real image;
seed recorded/randomized; empty prompt → 422 unchanged. Deployed md5s match
repo (`26b0e41e…`, `6c98ffcb…`).

### Not approved (do NOT implement)

Fix #2 (cap resolution for short prompts), fix #4 (idle-eviction `last_used` bug).

---

## Session 2026-08-24 — VRAM Eviction Fixed + Per-Process GPU Visibility

EditX OOM diagnosis: s2pro (always-resident, PID 368, 7.98 GiB) + editx vLLM mid-load (4.89 GiB) → 20 MiB alloc failed with 11 MiB free. "Evict VRAM" button was a no-op because (a) `/evict-all` never touched SGLang containers (s2pro un-evictable via button), (b) every engine container reported `current_engine: None` while processes held 6.6 GiB (failed vLLM loads leave their arena pinned; bookkeeping desyncs), (c) `/evict` returns `evicted: False` when nothing is bookmarked, so the UI showed "Evicted 0" with VRAM unchanged.

### Fixes (all 4)

1. **`/evict-all` + single evict now cover every engine class** (`tts_lab_dispatch.py`): SGLang (s2pro) → container stop; vLLM-backed (editx, orpheus) → container restart (in-process evict can't free vLLM's arena — `_VLLM_CONTAINERS` map, `_container_restart()` helper); standard torch containers → POST /evict. Dead services (legacy, orpheus) skip instead of DNS-erroring.
2. **vLLM load-failure self-recycle** (`tts_lab_engine_server.py`): on a failed load in a vLLM-backed stack (`_HAS_VLLM` = vllm importable), the server exits 1.5 s after the error response so Docker (`restart: unless-stopped`) recycles the container from 0 MiB — kills the "stuck 4.89 GiB from a failed OOM load" trap.
3. **Honest memory reporting**: engine `/evict` returns `freed_mb` (device-wide delta) + `held_mb` (what the process still pins); `/evict-all` aggregates `freed_mb_total`/`held_mb_total`; UI toasts now say "Evicted N · freed X · Y still held" instead of a hollow count. Engine `/health` adds per-process `proc_allocated_mb`/`proc_reserved_mb`.
4. **Per-process GPU breakdown in `/status`**: orchestrator runs `nvidia-smi --query-compute-apps` via Docker exec in a GPU container, maps host PIDs → containers via `docker top`, and returns `gpu.processes` (pid/mb/process/container) — incl. the bare-metal Image Lab. UI shows a live "GPU: engine-editx 3.9 GiB · engine-current 2.1 GiB · Image Lab 230 MiB" line under the VRAM bar.

### s2pro lighter-eviction options (checked on image)

- sgl-omni has **no unload endpoint** and `docker pause` frees nothing — stop/restart remains the eviction mechanism (already light: weights on host mount + flashinfer JIT persisted at `/opt/models/flashinfer-jit` → no re-download/recompile on restart).
- sgl-omni CLI **does accept `--cpu-offload-gb N`** (sglang 0.5.16 `OffloaderV1`): keeps the container alive with ~N GB of weights in CPU RAM (GPU footprint drops to ~3-4 GB) — coexistence mode without restarts, at the cost of per-request speed + CPU RAM. Not defaulted (would skew benchmark RTF); available as an opt-in compose flag.

### Verification (deployed + confirmed live, 2026-08-24)

`py_compile` clean on all four files. Live repro before fix: `/evict-all` → `evicted_count: 0`, VRAM 6624→6624 MiB. Deployed to the VM via scp + `docker compose up` (bridge — never `make deploy-*`, breaks engine-URL routing). Two additional bugs surfaced during live verification and fixed:

5. **`_docker_api` sent JSON bodies without `Content-Type`** → Docker Engine API rejects exec-create with 400 "malformed Content-Type header", so the process breakdown silently returned `[]` (bare except swallowed it). Fix: set `Content-Type: application/json` on content-bearing calls.
6. **nvidia-smi inside a normal container namespace-translates PIDs** (every CUDA process shows as pid 1) and hides other containers' processes — attribution impossible. Fix: a lazily-created, persistent `--pid=host` probe container (`tts-lab-gpu-probe`, AutoRemove, nvidia DeviceRequests, ~0 VRAM, recreated on demand) reports **host** PIDs for the whole box; `docker top` maps them to containers; bare-metal processes (Image Lab) are labelled `host`.
7. **Ghost VRAM survives `/evict`**: engine-current held 2144 MiB with `current_engine: None` and torch reporting 0 allocated/0 reserved — a native arena (ONNX Runtime/CTranslate2) or a failed-load traceback pin, invisible to torch and unreleasable in-process. Fix: after a `/evict` that evicted nothing, probe the container's real driver-level usage and **recycle the container** if it still pins > 768 MiB (idle CUDA contexts stay ~128-300 MiB). Live result: `/evict-all` → `container:tts-lab-engine-editx` (restart), `http://engine-current:8101` (`container-restart-after-evict`, "process pinned 2144 MiB after /evict — container recycled"), qwen/mid correctly left alone (128 MiB contexts), VRAM **2787 → 505 MiB**.

`/status` now returns `gpu.processes`: Image Lab 230 MiB (host), engine-current 2.1 GiB, qwen/mid/editx workers 128 MiB each — UI renders a live "GPU: container MiB · …" line under the VRAM bar. Commits: `ac3f71b` (4 fixes), `6844c42` (Content-Type), `0f26555` (pid=host probe), `31c4b3e` + `12d2ac5` (ghost recycle + structure repair).
