# Handoff — EditX + S2-Pro integration (2026-08-23)

> **Read first:** [CLAUDE.md](../../CLAUDE.md) (project identity, architecture, constraints) and the auto-memory index at `C:\Users\farid\.claude\projects\c--repos-TTS-LAB\memory\MEMORY.md`. This document is the continuity record for the current in-flight work; the repo docs under `docs/containerization/`, `docs/reference/`, and `docs/issues/` carry the deeper history.

## TL;DR — current status

| Item | State |
|---|---|
| **Fish S2-Pro** (Elo 1,125) | ✅ **UNBLOCKED + VERIFIED.** Served by custom `tts-lab-sglang-omni` image (`sgl-omni serve`), port 8005, orchestrator-routed via `S2PRO_SGLANG_URL`. First synth verified 2026-08-22 (4.6 s, 44.1 kHz, 200 OK). |
| **Step Audio EditX** (Elo 1,105, AWQ-4bit) | ✅ **FIRST SYNTH VERIFIED 2026-08-23 03:32.** Container `tts-lab-engine-editx` (port 8105). `POST /synthesize/editx` returned a 7.2 s / 24000 Hz WAV with `error: None`. Resident VRAM ~12.8 GiB. |
| Container status | All containers healthy. Orchestrator on 8009 shows editx + s2pro available. **LLM qwen36 (8006) currently stopped** (editx is resident; 16 GiB GPU can't hold both — eviction protocol is orchestrator-driven). |

**Remaining work** (was todo list 12→16): ① edit-type tests (emotion/style/speed), ② voice-clone tests (s2pro references + editx clone), ③ rebuild editx image on VM to bake in the live patches, ④ docs/memory updates. Details at the end of this file.

---

## What was delivered

### 1. S2-Pro via SGLang-Omni (`tts-lab-sglang-omni`)
- Rewrote `docker/Dockerfile.sglang` as an omni image: `sglang-omni==0.1.3` + `descript-audiotools==0.7.2` + `descript-audio-codec==1.0.0` + vendored `s2pro_tts.yaml`. Entrypoint `sgl-omni serve` (not `launch_server`).
- **Binary-body quirk (commit 337972f):** S2-Pro's `/v1/audio/speech` returns **raw WAV bytes**, not a JSON envelope. The dispatch layer handles this (`_do_synth_sglang`).
- Memory: `sglang-omni-binary-response.md` (in the session memory dir) — `/health` lives at the server root; first request JIT-compiles FlashInfer kernels (~2-3 min).
- `docker-compose.yml`: dropped the dead `sglang-runtime` service (port-8005 conflict); `s2pro` service now uses the omni image + `/opt/arthur/reference_voices` + `/tmp/tts_uploads` mounts.

### 2. EditX engine container (`tts-lab-engine-editx`, port 8105)
Own stack (incompatible with the lab's 3.11 containers — that's by design):

```
python 3.12 (ubuntu24.04 base) | torch 2.13.0+cu130 | vllm 0.26.1rc1.dev1117+g30b34171b
transformers 4.57.x | weights /opt/models/editx/ (Step-Audio-EditX-AWQ-4bit + Step-Audio-Tokenizer)
```

The repo's pinned vllm wheel (0.14.0rc2, built vs torch 2.9.1) can **never** run on this GPU — its `_C` links the `c10_cuda_check_implementation` symbol removed from every sm_120-capable torch. The nightly wheel from wheels.vllm.ai is the fix (see Dockerfile header comment).

**The engine-server wrapper** (`tts_lab_engine_server.py` lines 22-32) injects the pip-CUDA toolkit env so flashinfer JIT builds resolve correctly:
`CUDA_HOME`/`PATH` → `.../.venv/lib/python3.12/site-packages/nvidia/cu13`, `FLASHINFER_WORKSPACE_BASE=/opt/models/flashinfer-editx-jit`.

---

## The four fix layers on EditX (all baked into `docker/Dockerfile.engine-editx`)

1. **step1.py `intermediate_tensors` default** — vllm 0.26's Step1ForCausalLM.forward() lacks the `= None` default; CUDA-graph capture passes the model without that kwarg → TypeError at load. One-line sed.
2. **CCCL version-gate patch** — torch 2.13.0+cu130 pins `nvidia-cuda-runtime-cu12==13.0.88` (headers say CUDART 13000) while the unpinned nvcc install lands 13.3; the bundled CCCL's `cuda_toolkit.h:41` gate compares the nvcc tuple vs CUDART and fires `#error`. The sampling kernels only touch cudart APIs stable across 13.x → benign skew. Patch: sed the `#if !_CCCL_CUDACC_EQUAL(...)` to `#if 0`.
3. **pip cu13 layout symlinks** — flashinfer's generated `build.ninja` links `-L$CUDA_HOME/lib64 -lcudart` + `-L.../stubs -lcuda`, but pip CUDA-13 packages live under `lib/` with only versioned sonames. Symlinks: `lib64→lib`, `libcudart.so→libcudart.so.13`, `lib/stubs/libcuda.so→/usr/lib/x86_64-linux-gnu/libcuda.so.1` (driver lib absent at build time; falls back to a touch-empty stub — `ld -shared` never resolves symbols from it).
4. **tts.py `_generate` BatchEncoding coercion (upstream bug, commit pending)** — `model_loader.load_model` returns a **raw** `AutoTokenizer`; this checkpoint's `apply_chat_template(tokenize=True)` returns a **BatchEncoding** (dict-like). The repo's `_generate` iterates it as a plain list → string keys `'input_ids'`/`'attention_mask'` → `TypeError: '<=' not supported between instances of 'int' and 'str'` at the `<audio_N>` filter. Fix inserted at the top of `_generate`:
   ```python
   if hasattr(token_ids, "input_ids"):
       token_ids = list(token_ids["input_ids"])
   ```

### ⚠️ The trap that cost two synth attempts (module caching)
The engine-server imports `tts.py` **once per process** and keeps it in `sys.modules`. Patching the file on disk does NOT affect a running server — **you must `docker restart tts-lab-engine-editx` after patching any repo file.** Synth #10 failed with the identical TypeError after the patch was verified on disk; a restart + re-run (synth #11) succeeded immediately.

### ⚠️ Auto-evict retry VRAM race
After a synth failure the engine-server auto-evicts and retries — but the retry's EngineCore starts while the dead core's VRAM is still releasing (`Free memory 1.09/15.48 GiB` → load fails). This is a transient of the failure path; with the synth bug fixed there is no retry. Don't "fix" the retry logic.

### ⚠️ flashinfer JIT cache is env-sensitive
`/opt/models/flashinfer-editx-jit/.cache/flashinfer/0.6.17/120f/cached_ops/sampling/` holds a working `sampling.so`. Running python **without** the engine-server's env injection regenerates `build.ninja` with `cuda_home=/usr/local/cuda` → `nvcc: not found` → clobbers the good cache. Always reproduce with the full `CUDA_HOME`/`PATH` env (see `ninja_fix.sh` pattern).

---

## Verification evidence

**EditX first synth (2026-08-23 03:29-03:32):**
```
editx loaded in 174.36s
INPUT tokens: total=453, audio(65536-67583)=195, text(<65536)=153, other(>=67584)=105   ← debug block ran (coercion works)
Generated 301 tokens: min=3, max=70474, audio=195
POST /synthesize HTTP/1.1" 200 OK
```
Response: `sample_rate=24000, audio_dur_ms≈7201, synth_time_ms=188732 (includes 174 s load; generate alone ~5 s), rtf=26.21 (load-dominated), error=None`. WAV bytes are valid RIFF (`UklGR...WAVE...`). Resident VRAM 12.8/15.8 GiB.

**S2-Pro:** verified previously — `POST /synthesize/s2pro` → 4.6 s synth, 44.1 kHz, base64 audio.

---

## The code, in the repo

| File | What changed |
|---|---|
| `docker/Dockerfile.engine-editx` | NEW — full editx stack + all 4 fix layers (see above) |
| `docker/Dockerfile.sglang` | Rewritten as sglang-omni image |
| `docker-compose.yml` | `engine-editx` service (profile `editx`, 8105), `s2pro` on omni image, dropped `sglang-runtime`, orchestrator env `EDITX_URL` |
| `tts_lab_config.py` | `MODEL_INFO["editx"]` (heavy: True, ram_est 9000), `MODEL_ORDER` +editx after s2pro, `SYNTH_TIMEOUT["editx"]≈600`, `_ref_wav_path` moved here (orchestrator has no numpy) |
| `tts_lab_engines.py` | `_load_editx`/`_synth_editx` (~2288-2350) + registered in LOADERS/SYNTHERS; `_synth_editx` branches edit_type clone/emotion/style/speed/paralinguistic; `_tensor_to_wav` helper; s2pro stubs rewritten for sgl-omni |
| `tts_lab_dispatch.py` | `_check_available_remote` tolerant branch for SGLang `/health`; `pkg_map["editx"]`; `_do_synth_sglang` references mapping; HEAVY-branch container stop/start for editx ↔ s2pro/LLM coexistence |
| `tts_lab_ui.py` | editx panel (edit-type select, edit_info, n_edit_iter, language, ref upload); s2pro panel refreshed |
| `tts_lab_engine_server.py` | Env injection (lines 22-32) for the pip-cu13 toolkit + FLASHINFER_WORKSPACE_BASE |

**Not yet committed this session:** the tts.py coercion patch exists only as a live container edit + Dockerfile RUN step — it has NOT been committed (it's an upstream-repo file, not in this repo; the Dockerfile RUN is the deliverable, and the Dockerfile edits are uncommitted too). Check `git status` on the working repo.

---

## Commands for the next agent

```bash
# SSH
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87

# Status
curl -s http://192.168.0.87:8009/status | python3 -m json.tool | grep -A6 -E 'editx|s2pro'

# Synth (orchestrator, VM-local)
curl -s -m 900 -X POST http://localhost:8009/synthesize/editx -H 'Content-Type: application/json' \
  --data-raw '{"text": "Hello from the Arthur TTS lab.", "params": {"edit_type": "clone", "language": "en"}}'

# Edit types (payload shapes)
curl -s -m 900 -X POST http://localhost:8009/synthesize/editx -H 'Content-Type: application/json' \
  --data-raw '{"text": "The quick brown fox.", "params": {"edit_type": "emotion", "edit_info": "happy"}}'
  # style / speed / paralinguistic similarly; emotion/style/speed run clone → iterative tts.edit()

# Watch the engine
docker logs --tail 50 -f tts-lab-engine-editx        # (container restarted 03:27 — log file rolled)

# VRAM
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Rebuild the image with the baked-in patches (todo 15)
cd /opt/arthur-tts-lab && make build-engine ENGINE=editx
# then recreate: docker compose --profile editx up -d engine-editx
```

**Remote-file pattern that works** (heredocs through ssh+double-quotes get mangled — always):
Write script locally → `scp file arthur@192.168.0.87:/tmp/` → `ssh ... docker cp /tmp/file tts-lab-engine-editx:/tmp/` → `docker exec tts-lab-engine-editx ...`.
`docker exec bash -lc` resets PATH via /etc/profile — use plain `bash -c`.

---

## Pending work (todo 13→16, in order)

1. **Edit-type tests** — script staged locally at `C:\Users\farid\AppData\Local\Temp\editx_edits.sh` (emotion/style/speed × 3 curls, writes /tmp/editx_<type>.json, prints audio_bytes/sr/ms). scp → run on VM. Expect each ~30-90 s (no reload — engine is resident).
2. **Voice-clone tests** — s2pro: `params.references: [{audio_path: /opt/arthur/reference_voices/<wav>, text: ...}]` (binary-body path); editx: `edit_type=clone` with an uploaded ref WAV + `ref_text` (or orchestrator `audio_prompt_id`). Refs are mounted at `/opt/arthur/reference_voices/` (en-brian/en-chris/en-kebin/en-leo/en-leo2.wav + .json — note the .json files carry NO transcript; `_synth_editx` falls back to target text as prompt transcript, which works but is not a faithful clone prompt — worth improving).
3. **Rebuild editx image on VM** (bakes CCCL + layout + tts.py patches; container currently runs them live only) → `make build-engine ENGINE=editx` + recreate container + re-verify synth.
4. **Docs/memory** — update `docs/sessions/SESSION_SUMMARY.md`; add a memory file for the editx fixes (CCCL gate / cu13 layout / BatchEncoding + the module-caching trap); CLAUDE.md engine-count + torch-version staleness note (editx uses torch 2.13.0+cu130, not the 2.12 nightly in the header).

**Watch-outs:** editx is HEAVY (12.8 GiB resident) — s2pro (11 GiB) and qwen36 LLM (13.6 GiB) cannot coexist with it on the 16 GiB card; the orchestrator's eviction/stop-start protocol handles this, but a manual `docker compose up` of everything at once will OOM. GPU is RTX 5060 Ti 16 GB (sm_120) — no torch below 2.12 works, and only the vllm 0.26 nightly pair does.

## Context pointers
- Full prior-session transcript: `C:\Users\farid\.claude\projects\c--repos-TTS-LAB\92c70a3e-833d-473f-b677-05ad3beae645.jsonl`
- Plan (mostly executed): `C:\Users\farid\.claude\plans\tidy-dreaming-stroustrup.md`
- Engine compatibility: `docs/engine_compatibility.yaml` (needs the s2pro/editx entries updated)
- S2-Pro investigation: `docs/issues/s2pro-investigation.md`
- Memory: `memory/MEMORY.md` + `sglang-omni-binary-response.md`, `flashinfer-sm120-jit-first-request.md`, `s2pro-vocoder-cpu-fix.md`

---

## Continuation status (2026-08-23, same day) — all pending items done

- **① Edit-type tests ✅** — emotion (8.0 s, RTF 2.26), style (6.2 s, RTF 1.93), speed (10.4 s, RTF 1.70), all 200 OK / error=None, engine resident.
- **② Voice-clone tests ✅** — editx clone + ref_text → 6.5 s sane output; clone without ref_text → **62 s degenerate ramble** (fallback transcript is not faithful — needs a real transcript). s2pro clone (en-leo + transcript) → 3.8 s / 44.1 kHz, full orchestrator stop/start round-trip ✓.
- **③ Image rebuild ✅** — `make build-engine ENGINE=editx` on VM (first attempt failed: **root disk 99% full** — torch wheel extraction `failed to flush file`; freed ~220 GB by deleting dead images (torchtest pair, py311 stacks, llama.cpp LLM images per user: "never want qwen llm") + `docker builder prune -af`). Second build succeeded: `PATCHED_OK` + smoke `torch 2.13.0+cu130 | vllm 0.26.1rc1.dev1117`; container recreated with `--force-recreate`, synth re-verified post-rebuild.
- **④ Docs/memory ✅** — SESSION_SUMMARY updated; `engine_compatibility.yaml` s2pro/editx → supported with verification dates (counts 18/8/3); CLAUDE.md stack line + torch gotcha + engine count; memory files: `editx-fix-layers.md`, `llm-qwen-removed.md`, blocked-engines updated. All committed.
- **Infra changes this continuation:** engine-qwen (Qwen3-TTS) stopped + qwen36 LLM container/images removed (user: never want it — see memory `llm-qwen-removed`). ⚠️ Plain `docker compose up -d` would restart engine-qwen (default profile).
