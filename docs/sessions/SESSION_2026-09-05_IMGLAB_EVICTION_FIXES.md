# Session 2026-09-05 — Image Lab VRAM incident: the eviction that never fired

## Goal

A remote caller's client hit repeated 503s on the Image Lab (`FLUX.2 Klein 9B-KV
needs ~10 GiB free VRAM … only 5614 MiB available`). Their diagnosis: healthy
engine, leaked CUDA arena, fix = service restart. This session found the
failure chain ran much deeper — **the TTS-eviction helper had been dead code
since the containers moved onto the compose bridge** — and fixed all four
layers.

## Root-cause chain (what actually happened)

1. **Dead eviction.** `_evict_tts_engines()` POSTed `/evict` to
   `localhost:8101-8104`. Engine containers sit on the compose bridge and
   publish **no host ports** — every call was connection-refused and silently
   swallowed by `except Exception: pass`. Verified: `curl localhost:8101` →
   HTTP 000; bridge IP `172.19.0.4:8101` → HTTP 200, freed 1,960 MiB.
2. **Un-gated on-demand encoder load.** The NF4 Qwen3-8B encoder loads only on
   embed-cache misses (~5.2 GiB staging burst: process 7.55 → 12.58 GiB
   before OOMing on a 96 MiB alloc). With engine-current's ~2.1 GiB model
   resident, a fresh prompt always died here — no gate, no eviction, just OOM.
3. **No cleanup on generator failure.** `generate()` only cleaned up when the
   *loader* failed; a mid-generation OOM left the arena pinned at 12.6+ GiB →
   every later request 503'd on the 10.5 GiB gate until a service restart.
4. **Stale LLM-stop step.** Escalation still tried to stop the retired
   `tts-lab-llm-qwen36` container — HTTP 404 noise (~2 s) on every gate
   failure, misleading logs (this was the caller's quoted "stopping the LLM
   container" 503 text).

## Changes (`image_lab_engines.py`, all four deployed to `/opt/arthur-img/`)

- **Fix 1** — `generate()`: `generator_fn(params)` failures now call
  `_unload_current()` before re-raising (mirrors the `_ensure_engine` failure
  path). A failed generation can no longer brick the card.
- **Fix 2** — `_klein_prompt_embeds` cache-miss path now runs
  `_ensure_encoder_headroom(engine_key)` before loading the encoder:
  `_ENCODER_NEED_MB` (klein 4B 4000 / klein 9B 6000) evicts TTS via the
  (now working) helper when free VRAM is short, else raises a clear error.
  Thresholds sit between measured states: ~4,940 MiB free with a TTS model
  resident (evict → ~7.1 GiB) vs ~7,100 MiB idle-contexts floor (pass).
- **Fix 3** — removed `_stop_llm_container()`, `_LLM_CONTAINER_NAME`,
  `_DOCKER_SOCK`, its call site and docstring/error-message clauses.
- **Fix 4** — `_evict_tts_engines()` now POSTs the orchestrator's
  `/evict-all` (host-reachable on 8009, on-bridge by service name, covers all
  5 containers incl. editx) and logs the JSON summary instead of swallowing
  failures.

## Verification (live on the VM, 2026-09-05)

| Scenario | Result |
|---|---|
| Fresh prompt, TTS **omnivoice resident** (caller's incident state) | ✅ 200 in 55.1 s — gate logged "needs 6000 (only 4944) — evicting", evict-all freed 1,954 MiB, encoder 398/398, image 1024×1024 saved |
| Same prompt again (disk-cache hit, warm engine) | ✅ 200 in 23.2 s — "Prompt-embedding cache hit — skipping the text encoder", **no eviction** (surgical design holds) |
| Mid-load OOM (deliberate 96 MiB alloc fail, pre-Fix-4) | ✅ clean "Unloading engine" + 503 detail, next request healthy — no brick (Fix 1) |

## Notes

- First calibration attempt (6000 MiB) initially 503'd every fresh prompt —
  **that measurement had omnivoice resident**; the eviction couldn't free it,
  so the gate correctly refused. Lowering to 4500 then reproduced the OOM
  live (encoder staging really is ~5.2 GiB). Only after Fix 4 made eviction
  real did 6000 become the right threshold.
- Identity-run successes before this incident only happened because the TTS
  containers' own 900 s idle-eviction had dropped their models — never
  because imglab evicted them.
- The engine container `/evict` responses are still reachable by bridge IP
  for manual use; the orchestrator is the canonical broker going forward.

## Pending

- Commit (this doc + `image_lab_engines.py`). Unrelated dirty tree (Persian
  models work) left untouched.
- Caller should retest; TTS models lazy-reload on their next request
  (a few seconds of added latency is the designed cost of eviction).
