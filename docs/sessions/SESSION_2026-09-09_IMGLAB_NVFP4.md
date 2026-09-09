# SESSION_2026-09-09 — FLUX.2 Klein 9B NVFP4 fast lane + VRAM-gate deadlock hardening

**Date:** 2026-09-08/09 · **VM:** arthur@192.168.0.87 (RTX 5060 Ti 16 GB) · **Service:** arthur-imglab (:8002)

## What shipped

1. **11th engine `flux2klein9b-nvfp4`** on the live :8002 — the same 9B-KV klein
   model in NVFP4 (lite-infer nunchaku-lite checkpoint), registered as a new
   tab beside the Q6_K GGUF lane (`flux2klein9b`). Deployed 2026-09-09; live
   smoke draw at 720×1440 with the base face → 200, OCR'd verbatim by the
   shorts dev (5/5 identity, 5/5 headline).
2. **VRAM-gate deadlock hardening** — closes the pooled-allocator deadlock
   class first seen 2026-09-08 (see below).

## Measured numbers (probe + canvas sweep, 2026-09-08/09)

Engine: `lite-infer/flux.2-klein-9b-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`,
`Flux2KleinPipeline.from_pretrained(repo, torch_dtype=bfloat16).to("cuda")`
(diffusers 0.40 nunchaku-lite; sm_120 OK). Text prompt through the resident
bnb4 Qwen3-8B encoder — no embed cache, no encoder parking.

| Metric | Value |
|---|---|
| Load | 9.1 s → 11.22 GiB torch-alloc at ready (driver ~11.5-12.3 GiB — nunchaku scratch outside torch's allocator) |
| Gen (T2I 1024², 4 st) | 4.5-5.9 s |
| Gen (ref attached, ≤1.05 MP, 4 st) | 6.1-8.1 s |
| Gen (ref, 1536×1024, 4 st) | 10.0 s |
| Determinism (same-seed rerun) | near-identical, NOT byte-identical (MAE 3.66 / RMSE 9.85) |
| vs Q6 lane | Q6 25.3 s/draw @4 st; same-seed cross-lane MAE 23.95 (different images — judge by eye) |

Canvas sweep (ref attached, out-of-band probe next to the idle service —
the LAB's probe context; the dev's own draws run IN the service, which is a
different, cheaper context — see the FHD section below):

| Canvas | Context | Verdict | Driver peak | Notes |
|---|---|---|---|---|
| 720×1440 | out-of-band | ✅ | 15,483 MiB | dev's "crash" canvas — fits (crash root-caused separately, below) |
| 1366×768 | out-of-band | ✅ | 15,323 MiB | renders **1360×768** — 1366 isn't a multiple of 16 |
| 1536×1024 | out-of-band | ✅ | 15,416-15,659 MiB | every-context safe ceiling (~0.3-0.6 GiB margin on this card) |
| full-res 1536×1024 ref, 720×1440 | out-of-band | ✅ | 15,399 MiB | unthumbed ref also fits |
| 1920×1072 / 1072×1920 | in-service (API) | ✅ 6/6 | 15,371-15,697 MiB | dev 4/4 + lab re-draw 2/2 on a clear card; steady 13.4 s/draw; ~0.6-0.9 GiB margin |
| 1920×1072 / 1072×1920 | second process | ❌ OOM | died at 15,660 MiB | 1,006 MiB request vs 503 MiB free — the second process's own context hits the 15.48 GiB per-process torch wall |

## FHD correction — the dev's full-HD 4/4 was real (and in-service only)

The 09-09 verdict ("1536×1024 ceiling, nothing bigger fits") extrapolated the
**out-of-band** measurement into an absolute; it holds for every context with a
second CUDA process, but the dev's production draws run in the SERVICE, where
that ~1 GiB of duplicated process context does not exist. Their full-HD probe
(1080×1920/1920×1080 → pipe floors to 1072×1920/1920×1072) returned 4/4 × 200 —
confirmed in journald (real 32,160-token renders, 13-16 s each; the 2.5 s/it
step time vs 1.95 s/it at 1536×1024 matches the 31%-bigger latent exactly).

Lab re-measurement with a card-wide watcher while the service itself drew
(no second CUDA process): cold FHD draw (incl. 8.7 s load) peaked **15,697 MiB**
in 21.6 s; steady-state FHD (resident) peaked **15,371 MiB** in 13.4 s — both
200, matching the dev's timings. Margin over the 16,311 MiB card total is
~0.6-0.9 GiB: real but ambient-sensitive. A second process drawing the same
canvas OOM'd (this run) — the lab's original ceiling stands for that context.

**Bottom line:** canvases up to 1536×1024 are safe everywhere. Full-HD draws
are service-context-only: fine on a clear card (6/6 evidence), 503s mid-gen if
anything else holds the card (TTS container resident, comfy sidecar busy,
pooled blocks after a heavy gen). The FHD typography win is real at native
resolution — production use should evict-all first and treat a mid-batch 503
as retry-at-1536.

## Crash reconciliation — the dev's "720×1440 NEW ALL-TIME HIGH"

Dev's 5-draw probe reported 720×1440 "crashing". Root cause (from the dev's
logs + code): **not a CUDA OOM and not the canvas** — the load gate refused:

```
503: FLUX.2 Klein 9B-KV needs ~10 GiB free VRAM; only 10452 MiB available
```

That 503 names the **Q6** klein9b gate (10,500 MiB) — the NVFP4 route wasn't
registered yet and its 13,000 MiB gate was never hit. 10,452 vs 10,500 is the
48-MiB-short signature of the pooled-allocator deadlock first seen 2026-09-08:
after a heavy qwenimage-edit gen, ~4.4 GiB of torch caching-allocator blocks
stayed pooled at the driver level while `STATE.active_engine` was None. The
unload/evict paths **no-op'd on nothing-resident** (unload only poked the
ComfyUI sidecar; `/evict-all` skipped `unload_engine()` entirely when
`prev is None`), so no code path ever called `empty_cache()` on the pool →
every engine's gate failed ~48 MiB short → 503s until a process restart
returned the segments.

## Hardening (3 changes, one commit)

1. **`_unload_current`** (image_lab_engines.py) — the nothing-resident branch
   now also runs `free_vram()` (gc + empty_cache + ipc_collect) before
   returning, instead of just poking the Comfy sidecar.
2. **`_ensure_vram_headroom`** (image_lab_engines.py) — the gate escalates:
   TTS eviction → if still short **and nothing resident**, force
   gc + `empty_cache()` + `ipc_collect()`, settle 1 s, re-measure → only then
   raise the 503. Converts the 48-MiB-short class into a pass whenever the
   pool is torch-releasable; a genuinely foreign tenant still gets the clear
   error.
3. **`/evict-all`** (image_lab_dispatch.py) — runs the unload path whenever
   not busy (resident or not), so an evict-all call actually returns pooled
   memory instead of no-op'ing on `STATE.active_engine` None.

Verified live: `/engines/unload` and `/evict-all` both hit the new reclaim
branch (journald "Nothing resident — releasing pooled VRAM …"); 11/11 engines
available after restart. The gate's internal reclaim fires only under the
deadlock condition — reproduced by the dev's probe traffic rather than
forced here.

## Engine notes

- Whole-card engine (qwenimage-edit class): `_VRAM_NEED_MB` 13,000 — encoder
  NEVER parks (resident with the transformer). In-process gen peaks measured
  lower than the out-of-band sweep (the service's own context is reused).
- Single-engine-at-a-time dispatch means the registered engine can never
  collide with a resident engine — the out-of-band second-process context is
  the only way two engines coexist, which is what made the earlier probe
  numbers ~0.8 GiB higher.
- Same-seed A/B across Q6/NVFP4 yields different images — quality verdicts
  must be by eye on matched prompts (user eyeball pending on
  `c:\tmp\klein9b_nvfp4_probe\`).
- Params: `reference_image` (file), `prompt`, `width`/`height` (multiple of
  16 renders exact; else rounded), `num_inference_steps` (4), `seed`.
  No guidance (step-distilled), no quant select.
- Canvas rule for the shorts pipeline: short 720×1440, long 1360×768 (request
  1360, not 1366) — 1536×1024 is the every-context safe ceiling. Full-HD
  1920×1072 / 1072×1920 is verified in-service only (FHD section above):
  evict-all first, accept the thin margin.
