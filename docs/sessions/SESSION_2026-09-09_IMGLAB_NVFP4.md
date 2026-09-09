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

## 2026-09-09 addendum — canvas ladder with OmniVoice resident (TTS-only cleanup aftermath)

User stopped qwen/mid/editx containers (OmniVoice-only TTS) → engine-current
stays up with OmniVoice resident (~2.4 GiB card ambient vs ~0.87 GiB on the
clear card). Q: does near-FHD still OOM in that state? Measured through the
live :8002 API, card-wide watcher per draw, ref attached, 4 steps, fixed seed
(lab re-draw of the dev probe methodology):

| Canvas | Context | Verdict | Sampled card peak | Note |
|---|---|---|---|---|
| 720×1440 (1.04 MP) | OmniVoice resident | ✅ 200 | 15,712 MiB | ~600 MiB margin |
| 1360×768 (1.04 MP) | OmniVoice resident | ✅ 200 | 15,832 MiB | prod long canvas — ~480 MiB margin |
| 1024×1024 (1.05 MP) | OmniVoice resident | ❌ 503 | 15,798 MiB | CUDA OOM, 512 MiB request refused — same pixel area as the two ✅s: the ~1.04 MP boundary is razor-thin and allocation-order-dependent |
| 1536×1024 (1.57 MP) | OmniVoice resident | ❌ 503 | 15,696 MiB | **the previous every-context ceiling no longer fits** |
| 1792×1008 (1.81 MP) | OmniVoice resident | ❌ 503 | 15,836 MiB | nunchaku kernel OOM (Tensor.h:95) |
| 1920×1072 FHD (2.06 MP) | OmniVoice resident | ❌ 503 | 15,838 MiB | CUDA OOM, 256 MiB request refused |

**Verdict: the ambient tenant drops the NVFP4 canvas ceiling from 2.06 MP
(clear-card FHD, 6/6) to ~1.04 MP.** The +1.6-1.9 GiB of card OmniVoice holds
shrinks the imglab process's driver-allocatable ceiling from ~14.8 GiB (what
an in-service FHD gen needs) to ~13.2-13.4 GiB; 720×1440/1360×768 gens land
right at that wall (peaks 15,712-15,832 card-wide → only ~480-600 MiB of
16,311 left), and any canvas whose working pattern needs a single
allocation larger than the leftover (~480-600 MiB, or its aspect's peak
tensor) OOMs mid-gen. The 1.04 MP class itself is marginal: 1024×1024 failed
where same-area 720×1440/1360×768 succeeded — a 512 MiB transient vs ~480 MiB
free at that instant. All four OOMs surfaced as clean 503s (RuntimeError
mapping), zero service restarts, 11/11 engines stayed available, and the
generate() error path (`_unload_current()` at image_lab_engines.py ~2456)
released the engine after each failure — card fell from ~15.8 to ~3.9-4.2 GiB
post-failure, so the card was never bricked.

**Baked 2026-09-09 (user call — "auto evict, no other tenants when the
engine works, OK with OmniVoice cold loading"):**

1. **Load gate 13,000 → 15,000 MiB** (`_VRAM_NEED_MB["flux2klein9b-nvfp4"]`)
   — sits ~350-450 MiB under the post-eviction max free (~15.35-15.45 GiB)
   and above every TTS-resident free reading (OmniVoice: 13,876), so ANY
   NVFP4 load auto-evicts the TTS tenant and a cleared card passes
   untouched. Idle TTS containers whose CUDA contexts alone hold ~1.5-2 GiB
   now get an honest gate 503 instead of a mid-gen OOM.
2. **Warm tenant guard** (`_evict_tts_tenant_if_loaded` in generate() for
   warm draws, `_WARM_TENANT_GUARD_KEYS`) — the load gate never re-runs on
   a warm gen, so a TTS model that loaded since the last draw (card free
   ~3.1 → ~1.5 GiB with NVFP4 resident) would OOM the gen mid-flight. The
   warm path asks the orchestrator /status whether any TTS model is loaded
   (deterministic — a free-VRAM floor can't separate a tenant from the
   engine's own post-draw pooled blocks) and evicts first.
3. Gate 503 message now states the need in exact MiB ("requires ≥ 15,000
   MiB free") — the old "~14 GiB" integer-division wording contradicted
   itself at the new value.
4. Cost: every cold NVFP4 load evicts OmniVoice (~30 s reload on the next
   TTS synth — accepted). Warm draws within a batch pay one cheap
   orchestrator /status check each. The whole-card co-tenant OOM class is
   unchanged for OTHER whole-card engines (ernie, ideogram4, boogu, sana —
   their gates still pass the OmniVoice-resident state); add each to
   `_WARM_TENANT_GUARD_KEYS` + raise its gate as measured, if the user
   wants the same guarantee there.

**Live verification after the bake (2026-09-09):**

| Step | Result |
|---|---|
| Cold FHD 1920×1072 draw with OmniVoice resident | ✅ 200 in 31.8 s (incl. ~9 s load) — gate fired ("VRAM free 13,8xx < 15000 — evicting"), OmniVoice auto-unloaded, card peak 15,696 MiB |
| Warm FHD draw (tenant gone) | ✅ 200 in 15.2 s, peak 15,054 MiB — guard no-op'd (nothing to evict) |
| OmniVoice TTS synth while NVFP4 still resident | ❌ **500 CUDA OOM** — see gap below |
| Warm FHD draw after the failed TTS reload | ✅ 200 in 15.4 s (no tenant was ever loaded — the reload had failed) |

**Gap found — the reverse direction does not exist.** Nothing on the TTS
side can evict the image lab: engine-current's lazy load OOM'd trying to
fit OmniVoice (~1.14 GiB chunk vs ~794 MiB free) with NVFP4 resident
(~12.3 GiB), and the orchestrator dispatch has no image-lab hook (verified
in /opt/arthur-tts-lab/tts_lab_dispatch.py — the LLM-era evictions there
only cover TTS containers/LLM). The imglab idle-unload is 900 s, so NVFP4
squats the card up to 15 min after a batch → any TTS synth in that window
500s until the engine idles out or is unloaded manually. Both the
orchestrator and engine-current CAN reach the imglab from the compose
bridge at http://192.168.0.87:8002 (verified HTTP 200) — a reverse hook is
mechanically trivial but lives in the TTS-side files (other session's WIP).
Recommended shape (for the HEAVY class): orchestrator dispatch evicts
imglab (`POST /engines/unload`) before dispatching heavy TTS loads,
mirroring the existing `_stop_llm_container` pattern and the
imglab→orchestrator convention.

## 2026-09-09 addendum 2 — cohabitation instead of a reverse hook (user design, verified E2E)

User's counter-proposal to the reverse hook: "model load itself isn't
harmful — can the post-draw bloat be emptied so only the model remains,
then OmniVoice loads beside it? Next image evicts TTS. Is that logical?"
Answer verified live: **yes — for the small-tenant class.** The gap's OOM
was the POST-DRAW POOL, not the resident model: a fresh NVFP4 load leaves
~3.1-4.1 GiB free; the pool collapses it to ~0.5-1.2 GiB. The pool is
torch caching-allocator blocks — releasable via empty_cache without
unloading the model (nunchaku scratch is outside torch but measured
releasable too: the full working set returns).

**Baked:** `_compact_after_draw()` (image_lab_engines.py) — after every
successful guard-set draw, gc + empty_cache + ipc_collect while the engine
stays resident; logs the free-MiB delta. ~10 lines, image-side only, no
TTS-side change.

**Live E2E (2026-09-09, all through the live APIs, card-wide watcher):**

| Step | Result | Numbers |
|---|---|---|
| S1 cold OmniVoice load | ✅ 200 | 8 s, peak 2,469 MiB |
| D1 cold FHD 1920×1072 with OmniVoice resident | ✅ 200 | 30 s, gate fired (free 13,280 < 15,000 → evict-all freed 1,960 MiB), peak 15,696 |
| **Post-draw compaction** | ✅ | **released 2,878 MiB (free 796 → 3,674 MiB)** — engine kept resident, card ~12.2 GiB used |
| S2 synth beside resident NVFP4 | ✅ **200** | 9 s, peak 14,168 — the co-load the gap said was impossible |
| D2 warm FHD draw (tenant loaded again) | ✅ 200 | 18 s, guard fired ("tenant loaded — evicting TTS before warm gen", freed 1,960 MiB), peak 15,698 |
| Post-draw compaction | ✅ | released 3,212 MiB (free 462 → 3,674 MiB) |
| S3 synth again | ✅ 200 | 9 s, peak 14,168 |

Both-resident state: card 14,134 MiB used (~2.1 GiB free) — OmniVoice
synth ran comfortably in it. The floor is deterministic: free lands on
3,674 MiB after every draw, every time.

**Verdict:** the cycle the user proposed is the contract now — draws
always get the clear card (gate 15,000 on cold loads + the warm tenant
guard), and between draws the engine sits lean (~12.2 GiB) so a ~2 GiB
TTS tenant co-loads beside it instead of 500ing. No squat window, no
reverse hook needed, no TTS-side edit. Boundary: cohabitation only fits
tenants ≲ 2-3 GiB (OmniVoice ✓). Heavy engines (EditX ~12.8 GB AWQ alone,
s2pro) can never cohabit with a 12.3 GiB NVFP4 — for THAT class the
reverse hook (imglab unload from the orchestrator dispatch) remains the
completion if those containers come back. One per-draw cost: every draw
re-ascends from a cold pool (measured cold-start FHD peak 15,696 MiB —
still 200, ~0.6 GiB margin; per-draw overhead ~0.1-0.2 s).

State left after verification: NVFP4 unloaded, OmniVoice reloaded, card
2,630 MiB, 11/11 engines available.

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
