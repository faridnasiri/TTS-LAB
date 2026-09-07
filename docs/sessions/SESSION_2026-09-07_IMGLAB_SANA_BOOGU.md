# SANA 1.6B + Boogu-Image 0.1 Turbo — Image Lab Integration

> 2026-09-06 landscape round; both engines verified **live on the service**
> 2026-09-07. Adds engines #6 (`sana`) and #7 (`boogu`) to the Image Lab.
> Survey + fit-screening + watchlist: [`MODEL_LANDSCAPE_2026-09-06.md`](../image-lab/MODEL_LANDSCAPE_2026-09-06.md).

## Scope (user decisions, locked)

| Decision | Outcome |
|---|---|
| SANA | Two variants on **one engine key** via the existing `quant` field: `sprint-1.6b` (SanaSprintPipeline, 1–4 steps, no CFG) + `1.5-1.6b` (SanaPipeline, ~20 steps, CFG 4.5). NOT 4.8B, NOT 4K. |
| Boogu-Image-0.1-Turbo-fp8 | Added with an explicit **CPU-offload exception** to the GPU-only policy (user-approved 2026-09-06) — the only engine with one. |
| Z-Image Turbo | **Dropped** — the Qwen3-4B encoder's embedding space is model-locked (no API serves it in embedding mode); revisit only if a diffusers-layout fp8/int8 checkpoint ever appears. |
| Removals | None. |

## Engines at a glance (measured values)

### `sana` — SANA 1.6B, two checkpoints, whole-pipeline bf16 GPU-resident

| | `sprint-1.6b` (default) | `1.5-1.6b` |
|---|---|---|
| Pipeline | `SanaSprintPipeline` + SCMScheduler | `SanaPipeline` + DPM scheduler |
| Steps | 1–4 (default 4; >4 clamped server-side with a log line) | 1–24 (default 20 via UI variant preset) |
| CFG / negative | none (guidance-free) | 4.5 default; negative prompt supported |
| Repo | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` |
| Disk | 9.74 GB | ~4.5 GB net (Gemma-2-2B-IT shards dedup against Sprint) |

Measured (1024²): **resident 8,968 MiB, gen proc peak 9,886 MiB, device peak 10,866 MiB**.
Gate recalibrated 12,000 → **11,500 MiB** (still > TTS-resident free ~10.8 GiB → evicts TTS
on load; < idle free ~14.3 GiB → leaves an idle TTS alone).

### `boogu` — Boogu-Image 0.1 Turbo fp8 ⚠️ CPU-OFFLOAD EXCEPTION

Qwen3-VL-8B-class mllm (**fp8**) + custom `BooguImageTransformer2DModel`
(**bf16-stored** `.bin` shards — the repo name's "fp8" is the mllm + a runtime flag) +
FLUX.1 VAE. Turbo: 4 steps, CFG **forced 1.0** → no negative prompt, no quant select.

Measured (1024², 4 steps): **GPU peak 12,640 MiB, RAM peak 26,814 MB** (60% gate of
64,283 MB), first load 383 s incl. ~14 GB cache top-up, first gen ~100 s (module
staging), warm gens 37–41 s. Gate recalibrated 13,000 → **13,200 MiB**. Cross-checked
against the upstream direct run (B2, one-off venv): 12,164 MiB / 26,643 MB / 105 s — close.

## The Sprint 4-step SCM blocker (found + fixed this session)

diffusers' `SanaSprintPipeline.__call__` defaults `intermediate_timesteps: float = 1.3`,
which the SCMScheduler accepts **only at exactly 2 steps** (the SCM max→1.3→0 jump;
enforced in both the pipeline's `check_inputs` and `scheduling_scm.set_timesteps`).
Every non-2-step Sprint request therefore 400'd out of the box.

Fix in `_generate_sana` (Sprint branch, `image_lab_engines.py`):

```python
elif steps != 2:
    # diffusers SanaSprintPipeline passes intermediate_timesteps=1.3 (its
    # signature default) to the SCMScheduler, which accepts that only at
    # exactly 2 steps (the SCM max->1.3->0 jump). For 1/3/4 steps the
    # scheduler requires intermediate_timesteps=None to fall back to the
    # linear max_timesteps->0 schedule the Sprint distiller was trained
    # on — otherwise EVERY non-2-step Sprint request errors out.
    kw["intermediate_timesteps"] = None
```

The 2-step path keeps the default 1.3. SCM `step()` is a generic Euler
(t = timesteps[i+1], s = timesteps[i]) so the linear fallback is legitimate.

## Verification results (live service, 2026-09-07)

### SANA — all green

| Check | Result |
|---|---|
| Schema | 2-option variant select on `quant`; steps max 24; boogu exposes no neg/quant |
| Sprint preload | 124 s cold; 9 s warm reload |
| 4-step gen (the previously-broken path) | valid 1.06 MB PNG (mean 113.2, std 71.7); warm 20.8 s, **no reload** (load_s None) |
| Same-seed rerun | pixel-identical sha, still no reload |
| steps=20 on Sprint | clamped to 4 → **pixel-identical** output + journal log line |
| 2-step SCM jump path | intact, differs from 4-step, valid image |
| 1.5 variant switch | reload stamped (load_s 118.08 s), output differs from Sprint |
| 1.5 20-step CFG gen | valid; negative prompt engaged — sha differs; green-fraction 0.031 with vs 0.041 without negative (**33% suppression**) |
| Peaks | resident 8,968 MiB / gen proc 9,886 MiB / device 10,866 MiB — all under gate |

### Boogu — all green

| Check | Result |
|---|---|
| Cross-eviction direction 1 | sana sprint resident → boogu load OK (dispatch evicted sana) |
| Preload | 383 s incl. ~14 GB first cache top-up |
| r1 gen 1024²·4-step seed 1 | 1,213,819-byte PNG (mean 78.9, std 67.1), 100.6 s |
| r2 same seed | **pixel-identical**, no reload (41.3 s) |
| r3 seed 2 | differs (std 65.6, 37.4 s) — 3 consecutive gens, no OOM, no creep |
| Peaks | GPU 12,640 MiB < 13.5 GiB gate; RAM 26,814 MB < 60% of 64,283 |
| Cross-eviction direction 2 | flux2klein (Q6_K) load evicted boogu; cleanup evict returned `{"evicted": true, "engine": "flux2klein"}` |
| UI (deployed VM) | 7 gallery filter options, SANA/Boogu labels, variant preset switch + reload-warning hook |

### Idle unload (900 s)

_Result appended post-write: see commit note — the engine auto-evicted within the
expected 900–960 s window and the journal logged the idle-eviction line._

## Disk incident — the single 630 GB root disk (2026-09-07)

`/opt/arthur-img-models` is **NOT a separate mount** (contra CLAUDE.md's wording): it
shares the one 630 GB `/dev/sda1` with the OS **and the whole TTS docker stack**.
`du -x` accounted only ~400 GB while the disk sat 100% full — ~230 GB was held by
open-but-deleted docker layers (invisible to du, freed only by prune).

During Boogu's `snapshot_download` the disk filled: ENOSPC, curls exited 23,
`date: write error: No space left on device` in logs. Frees that resolved it:

1. Deleted the stale **unreferenced** `models--diffusers--FLUX.2-dev-bnb-4bit`
   cache entry (32 GB — flux2 was removed 2026-08-13; the grep hits that made it look
   live were the `flux2klein` GGUF code matching the `FLUX2` prefix only).
2. `docker builder prune -af` → 87 GB → **79 GB free** (65 GB after Boogu's ~14 GB top-up).
3. Post-verification orphan cleanup: deleted the legacy `hub/` subdir (~69 GB of full
   duplicates downloaded by a one-off upstream script run) → **133 GB free (79%)**.

**Two HF cache layouts on the VM:** the service (hf_hub 1.16.1, `.env` sets only
`HF_HOME`) reads/writes **top-level `models--*`** directly under
`/opt/arthur-img-models/huggingface/`. The `hub/` subdir duplicates came from a
different hf_hub generation during the B2 upstream run. No code path, env template,
or systemd unit references `hub/` (verified before deletion); top-level holds full
copies of everything (SANA ×2 9.1 G each, Boogu 20 G, ideogram-4-nf4 16 G, Qwen3-8B
16 G). `quantized/` (44 GB: sd35 + wan NF4) is **live** — left alone.

## Footnotes

- **Boogu fused-op gate**: `block_lumina2.py:17` checks `is_triton_available() and
  ("cuda" in os.getenv("device", "cpu"))` — the lowercase `device` env var is unset
  in the lab, so it takes the torch RMSNorm fallback. Benign — the identical path B2
  validated. No action.
- **DeepGEMM**: `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` must precede any
  transformers import; set in the VM `.env` (loaded via systemd EnvironmentFile) and
  the deploy Phase 5 env template. The fp8 mllm additionally needs
  `kernels>=0.14,<0.15` (Phase 3) — the finegrained-fp8 pack resolves from HF at
  first fp8 load.
- **Python 3.11.0rc1**: the initial deploy installed the 3.11 **release candidate**
  (deadsnakes resolves "3.11" to the newest build incl. rc). Phase 1 now upgrades
  rc→final; the VM runs CPython 3.11.15.
- **Never torch.compile the Boogu pipeline** (vendor-documented all-black outputs).

## Change set

Config/UI/dispatch wiring (variant select rides the `quant` field so variant switches
reload), `boogu_lab_engine.py` (new, ideogram4-pattern engine module), deploy script
Phases 3/4/5/8 (Boogu git clone + `--no-deps` editable install, pre-download list,
env template, tail output), `.gitignore` + `Boogu-Image/`. Gate comments now carry
the measured numbers. Docs: this session file, the landscape doc, the reference doc
(new §4.4/§4.5 + corrected disk/storage sections), deploy-image-lab skill, README,
CLAUDE.md.
