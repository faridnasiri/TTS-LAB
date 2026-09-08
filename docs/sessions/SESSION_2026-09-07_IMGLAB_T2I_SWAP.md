# sd35+Wan Removal / 4-Engine T2I Swap — Image Lab Integration

> 2026-09-07/08 typography round. Removed **sd35 + Wan2.2** (dated latent VAE /
> video-first; the lab is all-image now) and added **Z-Image Turbo, Qwen-Image
> 2512, HiDream O1-Dev (ComfyUI sidecar), ERNIE-Image-Turbo** — all four verified
> **live on the service** with measured VRAM. The diffusers bump that unlocked
> ERNIE's NVFP4 path (0.38 → **0.40.0**) was regression-swept across all 6 kept
> engines. Research basis:
> [`OPENWEIGHTS_T2I_16GB_FIT_2026-09-07.md`](../image-lab/OPENWEIGHTS_T2I_16GB_FIT_2026-09-07.md)
> (this doc supersedes the landscape doc's "Z-Image DROPPED" row).

## Scope (user decisions, locked)

| Decision | Outcome |
|---|---|
| sd35 + Wan2.2 removal | **Removed** (code + VM files). SD 3.5 "dated — latent VAE blurs glyphs"; Wan is a video model, not text-first. Video *plumbing* stays (type-driven, harmless). |
| Z-Image Turbo | **Added** — the 2026-09-06 DROPPED verdict is superseded: a diffusers `ZImagePipeline` + GGUF route now exists (`jayn7/Z-Image-Turbo-GGUF`, Q4_K_M ~4.9 GB). |
| Qwen-Image 2512 | **Added** — GGUF Q4_K_S transformer (`unsloth`) + bnb-4bit Qwen2.5-VL encoder (`OzzyGT`), embed cache. |
| HiDream O1-Dev | **Added** via **ComfyUI sidecar** (own venv + `arthur-comfy.service`, port 8188) — no diffusers pipeline exists for O1 (UiT); only verified 16 GB path is Comfy native Dev fp8_scaled. |
| ERNIE-Image-Turbo | **Added** via **Nunchaku-Lite NVFP4** (`lite-infer/...-nunchaku-lite-nvfp4-bnb4-text-encoder`) — Blackwell-native, whole-pipeline repo; FP8 fallback ladder kept. |
| SANA + Boogu | **Kept** (user correction) — lab is now **9 engines, all image**. |
| Disk | sd35/wan VM files deleted after verification (Phase F, pulled forward). |
| Diffusers | **0.38 → 0.40.0** in the shared lab venv (needed for NunchakuLite) — all 6 kept engines regression-swept before relying on it. |
| TTS lab | Shut down for VRAM during ERNIE verification (user-approved); **restored** via compose afterward. |
| `DIFFUSERS_TRUST_REMOTE_KERNELS=true` | **User-approved** (see blockers — diffusers' nunchaku quantizer gate on remote CUDA kernels). |

## Engines at a glance (measured on the 5060 Ti, 2026-09-08)

### `ernie` — ERNIE-Image-Turbo, Nunchaku-Lite NVFP4 + bnb4 text encoder

| | |
|---|---|
| Route | `lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder`, whole-pipeline `from_pretrained` (transformer NVFP4/svdq-w4a4 4.71 GB + text_encoder bnb4-nf4 2.75 GB + vae 168 MB; the `pe` 3B-bf16 7.66 GB is dropped pre-CUDA). Fallback: `rootlocalghost/ERNIE-Image-Turbo-FP8` |
| Steps / CFG | **Forced 8 steps, CFG 1.0** — clamps log correctly (28→8, 4.0→1.0) |
| Measured | **Load 31.2 s** (25.7 s post-cache); 7.18 GiB torch-alloc at ready **but 12,418 MiB process resident** (nunchaku scratch outside torch's allocator); **card peak 12,993 MiB** @1024²/8 steps; warm gen **12.8 s** |
| Determinism | **Near-deterministic** — seed honored (same-seed mean-abs-diff 8.48 vs cross-seed 39.03 @64×64 grayscale) but nunchaku kernels never produce pixel-identical shas (4/4 runs differed). Engine property, not a seed bug. |
| Gate | 11,000 → **13,000** (see gates table). |

### `zimage` — Z-Image Turbo (diffusers `ZImagePipeline` + GGUF)

| | |
|---|---|
| Route | `ZImageTransformer2DModel.from_single_file` + GGUF Q4_K_M (`jayn7`, default tier) + in-repo Qwen3-4B encoder (encode→park, klein precedent) |
| Measured | **Ready 16.3 s, 4.90 GiB CUDA** at load; warm gen **48 s** @1024² (default 8-step turbo); verified 200 |

### `qwenimage` — Qwen-Image 2512 (GGUF Q4_K_S)

| | |
|---|---|
| Route | `QwenImageTransformer2DModel.from_single_file` + Q4_K_S GGUF (`unsloth`, **the only tier — higher ones OOM**, ladder collapsed) + bnb-4bit Qwen2.5-VL encoder (`OzzyGT`), embed cache, cache-miss encodes run with the transformer unloaded |
| Measured | Cold gen 247 s / **warm 191 s** @1024²/20 steps CFG 4.0; same-seed rerun **pixel-identical sha**; ready 35.8 s / 11.67 GiB CUDA; gen peak 12.19 GiB torch-alloc = **14,166 MiB driver-accounted** |
| Gate | **14,000** (needs the whole card; idle free ~14.2 GiB passes, a resident TTS model trips it → evicts). |

### `hidream` — HiDream O1-Dev (ComfyUI sidecar)

| | |
|---|---|
| Route | `hidream_comfy_bridge.py` → headless ComfyUI (`arthur-comfy.service`, 8188) with the official Dev template; `Comfy-Org/HiDream-O1-Image` Dev fp8_scaled (~8.1 GB); 28 steps, CFG 0.0 fixed, ≤2048² |
| Measured | **Byte-identical sha `eef9e75e…` (1,221,917 B) across a comfy unload/reload**; handoff works end-to-end: lab "Unloading HiDream…" → comfy `/free` → gate re-measure → evict-all → SANA load → 200 |
| Gate | 11,500 (checked after `/free`, so it measures the true idle free state). |

## The diffusers 0.40.0 bump (unlocked ERNIE — changed the shared venv)

- **0.40.0 (PyPI release, not git-main) ships `NunchakuLiteQuantizationConfig` +
  `NunchakuLiteQuantizer`** — the ERNIE NVFP4 path only needed 0.38→0.40, plus
  huggingface-hub ≥1.23 and accelerate ≥0.31 (installed with deps: hub 1.30.0,
  accelerate 1.13.0, safetensors 0.8.0; transformers 5.9.0 unchanged).
- `GGUFQuantizationConfig` moved to `diffusers.quantizers.quantization_config` but
  stays top-level exported — the loaders' `from diffusers import GGUFQuantizationConfig`
  is unaffected.
- **Keep-engine regression (all 6 verified 200 on 0.40, 2026-09-08):**
  sana 53.7 s ✓ · flux2klein9b 118 s ✓ · boogu 86 s ✓ (its pip pin
  `diffusers[torch]<0.39,>=0.35.2` is a metadata conflict only) · zimage 48 s ✓ ·
  qwenimage 226 s ✓ · ideogram4 cold 758 s / warm 244 s ✓ (below).
- **Deploy script must pin these versions** for reproducibility (sync pending in
  `deploy_image_lab.ps1` Phase 3).

## Blockers found + fixed this round

### 1. ERNIE's fake OOM → real cause: the kernels trust gate

Both fp8 and NVFP4 attempts reported "CUDA out of memory. Tried to allocate 32.00 MiB"
at ~14.0 GiB — an identical, misleading signature. The CPU-side census probe showed
the real root cause: `ValueError` from `diffusers/quantizers/nunchaku/utils.py:27` —
diffusers' nunchaku quantizer refuses to fetch/execute
`rootonchair/nunchaku-lite-kernels` (a remote CUDA kernel repo) unless
`DIFFUSERS_TRUST_REMOTE_KERNELS=true` is set. The OOM text came from the *error
handling path* around it. After the flag: NVFP4 loads fine; **fp8 genuinely OOMs**
(8.03 + 3.85 + vae ≈ 14.02 GiB > 14.28 usable) — it stays as the fallback ladder
rung only.

`DIFFUSERS_TRUST_REMOTE_KERNELS=true` + `HF_HUB_CACHE` were appended to the lab's
`.env` (root EnvironmentFile). The flag disarms a security gate — **user-approved**
after an explicit ask.

### 2. HF cache layout split (hub ≥0.20 vs direct) — `HF_HUB_CACHE` fix

hub ≥0.20 writes `$HF_HOME/hub/`; the **direct layout** (`$HF_HOME/models--*`)
holds the pre-September repos (flux2klein, ideogram4, Qwen3-8B …). hub 1.30 has
**no legacy fallback**, so a local direct-layout repo reads as MISSING under
`local_files_only` → would silently re-download ~40 GB. Fix: `HF_HUB_CACHE=
/opt/arthur-img-models/huggingface` in `.env` — the service reads the direct
layout again. (Footnote: `du` does not follow cache symlinks — use `du -shL`.)

### 3. Ideogram 4 cold-load 502 s — network catch-up, not a regression

The 8.4-min load during the regression sweep was **2×5.2 GB re-downloaded**
(transformer + unconditional_transformer — two blob symlinks had gone missing from
the 16 GB snapshot; hub re-fetched just those blobs into the same snapshot
`f6643478…`, self-healing it). Since then: warm gens 236 s @1024²/28 steps.

### 4. Same-seed ≠ same image on bnb nf4 (ideogram4) — documented, not a bug

Cold `d85e46d6…` vs warm `d130cc3d…` for the same seed 12/prompt — bnb nf4 matmul
kernel noise (same class as ernie's nunchaku). Pre-existing engine property.

## Verification matrix (live service)

| Check | Result |
|---|---|
| /status schema | 9 engines, all probed; sd35/wan absent; UI gallery + sidebar updated |
| zimage | cold 200 (load 16.3 s) · warm 200/48 s · GGUF on 0.40 ✓ |
| qwenimage | cold 200/247 s · warm 200/191 s · **sha identical** · embed cache ✓ |
| hidream | 200, sha `eef9e75e` identical across checkpoint reload · comfy `/free` handoff → sana 200 |
| ernie | cold 200 (load 31.2 s) · warm 200/**12.8 s** · clamps logged · gate 13000 cold 200/40.9 s post-restart |
| ideogram4 (regression) | cold 200/758 s (502 s = re-download) · warm 200/244 s (236 s gen) · process peak **14,036 MiB** mid-gen (CFG pulls the unconditional transformer back) |
| sana / flux2klein9b / boogu / zimage / qwenimage (regression) | all 200 on 0.40 — see bump section |
| Cross-eviction | every sweep leg evicted the previous engine (journal "Unloading engine: …") · hidream↔sana handoff ✓ |
| 3 consecutive ernie gens | 200/200/200, **no reload** (single "ready in" line), process pinned 12,418 MiB, card 13,092→13,227 MiB (delta = two new 128 MiB TTS container contexts — foreign, not creep) |
| TTS co-tenancy | TTS compose restored (engine-current/qwen/orchestrator healthy); ernie gens pass with TTS contexts live |
| Idle eviction | **PASSED** — `03:59:39 Idle eviction: ernie unused for 912s — unloading from GPU` (900 s threshold + 60 s check cadence); process 12,418 → 188 MiB, card 997 MiB |

## VRAM gates (recalibrated from measured peaks, 2026-09-08)

| Engine | Before | After | Basis |
|---|---|---|---|
| ernie | 11,000 | **13,000** | resident 12,418 MiB / card peak 12,993 MiB (nunchaku scratch outside torch allocator → torch-alloc alone understates) |
| ideogram4 | 12,000 | **12,500** | governs load only (5,953 MiB used at ready after offloads) but **CFG gen peaks 14,036 MiB** — comment corrected; the engine needs the whole card mid-gen |
| zimage | 10,500 | 10,500 | comment now measured: 4.90 GiB CUDA ready / 48 s warm |
| qwenimage | 14,000 | 14,000 | comment now notes 12.19 GiB torch-alloc = 14,166 MiB driver-accounted |
| sana / boogu / hidream | — | unchanged | already measured (sana 8,968/9,886/10,866 · boogu 12,640 · hidream verified through gate after `/free`) |

All gates ship a measured-VRAM comment — the file is the number source of truth.

## Disk cleanup

1. **Phase F (2026-09-07, after verification):** sd35/wan VM files → **+113 GB**.
2. **2026-09-08:** dead `qwen-image-2512-Q4_K_M.gguf` (13 GB — Q4_K_S is the only
   live tier) + **8 hub-layout duplicates** (~82 GB: Boogu 20 G, rootlocalghost
   15 G, lite-infer 15 G, SANA×2 18.2 G, Tongyi-MAI 7.8 G, OzzyGT 5.9 G, Qwen-2512
   248 M — each has a live direct-layout twin the service now reads via
   `HF_HUB_CACHE`). → **df 93% → 78% (140 GB free)**.
3. **Kept in `hub/`** (25 GB, no direct twins, re-download protection):
   `unsloth/Qwen-Image-2512-GGUF` 13 G, `jayn7/Z-Image-Turbo-GGUF` 4.7 G,
   `Comfy-Org/HiDream-O1-Image` 7.6 G.

## Footnotes

- **ERNIE near-determinism** and **ideogram4 bnb nf4 non-determinism** are engine
  properties of the quantized kernels — never chase pixel-identical shas on these
  two; judge by seed-family similarity.
- **Ideogram 4 gen is CFG**: the load-ready state offloads the unconditional
  transformer, but every CFG step pulls it back → the 14,036 MiB mid-gen peak.
  It is the tightest engine on the card; single-engine-at-a-time dispatch covers it.
- **hub cache layout**: pre-September repos live direct (`models--*`); new hub
  versions write `hub/` unless `HF_HUB_CACHE` is set. The env var now makes the
  service read the direct layout — do not delete direct-layout repos expecting a
  `hub/` fallback; hub 1.30 has none.
- **`du` vs symlinks**: `du -sh` reports stub sizes for cache dirs — always
  `du -shL`.
- **qwenimage Q4_K_M OOMs** (12.34 GB file was the pre-verification default) —
  the loader ladder now exposes Q4_K_S only.

## Change set

Config/dispatch/UI/entry wiring for the 9-engine all-image lab; `_load/_generate/
_probe` triples for zimage/qwenimage/ernie (diffusers GGUF + NunchakuLite paths),
hidream comfy bridge (`hidream_comfy_bridge.py`, new), `ideogram4_lab_engine.py`
(sweep-era fixes); `_VRAM_NEED_MB` recalibration + measured comments; deploy script
Phases 3/4/5/8 (diffusers ≥0.40.0 + hub ≥1.23 pins, repo list swap, `.env`
additions, new modules); sd35/wan + nvfp4diag removal (code, scripts, docs);
`gguf_download.py`/`nvfp4_save.py`/`preq_save.py`/`bench_image_lab.py` updates;
docs (this file, OPENWEIGHTS implementation markers, MODEL_LANDSCAPE supersession
note, reference docs, README, CLAUDE.md, deploy skill). Committed in two rounds —
code first, then "verified live" with these numbers.
