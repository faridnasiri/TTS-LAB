# Image Lab — Deployment State & Session Reference

> **Snapshot 2026-09-08 (VM clock)** — the 9-engine all-image lab as it stands
> after the T2I swap round. One reference for *what is running, how fast it
> generates, what broke and how it was fixed*, plus the ops state to operate
> and re-deploy it. Live-verified on the RTX 5060 Ti 16 GB (sm_120) at
> `arthur@192.168.0.87`.
>
> Deep-dive companions (this doc is the consolidation, they are the detail):
> [`SESSION_2026-09-07_IMGLAB_T2I_SWAP.md`](../sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md)
> (sd35/wan removal + 4-engine typography swap), [`SESSION_2026-09-07_IMGLAB_SANA_BOOGU.md`](../sessions/SESSION_2026-09-07_IMGLAB_SANA_BOOGU.md)
> (SANA/Boogu integration + disk incident), [`FLUX2_KLEIN_9B_KV.md`](FLUX2_KLEIN_9B_KV.md)
> (Klein 9B-KV quant identity/VRAM study), [`IDEOGRAM4_FIX_2026-08-14.md`](IDEOGRAM4_FIX_2026-08-14.md)
> (blank-image root cause), [`OPENWEIGHTS_T2I_16GB_FIT_2026-09-07.md`](OPENWEIGHTS_T2I_16GB_FIT_2026-09-07.md)
> (engine-selection research).

## 1. Deployment at a glance

| What | Value |
|---|---|
| Service | `arthur-imglab.service` (systemd) — FastAPI on **port 8002**, `/opt/arthur-img/image_lab.py` |
| Sidecar | `arthur-comfy.service` — headless ComfyUI on **port 8188** (HiDream O1-Dev only), own venv `/opt/arthur-img-comfy-env` |
| Lab venv | `/opt/arthur-img-env` (Python 3.11.15, diffusers **0.40.0**, transformers 5.9.0, hub 1.30.0, accelerate 1.13.0) |
| TTS co-tenant | `docker compose` stack in `/opt/arthur-tts-lab` (orchestrator 8009 + engine containers 8101–8104) — **restored and healthy** since the ERNIE round |
| `.env` (root EnvironmentFile) | `HF_HOME` + **`HF_HUB_CACHE=/opt/arthur-img-models/huggingface`** + `DIFFUSERS_TRUST_REMOTE_KERNELS=true` + `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` (+ HF token, API keys) |
| Code path | `/opt/arthur-img/` — deploy via `scripts/deploy/deploy_image_lab.ps1` (Phase 5 code-only) |
| Models | `/opt/arthur-img-models/` — HF direct layout (`models--*`) + `gguf/` (GGUF_ROOT) — **not a separate mount** (single 630 GB root disk, see §7) |
| Gallery output | `/opt/arthur-img/…` web-served under the UI ("All engines" tab) |

## 2. Working-model roster — 10/10 green (probed + loaded this round)

Order = sidebar/gallery order. `quant` = engine default tier (UI "quant" field
when exposed). All measured on the 5060 Ti 16 GB.

| # | key | Label | Model / route | Default quant | Load-ready VRAM | Gate (MiB) | Status |
|---|---|---|---|---|---|---|---|
| 1 | `flux2klein` | FLUX.2 Klein | `black-forest-labs/FLUX.2-klein-4B`, **BF16** `from_pretrained` (no GGUF) — encode-then-park Qwen3 encoder | — (BF16) | 7.39 GiB | 10,500 | **Working** — verified 2026-09-08 |
| 2 | `flux2klein9b` | FLUX.2 Klein 9B-KV | GGUF transformer via `from_single_file` + Qwen3-8B NF4 encoder + Flux2 VAE | **Q6_K** (7.9 GB, default since 09-04) | ~7.55 GiB transformer | 10,500 | **Working** — Q8_0 documented-OOM rung |
| 3 | `ideogram4` | Ideogram 4 | `ideogram-ai/ideogram-4-nf4` (bnb nf4, `ideogram4_lab_engine.py`), hosted magic-prompt caption ladder | nf4 | 5,953 MiB after offloads | 12,500 | **Working** — gen needs whole card (CFG) |
| 4 | `sana` | SANA 1.6B | `SanaSprintPipeline` 1–4-step (**sprint-1.6b** default) / `SanaPipeline` 20-step (1.5-1.6b), both bf16 GPU | `sprint-1.6b` | 8,968 MiB | 11,500 | **Working** — SCM fix baked in |
| 5 | `boogu` | Boogu-Image 0.1 Turbo | Qwen3-VL-8B-class fp8 mllm + bf16 transformer + FLUX.1 VAE — **CPU-offload exception** (only engine) | fp8 | GPU 12,640 MiB peak / 26.8 GB RAM | 13,200 | **Working** — 4-step, CFG 1.0 |
| 6 | `zimage` | Z-Image Turbo | `ZImageTransformer2DModel.from_single_file` + GGUF + in-repo Qwen3-4B encoder (encode→park) | **Q4_K_M** (4.9 GB) | 4.90 GiB | 10,500 | **Working** |
| 7 | `qwenimage` | Qwen-Image 2512 | GGUF transformer + bnb-4bit Qwen2.5-VL encoder (`OzzyGT`), embed cache | **Q4_K_S** (11.5 GB — *only* tier) | 11.67 GiB | 14,000 | **Working** — needs the whole card |
| 8 | `qwenimage-edit` | Qwen-Image Edit | Qwen-Image-**Edit**-2511 (sibling of #7): native instruction **edits on an uploaded reference image** (identity + in-image text). Same GGUF Q4_K_S + **same** OzzyGT bnb4 encoder; image-conditioned embeds (vision tokens via Qwen2VL processor) keyed by prompt + image bytes | **Q4_K_S** (11.56 GB — *only* tier) | 11.81 GiB | 14,000 | **Working** — NEW 2026-09-08 (spike GO); whole card + more (~4 min/gen, see §3) |
| 9 | `hidream` | HiDream O1-Dev | **ComfyUI sidecar** via `hidream_comfy_bridge.py` → `Comfy-Org/HiDream-O1-Image` Dev **fp8_scaled** (~8.1 GB), 28 steps CFG 0.0 | fp8_scaled | (sidecar) | 11,500 | **Working** |
| 10 | `ernie` | ERNIE-Image-Turbo | Nunchaku-Lite **NVFP4** whole-pipeline (`lite-infer/...`), FP8 fallback ladder | NVFP4 | 12,418 MiB process resident | 13,000 | **Working** — 8 steps, CFG 1.0 |

Reference-capable engines (`image_input="reference"`, uniform `reference_image`
upload param — upload the base image with every request, nothing is stored):
**#1/#2/#8**. The other seven are text-only and **reject** a reference upload
with a 400 naming the native alternatives (never silently ignored).

Removed 2026-09-07 (not coming back): **sd35** + **Wan2.2** (dated latent VAE /
video-first; 113 GB of files deleted after verification). LLM/Qwen36 retired
2026-08-23 — the lab is 10 engines, all image; video *plumbing* stays
type-driven for a future engine. Session detail:
[`SESSION_2026-09-08_IMGLAB_QWENIMAGE_EDIT.md`](../sessions/SESSION_2026-09-08_IMGLAB_QWENIMAGE_EDIT.md).

## 3. Measured benchmark table (1024² unless noted)

Wall-clock via curl against the live service. **Methodology matters — read the
date tags:** `cold` = load+gen (engine not resident), `warm` = engine resident,
`ready` = load-to-serve only. "This round" = 2026-09-07/08 (VM clock).

| Engine | Load → ready | Warm gen | Cold wall | Determinism (same seed) |
|---|---|---|---|---|
| `flux2klein` (Klein 4B, BF16, 20 st CFG 3.5) | **12.9 s** / 7.39 GiB | **52 s** (gen only) | **77 s** (incl. ~12 s first encode-cache miss) | **pixel-identical** (`1ee1ab90`) — measured today |
| `flux2klein9b` (Q6_K, 4 st) | 12–24 s (within swap runs) | **25.3 s** (09-05) | 36–47 s swap runs (09-05); **118 s** cold on the 0.40 regression (09-08, incl. load) | quant-stable identity (cosine 0.958–0.985 Q4↔Q6) |
| `ideogram4` (nf4, 28 st) | ~90 s (cold first-run after cache heal ~500 s) | **236 s** gen / 244 s warm wall | 758 s cold (502 s = one-off re-download) | **NO** — bnb nf4 kernel noise (engine property) |
| `sana` sprint (4 st) | 124 s cold preload / **9 s warm reload** | **20.8 s** | 53.7 s (0.40 regression cold) | pixel-identical |
| `sana` 1.5 (20 st, CFG 4.5) | (same engine key) | not re-timed | reload 118 s | differs from Sprint (as designed) |
| `boogu` (4 st) | 383 s first (incl. 14 GB cache top-up) | **37–41 s** | 86 s (0.40 regression cold) | pixel-identical (r2) |
| `zimage` (Q4_K_M, 8 st turbo) | **16.3 s** / 4.90 GiB | **48 s** | 48 s class (0.40 regression row) | n/v this round |
| `qwenimage` (Q4_K_S, 20 st CFG 4.0) | 35.8 s / 11.67 GiB | **191 s** | 247 s | **pixel-identical** |
| `qwenimage-edit` (Q4_K_S, 20 st CFG 4.0, ref attached) | **20.5 s** warm / 11.81 GiB (51.2 s cold GGUF) | **223–241 s** gen (12.06 s/step — two-channel conditioning doubles the DiT seq to ~8.2K tokens) | **309.6 s** measured live on the first request (load 20.8 + image-keyed encode + reload + gen; gallery `total_s`) | **pixel-identical** — sha `9663ed91baf22a5e` (probe rerun) + run1/run2 same-seed compare on the live service |
| `hidream` (fp8_scaled, 28 st) | sidecar-owned (comfy) | not re-timed at 1024² this round — ~80 s class @2048×1376 per the research doc's 5060 Ti reference | — | **byte-identical** sha `eef9e75e` across comfy unload/reload |
| `ernie` (NVFP4, 8 st CFG 1.0) | 31.2 s cold-cache / **25.7 s** warm | **12.8 s** | 40.9 s (gate-13000 cold, post-restart) | **near** — seed honored but nunchaku kernels never pixel-identical (4/4 runs differed) |

VRAM peaks at generation (the number that governs gates, §4): klein-4B 12,724 MiB
process (today) · klein9b Q6 **14,234 MiB used / 1,615 free** · ideogram4 CFG
**14,036 MiB** · qwenimage 12.19 GiB torch-alloc = **14,166 MiB driver** ·
**qwenimage-edit torch 12.70 GiB = 15,400 MiB driver-total / 14.64 GiB pid —
the whole-card record, margin 0.53 GiB at both 1024² and 720×1440** · ernie
**12,993 MiB card peak** · boogu GPU 12,640 + 26.8 GB RAM · sana 10,866 · zimage
~5.3 GiB class. Canvas ceiling for qwenimage-edit: ~1.05M px (1024² and
720×1440 both verified; the API 400s anything larger).

## 4. VRAM gates + budget (how the 16 GB card is shared)

- Card: torch-total **15.48 GiB** (nvidia-smi 16,311 MiB). Idle floor ~997 MiB
  (lab CUDA context); each resident TTS engine container adds a **128–260 MiB**
  idle context even with no model loaded (lazy-load on synth request).
- **Gate semantics:** `_VRAM_NEED_MB` governs **load** only, against
  `torch.cuda.mem_get_info()` free MiB. Idle free ~14.2 GiB → gates ≤ ~10.5 pass
  untouched; a resident TTS model (~10.8 GiB free) trips gates ≥ ~11.5 → the lab
  escalates to the TTS evict-all endpoint first. **Generation peaks may exceed
  the load gate** (ideogram4 CFG = 14,036 mid-gen vs 12,500 gate) — the
  single-engine dispatch + 14 GB free idle is what makes that safe.
- All 9 gate values are in the table above; `image_lab_engines.py`
  `_VRAM_NEED_MB` (L507-554) carries the measured-VRAM comment per key — the
  file is the number source of truth.
- **Eviction paths:** every load evicts the previous lab engine ("Unloading
  engine: …" journal line) → TTS evict-all only when the gate trips → comfy
  `/free` for the sidecar (bridge + `_ensure_engine` poke it when reachable).
  **Idle eviction 900 s** (60 s check cadence; only generate/load touch the
  last-used stamp — `/status` polls do not): verified 3× — sana at 933 s,
  ernie at 912 s (`03:59:39 Idle eviction: ernie unused for 912s`; process
  12,418 → 188 MiB, card → 997 MiB).

## 5. Determinism matrix (never chase shas on the two noisy ones)

| Engine | Same-seed rerun | Notes |
|---|---|---|
| klein ×2, sana, boogu, qwenimage, hidream | **identical** | hidream byte-identical across a full checkpoint reload |
| ernie | near (mean-abs-diff 8.48 vs 39.03 cross-seed) | nunchaku NVFP4 kernel noise — judge by seed-family similarity |
| ideogram4 | **differs** (cold `d85e46d6` vs warm `d130cc3d`, seed 12) | bnb nf4 matmul noise — engine property, pre-existing |

## 6. Issues + fixes log (consolidated, newest last)

| Date | Issue | Root cause | Fix / state |
|---|---|---|---|
| 2026-06-10→08-14 | Ideogram 4 blank gray images | caption starvation — text <1% of the caption → nothing to render | hosted magic-prompt API auto-expansion + seed randomization (`ideogram4_lab_engine.py`); the fix doc is the reference |
| 09-07 | SANA Sprint 4-step requests 400 out of the box | `SanaSprintPipeline` passes `intermediate_timesteps=1.3` which SCMScheduler accepts only at exactly 2 steps | `_generate_sana`: `intermediate_timesteps=None` for every non-2-step count (linear max→0 fallback the distiller was trained on) |
| 09-07 | Disk 100% full mid-`snapshot_download` (ENOSPC, curl exit 23) | `du -x` hid ~230 GB of open-but-deleted docker layers; also a stale 32 GB `FLUX.2-dev-bnb-4bit` cache entry | `docker builder prune -af` (87 GB) + deleting the stale entry + a legacy `hub/` dupe subdir (69 GB) → 79% free |
| 09-07/08 | ERNIE "CUDA OOM 32 MiB" at ~14 GiB — fake signature | diffusers' nunchaku quantizer **refuses remote CUDA kernels** (`utils.py:27`) unless `DIFFUSERS_TRUST_REMOTE_KERNELS=true`; OOM text came from the error-handling path | env flag added (**user-approved** — disarms a security gate); NVFP4 loads clean after it. (fp8 genuinely OOMs: 8.03+3.85+vae ≈ 14.02 GiB > usable — fallback rung only) |
| 09-08 | 9 engines "missing" after hub upgrade | cache-layout split: hub ≥0.20 writes `hub/`, pre-September repos live **direct** (`models--*`); hub 1.30 has no legacy fallback | `HF_HUB_CACHE=/opt/arthur-img-models/huggingface` in `.env` → service reads the direct layout again |
| 09-08 | qwenimage Q4_K_M (12.34 GB) loads-then-OOMs at `to("cuda")` | ~14.2 GiB process floor vs 15.48 GiB card already hosting ~1.2 GiB TTS contexts (measured 3×) | ladder collapsed to **Q4_K_S-only** (`_QWENIMAGE_GGUF`); embed cache + cache-miss encodes run with the transformer unloaded; 13 GB dead Q4_K_M GGUF deleted |
| 09-08 | ideogram4 8.4-min cold load during the regression sweep | network catch-up: 2×5.2 GB blobs re-fetched (missing symlinks self-healed by hub) — not a diffusers 0.40 regression | none needed (one-off); warm gens 236 s after |
| 09-08 | deploy script would re-download the dead Q4_K_M | `deploy_image_lab.ps1` Phase 4 GGUF pre-warm still listed `qwen-image-2512-Q4_K_M.gguf` | **fixed today** — entry retargeted to Q4_K_S with an OOM-rationale comment (zimage's Q4_K_M entry stays — it is the live default tier) |
| 09-08 | `journalctl` empty for `arthur-imglab` | arthur not in `adm`/`systemd-journal` — root service logs invisible | read with `sudo journalctl -u arthur-imglab` (§9) |

Cross-cutting: the **diffusers 0.38 → 0.40.0** bump (unlocked NunchakuLite /
ERNIE) was regression-swept across all 6 kept engines before relying on it —
all 200. `GGUFQuantizationConfig` moved modules but stays top-level exported.
Deploy Phase 3 pins `diffusers>=0.40.0` + hub ≥1.23.

## 7. Cache layout + disk state (as of this round)

- **Direct layout is live** (`HF_HUB_CACHE`): `models--*` at the huggingface
  root holds flux2klein, ideogram4 (16 GB nf4), Qwen3-8B, SANA ×2, Boogu 20 G,
  lite-infer, rootlocalghost, Tongyi-MAI, OzzyGT encoder, ERNIE repo. **Do not
  delete direct repos expecting a `hub/` fallback** — hub 1.30 has none.
- Kept in `hub/` (25 GB, hub-only stubs — deleting would re-download):
  `unsloth/Qwen-Image-2512-GGUF` 13 G, `jayn7/Z-Image-Turbo-GGUF` 4.7 G,
  `Comfy-Org/HiDream-O1-Image` 7.6 G.
- `gguf/` root: klein9b Q6_K 7.9 GB + ladder (Q3_K_M 4.6 → Q8_0 10.0 — Q8 stays
  as a documented "bigger card / encoder-offload" rung, user decision), zimage
  Q4_K_M 4.9 GB (+Q4_K_S…Q6 ladder), qwenimage Q4_K_S 11.5 GB, transformer_cfg
  dirs per engine.
- **Disk: df 78% used, ~140 GB free** after this round's cleanup: −113 GB
  (sd35/wan Phase F), −13 GB (dead qwenimage Q4_K_M), −82 GB (8 hub-layout
  duplicates each verified to have a live direct twin).

## 8. Errata vs earlier docs

- `SESSION_2026-09-07_IMGLAB_SANA_BOOGU.md` says "flux2klein (Q6_K) load evicted
  boogu" — **the Klein 4B engine is BF16, not GGUF** (`image_lab_engines.py`
  `_load_flux2klein`, journal: "Loading FLUX.2 Klein 4B transformer (BF16)…").
  That row's quant label was wrong; the cross-eviction event itself stands.
- The SANA/Boogu doc's idle-unload row shows 943 s / `19:03:39` — the swap
  round's ernie run (912 s / `03:59:39`) is a second, independent PASS.

## 9. Ops cheatsheet

```bash
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87
sudo systemctl restart arthur-imglab        # restart lab (code redeploy = Phase 5 first)
sudo journalctl -u arthur-imglab -n 400     # logs — NEEDS sudo (arthur ∉ adm/journal group)
sudo systemctl restart arthur-comfy         # HiDream sidecar
curl -s localhost:8002/status | head -c 400 # schema/probe; empty JSON right after restart = still booting
curl -s -X POST localhost:8188/free -H 'Content-Type: application/json' -d '{"unload_models": true}'  # free sidecar VRAM
nvidia-smi                                  # card; process rows = per-container contexts
du -shL /opt/arthur-img-models/huggingface/*# du -sh LIES about cache symlinks — always -shL
df -h /                                     # single 630 GB root disk (imglab models share it with TTS stack)
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'   # TTS co-tenant
cd /opt/arthur-tts-lab && docker compose up -d                   # restore TTS stack (engine-current/qwen/orchestrator)
```

Deploy from the repo: `.\scripts\deploy\deploy_image_lab.ps1` (Phase 5 =
code-only ~30 s; full = Phases 1–8). Re-deploy note: Phase 4 pre-warm now
fetches qwenimage **Q4_K_S** (fixed this round — see §6).

## Change set

New file (this consolidation). Also `scripts/deploy/deploy_image_lab.ps1`
Phase 4 qwenimage GGUF entry: Q4_K_M → **Q4_K_S** + rationale comment.
