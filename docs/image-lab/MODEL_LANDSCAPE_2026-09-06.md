# Image Lab Model Landscape — 2026-09-06 Survey

> **What the user asked:** check for updates to the models already running in the
> Image Lab, and survey recently open-weighted image models that could actually run
> on the RTX 5060 Ti 16 GB — no massive downloads.
>
> **Bottom line:** nothing in the existing engine set has shipped new weights; two
> models fit and were added (SANA 1.6B Sprint + 1.5, Boogu-Image-0.1-Turbo fp8); one
> was screened out on the user's offload rule (Z-Image Turbo); everything else is
> closed-weights, too big for 16 GB, or an API-only update of an open line.

Survey date: 2026-09-06. All HF repo/org activity dates below were verified that day.

---

## 1. Existing engines — updates? No.

| Engine (in lab) | Open-weight lineage | Last weight activity | Verdict |
|---|---|---|---|
| FLUX.2 [dev] 32B | black-forest-labs open org | org frozen since **2026-04-06**; 9B-KV config last touched **Mar 12**; QuantStack GGUF quant ladder **Mar 13** | Frozen. Nothing newer on the open org |
| FLUX.2 Klein 9B-KV | black-forest-labs open org | same org freeze; the lab already runs the full GGUF quant ladder | Frozen |
| Ideogram 4 | Ideogram AI | repo frozen since **2026-06-04** | Frozen |
| SD 3.5 Large | stabilityai | weights unchanged since **Oct 2024** | Frozen — SD 3.5 Flash/Ultra have never shipped self-hostable weights |
| Wan 2.2 T2V/I2V | Wan-AI | open line tops at **2.2** | 2.5 / 2.6 / 3.0 are API-only |

**FLUX 3 [dev]:** announced for "later in 2026" — not shipped yet. Watchlist only
(§4). When it lands it is expected to be ~30B-class like FLUX.2 [dev], i.e. still a
16 GB VRAM question.

## 2. Candidates screened from the recent-weights table

| Candidate | Verdict | Why |
|---|---|---|
| **SANA 1.6B Sprint + SANA 1.5** (Efficient-Large-Model) | ✅ **ADDED** | Two bf16 diffusers repos, 1.6B DiT-class, whole pipeline ~10.4 GiB resident. Sprint = step-distilled 1–4 steps no CFG; 1.5 = ~20 steps CFG 4.5. Both ungated Apache-2.0 (+Gemma encoder terms). Repos share the Gemma-2-2B-IT encoder shards → HF cache dedups the pair to ~14 GB net disk. |
| **Boogu-Image-0.1-Turbo fp8** (Boogu) | ✅ **ADDED** | New 2026 Chinese T2I. Vendor's own 16 GB guidance is fp8 + `enable_model_cpu_offload` — the **one engine with a user-approved CPU-offload exception** to the lab's GPU-only policy. ~21 GB repo, Apache-2.0, ungated, research-only per README. |
| **Z-Image Turbo** (Z-Image/Z-Image-Turbo) | ❌ **DROPPED — user rule** | User rule: *"can the cpu offload part be offloaded to deepseek? if not ignore this model."* It cannot: the Qwen3-4B-class encoder emits a model-specific embedding space the DiT was trained against; DeepSeek has a different tokenizer/vocab and no API serves that encoder in embedding mode. (DeepSeek already covers the separate caption-expansion half of the Ideogram 4 flow — it cannot replace a diffusion *encoder*.) Full rationale + revisit condition in §3. |
| Qwen-Image 2.0 / 2.0-Pro / 3.0 (Qwen) | ❌ | Closed weights — API only, nothing to self-host |
| HunyuanImage-3.0 (Tencent) | ❌ | ~45 GB VRAM even at NF4. Not a 16 GB card, ever |
| SANA-WM | ❌ | **Video world model**, mislabeled as an image model in the source table — and the video slot is already covered by Wan 2.2 |
| "FLUX.2 turbo" | ❌ | fal-hosted **LoRA on the 32B dev base**, not a small checkpoint — 16 GB cannot run it at usable resolution |
| Nucleus-Image fp8 | ❌ | fp8 weights still need offload at 1024² → violates the GPU-only policy (and Boogu took the one sanctioned offload slot) |

Disk check for the two adds: **~36 GB net needed** (SANA pair ~14 GB after cache
dedup + Boogu ~21 GB) vs **63 GB free** at survey time → fits without removing
anything. sd35 + wan stay live and untouched.

## 3. Z-Image Turbo — drop rationale (recorded for the record)

> **SUPERSEDED 2026-09-07** — the drop verdict is obsolete. The next-day research
> report (`OPENWEIGHTS_T2I_16GB_FIT_2026-09-07.md`) re-ran the fit screen against
> the **GGUF transformer route** (`jayn7/Z-Image-Turbo-GGUF`, Q4_K_M ~5 GB via
> diffusers `ZImageTransformer2DModel.from_single_file`) with the in-repo Qwen3-4B
> encoder **encode-then-park** (the same embed-cache trick klein uses) — the
> offload rule is no longer triggered because nothing streams. User approved
> including it; the engine is implemented in the 2026-09-07 T2I swap. The
> historical rationale below is kept for the record.

- **What it is:** Z-Image Turbo (Apache-2.0) — a 7B-ish DiT distilled for ≤8-step
  generation. Attractive on paper: distilled, open, and the only entry in its row of
  the source table that wasn't obviously closed or huge.
- **The obstacle:** like SANA and Boogu, its instruction path runs through a **custom
  VLM encoder** (Qwen3-4B-class) that outputs a fixed, model-specific embedding space
  the DiT was trained against. Substituting any other LLM — DeepSeek included —
  changes the tokenizer/vocabulary and the embedding geometry; the DiT would receive
  garbage. The encoder is also ~7 GB fp32-on-disk, which is what forces CPU offload
  at 1024² on 16 GB in the first place.
- **The user's rule** (applied): *"can the cpu offload part be offloaded to
  deepseek? if not ignore this model."* It cannot → dropped.
- **Revisit condition:** only if a diffusers-layout **fp8/int8 Z-Image checkpoint**
  appears (encoder + DiT quantized, ~9-10 GB total like Boogu's layout). None exists
  today — the official repo ships FP32-on-disk (~33 GB). *(Met via GGUF — see the
  supersede note above.)*

## 4. Watchlist (not implemented, for the next landscape round)

| Item | Watch for | Trigger to act |
|---|---|---|
| **FLUX 3 [dev] weights** | "later in 2026" announcement on black-forest-labs HF org / X | Weights exist AND a 16 GB-runnable variant (GGUF/nf4/fp8 community ladder like Klein 9B's) appears |
| ~~**Z-Image fp8/int8 diffusers layout**~~ | — | ✅ Met via GGUF — **implemented 2026-09-07** (§3 superseded) |
| ~~**SD 3.5 Flash / Ultra open weights**~~ | — | Moot — sd35 itself was removed 2026-09-07 (typography superseded by Z-Image/Qwen-Image/HiDream/ERNIE) |
| **Wan 2.6 open release** | Wan-AI HF org | Currently API-only; if it ever opens, re-screen for 16 GB + disk |
| **Ideogram 4 text-editing follow-up** | Ideogram HF org | Upstream text-editing is the one Ideogram-4 capability the lab doesn't serve (T2I + partial edit only); watch for a distilled/edit-tuned checkpoint |
| **SANA 4.8B / 4K variants** | Efficient-Large-Model HF org | Config is already variant-extensible (`quant` field) — adding a 4.8B or 4K checkpoint later is a config + tooltip change, no new machinery |

## 5. What this round shipped

See `docs/sessions/SESSION_2026-09-06_IMGLAB_SANA_BOOGU.md` for the integration +
measured VRAM/RAM verdicts. Engine property tables and the offload-exception banner
live in `docs/image-lab/ARTHUR_IMAGE_LAB_REFERENCE.md`.
