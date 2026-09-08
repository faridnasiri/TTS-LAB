# Open-Weights T2I Leaderboard — 16 GB Fit Report (Typography Focus)

> **Generated:** 2026-09-07
> **GPU:** NVIDIA RTX 5060 Ti 16 GB GDDR7 (Blackwell sm_120)
> **Source:** [Artificial Analysis Text-to-Image Leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image?open-weights=true) — open-weights filter
> **Goal:** best open-weights models for **complex typography** — text rendering is the #1 criterion
> **Scope note:** `Qwen Image Max 2512` on the leaderboard is the **hosted-API name of the open Qwen-Image-2512 model** (Apache-2.0, downloadable) — not a separate architecture. The older Aug-2025 "Qwen-Image-Max" was a different, API-only model.

---

## TL;DR

Your lab already runs **Ideogram 4** (nf4/fp8) — the #1 open-weights typography model on this leaderboard (Elo 1017). Everything below is about what is **worth adding around it** on a 16 GB card.

| Priority | Model | Elo | License | Fits 16 GB? | Why |
|---|---|---|---|---|---|
| **Already have** | Ideogram 4 (nf4) | 1017 | gated, non-commercial | ✅ (~10 GB) | Consensus typography leader — keep as primary text engine |
| **Add #1** | **Z-Image Turbo** | 928 | Apache-2.0 | ✅ easy (~8 GB fp8) | Fastest install, plain-prompt, strong EN/ZH text, commercial-safe |
| **Add #2** | **Qwen-Image-2512** | 972 | Apache-2.0 | ✅ GGUF Q4_K_M ~13 GB | Best *independently verified* bilingual long-text of any fully-open model |
| **Add #3** | **HiDream-O1-Image (-Dev)** | 978 / 874 | MIT | ✅ fp8 ~10 GB | Highest claimed open LongText-Bench scores; pixel-native = sharp small glyphs |
| **Add #4** | **ERNIE-Image-Turbo** | 923 | Apache-2.0 | ✅ GGUF q8 ~14-15 GB | Poster/layout text, CN+EN, 8 steps |
| Won't fit | FLUX.2 dev, Cosmos3-Super-T2I, HunyuanImage 3.0, GLM-Image, Infinity 8B, HiDream-I1, FIBO | — | — | ❌ 24 GB+ class | See §5 — the best remaining text models are all out of reach on 16 GB |

**Bottom line:** on 16 GB, the text-rendering ceiling is Ideogram 4 (installed) + Z-Image-Turbo / Qwen-Image-2512 / HiDream-O1 as complements. The top-tier models that would *beat* Ideogram on some text dimensions (FLUX.2-dev 32B, GLM-Image 16B, HunyuanImage 3.0 80B, Cosmos3-Super 64B) all need ≥24 GB.

---

## 1. Method & evidence notes

- Leaderboard Elo = **overall blind-vote preference, not a text subscore**. No OCR/text subscore exists in the arena — treat Elo as a coarse prior, not a typography metric.
- Text evidence below is tagged by reliability: **(V)** vendor claim · **(I)** independent benchmark · **(C)** community consensus.
- De-facto text benchmarks: **LongText-Bench** (OCR word accuracy, EN/ZH tracks), **CVTG-2K**, **WeGenBench / OneIG-Bench**, **BizGenEval** (commercial layouts, first-class text dimension).
- All VRAM figures are for ~1024 px generation on a single consumer GPU; vendor "fits in X GB" claims usually target 24 GB cards.

---

## 2. Current Image Lab baseline (what the report builds on)

| Engine key | Model | Text strength | Fit method today |
|---|---|---|---|
| `ideogram4` | Ideogram 4 (9.3B, Qwen3-VL-8B encoder, JSON captions) | **Best open-weights typography** | official nf4 (~6 GB) / fp8 repo, magic-prompt |
| `flux2klein9b` | FLUX.2 Klein 9B-KV | runner-up tier | QuantStack GGUF Q6_K (~13.5 GB) |
| `flux2klein` | FLUX.2 Klein 4B | short strings OK | HF direct (~10 GB) |
| `sd35` | SD 3.5 Large | **dated now** — latent VAE blurs glyphs | city96 GGUF |
| `sana`, `wan`, `boogu` | Sana / Wan2.2 video / new engine | not text-first | — |

---

## 3. Full leaderboard fit table (all open-weights rows fetched 2026-09-07)

Ranked by leaderboard rank. ✅ = installable on 16 GB and relevant · 🟡 = marginal (offload/very tight) · ❌ = not feasible · Verdicts assume your existing quant toolkit (GGUF via ComfyUI-GGUF / city96 & QuantStack, nf4 bitsandbytes, fp8).

| # | Model | Elo | Size / arch | 16 GB | Typography | Verdict |
|---|---|---|---|---|---|---|
| 28 | Ideogram 4.0 | 1017 | 9.3B DiT | ✅ | **best-in-class EN** (V/I/C) | ✅ installed (`ideogram4`) |
| 29 | Ideogram 4.0 (Quality) | 1014 | same | ✅ | same | API preset of same weights |
| 39 | FLUX.2 [dev] | 1000 | 32B + 24B encoder | ❌ | good, < Ideogram long-text (I) | needs ≥24 GB (Q4 ~19 GB transformer + ~20 GB encoder) |
| 42 | FLUX.2 [dev] Turbo | 999 | 32B + 8-step LoRA | ❌ | same class | no size relief — still 32B |
| 43 | Cosmos3-Super-T2I (agentic) | 995 | 64B MoT | ❌ | no published text data | BF16 ≈128 GB; int4 ≈32 GB+ |
| 44 | Ideogram 4.0 Fast (Quality) | 992 | 9.28B (fal, FP4-targeted) | ✅ | small drop vs base | same stack as installed |
| 48 | Ideogram 4.0 Instant | 987 | 9.28B, 8-step no-CFG | ✅ | small drop vs base | optional fast tier, same stack |
| 53 | Cosmos3-Super-T2I | 982 | 64B | ❌ | — | data-center class |
| 54 | FLUX.2 [dev] Flash | 980 | 32B family | ❌ | — | see FLUX.2 dev |
| 55 | **HiDream-O1-Image** | 978 | 8B pixel-unified (no VAE) | ✅ fp8 | best open LongText claims (V) | ✅ **recommended** (see §4) |
| 57 | Ideogram 4.0 Fast | 974 | 9.28B | ✅ | — | see #44 |
| 60 | **Qwen Image Max 2512** (= Qwen-Image-2512) | 972 | 20B MMDiT | ✅ GGUF Q4 | strongest verified bilingual (I) | ✅ **recommended** (see §4) |
| 62 | Cosmos3-Super-T2I-4Step | 965 | 64B | ❌ | — | — |
| 65 | HunyuanImage 3.0 Instruct | 953 | 80B MoE / 13B active | ❌ | industry CN/EN (V), weak long EN | NF4 still 41-49 GB |
| 68 | FLUX.2 [klein] 9B | 945 | 9B | ✅ | runner-up tier (C) | ✅ installed (`flux2klein9b`) |
| 73 | HunyuanImage 3.0 | 935 | 80B MoE | ❌ | — | — |
| 74 | **Z-Image Turbo** | 928 | ~6B S3-DiT + Qwen3-4B enc | ✅ easy | LongText 0.922 (I); multi-font < Ideogram (C) | ✅ **recommended** (see §4) |
| 78 | **ERNIE Image Turbo** | 923 | ~8B DiT, 8-step | ✅ GGUF | CN press "~Nano Banana" (C), strong CN/EN | ✅ **recommended** (see §4) |
| 81 | ERNIE Image | 922 | ~8B, 50-step | 🟡 | same, slower | Turbo preferred |
| 94 | FLUX.2 [klein] Base 9B | 899 | 9B | ✅ | — | same family as installed |
| 100 | SRPO | 880 | — | ? | — | no install path found (Tencent training entry) |
| 101 | HunyuanImage 2.1 | 879 | 17B DiT | 🟡 fp8 ~18-20 GB | ~3% text error, 2K native (V) | offload-only on 16 GB — skip |
| 102 | Qwen Image (orig.) | 878 | 20B | ✅ GGUF Q4 | LongText 0.945 (I) | superseded by 2512 |
| 105 | FIBO | 876 | 8B, JSON captions | ❌ | **spelling unreliable** (I) | gated CC-BY-NC + 24 GB+ |
| 106 | HiDream-O1-Image-Dev | 874 | 8B, 28-step CFG-0 | ✅ fp8 | same as #55 | ✅ **recommended** (see §4) |
| 109 | HiDream-I1-Dev | 872 | ~17B + 4 encoders | ❌ | strong (C) | 24 GB+ class (GGUF/NF4 only) |
| 111 | GLM-Image | 869 | 16B hybrid AR+DiT | ❌ | best-in-class claims (V/I) | BF16 ~34-38 GB; no GGUF — **24 GB+** |
| 112 | Z-Image Base | 867 | ~6B, 28-50 steps | ✅ | LongText 0.936 (I) | Turbo covers this |
| 113 | HiDream-I1-Fast | 864 | ~17B | ❌ | — | — |
| 115 | FLUX.2 [klein] 4B | 861 | 4B | ✅ | short-text OK (C) | ✅ installed (`flux2klein`) |
| 117 | Infinity 8B | 857 | 8B bitwise-AR | ❌ | no text focus | fp32 repo → fp16 ~16.7 GB + enc; no quants; 24 GB+ |
| 118 | LongCat Image | 853 | ~6B + Qwen2.5-VL-7B | 🟡 | excellent bilingual (V/I) | ~17 GB w/ offload; fp8 would fit — long-shot |
| 120 | FLUX.1 [dev] | 843 | 12B | ✅ | weaker than FLUX.2 klein | skip — superseded |
| 122 | FLUX.1 Krea [dev] | 837 | 12B | ✅ | — | skip |
| 123 | FIBO Lite | 835 | 8B few-step | ❌ | spelling unreliable | gated CC-BY-NC |
| 124 | SD 3.5 Large Turbo | 835 | 8B | ✅ | dated glyphs | ✅ installed (`sd35` turbo preset) |
| 125 | SD 3.5 Large | 835 | 8B | ✅ | dated glyphs | ✅ installed |
| 134 | FLUX.1 [schnell] | 799 | 12B 4-step | ✅ | poor | skip |
| 137 | Lumina Image v2 | 778 | 2.6B | ✅ | weak (C) | skip for typography |
| 138 | FLUX.2 [klein] Base 4B | 773 | 4B | ✅ | — | Apache-2.0 sibling of installed |
| 139 | Playground v2.5 | 773 | 5B | ✅ | old era | skip |
| 142 | SD 3.5 Medium | 764 | 2.5B | ✅ | dated | skip |
| 143 | Sana Sprint 1.6B | 741 | 1.6B | ✅ | **acknowledged weak spot** | license ambiguity (NSCL vs Apache); skip |
| 145 | SD 3 Medium | 723 | 2B | ✅ | dated | skip |
| 149 | OmniGen V2 | 712 | — | ? | — | not text-first |
| 150 | SDXL Lightning | 712 | 3.5B | ✅ | dated | skip |
| 151 | Bagel (BAGEL-7B-MoT) | 710 | 14B tot/7B act AR | ✅ fp8 | no text evidence | GenEval 0.88; not typography-focused |
| 152 | Bria 3.2 | 707 | — | ❌ | — | gated |
| 153 | SDXL 1.0 | 685 | 3.5B | ✅ | dated | skip |
| 154 | SD 2.1 | 556 | 0.9B | ✅ | very dated | skip |
| 156 | Janus Pro | 522 | — | ✅ | weak | skip |
| 157 | SD 1.5 | 465 | 0.9B | ✅ | very dated | skip |

---

## 4. Recommended additions — deep dive

> **All four implemented 2026-09-07** — see
> `docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md` for the change set,
> live-verified numbers, and gate recalibrations.

### #1 — Z-Image Turbo (Elo 928) — *easiest win, plain prompts, fast* — ✅ IMPLEMENTED
- **Arch:** ~6B single-stream S3-DiT + Qwen3-4B text encoder + FLUX-style 16-ch VAE. Distilled to **8 steps, CFG must be 0.0**.
- **License:** Apache-2.0, ungated — `Tongyi-MAI/Z-Image-Turbo` (HF/ModelScope). Commercial-safe.
- **16 GB:** BF16 ~14-16 GB (tight but fits); **fp8 ~8 GB; GGUF Q4 ~5-6 GB**. Easiest fit of any text-capable model on the list. ~2-3 s/img @1024px on a 4090 → roughly 5-8 s on the 5060 Ti.
- **Typography:** native EN+ZH in-image text (LongText-Bench **0.922** Turbo / 0.936 base — I). Strongest for short-to-medium phrases and bilingual signage; below Ideogram for dense multi-font design layouts (C). Plain-prompt friendly — no JSON caption requirement (unlike Ideogram 4).
- **Note:** Turbo is not LoRA-trainable; the base (Z-Image Base, 28-50 steps) is the fine-tune target. ComfyUI native (CLIPLoader `lumina2`).

### #2 — Qwen-Image-2512 (Elo 972) — *best verified bilingual long text, fully open* — ✅ IMPLEMENTED
- **Arch:** ~20B MMDiT (+ ~4B VL encoder, Wan-VAE-derived VAE in 2512; original used Qwen2.5-VL-7B). Apache-2.0. ComfyUI-native since Aug 2025; GGUF via `city96/Qwen-Image-gguf`; diffusers pipeline exists.
- **16 GB:** BF16 ~40 GB+ / **FP8 ~20 GB → OOMs on 16 GB**. Working path: **GGUF Q4_K_M ~13.3 GB peak** — proven on an RTX 4060 Ti 16 GB and on a **sm_120 RTX 5080 16 GB** (same Blackwell arch as your 5060 Ti). Expect **~100-200 s/image at 50 steps**; halve with CFG 1.0; a **4-step Lightning LoRA cuts this to tens of seconds**.
- **Typography:** strongest *independently verified* bilingual long-text of any fully-open model — LongText-Bench **0.956 EN / 0.965 ZH** (I, cited in ERNIE tech report). Original Qwen-Image scored 0.945. Bilingual infographics/slides/posters; English alone still trails Ideogram for design-grade typography.
- **Why not the original "Qwen Image" row (878)?** 2512 is the December-2025 refresh, materially better at dense text layouts — pick 2512.

### #3 — HiDream-O1-Image / -Dev (Elo 978 / 874) — *pixel-native sharpness, most permissive* — ✅ IMPLEMENTED (Dev, ComfyUI sidecar)
- **Arch:** **8B pixel-space "UiT" — no VAE, no separate text encoder**; unified decoder-only transformer over raw pixel tokens (extends Qwen3-VL-8B). No latent compression ⇒ no glyph blur on small type.
- **License:** **MIT**, ungated — `HiDream-ai/HiDream-O1-Image` / `-Dev`.
- **16 GB:** FP8-mixed **~10 GB fits comfortably**; BF16 ~17-20 GB needs offload. FP8 conversions on HF (e.g. `drbaph/HiDream-O1-Image-BF16`). Pixel-space ⇒ higher compute per image than latent DiTs — expect slower gens (no 16 GB timing data found).
- **Typography:** vendor-reported **LongText-Bench 0.979 EN / 0.978 ZH — highest open-weights numbers published (V)**; CVTG-2K 0.9128 (V); community notes residual long-text errors and less independent confirmation. **Dev** variant = 28 steps, CFG 0.0, rank 874 — best speed/quality balance.
- **Recommendation:** start with the **Dev** checkpoint on this GPU.

### #4 — ERNIE-Image-Turbo (Elo 923) — *poster/layout bilingual, 8-step* — ✅ IMPLEMENTED (Nunchaku-Lite NVFP4)
- **Arch:** ~8B single-stream DiT (ERNIE-ViLG 3.0 lineage) + Ministral-3-3B encoder + FLUX-2 VAE. Turbo = 8 steps, CFG 1.0.
- **License:** Apache-2.0, ungated (`baidu/ERNIE-Image-Turbo`). Community GGUF q8/q6/q4 UNet on Civitai; ComfyUI workflows published.
- **16 GB:** BF16 full stack ≈ 24 GB → no. **GGUF q8 UNet (~8.5 GB) + quantized encoder + VAE ≈ 14-15 GB — fits** (estimate from param count; not officially published).
- **Typography:** CN press headlined text as "comparable to Nano Banana" (C); strong CN/EN posters, comics, structured layouts; occasional artifacts and Asian-face default bias (C). Good **cheap 8-step alternative** with Apache license if you want a second fast text engine.

### Optional — Ideogram 4.0 Instant (Elo 987) — *same stack as installed*
- `fal/ideogram-v4-instant` is the **8-step, no-CFG** open release of the same 9.3B family you already run (gated, non-commercial). Your `ideogram4` engine already has V4_TURBO_12 — Instant would add an 8-step tier if you want max speed from the same captioning stack. Marginal gain.

---

## 5. Won't fit on 16 GB — and why it matters

The pattern is brutal: **the models that would actually extend your typography ceiling past Ideogram 4 mostly sit at 16-80B** and need ≥24 GB.

| Model | Elo | Size | What it would add | Why it won't fit |
|---|---|---|---|---|
| FLUX.2 [dev] | 1000 | 32B + Mistral-Small-3.2-24B encoder | best-of-class EN display type among big models | Q4_K_S ~19 GB transformer **+ ~20 GB quantized encoder**; 24 GB+ |
| Cosmos3-Super-T2I | 982 | 64B hybrid MoT | arena top open; huge prompt context | BF16 ≈128 GB; community int4 ≈32 GB+ |
| HunyuanImage 3.0 | 935 | 80B MoE (13B act.) | industry bilingual (V) | NF4 still 41-49 GB VRAM; GGUF unsupported (MoE) |
| GLM-Image | 869 | 16B (9B AR + 7B DiT) | best-in-class text claims, Glyph encoder | BF16 ~34-38 GB; no GGUF; int8 ≈20 GB disk, no 16 GB demo |
| HiDream-I1-Dev/Fast | 872/864 | ~17B + 4 encoders | strong typography, MIT | GGUF/NF4 class, 24 GB+ |
| Infinity 8B | 857 | 8B bitwise-AR | — | fp32 weights only → fp16 ≈16.7 GB + encoder, no quants |
| FIBO / Lite | 876/835 | 8B | licensed-data, JSON captions | gated **CC-BY-NC** + 24 GB+; and spelling unreliable (I) |

**Marginal (offload-only, not recommended):** HunyuanImage 2.1 (fp8 ~18-20 GB at 1024px — CPU-offload territory, very slow), LongCat-Image (fp16 ~17 GB w/ offload; fp8 would fit — Apache, strong bilingual text, worth a look if you want to push the 16 GB edge).

**Upgrade-path note:** a used/24 GB card (e.g. RTX 4090/5090, or 24 GB Blackwell-class) unlocks FLUX.2-dev Q4, GLM-Image int8, HiDream-I1, HunyuanImage 3.0 NF4 — the entire next typography tier. On the current 16 GB card, Ideogram 4 remains the correct ceiling and the four additions above are the sensible complement.

---

## 6. Caveats

1. **Elo ≠ typography.** Use the table for fit; judge text with your own eyes on poster-style prompts before wiring an engine in.
2. **Distillation degrades text** — general finding (I/C): few-step variants (Instant/Fast/Turbo) drop a little legibility; modern ones hold much better than the SDXL-Turbo era. When legibility matters: ≥4-8 steps and CFG≈1.
3. **Speed numbers for Qwen-Image GGUF** are from community 16 GB reports (4060 Ti / sm_120 5080) — expect the same order of magnitude on the 5060 Ti but verify.
4. **Licenses:** Z-Image / Qwen-Image / ERNIE-Image = Apache-2.0 (commercial-safe). HiDream-O1 = MIT. Ideogram 4 = **gated non-commercial** (research/personal free; paid license for commercial self-host) — fine for the lab, matters if outputs ship in SpamBlocker. Sana Sprint has a license ambiguity (NSCL v2 vs Apache metadata) — moot since its text is weak anyway. Cosmos3 uses OpenMDW 1.1.
5. **The 4 recommended models share the winning recipe** (per the typography research): long-context **VLM text encoders** (Qwen3-VL / Qwen3-4B / Qwen2.5-VL), **native high-res training**, and (HiDream) **pixel-level generation** — which is exactly why small 6-9B models out-type 32-80B ones.
6. This report is a point-in-time snapshot (2026-09-07); verify repos/quants/licenses at install time.

---

## 7. Suggested next steps

1. Install **Z-Image Turbo** first (smallest download, fastest, Apache-2.0, plain prompts) — fits the lab's ComfyUI-GGUF or diffusers route.
2. Install **Qwen-Image-2512 GGUF Q4_K_M** + Lightning LoRA if bilingual/dense-text is in your workloads; reuse the same city96/QuantStack GGUF pattern as `sd35`/`flux2klein9b`.
3. Trial **HiDream-O1-Image-Dev fp8** head-to-head vs Ideogram 4 on 10 poster prompts with long text — it is the only candidate that could displace Ideogram 4 in your lab, and it's MIT.
4. If a fast second text engine is wanted: **ERNIE-Image-Turbo** (8-step, Apache).

---

## Sources

- Artificial Analysis leaderboard (open-weights rows): https://artificialanalysis.ai/image/leaderboard/text-to-image?open-weights=true
- Ideogram 4: [HF ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) · [licensing](https://ideogram.ai/licensing) · [fal Instant/Fast](https://huggingface.co/fal/ideogram-v4-instant)
- FLUX.2: [HF FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) · [fal Turbo comparison](https://fal.ai/learn/biz/flux-2-turbo-vs-flux-2-which-model-should-you-choose) · [QuantStack Klein 9B-KV GGUF](https://huggingface.co/QuantStack/FLUX.2-Klein-9B-KV-GGUF)
- Cosmos3-Super-T2I: [HF card](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) · [OpenMDW 1.1](https://openmdw.ai/license/1-1/)
- HiDream-O1-Image: [HF](https://huggingface.co/HiDream-ai/HiDream-O1-Image) · [WaveSpeed write-up](https://wavespeed.ai/blog/posts/hidream-o1-image-dev-pixel-unified-transformer/)
- Qwen-Image-2512: [HF](https://huggingface.co/Qwen/Qwen-Image-2512) · 16 GB GGUF reports: [4060 Ti 16GB](https://smeltcore.com/recipes/qwen-image-on-rtx-4060-ti-16gb-20b-text-to-image-via-comfyui-gguf/) · [sm_120 16GB](https://smeltcore.com/recipes/qwen-image-on-rtx-5080-20b-text-to-image-via-comfyui-gguf-blackwell-sm-120-16-gb/) · [arXiv 2508.02324](https://arxiv.org/abs/2508.02324)
- Z-Image: [zimage.design guide](https://zimage.design/blog/getting-started-with-z-image/) · [Turbo under ComfyUI](https://www.thundercompute.com/blog/z-image-turbo-comfyui)
- ERNIE-Image: [tech reference](https://ernie-image.github.io/) · [HF ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)
- HunyuanImage 3.0/2.1: [HF 3.0](https://huggingface.co/tencent/HunyuanImage-3.0) · [HF 2.1](https://huggingface.co/tencent/HunyuanImage-2.1)
- GLM-Image: [HF](https://huggingface.co/zai-org/GLM-Image) · [diffusers VRAM notes](https://www.glmimage.blog/blog/glm-image-diffusers-pipeline)
- Infinity 8B: [HF](https://huggingface.co/FoundationVision/Infinity) · FIBO: [HF](https://huggingface.co/briaai/FIBO) · LongCat: [HF](https://huggingface.co/meituan-longcat/LongCat-Image)
- Text benchmarks: [LongText-Bench](https://huggingface.co/datasets/X-Omni/LongText-Bench) · [BizGenEval](https://huggingface.co/papers/2603.25732) · [TextPecker (CVPR 2026)](https://www.openaccess.thecvf.com/content/CVPR2026/papers/Zhu_TextPecker_Rewarding_Structural_Anomaly_Quantification_for_Enhancing_Visual_Text_Rendering_CVPR_2026_paper.pdf)
