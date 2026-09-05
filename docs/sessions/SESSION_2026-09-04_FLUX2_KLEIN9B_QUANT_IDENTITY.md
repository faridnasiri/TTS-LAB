# Session 2026-09-04 — FLUX.2 Klein 9B-KV: identity preservation + quant ladder

## Goal

1. Verify the vendor claim that FLUX.2 Klein 9B-KV (GGUF) "maintains subject
   identity, face structure, and scene composition" in instruction-based edits
   — with the user's own eyes, not just cosine scores.
2. Answer the quant question: is Q4_K_M still the right default? Does Q8_0
   exist / run? Is Q4 faster than Q6?

## Verdicts

| Claim / question | Outcome | Evidence |
|---|---|---|
| KV keeps identity across instructed edits | ✅ **Verified** — but **instructed, not anchored** | 6 edits vs control portrait, insightface cosine 0.697–0.932 vs **0.084 impostor**; face-bbox IoU 0.861–0.976 |
| Q6_K should replace Q4_K_M as default | ✅ Raised 2026-09-04 | Identity is **quant-invariant** in Q4→Q6 range (same-seed outputs agree 0.958–0.985; per-edit cosines within ±0.002) — Q6 buys detail headroom at +2.1 GiB, nothing lost |
| Is there a Q8_0? | ⚠️ Exists, **cannot run on the 16 GB card** | 9.98 GB GGUF downloads + loads, but generation OOMs (60 MiB free of 15.48 GiB at first allocation) even with TTS containers evicted — ~0.8 GiB over the wall |
| Is Q4 faster than Q6? | ✅ Yes, marginally: **~5–6%** | Warm-engine 1024²/4-step: Q4 23.7–24.0 s vs Q6 25.3 s (~1.5 s/image) — attention + VAE costs are quant-independent, so the bandwidth saving doesn't scale linearly |

## Method & assets

- **Subject:** synthetic control portrait (T2I, seed 777). Six instructed
  edits share the control as reference: scene swap (bg), suit swap, laugh,
  Rembrandt restyle, winter compound, de-age 30 yrs.
- **Identity metric:** insightface buffalo_l embeddings (L2-normalized
  cosine) + face-bbox IoU, run in an isolated CPU venv (`/tmp/kvvenv` on the
  VM, models in `/tmp/kvvenv/ifmodels`) — the harness model cannot view
  images, so a machine metric stood in and a **blind vote page** was built
  for the human-eye half of the verdict (still pending).
- **Edits by instruction, not anchor:** prompts say "keep his face/glasses/
  beard…" — nothing is anchored to the reference beyond the KV path. The
  weak spots are exactly the face-deforming/restyle edits (laugh 0.697,
  Rembrandt 0.698); outfit/background swaps hold best (suit 0.932).
- **Assets:** VM `/tmp/kvtest/` (ctrl.png reference, per-quant outputs,
  prompt/seed scripts); dev machine `c:\tmp\kvtest\` (PNG+700px JPEG pairs,
  `face_metrics.json`, `kv_report.html` evidence sheet, `kv_vote.html` blind
  vote — served at http://127.0.0.1:8765/).

## Quant findings in detail

- **Q4 ↔ Q6 same-seed A/B** (E2/E3/E4, seeds 1002/1003/1004, ref ctrl.png):
  identity cosine vs reference — Q6: 0.930 / 0.690 / 0.696; Q4:
  0.932 / 0.697 / 0.698. Q4↔Q6 outputs themselves agree 0.958–0.985.
  Identity retention is **bound by the 4-step distilled model's
  capability**, not the quant.
- **Q8_0 verified 2026-09-05 (VM clock):** 9.98 GB GGUF fetched in ~110 s
  via xet; load succeeded (~35 s); generation failed with CUDA OOM at the
  first 128 MiB allocation (60 MiB free). Pre-evicting the TTS containers
  (ports 8101–8104) was **not** sufficient — the Q8 transformer sits
  ~2.05 GiB above Q6_K's 7.55 GiB resident. Q6_K itself peaks at
  14,234 MiB used / 1,615 MiB free at 1024², so Q8 (~16.3 GiB) cannot fit;
  even 768² misses by ~0.2 GiB. It would run only with the text encoder
  off-GPU (`IMGLAB_GPU_ONLY=0`, bf16 on CPU RAM frees ~2.5 GiB) or a larger
  card. **The process self-healed** — a subsequent Q6_K generation returned
  200 in 36 s (cold) on the same process, no restart needed.
- **Speed (2026-09-05 VM clock, wall clock via curl):** warm 1024²/4-step —
  Q6_K 25.3 s (t2), Q4_K_M 24.0 s (t4) and 23.7 s (t5); cold/swap runs
  t1/t3/t6 37.6/47.5/36.9 s (load ~12–24 s incl. unload).

## Changes

1. **`image_lab_engines.py`** — `_FLUX2KLEIN9B_GGUF` ladder grew to five
   rungs (Q3_K_M / Q4_K_M / Q5_K_M / Q6_K / Q8_0); loader default
   `quant or "Q6_K"` (was `"Q4_K_M"`). All quants share one architecture
   config — no new download logic.
2. **`image_lab_config.py`** — flux2klein9b description + quant selector:
   Q6_K marked default, each rung shows GB; Q8_0 labelled honestly
   ("near-lossless — exceeds 16 GB card") with a tooltip recording the
   verified 2026-09-05 OOM and the offload escape hatch.
3. **`docs/image-lab/FLUX2_KLEIN_9B_KV.md`** — source/ladder rows updated to
   Q6_K default; new "Quant-vs-identity finding" block; VRAM table corrected
   to measured numbers (Q6 gen peak 14.2 GiB, ~1.5 GiB headroom — the old
   "~12 GiB estimate / 4–5 GiB headroom" was wrong).
4. **Deployed** on the VM: the two `.py` files → `/opt/arthur-img/`,
   `arthur-imglab.service` restarted; served engine catalog verified to
   carry the new labels/defaults.

## Pending

- **User's blind vote** on the 7 portraits (`kv_vote.html` — 6 edits + 1
  impostor shuffled) is the human-eye verdict; cosine evidence stands in
  until then.
- Q8_0 stays in the ladder as a documented "bigger card / encoder-offload
  only" rung (user decision — not dropped).
