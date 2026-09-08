# Session 2026-09-08 — Native base-face consumption: capability layer + Qwen-Image-Edit engine

## Context / requirement

The YouTube-shorts pipeline needed a fixed channel face rendered across emotions
and headline text. Hard constraints from the user:

- **Native AI conditioning only — NO face-swap / post-processing** ("that post
  processing sucks locally"). The base face must be consumed by the model
  itself.
- **Per-request upload** — the API caller uploads the reference image with
  every request; no stored / pre-attached base face anywhere.
- Decision: the edit capability is a **new 10th engine tab** (not a variant
  select); the VM spike runs over SSH.

Outcome of the spike (below): **GO** — the engine shipped as `qwenimage-edit`
and was verified live. Companion doc: the deployment-state consolidation
[`IMGLAB_DEPLOYMENT_STATE_2026-09-08.md`](../image-lab/IMGLAB_DEPLOYMENT_STATE_2026-09-08.md)
carries the roster + benchmark tables.

## Deliverable A — capability layer (commit 06937fa)

Uniform, honest reference-image surface across all engines, no VM dependency:

- `EngineInfo.image_input: "none" | "reference"` — declared by the checkpoint's
  real capability. klein×2 declare `"reference"`; the six text-only engines
  declare `"none"`; `qwenimage-edit` (Part C) declares `"reference"`.
- Every engine's param schema carries the `reference_image` file param (the UI
  dropzone appears on every tab) — but text-only engines **reject** an upload
  up front with a 400 naming the native alternatives instead of silently
  ignoring the caller's face:

  ```
  "SANA 1.6B is text-only — no checkpoint exists that consumes reference
  images. Native reference support: FLUX.2 Klein 4B, FLUX.2 Klein 9B-KV,
  Qwen-Image Edit."
  ```

- Upload >10 MB → 400; corrupt/undecodable uploads → 400 (never a silent
  None).
- `/status` + `/engines` expose `image_input`; UI shows a capability badge
  per tab.

## Deliverable B — VM spike (kill criteria met: GO)

Probe harness kept in-repo: `scripts/probe/qwenimage_edit_probe.py`. Mirrors
the production park rhythm exactly. Checkpoint: **Qwen/Qwen-Image-Edit-2511**
(official diffusers `QwenImageEditPlusPipeline`) + **unsloth GGUF Q4_K_S**
(`qwen-image-edit-2511-Q4_K_S.gguf`, 12,410,747,488 B ≈ 11.56 GiB).

### Edit conditioning (why this engine is special)

Two-channel: (1) the prompt is image-conditioned — vision tokens from the
Qwen2VL processor merge into the text (condition image resized to **384²
area** first); (2) the VAE-encoded ref latents (**1024² area**) concatenate to
the noise latents, doubling the DiT sequence (~4,096 canvas + ~4,072 ref ≈
8.2K tokens at 1024² → **12.06 s/step** vs the base 2512's ~5 s/step). The
EditPlus system template + 64-token prefix drop are hardcoded in the pipeline
`__init__` — the encode path runs the **pipeline-native
`_get_qwen_prompt_embeds` on a partial pipeline** (encoder + processor, no
transformer/VAE — `register_modules` accepts None), never a static
replication.

### Measured numbers (all on the 5060 Ti 16 GB, 2026-09-08)

| Stage | Value |
|---|---|
| Transformer load (warm) | 20.5 s / 11.81 GiB torch-alloc resident (cold 51.2 s) |
| Encoder load (warm) | 5.5 s — bnb-4bit OzzyGT 2512 mirror (~5.83 GiB resident), **shared with the base engine** (same Qwen2.5-VL architecture — probe-verified) |
| Encode phase peak | 6.31 GiB torch-alloc / ~6,840 MiB driver (encoder never coexists with the transformer) |
| **Gen peak 1024²/20 st/CFG 4.0** | torch 12.70 GiB · driver-total **15.40 GiB** · pid 14.64 GiB (margin **0.53 GiB** vs 15.93 GiB usable) |
| Gen peak 720×1440/20 st/CFG 4.0 | identical 12.70 GiB torch / 15.40 GiB driver-total — canvas ceiling ~1.05M px |
| Wall | 241.2 s (1024²) / 238.1 s (720×1440) — 12.06 s/step ≈ **~4 min/image at 20 st** |
| Determinism | same-seed rerun **pixel-identical** (sha `9663ed91baf22a5e`); 40-step run also sha-identical vs its own rerun |
| First OOM | fleet artifact only: the idle arthur-imglab service twin (506 MiB) pushed the process 70-190 MiB over; `systemctl stop arthur-imglab` → rerun passed. Production gens RUN inside that process, so the honest fleet is ~804 MiB (TTS contexts + comfy sidecar) |

`expandable_segments` is NOT a fix (reverted 2026-08-14 — pins ~3 GB of freed
encoder memory, see image_lab.py:38-52); native allocator numbers above.

### Verdict ladder

GO — gen peak ≤ usable − ~400 MiB at 1024² default steps, both square and
9:16. No reduced-steps fallback needed; no nunchaku route needed.

## Deliverable C — `qwenimage-edit` engine (shipped on GO)

10th engine — "Qwen-Image Edit". Native reference consumption, whole-card
engine (gate 14,000 MiB like the base; encode gate 8,000):

- `_QWENIMAGE_EDIT_GGUF`: **Q4_K_S only** (two-channel conditioning peaks
  15.40 GiB driver — Q4_K_M-class is OOM-class; keep the dict shape).
- Loader mirrors the base (`AutoencoderKLQwenImage` concrete-class VAE gotcha,
  `_execution_device` patch, slicing/tiling) + the **Qwen2VLProcessor**
  (global-cached for the encode-without-transformer path, like the tokenizer).
- Generator: reference **required** (400 without it — no text-only path in an
  edit checkpoint); canvas area ≤ 1024² px (400 beyond — 1024² and 720×1440
  are the measured ceiling); ref thumbnailed ≤1024 px before the pipe (it
  re-resizes internally anyway).
- **Image-conditioned embed cache**: the embeds depend on the reference as
  much as the prompt, so the cache key is sha256(prompt text + uploaded image
  bytes) — same (prompt, image) pair → same cond image → same embeds. Cache
  miss → unload transformer → bnb4 encoder encode → save → reload (paid once
  per unique pair).
- Registered in `_LOADERS`/`_GENERATORS`/`probe_availability`; deploy pre-warm
  job list gained the GGUF + transformer-config rows.

## E2E verification (live, 2026-09-08 — all green)

On the deployed service (`arthur-imglab`, the 10-engine code), channel base
face uploaded fresh per request:

- `/status`: **10 engines** all probed available, `qwenimage-edit`
  `image_input=reference`; `sana` + ref upload → **400** naming "FLUX.2 Klein
  4B, FLUX.2 Klein 9B-KV, Qwen-Image Edit".
- **run1** (first live request — "Make this person smile warmly… bold white
  headline PIZZA IS LIFE…", 1024²/20 st/CFG 4.0, seed 20260908): **200 OK**,
  gallery `total_s` **309.56** (`load_s` 20.81). Journald lifecycle as
  designed: `20:04:01 Qwen-Image-Edit embed cache miss (2 prompt(s),
  image-keyed) — unloading the transformer to encode` → encoder load →
  `20:04:28 encode done — reloading the transformer` → gen → saved
  `qwenimage-edit_61cb2e27-a3cb-4a6d-8a26-900a9cc33322.png`.
- **run2** (same-seed rerun, same prompt + image bytes): **200 OK**, gallery
  `total_s` **242.47** — and the journald window between run1's 200 and run2's
  gen start is **clean: no cache-miss / encode / reload markers** (embed-cache
  hit; `load_s` null — the transformer never unloaded). Gen wall 243 s ≈
  20 × 12.06 s/step.
- **Determinism: run1 vs run2 byte-identical** — sha1
  `4630f8e471245365916bd3e3a49d68ad22437ecc`, both 1,103,919 B (live-service
  confirmation of the probe's finding; qwenimage-family GGUF route is
  pixel-identical on same seed).
- **No-ref → 400**: "reference_image is required for Qwen-Image Edit — the
  edit checkpoint consumes a base image natively (no text-only path). Upload
  the image with this request."
- **Oversize canvas → 400**: "Canvas 1536×1536 = 2.4M px exceeds the ~1.05M
  px ceiling verified on the 16 GB card (1024² and 720×1440 both fit —
  measured 2026-09-08)."
- Quality eyeball: spike PNGs (20 st ×2 sizes + 40-step quality + rerun) were
  pulled to the caller at `c:\tmp\qwenimage_edit_spike\` — identity vs
  `channel-base-face.jpg` + "PIZZA IS LIFE" headline legibility are the human
  verdicts that remain (files unviewable in the agent environment).

## Notes for the next session

- Remote dev's native klein9b emotion+headline test pair is independent of
  this work (answers klein prompt-guidance quality).
- The user authorised stopping TTS containers during the spike — not needed
  beyond the service-twin stop already described.
