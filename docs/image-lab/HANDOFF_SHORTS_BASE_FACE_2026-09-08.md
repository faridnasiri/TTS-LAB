# Handoff — Base-face API for the shorts pipeline (2026-09-08)

For the dev consuming the Image Lab to render the fixed channel face across
emotions + headline text. Covers the capability layer (commit `06937fa`) and
the new **Qwen-Image Edit** engine (commit `e0e79ae`).

## The rules of this API (locked with the user)

1. **Upload the base image with EVERY request** — `reference_image` multipart
   file. Nothing is stored or pre-attached server-side; there is no "registered
   face" endpoint. Pick the file fresh per call.
2. **Native AI conditioning only — no face-swap / no post-processing.** The
   checkpoint itself consumes the face. Do not build a swap/composite step into
   the pipeline.
3. If an engine can't natively consume a face it **rejects** the upload with a
   400 naming the native alternatives — never silently ignores it.

## Which engines accept the face (3 of 10)

| Engine key | Lane | Per-request ref upload | What it's for | Wall time (measured) |
|---|---|---|---|---|
| `flux2klein` | fast | native subject conditioning | identity + emotion, lighter card load | ~52 s gen |
| `flux2klein9b` | **fast lane** | native subject conditioning (FLUX.2 klein 9B does prompt-driven edits of the ref) | identity + emotion, 4-step distilled | **25–47 s gen** |
| `qwenimage-edit` | **text lane** | native **instruction-edit** conditioning (Qwen-Image-Edit-2511) | identity + emotion + **in-image headline text**, instruction fidelity | **~4 min gen** (12.06 s/step) |

The other seven (`sana`, `boogu`, `zimage`, `ideogram4`, `qwenimage`, `hidream`,
`ernie`) are text-only — a ref upload returns **400** naming the three above.

**Lane guidance:** if the klein 9B-KV emotion+headline test pair (dev-side)
shows legible in-image headline text at 4 steps, prefer `flux2klein9b` for
everything — it is ~6× cheaper per frame. `qwenimage-edit` is the fidelity
engine when text must be crisp/editable and time is acceptable.

## Calling `qwenimage-edit`

`POST http://192.168.0.87:8002/generate/qwenimage-edit` (multipart/form-data)

| Field | Required | Meaning |
|---|---|---|
| `prompt` | yes | **Edit instruction** for the uploaded image — state the change ("make them smile warmly") AND anything to preserve; text to render in-image should be spelled explicitly |
| `reference_image` | **yes** | the base face upload (jpg/png, ≤ 10 MB). No-ref → 400 — this checkpoint has no text-only path |
| `width`, `height` | no | default 1024×1024. **Canvas area cap ~1.05M px** — 1024² and 720×1440 (9:16 shorts) both fit; larger → 400 (measured VRAM ceiling) |
| `num_inference_steps` | no | default 20 (**~4 min**). Model card recommends 40 (~8 min) for max quality |
| `guidance_scale` | no | default 4.0 (maps to `true_cfg_scale` — the model-card setting) |
| `seed` | no | int. **Same seed + same params → byte-identical PNG** (verified live). Use it to reroll/compare |
| `negative_prompt` | no | optional; empty is fine |
| `num_images` | no | max **2** (each ~4 min at the 20-step default) |
| `quant` | no | Q4_K_S only tier (leave unset) |

**Example (bash):**

```bash
curl -X POST http://192.168.0.87:8002/generate/qwenimage-edit \
  -F 'prompt=Make this person smile warmly with eyes bright, and add bold \
white headline text across the top of the image that reads PIZZA IS LIFE in \
clean modern lettering, bright studio lighting, photorealistic, sharp detail' \
  -F 'reference_image=@channel-base-face.jpg' \
  -F 'width=720' -F 'height=1440' \
  -F 'num_inference_steps=20' -F 'guidance_scale=4.0' \
  -F 'seed=20260908'
```

**Example (python):**

```python
import requests
r = requests.post(
    "http://192.168.0.87:8002/generate/qwenimage-edit",
    files={"reference_image": ("base.jpg", open("channel-base-face.jpg", "rb"), "image/jpeg")},
    data={
        "prompt": "Make this person smile warmly ... reads PIZZA IS LIFE ...",
        "width": "720", "height": "1440",
        "num_inference_steps": "20", "guidance_scale": "4.0",
        "seed": "20260908",
    },
)
png = r.json()["results"][0]["base64"]   # or grab results[0]["url"] from the gallery
```

**Response:** `{"results": [{id, engine, filename, url, base64, width, height}]}`
— `base64` is the full PNG. The gallery (`GET /gallery`, `DELETE /gallery/{id}`)
keeps every run with `stats.total_s` / `stats.load_s` (epoch timing).

## Timing budget (the part that bites)

First request for a **unique (face bytes + prompt)** pair: **~5:10 total**
(load ~20 s + image-keyed embed encode + reload + ~4 min gen). Every repeat of
the same pair (e.g. a seed reroll): **~4:05** — the embeds are cached keyed by
sha256(prompt + image bytes), so **changing any byte of the face file or any
character of the prompt pays the ~65 s encode again**. The engine shares the
card with nothing else; idle engines are auto-evicted, so the first call after
a quiet period pays the load too. If the card is busy: 503 "Another generation
is already in progress" — poll and retry.

## Determinism cheat-sheet

- `qwenimage-edit`, `qwenimage`: same seed → pixel-identical (sha-verified).
- `flux2klein9b`: quant-stable identity; deterministic same-config reruns.
- `ideogram4`, `ernie`: **not** deterministic (kernel noise) — never rely on
  seeds there.

## Gotchas seen live (E2E 2026-09-08)

- No-ref and oversize-canvas requests 400 **fast** — do not treat a 400 as a
  queue issue; read `detail`.
- The prompt drives everything text-related: quote the headline exactly, keep
  the face-preservation language explicit ("this person", "the same person").
- Keep `width*height` ≤ ~1.05M px — 1536² is rejected by design.

## References

- [IMAGE_LAB_API_REFERENCE.md](IMAGE_LAB_API_REFERENCE.md) — full endpoint doc
  (status, gallery, evict, refresh)
- [IMGLAB_DEPLOYMENT_STATE_2026-09-08.md](IMGLAB_DEPLOYMENT_STATE_2026-09-08.md)
  — roster + measured benchmark table
- [SESSION_2026-09-08_IMGLAB_QWENIMAGE_EDIT.md](../sessions/SESSION_2026-09-08_IMGLAB_QWENIMAGE_EDIT.md)
  — requirement context + spike/E2E numbers
- Spike harness (reproducible): `scripts/probe/qwenimage_edit_probe.py`
