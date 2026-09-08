# Arthur Image Lab — API Reference

**Base URL:** `http://192.168.0.87:8002`  
**Protocol:** HTTP/1.1 — all generation requests are **synchronous** (connection held open until complete)  
**Auth:** None (local network only)  
**Revised:** 2026-09-08 — sd35 + wan removed (09-07); Z-Image, Qwen-Image 2512, HiDream O1, ERNIE-Image added; §3 now carries the lab's own measured benchmark timings (5060 Ti)

---

## Table of Contents

1. [Endpoints at a Glance](#1-endpoints-at-a-glance)
2. [GET /status](#2-get-status)
3. [POST /generate/{engine}](#3-post-generateengine)
4. [GET /engines](#4-get-engines)
5. [POST /engines/{engine}/load](#5-post-enginesengineload)
6. [POST /engines/unload](#6-post-enginesunload)
7. [POST /engines/{engine}/evict](#7-post-enginesengineevict)
8. [POST /evict-all](#8-post-evict-all)
9. [POST /refresh](#9-post-refresh)
10. [GET /files/{subdir}/{filename}](#10-get-filessubdirfilename)
11. [GET /gallery](#11-get-gallery)
12. [DELETE /gallery/{id}](#12-delete-galleryid)
13. [Engine Parameters Reference](#13-engine-parameters-reference)
14. [Response Schemas](#14-response-schemas)
15. [Error Reference](#15-error-reference)
16. [curl Cookbook](#16-curl-cookbook)
17. [Python Cookbook](#17-python-cookbook)

---

## 1. Endpoints at a Glance

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/status` | Live engine status, VRAM, active engine |
| `POST` | `/generate/{engine}` | Run generation (blocks until done) |
| `GET` | `/engines` | Engine metadata (static, no state) |
| `POST` | `/engines/{engine}/load` | Preload engine into VRAM (non-blocking, optional `quant`) |
| `POST` | `/engines/unload` | Evict current engine from VRAM |
| `POST` | `/engines/{engine}/evict` | Evict ONE engine (only if it is the resident one) |
| `POST` | `/evict-all` | Whole-card eviction — Image Lab engine **and** all TTS engines via the orchestrator |
| `POST` | `/refresh` | Re-probe engine availability without restart |
| `GET` | `/files/images/{filename}` | Download generated PNG |
| `GET` | `/files/videos/{filename}` | Download generated MP4 |
| `GET` | `/gallery` | List past generations |
| `DELETE` | `/gallery/{id}` | Delete a gallery entry + file |

---

## 2. GET /status

Returns live service state: all engine availability, which engine is loaded, VRAM, host RAM, and a device-wide GPU report with per-process attribution. The web UI polls this endpoint every 4 s with `?brief=1`.

### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `brief` | int | `0` | `?brief=1` drops each engine's `description` and `params` (keeps `key`/`label`/`available`/`loaded`/`error`). The full `/status` is fetched once at UI boot; the 4 s poll always uses `brief=1` to stay light. |

### Response — 200

```json
{
  "engines": [
    {
      "key":         "flux2klein",
      "label":       "FLUX.2 Klein 4B",
      "description": "FLUX.2 Klein 4B — compact 4B flow transformer...",
      "output_type": "image",
      "vram_gb":     13.0,
      "available":   true,
      "loaded":      true,
      "error":       "",
      "params":      [ ... ]
    },
    { "key": "zimage",    "label": "Z-Image Turbo",      "available": true, "loaded": false, "error": "" },
    { "key": "qwenimage", "label": "Qwen-Image 2512",    "available": true, "loaded": false, "error": "" },
    { "key": "hidream",   "label": "HiDream O1",         "available": true, "loaded": false, "error": "" },
    { "key": "ernie",     "label": "ERNIE-Image",        "available": true, "loaded": false, "error": "" }
  ],
  "active_engine": "flux2klein",
  "active_quant":  "",
  "generating":    false,
  "loading":       false,
  "vram": {                                // this process's torch view (GB)
    "available":    true,
    "allocated_gb": 0.01,
    "reserved_gb":  0.04,
    "total_gb":     15.48,
    "free_gb":      15.43,
    "device_name":  "NVIDIA GeForce RTX 5060 Ti"
  },
  "system": {                              // host RAM (MB) — whole-card context
    "total": 31914,
    "used":  15022,
    "free":  16892
  },
  "gpu": {                                 // device-wide nvidia-smi view (MB) + processes
    "available":     true,
    "name":          "NVIDIA GeForce RTX 5060 Ti",
    "vram_total_mb": 16280,
    "vram_used_mb":  9216,
    "vram_free_mb":  7064,
    "source":        "nvidia-smi",
    "ts":            1779640492.3,
    "processes": [
      { "pid": 5121, "mb": 6144, "process": "python",   "container": "" },
      { "pid": 2093, "mb": 2970, "process": "python3",  "container": "tts-lab-engine-current" }
    ]
  }
}
```

### Field Notes

| Field | Description |
|---|---|
| `available` | `true` if engine dependencies are importable and model files exist |
| `loaded` | `true` if this engine is currently in VRAM (only one can be `true` at a time) |
| `active_engine` | Key of the loaded engine, or `null` if nothing is loaded |
| `active_quant` | Quantization level of the loaded engine (e.g. `"Q3_K_M"`), empty for BF16 |
| `generating` | `true` while a generation is running — further `/generate` calls return `503` |
| `loading` | `true` while a model is being loaded (30–90 s) — the load endpoints are async, so `/status` stays live during a load |
| `vram` | This process's PyTorch allocator view in GB (`reserved_gb` = model + cache). Kept for compatibility with the older UI. |
| `system` | Host RAM in MB — psutil, else `/proc/meminfo`. Zeros when unavailable. |
| `gpu` | Device-wide driver view in MB (`nvidia-smi`, TTL-cached ~2 s). `processes` lists every CUDA process on the card with its container name (`""` = bare-metal host process, e.g. Image Lab itself) — this is what the UI's "who holds the VRAM" line renders. Falls back to the torch view with `source: "torch"` and empty `processes` when nvidia-smi is missing. |
| `brief=1` | Engine items drop `description`/`params` — the UI merges these only from full polls. |

---

## 3. POST /generate/{engine}

Runs image or video generation. **Synchronous — the connection is held open until the result is ready.**

### Measured timings (the lab's own RTX 5060 Ti 16 GB, 1024×1024)

Measured **2026-09-07/08** via curl against the live service — wall-clock, round-trip and file save included, at each engine's **default configuration** unless noted. `cold` = engine not resident (load + generate); `warm` = engine already loaded; `load → ready` = load portion alone.

| Engine (config) | Load → ready | Warm gen | Cold wall (load + gen) | Same-seed determinism |
|---|---|---|---|---|
| `flux2klein` (BF16, 4 st) | 13.0 s | ~20.4 s | 33.4 s | pixel-identical (BF16) |
| `flux2klein9b` (Q6_K, 4 st) | 12–24 s | 25.3 s (09-05) | 36–47 s swap runs (09-05); 118 s cold on the 09-08 regression | quant-stable identity (cosine 0.958–0.985 Q4↔Q6) |
| `ideogram4` (nf4, **28 st** — default preset is 20) | ~90 s | 236 s gen / 244 s wall | 758 s (incl. 502 s one-off re-download) | **NO** — bnb nf4 matmul noise |
| `sana` sprint (4 st) | 124 s cold / 9 s warm reload | 20.8 s | 53.7 s | pixel-identical |
| `sana` 1.5 (20 st, CFG 4.5) | same key | not re-timed | ~118 s reload | differs from Sprint (as designed) |
| `boogu` (4 st) | 383 s first (incl. 14 GB cache top-up) | 37–41 s | 86 s | pixel-identical |
| `zimage` (Q4_K_M, 8 st) | 16.3 s / 4.90 GiB | 48 s | ~48 s class | n/v |
| `qwenimage` (Q4_K_S, 20 st, CFG 4.0) | 35.8 s / 11.67 GiB | 191 s | 247 s | pixel-identical |
| `hidream` (fp8_scaled, 28 st — Comfy sidecar) | sidecar-owned | not re-timed at 1024² — ~80 s @2048×1376 (research-doc ref) | — | byte-identical across sidecar reload |
| `ernie` (NVFP4, 8 st) | 31.2 s cold / 25.7 s warm | **12.8 s** | 40.9 s | near — seed honored, NVFP4 kernels never pixel-identical |

**Speed guide — fastest → slowest, one 1024² image, warm, default settings:** `ernie` 12.8 s < `flux2klein` ~20.4 s ≈ `sana` sprint 20.8 s < `flux2klein9b` 25.3 s < `boogu` 37–41 s < `zimage` 48 s < `hidream` ~80 s (28 st, sidecar ref) < `qwenimage` 191 s < `ideogram4` 236 s @28 st (≈ ¾ of that at the default 20-step preset — estimated).

Planning notes for remote users:

- **Scaling:** generation time scales roughly with step count. Distilled engines give their best at 1–8 steps and are the fast tier (ernie, klein ×2, zimage, boogu, sana-sprint, hidream — all CFG-free). The slow tier is the text-rendering specialists that pay a CFG double pass: qwenimage (20 st ≈ 191 s) and ideogram4 (20–48 st ≈ 4–8 min).
- **VRAM:** every load evicts the previous resident engine, and mid-gen peaks exceed the load gate — ideogram4 CFG ≈ 14.0 GiB, qwenimage ≈ 14.2 GiB driver footprint, klein9b Q6 14,234 MiB used / 1,615 free. Heavy engines need a nearly clean card; a resident TTS container model can trip the gate → `503 CUDA out of memory` (clear with `POST /evict-all`, §8).
- **Reproducibility:** `seed=-1` = random; the resolved seed is echoed in `params.seed`. ideogram4 and ernie reruns differ slightly on the same seed (quantised-kernel noise) — judge those two by seed-family similarity, not pixels.

Detailed methodology / per-run record: `IMGLAB_DEPLOYMENT_STATE_2026-09-08.md` §3 (internal doc).

> `flux2` (FLUX.2 [dev] 32B) was REMOVED 2026-08-13; `sd35` + `wan` were
> REMOVED 2026-09-07 — see `ARTHUR_IMAGE_LAB_REFERENCE.md` §4 for the history.

### Content-Type

`multipart/form-data` — all fields are form fields (not JSON body).

### URL Parameter

| Parameter | Description |
|---|---|
| `engine` | `flux2klein` \| `flux2klein9b` \| `ideogram4` \| `sana` \| `boogu` \| `zimage` \| `qwenimage` \| `hidream` \| `ernie` |

### Common Form Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Text description of the image or video to generate |
| `negative_prompt` | string | `""` | What NOT to include. Only the CFG-capable engines take it: flux2klein ×2, qwenimage, sana's 1.5-1.6b variant. Ignored elsewhere — CFG-free models clamp guidance (zimage 0.0; boogu/hidream/ernie 1.0) and have no negative-prompt path. |
| `width` | int | `1024` | Output width in pixels. Must be a multiple of 64. |
| `height` | int | `1024` | Output height in pixels. Must be a multiple of 64. |
| `num_inference_steps` | int | engine default | Denoising steps. More = better quality, slower. |
| `guidance_scale` | float | engine default | Prompt adherence strength. |
| `seed` | int | `-1` | `-1` = random. Fixed value = reproducible output. |
| `quant` | string | engine default | Quantization level. See [Engine Parameters Reference](#13-engine-parameters-reference). |
| `reference_image` | file | `null` | Optional image upload for I2I — FLUX.2 Klein engines only (wan I2V removed 2026-09-07). |

### Engine-Specific Fields

| Field | Type | Default | Engines | Description |
|---|---|---|---|---|
| `num_images` | int | `1` | engines with num_images in §13 | How many images to generate per request (1–2 qwenimage, 1–4 others that expose it). |

### Success Response — 200

```json
{
  "results": [
    {
      "id":         "3f2a1b9c-4d5e-6789-abcd-ef0123456789",
      "engine":     "flux2klein",
      "filename":   "flux2klein_3f2a1b9c-4d5e-6789-abcd-ef0123456789.png",
      "url":        "/files/images/flux2klein_3f2a1b9c-...png",
      "base64":     "iVBORw0KGgo...",
      "type":       "image",
      "width":      1024,
      "height":     1024,
      "params": {
        "prompt": "a red fox in a snowy forest",
        "seed":   1847392810,
        "width":  1024,
        "height": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 3.5
      },
      "stats": {
        "started_at":  1779640490.2,
        "finished_at": 1779640492.3,
        "load_s":      1.9,
        "total_s":     2.1
      },
      "created_at": 1779640492.3
    }
  ]
}
```

**Notes:**
- `results` is always an array. Most engines return 1 item; engines exposing `num_images` return up to their per-engine max (1–4).
- `base64` contains the full PNG encoded as base64. For videos, `base64` is `null` (too large).
- `url` is a relative path — prepend the base URL to fetch the file.
- `params.seed` is the actual seed used (even if you sent `-1`, the resolved random seed is returned).
- `stats` (per-image run timing, present on every entry saved since 2026-09-06) holds epoch `started_at`/`finished_at` and the load/generation split: `total_s` spans request start → file saved, `load_s` is the model-load portion (`null` if the model was already resident). Generation time ≈ `total_s − load_s`. Older gallery rows simply lack `stats`.
- `params.prompt` is the raw submitted prompt. For Ideogram 4 the text **actually sent to the model** is recorded separately under `params.caption` — the magic-prompt-expanded caption (equals `prompt` when expansion is off or fails). The UI shows the caption as the primary prompt block when the two differ.

### Error Responses

| HTTP | Condition |
|---|---|
| `400` | Invalid parameter value (e.g. unknown quant level) |
| `404` | Unknown engine key |
| `500` | Unhandled exception during generation |
| `503` | CUDA OOM or other runtime failure |

---

## 4. GET /engines

Returns static engine metadata (no live state — for available/loaded, use `/status`).

### Response — 200

```json
{
  "flux2klein": { ... },
  "flux2klein9b": { ... },
  "ideogram4": { ... },
  "sana":      { ... },
  "boogu":     { ... },
  "zimage":    { ... },
  "qwenimage": { ... },
  "hidream":   { ... },
  "ernie":     { ... }
}
```

---

## 5. POST /engines/{engine}/load

Pre-loads an engine into VRAM without generating anything. Useful for warming up before the first request.

**Non-blocking:** the load runs in a worker thread — the HTTP call stays open for the duration (seconds to several minutes — see the §3 load column), but `/status` and `/logs` keep responding and report `"loading": true` so clients can poll rather than hang. Requests are single-flight: a second preload while one is running (or during generation) returns `503`.

### Form Field

| Field | Type | Default | Description |
|---|---|---|---|
| `quant` | string | `""` | Quantization level to load (e.g. `"Q5_K_M"` for zimage, `"Q6_K"` for flux2klein9b, or a SANA variant like `"1.5-1.6b"`). Empty = engine default. Changing quant on an already-loaded engine reloads it. |

### Response — 200

```json
{ "loaded": "zimage", "quant": "Q5_K_M" }
```

### Response — 503 (server busy)

```json
{ "detail": "Server is busy" }
```

---

## 6. POST /engines/unload

Evicts the currently-loaded engine from VRAM. Useful for freeing VRAM between sessions.

### Response — 200

```json
{ "unloaded": true }
```

---

## 7. POST /engines/{engine}/evict

Evicts ONE engine — but only if it is the currently-resident one (the Image Lab is single-resident, so this is effectively the per-engine form of `/engines/unload`). The UI's resident-chip ✕ button calls this with the loaded engine's key.

Returns `200` (not an error) when the engine is not resident — nothing to do.

### Response — 200 (was resident → unloaded)

```json
{ "evicted": true, "engine": "flux2klein", "mode": "local-unload" }
```

### Response — 200 (not resident)

```json
{ "evicted": false, "engine": "ernie", "note": "not resident" }
```

### Response — 503 (server busy)

```json
{ "detail": "Server is busy" }
```

---

## 8. POST /evict-all

**Whole-card eviction** — the Image Lab shares its 16 GB RTX 5060 Ti with the TTS engine containers, so this unloads:

1. The Image Lab's resident engine (if any — skipped with a note when a generation/load is in flight), then
2. Every TTS engine container, by POSTing `http://localhost:8009/evict-all` on the TTS orchestrator (the canonical broker — TTS containers publish no host ports).

The TTS call is best-effort: the orchestrator being down is reported in `errors`, not raised. Used by the UI's **Evict VRAM** button.

### Response — 200

```json
{
  "image_lab": { "unloaded": true, "engine": "flux2klein" },
  "tts":       { "evicted_count": 3, "freed_mb_total": 9216, ... },
  "errors":    []
}
```

`tts` is the orchestrator's payload verbatim, or `{ "error": "..." }` when unreachable. Each side's failure lands in `errors` (`{ "side": "image_lab" | "tts", "error": "..." }`).

---

## 9. POST /refresh

Re-runs the engine availability probe (`importlib` spec checks) without restarting the service. Use after installing a missing dependency or adding a model file, to flip an engine's `available` flag without a redeploy.

### Response — 200

```json
{ "refreshed": true }
```

### Response — 503 (server busy)

```json
{ "detail": "Server is busy" }
```

---

## 10. GET /files/{subdir}/{filename}

Serves a generated image or video file directly.

| `subdir` | Content-Type | File extension |
|---|---|---|
| `images` | `image/png` | `.png` |
| `videos` | `video/mp4` | `.mp4` |

`filename` must be the exact filename returned in the `url` field of a generate response (e.g. `flux2klein_3f2a1b9c-....png`). Path traversal is rejected (`../` etc.).

### Response — 200

Raw PNG or MP4 binary.

### Response — 404

File not found or invalid subdir.

---

## 11. GET /gallery

Returns a paginated list of past generations.

### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `50` | Max entries to return |
| `offset` | int | `0` | Skip N entries (for pagination) |
| `engine` | string | *(all)* | Filter by engine key (e.g. `?engine=zimage`) |

### Response — 200

```json
{
  "entries": [
    {
      "id":         "3f2a1b9c-...",
      "engine":     "flux2klein",
      "filename":   "flux2klein_3f2a1b9c-....png",
      "url":        "/files/images/flux2klein_3f2a1b9c-....png",
      "base64":     "iVBORw0KGgo...",
      "type":       "image",
      "width":      1024,
      "height":     1024,
      "params":     { ... },
      "stats": {
        "started_at":  1779640490.2,
        "finished_at": 1779640492.3,
        "load_s":      1.9,
        "total_s":     2.1
      },
      "created_at": 1779640492.3
    }
  ],
  "limit":  50,
  "offset": 0
}
```

Entries are ordered newest-first. The gallery stores the last 500 entries on disk. `base64` is stripped from listings (fetch `/files/...` instead). Entries saved before 2026-09-06 have no `stats` key.

---

## 12. DELETE /gallery/{id}

Deletes a gallery entry and its associated file from disk.

`id` is the UUID string from the `id` field of any gallery entry.

### Response — 200

```json
{ "deleted": "3f2a1b9c-4d5e-6789-abcd-ef0123456789" }
```

### Response — 404

```json
{ "detail": "Generation not found" }
```

---

## 13. Engine Parameters Reference

### `flux2` — FLUX.2 [dev] — 🗑️ REMOVED

> Removed 2026-08-13. The 32B dev model (~27 GB VRAM needed) cannot run on the
> 15.5 GiB card without CPU offloading, which is against the GPU-only policy.
> Engine entry, loaders, UI tab, and all model files (~51 GB) deleted. See
> `ARTHUR_IMAGE_LAB_REFERENCE.md` §4.1 for the full history. Former params:
> prompt / reference_image / width-height 256-2048 / steps 28 / guidance 3.5 /
> seed / quant (Q3_K_M 16 GB · Q4_K_M 20 GB · Q5_K_M 24 GB · Q8_0 35 GB ·
> nvfp4 8 GB).

---

### `flux2klein` — FLUX.2 Klein 4B

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `negative_prompt` | string | `""` | — | |
| `reference_image` | file | null | — | I2I / style transfer |
| `width` | int | `1024` | 256–2048, step 64 | |
| `height` | int | `1024` | 256–2048, step 64 | |
| `num_inference_steps` | int | `4` | 1–20 | Step-distilled — 4 is optimal |
| `guidance_scale` | float | `3.5` | 1.0–10.0 | Ignored by distilled model |
| `seed` | int | `-1` | -1 to 2³¹-1 | |

`quant` field is ignored — always runs in BF16. No GGUF available for this model.

---

### `flux2klein9b` — FLUX.2 Klein 9B-KV

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `negative_prompt` | string | `""` | — | |
| `reference_image` | file | null | — | I2I / style transfer |
| `width` | int | `1024` | 256–2048, step 64 | |
| `height` | int | `1024` | 256–2048, step 64 | |
| `num_inference_steps` | int | `4` | 1–20 | Step-distilled — 4 is optimal |
| `guidance_scale` | float | `3.5` | 1.0–10.0 | Ignored by distilled model (no CFG pass) |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q6_K` | see below | GGUF transformer (`from_single_file`) + Qwen3-8B **NF4** encoder + Flux2 VAE |

**`quant` options:** `Q3_K_M` (4.6 GB — smallest usable for faces) · `Q4_K_M` (5.7 GB — previous default) · `Q5_K_M` (6.8 GB) · `Q6_K` (7.9 GB) ✓ default · `Q8_0` (10.0 GB — near-lossless, **exceeds the 16 GB card** — documented OOM rung, kept for a future bigger card / encoder-offload).

---

### `ideogram4` — Ideogram 4 (hosted-API preset ladder)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | Plain-English caption — **no** `negative_prompt` on this engine |
| `width` | int | `1024` | 256–2048, step 64 | |
| `height` | int | `1024` | 256–2048, step 64 | |
| `preset` | string | `V4_DEFAULT_20` | see below | Ladder mirrors the hosted Ideogram API tiers |
| `num_inference_steps` | int | `0` | 0–128 | `0` = follow the preset's step count; nonzero overrides it |
| `guidance_scale` | float | `7.0` | 1.0–30.0 | |
| `mu` | float | `0.0` | -5.0–5.0 | Caption-lift sampler param |
| `std` | float | `1.75` | 0.1–5.0 | Caption-lift sampler param |
| `use_magic_prompt` | bool | `false` | — | Magic-prompt caption expansion (hosted API — needs `IDEOGRAM_API_KEY`) |
| `magic_prompt_aspect_ratio` | string | `1:1` | 1:1 · 3:2 · 2:3 · 16:9 … | Aspect hint for the expansion service |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `nf4` | `nf4` · `fp8` | `nf4` ~6 GB, CUDA only ✓ default; `fp8` ~10 GB, any device |

**`preset` options:** `V4_QUALITY_48` (48 steps, best quality) · `V4_DEFAULT_20` (20 steps, recommended) ✓ default · `V4_TURBO_12` (12 steps, fastest).

Slowest engine in the lab (~4 min/image at 28 steps measured; see §3) but the strongest pure photo/illustration quality, with its own caption-expansion path. The 2026-08-14 blank-image bug is fixed — see `IDEOGRAM4_FIX_2026-08-14.md`. **Not deterministic on the same seed** (bnb nf4 matmul noise).

---

### `sana` — SANA 1.6B (Sprint)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `negative_prompt` | string | `""` | — | Only the 1.5-1.6b (CFG) variant takes it |
| `width` | int | `1024` | 256–2048, step 32 | |
| `height` | int | `1024` | 256–2048, step 32 | |
| `num_inference_steps` | int | `4` | 1–24 | Sprint: 1–4 (distilled); 1.5-1.6b: ~20. Non-2-step Sprint runs get the linear timestep fallback (`intermediate_timesteps=None`, the SCM fix) |
| `guidance_scale` | float | `4.5` | 1.0–20.0 | Used by the 1.5-1.6b variant; Sprint runs CFG-free |
| `num_images` | int | `1` | 1–4 | |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `sprint-1.6b` | see below | Picks the pipeline + checkpoint |

**`quant` options:** `sprint-1.6b` ✓ default — `SanaSprintPipeline`, 1–4 steps, CFG-free (~21 s/img measured); `1.5-1.6b` — `SanaPipeline` + SCM scheduler, ~20 steps CFG 4.5, higher fidelity, slower (not re-timed on this card).

---

### `boogu` — Boogu-Image 0.1 Turbo

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `width` | int | `1024` | 256–1536, step 16 | |
| `height` | int | `1024` | 256–1536, step 16 | |
| `num_inference_steps` | int | `4` | 1–8 | Distilled — 4 optimal |
| `guidance_scale` | float | `1.0` | fixed 1.0 | Distilled guidance-free — clamped |
| `num_images` | int | `1` | 1–2 | |
| `seed` | int | `-1` | -1 to 2³¹-1 | |

**Resource exception — the only CPU-offload engine:** fp8 Qwen3-VL-8B mllm + bf16 transformer + FLUX.1 VAE; GPU peaks ~12.6 GiB **plus ~27 GB host RAM**. No `quant` param.

---

### `zimage` — Z-Image Turbo

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | Plain-prompt friendly (no JSON caption needed) |
| `width` | int | `1024` | 256–1536, step 16 | |
| `height` | int | `1024` | 256–1536, step 16 | |
| `num_inference_steps` | int | `8` | 1–8 | Distilled — 8 is the training target |
| `guidance_scale` | float | `0.0` | fixed 0.0 | CFG-free model — server forces 0.0 (no negative prompt) |
| `num_images` | int | `1` | 1–4 | |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q4_K_M` | see below | GGUF transformer (`jayn7/Z-Image-Turbo-GGUF`) |

**`quant` options:** `Q3_K_S` (~3.5 GB — fastest) · `Q3_K_M` (~3.8 GB) · `Q4_K_S` (~4.3 GB) · `Q4_K_M` (~4.6 GB) ✓ default/recommended · `Q5_K_S` (~4.8 GB) · `Q5_K_M` (~5.1 GB) · `Q6_K` (~5.5 GB) · `Q8_0` (~6.7 GB — best quality). GGUF transformer ladder (`jayn7/Z-Image-Turbo-GGUF`); Q4_K_M is the quality/speed sweet spot — drop rungs only when VRAM is tight.

---

### `qwenimage` — Qwen-Image 2512

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | Dense bilingual (EN/ZH) layouts are the strength |
| `negative_prompt` | string | `""` | — | Supported (CFG path) |
| `width` | int | `1024` | 256–1536, step 16 | |
| `height` | int | `1024` | 256–1536, step 16 | |
| `num_inference_steps` | int | `20` | 1–50 | 20 ✓ daily tier (191 s measured); 50 = max quality (≈ 3× the 20-step time) |
| `guidance_scale` | float | `4.0` | 1.0–8.0 | Maps to `true_cfg_scale` |
| `num_images` | int | `1` | 1–2 | ≈ 191 s per image at 20 steps — a 2-image request roughly doubles it |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q4_K_S` | see below | GGUF transformer (`unsloth/Qwen-Image-2512-GGUF`) |

**`quant` options:** `Q4_K_S` (~11.5 GB) ✓ default — the **only** tier on this card. The Q4_K_M rung (12.34 GB) loads then OOMs at `to("cuda")` with the card's TTS co-tenant contexts (~14.2 GiB process floor vs 15.48 GiB), so the ladder was collapsed 2026-09-08 and the dead GGUF deleted. Embed-cache encodes run with the transformer unloaded.

---

### `hidream` — HiDream O1-Dev (ComfyUI sidecar)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `width` | int | `1024` | 256–2048, step 64 | |
| `height` | int | `1024` | 256–2048, step 64 | |
| `num_inference_steps` | int | `28` | fixed 28 | Dev checkpoint — fixed-step sampler |
| `guidance_scale` | float | `1.0` | fixed 1.0 | Distilled CFG-free sampler — field is clamped to 1.0 (no negative-prompt path) |
| `num_images` | int | `1` | 1–4 | |
| `seed` | int | `-1` | -1 to 2³¹-1 | |

Generation runs **out-of-process**: the lab unloads its resident engine, POSTs
the workflow JSON to the ComfyUI sidecar (port 8188, `arthur-comfy.service`),
polls `/history`, pulls the PNG via `/view`, and hands VRAM back via
`/free {"unload_models": true}`. No `quant` param — the Dev fp8_scaled
checkpoint is fixed.

---

### `ernie` — ERNIE-Image-Turbo

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | CN+EN poster/layout strength |
| `width` | int | `1024` | 256–1536, step 16 | |
| `height` | int | `1024` | 256–1536, step 16 | |
| `num_inference_steps` | int | `8` | fixed 8 | Distilled — fixed 8-step sampler |
| `guidance_scale` | float | `1.0` | fixed 1.0 | Distilled guidance-free — server forces 1.0 (no negative prompt) |
| `num_images` | int | `1` | 1–4 | |
| `seed` | int | `-1` | -1 to 2³¹-1 | |

No `quant` param — pre-quantised Nunchaku-Lite NVFP4 transformer + bnb-4bit
Ministral-3 text encoder (lite-infer repo), fp8/bf16 fallback if that route
can't load.

---

### `sd35` — Stable Diffusion 3.5 Large — 🗑️ REMOVED

> Removed 2026-09-07 — superseded by the Z-Image / Qwen-Image-2512 /
> HiDream-O1 / ERNIE-Image round. Former params: prompt / negative_prompt /
> width-height 256-1536 step 64 / steps 1-100 (28) / guidance 1.0-20.0 (4.5) /
> num_images 1-4 / seed / quant (Q4_0 ~4.8 GB · Q5_0 ~5.8 GB · Q8_0 ~8.8 GB ·
> nvfp4 ~2 GB). History: `ARTHUR_IMAGE_LAB_REFERENCE.md` §4.2.

---

### `wan` — Wan2.2 (Text-to-Video / Image-to-Video) — 🗑️ REMOVED

> Removed 2026-09-07 — video engines dropped (user decision); the lab is
> all-image. Former fields: mode (t2v/i2v) / reference_image (i2v) /
> num_frames 16-120 / fps 8-24 / resolution 480p-720p / quant (Q3_K_M–Q8_0 +
> nvfp4 per dual transformer). History: `ARTHUR_IMAGE_LAB_REFERENCE.md` §4.3.

---

## 14. Response Schemas

### Generation Result Object

```typescript
{
  id:         string;        // UUID — use this for gallery DELETE
  engine:     string;        // any key from GET /engines (e.g. "zimage" | "qwenimage" | "hidream" | "ernie")
  filename:   string;        // e.g. "flux2klein_3f2a1b9c-....png"
  url:        string;        // Relative URL — prepend base URL to fetch
  base64:     string | null; // PNG as base64 string; null for videos
  type:       "image" | "video";
  width?:     number;        // Image width in pixels (images only)
  height?:    number;        // Image height in pixels (images only)
  fps?:       number;        // Frame rate (videos only)
  num_frames?: number;       // Frame count (videos only)
  params:     object;        // Echo of generation params (seed resolved;
                             //   ideogram4 additionally stores the effective
                             //   magic-prompt caption under params.caption)
  stats?:     {              // Present on entries saved since 2026-09-06
    started_at:  number;     //   epoch — generate() request began
    finished_at: number;     //   epoch — file saved
    load_s:      number|null;//   model-load portion (null when already resident)
    total_s:     number;     //   total wall time (request start → saved)
  };
  created_at: number;        // Unix timestamp (float)
}
```

### VRAM Object (inside /status — `vram`)

```typescript
{
  available:    boolean;   // false if CUDA not available
  allocated_gb: number;    // PyTorch actively allocated
  reserved_gb:  number;    // PyTorch reserved (model + cache)
  total_gb:     number;    // GPU total capacity (15.48 on RTX 5060 Ti)
  free_gb:      number;    // total_gb - reserved_gb
  device_name:  string;    // "NVIDIA GeForce RTX 5060 Ti"
}
```

### GPU Report Object (inside /status — `gpu`, device-wide)

```typescript
{
  available:     boolean;   // false if no GPU / nvidia-smi and torch both fail
  name:          string;    // "NVIDIA GeForce RTX 5060 Ti"
  vram_total_mb: number;    // 16280 on this card
  vram_used_mb:  number;    // includes ALL processes (Image Lab + TTS containers)
  vram_free_mb:  number;
  source:        string;    // "nvidia-smi" | "torch" (fallback)
  ts:            number;    // cache timestamp
  processes:     Array<{   // every CUDA process on the card, sorted by MB desc
    pid:        number;
    mb:         number;
    process:    string;     // process name (e.g. "python3")
    container:  string;     // container name, "" = bare-metal host process
  }>;
}
```

### System Object (inside /status — `system`, host RAM)

```typescript
{ total: number; used: number; free: number; }   // MB
```

---

## 15. Error Reference

All errors follow FastAPI's default shape:

```json
{ "detail": "Human-readable error message" }
```

| HTTP | When |
|---|---|
| `400 Bad Request` | Invalid field value — e.g. unknown `quant` string |
| `404 Not Found` | Unknown engine key, missing file, missing gallery entry |
| `500 Internal Server Error` | Unhandled Python exception (check `journalctl` on VM) |
| `503 Service Unavailable` | CUDA OOM, model load failed, or server busy during preload |

**Common 503 messages:**

```
CUDA out of memory. Tried to allocate X GiB.
  → Evict other engines (POST /evict-all), or switch to a smaller quant

Server is busy
  → Poll /status until generating=false, then retry

ComfyUI sidecar unreachable (hidream)
  → Check arthur-comfy.service + http://127.0.0.1:8188/system_stats
```

---

## 16. curl Cookbook

### Quick image — FLUX.2 Klein 4B (4 steps, ~20 s warm)

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2klein \
  -F "prompt=a golden retriever in a meadow at sunset" \
  -F "num_inference_steps=4" \
  -F "seed=42"
```

### High quality image — Qwen-Image 2512 with negative prompt

```bash
curl -X POST http://192.168.0.87:8002/generate/qwenimage \
  -F "prompt=a golden retriever in a meadow at sunset, golden hour, bokeh" \
  -F "negative_prompt=blurry, low quality, watermark" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=4.0" \
  -F "quant=Q4_K_S"   # only tier on this card (see §13)
```

### Typographic poster — Z-Image Turbo (plain prompt, 8 steps)

```bash
curl -X POST http://192.168.0.87:8002/generate/zimage \
  -F "prompt=a minimal concert poster with the text 'ARTHUR LAB' in bold white letters" \
  -F "num_images=2" \
  -F "seed=100"
```

### Fast high-quality — FLUX.2 Klein 4B (distilled)

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2klein \
  -F "prompt=a photorealistic mountain lake at dawn, misty, reflections" \
  -F "num_inference_steps=4" \
  -F "width=1024" \
  -F "height=1024"
```

### Image editing — FLUX.2 Klein with reference image

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2klein \
  -F "prompt=same scene but at night, moonlight, stars" \
  -F "reference_image=@/path/to/input.png"
```

### Out-of-process — HiDream O1 (ComfyUI sidecar)

```bash
# The lab unloads its own engine, runs the job in the ComfyUI sidecar,
# pulls the PNG, and frees ComfyUI's VRAM afterwards.
curl -X POST http://192.168.0.87:8002/generate/hidream \
  -F "prompt=editorial illustration with the headline 'ARTHUR' in huge pixel-sharp type" \
  -F "width=1024" \
  -F "height=1024" \
  -F "seed=7"
```

### Fastest — ERNIE-Image-Turbo (NVFP4, fixed 8 steps, ~13 s warm)

```bash
curl -X POST http://192.168.0.87:8002/generate/ernie \
  -F "prompt=event poster layout, CN+EN bilingual text, autumn colors" \
  -F "seed=2026"
```

### Save image from response (jq)

```bash
curl -s -X POST http://192.168.0.87:8002/generate/flux2klein \
  -F "prompt=a lighthouse on a rocky coast" \
  -F "seed=999" \
| jq -r '.results[0].base64' \
| base64 -d > output.png
```

### Preload engine before generating

```bash
# Preload zimage into VRAM (~30-60 s load time)
curl -X POST http://192.168.0.87:8002/engines/zimage/load

# Poll until loading=false
watch -n 2 'curl -s http://192.168.0.87:8002/status | jq "{loading,active_engine}"'

# Now generate instantly (model already loaded)
curl -X POST http://192.168.0.87:8002/generate/zimage \
  -F "prompt=a futuristic city at night"
```

### Check VRAM before generating

```bash
# This process's torch view (GB)
curl -s http://192.168.0.87:8002/status | jq '.vram | {reserved_gb, free_gb, total_gb}'

# Device-wide view incl. TTS containers (MB) + who holds what
curl -s http://192.168.0.87:8002/status | jq '.gpu | {used_mb: .vram_used_mb, total_mb: .vram_total_mb}'
curl -s http://192.168.0.87:8002/status | jq '.gpu.processes[] | "\(.container // "host") \(.mb) MB"'

# Lightweight poll payload (no per-engine param schemas)
curl -s "http://192.168.0.87:8002/status?brief=1"
```

### Evict everything from VRAM (Image Lab + all TTS engines)

```bash
curl -s -X POST http://192.168.0.87:8002/evict-all
# → {"image_lab":{"unloaded":true,"engine":"flux2klein"},
#    "tts":{"evicted_count":3,"freed_mb_total":9216,...},
#    "errors":[]}

# Evict one engine (resident only)
curl -s -X POST http://192.168.0.87:8002/engines/flux2klein/evict

# Re-probe availability after fixing a dependency / adding model files
curl -s -X POST http://192.168.0.87:8002/refresh
```

### Preload with a specific quant

```bash
curl -s -X POST http://192.168.0.87:8002/engines/zimage/load \
  -F "quant=Q5_K_M"     # async — /status reports loading:true while it runs
```

### Filter gallery by engine

```bash
curl "http://192.168.0.87:8002/gallery?engine=flux2klein&limit=10"
```

### Delete a gallery entry

```bash
curl -X DELETE http://192.168.0.87:8002/gallery/3f2a1b9c-4d5e-6789-abcd-ef0123456789
```

### Download a file directly

```bash
curl -o image.png "http://192.168.0.87:8002/files/images/flux2klein_3f2a1b9c-....png"
```

---

## 17. Python Cookbook

### Simple generation + save

```python
import requests, base64, json

BASE = "http://192.168.0.87:8002"

def generate(engine: str, **kwargs) -> list[dict]:
    resp = requests.post(
        f"{BASE}/generate/{engine}",
        data=kwargs,          # multipart/form-data
        timeout=900,          # 15 min — large models take time
    )
    resp.raise_for_status()
    return resp.json()["results"]

# Text-to-image
results = generate("flux2klein",
    prompt="a futuristic city skyline at dusk",
    num_inference_steps=4,
    seed=42,
)

# Save the PNG
for r in results:
    img_bytes = base64.b64decode(r["base64"])
    with open(r["filename"], "wb") as f:
        f.write(img_bytes)
    print(f"Saved {r['filename']} ({r['width']}x{r['height']})")
```

### With reference image (I2I)

```python
def generate_i2i(engine: str, prompt: str, image_path: str, **kwargs) -> list[dict]:
    with open(image_path, "rb") as img:
        resp = requests.post(
            f"{BASE}/generate/{engine}",
            data={"prompt": prompt, **kwargs},
            files={"reference_image": (image_path, img, "image/png")},
            timeout=900,
        )
    resp.raise_for_status()
    return resp.json()["results"]

results = generate_i2i("flux2klein",
    prompt="same scene but in winter with snow",
    image_path="input.png",
)
```

### Poll status until ready

```python
import time

def wait_for_idle(poll_interval: float = 3.0, timeout: float = 300.0):
    """Wait until the server is not loading or generating."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = requests.get(f"{BASE}/status", timeout=10).json()
        if not s["loading"] and not s["generating"]:
            return s
        print(f"  Server busy — loading={s['loading']} generating={s['generating']}")
        time.sleep(poll_interval)
    raise TimeoutError("Server did not become idle within timeout")

# Preload zimage then generate
requests.post(f"{BASE}/engines/zimage/load", timeout=10)
wait_for_idle()

results = generate("zimage",
    prompt="a photorealistic forest in autumn",
    num_inference_steps=8,
)
```

> Wan/video cookbook entries removed with the engine (2026-09-07). All current
> engines return `type: "image"` with base64 — the download-video helper is no
> longer needed until a future video engine lands.

### Batch with automatic engine switching

```python
tasks = [
    ("flux2klein", dict(prompt="a cat on a rooftop", num_inference_steps=4)),
    ("qwenimage",  dict(prompt="a bilingual poster, CN+EN text")),
    ("ernie",      dict(prompt="abstract digital art, neon colours")),
]

for engine, params in tasks:
    print(f"Generating with {engine}: {params['prompt']}")
    results = generate(engine, **params)
    for r in results:
        img = base64.b64decode(r["base64"])
        out = f"{r['engine']}_{r['params']['seed']}.png"
        open(out, "wb").write(img)
        print(f"  → {out}")
```
