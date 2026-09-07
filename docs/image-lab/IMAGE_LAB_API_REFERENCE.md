# Arthur Image Lab — API Reference

**Base URL:** `http://192.168.0.87:8002`  
**Protocol:** HTTP/1.1 — all generation requests are **synchronous** (connection held open until complete)  
**Auth:** None (local network only)

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
      "vram_gb":     10.0,
      "available":   true,
      "loaded":      true,
      "error":       "",
      "params":      [ ... ]
    },
    { "key": "sd35", "label": "SD 3.5 Large", "available": true, "loaded": false, "error": "" },
    { "key": "wan",  "label": "Wan2.2",       "available": true, "loaded": false, "error": "" }
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

Runs image or video generation. **Synchronous — the connection is held open until the result is ready.** Typical durations:

| Engine | Quant | Resolution | Steps | Expected time |
|---|---|---|---|---|
| `flux2klein` | BF16/NF4 | 1024×1024 | 4 | 50 s |
| `flux2klein9b` | Q4_K_M | 1024×1024 | 4 | 40 s |
| `sd35` | Q4_0 | 1024×1024 | 28 | 60–90 s |
| `wan` | Q4_K_M | 720p, 49 frames | — | 5–10 min |
| `ideogram4` | NF4 (API) | 1024×1024 | 20 | 30–60 s |

> `flux2` (FLUX.2 [dev] 32B) was REMOVED 2026-08-13 — see
> `ARTHUR_IMAGE_LAB_REFERENCE.md` §4.1 for the full history.

### Content-Type

`multipart/form-data` — all fields are form fields (not JSON body).

### URL Parameter

| Parameter | Description |
|---|---|
| `engine` | `flux2klein` \| `flux2klein9b` \| `sd35` \| `wan` \| `ideogram4` |

### Common Form Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Text description of the image or video to generate |
| `negative_prompt` | string | `""` | What NOT to include. Supported by `sd35` and `wan`. Ignored by the FLUX.2 Klein engines. |
| `width` | int | `1024` | Output width in pixels. Must be a multiple of 64. |
| `height` | int | `1024` | Output height in pixels. Must be a multiple of 64. |
| `num_inference_steps` | int | engine default | Denoising steps. More = better quality, slower. |
| `guidance_scale` | float | engine default | Prompt adherence strength. |
| `seed` | int | `-1` | `-1` = random. Fixed value = reproducible output. |
| `quant` | string | engine default | Quantization level. See [Engine Parameters Reference](#13-engine-parameters-reference). |
| `reference_image` | file | `null` | Optional image upload for I2I (FLUX.2) or I2V first frame (Wan). |

### Engine-Specific Fields

| Field | Type | Default | Engines | Description |
|---|---|---|---|---|
| `num_images` | int | `1` | `sd35` only | How many images to generate per request (1–4). |
| `mode` | string | `t2v` | `wan` only | `t2v` = text-to-video \| `i2v` = image-to-video |
| `num_frames` | int | `49` | `wan` only | Number of video frames. At 16 fps, 49 frames ≈ 3 s. |
| `fps` | int | `16` | `wan` only | Output video frame rate (8–24). |
| `resolution` | string | `720p` | `wan` only | `480p` (854×480) or `720p` (1280×720). |

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
- `results` is always an array. Most engines return 1 item; `sd35` returns up to 4.
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
  "sd35":      { ... },
  "wan":       { ... }
}
```

---

## 5. POST /engines/{engine}/load

Pre-loads an engine into VRAM without generating anything. Useful for warming up before the first request.

**Non-blocking:** the load runs in a worker thread — the HTTP call stays open for the duration (30–90 s), but `/status` and `/logs` keep responding and report `"loading": true` so clients can poll rather than hang. Requests are single-flight: a second preload while one is running (or during generation) returns `503`.

### Form Field

| Field | Type | Default | Description |
|---|---|---|---|
| `quant` | string | `""` | Quantization level to load (e.g. `"Q3_K_M"` for `wan`). Empty = engine default. Changing quant on an already-loaded engine reloads it. |

### Response — 200

```json
{ "loaded": "sd35", "quant": "Q4_0" }
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
{ "evicted": false, "engine": "sd35", "note": "not resident" }
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
| `engine` | string | *(all)* | Filter by engine key (e.g. `?engine=sd35`) |

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

### `sd35` — Stable Diffusion 3.5 Large

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `negative_prompt` | string | `""` | — | Supported |
| `width` | int | `1024` | 256–1536, step 64 | |
| `height` | int | `1024` | 256–1536, step 64 | |
| `num_inference_steps` | int | `28` | 1–100 | |
| `guidance_scale` | float | `4.5` | 1.0–20.0 | |
| `num_images` | int | `1` | 1–4 | Images per request |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q4_0` | see below | |

**`quant` options:**

| Value | Transformer size | Notes |
|---|---|---|
| `Q4_0` | ~4.8 GB | ✓ Recommended |
| `Q5_0` | ~5.8 GB | Higher quality |
| `Q8_0` | ~8.8 GB | Near-lossless |
| `nvfp4` | ~2 GB | Blackwell native — run `nvfp4_save.py` first |

---

### `wan` — Wan2.2 (Text-to-Video / Image-to-Video)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | Describe motion and scene |
| `negative_prompt` | string | `"low quality, blurry, distorted"` | — | |
| `mode` | string | `t2v` | `t2v` \| `i2v` | |
| `reference_image` | file | null | — | Required when `mode=i2v` |
| `num_frames` | int | `49` | 16–120, step 8 | 49 ≈ 3 s at 16 fps |
| `fps` | int | `16` | 8–24 | Output video frame rate |
| `resolution` | string | `720p` | `480p` \| `720p` | 480p = 854×480 \| 720p = 1280×720 |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q4_K_M` | see below | Applies to both HighNoise + LowNoise transformers |

**`quant` options:**

| Value | Per-transformer size | Total (×2) | Notes |
|---|---|---|---|
| `Q3_K_M` | ~7.2 GB | ~14.4 GB | Smallest |
| `Q4_K_M` | ~9.7 GB | ~19.4 GB | ✓ Recommended |
| `Q5_K_M` | ~10.8 GB | ~21.6 GB | Higher quality |
| `Q8_0` | ~15.4 GB | ~30.8 GB | Near-lossless |
| `nvfp4` | ~4 GB | ~8 GB | Blackwell native — run `nvfp4_save.py` first |

---

## 14. Response Schemas

### Generation Result Object

```typescript
{
  id:         string;        // UUID — use this for gallery DELETE
  engine:     string;        // "flux2klein" | "flux2klein9b" | "sd35" | "wan" | "ideogram4"
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
  → Switch to a smaller quant (e.g. Q3_K_M instead of Q4_K_M)

NVFP4 transformer not found at /opt/arthur-img-models/nvfp4/sd35/transformer.
  → Run nvfp4_save.py on the VM first

SD 3.5 shared pipeline components not found at: .../quantized/sd35/shared
  → Run preq_save.py on the VM first

Server is busy
  → Poll /status until generating=false, then retry
```

---

## 16. curl Cookbook

### Quick image — fastest (FLUX.2 Klein, 4 steps)

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2klein \
  -F "prompt=a golden retriever in a meadow at sunset" \
  -F "num_inference_steps=4" \
  -F "seed=42"
```

### High quality image — SD 3.5 with negative prompt

```bash
curl -X POST http://192.168.0.87:8002/generate/sd35 \
  -F "prompt=a golden retriever in a meadow at sunset, golden hour, bokeh" \
  -F "negative_prompt=blurry, low quality, watermark" \
  -F "num_inference_steps=28" \
  -F "guidance_scale=4.5" \
  -F "quant=Q4_0"
```

### Multiple images — SD 3.5 (up to 4)

```bash
curl -X POST http://192.168.0.87:8002/generate/sd35 \
  -F "prompt=portrait of a mountain climber" \
  -F "num_images=4" \
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

### Text-to-video — Wan2.2

```bash
curl -X POST http://192.168.0.87:8002/generate/wan \
  -F "prompt=a red fox running through a snowy forest, cinematic, slow motion" \
  -F "negative_prompt=low quality, blurry" \
  -F "mode=t2v" \
  -F "num_frames=49" \
  -F "resolution=720p" \
  -F "quant=Q4_K_M"
```

### Image-to-video — Wan2.2

```bash
curl -X POST http://192.168.0.87:8002/generate/wan \
  -F "prompt=the fox slowly turns its head and looks at the camera" \
  -F "mode=i2v" \
  -F "reference_image=@/path/to/fox.png" \
  -F "num_frames=49" \
  -F "resolution=480p"
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
# Preload sd35 into VRAM (~35 s load time)
curl -X POST http://192.168.0.87:8002/engines/sd35/load

# Poll until loading=false
watch -n 2 'curl -s http://192.168.0.87:8002/status | jq "{loading,active_engine}"'

# Now generate instantly (model already loaded)
curl -X POST http://192.168.0.87:8002/generate/sd35 \
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
curl -s -X POST http://192.168.0.87:8002/engines/wan/load \
  -F "quant=Q3_K_M"     # async — /status reports loading:true while it runs
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

# Preload sd35 then generate
requests.post(f"{BASE}/engines/sd35/load", timeout=10)
wait_for_idle()

results = generate("sd35",
    prompt="a photorealistic forest in autumn",
    num_inference_steps=28,
    quant="Q4_0",
)
```

### Download video (no base64 for videos)

```python
def save_video_result(result: dict, output_path: str):
    url = f"{BASE}{result['url']}"          # e.g. /files/videos/wan_uuid.mp4
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"Saved {output_path} ({result['num_frames']} frames @ {result['fps']} fps)")

results = generate("wan",
    prompt="ocean waves crashing on a beach, cinematic",
    mode="t2v",
    num_frames=49,
    resolution="720p",
    quant="Q4_K_M",
)
save_video_result(results[0], "ocean.mp4")
```

### Batch with automatic engine switching

```python
tasks = [
    ("flux2klein", dict(prompt="a cat on a rooftop", num_inference_steps=4)),
    ("flux2klein", dict(prompt="a dog in a park",    num_inference_steps=4)),
    ("sd35",       dict(prompt="abstract digital art, neon colours", quant="Q4_0")),
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
