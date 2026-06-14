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
7. [GET /files/{subdir}/{filename}](#7-get-filessubdirfilename)
8. [GET /gallery](#8-get-gallery)
9. [DELETE /gallery/{id}](#9-delete-galleryid)
10. [Engine Parameters Reference](#10-engine-parameters-reference)
11. [Response Schemas](#11-response-schemas)
12. [Error Reference](#12-error-reference)
13. [curl Cookbook](#13-curl-cookbook)
14. [Python Cookbook](#14-python-cookbook)

---

## 1. Endpoints at a Glance

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/status` | Live engine status, VRAM, active engine |
| `POST` | `/generate/{engine}` | Run generation (blocks until done) |
| `GET` | `/engines` | Engine metadata (static, no state) |
| `POST` | `/engines/{engine}/load` | Preload engine into VRAM |
| `POST` | `/engines/unload` | Evict current engine from VRAM |
| `GET` | `/files/images/{filename}` | Download generated PNG |
| `GET` | `/files/videos/{filename}` | Download generated MP4 |
| `GET` | `/gallery` | List past generations |
| `DELETE` | `/gallery/{id}` | Delete a gallery entry + file |

---

## 2. GET /status

Returns live service state: all engine availability, which engine is loaded, VRAM usage.

### Response — 200

```json
{
  "engines": [
    {
      "key":         "flux2",
      "label":       "FLUX.2 [dev]",
      "description": "32B rectified flow transformer...",
      "output_type": "image",
      "vram_gb":     16.0,
      "available":   true,
      "loaded":      false,
      "error":       "",
      "params":      [ ... ]
    },
    {
      "key":       "sd35",
      "available": true,
      "loaded":    false,
      ...
    },
    {
      "key":       "flux2klein",
      "available": true,
      "loaded":    true,
      ...
    },
    {
      "key":       "wan",
      "available": true,
      "loaded":    false,
      ...
    }
  ],
  "active_engine": "flux2klein",
  "active_quant":  "",
  "generating":    false,
  "loading":       false,
  "vram": {
    "available":    true,
    "allocated_gb": 0.01,
    "reserved_gb":  0.04,
    "total_gb":     15.48,
    "free_gb":      15.43,
    "device_name":  "NVIDIA GeForce RTX 5060 Ti"
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
| `generating` | `true` while a generation is running — further `/generate` calls will queue |
| `loading` | `true` while a model is being loaded — takes 30–90 s |
| `vram.reserved_gb` | PyTorch reserved (includes loaded model + KV cache) |
| `vram.allocated_gb` | PyTorch actively allocated (subset of reserved) |

---

## 3. POST /generate/{engine}

Runs image or video generation. **Synchronous — the connection is held open until the result is ready.** Typical durations:

| Engine | Quant | Resolution | Steps | Expected time |
|---|---|---|---|---|
| `flux2klein` | BF16 | 1024×1024 | 4 | 10–30 s |
| `sd35` | Q4_0 | 1024×1024 | 28 | 60–90 s |
| `flux2` | Q3_K_M | 1024×1024 | 28 | 8–12 min |
| `wan` | Q4_K_M | 720p, 49 frames | — | 5–10 min |

### Content-Type

`multipart/form-data` — all fields are form fields (not JSON body).

### URL Parameter

| Parameter | Description |
|---|---|
| `engine` | `flux2` \| `flux2klein` \| `sd35` \| `wan` |

### Common Form Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Text description of the image or video to generate |
| `negative_prompt` | string | `""` | What NOT to include. Supported by `sd35` and `wan`. Ignored by `flux2` and `flux2klein`. |
| `width` | int | `1024` | Output width in pixels. Must be a multiple of 64. |
| `height` | int | `1024` | Output height in pixels. Must be a multiple of 64. |
| `num_inference_steps` | int | engine default | Denoising steps. More = better quality, slower. |
| `guidance_scale` | float | engine default | Prompt adherence strength. |
| `seed` | int | `-1` | `-1` = random. Fixed value = reproducible output. |
| `quant` | string | engine default | Quantization level. See [Engine Parameters Reference](#10-engine-parameters-reference). |
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
  "flux2": {
    "label":       "FLUX.2 [dev]",
    "description": "32B rectified flow transformer...",
    "output_type": "image",
    "vram_gb":     16.0,
    "hf_repo":     "diffusers/FLUX.2-dev-bnb-4bit",
    "params":      [ ... ]
  },
  "sd35":      { ... },
  "flux2klein": { ... },
  "wan":       { ... }
}
```

---

## 5. POST /engines/{engine}/load

Pre-loads an engine into VRAM without generating anything. Useful for warming up before the first request.

Returns immediately with `503` if the server is currently generating or loading.

### Response — 200

```json
{ "loaded": "sd35" }
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

## 7. GET /files/{subdir}/{filename}

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

## 8. GET /gallery

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
      "created_at": 1779640492.3
    }
  ],
  "limit":  50,
  "offset": 0
}
```

Entries are ordered newest-first. The gallery stores the last 500 entries on disk.

---

## 9. DELETE /gallery/{id}

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

## 10. Engine Parameters Reference

### `flux2` — FLUX.2 [dev]

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `prompt` | string | required | — | |
| `reference_image` | file | null | — | Enables image editing (I2I) mode |
| `width` | int | `1024` | 256–2048, step 64 | |
| `height` | int | `1024` | 256–2048, step 64 | |
| `num_inference_steps` | int | `28` | 1–50 | |
| `guidance_scale` | float | `3.5` | 1.0–20.0 | 3.5–4.0 typical |
| `seed` | int | `-1` | -1 to 2³¹-1 | |
| `quant` | string | `Q4_K_M` | see below | |

**`quant` options:**

| Value | Transformer size | Notes |
|---|---|---|
| `Q3_K_M` | ~16 GB | Smallest; good for machines with limited CPU RAM |
| `Q4_K_M` | ~20 GB | ✓ Recommended — best quality/size tradeoff |
| `Q5_K_M` | ~24 GB | Higher quality, needs ~38 GB CPU RAM total |
| `Q8_0` | ~35 GB | Near-lossless; needs ~50 GB CPU RAM total |
| `nvfp4` | ~8 GB | Blackwell native FP4 — run `nvfp4_save.py` first |

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

## 11. Response Schemas

### Generation Result Object

```typescript
{
  id:         string;        // UUID — use this for gallery DELETE
  engine:     string;        // "flux2" | "flux2klein" | "sd35" | "wan"
  filename:   string;        // e.g. "flux2klein_3f2a1b9c-....png"
  url:        string;        // Relative URL — prepend base URL to fetch
  base64:     string | null; // PNG as base64 string; null for videos
  type:       "image" | "video";
  width?:     number;        // Image width in pixels (images only)
  height?:    number;        // Image height in pixels (images only)
  fps?:       number;        // Frame rate (videos only)
  num_frames?: number;       // Frame count (videos only)
  params:     object;        // Echo of generation params (seed resolved)
  created_at: number;        // Unix timestamp (float)
}
```

### VRAM Object (inside /status)

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

---

## 12. Error Reference

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

NVFP4 transformer not found at /opt/arthur-img-models/nvfp4/flux2/transformer.
  → Run nvfp4_save.py on the VM first

SD 3.5 shared pipeline components not found at: .../quantized/sd35/shared
  → Run preq_save.py on the VM first

Server is busy
  → Poll /status until generating=false, then retry
```

---

## 13. curl Cookbook

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

### State-of-the-art quality — FLUX.2 full 32B

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2 \
  -F "prompt=a photorealistic mountain lake at dawn, misty, reflections" \
  -F "num_inference_steps=28" \
  -F "guidance_scale=3.5" \
  -F "quant=Q3_K_M" \
  -F "width=1024" \
  -F "height=1024"
```

### Image editing — FLUX.2 with reference image

```bash
curl -X POST http://192.168.0.87:8002/generate/flux2 \
  -F "prompt=same scene but at night, moonlight, stars" \
  -F "reference_image=@/path/to/input.png" \
  -F "quant=Q3_K_M"
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
curl -s http://192.168.0.87:8002/status | jq '.vram | {reserved_gb, free_gb, total_gb}'
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

## 14. Python Cookbook

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

results = generate_i2i("flux2",
    prompt="same scene but in winter with snow",
    image_path="input.png",
    quant="Q3_K_M",
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
