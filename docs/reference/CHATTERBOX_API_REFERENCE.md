# Chatterbox TTS — API Reference

> **Base URL:** `http://<host>:8001`
> **Auth:** None (internal lab service)
> **Content-Type:** `application/json`

---

## 1. `POST /synthesize/chatterbox`

Synthesize speech from text using Chatterbox TTS.

### Request Body

```json
{
  "text": "string (required)",
  "params": {
    "model": "persian",
    "audio_prompt_id": "",
    "seed": "0",
    "exaggeration": "0.65",
    "cfg_weight": "0.5",
    "repetition_penalty": "1.5",
    "use_g2p": "none",
    "max_length": "20000",
    "chunk_silence_ms": "350"
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | **Yes** | — | Persian or English text to synthesize. Any length — the engine auto-chunks at sentence boundaries. |
| `params` | object | No | `{}` | Synthesis parameters (see below). |

### `params` — Full Reference

#### Model Selection

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"persian"` | **`"persian"`** — Persian fine-tune 0.5B (30 layers, 2454 BPE tokens) from `hootan09/ChatterBox`. Default. Handles both Persian and English.<br>**`"default"`** — English-only 0.5B (16 layers, 704 BPE tokens). Faster startup, smaller VRAM.<br>**`"v3"`** — Multilingual v3 1.0B (30 layers, 2454 grapheme tokens) from `ResembleAI/chatterbox`. 23 languages. |

Switching `model` between calls triggers a full unload + reload (~10-15s). Same model across calls reuses the loaded instance. Persian is the default — no model param needed for Persian/English text.

#### Voice Cloning (Speaker Conditioning)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference voice ID from the Voice Library or upload. The model extracts a speaker embedding from this WAV file and clones the voice. **Leave empty to use the built-in voice.** Example: `"el_brian_deep_funny_and_cocky"` |

Reference WAVs should be **5-10 seconds** for best speaker embedding extraction. Voices can be uploaded via the UI or `POST /upload`.

#### Emotion / Expressiveness

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `exaggeration` | float | `0.65` | 0.0–1.0 | Emotion/expressiveness strength. Higher = more dramatic delivery. Maps to `emotion_adv` in the T3 conditionals. |
| `cfg_weight` | float | `0.5` | 0.0–∞ | Classifier-Free Guidance weight. `0.0` = no CFG (faster, less controlled). `0.5` = moderate CFG. Higher values trade naturalness for adherence. |

#### Generation Control

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `seed` | int/string | `"0"` | 0–2³¹ | **`0`** = random (non-deterministic; output varies per call).<br>**Any non-zero value** = deterministic seed. Sets `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, and enables `cudnn.deterministic=True`. Each chunk gets `seed + chunk_index`.<br>⚠️ Output sizes will be identical across runs; audio content is perceptually identical but not byte-identical (irreducible CUDA non-determinism). |
| `repetition_penalty` | float | `1.5` | 1.0–3.0 | Penalizes token repetition during autoregressive generation. Higher values reduce the chance of repeated tokens (which trigger early EOS via the alignment stream analyzer). Default increased from model's 1.2 to 1.5 for long-form reliability. |

#### Text Processing

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `use_g2p` | string | `"none"` | Grapheme-to-phoneme provider for Persian. **`"none"`** — raw text (default). Persian T3 handles raw text natively.<br>**`"persian_phonemizer"`** — adds combining vowel marks. May improve pronunciation but doubles token count (~1.5–2× expansion). |

#### Chunking (Long-Form Synthesis)

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `chunk_silence_ms` | float | `350` | 0–5000 | Silence gap inserted between chunks in milliseconds. `0` = no gap (chunks abut directly). |
| `max_length` | int/string | `"20000"` | 1–50000 | Maximum speech tokens per chunk. Patches over the model's internal hardcoded `max_new_tokens=1000`. Only relevant if a single chunk produces very long audio. |

**Chunking behavior:**
- Text is split at sentence boundaries (`.!?۔！？`) first
- Sentences longer than the chunk limit are split at commas (`،,؛;`)
- Any remaining chunk > 100 chars (Persian) or > 187 chars (English) is force-split by character count
- Persian chunk size: **80 chars** (pre-G2P)
- English chunk size: **150 chars** (pre-G2P)
- Failed chunks are logged and skipped; survivors are stitched with silence gaps
- Raises error only if ALL chunks fail

---

### Response

**Success (HTTP 200):**
```json
{
  "audio_b64": "<base64-encoded WAV>",
  "sample_rate": 24000,
  "synth_time_ms": 45231,
  "audio_dur_ms": 47900,
  "rtf": 0.944
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_b64` | string | Base64-encoded WAV audio (24kHz, 16-bit, mono). |
| `sample_rate` | int | Always `24000` (S3Gen vocoder output rate). |
| `synth_time_ms` | int | Wall-clock synthesis time in milliseconds. |
| `audio_dur_ms` | int | Total audio duration in milliseconds (including silence gaps). |
| `rtf` | float | Real-Time Factor (`synth_time / audio_dur`). < 1.0 = faster than real-time. |

**Error (HTTP 400):**
```json
{"error": "Unknown engine: chatterbox"}
{"error": "Empty text"}
```

**Error (HTTP 408 — Timeout):**
```json
{"error": "Synthesis timeout after 300s -- 'chatterbox' requires a GPU."}
```

**Error (HTTP 500 — Synthesis Failure):**
```json
{
  "error": "All 3/3 chunks failed.\nFirst error:\n...",
  "trace": "..."
}
```

The `error` field includes the first chunk failure's full traceback. The `trace` field is a compact 4-frame summary.

---

## 2. `GET /status`

Returns engine availability, loaded model, and resource usage.

### Response
```json
{
  "models": {
    "chatterbox": {
      "label": "Chatterbox",
      "size": "3.0 GB",
      "rtf_est": "RTF 1.67 (GPU)",
      "ram_est_mb": 1800,
      "heavy": true,
      "notes": "Exaggeration slider + voice cloning.",
      "available": true,
      "reason": "",
      "status": "loaded",
      "load_time_s": 12.3,
      "error": null,
      "loaded_model": "persian"
    }
  },
  "system": {"total": 32000, "used": 8500, "free": 23500},
  "gpu": {
    "name": "NVIDIA GeForce RTX 3090",
    "vram_total": 24576,
    "vram_used": 4200,
    "vram_free": 20376
  },
  "device": "cuda"
}
```

Key fields for Chatterbox:
- `models.chatterbox.available` — `true` if the model can be loaded (dependencies present, not mid-sweep)
- `models.chatterbox.status` — `"loaded"` | `"loading"` | `"unloaded"` | `"error"`
- `models.chatterbox.loaded_model` — which variant is loaded: `"default"` | `"persian"` | `"v3"` | `null`
- `models.chatterbox.error` — error message if status is `"error"`

---

## 3. `GET /voices/chatterbox`

> **Note:** Chatterbox does not expose a predefined voice list via this endpoint (returns empty `[]`). Voice selection is done via `audio_prompt_id` in the synthesis params, referencing uploaded WAV files or Voice Library entries.

---

## 4. `GET /refs`

List available reference WAVs for the `audio_prompt_id` dropdown.

### Response
```json
{
  "refs": [
    {"id": "el_liam_energetic", "name": "Liam — Energetic Social Media Creator"},
    {"id": "el_sarah_calm", "name": "Sarah — Calm Narration"}
  ]
}
```

These IDs come from files in `UPLOAD_DIR` (`/tmp/tts_uploads/`) and Voice Library entries.

---

## 5. `GET /voice-library`

List all voices in the Voice Library with metadata.

### Response
```json
{
  "voices": [
    {
      "id": "el_liam_energetic_social_media_creator",
      "name": "Liam — Energetic Social Media Creator",
      "duration_s": 5.2,
      "sample_rate": 24000,
      "source": "elevenlabs",
      "created": "2026-06-14T12:00:00"
    }
  ]
}
```

---

## 6. `GET /voice-library/{voice_id}/audio`

Stream the WAV audio for a voice library entry. Returns `audio/wav` binary.

Use this for the ▶ play button preview in the UI.

---

## 7. `GET /logs?since={seq}`

Poll for server-side log entries (streaming log view).

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `since` | int | `0` | Return log entries with `seq > since`. |

### Response
```json
{
  "entries": [
    {"seq": 1, "ts": "2026-06-17T10:00:00", "tag": "SEED", "engine": "chatterbox", "msg": "Seed set to 1234"},
    {"seq": 2, "ts": "2026-06-17T10:00:01", "tag": "CHUNK", "engine": "chatterbox", "msg": "Splitting 264 chars → 5 chunks (avg 60 chars, 350ms gap)"},
    {"seq": 3, "ts": "2026-06-17T10:00:05", "tag": "CHUNK", "engine": "chatterbox", "msg": "  [1/5] 8.2s gen → 9.5s audio  RTF 0.9×  VRAM 4.20GB  elapsed 8s  ETA 33s"}
  ],
  "seq": 3
}
```

**Chatterbox log tags:** `SEED`, `CHUNK`, `LOAD`, `SYNTH`, `RESULT`, `PARAMS`.

---

## 8. `POST /models/chatterbox/load`

Pre-load the Chatterbox model (avoids cold-start latency on first synthesis).

### Request Body (optional)
```json
{"params": {"model": "persian"}}
```

### Response
```json
{"loaded": "chatterbox", "model": "persian"}
```

---

## 9. `DELETE /models/chatterbox`

Unload the Chatterbox model from VRAM.

### Response
```json
{"unloaded": "chatterbox"}
```

---

## Request Flow

```
POST /synthesize/chatterbox
  │
  ├─ 1. Validate engine + text
  ├─ 2. _ensure_loaded("chatterbox", params)
  │     ├─ Load if unloaded
  │     ├─ Reload if model variant changed (default ↔ persian ↔ v3)
  │     └─ Skip if already loaded with same variant
  ├─ 3. _synth_chatterbox(inst, text, params)
  │     ├─ Set seed + CUDA determinism (if seed ≠ 0)
  │     ├─ Apply persian_phonemizer G2P (if use_g2p ≠ "none")
  │     ├─ Decompose Persian chars (if persian model)
  │     ├─ Split text → chunks (sentence → comma → force-split)
  │     ├─ For each chunk:
  │     │   ├─ Set per-chunk seed (seed + i)
  │     │   ├─ Tokenize, truncate if > 2048 tokens
  │     │   ├─ Patch max_new_tokens (1000 → max_length)
  │     │   ├─ inst.generate(chunk_text, **kw)
  │     │   └─ Restore inference patch + CUDA flags
  │     ├─ Stitch audio arrays with silence gaps
  │     └─ Encode to WAV bytes
  └─ 4. Return {audio_b64, sample_rate, synth_time_ms, audio_dur_ms, rtf}
```

---

## Python: Minimal API Call

```python
import requests, base64

resp = requests.post(
    "http://192.168.0.87:8001/synthesize/chatterbox",
    json={
        "text": "در صورت امضای توافق صلح، بیت کوین پتانسیل تست سطوح بالای ۶۶,۰۰۰ دلار را دارد.",
        "params": {
            "model": "persian",
            "seed": "1234",
            "exaggeration": "0.65",
        },
    },
    timeout=300,
)

data = resp.json()
if "error" in data:
    print(f"Error: {data['error']}")
else:
    wav_bytes = base64.b64decode(data["audio_b64"])
    with open("output.wav", "wb") as f:
        f.write(wav_bytes)
    print(f"Done: {data['audio_dur_ms']}ms, RTF {data['rtf']}")
```

---

## cURL: Minimal Example

```bash
curl -X POST http://192.168.0.87:8001/synthesize/chatterbox \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello, this is a test of Chatterbox TTS.",
    "params": {"model": "default", "seed": "42"}
  }' \
  -o output.wav
```

---

## Performance Notes

| Metric | Value |
|--------|-------|
| Sample rate | 24000 Hz (S3Gen vocoder) |
| Bit depth | 16-bit PCM |
| Channels | Mono |
| RTF (GPU) | ~0.9–1.7× (slower than real-time) |
| GPU utilization | 15–20% (memory-bandwidth-bound, normal) |
| VRAM (0.5B models) | ~2–3 GB |
| VRAM (1.0B v3 model) | ~4–5 GB |
| Timeout | 300s (DEFAULT_SYNTH_TIMEOUT) |
| Cold-start load time | ~10–15s (downloads ~2 GB on first run) |
| Warm start | ~0s (reuses loaded instance) |

### Chunk Size Tuning

| Model | Chunk Size | Force-Split At | Notes |
|-------|-----------|----------------|-------|
| Persian (fa) | 80 chars | 100 chars | Matches 5–8s training distribution after G2P expansion |
| English (default) | 150 chars | 187 chars | Longer sequences tolerated by English-only model |
| v3 (multilingual) | 150 chars | 187 chars | v3 has better long-form handling |

To customize chunk size, modify `_CHUNK_CHARS` in `tts_lab_engines.py:_synth_chatterbox()`.

---

*Last updated: 2026-06-17*
