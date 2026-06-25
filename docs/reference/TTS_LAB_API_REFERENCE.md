# Arthur TTS Lab — Master API Reference

> **Base URL:** `http://192.168.0.87:8001`
> **Auth:** None (internal lab service)
> **Content-Type:** `application/json`
> **Response:** WAV audio as base64-encoded string in JSON envelope

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Endpoints](#core-endpoints)
3. [Engine Catalog](#engine-catalog)
4. [Voice Cloning](#voice-cloning)
5. [Container Architecture](#container-architecture)
6. [Code Examples](#code-examples)

---

## Quick Start

```bash
# Minimal synthesis — Piper (fast, CPU-only, English)
curl -X POST http://192.168.0.87:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}'

# Get engine status and availability
curl http://192.168.0.87:8001/status

# Persian TTS with Chatterbox (default Persian model)
curl -X POST http://192.168.0.87:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"سلام، خوبی؟"}'

# Voice cloning with Qwen3-TTS (with transcript — best quality)
curl -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a cloned voice.","params":{"audio_prompt_id":"abc12345","ref_text":"exact words in the reference clip","language":"english"}}'

# Voice cloning without transcript (x-vector only — works with any clip)
curl -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a cloned voice.","params":{"audio_prompt_id":"abc12345","language":"english"}}'
```

---

## Core Endpoints

### `GET /status`

Full system status — all engines, RAM, VRAM, GPU info.

**Response:**
```json
{
  "system": { "total": 32768, "used": 12450 },
  "gpu": {
    "name": "NVIDIA RTX 5060 Ti",
    "vram_total": 16384,
    "vram_used": 3200,
    "vram_free": 13184
  },
  "models": {
    "piper":    { "available": true, "loaded": false, "status": "unloaded" },
    "kokoro":   { "available": true, "loaded": false, "status": "unloaded" },
    "qwen3tts": { "available": true, "loaded": false, "status": "unloaded" }
  }
}
```

Engine status values: `"unloaded"` | `"loading"` | `"loaded"` | `"error"` | `"evicted"`

---

### `POST /synthesize/{engine}`

**Primary endpoint.** Synthesize text to speech. Returns JSON with base64-encoded WAV.

**Request Body:**
```json
{
  "text": "string (required — any length, auto-chunked per engine)",
  "params": {
    "voice": "aiden",
    "language": "english",
    "speed": "1.0"
  }
}
```

**Success Response (200):**
```json
{
  "audio_b64": "UklGRiRAAABXQVZFZm10... (base64 WAV)",
  "sample_rate": 24000,
  "synth_time_ms": 342,
  "audio_dur_ms": 2150,
  "rtf": 0.159,
  "load_time_s": 2.4
}
```

**Error Responses:**
- `400` — Unknown engine or empty text
- `408` — Synthesis timeout (see `SYNTH_TIMEOUT` per engine)
- `500` — Engine load/synth error (includes `trace` field)
- `503` — Engine not available (missing dependencies, failed probe)

**Decoding the audio:**
```python
import base64
wav_bytes = base64.b64decode(response["audio_b64"])
with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

---

### `POST /upload`

Upload a reference WAV file for voice cloning. Returns an ID that you pass as `audio_prompt_id` in synthesis params.

**Request:** `multipart/form-data` with field `file` (WAV file)

**Response:**
```json
{
  "id": "a1b2c3d4",
  "filename": "my_voice.wav",
  "size": 48044
}
```

**⚠️ Container mode limitation:** Reference audio is uploaded to the orchestrator but file paths are not synced to engine containers. Voice cloning with `audio_prompt_id` only works in bare-metal (local) mode. See [Voice Cloning](#voice-cloning) for workarounds.

---

### `GET /refs`

List all available reference WAV files (pre-shipped + uploaded).

**Response:**
```json
{
  "refs": [
    { "id": "arthur_ref", "name": "arthur_ref.wav", "size": 144012 },
    { "id": "a1b2c3d4",  "name": "my_voice.wav  (uploaded)", "size": 48044 }
  ]
}
```

---

### `GET /voices/{engine}`

Get available voice/speaker names for an engine.

```bash
curl http://192.168.0.87:8001/voices/kokoro
# → {"voices": ["af_heart","af_bella","bm_lewis","am_adam",...]}
```

---

### `POST /models/{engine}/load`

Pre-load an engine into memory (warmup).

```json
{
  "params": { "voice": "khadijah" }
}
```

**Response:** `{"status": "loaded", "model": "matcha", "load_time_s": 1.2}`

---

### `DELETE /models/{engine}`

Unload an engine from memory / VRAM.

**Response:** `{"unloaded": "chatterbox"}`

---

### `POST /refresh`

Re-probe all engine availability without restarting the server.

**Response:** `{"refreshed": true, "models": [...]}`

---

### `GET /logs`

Last 200 server-side log entries (ring buffer).

---

## Engine Catalog

### Engine Status Legend

| Status | Meaning |
|--------|---------|
| ✅ **supported** | Production-ready, deployed and tested |
| 🧪 **experimental** | Deployed but not fully validated |
| ⚠️ **blocked** | Dependency/config issue — unavailable |
| 🔧 **not built** | Code exists, not deployed |

---

### Lightweight Engines (CPU-friendly, no GPU needed)

#### `piper` — Piper TTS ✅

| | |
|---|---|
| **Size** | 61–116 MB per voice (ONNX) |
| **RTF** | ~0.43× ⚡ (real-time) |
| **Voices** | 6 (en_US-ryan-high, en_US-lessac-medium, en_GB-alan-medium, etc.) |
| **Languages** | English (US/GB) |
| **Voice Cloning** | ❌ |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"en_US-ryan-high"` | Voice model name (must match installed `.onnx` file) |

---

#### `kokoro` — Kokoro-82M ✅

| | |
|---|---|
| **Size** | 89 MB (ONNX) |
| **RTF** | ~3.20× |
| **Voices** | 54 (9 languages) |
| **Languages** | en, ja, zh, fr, ko, it, es, pt, hi |
| **Voice Cloning** | ❌ |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"af_heart"` | One of 54 speaker names (e.g., `bm_lewis`, `af_bella`, `am_adam`) |
| `language` | string | `"en-us"` | Language code |
| `speed` | float | `1.0` | 0.5–2.0 |

---

#### `melo` — MeloTTS ✅

| | |
|---|---|
| **Size** | 200 MB |
| **RTF** | ~0.46× ⚡ |
| **Voices** | 5 English accents |
| **Languages** | EN-US, EN-BR, EN-AU, EN-INDIA, EN-Default |
| **Voice Cloning** | ❌ |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"EN-Default"` | `EN-Default`, `EN-US`, `EN-BR`, `EN-AU`, `EN_INDIA` |
| `speed` | float | `1.0` | 0.5–2.0 |

---

#### `matcha` — Matcha-TTS (FA/EN) ✅

| | |
|---|---|
| **Size** | 74 MB per voice (ONNX) |
| **RTF** | ~0.24× ⚡⚡ |
| **Voices** | 2: Khadijah (Persian F) + Musa (Persian M) |
| **Languages** | Persian, English |
| **Voice Cloning** | ❌ |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"khadijah"` | `khadijah` (F) or `musa` (M) |
| `speed` | float | `1.0` | 0.5–2.0 |
| `temperature` | float | `0.333` | Noise scale — higher = more variation |

---

#### `mmsfas` — MMS Persian (Meta) ✅

| | |
|---|---|
| **Size** | ~150 MB |
| **RTF** | ~0.5× ⚡ |
| **Languages** | Persian/Farsi only |
| **Voice Cloning** | ❌ |
| **Container** | engine-current |
| **License** | CC-BY-NC 4.0 (Meta) |

**Params:** None (VITS single-speaker model).

---

### Heavy Engines (GPU recommended/required)

#### `chatterbox` — Chatterbox (Persian T3) ✅

| | |
|---|---|
| **Size** | ~3.0 GB |
| **RTF** | ~2.42× |
| **VRAM** | ~1.8 GB |
| **Voices** | Voice cloning via reference WAV |
| **Languages** | Persian (T3 fine-tune), English, 23 languages (v3) |
| **Voice Cloning** | ✅ Zero-shot (reference WAV) |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"persian"` | `"persian"` (T3 fine-tune), `"default"` (English 0.5B), `"v3"` (multilingual 1.0B) |
| `audio_prompt_id` | string | `""` | Uploaded reference WAV ID for voice cloning |
| `exaggeration` | float | `0.65` | Emotion exaggeration (0.0–1.0) |
| `cfg_weight` | float | `0.5` | Classifier-free guidance weight |
| `temperature` | float | — | Sampling temperature (optional) |
| `top_p` | float | — | Nucleus sampling (optional) |
| `top_k` | int | — | Top-k sampling (optional) |
| `repetition_penalty` | float | `1.5` | Higher = less repetition |
| `min_p` | float | — | Minimum probability threshold |
| `seed` | string | `"0"` | Random seed (`"0"` = random) |
| `use_g2p` | string | `"none"` | Persian G2P: `"persian_phonemizer"`, `"hazm"`, `"parsivar"`, `"none"` |
| `max_length` | string | `"20000"` | Max tokens per chunk |
| `chunk_silence_ms` | string | `"350"` | Silence between chunks |

---

#### `chatterboxturbo` — Chatterbox Turbo ✅

| | |
|---|---|
| **Size** | ~700 MB |
| **RTF** | ~1.11× (near real-time) |
| **VRAM** | ~1.5 GB |
| **Languages** | 23 languages |
| **Voice Cloning** | ✅ Zero-shot (reference WAV) |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference WAV ID for voice cloning |
| `exaggeration` | float | `0.5` | Emotion exaggeration |
| `cfg_weight` | float | `0.5` | CFG weight |
| `temperature` | float | — | Sampling temperature |
| `top_p` | float | — | Nucleus sampling |
| `top_k` | int | — | Top-k sampling |
| `repetition_penalty` | float | — | Repetition penalty |
| `min_p` | float | — | Min probability |
| `norm_loudness` | bool | `true` | Normalize output loudness |
| `seed` | string | `"0"` | Random seed |
| `use_g2p` | string | `"none"` | Persian G2P provider |

---

#### `chattts` — ChatTTS ✅

| | |
|---|---|
| **Size** | 1.2–2.3 GB |
| **RTF** | ~2.14× |
| **VRAM** | ~1.8 GB |
| **Languages** | Chinese, English |
| **Voice Cloning** | ✅ Reference WAV (known bug: library falls back to random speaker) |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference WAV ID |
| `voice_characteristics` | string | — | Voice description text (e.g., "deep warm elderly male") |
| `transcript` | string | — | Transcript of the reference WAV |
| `speed` | float | `1.0` | Speed multiplier |
| `temperature` | float | `0.3` | Sampling temperature |
| `top_p` | float | `0.7` | Nucleus sampling |
| `top_k` | int | `20` | Top-k sampling |
| `repetition_penalty` | float | `2.0` | Repetition penalty |
| `max_new_tokens` | int | `2048` | Max tokens |

---

#### `f5tts` — F5-TTS ✅

| | |
|---|---|
| **Size** | ~1.2 GB |
| **RTF** | ~5.45× |
| **VRAM** | ~2.0 GB |
| **Languages** | English (Arabic-script languages via character mapping) |
| **Voice Cloning** | ✅ **REQUIRED** — needs 5–15s reference WAV |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | — | Reference WAV ID (**required** — falls back to first available if missing) |
| `ref_text` | string | `""` | Transcript of reference audio (improves quality) |
| `speed` | float | `1.0` | Speed multiplier |
| `nfe_step` | int | `32` | Number of function evaluations (higher = better quality, slower) |

---

#### `styletts2` — StyleTTS 2 ✅

| | |
|---|---|
| **Size** | ~0.7 GB |
| **RTF** | ~0.22× ⚡⚡ |
| **VRAM** | ~1.5 GB |
| **Languages** | English |
| **Voice Cloning** | ✅ Style transfer from reference WAV |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference WAV ID for style transfer |
| `alpha` | float | `0.3` | Style mixing (0.0 = base voice, 1.0 = full style) |
| `beta` | float | `0.7` | Prosody mixing |
| `diffusion_steps` | int | `5` | Diffusion steps (higher = better quality) |
| `embedding_scale` | float | `1.0` | Speaker embedding strength |

---

#### `fishspeech` — Fish Speech ✅

| | |
|---|---|
| **Size** | ~1.1 GB |
| **RTF** | ~3.48× |
| **VRAM** | ~1.5 GB |
| **Languages** | Multilingual (Persian via LM tokenizer) |
| **Voice Cloning** | ✅ Zero-shot |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `max_new_tokens` | int | `256` | Max tokens |
| `top_p` | float | `0.7` | Nucleus sampling |
| `repetition_penalty` | float | `1.5` | Repetition penalty |
| `temperature` | float | `0.7` | Sampling temperature |

---

#### `bark` — Bark ✅

| | |
|---|---|
| **Size** | ~2.5 GB |
| **RTF** | ~5.92× |
| **VRAM** | ~12 GB (heavy!) |
| **Languages** | Multilingual |
| **Voice Cloning** | ❌ (uses text prompt style only) |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"v2/en_speaker_6"` | Speaker preset |
| `voice_preset` | string | — | Style preset (overrides voice) |

**Emotion tokens** (embed in text): `[laughs]`, `[sighs]`, `[clears throat]`, `[long pause]`, `[hesitantly]`, `[nervously]`

---

#### `dia` — Dia-1.6B ✅

| | |
|---|---|
| **Size** | ~3 GB |
| **RTF** | ~7.20× |
| **VRAM** | ~3.0 GB |
| **Languages** | English |
| **Voice Cloning** | ✅ Reference WAV for speaker cloning |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference WAV for speaker cloning |
| `voice` | string | — | Speaker tag (e.g., `"[S1]"`, `"[S2]"`) |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling |

**Multi-speaker tags** (embed in text): `[S1]`, `[S2]`

---

#### `outetts` — OuteTTS 1.0 ✅

| | |
|---|---|
| **Size** | 384 MB (Q4_K_M GGUF) |
| **RTF** | ~15–26× 🐌 |
| **VRAM** | ~800 MB |
| **Languages** | English |
| **Voice Cloning** | ✅ Reference WAV creates speaker |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | string | `/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf` | Path to GGUF file |
| `voice` | string | `"default"` | Speaker name |
| `audio_prompt_id` | string | `""` | Reference WAV to create a speaker |
| `temperature` | float | `0.1` | Sampling temperature |
| `repetition_penalty` | float | `1.1` | Repetition penalty |
| `max_length` | int | `4096` | Max tokens (auto-capped for speed) |

---

#### `zonos` — Zonos v0.1 ✅

| | |
|---|---|
| **Size** | ~1.2 GB |
| **RTF** | ~4.29× |
| **VRAM** | ~2.5 GB |
| **SR** | 44 kHz |
| **Languages** | en-us |
| **Voice Cloning** | ✅ Speaker embedding from reference WAV |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `variant` | string | `"transformer"` | `"transformer"` or `"hybrid"` |
| `audio_prompt_id` | string | `""` | Reference WAV for speaker embedding |
| `language` | string | `"en-us"` | Language code |
| `speaking_rate` | float | `13.0` | Speaking rate (1–30) |
| `cfg_scale` | float | `2.0` | CFG scale |
| `max_new_tokens` | int | `1024` | Max tokens |
| `happiness` | float | `0.3` | Emotion vector (0–1) |
| `sadness` | float | `0.05` | Emotion vector |
| `disgust` | float | `0.05` | Emotion vector |
| `fear` | float | `0.05` | Emotion vector |
| `surprise` | float | `0.1` | Emotion vector |
| `anger` | float | `0.05` | Emotion vector |
| `other` | float | `0.2` | Emotion vector |
| `neutral` | float | `0.2` | Emotion vector |

---

#### `omnivoice` — OmniVoice ✅

| | |
|---|---|
| **Size** | ~1.2 GB (BF16) |
| **RTF** | ~0.67× ⚡ |
| **VRAM** | ~2.0 GB |
| **Languages** | **600+ languages** |
| **Voice Cloning** | ✅ Reference WAV or text-only voice description |
| **Container** | engine-current |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `language` | string | `""` | Language code (e.g., `"fa"`, `"en"`, `"zh"`) — 600+ supported |
| `audio_prompt_id` | string | `""` | Reference WAV for voice cloning |
| `ref_text` | string | `""` | Transcript of reference, OR text-only voice clone prompt (e.g., "deep warm male voice") |
| `instruct` | string | `""` | Style instruction (e.g., "speak slowly and clearly") |
| `speed` | float | — | Speed multiplier (optional) |
| `duration` | float | — | Target duration in seconds (optional hint) |

---

#### `orpheus` — Orpheus 3B 🧪

| | |
|---|---|
| **Size** | ~3 GB |
| **VRAM** | ~3.0 GB |
| **Languages** | English |
| **Voice Cloning** | ❌ |
| **Container** | engine-current (needs vllm) |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `voice` | string | `"tara"` | Speaker name |

**Emotion tags** (embed in text): `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

---

#### `qwen3tts` — Qwen3-TTS 1.7B Base ✅

| | |
|---|---|
| **Size** | ~3 GB |
| **VRAM** | ~6 GB (1.7B Base) |
| **Languages** | 10 (en, zh, ja, ko, fr, de, es, pt, more) |
| **Voice Cloning** | ✅ **Primary feature** — 3s reference WAV, ICL mode (ref_audio + ref_text) or x-vector only |
| **Built-in Speakers** | ❌ (Base model — voice clone only, no preset speakers) |
| **Voice Design** | ❌ (Base model — no `instruct` parameter; use CustomVoice variant for that) |
| **Container** | engine-qwen (mid stack) |

**Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_prompt_id` | string | `""` | Reference WAV ID for voice cloning — **upload a WAV to enable voice clone** |
| `ref_text` | string | `""` | Transcript of reference audio. **Optional** — without it, uses x-vector only mode (speaker embedding from audio alone). Providing transcript enables ICL mode for best quality. |
| `language` | string | `"english"` | `english`, `chinese`, `japanese`, `korean`, `french`, `german`, `spanish`, `portuguese` |
| `temperature` | float | `0.9` | Sampling temperature — lower = more stable, higher = more expressive |
| `top_p` | float | `1.0` | Nucleus sampling cutoff |
| `top_k` | int | `50` | Top-k vocabulary cutoff |
| `repetition_penalty` | float | `1.05` | Higher = less repetition |
| `max_new_tokens` | int | `2048` | Max codec tokens (≈ audio length cap) |

**Voice Cloning Modes:**
| Mode | Requirements | Quality |
|------|-------------|---------|
| **ICL** (In-Context Learning) | `audio_prompt_id` + `ref_text` | Best speaker similarity (~0.95 SIM) |
| **x-vector only** | `audio_prompt_id` only (no transcript) | Good — speaker embedding from audio alone |
| **Default voice** | No reference WAV | Neutral voice, no cloning |

**Usage:**
```bash
# Voice clone with transcript (ICL — best quality)
curl -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a cloned voice.","params":{"audio_prompt_id":"abc123","ref_text":"exact words in the reference","language":"english"}}'

# Voice clone without transcript (x-vector only)
curl -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a cloned voice.","params":{"audio_prompt_id":"abc123","language":"english"}}'

# Default voice (no reference)
curl -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world.","params":{"language":"english"}}'
```

> **Note:** The Base model replaces the CustomVoice variant (which had 9 built-in speakers + `instruct` voice design but did NOT support voice cloning). The CustomVoice model's `generate_voice_clone()` raised `ValueError: model with tts_model_type: custom_voice does not support generate_voice_clone`. Reference WAV files are shared between orchestrator and engine-qwen via Docker volume mount at `/tmp/tts_uploads`.

---

### Experimental / Blocked Engines

#### `vibevoice` — VibeVoice-1.5B 🧪

| | |
|---|---|
| **Size** | ~6 GB (BF16) |
| **VRAM** | ~6.5 GB |
| **Languages** | English, Chinese |
| **Voice Cloning** | ✅ Multi-speaker, voice design |
| **Container** | engine-mid (needs SGLang) |

**Status:** ⚠️ SGLang image needs upstream update. Served via SGLang-Omni OpenAI-compatible endpoint.

---

#### `higgs` — Higgs Audio v3 🧪

| | |
|---|---|
| **Size** | ~8 GB (BF16) |
| **VRAM** | ~8.5 GB |
| **Languages** | English |
| **Voice Cloning** | ✅ Reference WAV |
| **Container** | engine-mid |

**Status:** ⚠️ Needs SGLang update.

---

#### `s2pro` — Fish S2-Pro (5B) 🧪

| | |
|---|---|
| **Size** | ~10 GB (BF16) |
| **VRAM** | ~10 GB |
| **Languages** | 80+ languages |
| **Voice Cloning** | ✅ |
| **Container** | SGLang (OpenAI-compatible `/v1/audio/speech`) |

**Status:** ⚠️ Needs SGLang serving infra.

---

#### `indextts` — IndexTTS-2 ⚠️

| | |
|---|---|
| **Size** | ~1.5 GB |
| **Languages** | Multilingual |
| **Voice Cloning** | ✅ Reference WAV required |
| **Container** | engine-legacy |

**Status:** ⚠️ SKIPPED — needs legacy stack (torch 1.13 + tf 4.46).

---

#### `xtts` — XTTS-v2 ⚠️

**Status:** ⚠️ Broken — torchcodec incompatibility with torch nightly.

---

#### `cosyvoice` — CosyVoice2 ⚠️

**Status:** ⚠️ Not built — git clone + openai-whisper build failure.

---

#### `parler` — Parler-TTS ⚠️

**Status:** ⚠️ SKIPPED — needs legacy stack (torch 1.x + tf 4.x).

---

#### `csm` — Sesame CSM 1B ⚠️

**Status:** ⚠️ Blocked — meta-llama/Llama-3.2-1B gated model.

---

#### `openvoice` — OpenVoice v2 ⚠️

**Status:** ⚠️ Build failure — av package Cython compilation error.

---

#### `neutts` — NeuTTS Air ⚠️

**Status:** ⚠️ Not yet configured — needs `_load_neutts()` implementation.

---

#### `manatts` — ManaTTS (FA) ⚠️

**Status:** ⚠️ parallel-wavegan not available on PyPI.

---

## Voice Cloning

### Which engines support voice cloning?

| Engine | Method | Ref Audio | Ref Text | Container Mode Works? |
|--------|--------|-----------|----------|----------------------|
| `f5tts` | Zero-shot clone | **Required** | Optional (improves quality) | ❌ (same file-path issue) |
| `chatterbox` | Zero-shot clone | Optional | No | ❌ |
| `chatterboxturbo` | Zero-shot clone | Optional | No | ❌ |
| `qwen3tts` | Voice clone ICL | Required | Required (best quality) | ✅ |
| `qwen3tts` | Voice clone x-vector | Required | No (embedding only) | ✅ |
| `styletts2` | Style transfer | Optional | No | ❌ |
| `fishspeech` | Zero-shot clone | Via tokenizer | No | ❌ |
| `zonos` | Speaker embedding | Optional | No | ❌ |
| `omnivoice` | Zero-shot clone | Optional | Optional | ❌ (file), ✅ (text-only) |
| `dia` | Speaker clone | Optional | No | ❌ |
| `outetts` | Speaker creation | Optional | No | ❌ |

### How voice cloning works

1. **Upload** reference WAV via `POST /upload` → get back an `{id}`
2. **Pass** `audio_prompt_id: "{id}"` in `params` to `POST /synthesize/{engine}`
3. The engine's `_synth_*` function looks up `/tmp/tts_uploads/{id}.wav` and uses it
4. Docker volumes (`/tmp/tts_uploads:/tmp/tts_uploads`) share files between orchestrator and engine containers — verified working on engine-qwen

### Container mode — per-engine status

| Engine | Container mode status | Notes |
|--------|----------------------|-------|
| `qwen3tts` | ✅ Working | Volume mount verified — `/tmp/tts_uploads` shared with engine-qwen |
| `chatterbox` | ❌ Broken | Same file-path issue (needs engine-current volume mount verified) |
| `chatterboxturbo` | ❌ Broken | Same as chatterbox |
| `f5tts` | ❌ Broken | Same file-path issue |
| `styletts2` | ❌ Broken | Same file-path issue |
| `zonos` | ❌ Broken | Same file-path issue |
| `omnivoice` | ❌ (file), ✅ (text-only) | Text-only `voice_clone_prompt` works without files |

All engine containers have `/tmp/tts_uploads` mounted in docker-compose. The remaining "broken" entries need per-engine verification — they may work if the engine container has the volume mount.

---

## Container Architecture

For external services integrating with TTS Lab, understanding the topology helps with reliability and latency planning:

```
External Service
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  Orchestrator (port 8001)                             │
│  - No ML libraries loaded                             │
│  - Pure HTTP dispatch to engine containers            │
│  - Serves Web UI at GET /                             │
│  - Forwards synthesis requests to correct container   │
└────┬──────────────┬──────────────┬───────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│ engine- │  │ engine-  │  │ engine-  │
│ current │  │ mid      │  │ legacy   │
│ :8101   │  │ :8103    │  │ :8102    │
│         │  │          │  │          │
│ 21      │  │ Qwen3TTS │  │ IndexTTS │
│ engines │  │ VibeVoice│  │ (skipped)│
│         │  │ Higgs    │  │          │
└─────────┘  └──────────┘  └──────────┘
     │
     ▼
┌──────────┐  ┌──────────┐
│ Orpheus  │  │ SGLang   │
│ vllm     │  │ :8005    │
│ :8002    │  │ S2-Pro   │
└──────────┘  └──────────┘
```

### Engine URL Environment Variables

In orchestrator mode, engine dispatch is configured via env vars:

```bash
PIPER_URL=http://engine-current:8101
KOKORO_URL=http://engine-current:8101
CHATTERBOX_URL=http://engine-current:8101
QWEN3TTS_URL=http://engine-qwen:8104
VIBEVOICE_URL=http://engine-mid:8103
S2PRO_SGLANG_URL=http://sglang:8000/v1/audio/speech
```

Each engine container runs `tts_lab_engine_server.py` and exposes:
- `GET /health` — engine availability + GPU info
- `POST /synthesize` — `{"engine": "...", "text": "...", "params": {...}}`
- `POST /unload` — evict current engine from VRAM

### Timeouts

Default synthesis timeout: **300 seconds**. Per-engine overrides:

| Engine | Timeout |
|--------|---------|
| `fishspeech` | 360s |
| `orpheus` | 240s |
| `dia` | 180s |
| `bark` | 180s |
| `qwen3tts` | 180s |
| `outetts` | 120s |
| `f5tts` | 120s |
| `manatts` | 120s |
| `chattts` | 90s |

---

## Code Examples

### Python

```python
import requests
import base64
import json

BASE = "http://192.168.0.87:8001"

# 1. Check status
r = requests.get(f"{BASE}/status")
status = r.json()
available = [n for n, m in status["models"].items() if m.get("available")]
print(f"Available engines: {available}")

# 2. Simple synthesis (Kokoro — fast CPU TTS)
r = requests.post(f"{BASE}/synthesize/kokoro", json={
    "text": "Hello, this is a test of the Kokoro text to speech engine.",
    "params": {"voice": "bm_lewis"}
})
result = r.json()
wav_bytes = base64.b64decode(result["audio_b64"])
with open("output.wav", "wb") as f:
    f.write(wav_bytes)
print(f"Generated {result['audio_dur_ms']}ms audio in {result['synth_time_ms']}ms (RTF {result['rtf']}×)")

# 3. Persian TTS (Chatterbox — default Persian model)
r = requests.post(f"{BASE}/synthesize/chatterbox", json={
    "text": "سلام، حالت چطوره؟ امروز هوا خیلی خوبه.",
    "params": {"exaggeration": "0.65", "temperature": "0.8"}
})
persian_wav = base64.b64decode(r.json()["audio_b64"])
with open("persian_output.wav", "wb") as f:
    f.write(persian_wav)

# 4. Voice cloning (bare-metal mode only)
# First upload reference audio
with open("my_voice.wav", "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": f})
ref_id = r.json()["id"]

# Then synthesize with it
r = requests.post(f"{BASE}/synthesize/f5tts", json={
    "text": "This is my cloned voice speaking.",
    "params": {
        "audio_prompt_id": ref_id,
        "ref_text": "exact words spoken in my_voice.wav",
        "nfe_step": "32"
    }
})

# 5. Long-form text (auto-chunked by engine)
long_text = "..."  # paragraphs of text
r = requests.post(f"{BASE}/synthesize/chatterbox", json={
    "text": long_text,
    "params": {"max_length": "20000", "chunk_silence_ms": "350"}
})

# 6. Multi-engine batch synthesis
engines = ["piper", "kokoro", "melo", "f5tts"]
for engine in engines:
    try:
        r = requests.post(f"{BASE}/synthesize/{engine}", json={
            "text": "The quick brown fox jumps over the lazy dog."
        }, timeout=120)
        result = r.json()
        if "error" in result:
            print(f"{engine}: ERROR — {result['error']}")
        else:
            print(f"{engine}: {result['audio_dur_ms']}ms audio, {result['synth_time_ms']}ms synth, RTF {result['rtf']}×")
    except Exception as e:
        print(f"{engine}: FAILED — {e}")
```

### JavaScript (Browser / Node.js)

```javascript
const API = "http://192.168.0.87:8001";

// Simple synthesis
async function synthesize(engine, text, params = {}) {
  const res = await fetch(`${API}/synthesize/${engine}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, params }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);

  // Decode base64 WAV and play
  const bytes = Uint8Array.from(atob(data.audio_b64), c => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
  return data;
}

// Upload reference audio for voice cloning
async function uploadReference(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
  return (await res.json()).id;  // Use as audio_prompt_id
}

// Get available voices for an engine
async function getVoices(engine) {
  const res = await fetch(`${API}/voices/${engine}`);
  return (await res.json()).voices;
}

// Status poller
async function pollStatus() {
  const res = await fetch(`${API}/status`);
  const status = await res.json();
  console.log(`RAM: ${status.system.used}/${status.system.total} MB`);
  if (status.gpu) {
    console.log(`VRAM: ${status.gpu.vram_used}/${status.gpu.vram_total} MB`);
  }
  return status;
}

// Usage
await synthesize("piper", "Hello world");
await synthesize("kokoro", "Hello world", { voice: "bm_lewis" });
await synthesize("chatterbox", "سلام دنیا", { exaggeration: "0.65" });

const refId = await uploadReference(document.querySelector("input[type=file]").files[0]);
await synthesize("f5tts", "Cloned voice speaking", {
  audio_prompt_id: refId,
  ref_text: "exact transcript of uploaded audio",
});
```

### cURL

```bash
# Minimal — defaults for everything
curl -s -X POST http://192.168.0.87:8001/synthesize/piper \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}' | jq -r '.audio_b64' | base64 -d > out.wav

# With all params
curl -s -X POST http://192.168.0.87:8001/synthesize/kokoro \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "params": {
      "voice": "bm_lewis",
      "speed": "1.2"
    }
  }' | jq '.audio_dur_ms, .synth_time_ms, .rtf'

# Persian
curl -s -X POST http://192.168.0.87:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"سلام، خوبی؟ امروز چطور بود؟"}'

# Voice clone (Qwen3-TTS Base — with transcript, best quality)
curl -s -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is my cloned voice.",
    "params": {
      "audio_prompt_id": "abc12345",
      "ref_text": "exact words in the reference clip",
      "language": "english",
      "temperature": "0.9"
    }
  }'

# Voice clone without transcript (x-vector only)
curl -s -X POST http://192.168.0.87:8001/synthesize/qwen3tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is my cloned voice.",
    "params": {
      "audio_prompt_id": "abc12345",
      "language": "english"
    }
  }'

# Upload reference audio
curl -s -X POST http://192.168.0.87:8001/upload \
  -F "file=@my_voice.wav"
# → {"id":"a1b2c3d4","filename":"my_voice.wav","size":48044}

# Engine status
curl -s http://192.168.0.87:8001/status | jq '.models | to_entries | map(select(.value.available)) | map(.key)'

# Get voices for an engine
curl -s http://192.168.0.87:8001/voices/kokoro | jq '.voices'

# Pre-load an engine
curl -s -X POST http://192.168.0.87:8001/models/chatterbox/load \
  -H "Content-Type: application/json" \
  -d '{"params": {"model": "persian"}}'

# Unload an engine
curl -s -X DELETE http://192.168.0.87:8001/models/chatterbox

# Re-probe availability
curl -s -X POST http://192.168.0.87:8001/refresh

# Server logs
curl -s http://192.168.0.87:8001/logs
```

---

## Response Envelope

All synthesis responses follow this structure:

```json
{
  "audio_b64": "base64-encoded WAV bytes",
  "sample_rate": 24000,
  "synth_time_ms": 342,
  "audio_dur_ms": 2150,
  "rtf": 0.159,
  "load_time_s": 2.4
}
```

| Field | Description |
|-------|-------------|
| `audio_b64` | Full WAV file as base64 string (decode to get raw bytes) |
| `sample_rate` | Audio sample rate in Hz (varies by engine: 16000–44100) |
| `synth_time_ms` | Wall-clock time spent generating audio (excludes model loading) |
| `audio_dur_ms` | Duration of the generated audio in milliseconds |
| `rtf` | Real-Time Factor — values < 1.0 are faster than real-time |
| `load_time_s` | Time spent loading the model (0 if already loaded in VRAM) |

**Error responses:**
```json
{
  "error": "Human-readable error message",
  "trace": "Python traceback (last 4 frames)"
}
```

---

## Rate Limiting & Concurrency

- **No rate limiting** — this is an internal lab service
- **Single engine at a time in VRAM** — engine containers evict the previous engine before loading a new one (lazy-loading design)
- **Concurrent synthesis** — possible for different engines in different containers, but same-container engines serialize
- **GPU memory** — RTX 5060 Ti 16 GB GDDR7. Largest engines (VibeVoice 6.5 GB, Higgs 8.5 GB) can't co-reside

---

## See Also

- [Chatterbox API Reference](CHATTERBOX_API_REFERENCE.md) — detailed Chatterbox-specific docs
- [Engine Compatibility YAML](../engine_compatibility.yaml) — canonical engine status and stack assignments
- [Architecture Reference](ARCHITECTURE_REFERENCE.md) — container topology, VRAM budget
- [Known Issues](KNOWN_ISSUES.md) — current bugs and fix history
