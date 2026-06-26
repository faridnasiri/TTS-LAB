# Qwen 3.6 LLM — Deployment Reference

> **Deployed:** 2026-06-26
> **VM:** arthur@192.168.0.87
> **GPU:** RTX 5060 Ti 16 GB GDDR7 (Blackwell sm_120)

---

## 1. Overview

Qwen 3.6 27B dense model deployed as a reasoning and programming assistant. Runs in a standalone Docker container using the pre-built `ghcr.io/ggml-org/llama.cpp:server-cuda` image — **no source compilation required**. Exposes an OpenAI-compatible chat API on port 8006. Integrated into the TTS Lab orchestrator with automatic VRAM eviction of all TTS engines before inference.

### Key Metrics

| Metric | Value |
|--------|-------|
| Model | Qwen3.6-27B |
| Parameters | 27B dense (all active) |
| Quantization | Q3_K_M (3-bit, standard GGUF) |
| File size | 13 GB |
| VRAM (model only) | ~12.4 GB |
| VRAM (model + 32K KV cache) | ~13.6 GB |
| Inference speed | ~23 tok/s |
| Context window | 32,768 tokens |
| KV cache type | q4_0 (4-bit quantized) |
| Thinking mode | Enabled (`<think>` tags) |
| Native context | 262K (we cap at 32K for VRAM) |
| SWE-bench Verified | 77.2% |
| AIME 2025 | 94.1% |
| License | Apache 2.0 |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOST: RTX 5060 Ti 16 GB VRAM                  │
│                    arthur@192.168.0.87                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  tts-lab-orchestrator │    │  tts-lab-llm-qwen36           │   │
│  │  port 8009            │    │  port 8006                    │   │
│  │──────────────────────│    │──────────────────────────────│   │
│  │  FastAPI + uvicorn   │    │  llama-server (llama.cpp)     │   │
│  │  NO ML libraries      │    │  CUDA 12.8 (pre-built)        │   │
│  │──────────────────────│    │──────────────────────────────│   │
│  │  Routes:              │    │  Model: 27B Q3_K_M GGUF       │   │
│  │  /synthesize/qwen36   │    │  Context: 32K                 │   │
│  │    ↓                  │    │  KV cache: q4_0               │   │
│  │  Phase 1: Evict TTS   │    │  Flash Attn: ON               │   │
│  │  Phase 2: Verify VRAM │    │  API: /v1/chat/completions    │   │
│  │  Phase 3: Route → LLM │───▶│  VRAM: ~13.6 GB               │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  tts-lab-engine-current│   │  tts-lab-engine-qwen          │   │
│  │  port 8101            │    │  port 8104                    │   │
│  │──────────────────────│    │──────────────────────────────│   │
│  │  21 TTS engines       │    │  Qwen3TTS 1.7B               │   │
│  │  POST /evict endpoint │    │  POST /evict endpoint         │   │
│  │  VRAM: 0-12 GB (lazy) │    │  VRAM: 0-6 GB (lazy)          │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│                                                                  │
│  All containers use --network host for direct localhost access   │
└─────────────────────────────────────────────────────────────────┘
```

### Dispatch Flow (qwen36 synthesis)

```
User types prompt in UI
  │
  ▼
POST /synthesize/qwen36 → orchestrator:8009
  │
  ├─ Phase 1: GLOBAL EVICTION ─────────────────────────────
  │   POST /evict → engine-current:8101   (unloads any TTS model)
  │   POST /evict → engine-qwen:8104      (unloads Qwen3TTS if loaded)
  │   POST /evict → engine-mid:8103       (unloads VibeVoice/Higgs if loaded)
  │   Each returns: {evicted: true/false, engine_was: "...", vram_free_mb: N}
  │
  ├─ Phase 2: VERIFY ─────────────────────────────────────
  │   All TTS containers confirmed evicted
  │   VRAM: ~15.5 GB free (only container overhead remains)
  │
  ├─ Phase 3: ROUTE TO LLM ───────────────────────────────
  │   POST /v1/chat/completions → llm-qwen36:8006
  │   Payload: {messages, temperature, max_tokens, top_p, ...}
  │   llama.cpp loads model into GPU, runs inference
  │
  ▼
Response → orchestrator → JSON to UI
  {
    "text": "def fibonacci(n): ...",
    "reasoning": "Here's a thinking process:\n1. ...",
    "tokens": 304,
    "tokens_per_sec": 23.1,
    "model": "/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf",
    "finish_reason": "stop",
    "synth_time_ms": 13149
  }

TTS engines stay evicted. Next TTS synthesis request triggers lazy reload.
```

---

## 3. Deployment

### 3.1 — Prerequisites

```bash
# On VM (arthur@192.168.0.87)

# 1. Pre-built llama.cpp image
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

# 2. Model file (~13 GB download)
mkdir -p /opt/models/llm
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('batiai/Qwen3.6-27B-GGUF', 'Qwen-Qwen3.6-27B-Q3_K_M.gguf',
                local_dir='/opt/models/llm/')
"

# 3. Verify model
ls -lh /opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf
# Expected: ~13 GB
```

### 3.2 — Deploy Container

```bash
# Stop existing if any
docker rm -f tts-lab-llm-qwen36 2>/dev/null

# Deploy
docker run -d \
  --name tts-lab-llm-qwen36 \
  --gpus all \
  --network host \
  -v /opt/models:/opt/models \
  --restart unless-stopped \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf \
  --host 0.0.0.0 \
  --port 8006 \
  --n-gpu-layers 99 \
  --ctx-size 32768 \
  --parallel 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --threads 4 \
  --threads-batch 8
```

### 3.3 — Verify

```bash
# Check container
docker ps --filter name=llm-qwen36

# Check logs (should show "model loaded" + "server is listening")
docker logs tts-lab-llm-qwen36

# Test health
curl -s http://localhost:8006/health
# Expected: HTTP 200 OK

# Test inference
curl -s http://localhost:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "messages": [{"role": "user", "content": "what is 2+2"}],
    "max_tokens": 100
  }' | python3 -m json.tool

# Check VRAM
nvidia-smi
# Expected: ~13500 MiB used for LLM
```

### 3.4 — Orchestrator Integration

```bash
# The orchestrator needs QWEN36_URL env var
docker rm -f tts-lab-orchestrator 2>/dev/null
docker run -d \
  --name tts-lab-orchestrator \
  --network host \
  -v /opt/models:/opt/models \
  -e ORCHESTRATOR_MODE=1 \
  -e PIPER_URL=http://localhost:8101 \
  -e KOKORO_URL=http://localhost:8101 \
  -e MELO_URL=http://localhost:8101 \
  -e CHATTTS_URL=http://localhost:8101 \
  -e BARK_URL=http://localhost:8101 \
  -e QWEN3TTS_URL=http://localhost:8104 \
  -e QWEN36_URL=http://localhost:8006 \
  --restart unless-stopped \
  tts-lab-orchestrator:latest \
  uvicorn tts_lab:app --host 0.0.0.0 --port 8009 --workers 1
```

---

## 4. Configuration Reference

### 4.1 — llama-server Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--model` | `/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf` | GGUF model file |
| `--host` | `0.0.0.0` | Bind to all interfaces |
| `--port` | `8006` | API port |
| `--n-gpu-layers` | `99` | All layers on GPU (27B model has ~64 layers, 99 ensures all) |
| `--ctx-size` | `32768` | Max context window (32K tokens) |
| `--parallel` | `1` | Single concurrent request (saves VRAM) |
| `--cache-type-k` | `q4_0` | 4-bit KV cache keys (~4× smaller than f16) |
| `--cache-type-v` | `q4_0` | 4-bit KV cache values |
| `--flash-attn` | `on` | Flash Attention (faster, less VRAM) |
| `--jinja` | — | Enable Jinja chat template (required for Qwen 3.6 thinking mode) |
| `--threads` | `4` | CPU threads for tokenization |
| `--threads-batch` | `8` | CPU threads for batch processing |

### 4.2 — Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN36_URL` | `http://localhost:8006` | LLM URL (set in orchestrator) |
| `HF_HOME` | — | HuggingFace cache (not needed at runtime — model is mounted) |

### 4.3 — Model Parameters (sent by orchestrator)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | 0.7 | 0.1–2.0 | Sampling temperature |
| `top_p` | float | 0.9 | 0.1–1.0 | Nucleus sampling |
| `max_tokens` | int | 2048 | 256–8192 | Max response tokens |
| `presence_penalty` | float | 0.0 | -2.0–2.0 | Penalize repeated topics |
| `frequency_penalty` | float | 0.0 | -2.0–2.0 | Penalize word repetition |
| `seed` | int | -1 | -1–99999 | -1 = random, ≥0 = deterministic |
| `system_prompt` | string | *see below* | — | System prompt for chat |

Default system prompt:
```
You are a helpful AI assistant specialized in reasoning and programming.
```

---

## 5. VRAM Budget

### 5.1 — Breakdown

```
RTX 5060 Ti — 16 GB total (15,847 MiB usable)

Component                         VRAM
─────────────────────────────────────────
Model weights (Q3_K_M, 27B)      ~12.4 GB
KV cache (32K tokens, q4_0)      ~1.5 GB
CUDA context + overhead           ~0.2 GB
llama-server runtime              ~0.1 GB
─────────────────────────────────────────
Total LLM idle                   ~13.6 GB
Free                              ~2.4 GB
```

### 5.2 — Coexistence Matrix

| Scenario | LLM VRAM | TTS VRAM | Total | Fits? |
|----------|:--------:|:--------:|:-----:|:-----:|
| LLM only | 13.6 GB | 0 GB | 13.6 GB | ✅ |
| LLM + kokoro (0.2 GB) | 13.6 GB | 0.2 GB | 13.8 GB | ✅ |
| LLM + piper (0.2 GB) | 13.6 GB | 0.2 GB | 13.8 GB | ✅ |
| LLM + melo (1.4 GB) | 13.6 GB | 1.4 GB | 15.0 GB | ✅ |
| LLM + chattts (2 GB) | 13.6 GB | 2.0 GB | 15.6 GB | ⚠️ tight |
| LLM + bark (12 GB) | 13.6 GB | 12.0 GB | 25.6 GB | ❌ OOM |
| LLM + Qwen3TTS (6 GB) | 13.6 GB | 6.0 GB | 19.6 GB | ❌ OOM |

**Rule of thumb:** With the LLM loaded, only light TTS engines (≤2 GB) can coexist. Heavy TTS engines must be evicted before LLM loads. The global eviction protocol handles this automatically.

### 5.3 — Context Size vs VRAM

| Context Size | KV Cache (approx) | Total VRAM | Fits 16GB? |
|-------------|:-------------------:|:----------:|:----------:|
| 4096 | 0.3 GB | 13.0 GB | ✅ |
| 8192 | 0.5 GB | 13.2 GB | ✅ |
| 16384 | 0.9 GB | 13.6 GB | ✅ |
| 32768 | 1.5 GB | 14.2 GB | ✅ |
| 65536 | 3.0 GB | 15.7 GB | ⚠️ very tight |
| 131072 | 6.0 GB | 18.7 GB | ❌ OOM |

---

## 6. API Reference

### 6.1 — Health Check

```bash
GET http://192.168.0.87:8006/health
```
Returns: HTTP 200 OK (no body)

### 6.2 — Chat Completion

```bash
POST http://192.168.0.87:8006/v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a Python function to reverse a linked list."}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "choices": [{
    "finish_reason": "stop",
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Here's a Python function to reverse a linked list:\n\n```python\n...\n```",
      "reasoning_content": "Here's a thinking process:\n\n1. Understand the task..."
    }
  }],
  "created": 1782463624,
  "model": "/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf",
  "usage": {
    "completion_tokens": 304,
    "prompt_tokens": 23,
    "total_tokens": 327,
    "prompt_tokens_details": {"cached_tokens": 0}
  }
}
```

**Key fields:**
- `choices[0].message.content` — the actual response text
- `choices[0].message.reasoning_content` — the thinking process (Qwen 3.6 native)
- `choices[0].finish_reason` — `"stop"` (natural end), `"length"` (hit max_tokens), `"content_filter"` (safety)
- `usage.total_tokens` — total tokens consumed

### 6.3 — Orchestrator Synthesis (qwen36)

```bash
POST http://192.168.0.87:8009/synthesize/qwen36
Content-Type: application/json

{
  "text": "Write a Python function to reverse a linked list.",
  "params": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "system_prompt": "You are an expert Python programmer."
  }
}
```

**Response (differs from TTS):**
```json
{
  "text": "Here's a Python function...",
  "reasoning": "Here's a thinking process:\n1. ...",
  "tokens": 304,
  "tokens_per_sec": 23.1,
  "model": "/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf",
  "finish_reason": "stop",
  "synth_time_ms": 13149
}
```

**TTS response (for comparison):**
```json
{
  "audio_b64": "UklGRiT4AABXQVZF...",
  "sample_rate": 24000,
  "synth_time_ms": 150,
  "audio_dur_ms": 3200,
  "rtf": 0.047,
  "load_time_s": 2.3
}
```

The UI detects `data.text !== undefined` to branch between chat (LLM) and audio (TTS) rendering.

---

## 7. Thinking Mode

Qwen 3.6 has native thinking tokens (`<think>...</think>`) that enable chain-of-thought reasoning before generating the final answer. The `--jinja` flag enables the chat template that triggers thinking mode.

### 7.1 — How It Works

1. User sends: "Write a function to detect palindromes"
2. Model generates thinking: `<think>1. Understand task: detect palindromes\n2. Approach: compare string to reverse\n3. Edge cases: empty strings, case sensitivity...</think>`
3. Model generates answer: `def is_palindrome(s): ...`

### 7.2 — Token Allocation

Thinking tokens are included in `max_tokens`. With `max_tokens=200`:
- ~180 tokens go to thinking
- ~20 tokens left for the actual answer
- Result: `finish_reason: "length"`, empty or truncated `content`

**Recommendation:** Use `max_tokens >= 512` for simple queries, `>= 2048` for coding tasks.

### 7.3 — Disabling Thinking

To disable thinking mode, remove the `--jinja` flag from the llama-server command. The model will generate answers directly without reasoning. Speed increases ~2× but answer quality may decrease for complex tasks.

### 7.4 — Thinking in the UI

The `reasoning` field is available in the API response but the current UI does not render it separately. Future enhancement: collapsible "Show reasoning" section in chat messages.

---

## 8. Global VRAM Eviction Protocol

### 8.1 — Why

The LLM needs ~13.6 GB of clean VRAM. Any resident TTS model would cause OOM. The orchestrator evicts ALL TTS engines from ALL containers before routing to the LLM.

### 8.2 — Endpoints Involved

| Endpoint | Container | Purpose |
|----------|-----------|---------|
| `POST /evict` | engine-current:8101 | Evict current TTS model |
| `POST /evict` | engine-qwen:8104 | Evict Qwen3TTS |
| `POST /evict` | engine-mid:8103 | Evict VibeVoice/Higgs |
| `POST /evict` | engine-legacy:8102 | Evict IndexTTS/Parler |

Each `/evict` call:
1. Calls `_evict_current()` — deletes Python instance reference
2. Runs `gc.collect()` — forces garbage collection
3. Calls `torch.cuda.empty_cache()` — releases CUDA caching allocator
4. Returns `{evicted: bool, engine_was: str|null, vram_free_mb: int, vram_total_mb: int}`

### 8.3 — Code Path

```python
# tts_lab_dispatch.py

def _do_synth(name, text, params):
    if name in _REMOTE_ENGINES:
        if MODEL_INFO.get(name, {}).get("engine_type") == "llm":
            return _do_synth_llm(name, text, params)  # ← LLM path
        return _do_synth_remote(name, text, params)    # ← TTS path

def _do_synth_llm(name, text, params):
    # Phase 1: Evict ALL TTS engines
    evict_results = _evict_all_tts_engines()

    # Phase 2: Route to LLM
    llm_url = _REMOTE_ENGINES.get(name, "http://llm-qwen36:8006")
    resp = httpx.post(f"{llm_url}/v1/chat/completions", json=payload, timeout=600)

    # Phase 3: Return text response
    return {"text": ..., "reasoning": ..., "tokens": ...}
```

### 8.4 — Failure Handling

| Scenario | Behavior |
|----------|----------|
| Engine container unreachable | Log warning, continue (best-effort) |
| Engine returns HTTP error | Log error, continue |
| All evictions succeed | LLM loads into ~15.5 GB free VRAM |
| SGLang containers running | Must be stopped manually (no `/evict` endpoint) |
| Orpheus container running | Must be stopped manually |

---

## 9. Monitoring & Debugging

### 9.1 — Health Checks

```bash
# LLM container
curl -s http://localhost:8006/health          # HTTP 200 if running

# LLM slots (VRAM usage per slot)
curl -s http://localhost:8006/slots | python3 -m json.tool

# Orchestrator status (shows qwen36 availability)
curl -s http://localhost:8009/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
q = d['models'].get('qwen36', {})
print(f\"Available: {q.get('available')}\")
print(f\"Reason: {q.get('reason', 'N/A')}\")
"

# Engine eviction endpoint
curl -s -X POST http://localhost:8101/evict | python3 -m json.tool
```

### 9.2 — Logs

```bash
# LLM container logs (model loading, inference)
docker logs tts-lab-llm-qwen36

# Follow logs
docker logs -f tts-lab-llm-qwen36

# Orchestrator logs (eviction, dispatch)
docker logs tts-lab-orchestrator | grep -E 'LLM|evict|qwen36'

# Engine server logs
docker logs tts-lab-engine-current | grep -E 'evict|VRAM'
```

### 9.3 — VRAM Monitoring

```bash
# All containers
docker exec tts-lab-llm-qwen36 nvidia-smi

# Or on host
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Per-container VRAM (if using host network — all share GPU)
# The LLM uses ~13.6 GB when loaded
# Engine containers use 0 when evicted, 0-12 GB when TTS loaded
```

### 9.4 — Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `"reason": "Connection refused"` | LLM container not running | `docker start tts-lab-llm-qwen36` |
| `request exceeds available context size` | Prompt too long for ctx-size | Increase `--ctx-size` or shorten prompt |
| `finish_reason: "length"` with empty content | max_tokens too small (all used by thinking) | Increase `max_tokens` to 512+ |
| OOM / CUDA out of memory | TTS engine not evicted before LLM load | Check eviction logs, restart LLM container |
| `model loaded` but no response | Model file corrupt or wrong format | Re-download GGUF, verify checksum |
| Slow response (>30s) | Thinking mode with complex query, or CPU fallback | Check `--n-gpu-layers 99`, verify GPU usage |

---

## 10. Model Source & Updates

### 10.1 — Current Model

| Field | Value |
|-------|-------|
| HF repo | `batiai/Qwen3.6-27B-GGUF` |
| File | `Qwen-Qwen3.6-27B-Q3_K_M.gguf` |
| Quant type | Q3_K_M (standard 3-bit) |
| File size | 13 GB |
| VRAM usage | ~12.4 GB (model) + ~1.5 GB (32K KV cache) |
| Public | Yes, no auth required |

### 10.2 — Alternative Models (same repo)

| File | Quant | Size | Quality | Speed | Fits 16GB? |
|------|-------|------|---------|-------|:----------:|
| `Qwen-Qwen3.6-27B-IQ4_XS.gguf` | IQ4_XS | 15 GB | Better | Same | ❌ (need 17+ GB) |
| `Qwen-Qwen3.6-27B-Q3_K_M.gguf` | Q3_K_M | 13 GB | Good | Same | ✅ (current) |
| `Qwen-Qwen3.6-27B-IQ3_XXS.gguf` | IQ3_XXS | 11 GB | Decent | Faster | ✅ |

### 10.3 — Future: MoE Model (35B-A3B)

The Qwen3.6-35B-A3B MoE model would be ~3× faster (107 tok/s vs 23 tok/s) with similar VRAM usage. It requires:

1. llama.cpp built from source with CUDA Driver API stubs (the pre-built image doesn't support TQ quantization)
2. TQ3_4S GGUF file from `YTan2000/Qwen3.6-35B-A3B-TQ3_4S`
3. The source build failed 5 times due to `undefined reference to cuGetErrorString` linker error — NVIDIA removed libcuda.so stubs from CUDA 12.x devel images

**Status:** Blocked until llama.cpp or NVIDIA provides a fix for CUDA Driver API linking in devel containers.

### 10.4 — Updating the Model

```bash
# 1. Download new GGUF
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('batiai/Qwen3.6-27B-GGUF', 'Qwen-Qwen3.6-27B-Q3_K_M.gguf',
                local_dir='/opt/models/llm/', force_download=True)
"

# 2. Restart LLM container
docker restart tts-lab-llm-qwen36

# 3. Verify
docker logs tts-lab-llm-qwen36 | grep "model loaded"
```

---

## 11. Full Deploy Script

```bash
#!/bin/bash
# deploy-qwen36.sh — Full Qwen 3.6 LLM deployment
# Run on VM: arthur@192.168.0.87

set -e

echo "=== Qwen 3.6 LLM Deployment ==="
echo ""

# ── 1. Pull pre-built image ─────────────────────────────────
echo "[1/4] Pulling llama.cpp server-cuda image..."
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

# ── 2. Download model (if not exists) ───────────────────────
MODEL_FILE="/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "[2/4] Downloading model (~13 GB)..."
    mkdir -p /opt/models/llm
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('batiai/Qwen3.6-27B-GGUF', 'Qwen-Qwen3.6-27B-Q3_K_M.gguf',
                local_dir='/opt/models/llm/')
"
else
    echo "[2/4] Model already exists: $MODEL_FILE ($(du -sh $MODEL_FILE | cut -f1))"
fi

# ── 3. Deploy container ─────────────────────────────────────
echo "[3/4] Deploying LLM container..."
docker rm -f tts-lab-llm-qwen36 2>/dev/null || true
docker run -d \
  --name tts-lab-llm-qwen36 \
  --gpus all \
  --network host \
  -v /opt/models:/opt/models \
  --restart unless-stopped \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf \
  --host 0.0.0.0 \
  --port 8006 \
  --n-gpu-layers 99 \
  --ctx-size 32768 \
  --parallel 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --threads 4 \
  --threads-batch 8

# ── 4. Wait for model to load ───────────────────────────────
echo "[4/4] Waiting for model to load..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8006/health > /dev/null 2>&1; then
        echo "✅ LLM is ready!"
        break
    fi
    sleep 2
done

# Verify
echo ""
echo "=== Deployment Complete ==="
docker ps --filter name=llm-qwen36 --format '{{.Names}}  {{.Status}}'
echo ""
echo "Test: curl -s http://localhost:8006/v1/chat/completions -H 'Content-Type: application/json' --data-raw '{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":50}'"
```

---

## 12. Troubleshooting Decision Tree

```
LLM not responding?
├─ Container running?
│  ├─ NO → docker start tts-lab-llm-qwen36
│  └─ YES → docker logs tts-lab-llm-qwen36
│     ├─ "model loaded" + "server is listening" → ✅ Running
│     │  └─ Check orchestrator: curl localhost:8009/status → qwen36 available?
│     │     ├─ YES → Try: curl localhost:8006/v1/chat/completions
│     │     └─ NO → orchestrator missing QWEN36_URL
│     ├─ "failed to load model" → Model file missing/corrupt
│     │  └─ ls -lh /opt/models/llm/ → re-download if missing
│     ├─ "CUDA error" / "out of memory" → VRAM conflict
│     │  └─ docker restart tts-lab-llm-qwen36 (and stop heavy TTS first)
│     └─ "exiting" → Crash loop
│        └─ Check exact error, may need flag adjustment

"request exceeds available context size"?
└─ Increase --ctx-size (restart container with larger value)
   Current: 32768. Max for 16GB: ~65536 (very tight)

Empty response / "finish_reason": "length"?
└─ max_tokens too small for thinking mode
   └─ Increase max_tokens to 512+ (simple) or 2048+ (coding)

Slow inference?
├─ nvidia-smi → GPU util 0%? → CPU fallback (--n-gpu-layers not working)
│  └─ Check: docker inspect tts-lab-llm-qwen36 --format '{{.Args}}'
├─ GPU util >80%? → Normal, model is running on GPU
└─ 23 tok/s is expected for 27B Q3_K_M — this is the speed limit
```
