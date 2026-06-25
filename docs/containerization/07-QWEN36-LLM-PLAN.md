# Qwen 3.6 LLM Integration Plan — Reasoning & Programming Engine

> **Status:** DRAFT — pending review
> **Date:** 2026-06-25
> **Author:** Claude (research + planning)
> **Target:** Add Qwen 3.6 as a reasoning/programming LLM to the TTS Lab

---

## 1. Executive Summary

Add **Qwen 3.6** (Alibaba's April 2026 open-weight model) as a dedicated reasoning and programming assistant in the TTS Lab. The model will run in a **standalone container** using **llama.cpp** with **GGUF quantization**, exposed via an **OpenAI-compatible API**. It integrates with the orchestrator as a remote engine, leveraging existing dispatch patterns.

**Primary recommendation:** Qwen3.6-35B-A3B (MoE) with TQ3_4S quantization (~12.4 GiB, fits 16 GB VRAM with room for context).

**Architecture:** Standalone container (orpheus pattern) — does NOT inherit from tts-lab-base due to completely disjoint dependency tree.

---

## 2. Model Selection Analysis

### 2.1 — Qwen 3.6 Variants

| Variant | Architecture | Total Params | Active Params | BF16 Size | Best Quant for 16GB | Quant Size |
|---------|-------------|-------------|---------------|-----------|---------------------|------------|
| **Qwen3.6-3B** | Dense | 3B | 3B | ~6 GB | Q4_K_M | ~2.5 GB |
| **Qwen3.6-8B** | Dense | 8B | 8B | ~16 GB | Q4_K_M | ~5.5 GB |
| **Qwen3.6-14B** | Dense | 14B | 14B | ~28 GB | Q4_K_M | ~9 GB |
| **Qwen3.6-27B** | Dense (flagship) | 27B | 27B | ~54 GB | IQ4_XS / Q3_K_M | ~14-15 GB |
| **Qwen3.6-32B** | Dense | 32B | 32B | ~64 GB | Q3_K_M | ~16 GB |
| **Qwen3.6-35B-A3B** | MoE (sparse) | 35B | **~3B** | ~69 GB | **TQ3_4S** | **~12.4 GB** |
| **Qwen3.6-72B** | Dense | 72B | 72B | ~140 GB | ❌ Too large | — |
| **Qwen3.6-480B-A24B** | MoE (flagship) | 480B | ~24B | — | ❌ Too large | — |

### 2.2 — Coding & Reasoning Benchmarks

| Benchmark | Qwen3.6-27B | Qwen3.6-35B-A3B | Qwen3.5-397B-A17B | Claude 4.5 Opus |
|-----------|:-----------:|:----------------:|:-----------------:|:---------------:|
| **SWE-bench Verified** | **77.2** | **73.4** | 76.2 | 80.9 |
| **SWE-bench Pro** | **53.5** | — | 50.9 | 57.1 |
| **Terminal-Bench 2.0** | **59.3** | **51.5** | 52.5 | 59.3 |
| **SkillsBench** | **48.2** | — | 30.0 | — |
| **AIME 2025/26** | **94.1** | **92.7** | 91.3 | — |
| **GPQA Diamond** | **87.8** | **86.0** | 88.4 | — |
| **MMLU-Pro** | **86.2** | — | — | — |

**Key takeaway:** The 27B dense model is the strongest coder (beats their previous 397B MoE flagship). The 35B-A3B MoE is close behind while using only ~3B active parameters per token — making it much faster at inference.

### 2.3 — Recommendation

| Priority | Model | Quantization | Why |
|----------|-------|-------------|-----|
| **🏆 Primary** | **Qwen3.6-35B-A3B** | **TQ3_4S** | Fits 16GB with room for 4K context. 107 tok/s. Fast MoE (3B active). Excellent reasoning. |
| **🥈 Alternative** | **Qwen3.6-27B** | **IQ4_XS pure** | Strongest coding benchmarks. ~14-15 GB, tight on 16GB. ~43 tok/s (stock) / ~120 tok/s (speculative). |
| **🥉 Fallback** | **Qwen3.6-14B** | **Q4_K_M** | Smaller, faster. ~9 GB — leaves 7 GB for a TTS engine. Still decent for coding. |

**Decision: Start with Qwen3.6-35B-A3B TQ3_4S.** If quality is insufficient for heavy programming, switch to 27B IQ4_XS. If VRAM contention with TTS engines is an issue, fall back to 14B Q4_K_M.

---

## 3. Inference Engine Selection

### 3.1 — Comparison for Single-GPU Consumer Hardware

| Engine | VRAM Efficiency | Setup Complexity | API Compatibility | Long Context | Tok/s (est.) |
|--------|:---------------:|:----------------:|:-----------------:|:------------:|:------------:|
| **llama.cpp** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OpenAI-compatible | 262K on 24GB | 107 (MoE) / 43 (27B) |
| Ollama | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OpenAI-compatible | Limited | ~38 |
| vLLM | ⭐⭐⭐ | ⭐⭐ | OpenAI-compatible | Needs 2× GPU | ~68 |
| SGLang | ⭐⭐⭐ | ⭐⭐ | OpenAI-compatible | Needs 2× GPU | ~72 |
| Transformers | ⭐⭐ | ⭐⭐⭐ | Custom | Limited | Slowest |

### 3.2 — Decision: llama.cpp

**Rationale:**
1. **Best VRAM efficiency** — Only engine that can fit 262K context on a single consumer GPU via q4_0 KV cache
2. **Native GGUF support** — Works directly with community quants from HuggingFace
3. **OpenAI-compatible API** — `llama-server` exposes `/v1/chat/completions` endpoint
4. **Simple Docker deployment** — Single binary, no Python ML stack dependency hell
5. **No torch/transformers/CUDA version conflicts** — Completely isolated from TTS stacks
6. **Active maintenance** — Qwen 3.6 architecture (GatedDeltaNet + MTP) already supported
7. **Speculative decoding** — Optional DFlash for 2-3× speedup on code

---

## 4. Container Architecture

### 4.1 — Stack Positioning

```
Base (nvidia/cuda:12.8.2-runtime-ubuntu22.04)
  ├── Stack:current    torch 2.12 + tf 5.12.1
  │   └── Engine:current    21 TTS engines (port 8101)
  ├── Stack:mid        torch 2.12 + tf 4.51.3
  │   ├── Engine:qwen       Qwen3TTS (port 8104)
  │   └── Engine:mid        VibeVoice, Higgs (port 8103)
  ├── Stack:legacy     torch 1.13 + tf 4.46
  │   └── Engine:legacy     IndexTTS, Parler (port 8102)
  ├── Orchestrator     No ML — HTTP dispatch (port 8001)
  ├── Orpheus          vllm + CUDA 12.1 (port 8002, blocked)
  └── LLM:qwen36       llama.cpp + CUDA 12.8 (port 8006)  ← NEW
```

### 4.2 — Why a New Standalone Container

| Factor | Assessment |
|--------|-----------|
| **Dependency tree** | llama.cpp has ZERO overlap with any existing stack (no torch, no transformers, no Python ML libs) |
| **CUDA version** | llama.cpp supports CUDA 12.8 (same as base image) — unlike Orpheus which needs CUDA 12.1 |
| **VRAM conflict** | LLM at 12-15 GB + any TTS engine = OOM on 16 GB. Must be time-multiplexed |
| **Base image** | Can inherit from `nvidia/cuda:12.8.2-runtime-ubuntu22.04` (same as tts-lab-base) for consistency |
| **Pattern** | Follow `orpheus` standalone pattern — own Dockerfile, own server, no base inheritance |

### 4.3 — Container Design

```
Dockerfile.llm-qwen36:
  FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04
  → Build llama.cpp from source with CUDA support
  → Copy GGUF model file (or mount from /opt/models)
  → Entrypoint: llama-server with OpenAI-compatible API
  → Port: 8006
  → GPU: Required
  → VRAM: ~13 GB (model + KV cache for 4K context)
```

---

## 5. VRAM Budget & Coexistence Strategy

### 5.1 — VRAM Budget (16 GB Total)

| Scenario | LLM | TTS Engine | Total | Fits? |
|----------|-----|-----------|-------|:-----:|
| LLM only (idle TTS) | 12.4 GB | 0 GB | 12.4 GB | ✅ |
| LLM only (idle TTS) 27B | 14.5 GB | 0 GB | 14.5 GB | ✅ tight |
| LLM + light TTS (kokoro) | 12.4 GB | 0.2 GB | 12.6 GB | ✅ |
| LLM + medium TTS (chattts) | 12.4 GB | 2 GB | 14.4 GB | ✅ tight |
| LLM + heavy TTS (bark) | 12.4 GB | 12 GB | **24.4 GB** | ❌ OOM |
| LLM + Qwen3TTS | 12.4 GB | 6 GB | **18.4 GB** | ❌ OOM |
| No LLM + heavy TTS | 0 GB | 12 GB | 12 GB | ✅ |

### 5.2 — Coexistence Strategy

**Option A: Profile-based (Simple, Recommended)**
- LLM container is behind a `llm` Docker Compose profile
- User stops TTS-heavy containers before starting LLM, or vice versa
- Manual VRAM management — user decides what runs

**Option B: VRAM-aware orchestration (Complex, Future)**
- Add a `/vram` endpoint to each container that reports `nvidia-smi` usage
- Orchestrator checks VRAM before dispatching
- Auto-evict TTS engines before LLM inference
- Requires significant rework

**Decision: Start with Option A (profile-based).** The lab is single-user; manual switching is acceptable. Add Option B later if needed.

### 5.3 — Runtime Configuration

```bash
# Docker Compose profile: llm
docker compose --profile llm up -d     # Start LLM (stop heavy TTS first)
docker compose --profile llm down      # Stop LLM (free VRAM for TTS)

# Alternative: direct docker run
docker run -d --name tts-lab-llm-qwen36 --gpus all --network host \
  -v /opt/models:/opt/models \
  -p 8006:8006 \
  tts-lab-llm-qwen36:latest
```

---

## 6. llama.cpp Server Configuration

### 6.1 — Model File

**Primary:**
- Model: `Qwen3.6-35B-A3B-TQ3_4S.gguf` (~12.4 GiB)
- Source: [YTan2000/Qwen3.6-35B-A3B-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-TQ3_4S)
- Path on VM: `/opt/models/llm/qwen3.6-35b-a3b-tq3_4s.gguf`

**Alternative (stronger coding):**
- Model: `Qwen3.6-27B-IQ4_XS-pure.gguf` (~14.5 GiB)
- Source: [Ununnilium/Qwen3.6-27B-IQ4_XS-pure-GGUF](https://huggingface.co/Ununnilium/Qwen3.6-27B-IQ4_XS-pure-GGUF)
- Path on VM: `/opt/models/llm/qwen3.6-27b-iq4_xs-pure.gguf`

### 6.2 — llama-server Flags

```bash
llama-server \
  --model /opt/models/llm/qwen3.6-35b-a3b-tq3_4s.gguf \
  --host 0.0.0.0 \
  --port 8006 \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --parallel 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --threads 4 \
  --threads-batch 8
```

**Flag explanations:**
| Flag | Value | Rationale |
|------|-------|-----------|
| `--n-gpu-layers 99` | All layers on GPU | Max speed on 16 GB VRAM |
| `--ctx-size 4096` | 4K context | Conservative — leaves VRAM headroom. Bump to 8192 if VRAM permits |
| `--parallel 1` | Single request | Single-user lab; saves VRAM |
| `--cache-type-k q4_0` | 4-bit KV cache | Saves ~2 GB vs f16 KV cache |
| `--cache-type-v q4_0` | 4-bit value cache | Same |
| `--flash-attn on` | Flash attention | Faster, less VRAM |
| `--jinja` | Jinja templates | Qwen 3.6 chat template support |
| `--threads 4` | 4 CPU threads | For CPU-bound ops (tokenization) |

### 6.3 — API Endpoints

llama-server exposes these endpoints natively:
- `POST /v1/chat/completions` — OpenAI-compatible chat API
- `GET /health` — Health check (returns 200)
- `GET /v1/models` — Model list
- `GET /slots` — Slot status
- `POST /tokenize` — Token counting

---

## 7. Orchestrator Integration

### 7.1 — Registration Pattern

Follow the existing remote URL pattern. The orchestrator already auto-discovers `{ENGINE}_URL` env vars.

**In `docker-compose.yml` orchestrator environment:**
```yaml
QWEN36_URL: http://llm-qwen36:8006
```

**In `tts_lab_dispatch.py` `_build_remote_urls()`:**
- Auto-discovered from `QWEN36_URL` env var — no code changes needed

### 7.2 — Engine Registration

**`tts_lab_config.py` — MODEL_INFO:**
```python
"qwen36": {
    "label": "Qwen3.6-35B-A3B",
    "size": "~13 GB (TQ3_4S GGUF)",
    "rtf_est": "LLM — N/A",
    "ram_est_mb": 13000,
    "heavy": True,
    "notes": "Alibaba Qwen 3.6 MoE. 35B total, 3B active. Reasoning + coding. llama.cpp.",
    "arthur_fit": 3,
}
```

**`tts_lab_config.py` — MODEL_ORDER:**
```python
MODEL_ORDER = [
    # ... existing engines ...
    "qwen36",
]
```

### 7.3 — Synthesis Function (Text Generation)

Since this is a text→text LLM (not text→audio), the "synthesis" function differs from TTS engines. Options:

**Option A: Return text directly (Recommended)**
- `_synth_qwen36()` sends chat request, returns text response as JSON
- UI renders text response instead of audio player
- Clean separation of concerns

**Option B: TTS the output**
- LLM generates text, then pipe through a TTS engine
- Adds latency, but produces audio
- Confusing: which TTS engine to use?

**Decision: Option A.** The LLM is fundamentally a text service. The UI should show a chat interface for it, distinct from the TTS audio player. This means:

1. Add a chat/text UI section in `tts_lab_ui.py`
2. The synth function returns `{"text": response, "tokens": n, "tokens_per_sec": rtf}`
3. The dispatch layer handles text responses differently from audio responses

### 7.4 — Dispatch Flow

```
User types prompt in UI → POST /synthesize/qwen36
  → Orchestrator checks _REMOTE_ENGINES["qwen36"]
  → _do_synth_remote() POSTs to http://llm-qwen36:8006/v1/chat/completions
  → llama-server generates tokens
  → Response: {"text": "...", "usage": {"total_tokens": 150, ...}}
  → Orchestrator returns text response to UI
  → UI renders in chat panel
```

---

## 8. UI Design Considerations

### 8.1 — Chat Interface (New)

The existing UI is TTS-centric (text input → audio player). For the LLM, we need a chat panel:

```
┌─────────────────────────────────────────┐
│  Qwen 3.6 — Reasoning & Programming     │
│  [Model: 35B-A3B ▼]  [Temperature: 0.7] │
├─────────────────────────────────────────┤
│  System: You are a helpful AI assistant  │
│  specialized in reasoning and coding.    │
├─────────────────────────────────────────┤
│  👤 User: Write a Python function to     │
│  sort a list using quicksort.            │
│                                          │
│  🤖 Qwen: Here's a Python implementation │
│  of quicksort...                         │
│  ```python                               │
│  def quicksort(arr):                     │
│      ...                                 │
│  ```                                     │
├─────────────────────────────────────────┤
│  [___________________________________]   │
│  [Send]  [Clear]  [Copy Last Response]   │
└─────────────────────────────────────────┘
```

### 8.2 — Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | select | `35b-a3b` | Model variant (if multiple GGUF files available) |
| `temperature` | slider (0-2) | 0.7 | Sampling temperature |
| `max_tokens` | number | 2048 | Max tokens to generate |
| `top_p` | slider (0-1) | 0.9 | Nucleus sampling |
| `system_prompt` | textarea | "You are..." | System prompt |
| `thinking` | toggle | off | Enable Qwen 3.6 thinking/reasoning mode |

### 8.3 — UI Integration Points

The LLM chat UI can be:
- **A new tab** in the existing UI (alongside each TTS engine tab)
- **A separate page** accessible from the sidebar
- **A collapsible panel** at the bottom of the page

**Recommendation: New tab approach.** Each engine already has a tab in the UI. Add `qwen36` as a tab with chat UI instead of audio controls. The `_build_params()` function in `tts_lab_ui.py` handles engine-specific UI — add a chat interface for `name == "qwen36"`.

---

## 9. Implementation Plan

### Phase 1: Container & Model (Day 1)

| Step | Task | Files |
|------|------|-------|
| 1.1 | Download GGUF model to VM | `/opt/models/llm/qwen3.6-35b-a3b-tq3_4s.gguf` |
| 1.2 | Create `Dockerfile.llm-qwen36` | `docker/Dockerfile.llm-qwen36` |
| 1.3 | Build and test container locally | — |
| 1.4 | Verify llama-server starts, test with curl | — |
| 1.5 | Add `llm-qwen36` service to `docker-compose.yml` (profile: `llm`) | `docker-compose.yml` |

### Phase 2: Orchestrator Integration (Day 1-2)

| Step | Task | Files |
|------|------|-------|
| 2.1 | Add `MODEL_INFO["qwen36"]` entry | `tts_lab_config.py` |
| 2.2 | Add to `MODEL_ORDER` | `tts_lab_config.py` |
| 2.3 | Add `_load_qwen36()` and `_synth_qwen36()` | `tts_lab_engines.py` |
| 2.4 | Register in `LOADERS`/`SYNTHERS` dicts | `tts_lab_engines.py` |
| 2.5 | Handle text responses in dispatch layer | `tts_lab_dispatch.py` |
| 2.6 | Add `QWEN36_URL` to orchestrator env vars | `docker-compose.yml`, `Makefile` |

### Phase 3: UI (Day 2-3)

| Step | Task | Files |
|------|------|-------|
| 3.1 | Add `if name == "qwen36":` block for chat interface | `tts_lab_ui.py` |
| 3.2 | Build chat panel HTML/JS (message list, input, send) | `tts_lab_ui.py` |
| 3.3 | Handle streaming vs. non-streaming responses | `tts_lab_ui.py` |
| 3.4 | Add model/temperature/max_tokens controls | `tts_lab_ui.py` |
| 3.5 | Style chat bubbles, code blocks (syntax highlighting?) | `tts_lab_ui.py` |

### Phase 4: Documentation & Polish (Day 3)

| Step | Task | Files |
|------|------|-------|
| 4.1 | Add engine entry to `engine_compatibility.yaml` | `docs/engine_compatibility.yaml` |
| 4.2 | Update `CLAUDE.md` with LLM engine info | `CLAUDE.md` |
| 4.3 | Add Makefile targets (`build-llm`, `deploy-llm`) | `Makefile` |
| 4.4 | Test VRAM coexistence scenarios | — |
| 4.5 | E2E test: prompt → response in UI | — |

---

## 10. Dockerfile Specification

### `docker/Dockerfile.llm-qwen36`

```dockerfile
# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
# TIER 3 — llm-qwen36 (Qwen 3.6 LLM on CUDA 12.8 + llama.cpp)
# ═══════════════════════════════════════════════════════════════════
#
# Standalone container — does NOT inherit from tts-lab-base.
# llama.cpp has zero dependency overlap with TTS stacks.
#
# SIZE: ~3 GB (image) + ~13 GB (model, mounted)
# GPU: Required (CUDA 12.8)
# VRAM: ~13 GB (model + KV cache)
# ═══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="TTS Lab — Qwen 3.6 LLM"
LABEL tts-lab.tier="3-engine"
LABEL tts-lab.stack="llm"
LABEL tts-lab.gpu="required"

# ── Build deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Build llama.cpp with CUDA ────────────────────────────────────
# Pin to a known-good commit; update periodically
ARG LLAMA_CPP_VERSION=master
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120" \
      -DGGML_CUDA_F16=ON \
      -DGGML_CUDA_FA=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/

# ── Runtime deps only ────────────────────────────────────────────
RUN apt-get remove -y build-essential cmake git && \
    apt-get autoremove -y && \
    apt-get clean

# ── Health check tool ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# ── Model directory (mounted at runtime) ─────────────────────────
RUN mkdir -p /opt/models/llm

ENV PYTHONUNBUFFERED=1
WORKDIR /opt/llama.cpp
EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8006/health || exit 1

# Default: Qwen3.6-35B-A3B TQ3_4S
# Override MODEL_PATH env var for different model/quant
ENV MODEL_PATH=/opt/models/llm/qwen3.6-35b-a3b-tq3_4s.gguf
ENV CTX_SIZE=4096

CMD llama-server \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8006 \
    --n-gpu-layers 99 \
    --ctx-size ${CTX_SIZE} \
    --parallel 1 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --jinja \
    --threads 4 \
    --threads-batch 8
```

---

## 11. Trade-offs & Risks

### 11.1 — VRAM Contention (Primary Risk)

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM + any TTS engine > 4 GB = OOM | Crashes, lost work | Profile-based isolation. Document clearly. |
| 27B IQ4_XS leaves < 1.5 GB free | Can't run even light TTS | Stick to 35B-A3B for daily use |
| Multiple users (future) | VRAM exhaustion | Single-user by design; add queue if needed |

### 11.2 — Model Quality Trade-offs

| Trade-off | 35B-A3B (MoE) | 27B Dense |
|-----------|:-------------:|:---------:|
| Coding (SWE-bench) | 73.4 | **77.2** |
| Speed | **107 tok/s** | 43 tok/s |
| VRAM | **12.4 GB** | 14.5 GB |
| Context capacity | 4K-8K | 2K-4K |
| MoE routing quirks | Occasional weird routing | Consistent dense output |

### 11.3 — Not a TTS Engine

The LLM returns text, not audio. This breaks the assumption that all engines return WAV bytes. The dispatch layer needs to distinguish text engines from audio engines. Options:
1. Use `response_type` field in MODEL_INFO
2. Check if response is bytes vs string in dispatch
3. Add `engine_type: "tts" | "llm"` to engine metadata

**Recommendation: Add `engine_type` to MODEL_INFO.** Clean and extensible for future non-TTS engines.

### 11.4 — llama.cpp Build Time

Building llama.cpp from source takes ~5-10 minutes on the VM. Consider:
- Pre-built binaries in the image (current approach)
- CI/CD pipeline for weekly rebuilds against latest llama.cpp

### 11.5 — Model Updates

Qwen team releases models regularly. The 3.6 series may be superseded. Design for easy model swapping:
- `MODEL_PATH` env var overrides the default GGUF file
- `CTX_SIZE` env var adjusts context window
- Multiple model files can coexist in `/opt/models/llm/`

---

## 12. Cost & Time Estimate

| Phase | Effort | Calendar Time |
|-------|--------|---------------|
| Phase 1: Container & Model | 2-3 hours | 1 day |
| Phase 2: Orchestrator Integration | 2-3 hours | 1 day |
| Phase 3: UI (Chat Interface) | 4-6 hours | 1-2 days |
| Phase 4: Documentation & Polish | 1-2 hours | 1 day |
| **Total** | **9-14 hours** | **3-4 days** |

Model download: ~12.4 GB, ~10-20 minutes on the VM's connection.

---

## 13. Open Questions

1. **Chat history persistence?** Should conversations be saved? Where? (localStorage vs server-side)
2. **Streaming responses?** llama-server supports SSE streaming. Worth the UI complexity?
3. **System prompt customization?** Should users set their own system prompt per conversation?
4. **Multiple model support?** Should we support switching between 35B-A3B, 27B, and 14B from the UI?
5. **Thinking/reasoning mode?** Qwen 3.6 has a `preserve_thinking` flag for multi-turn reasoning. Expose in UI?
6. **Should we add a dedicated TTS→speech path?** Pipe LLM output through a TTS engine for spoken responses?

---

## 14. Appendix: Model Download Commands

```bash
# On the VM (arthur@192.168.0.87)
mkdir -p /opt/models/llm

# Primary: Qwen3.6-35B-A3B TQ3_4S (~12.4 GB)
huggingface-cli download YTan2000/Qwen3.6-35B-A3B-TQ3_4S \
  --local-dir /opt/models/llm/ \
  --local-dir-use-symlinks False

# Alternative: Qwen3.6-27B IQ4_XS pure (~14.5 GB)
huggingface-cli download Ununnilium/Qwen3.6-27B-IQ4_XS-pure-GGUF \
  --local-dir /opt/models/llm/ \
  --local-dir-use-symlinks False

# Fallback: Qwen3.6-14B Q4_K_M (~9 GB)
huggingface-cli download bartowski/Qwen3.6-14B-GGUF \
  --include "*Q4_K_M*" \
  --local-dir /opt/models/llm/ \
  --local-dir-use-symlinks False
```

---

## 15. Appendix: Quick Test Commands

```bash
# Build the container
docker build -f docker/Dockerfile.llm-qwen36 -t tts-lab-llm-qwen36:latest .

# Run interactively (test)
docker run --rm -it --gpus all --network host \
  -v /opt/models:/opt/models \
  tts-lab-llm-qwen36:latest

# Test chat completion
curl http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a linked list."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'

# Test health
curl http://localhost:8006/health

# Check VRAM
nvidia-smi
```
