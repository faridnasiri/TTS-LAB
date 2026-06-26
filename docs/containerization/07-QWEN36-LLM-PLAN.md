# Qwen 3.6 LLM Integration Plan — Reasoning & Programming Engine

> **Status:** DEPLOYED — 2026-06-26
> **Date:** 2026-06-25 (plan) / 2026-06-26 (deployment)
> **Author:** Claude (research + planning + implementation)
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

### 4.1 — Container Topology (Full Lab + LLM)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              HOST: RTX 5060 Ti 16 GB VRAM                             │
│                              arthur@192.168.0.87 :8001                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
┌──────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐
│     orchestrator         │ │    engine-current         │ │    engine-qwen            │
│     (port 8001)          │ │    (port 8101)            │ │    (port 8104)            │
│──────────────────────────│ │──────────────────────────│ │──────────────────────────│
│ Base: tts-lab-base       │ │ Base: stack-current       │ │ Base: stack-mid           │
│ Stack: none (no ML)      │ │ Stack: current            │ │ Stack: mid                │
│──────────────────────────│ │──────────────────────────│ │──────────────────────────│
│ Dependencies:            │ │ torch 2.12 nightly        │ │ torch 2.12 nightly        │
│   fastapi, uvicorn       │ │ transformers 5.12.1        │ │ transformers 4.51.3        │
│   httpx, soundfile       │ │ CUDA 12.8                 │ │ CUDA 12.8                 │
│──────────────────────────│ │──────────────────────────│ │──────────────────────────│
│ VRAM: ~0 MB (no GPU)     │ │ VRAM: 0-12 GB (lazy)      │ │ VRAM: 0-6 GB (lazy)       │
│                          │ │ Engines: 21 TTS            │ │ Engines: 1 (Qwen3TTS)     │
│ Routes to ALL engines    │ │  piper, kokoro, melo,      │ │  gated: Qwen/Qwen3-TTS     │
│ via {ENGINE}_URL env vars│ │  matcha, chattts, bark,    │ │                            │
│                          │ │  styletts2, f5tts, dia,    │ │                            │
│                          │ │  chatterbox, zonos, ...    │ │                            │
└──────────┬───────────────┘ └──────────────────────────┘ └──────────────────────────┘
           │
           ├──────────────────────────────────────────────────────────────────┐
           │                     │                     │                       │
           ▼                     ▼                     ▼                       ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   engine-mid          │ │   engine-legacy       │ │   orpheus             │ │  ★ llm-qwen36  (NEW) │
│   (port 8103)         │ │   (port 8102)         │ │   (port 8002)         │ │   (port 8006)         │
│───────────────────────│ │───────────────────────│ │───────────────────────│ │───────────────────────│
│ Base: stack-mid       │ │ Base: stack-legacy    │ │ Base: CUDA 12.1       │ │ Base: CUDA 12.8       │
│ Stack: mid            │ │ Stack: legacy         │ │ Stack: standalone     │ │ Stack: standalone     │
│───────────────────────│ │───────────────────────│ │───────────────────────│ │───────────────────────│
│ torch 2.12 nightly    │ │ torch 1.13             │ │ vllm + CUDA 12.1      │ │ llama.cpp (source)    │
│ transformers 4.51.3   │ │ transformers 4.46      │ │ numpy >= 2.0           │ │ CUDA 12.8             │
│ CUDA 12.8             │ │ CUDA 11.7              │ │ protobuf >= 5.0        │ │ GPU: sm_120           │
│───────────────────────│ │───────────────────────│ │───────────────────────│ │───────────────────────│
│ VRAM: 0-7 GB (lazy)   │ │ VRAM: 0-4 GB (lazy)   │ │ VRAM: ~6 GB (fixed)    │ │ VRAM: ~13 GB (fixed)  │
│ Engines: 2             │ │ Engines: 2             │ │ Engines: 1             │ │ Engines: 1 (LLM)      │
│  VibeVoice, Higgs      │ │  IndexTTS, Parler      │ │  orpheus-3b             │ │  qwen3.6-35b-a3b      │
│ Profile: mid           │ │ Profile: legacy        │ │ Profile: gpu            │ │ Profile: llm           │
│ Status: experimental   │ │ Status: BLOCKED        │ │ Status: BLOCKED         │ │ Status: PLANNED        │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘ └───────────────────────┘

                    ┌─────────────────────────────────────────────────┐
                    │          SGLang Instances (profile: sglang)     │
                    │  All share tts-lab-sglang image                 │
                    ├──────────────┬──────────────┬───────────────────┤
                    │  vibevoice   │   higgs      │   s2pro           │
                    │  port 8003   │   port 8004  │   port 8005       │
                    │  ~7 GB VRAM  │   ~9 GB VRAM │   ~11 GB VRAM     │
                    │  EXPERIMENTAL│   EXPERIMENTAL│   BLOCKED        │
                    └──────────────┴──────────────┴───────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         VRAM COEXISTENCE MATRIX (16 GB total)                         │
├────────────────────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│                        │ LLM idle │ LLM 35B  │ LLM 27B  │ LLM 14B  │                 │
│                        │ (0 GB)   │(12.4 GB) │(14.5 GB) │ (9 GB)   │                 │
├────────────────────────┼──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ No TTS loaded          │    ✅    │    ✅    │  ✅ tight│    ✅    │                 │
│ Light TTS (0.2 GB)     │    ✅    │    ✅    │  ✅ tight│    ✅    │ kokoro, piper    │
│ Medium TTS (2 GB)      │    ✅    │  ✅ tight│    ❌    │    ✅    │ chattts, omnivoice│
│ Heavy TTS (4-6 GB)     │    ✅    │    ❌    │    ❌    │  ✅ tight│ dia, fishspeech   │
│ Qwen3TTS (6 GB)        │    ✅    │    ❌    │    ❌    │  ✅ tight│ engine-qwen       │
│ Bark (12 GB)           │    ✅    │    ❌    │    ❌    │    ❌    │ OOM with any LLM │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┴─────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              NETWORK FLOW (key paths)                                 │
└──────────────────────────────────────────────────────────────────────────────────────┘

  ── TTS Synthesis (unchanged) ──────────────────────────────────────────────────────────
  Browser ──POST /synthesize/piper──▶ orchestrator:8001 ──POST /synthesize──▶ engine-current:8101
  Browser ──POST /synthesize/qwen3tts▶ orchestrator:8001 ──POST /synthesize──▶ engine-qwen:8104

  ── LLM Synthesis with Global Eviction (NEW) ───────────────────────────────────────────
  Browser ──POST /synthesize/qwen36──▶ orchestrator:8001
                                          │
                                          │ ★ Phase 1: EVICT ALL TTS ★
                                          ├──POST /evict──▶ engine-current:8101    (evicts e.g. chattts)
                                          ├──POST /evict──▶ engine-qwen:8104       (evicts e.g. qwen3tts)
                                          ├──POST /evict──▶ engine-mid:8103        (evicts if loaded)
                                          └──POST /evict──▶ engine-legacy:8102     (evicts if loaded)
                                          │
                                          │ ★ Phase 2: VERIFY — all evicted ★
                                          │
                                          │ ★ Phase 3: ROUTE TO LLM ★
                                          └──POST /v1/chat/completions──▶ llm-qwen36:8006
                                                                              │
                                                                              │ llama.cpp inference
                                                                              │ 12.4 GB VRAM used
                                                                              │ 3.6 GB free
                                                                              │
  Browser ◀── JSON text response ──────────│◀──────────────────────────────────┘

  ── TTS engines stay evicted. Next TTS synthesis request triggers lazy reload. ─────────

  Docker network: tts-lab-net (bridge) — all containers communicate by service name
  Host network: used in Makefile deploy targets (--network host) — direct localhost ports
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

### 5.2 — Coexistence Strategy: Global VRAM Eviction (PRIMARY)

**The LLM must load into 100% clean VRAM — zero TTS models resident.** This is non-negotiable
because the 35B-A3B GGUF (~12.4 GB) + any medium TTS engine (>2 GB) exceeds the 16 GB budget.

The mechanism: **orchestrator-coordinated cross-container eviction**. When a synthesis request
arrives for `qwen36`, the orchestrator sends `POST /evict` to EVERY engine container BEFORE
routing to the LLM. Each engine container already has `_evict_current()` — we simply expose it
as an HTTP endpoint.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    GLOBAL VRAM EVICTION — SEQUENCE DIAGRAM                             │
└──────────────────────────────────────────────────────────────────────────────────────┘

  User                Orchestrator           engine-current        engine-qwen         llm-qwen36
  │                       │                       │                    │                   │
  │ POST /synthesize/qwen36                       │                    │                   │
  │──────────────────────▶│                       │                    │                   │
  │                       │                       │                    │                   │
  │                       │  ★ Phase 1: EVICT ALL TTS ENGINES ★                        │
  │                       │                       │                    │                   │
  │                       │ POST /evict (timeout 10s)                   │                   │
  │                       │──────────────────────▶│                    │                   │
  │                       │                       │ _evict_current()   │                   │
  │                       │                       │ torch.cuda.empty   │                   │
  │                       │                       │ gc.collect()       │                   │
  │                       │  {"evicted":"chattts",│                    │                   │
  │                       │   "vram_free_mb":15800}                    │                   │
  │                       │◀──────────────────────│                    │                   │
  │                       │                       │                    │                   │
  │                       │ POST /evict (timeout 10s)                   │                   │
  │                       │────────────────────────────────────────────▶│                   │
  │                       │                       │                    │ _evict_current()  │
  │                       │  {"evicted":"qwen3tts",                    │                   │
  │                       │   "vram_free_mb":15800}                    │                   │
  │                       │◀────────────────────────────────────────────│                   │
  │                       │                       │                    │                   │
  │                       │  ★ Phase 2: VERIFY — all engines evicted ★                    │
  │                       │  All TTS engines report "nothing loaded" or evicted OK        │
  │                       │  Total VRAM free across all TTS containers: ~15.8 GB          │
  │                       │                       │                    │                   │
  │                       │  ★ Phase 3: ROUTE TO LLM ★                                    │
  │                       │                                                              │
  │                       │ POST /v1/chat/completions                                     │
  │                       │─────────────────────────────────────────────────────────────▶│
  │                       │                                                              │
  │                       │  llama.cpp inference (12.4 GB VRAM used, 3.6 GB free)         │
  │                       │                                                              │
  │                       │  {"choices":[{"message":{"content":"def quicksort..."}}]}     │
  │                       │◀─────────────────────────────────────────────────────────────│
  │                       │                                                              │
  │  {"text":"def quicksort..."}                                                          │
  │◀──────────────────────│                                                              │
  │                       │                                                              │
  │  ★ TTS engines stay evicted — reload lazily on next TTS synthesis request ★          │
  │                       │                                                              │
```

### 5.3 — Implementation: New `/evict` Endpoint on Engine Containers

Each engine container (`tts_lab_engine_server.py`) already has `_evict_current()` (line 77).
We expose it as an HTTP endpoint. **This is ~15 lines of code.**

**Added to `tts_lab_engine_server.py`:**

```python
class EvictResponse(BaseModel):
    evicted: bool
    engine_was: str | None = None
    vram_free_mb: int = 0
    vram_total_mb: int = 0

@app.post("/evict", response_model=EvictResponse)
async def evict():
    """Evict the currently loaded engine. Called by orchestrator before LLM loads."""
    global _current_engine
    was = _current_engine
    _evict_current()
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        free_mb = free // 1048576
        total_mb = total // 1048576
    except Exception:
        free_mb, total_mb = 0, 0
    return EvictResponse(
        evicted=was is not None,
        engine_was=was,
        vram_free_mb=free_mb,
        vram_total_mb=total_mb,
    )
```

### 5.4 — Implementation: Orchestrator Pre-Dispatch Eviction Hook

The orchestrator needs a **global eviction function** that fires before LLM dispatch.
This lives in `tts_lab_dispatch.py`.

**Added to `tts_lab_dispatch.py`:**

```python
# ── Engine container URL registry (populated by _build_remote_urls) ──
_ENGINE_CONTAINER_URLS: set[str] = set()

def _build_remote_urls():
    global _ENGINE_CONTAINER_URLS
    _REMOTE_ENGINES.clear()
    _ENGINE_CONTAINER_URLS.clear()
    for key, val in os.environ.items():
        if key.endswith("_URL") and not key.endswith("_SGLANG_URL"):
            engine_name = key[:-4].lower()
            _REMOTE_ENGINES[engine_name] = val
            _ENGINE_CONTAINER_URLS.add(val.rstrip("/"))


async def _evict_all_tts_engines(http_client) -> dict[str, dict]:
    """POST /evict to every known engine container. Returns per-URL results.
    Called before LLM synthesis to guarantee 100% clean VRAM."""
    results = {}
    for base_url in _ENGINE_CONTAINER_URLS:
        try:
            evict_url = f"{base_url}/evict"
            resp = await http_client.post(evict_url, timeout=10.0)
            results[base_url] = resp.json() if resp.status_code == 200 else {"error": resp.text}
        except Exception as e:
            results[base_url] = {"error": str(e)}
    return results


async def _do_synth_qwen36(name: str, text: str, params: dict):
    """LLM synthesis — evicts all TTS engines first, then dispatches to llama.cpp."""
    import httpx
    async with httpx.AsyncClient() as client:
        # ★ Phase 1: Evict ALL TTS engines from VRAM ★
        evict_results = await _evict_all_tts_engines(client)
        slog(f"[dispatch] Global eviction results: {evict_results}")

        # ★ Phase 2: Route to LLM ★
        llm_url = _REMOTE_ENGINES.get("qwen36", "http://llm-qwen36:8006")
        payload = {
            "model": params.get("model", "qwen3.6"),
            "messages": [
                {"role": "system", "content": params.get("system_prompt", "You are a helpful assistant.")},
                {"role": "user", "content": text},
            ],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 2048),
            "top_p": params.get("top_p", 0.9),
        }
        resp = await client.post(
            f"{llm_url}/v1/chat/completions",
            json=payload,
            timeout=300.0,
        )
        data = resp.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "tokens": data.get("usage", {}).get("total_tokens", 0),
            "model": data.get("model", ""),
        }
```

### 5.5 — Eviction Failure Handling

| Scenario | Behavior |
|----------|----------|
| Engine container unreachable | Log warning, continue — container may be down/stopped |
| Engine container returns error | Log error, continue — best-effort eviction |
| All evictions succeed | LLM loads into ~15.8 GB free VRAM |
| SGLang containers running | Must be **manually stopped** (`docker compose --profile sglang down`) — SGLang doesn't have `/evict` endpoint |
| Orpheus container running | Must be **manually stopped** — blocked anyway |

**SGLang note:** The SGLang containers (vibevoice, higgs, s2pro) don't run `tts_lab_engine_server.py`
and don't have an eviction endpoint. If SGLang services are running, they must be stopped manually
before LLM use. The orchestrator will log a warning if SGLang URLs are configured but unreachable
during eviction.

### 5.6 — Container Lifecycle (Simplified)

```bash
# Normal TTS operation — LLM container can be running (idle):
docker compose up -d                          # orchestrator + engine-current + engine-qwen
docker compose --profile llm up -d            # + LLM container (llama-server running, no model loaded yet)

# First LLM request triggers eviction → llama.cpp loads model into VRAM
# After LLM response: model stays loaded until LLM container is stopped

# Free VRAM for heavy TTS work:
docker compose --profile llm down             # Stop LLM container → VRAM fully freed

# Or keep LLM loaded, TTS engines reload lazily when needed (light engines only):
# kokoro/piper (0.2 GB) can reload alongside LLM (12.4 + 0.2 = 12.6 GB, fits)
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

### 7.4 — Dispatch Flow (with Global Eviction)

```
User types prompt in UI → POST /synthesize/qwen36
  │
  │  ┌─────────────────────────────────────────────────────────────┐
  │  │           Orchestrator: _do_synth_qwen36()                   │
  │  │                                                              │
  │  │  ★ Phase 1: GLOBAL EVICTION ★                                │
  │  │  _evict_all_tts_engines(http_client)                         │
  │  │    ├─ POST /evict → engine-current:8101   "chattts → freed"  │
  │  │    ├─ POST /evict → engine-qwen:8104      "qwen3tts → freed" │
  │  │    ├─ POST /evict → engine-mid:8103       "nothing loaded"   │
  │  │    └─ POST /evict → engine-legacy:8102    "nothing loaded"   │
  │  │                                                              │
  │  │  ★ Phase 2: VERIFY ★                                        │
  │  │  All engines report evicted — VRAM: ~15.8 GB free            │
  │  │                                                              │
  │  │  ★ Phase 3: ROUTE TO LLM ★                                   │
  │  │  POST /v1/chat/completions → llm-qwen36:8006                 │
  │  │  Payload: {model, messages, temperature, max_tokens}         │
  │  │                                                              │
  │  └─────────────────────────────────────────────────────────────┘
  │
  ▼
llama-server loads model → GPU inference (12.4 GB VRAM, ~107 tok/s)
  │
  ▼
Response: {"choices":[{"message":{"content":"def quicksort..."}}], "usage":{...}}
  │
  ▼
Orchestrator returns: {"text": "def quicksort...", "tokens": 150, "model": "qwen3.6"}
  │
  ▼
UI renders in chat panel (NOT audio player — text-only response)

TTS engines stay evicted. Next TTS synthesis request triggers lazy reload automatically.
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

### Phase 2: Global VRAM Eviction Mechanism (Day 1-2) ⭐ CRITICAL PATH

| Step | Task | Files |
|------|------|-------|
| **2.1** | **Add `POST /evict` endpoint to engine server** | `tts_lab_engine_server.py` |
| | — Exposes existing `_evict_current()` (line 77) as HTTP endpoint | |
| | — Returns `{evicted, engine_was, vram_free_mb, vram_total_mb}` | |
| | — ~15 lines of code | |
| **2.2** | **Add `_evict_all_tts_engines()` to dispatch** | `tts_lab_dispatch.py` |
| | — Collects unique engine container URLs from `_ENGINE_CONTAINER_URLS` | |
| | — `POST /evict` to each container (async, 10s timeout) | |
| | — Returns per-URL results dict | |
| **2.3** | **Add `_do_synth_qwen36()` to dispatch** | `tts_lab_dispatch.py` |
| | — Calls `_evict_all_tts_engines()` before LLM dispatch | |
| | — Routes to `llm-qwen36:8006/v1/chat/completions` | |
| | — Returns `{text, tokens, model}` | |
| **2.4** | **Test eviction + LLM load sequence** | — |
| | — Load a TTS engine in engine-current, verify VRAM usage | |
| | — Send LLM request, verify eviction fires | |
| | — Verify VRAM is clean before llama.cpp loads | |
| | — Verify TTS engine reloads lazily on next TTS request | |

### Phase 3: Engine Registration & Config (Day 2)

| Step | Task | Files |
|------|------|-------|
| 3.1 | Add `MODEL_INFO["qwen36"]` entry with `engine_type: "llm"` | `tts_lab_config.py` |
| 3.2 | Add to `MODEL_ORDER` | `tts_lab_config.py` |
| 3.3 | Add `_load_qwen36()` (returns URL dict) and `_synth_qwen36()` (delegates to dispatch) | `tts_lab_engines.py` |
| 3.4 | Register in `LOADERS`/`SYNTHERS` dicts | `tts_lab_engines.py` |
| 3.5 | Add `QWEN36_URL` to orchestrator env vars | `docker-compose.yml`, `Makefile` |

### Phase 4: UI — Chat Interface (Day 2-3)

| Step | Task | Files |
|------|------|-------|
| 4.1 | Add `if name == "qwen36":` block for chat interface | `tts_lab_ui.py` |
| 4.2 | Build chat panel HTML/JS (message list, input, send) | `tts_lab_ui.py` |
| 4.3 | Add model/temperature/max_tokens/system_prompt controls | `tts_lab_ui.py` |
| 4.4 | Style chat bubbles, code blocks with syntax highlighting | `tts_lab_ui.py` |
| 4.5 | Add "LLM engine — text response" indicator (no audio player) | `tts_lab_ui.py` |

### Phase 5: Documentation & Polish (Day 3)

| Step | Task | Files |
|------|------|-------|
| 5.1 | Add engine entry to `engine_compatibility.yaml` | `docs/engine_compatibility.yaml` |
| 5.2 | Update `CLAUDE.md` with LLM engine info + eviction flow | `CLAUDE.md` |
| 5.3 | Add Makefile targets (`build-llm`, `deploy-llm`) | `Makefile` |
| 5.4 | Test VRAM coexistence scenarios (all combinations from matrix) | — |
| 5.5 | E2E test: TTS loaded → LLM request → eviction → text response → TTS reload | — |

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
| Phase 2: Global VRAM Eviction ⭐ | 2-3 hours | 1 day |
| Phase 3: Engine Registration & Config | 1-2 hours | 0.5 day |
| Phase 4: UI (Chat Interface) | 4-6 hours | 1-2 days |
| Phase 5: Documentation & Polish | 1-2 hours | 0.5 day |
| **Total** | **10-16 hours** | **3-4 days** |

Model download: ~12.4 GB, ~10-20 minutes on the VM's connection.

**Critical path:** Phase 2 (eviction) must be complete before end-to-end testing. The `/evict`
endpoint is the linchpin — without it, VRAM collisions will cause OOM crashes.

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


---
---

## 16. ACTUAL DEPLOYMENT — What Shipped (2026-06-26)

### Model Delivered

| Field | Planned | Actual | Why Changed |
|-------|---------|--------|-------------|
| Model | Qwen3.6-35B-A3B (MoE) | **Qwen3.6-27B (Dense)** | TQ3_4S quantization (ggml type 46) not supported by pre-built llama.cpp |
| Quant | TQ3_4S | **Q3_K_M** | Standard quantization, works with pre-built images |
| Source | YTan2000/Qwen3.6-35B-A3B-TQ3_4S | **batiai/Qwen3.6-27B-GGUF** | Public repo, no auth needed |
| File | ~12.4 GB | **~13 GB** | Similar size |
| Speed | ~107 tok/s (planned) | **~23 tok/s** | Dense 27B is slower than MoE 35B-A3B |
| SWE-bench | 73.4 (35B-A3B) | **77.2 (27B)** | 27B actually BETTER at coding |
| Context | 4096 (planned) | **32768** | 32K fits with q4_0 KV cache |
| Thinking | Not planned | **Enabled** | Qwen 3.6 has native thinking tokens (`<think>` tags) |

### Container

| Field | Planned | Actual | Why |
|-------|---------|--------|-----|
| Base image | Custom Dockerfile building llama.cpp from source | **Pre-built `ghcr.io/ggml-org/llama.cpp:server-cuda`** | Source build failed on CUDA linker error (`undefined reference to cuGetErrorString`) — pre-built image works immediately |
| Port | 8006 | **8006** | As planned |
| Network | Docker bridge | **Host network** (`--network host`) | Matches existing TTS container pattern (Makefile deploy) |
| GPU | `--gpus all` | **`--gpus all`** | As planned |

### Exact Deployment Command

```bash
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

### VRAM Budget (Actual)

```
RTX 5060 Ti — 16 GB (16311 MiB)

LLM model:        ~13 GB  (Q3_K_M, all 99 layers on GPU)
KV cache (32K):   ~1.5 GB (q4_0 quantized, 32768 tokens)
llama.cpp overhead: ~0.5 GB
─────────────────────────────────
LLM idle:         ~13.6 GB used, ~2.4 GB free

LLM + kokoro:     ~13.8 GB used, ~2.1 GB free  ✅ fits
LLM + chattts:    ~15.6 GB used, ~0.4 GB free  ⚠️ very tight
LLM + bark:       ~25.6 GB needed               ❌ OOM
```

### Port Map (Actual)

```
Host: arthur@192.168.0.87

8009 — TTS Lab Orchestrator (Web UI + HTTP dispatch)
8006 — Qwen 3.6 LLM (llama.cpp OpenAI-compatible API)
8101 — Engine-Current (21 TTS engines)
8104 — Engine-Qwen (Qwen3TTS)
8103 — Engine-Mid (VibeVoice, Higgs)
```

### Source Build Attempts (Failed)

Five Docker build attempts were made before switching to the pre-built image:

| Attempt | Issue | Fix Tried | Result |
|---------|-------|-----------|--------|
| 1 | 9 GPU architectures → 3+ hour build time | — | Cancelled (too slow) |
| 2 | `CMAKE_CUDA_ARCHITECTURES="120a-real"` invalid for CMake 3.22 | Single arch flag | Failed |
| 3 | Build reached 66% then `make: Error 2` (unnecessary targets) | `--target llama-server` | Failed |
| 4 | `undefined reference to cuGetErrorString` (linker) | Symlink stubs | Failed |
| 5 | Same linker error — stubs not in CUDA 12.8 devel image | — | Failed |

**Root cause:** NVIDIA removed `libcuda.so` stubs from CUDA 12.x devel images. llama.cpp HEAD requires CUDA Driver API symbols at link time. Pre-built images from ghcr.io are compiled in a CI environment that has the driver.

### End-to-End Test (Verified)

```
1. TTS synthesis (kokoro):         ✅ 200 OK, audio returned
2. LLM request (qwen36):           ✅ Global eviction fires → LLM responds
3. Reasoning visible:               ✅ thinking process in reasoning_content
4. TTS after LLM:                   ✅ kokoro reloads lazily, works alongside LLM
5. VRAM coexistence (LLM + kokoro): ✅ 13.8 GB used, both work
6. 32K context:                     ✅ Model loads with ctx-size 32768
```

### What's NOT Yet Done

| Item | Status | Notes |
|------|--------|-------|
| TQ3_4S MoE model support | ❌ Blocked | Requires llama.cpp built from source with CUDA driver stubs |
| Faster inference (107 tok/s) | ❌ Blocked | Blocked on MoE model above |
| Dockerfile.llm-qwen36 | ⚠️ Not used | Running pre-built image directly; Dockerfile kept as reference |
| SGLang/Orpheus coexistence | ⚠️ Manual | Must stop SGLang containers before LLM (no /evict endpoint on SGLang) |
| Chat history persistence | ❌ Not implemented | Conversations lost on page refresh |

# Test health
curl http://localhost:8006/health

# Check VRAM
nvidia-smi
```
