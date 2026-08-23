# Arthur TTS Lab — Benchmark Results 2026-08-23 (EditX + S2-Pro)

> VM: arthur@192.168.0.87 · RTX 5060 Ti 16 GB GDDR7 (sm_120) · container-only mode (orchestrator 8009)
> All runs orchestrator-routed (`POST /synthesize/<engine>`), engine containers as deployed 2026-08-23.
> RTF = total synth time ÷ audio duration (includes load on first runs; per-test footnotes).
> Both engines are **HEAVY** — only one can be resident at a time (orchestrator stop/start protocol).

---

## Step Audio EditX — AWQ-4bit (`editx`, port 8105)

**Stack:** torch 2.13.0+cu130 (stable) · vllm 0.26.1rc1.dev1117 nightly wheel · transformers 4.57 · python 3.12
**VRAM resident:** ~12.8 GiB (measured 12,643 MiB post-rebuild re-verify; card total 15.8 GiB)
**Output:** 24 kHz WAV

| Run | Load | Audio dur | Synth ms | RTF | Notes |
|---|---|---|---|---|---|
| First synth (2026-08-23 03:32) | 174.4 s | 7.2 s | 188,732 | 26.2× | load-dominated; generate alone ~5 s |
| emotion (`happy`) | — | 8.0 s | 18,200 | **2.26×** | resident |
| style (`narrator`) | — | 6.2 s | 12,035 | **1.93×** | resident |
| speed (1.5×) | — | 10.4 s | 17,762 | **1.70×** | resident |
| clone + `ref_text` (en-leo) | — | 6.5 s | 10,783 | **1.66×** | resident; sane output |
| clone w/o `ref_text` (en-leo2) | — | **62.0 s** ⚠️ | 26,866 | 0.43× | degenerate ramble — fallback target-text transcript |
| Post-rebuild re-verify | 174 s (reload) | 7.2 s | 445,552 | 61.9× | includes orchestrator stopping s2pro first |

**Resident generate speed: ~5 s for ~7 s audio ≈ RTF 0.7×** — the model itself is fast; RTF >1 on
single runs is load-dominated (174 s engine load).

---

## Fish S2-Pro — 5B Dual-AR (`s2pro`, port 8005, sglang-omni)

**Stack:** sglang-omni 0.1.3 (`sgl-omni serve`) · torch nightly cu130 · flashinfer 0.6.x JIT (sm_120)
**VRAM:** ~11.75 GB pipeline; peak 15,540 MiB; measured **14,428 MiB** during clone run (vocoder codec on CPU)
**Output:** 44.1 kHz WAV

| Run | Load | Audio dur | Synth ms | RTF | Notes |
|---|---|---|---|---|---|
| First synth (2026-08-22) | 182 s | 3.3 s | ~182,000+ | ~55× | flashinfer decode kernels JIT-compiled on first request |
| Steady state (resident) | — | 3.3 s | **4,600** | ~1.4× | per-step latency from CPU-vocoder IPC |
| Clone + transcript (2026-08-23, en-leo) | container start + model load | 3.8 s | 116,478 | 30.6× | full orchestrator stop/start round-trip (editx evicted first) |

---

## Coexistence / eviction (measured)

| Resident combo | VRAM | Result |
|---|---|---|
| editx alone | 12,643 MiB | ✅ (3.2 GiB free) |
| s2pro alone (peak) | 15,540 MiB | ✅ tight |
| editx + s2pro | ~24.5 GiB | ❌ OOM — orchestrator stop/start protocol handles (verified round-trip 2026-08-23) |
| LLM qwen36 (retired 2026-08-23) | 13.6 GiB | never coexisted; container + images removed |

**Engine-load times (cold):** editx 174 s · s2pro ~90–120 s (JIT cached at `/opt/models/flashinfer-editx-jit` / `/opt/models/flashinfer-jit`).
