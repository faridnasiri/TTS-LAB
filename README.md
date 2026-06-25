# Arthur TTS Lab

> 28-engine TTS benchmark + 5-engine Image/Video lab | FastAPI | Docker 7-container | Bare-metal fallback
> **Deployed to:** `arthur@192.168.0.87:8001` | **GPU:** RTX 5060 Ti 16 GB GDDR7 (Blackwell sm_120)

A self-hosted, multi-engine Text-to-Speech benchmark and evaluation lab. Compare every major open-source TTS model side-by-side through a single web UI. Also includes an Image/Video generation lab (FLUX.2, SD 3.5, Ideogram 4, Wan2.2).

Originally built for an Android scam-baiting app ("SpamBlocker") that uses a character named Arthur Henderson as an AI decoy — hence the "Arthur" naming throughout.

---

## Features

- **28 TTS engines** — every major open-source TTS model, 16 validated + 8 experimental + 4 blocked
- **5 Image & Video engines** — FLUX.2, SD 3.5, Ideogram 4, Wan2.2, FLUX.2 Klein
- **Side-by-side comparison** — switch engines instantly, compare voices, measure quality
- **Voice cloning** — zero-shot cloning on 8 engines (F5-TTS, Chatterbox, Zonos, Fish Speech, StyleTTS2, XTTS, CosyVoice2, IndexTTS-2)
- **Voice Library** — browse, play, download Persian reference voices from Common Voice
- **RTF benchmarking** — automated Real-Time Factor measurement across all TTS engines
- **Persian text processing** — G2P, hazm, parsivar providers with live preview
- **NVFP4 native quantization** — Blackwell-optimized weights for FLUX.2 and Wan2.2
- **Containerized orchestration** — 7 Docker containers organized by dependency compatibility, not engine count
- **Engine maturity framework** — deterministic SUPPORTED/EXPERIMENTAL/BLOCKED/DEPRECATED lifecycle
- **Validation automation** — scripted gate tracking, auto-populated compatibility matrix, anti-drift mechanisms
- **Qwen 3.6 LLM** — optional reasoning & coding assistant via llama.cpp (GGUF Q3_K_S)

---

## Quick Start

```bash
# Default (orchestrator + engine-current + engine-qwen):
docker compose up -d

# + engine-mid (VibeVoice + Higgs):
docker compose --profile mid up -d

# + GPU-only engines (Orpheus):
docker compose --profile gpu up -d

# + SGLang engines (S2-Pro, VibeVoice-SGLang, Higgs-SGLang):
docker compose --profile sglang up -d

# + LLM (Qwen 3.6):
docker compose --profile llm up -d

# Everything:
docker compose --profile mid --profile gpu --profile sglang --profile llm --profile legacy up -d

# Open in browser:
# http://192.168.0.87:8001
```

---

## Architecture

### Design Principle: Compatibility-Domain

Containers are organized by **dependency compatibility boundary**, not engine count. An engine is placed in a container because it shares a torch+transformers+CUDA compatibility domain with other engines — not because it needs isolation.

```
Before (engine-centric):               After (compatibility-domain):

28 containers (one per engine)          7 containers (one per stack)
28 Dockerfiles                          7 Dockerfiles (6 custom + 1 pre-built)
28 build targets                        7 build targets
Duplicated ML stacks on disk            Base shared once, stacks shared
                                        → ~123 GB total images
```

### Container Topology

```
                          TTS-LAB Orchestrator (port 8001)
                          No ML libs — pure HTTP dispatch

        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼

┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  engine-current   │  │    engine-mid     │  │   engine-qwen     │
│  torch 2.14 n.    │  │  torch 2.12 n.    │  │  torch 2.12 n.    │
│  tf 5.12.1        │  │  tf 4.51.3        │  │  tf 4.57.3        │
│  CUDA 13.0        │  │  CUDA 12.8        │  │  CUDA 12.8        │
│  Python 3.10      │  │  Python 3.10      │  │  Python 3.10      │
│  port 8101        │  │  port 8103        │  │  port 8104        │
│  profile: default │  │  profile: mid     │  │  profile: default │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ 16 SUPP + 5 EXPER │  │ 2 EXPER           │  │ 1 SUPP            │
│                   │  │                   │  │                   │
│ piper      SUPP   │  │ vibevoice  EXPER  │  │ qwen3tts   SUPP   │
│ kokoro     SUPP   │  │ higgs      EXPER  │  │                   │
│ melo       SUPP   │  └───────────────────┘  └───────────────────┘
│ matcha     SUPP   │
│ chattts    SUPP   │  ┌───────────────────┐  ┌───────────────────┐
│ outetts    SUPP   │  │  engine-legacy    │  │     orpheus       │
│ bark       SUPP   │  │  torch 1.13       │  │  vllm + CUDA 12.1 │
│ styletts2  SUPP   │  │  tf 4.46          │  │  port 8002        │
│ f5tts      SUPP   │  │  CUDA 11.7        │  │  profile: gpu     │
│ dia        SUPP   │  │  port 8102        │  ├───────────────────┤
│ chatterbox SUPP   │  │  profile: legacy  │  │ orpheus    BLOCK  │
│ chatter-   SUPP   │  ├───────────────────┤  └───────────────────┘
│   boxturbo        │  │ indextts   BLOCK  │
│ fishspeech SUPP   │  │ parler     BLOCK  │  ┌───────────────────┐
│ omnivoice  SUPP   │  └───────────────────┘  │      s2pro        │
│ zonos      SUPP   │                         │  SGLang pre-built │
│ xtts       SUPP   │                         │  port 8005        │
│ cosyvoice  EXPER  │                         │  profile: sglang  │
│ csm        EXPER  │                         ├───────────────────┤
│ manatts    EXPER  │                         │ s2pro      BLOCK  │
│ neutts     EXPER  │                         └───────────────────┘
│ openvoice  EXPER  │
└───────────────────┘

┌───────────────────┐
│   llm-qwen36      │  (optional)
│   Qwen 3.6 35B    │
│   llama.cpp GGUF  │
│   Q3_K_S 4-bit    │
│   port 8006       │
│   profile: llm    │
└───────────────────┘
```

**7 containers — 6 custom + 1 pre-built (SGLang).** 4 deployed by default; 3 on-demand via profiles.

### Tiered Image Inheritance

```
                         nvidia/cuda:12.8.2-runtime-ubuntu22.04
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              tts-lab-base (7.5 GB)              (future: base-py311)
          System + Python utils + app code            CUDA 13 + Python 3.11
                    │
        ┌───────────┼───────────┐
        │           │           │
   stack-current  stack-mid  stack-legacy
   (16.2 GB)     (19.4 GB)   (not built)
        │           │
   engine-current engine-mid   engine-qwen
   (71.5 GB)     (20.7 GB)    (20.3 GB)
```

**Shared base stored once on disk.** Engine images only add their engine-specific packages (~55 GB for current, ~1 GB each for mid/qwen).

### Network Layout

```
                         Host: 192.168.0.87
                              │
                    ┌─────────┴─────────┐
                    │  Docker Bridge    │
                    │  172.18.0.0/16    │
                    │                   │
         ┌──────────┼──────────┬────────┼──────────┐
         │          │          │        │          │
    172.18.0.5  172.18.0.2  172.18.0.3  172.18.0.4  (future: .6,.7)
    orchestrator engine-     engine-     engine-
                current      qwen        mid
    Port 8001    Port 8101   Port 8104   Port 8103
    (external)   (internal)  (internal)  (internal)
```

### Orchestrator Routing

```
Browser (192.168.0.87:8001)
    │
    ▼
orchestrator:/synthesize/{engine}
    │
    ├── HTTP POST → engine-current:8101/synthesize   (21 engines)
    ├── HTTP POST → engine-qwen:8104/synthesize       (qwen3tts)
    ├── HTTP POST → engine-mid:8103/synthesize        (vibevoice, higgs)
    ├── HTTP POST → engine-legacy:8102/synthesize     (indextts, parler — not deployed)
    ├── HTTP POST → orpheus:8002/synthesize            (orpheus — not deployed)
    └── HTTP POST → s2pro:8000/v1/audio/speech        (s2pro — not deployed)
```

---

## Engine Distribution

### By Container

| Container | Total | Supported | Experimental | Blocked | Deployed |
|-----------|:-----:|:---------:|:------------:|:-------:|:--------:|
| `engine-current` | **21** | 16 | 5 | 0 | ✅ |
| `engine-mid` | **2** | 0 | 2 | 0 | ✅ (profile) |
| `engine-qwen` | **1** | 1 | 0 | 0 | ✅ |
| `engine-legacy` | **2** | 0 | 0 | 2 | ❌ |
| `orpheus` | **1** | 0 | 0 | 1 | ❌ |
| `s2pro` | **1** | 0 | 0 | 1 | ❌ |
| **Total** | **28** | **17** | **7** | **4** | 4/7 |

### Per-Container Engine List

**engine-current** — torch 2.14 nightly + tf 5.12.1, CUDA 13.0 (21 engines)

| # | Engine | Status | VRAM | RTF | Voice Clone | Notes |
|---|--------|:------:|:----:|:---:|:-----------:|-------|
| 1 | `piper` | SUPP | ~200 MB | 3.7× | — | ONNX CPU, real-time on any hardware |
| 2 | `kokoro` | SUPP | ~200 MB | 6.5× | — | 54 voices, 9 languages |
| 3 | `melo` | SUPP | ~1.4 GB | 9.8× | — | 5 English accents, MeCab+unidic |
| 4 | `matcha` | SUPP | ~200 MB | 8.0× | — | ONNX flow-matching, real-time |
| 5 | `chattts` | SUPP | ~2 GB | 11.5× | — | Speed prompts `[speed_N]`, speaker sampling |
| 6 | `outetts` | SUPP | ~2 GB | 34.9× | — | LLM-based, GGUF via llama.cpp |
| 7 | `bark` | SUPP | ~12 GB | 19.0× | — | Emotion tokens: `[laughs]` `[sighs]` |
| 8 | `styletts2` | SUPP | ~1.5 GB | 35.8× | ✅ | Style transfer from reference WAV |
| 9 | `f5tts` | SUPP | ~3 GB | 13.4× | ✅ | Best zero-shot voice cloning |
| 10 | `dia` | SUPP | ~4 GB | — | ✅ | Dialogue-native, `[S1]`/`[S2]` speakers |
| 11 | `chatterbox` | SUPP | ~2 GB | 25.2× | ✅ | Exaggeration slider + voice cloning |
| 12 | `chatterboxturbo` | SUPP | ~1.5 GB | 13.6× | — | One-step distilled, near real-time |
| 13 | `fishspeech` | SUPP | ~3 GB | 9.4× | ✅ | Zero-shot cloning |
| 14 | `omnivoice` | SUPP | ~2 GB | 4.5× | — | 600+ languages, real-time |
| 15 | `zonos` | SUPP | ~3 GB | 19.0× | ✅ | Emotion vector + speaking-rate, 44 kHz |
| 16 | `xtts` | SUPP | ~2 GB | 44.7× | ✅ | 58 speakers, 17 languages |
| 17 | `cosyvoice` | EXPER | ~3 GB | — | ✅ | Needs model download + hyperpyyaml |
| 18 | `csm` | EXPER | ~4 GB | — | — | Meta license gated, needs torchtune/torchao |
| 19 | `manatts` | EXPER | ~1.5 GB | — | — | Persian multi-speaker Tacotron2 |
| 20 | `neutts` | EXPER | ~1 GB | — | — | NeuTTS Air, not yet configured |
| 21 | `openvoice` | EXPER | ~3 GB | — | ✅ | MeloTTS base + tone-color conversion |

**engine-mid** — torch 2.12 nightly + tf 4.51.3, CUDA 12.8 (2 engines, profile: `mid`)

| # | Engine | Status | VRAM | Notes |
|---|--------|:------:|:----:|-------|
| 22 | `vibevoice` | EXPER | ~6.5 GB | POC pending — config_load gate failed (not in CONFIG_MAPPING) |
| 23 | `higgs` | EXPER | ~9 GB | POC pending — untested |

**engine-qwen** — torch 2.12 nightly + tf 4.57.3, CUDA 12.8 (1 engine)

| # | Engine | Status | VRAM | RTF | Voice Clone | Notes |
|---|--------|:------:|:----:|:---:|:-----------:|-------|
| 24 | `qwen3tts` | SUPP | ~4.2 GB | 4.7× | ✅ | Qwen3-TTS-12Hz-1.7B-Base. ICL + x-vector-only modes. 10 languages. HF_TOKEN required. |

**engine-legacy** — torch 1.13 + tf 4.46, CUDA 11.7 (2 engines, profile: `legacy`, not deployed)

| # | Engine | Status | Notes |
|---|--------|:------:|-------|
| 25 | `indextts` | BLOCK | Needs legacy stack build |
| 26 | `parler` | BLOCK | Needs legacy stack build |

**orpheus** — vllm + CUDA 12.1 (1 engine, profile: `gpu`, not deployed)

| # | Engine | Status | Notes |
|---|--------|:------:|-------|
| 27 | `orpheus` | BLOCK | vllm incompatible with torch nightly. Gated model (HF_TOKEN). |

**s2pro** — SGLang pre-built (1 engine, profile: `sglang`, not deployed)

| # | Engine | Status | Notes |
|---|--------|:------:|-------|
| 28 | `s2pro` | BLOCK | SGLang image tf 5.6.0 too old. Requires paged KV cache + RadixAttention. |

---

## Engine Maturity Classification

Every engine has one of four states, stored in [`docs/engine_compatibility.yaml`](docs/engine_compatibility.yaml) — the **machine-readable single source of truth**.

| State | Icon | Meaning | CI: Build | CI: Smoke Test | Required |
|-------|:----:|---------|:---------:|:--------------:|:--------:|
| **SUPPORTED** | ✅ | Synthesis confirmed on target hardware | Yes | Yes | Pass |
| **DEPRECATED** | ⚠ | Still works, no longer recommended | Yes | Yes | Warn |
| **EXPERIMENTAL** | 🧪 | Container defined, not yet validated | Yes | Best effort | Warn |
| **BLOCKED** | ❌ | Cannot work — upstream missing or incompatible | No | No | Skip |

### Lifecycle

```
EXPERIMENTAL  ──(all promotion gates pass)──▶  SUPPORTED
EXPERIMENTAL  ──(blocker found)─────────────▶  BLOCKED
BLOCKED       ──(upstream releases fix)─────▶  EXPERIMENTAL
SUPPORTED     ──(superseded by better)──────▶  DEPRECATED
DEPRECATED    ──(eventually breaks)─────────▶  BLOCKED
```

Promotion from EXPERIMENTAL to SUPPORTED is **deterministic** — every promotion gate must pass. No manual judgment. The [`update_engine_status.py`](scripts/utils/update_engine_status.py) script enforces this automatically.

---

## Docker Compose Profiles

| Profile | Containers Added | Engines | Command |
|:-------:|-----------------|---------|---------|
| *(default)* | orchestrator, engine-current, engine-qwen | 17 SUPP/EXPER | `docker compose up -d` |
| `mid` | + engine-mid | + 2 EXPER | `docker compose --profile mid up -d` |
| `gpu` | + orpheus | + 1 BLOCKED | `docker compose --profile gpu up -d` |
| `sglang` | + s2pro, vibevoice (SGLang), higgs (SGLang) | + 3 BLOCKED | `docker compose --profile sglang up -d` |
| `legacy` | + engine-legacy | + 2 BLOCKED | `docker compose --profile legacy up -d` |
| `llm` | + llm-qwen36 | Qwen 3.6 LLM | `docker compose --profile llm up -d` |
| *(all)* | All 7 + LLM | All 28 + LLM | `docker compose --profile mid --profile gpu --profile sglang --profile legacy --profile llm up -d` |

### Startup Order

```
Default start:
  engine-current ──(healthy)──┐
  engine-qwen    ──(healthy)──┼── orchestrator ──(port 8001)── ready

With --profile mid:
  engine-mid     ──(healthy)──┘
```

---

## Architecture — Two Modes

### Mode 1: Containerized / Orchestrator (primary)

**Compatibility-domain design** — containers organized by dependency boundaries:

```
Base (nvidia/cuda:12.8.2-runtime-ubuntu22.04)
  ├── Stack:current    torch 2.14 nightly + transformers 5.12.1 + CUDA 13.0
  │   └── Engine:current    21 engines (workhorse), port 8101
  ├── Stack:mid        torch 2.12 nightly + transformers 4.51.3 + CUDA 12.8
  │   ├── Engine:qwen       Qwen3TTS, port 8104 (tf pinned 4.57.3)
  │   └── Engine:mid        VibeVoice, Higgs (experimental), port 8103
  ├── Stack:legacy     torch 1.13 + transformers 4.46 + CUDA 11.7
  │   └── Engine:legacy     IndexTTS, Parler (blocked — not deployed), port 8102
  └── Orchestrator     No ML libs — pure HTTP dispatch, port 8001

GPU containers (profiles: gpu, sglang):
  ├── Orpheus   vllm + CUDA 12.1, port 8002 (blocked)
  └── SGLang    Custom pip-built SGLang, port 8005 (S2-Pro/VibeVoice/Higgs — blocked)
```

**Orchestrator mode** (`ORCHESTRATOR_MODE=1`): the orchestrator loads zero ML libraries. All engine requests route via HTTP to engine containers. The web UI is served by the orchestrator.

**Lazy-load engine servers:** Each engine container starts with 0 engines loaded. On synthesis:
1. Evict currently loaded engine (`gc.collect()`, `torch.cuda.empty_cache()`)
2. Load requested engine
3. Synthesize
4. Keep loaded until another engine needs the VRAM

### Mode 2: Bare-Metal (legacy, PowerShell-deployed)

All engines load in-process in a single Python process on the VM. Single systemd service (`arthur-lab.service`). Files deployed by `scripts/deploy/deploy_lab.ps1` to `/opt/arthur/`. Simpler to debug, faster iteration.

---

## VM Details

| Property | Value |
|---|---|
| Host | Proxmox VM 104, Ubuntu 22.04 |
| IP | 192.168.0.87 |
| SSH | `ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87` |
| GPU | NVIDIA RTX 5060 Ti, 16 GB GDDR7, Blackwell sm_120 |
| Driver | 580.159.03 |
| CUDA | 13.0 (current stack), 12.8 (mid/qwen) |
| Torch | 2.14 nightly cu130 (current), 2.12 nightly cu128 (mid/qwen) |
| RAM | 32 GB |
| Web UI | http://192.168.0.87:8001 |
| Models | `/opt/models/` (137 GB used / 177 GB total) |
| HF cache | `/opt/models/huggingface/` |
| Docker images | ~123 GB total (all 7) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/status` | JSON: all engines, availability, RAM estimates |
| `POST` | `/synthesize/{engine}` | Synthesize audio — returns WAV binary |
| `POST` | `/synthesize/{engine}` (multipart) | With reference WAV upload for voice cloning |
| `GET` | `/logs` | Last 200 server-side log entries (ring buffer) |
| `POST` | `/refresh` | Re-probe all engine availability without restart |
| `GET` | `/models/{engine}` | Engine metadata |
| `POST` | `/models/{engine}/load` | Force-load an engine into memory |
| `POST` | `/unload` | Evict currently loaded engine (free VRAM) |

### Synthesize Request

```json
{
  "text": "Hello world.",
  "params": {
    "voice":        "bm_lewis",
    "speed":        1.0,
    "speaker":      "Baldur Sanjin",
    "language":     "en",
    "description":  "A warm elderly male voice speaking slowly and clearly.",
    "exaggeration": 0.5
  }
}
```

Parameters are engine-specific — unused ones are silently ignored.

---

## Image Lab — 5 Image & Video Engines

The project also includes a separate Image & Video generation lab on port 8002:

| # | Key | Label | Type | VRAM | Notes |
|---|---|---|---|---|---|
| 1 | `flux2` | FLUX.2 [dev] | Image | ~16 GB | 32B rectified flow transformer. GGUF quantized. I2I editing. |
| 2 | `flux2klein` | FLUX.2 Klein 4B | Image | ~13 GB | Compact 4B model. Apache 2.0. Step-distilled. |
| 3 | `sd35` | SD 3.5 Large | Image | ~12 GB | 8B MMDiT. GGUF quantized. Turbo/Lightning presets. |
| 4 | `wan` | Wan2.2 | Video | ~14 GB | T2V + I2V. Up to 5s cinematic video. Dual-transformer GGUF. |
| 5 | `ideogram4` | Ideogram 4 | Image | ~6-10 GB | 9.3B DiT + Qwen3-VL. Native text rendering. NF4/FP8 quants. |

> **Optional:** ComfyUI integration toggle via `IMGLAB_USE_COMFYUI=1` env var.

Image Lab deploys separately: `.\scripts\deploy\deploy_image_lab.ps1` — see [`docs/image-lab/`](docs/image-lab/).

---

## Key Source Files

| File | Lines | Role |
|---|---|---|
| `tts_lab.py` | 188 | FastAPI app entry-point, lifespan, route wiring |
| `tts_lab_shims.py` | 590 | **Imported FIRST** — `sys.modules` stubs, transformers compat patches, thread pinning |
| `tts_lab_shims_legacy.py` | 50 | Minimal shims for legacy container (torch 1.13 / tf 4.46) |
| `tts_lab_config.py` | 292 | `MODEL_INFO` catalogue, `MODEL_ORDER`, voice lists, per-engine `_state`, paths |
| `tts_lab_engines.py` | 1,930 | All 28 `_load_X()` + `_synth_X()` pairs, `LOADERS`/`SYNTHERS` dicts |
| `tts_lab_dispatch.py` | 513 | Availability probing, `_ensure_loaded()`, `_do_synth()`, local + remote dispatch |
| `tts_lab_engine_server.py` | 295 | Engine-container FastAPI server with lazy-loading + VRAM eviction |
| `tts_lab_orpheus_server.py` | 107 | Orpheus-specific vllm server |
| `tts_lab_ui.py` | 1,793 | Full HTML/JS web UI inlined as Python strings |
| `tts_lab_utils.py` | 103 | `_to_wav()`, `_wav_dur()`, `_safe_del()`, `_ram_mb()`, `_require_gpu()` |
| `voice_library.py` | 593 | Persian Voice Library — Common Voice download, speaker embeddings |
| `image_lab.py` | 188 | Image Lab FastAPI entry-point (port 8002) |
| `image_lab_engines.py` | 884 | 5 image/video engine load/synth pairs |
| `image_lab_ui.py` | 891 | Image Lab web UI |

---

## Build / Run / Test / Deploy Commands

### Docker (primary path)

```bash
# Build chain
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current:latest .
docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .

# Compose
docker compose up -d                                              # default services
docker compose --profile mid up -d                                # + engine-mid
docker compose --profile mid --profile gpu --profile sglang up -d # everything
docker compose down                                               # stop all

# Engine rebuild
make build-engine ENGINE=current                                  # single engine
make rebuild                                                      # full chain (7 images, ~35 min)
```

### PowerShell Deploy (legacy bare-metal)

```powershell
.\scripts\deploy\deploy_lab.ps1                    # Full fresh deploy (all 8 phases, 30-60 min)
.\scripts\deploy\deploy_lab.ps1 -Phase 5           # Code-only redeploy (~30 sec)
.\scripts\deploy\deploy_lab.ps1 -GPU               # Use CUDA PyTorch instead of CPU
.\scripts\deploy\deploy_lab.ps1 -SkipPhases "4"    # Skip model downloads
```

### Tests & Benchmarks

| Script | Description |
|---|---|
| `bash scripts/test/quick_test.sh` | Fast smoke test (10 engines) |
| `bash scripts/test/test_slow_engines.sh` | 5-min timeout for slow engines |
| `python scripts/benchmark/tts_benchmark.py` | Automated RTF benchmark |
| `python scripts/benchmark/bench_all.py` | Batch benchmark against server |

### Validation Framework

```bash
# Check fingerprints
python scripts/utils/update_engine_status.py --fingerprint --container tts-lab-engine-current

# Record a gate pass
python scripts/utils/update_engine_status.py vibevoice model_load passed --duration 41 --vram-mb 6420

# Check promotion eligibility
python scripts/utils/update_engine_status.py qwen3tts --check

# Promote when all gates pass
python scripts/utils/update_engine_status.py qwen3tts --promote
```

### VM Management

```bash
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87
sudo journalctl -u arthur-lab -f              # Bare-metal logs
docker compose logs -f --tail 50              # Container logs
curl -s http://192.168.0.87:8001/status       # Engine status JSON
nvidia-smi                                      # GPU status
```

---

## VRAM Budget

With lazy-load (one engine in VRAM per container at a time on RTX 5060 Ti 16 GB):

| Scenario | VRAM Used | Free | Viable? |
|----------|:---------:|:----:|:--------:|
| Idle (4 containers, 0 engines) | ~1.2 GB | 14.8 GB | ✅ |
| piper + idle + idle | ~1.4 GB | 14.6 GB | ✅ |
| f5tts + idle + idle | ~4.2 GB | 11.8 GB | ✅ |
| bark + idle + idle | ~13 GB | 3 GB | ✅ (tight) |
| matcha + qwen3tts + idle | ~4.6 GB | 11.4 GB | ✅ |
| bark + qwen3tts | ~17 GB | OOM | ❌ |
| bark + VibeVoice | ~19 GB | OOM | ❌ |

---

## Compatibility Patches

The project applies patches to bridge API gaps between engine requirements and installed library versions. All patches are idempotent and baked into Dockerfiles as `RUN` steps.

| Patch | Purpose |
|---|---|
| `patches/patch_parler_tts.py` | `parler_tts` 0.2.3 → `transformers` 4.51+ compat (6 fixes) |
| `patches/patch_transformers_stubs.py` | Missing `transformers` 4.54+ modules |
| `patches/fix_transformers_shims.py` | Decorator shims + `GeneralInterface` |
| `patches/patch_torchaudio.py` | `torchaudio` backend compat for CosyVoice |
| `patches/patch_torchaudio_init.py` | Additional torchaudio import guards |

---

## Known Constraints

| Constraint | Detail |
|---|---|
| **GPU** | RTX 5060 Ti 16 GB GDDR7, Blackwell sm_120 |
| **Torch nightly required** | sm_120 needs torch >= 2.12 nightly with CUDA 12.8+ |
| **Single-engine VRAM** | One engine at a time per container, evicted on switch |
| **Gated models** | qwen3tts, orpheus, csm require `HF_TOKEN` in `.env` |
| **Blocked engines** | indextts, parler (legacy stack not built), orpheus (vllm incompat), s2pro (SGLang too old) |
| **Maximum concurrent** | 4 containers deployed; bark (12 GB) precludes loading anything else |

---

## Adding a New Engine

1. Add entry to `MODEL_INFO` dict in [`tts_lab_config.py`](tts_lab_config.py)
2. Add key to `MODEL_ORDER` list in [`tts_lab_config.py`](tts_lab_config.py)
3. Add `_load_xxx()` and `_synth_xxx()` in [`tts_lab_engines.py`](tts_lab_engines.py)
4. Register both in `LOADERS` and `SYNTHERS` dicts at bottom of [`tts_lab_engines.py`](tts_lab_engines.py)
5. Add entry to [`docs/engine_compatibility.yaml`](docs/engine_compatibility.yaml) with correct stack/container assignment
6. Add engine URL to orchestrator env vars in [`docker-compose.yml`](docker-compose.yml)
7. Add to correct engine container Dockerfile (tier 3)
8. Deploy: rebuild the affected container image

---

## Documentation Index

### Architecture & Design
| Document | Topic |
|---|---|
| [`docs/containerization/01-ARCHITECTURE.md`](docs/containerization/01-ARCHITECTURE.md) | **Canonical** container architecture — topology, stacks, engine distribution |
| [`docs/reference/ARCHITECTURE_REFERENCE.md`](docs/reference/ARCHITECTURE_REFERENCE.md) | Deployed architecture, fingerprints, performance data |
| [`docs/engine_compatibility.yaml`](docs/engine_compatibility.yaml) | **Single source of truth** — stacks, engines, versions, validation status |
| [`docs/containerization/04-ADHOC-LOG.md`](docs/containerization/04-ADHOC-LOG.md) | Day-by-day fix log (~50KB, very detailed) |
| [`docs/containerization/05-STATE-2026-06-21.md`](docs/containerization/05-STATE-2026-06-21.md) | Deployment state snapshot |

### Reference
| Document | Topic |
|---|---|
| [`docs/reference/TTS_MODEL_COMPARISON.md`](docs/reference/TTS_MODEL_COMPARISON.md) | Side-by-side quality comparison v1 |
| [`docs/reference/TTS_MODEL_COMPARISON2.md`](docs/reference/TTS_MODEL_COMPARISON2.md) | Side-by-side quality comparison v2 |
| [`docs/reference/PERSIAN_TTS_MODELS.md`](docs/reference/PERSIAN_TTS_MODELS.md) | Comprehensive Persian/Farsi TTS reference |
| [`docs/reference/KNOWN_ISSUES.md`](docs/reference/KNOWN_ISSUES.md) | Current bugs and planned fixes |
| [`docs/reference/VM_SETUP_REFERENCE.md`](docs/reference/VM_SETUP_REFERENCE.md) | Proxmox VM setup, disk expansion |
| [`docs/reference/GPU_QA_REFERENCE.md`](docs/reference/GPU_QA_REFERENCE.md) | Blackwell sm_120 library compatibility |
| [`docs/reference/GPU_UPGRADE_ANALYSIS.md`](docs/reference/GPU_UPGRADE_ANALYSIS.md) | GPU upgrade analysis |

### Sessions & Issues
| Document | Topic |
|---|---|
| [`docs/sessions/SESSION_SUMMARY.md`](docs/sessions/SESSION_SUMMARY.md) | Rolling master session summary |
| [`docs/image-lab/`](docs/image-lab/) | Image Lab subsystem documentation |
| [`docs/issues/`](docs/issues/) | Bug investigations |

---

## Environment Variables

### Common (all containers)

| Variable | Value | Purpose |
|---|---|---|
| `HF_HOME` | `/opt/models/huggingface` | HF model cache on data disk |
| `XDG_CACHE_HOME` | `/opt/models/cache` | Pip/suno/whisper caches on data disk |
| `COQUI_TOS_AGREED` | `1` | Suppress XTTS ToS prompt |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer fork warning |
| `PYTHONUNBUFFERED` | `1` | Real-time log output |

### Container-Specific

| Variable | Purpose |
|---|---|
| `ORCHESTRATOR_MODE=1` | Skip ML imports, HTTP-dispatch only (orchestrator) |
| `{ENGINE}_URL=http://engine-xxx:81xx` | Engine routing (orchestrator) |
| `SUNO_USE_SMALL_MODELS=False` | Full Bark model (engine-current) |
| `HF_TOKEN=${HF_TOKEN:-}` | Gated model access (engine-qwen, orpheus) |

Set `HF_TOKEN` in a gitignored `.env` file:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Git Workflow

- **Single branch:** `main` only — no feature branches
- **CI/CD:** GitHub Actions — `build-images.yml` (Docker images to GHCR), `deploy.yml` (manual dispatch)
- **Never commit:** `secrets.env`, `*.env.local`, `.env`, `.claude/settings.local.json`, `__pycache__/`, output audio files

---

## License

MIT — see [LICENSE](LICENSE).
