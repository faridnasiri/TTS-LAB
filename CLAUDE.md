# Arthur TTS Lab

> 28-engine TTS benchmark + 1 LLM (Qwen 3.6 27B) + 5-engine Image/Video lab | FastAPI | Docker multi-container | Ansible IaC
> **Deployed to:** `arthur@192.168.0.87:8009` (TTS) / `:8006` (LLM) | **GPU:** RTX 5060 Ti 16 GB GDDR7 (Blackwell sm_120)

## Project Identity

A self-hosted, multi-engine Text-to-Speech benchmark and evaluation lab. Compare every major open-source TTS model side-by-side through a single web UI. Also includes an Image/Video generation lab (FLUX.2, SD 3.5, Ideogram 4, Wan2.2).

Originally built for an Android scam-baiting app ("SpamBlocker") that uses a character named Arthur Henderson as an AI decoy — hence the "Arthur" naming throughout.

## Architecture — Two Modes

### Mode 1: Bare-Metal (legacy, PowerShell-deployed)
All engines load in-process in a single Python process. Single systemd service (`arthur-lab.service`). Files deployed by `scripts/deploy/deploy_lab.ps1` to `/opt/arthur/`. Simpler to debug, faster iteration.

### Mode 2: Containerized / Orchestrator (current target)
**Compatibility-domain design** — containers are organized by dependency boundaries (torch + transformers + CUDA versions), not per-engine:

```
Base (nvidia/cuda:12.8.2-runtime-ubuntu22.04)
  ├── Stack:current    torch 2.12 nightly + transformers 5.12.1 + CUDA 12.8
  │   └── Engine:current    21 engines (workhorse), port 8101
  ├── Stack:mid        torch 2.12 nightly + transformers 4.51.3 + CUDA 12.8
  │   ├── Engine:qwen       Qwen3TTS, port 8104
  │   └── Engine:mid        VibeVoice, Higgs (experimental), port 8103
  ├── Stack:legacy     torch 1.13 + transformers 4.46 + CUDA 11.7
  │   └── Engine:legacy     IndexTTS, Parler (blocked), port 8102
  ├── LLM:qwen36       Pre-built llama.cpp server-cuda, port 8006 (Qwen 3.6 27B reasoning/coding)
  └── Orchestrator     No ML libs — pure HTTP dispatch, port 8009

GPU containers (profiles: gpu, sglang, llm):
  ├── Orpheus   vllm + CUDA 12.1, port 8002 (blocked)
  ├── SGLang    Custom pip-built SGLang, port 8005 (for VibeVoice/Higgs/S2-Pro)
  └── LLM       Pre-built ghcr.io/ggml-org/llama.cpp:server-cuda (Qwen 3.6 27B Q3_K_M, ~14 GB VRAM)
```

### LLM Deployment (Actual)

```
Container:  tts-lab-llm-qwen36
Image:      ghcr.io/ggml-org/llama.cpp:server-cuda (pre-built, NO source compilation)
Model:      batiai/Qwen3.6-27B-GGUF → Qwen-Qwen3.6-27B-Q3_K_M.gguf (~13 GB)
VRAM:       ~13.6 GB (model + 32K KV cache on GPU)
Speed:      ~23 tok/s (all 99 layers on GPU)
Context:    32768 tokens (q4_0 KV cache, ~1.5 GB)
Thinking:   Enabled (<think> tags — reasoning_content in API response)
API:        OpenAI-compatible POST /v1/chat/completions on port 8006

Deploy command (on VM):
  docker run -d --name tts-lab-llm-qwen36 --gpus all --network host \
    -v /opt/models:/opt/models --restart unless-stopped \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    --model /opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf \
    --host 0.0.0.0 --port 8006 --n-gpu-layers 99 --ctx-size 32768 \
    --parallel 1 --cache-type-k q4_0 --cache-type-v q4_0 \
    --flash-attn on --jinja --threads 4 --threads-batch 8
```

**Orchestrator mode** (`ORCHESTRATOR_MODE=1`): the orchestrator loads zero ML libraries. All engine requests route via HTTP to engine containers using `{ENGINE_NAME}_URL` environment variables. The web UI is served by the orchestrator.

## Key Source Files

| File | Lines | Role |
|---|---|---|
| `tts_lab.py` | 188 | FastAPI app entry-point, lifespan, route wiring |
| `tts_lab_shims.py` | 590 | **Imported FIRST** — `sys.modules` stubs, transformers compat patches, thread pinning |
| `tts_lab_shims_legacy.py` | 50 | Minimal shims for legacy container (torch 1.13 / tf 4.46) |
| `tts_lab_config.py` | 293 | `MODEL_INFO` catalogue, `MODEL_ORDER`, voice lists, per-engine `_state`, paths |
| `tts_lab_engines.py` | 2,100 | All 29 `_load_X()` + `_synth_X()` pairs (28 TTS + 1 LLM), `LOADERS`/`SYNTHERS` dicts |
| `tts_lab_dispatch.py` | 600 | Availability probing, `_ensure_loaded()`, `_do_synth()`, global TTS eviction, LLM dispatch |
| `tts_lab_engine_server.py` | 340 | Engine-container FastAPI server with lazy-loading + VRAM eviction + `/evict` endpoint |
| `tts_lab_orpheus_server.py` | 107 | Orpheus-specific vllm server |
| `tts_lab_ui.py` | 1,900 | Full HTML/JS web UI inlined as Python strings (TTS + LLM chat) |
| `tts_lab_utils.py` | 103 | `_to_wav()`, `_wav_dur()`, `_safe_del()`, `_ram_mb()`, `_require_gpu()` |
| `voice_library.py` | 593 | Persian Voice Library — Common Voice download, speaker embeddings |
| `image_lab.py` | 188 | Image Lab FastAPI entry-point (port 8002) |
| `image_lab_engines.py` | 884 | 5 image/video engine load/synth pairs |
| `image_lab_ui.py` | 891 | Image Lab web UI |

## Build / Run / Test / Deploy Commands

### PowerShell Deploy (primary path for bare-metal)
```powershell
.\scripts\deploy\deploy_lab.ps1                    # Full fresh deploy (all 8 phases, 30-60 min)
.\scripts\deploy\deploy_lab.ps1 -Phase 5           # Code-only redeploy (most common, ~30 sec)
.\scripts\deploy\deploy_lab.ps1 -Phase 6           # Re-patch + restart service
.\scripts\deploy\deploy_lab.ps1 -Phase 7           # Restart service only
.\scripts\deploy\deploy_lab.ps1 -GPU               # Use CUDA PyTorch instead of CPU
.\scripts\deploy\deploy_lab.ps1 -SkipPhases "4"    # Skip model downloads
.\scripts\deploy\deploy_image_lab.ps1              # Deploy Image Lab (separate script)
```

### Makefile (Docker builds on the VM)
```bash
make build-engine ENGINE=current     # Single engine image
make deploy-engine ENGINE=current    # Build + run one engine container
make rebuild                         # Full chain rebuild (all 7 images, ~1-2 hrs)
make deploy-orchestrator             # Build + run orchestrator
make build-llm                       # Build Qwen 3.6 LLM image
make deploy-llm                      # Build + deploy Qwen 3.6 LLM
make sweep                           # Engine synthesis sweep (bare-metal only)
make build-engine ENGINE=qwen TORCH_VER=2.12.0.dev20260409+cu128  # Override torch
```

### Docker Compose
```bash
docker compose up -d                         # orchestrator + engine-current + engine-qwen
docker compose --profile mid up -d           # + engine-mid (VibeVoice, Higgs)
docker compose --profile gpu up -d           # + Orpheus (needs GPU)
docker compose --profile sglang up -d        # + SGLang engines (vibevoice, higgs, s2pro)
docker compose --profile llm up -d           # + Qwen 3.6 LLM (~13 GB VRAM — evicts TTS first)
docker compose down                          # Stop all
```

### Tests & Benchmarks
```bash
.\scripts\test\e2e_test.ps1                  # Full E2E (10 sections)
bash scripts/test/quick_test.sh               # Fast smoke test (10 engines)
bash scripts/test/test_slow_engines.sh        # 5-min timeout for slow engines
python scripts/benchmark/tts_benchmark.py     # Automated RTF benchmark
python scripts/benchmark/bench_all.py         # Batch benchmark against server
```

### VM Management
```bash
ssh -i ~/.ssh/id_arthur_vm arthur@192.168.0.87

# Container status
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# TTS Lab
curl -s http://192.168.0.87:8009/status       # Engine status JSON
curl -s http://localhost:8009/                # Web UI

# LLM (Qwen 3.6)
curl -s http://localhost:8006/health           # LLM health
curl -s http://localhost:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-raw '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'  # Direct LLM test

# VRAM
docker exec tts-lab-llm-qwen36 nvidia-smi      # Or: nvidia-smi on host
docker logs tts-lab-llm-qwen36                 # LLM container logs
docker logs tts-lab-orchestrator               # Orchestrator logs
```

### Ansible
```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml
ansible-playbook -i ansible/inventory.yml ansible/site.yml --tags deploy
```

## API Endpoints

### Orchestrator (port 8009)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (TTS + LLM chat) |
| `GET` | `/status` | JSON: all engines, availability, RAM estimates |
| `POST` | `/synthesize/{engine}` | Synthesize audio (TTS) or generate text (LLM) |
| `POST` | `/synthesize/{engine}` (multipart) | With reference WAV upload for voice cloning |
| `GET` | `/logs` | Last 400 server-side log entries (ring buffer) |
| `POST` | `/refresh` | Re-probe all engine availability without restart |
| `GET` | `/models/{engine}` | Engine metadata |
| `POST` | `/models/{engine}/load` | Force-load an engine into memory |

### LLM Container (port 8006) — OpenAI-compatible

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `GET` | `/health` | Health check (returns 200 OK) |
| `GET` | `/v1/models` | Model list |
| `GET` | `/slots` | Slot/VRAM status |

### Engine Containers (ports 8101-8104)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Engine availability + GPU VRAM info |
| `POST` | `/synthesize` | Synthesize audio (returns base64 WAV) |
| `POST` | `/evict` | **Evict current engine from VRAM** — called by orchestrator before LLM loads |
| `POST` | `/unload` | Manually unload current engine (debug) |

### LLM Synthesis Flow (qwen36)

```
POST /synthesize/qwen36 → orchestrator → Phase 1: POST /evict to ALL TTS containers
                                       → Phase 2: Verify VRAM clean
                                       → Phase 3: POST /v1/chat/completions → llama.cpp
                                       → Response: {text, reasoning, tokens, tokens_per_sec, model, finish_reason}

Response differs from TTS engines:
  TTS:  {audio_b64, sample_rate, synth_time_ms, rtf}
  LLM:  {text, reasoning, tokens, tokens_per_sec, model, finish_reason, synth_time_ms}
```

## Code Conventions

1. **Shims-first import:** `tts_lab_shims.py` MUST be imported before any ML library. It applies `sys.modules` stubs, thread-pin env vars, and transformers compatibility patches before torch/transformers are touched.
2. **Engine load/synth pairs:** Every engine has `_load_ENGINENAME()` and `_synth_ENGINENAME()` in `tts_lab_engines.py`. Load returns an instance; synth returns `(wav_bytes, sample_rate)`.
3. **MODEL_INFO catalogue:** All engine metadata lives in `tts_lab_config.py` as the `MODEL_INFO` dict — display name, description, model size, supported params, default voice.
4. **MODEL_ORDER list:** Controls sidebar order and availability sweep order in `tts_lab_config.py`.
5. **LOADERS/SYNTHERS dicts:** Registered at the bottom of `tts_lab_engines.py` — maps engine key strings to their load/synth functions.
6. **Availability probing:** Uses `find_spec()` (no C-ext imports) for local mode; HTTP health check for remote mode. Defined in `tts_lab_dispatch.py`.
7. **VRAM management (containers):** Single-engine-at-a-time loading with `_evict_current()` — clears references, runs GC, calls `torch.cuda.empty_cache()`.
8. **Patches:** All compatibility patches live in `/patches/` and are re-applied on deploy via Phase 6.
9. **Orchestrator guard:** Code paths that only work in bare-metal mode are wrapped in `if not _ORCHESTRATOR_MODE:`.
10. **Orchestrator-safe imports:** `tts_lab_config.py` has try/except for `tts_lab_shims` imports because the orchestrator has no torch.

## Common Gotchas

- **numpy<2.0 requirement:** `vllm` pulls in numpy 2.x which breaks `numpy.core.multiarray`. Pinned as `numpy>=1.24,<2.0` and `protobuf>=3.20,<4.0` in `requirements.txt`.
- **torch nightly for sm_120:** RTX 5060 Ti (Blackwell) needs torch >= 2.12 nightly builds with CUDA 12.8+. Stable torch releases before 2.12 lack sm_120 support. Current: `2.12.0.dev20260408+cu128`.
- **torchcodec metadata stub:** Must create a dummy `torchcodec-99.0.0.dist-info/METADATA` in site-packages (see `Dockerfile.stack.current` line 12). If missing, Chatterbox and Zonos fail.
- **transformers version conflicts:** Engine-current uses tf 5.12.1; engine-mid uses tf 4.51.3; engine-legacy uses tf 4.46. `ROPE_INIT_FUNCTIONS` removed in 5.x. `TransformGetItemToIndex` added in 4.54.
- **inspect.getsourcefile crash:** `torch._dynamo` import chain corrupts module `__file__` attributes on Python 3.11. Fixed by patching `inspect.getsourcefile` in `tts_lab_shims.py`.
- **ChatTTS narrow() bug:** PyTorch 2.10 strict validation rejects `narrow(1, -n, n)` when n=0. Patched in VM's gpt.py.
- **OpenVoice device mismatch:** Speaker SE tensors load on CPU while model is on CUDA. Fix: `map_location=DEVICE`.
- **OuteTTS max_length:** HF backend encodes any text as ~15K tokens. Use GGUF + LLAMACPP backend instead.
- **Piper/Kokoro GPU EP slower:** Tiny ONNX models are slower via GPU due to memory transfer overhead. Keep CPU ONNX execution provider.

## Documentation Index

| Path | Topic |
|---|---|
| `docs/containerization/01-ARCHITECTURE.md` | **Canonical** container architecture — topology, stacks, engine distribution |
| `docs/containerization/04-ADHOC-LOG.md` | Day-by-day fix log (~50KB, very detailed) |
| `docs/containerization/05-STATE-2026-06-21.md` | Deployment state snapshot |
| `docs/engine_compatibility.yaml` | **Single source of truth** — stacks, engines, versions, validation status |
| `docs/reference/ARCHITECTURE_REFERENCE.md` | Deployed architecture, container topology, VRAM budget |
| `docs/reference/KNOWN_ISSUES.md` | Current bugs, engine fix history |
| `docs/reference/TTS_MODEL_COMPARISON.md` / `*2.md` | Side-by-side quality comparison v1/v2 |
| `docs/reference/PERSIAN_TTS_MODELS.md` | Comprehensive Persian/Farsi TTS reference |
| `docs/reference/VM_SETUP_REFERENCE.md` | Proxmox VM setup, disk expansion |
| `docs/reference/GPU_QA_REFERENCE.md` | Blackwell sm_120 library compatibility |
| `docs/reference/GPU_UPGRADE_ANALYSIS.md` | GPU selection analysis |
| `docs/benchmarks/*.md` | RTF benchmark results by date |
| `docs/containerization/07-QWEN36-LLM-PLAN.md` | Qwen 3.6 LLM integration plan — model selection, VRAM strategy, eviction protocol |
| `docs/containerization/08-QWEN36-DEPLOYMENT.md` | **Qwen 3.6 LLM deployment reference** — full deploy guide, API, VRAM budget, troubleshooting |
| `docs/image-lab/*.md` | Image Lab subsystem docs |
| `docs/sessions/SESSION_SUMMARY.md` | Rolling master session summary |
| `docs/issues/*.md` | Bug investigations (VibeVoice, S2-Pro, ChatTTS) |

## Git Workflow

- **Single branch:** `main` only — no feature branches
- **Commit style:** Descriptive first line, body with rationale. End with `Co-Authored-By: Claude <noreply@anthropic.com>` when Claude-authored.
- **CI/CD:** GitHub Actions on push to main — `build-images.yml` (Docker images to GHCR), `deploy.yml` (manual dispatch, pulls from GHCR)
- **Never commit:** `secrets.env`, `*.env.local`, `.env`, `.claude/settings.local.json`, `__pycache__/`, output audio files

## Key Constraints

- **GPU:** NVIDIA RTX 5060 Ti, 16 GB GDDR7, Blackwell sm_120
- **VM:** `arthur@192.168.0.87`, Ubuntu 22.04, Proxmox VM 104
- **SSH key:** `~/.ssh/id_arthur_vm`
- **Models:** `/opt/models/` (~650 GB), HF cache: `/opt/models/huggingface/`
- **Bare-metal venv:** `/opt/arthur-bench-env/` (Python 3.11)
- **Lab code:** `/opt/arthur/`
- **Services:** `arthur-lab.service` (port 8001), `arthur-imglab.service` (port 8002)
- **Image Lab models:** `/opt/arthur-img-models/` (separate disk)
- **Container registry:** `ghcr.io/farid-nasiri/tts-lab-*`
- **Engine status:** See `docs/engine_compatibility.yaml` — 16 supported, 7 experimental, 5 blocked
