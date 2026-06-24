---
name: 'rebuild-container'
description: 'Container rebuild workflow — Dockerfile hierarchy, build order, cache management'
---

# Container Rebuild Workflow

## Image Layer Hierarchy (Build Order)

```
Tier 1 — Base (rarely rebuilt):
  Dockerfile.base                → tts-lab-base:latest

Tier 2 — Stacks (rebuilt when torch/transformers versions change):
  Dockerfile.stack.current       → tts-lab-stack-current:latest
  Dockerfile.stack.mid           → tts-lab-stack-mid:latest
  Dockerfile.stack.legacy        → tts-lab-stack-legacy:latest

Tier 3 — Engines (rebuilt when engine code or pip packages change):
  Dockerfile.engine-current      → tts-lab-engine-current:latest (21 engines, port 8101)
  Dockerfile.engine-mid          → tts-lab-engine-mid:latest (VibeVoice, Higgs, port 8103)
  Dockerfile.engine-qwen         → tts-lab-engine-qwen:latest (Qwen3TTS, port 8104)
  Dockerfile.engine-legacy       → tts-lab-engine-legacy:latest (IndexTTS, Parler, port 8102)

Special:
  Dockerfile.orchestrator        → tts-lab-orchestrator:latest (no ML libs, port 8001)
  Dockerfile.orpheus             → tts-lab-orpheus:latest (vllm, port 8002)
  Dockerfile.sglang              → tts-lab-sglang:latest (custom pip-built)
```

## Build Commands

```bash
# Single engine (fastest — common case)
make build-engine ENGINE=current

# Full chain rebuild (all 7 images, ~1-2 hours)
make rebuild

# Individual builds:
docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current:latest .
docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .

# Override torch version for testing:
make build-engine ENGINE=current TORCH_VER=2.12.0.dev20260409+cu128
```

## Cache Management
```bash
make clean-cache                           # Prune BuildKit cache older than 24h
docker builder prune -a -f                 # Prune all build cache
docker system df                           # Check disk usage
```

## Build Order Dependencies
- `stack.current` depends on `base` being built first
- `engine.current` depends on `stack.current` being built first
- `engine.mid` depends on `base` (different stack)
- `orchestrator` depends on `base` (inherits directly, no stack)
- When `base` changes, ALL downstream images must be rebuilt

## Torch Version — Single Source of Truth
See `Makefile` lines 23-25:
- Current torch: `2.12.0.dev20260408+cu128`
- Current torchaudio: `2.11.0.dev20260407+cu128`

## After Rebuild
1. Push to GHCR (or rebuild directly on VM)
2. Update `docker-compose.yml` if image tags changed
3. Restart: `docker compose up -d`
4. Health check: `curl -s http://localhost:8001/status`
