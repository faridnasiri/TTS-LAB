# ═══════════════════════════════════════════════════════════════════════
# TTS Lab — Build Automation
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   make build-engine ENGINE=current        # Build one engine image
#   make deploy-engine ENGINE=current       # Build + deploy one engine
#   make rebuild                            # Full chain rebuild (all 7 images)
#   make sweep                              # Run engine synthesis sweep
#
#   ENGINE values:  current | mid | qwen | legacy | orpheus | editx
#   IMAGE values:   tts-lab-engine-$(ENGINE) | tts-lab-orchestrator | tts-lab-sglang-omni
#   PORT values:    8101 | 8102 | 8103 | 8104 | 8105
#
#   S2-Pro (sglang-omni image):
#     docker build -f docker/Dockerfile.sglang -t tts-lab-sglang-omni:latest .
#
#   Override torch version for testing:
#     make build-engine ENGINE=current TORCH_VER=2.12.0.dev20260409+cu128
# ═══════════════════════════════════════════════════════════════════════

ENGINE ?= current
IMAGE  ?= tts-lab-engine-$(ENGINE)
PORT   ?= 8101

# ── Dependency versions (single source of truth) ─────────────────────
TORCH_VER      ?= 2.12.0.dev20260408+cu128
TORCHAUDIO_VER ?= 2.11.0.dev20260407+cu128

.PHONY: build-engine deploy-engine clean-cache pull rebuild sweep deploy-orchestrator build-llm deploy-llm build-sglang-omni deploy-sglang-omni

# ── Help ──────────────────────────────────────────────────────────────
help:
	@echo "TTS Lab Build Automation"
	@echo ""
	@echo "  make build-engine ENGINE=current    Build one engine image"
	@echo "  make deploy-engine ENGINE=current   Build + deploy one engine"
	@echo "  make rebuild                        Full chain rebuild (8 images)"
	@echo "  make sweep                          Run engine synthesis sweep"
	@echo "  make deploy-orchestrator            Build + deploy orchestrator"
	@echo "  make build-llm                      Build Qwen 3.6 LLM image"
	@echo "  make deploy-llm                     Build + deploy Qwen 3.6 LLM"
	@echo "  make build-sglang-omni              Build S2-Pro omni image (sgl-omni serve)"
	@echo "  make deploy-sglang-omni             Build + deploy S2-Pro container"
	@echo ""
	@echo "  ENGINE values: current | mid | qwen | legacy | orpheus | editx"

# ── Cache Management ──────────────────────────────────────────────────
clean-cache:
	@echo "Clearing BuildKit cache older than 24h..."
	docker builder prune -f --filter until=24h

# ── Git Sync ──────────────────────────────────────────────────────────
pull:
	@echo "Pulling latest code..."
	git pull

# ── Build ─────────────────────────────────────────────────────────────
build-engine: pull clean-cache
	@echo "Building $(IMAGE)..."
	docker build \
		--build-arg TORCH_VERSION=$(TORCH_VER) \
		--build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VER) \
		-f docker/Dockerfile.engine-$(ENGINE) \
		-t $(IMAGE):latest .

build-orchestrator: pull clean-cache
	@echo "Building tts-lab-orchestrator..."
	docker build \
		-f docker/Dockerfile.orchestrator \
		-t tts-lab-orchestrator:latest .

# ── Deploy ────────────────────────────────────────────────────────────
deploy-engine: build-engine
	-docker stop $(IMAGE) 2>/dev/null || true
	-docker rm $(IMAGE) 2>/dev/null || true
	docker run -d --name $(IMAGE) --gpus all --network host \
		-v /opt/models:/opt/models \
		-v /tmp/tts_uploads:/tmp/tts_uploads \
		-v /opt/arthur/reference_voices:/opt/arthur/reference_voices \
		-e HF_HOME=/opt/models/huggingface \
		-e XDG_CACHE_HOME=/opt/models/cache \
		-e COQUI_TOS_AGREED=1 \
		-e TOKENIZERS_PARALLELISM=false \
		-e PYTHONUNBUFFERED=1 \
		-e SUNO_USE_SMALL_MODELS=False \
		--restart unless-stopped \
		$(IMAGE):latest

deploy-orchestrator: build-orchestrator
	-docker stop tts-lab-orchestrator 2>/dev/null || true
	-docker rm tts-lab-orchestrator 2>/dev/null || true
	docker run -d --name tts-lab-orchestrator --network host \
		-v /opt/models:/opt/models \
		-v /tmp/tts_uploads:/tmp/tts_uploads \
		-v /opt/arthur/reference_voices:/opt/arthur/reference_voices \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e ORCHESTRATOR_MODE=1 \
		-e HF_HOME=/opt/models/huggingface \
		-e XDG_CACHE_HOME=/opt/models/cache \
		-e COQUI_TOS_AGREED=1 \
		-e TOKENIZERS_PARALLELISM=false \
		-e PYTHONUNBUFFERED=1 \
		-e PIPER_URL=http://localhost:8101 \
		-e MATCHA_URL=http://localhost:8101 \
		-e MELO_URL=http://localhost:8101 \
		-e KOKORO_URL=http://localhost:8101 \
		-e CHATTTS_URL=http://localhost:8101 \
		-e BARK_URL=http://localhost:8101 \
		-e OUTETTS_URL=http://localhost:8101 \
		-e STYLETTS2_URL=http://localhost:8101 \
		-e F5TTS_URL=http://localhost:8101 \
		-e DIA_URL=http://localhost:8101 \
		-e FISHSPEECH_URL=http://localhost:8101 \
		-e CHATTERBOX_URL=http://localhost:8101 \
		-e CHATTERBOXTURBO_URL=http://localhost:8101 \
		-e OMNIVOICE_URL=http://localhost:8101 \
		-e ZONOS_URL=http://localhost:8101 \
		-e CSM_URL=http://localhost:8101 \
		-e XTTS_URL=http://localhost:8101 \
		-e COSYVOICE_URL=http://localhost:8101 \
		-e OPENVOICE_URL=http://localhost:8101 \
		-e MANATTS_URL=http://localhost:8101 \
		-e MMSFAS_URL=http://localhost:8101 \
		-e QWEN3TTS_URL=http://localhost:8104 \
		-e VIBEVOICE_URL=http://localhost:8103 \
		-e HIGGS_URL=http://localhost:8103 \
		-e INDEXTTS_URL=http://localhost:8102 \
		-e PARLER_URL=http://localhost:8102 \
		-e ORPHEUS_URL=http://localhost:8002 \
		-e S2PRO_SGLANG_URL=http://localhost:8005/v1/audio/speech \
		-e EDITX_URL=http://localhost:8105 \
		-e QWEN36_URL=http://localhost:8006 \
		--restart unless-stopped \
		tts-lab-orchestrator:latest \
		uvicorn tts_lab:app --host 0.0.0.0 --port 8009 --workers 1

# ── Full Chain ────────────────────────────────────────────────────────
rebuild: clean-cache
	@echo "=== Tier 1: Base ==="
	docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
	@echo "=== Tier 2: Stacks ==="
	docker build -f docker/Dockerfile.stack.current \
		--build-arg TORCH_VERSION=$(TORCH_VER) \
		--build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VER) \
		-t tts-lab-stack-current:latest .
	docker build -f docker/Dockerfile.stack.mid -t tts-lab-stack-mid:latest .
	@echo "=== Tier 3: Engines ==="
	docker build -f docker/Dockerfile.engine-current \
		--build-arg TORCH_VERSION=$(TORCH_VER) \
		--build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VER) \
		-t tts-lab-engine-current:latest .
	docker build -f docker/Dockerfile.engine-mid -t tts-lab-engine-mid:latest .
	docker build \
		--build-arg TORCH_VERSION=$(TORCH_VER) \
		--build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VER) \
		-f docker/Dockerfile.engine-qwen -t tts-lab-engine-qwen:latest .
	@echo "=== Standalone GPU images ==="
	docker build -f docker/Dockerfile.sglang -t tts-lab-sglang-omni:latest .
	docker build -f docker/Dockerfile.engine-editx -t tts-lab-engine-editx:latest .
	@echo "=== Orchestrator ==="
	docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .
	@echo "=== Done ==="

# ── Test ──────────────────────────────────────────────────────────────
sweep:
	python3 sweep.py

# ── S2-Pro (SGLang-Omni) ────────────────────────────────────────────
# NOTE: do NOT reuse the tts-lab-sglang tag — vibevoice/higgs POC services
# still use that image (old launch_server entrypoint).
build-sglang-omni: pull clean-cache
	@echo "Building tts-lab-sglang-omni (sgl-omni serve, ~25 GB, 30-60 min)..."
	docker build \
		-f docker/Dockerfile.sglang \
		-t tts-lab-sglang-omni:latest .

deploy-sglang-omni: build-sglang-omni
	-docker stop tts-lab-s2pro 2>/dev/null || true
	-docker rm tts-lab-s2pro 2>/dev/null || true
	docker run -d --name tts-lab-s2pro --gpus all \
		-p 8005:8000 \
		-v /opt/models/huggingface:/root/.cache/huggingface \
		-v /opt/models/flashinfer-jit:/root/.cache/flashinfer \
		-v /tmp/tts_uploads:/tmp/tts_uploads \
		-v /opt/arthur/reference_voices:/opt/arthur/reference_voices \
		-e HF_HOME=/opt/models/huggingface \
		--restart unless-stopped \
		tts-lab-sglang-omni:latest

# ── LLM (Qwen 3.6) ─────────────────────────────────────────────────
build-llm: pull clean-cache
	@echo "Building tts-lab-llm-qwen36..."
	docker build \
		-f docker/Dockerfile.llm-qwen36 \
		-t tts-lab-llm-qwen36:latest .

deploy-llm: build-llm
	-docker stop tts-lab-llm-qwen36 2>/dev/null || true
	-docker rm tts-lab-llm-qwen36 2>/dev/null || true
	docker run -d --name tts-lab-llm-qwen36 --gpus all --network host \
		-v /opt/models:/opt/models \
		-e MODEL_PATH=/opt/models/llm/Qwen-Qwen3.6-27B-Q3_K_M.gguf \
		-e CTX_SIZE=32768 \
		--restart unless-stopped \
		tts-lab-llm-qwen36:latest
