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
#   ENGINE values:  current | mid | qwen | legacy | orpheus
#   IMAGE values:   tts-lab-engine-$(ENGINE) | tts-lab-orchestrator
#   PORT values:    8101 | 8102 | 8103 | 8104
# ═══════════════════════════════════════════════════════════════════════

ENGINE ?= current
IMAGE  ?= tts-lab-engine-$(ENGINE)
PORT   ?= 8101

.PHONY: build-engine deploy-engine clean-cache pull rebuild sweep deploy-orchestrator

# ── Help ──────────────────────────────────────────────────────────────
help:
	@echo "TTS Lab Build Automation"
	@echo ""
	@echo "  make build-engine ENGINE=current    Build one engine image"
	@echo "  make deploy-engine ENGINE=current   Build + deploy one engine"
	@echo "  make rebuild                        Full chain rebuild (7 images)"
	@echo "  make sweep                          Run engine synthesis sweep"
	@echo "  make deploy-orchestrator            Build + deploy orchestrator"
	@echo ""
	@echo "  ENGINE values: current | mid | qwen | legacy | orpheus"

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
		--build-arg CACHEBUST=$$(date +%s) \
		-f docker/Dockerfile.engine-$(ENGINE) \
		-t $(IMAGE):latest .

build-orchestrator: pull clean-cache
	@echo "Building tts-lab-orchestrator..."
	docker build \
		--build-arg CACHEBUST=$$(date +%s) \
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
		-e QWEN3TTS_URL=http://localhost:8104 \
		-e VIBEVOICE_URL=http://localhost:8103 \
		-e HIGGS_URL=http://localhost:8103 \
		--restart unless-stopped \
		tts-lab-orchestrator:latest \
		uvicorn tts_lab:app --host 0.0.0.0 --port 8009 --workers 1

# ── Full Chain ────────────────────────────────────────────────────────
rebuild: clean-cache
	@echo "=== Tier 1: Base ==="
	docker build -f docker/Dockerfile.base -t tts-lab-base:latest .
	@echo "=== Tier 2: Stacks ==="
	docker build -f docker/Dockerfile.stack.current -t tts-lab-stack-current:latest .
	docker build -f docker/Dockerfile.stack.mid -t tts-lab-stack-mid:latest .
	@echo "=== Tier 3: Engines ==="
	docker build -f docker/Dockerfile.engine-current -t tts-lab-engine-current:latest .
	docker build -f docker/Dockerfile.engine-mid -t tts-lab-engine-mid:latest .
	docker build -f docker/Dockerfile.engine-qwen -t tts-lab-engine-qwen:latest .
	@echo "=== Orchestrator ==="
	docker build -f docker/Dockerfile.orchestrator -t tts-lab-orchestrator:latest .
	@echo "=== Done ==="

# ── Test ──────────────────────────────────────────────────────────────
sweep:
	python3 sweep.py
