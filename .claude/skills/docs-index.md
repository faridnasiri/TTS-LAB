---
name: 'docs-index'
description: 'Navigate the 45+ markdown docs by topic — find the right document fast'
---

# Documentation Index

## Architecture & Design

| File | What It Covers |
|---|---|
| `docs/containerization/01-ARCHITECTURE.md` | **Canonical** container architecture — compatibility-domain design, topology, stack defs, engine distribution, validation framework |
| `docs/containerization/06-ARCHITECTURE_REFERENCE.md` | Deployed-state architecture reference, builds on 01-ARCHITECTURE |
| `docs/reference/ARCHITECTURE_REFERENCE.md` | Deployed architecture — container topology, VRAM budget, health checks, network layout |
| `docs/engine_compatibility.yaml` | **Single source of truth** — machine-readable: stacks, engines, versions, status, validation gates |

## Deployment & Operations

| File | What It Covers |
|---|---|
| `docs/reference/VM_SETUP_REFERENCE.md` | Proxmox VM 104 setup, data disk expansion, network config |
| `deploy_lab.ps1` | **The primary deploy script** — 8 phases, idempotent |
| `deploy_image_lab.ps1` | Image Lab deploy script |
| `ansible/site.yml` | Ansible IaC (4 roles: docker, disk, deploy, monitoring) |
| `docker-compose.yml` | Docker Compose — 6 containers + profiles |
| `Makefile` | Build automation for Docker images (single-source torch versions) |

## Engine Reference

| File | What It Covers |
|---|---|
| `tts_lab_config.py` | `MODEL_INFO` and `MODEL_ORDER` — all engine metadata, voice lists, paths |
| `tts_lab_engines.py` | All 28 `_load_X()` + `_synth_X()` pairs (~1930 lines) |
| `docs/engine_compatibility.yaml` | Per-engine: status, container assignment, VRAM est, deps, validation gates |
| `docs/reference/TTS_MODEL_COMPARISON.md` | Side-by-side quality comparison v1 |
| `docs/reference/TTS_MODEL_COMPARISON2.md` | Side-by-side quality comparison v2 |
| `docs/reference/PERSIAN_TTS_MODELS.md` | Comprehensive Persian/Farsi TTS model reference |

## GPU & Hardware

| File | What It Covers |
|---|---|
| `docs/reference/GPU_QA_REFERENCE.md` | Blackwell SM 12.0 compatibility, flash-attn verdict |
| `docs/reference/GPU_UPGRADE_ANALYSIS.md` | GPU selection, DDA vs GPU-PV, Hyper-V GPU virtualization |
| `scripts/deploy/setup_proxmox_gpu_passthrough.sh` | Proxmox GPU passthrough setup |

## Benchmark Results

| File | What It Covers |
|---|---|
| `docs/benchmarks/BENCHMARK_RESULTS_2026-04-20_RTX5060Ti.md` | GPU benchmark after RTX 5060 Ti |
| `docs/benchmarks/BENCHMARK_RESULTS_2026-04-23.md` | Post-fix benchmark results |
| `docs/benchmarks/BENCHMARK_RESULTS_2026-03-26.md` | CPU-only benchmark (pre-GPU) |

## Image Lab

| File | What It Covers |
|---|---|
| `docs/image-lab/ARTHUR_IMAGE_LAB_REFERENCE.md` | Image Lab architecture and API reference |
| `docs/image-lab/IMAGE_LAB_API_REFERENCE.md` | Image Lab API endpoints |
| `docs/image-lab/IDEOGRAM4_API_REFERENCE.md` | Ideogram4 API (v1) |
| `docs/image-lab/IDEOGRAM4_API_REFERENCE_V2.md` | Ideogram4 API v2 |
| `docs/image-lab/IDEOGRAM4_OPTIMIZATIONS.md` | VRAM and speed optimizations |
| `docs/image-lab/IDEOGRAM4_VRAM_FIX.md` | VRAM fix details |

## Bug Investigations

| File | Engine/Issue |
|---|---|
| `docs/issues/vibevoice-investigation.md` | VibeVoice deep-dive (~10KB) |
| `docs/issues/vibevoice-upstream-report.md` | Upstream bug report for VibeVoice |
| `docs/issues/s2pro-investigation.md` | Fish S2-Pro investigation |
| `docs/issues/chattts-encode-prompt-decode-bug.md` | ChatTTS LZMA encode/decode bug |

## Historical Session Notes

| File | Date | Topic |
|---|---|---|
| `docs/sessions/SESSION_SUMMARY.md` | Rolling master | All sessions since 2026-03-23 |
| `docs/sessions/SESSION_2026-04-20_TTS_LAB_ENGINE_FIXES.md` | 2026-04-20 | GPU engine fixes for RTX 5060 Ti |
| `docs/sessions/SESSION_2026-04-22_GPU_FLASH_ATTN.md` | 2026-04-22 | Flash attention on Blackwell |
| `docs/sessions/SESSION_2026-04-22_QWEN3TTS.md` | 2026-04-22 | Qwen3TTS integration |
| `docs/sessions/SESSION_2026-05-25_IMAGE_LAB_EVOLUTION.md` | 2026-05-25 | Image Lab evolution |
| `docs/sessions/SESSION_2026-05-25_SD35_NVFP4_FIX.md` | 2026-05-25 | SD3.5 NVFP4 fix |

## CI/CD

| File | What It Covers |
|---|---|
| `.github/workflows/build-images.yml` | CI — build + push 7 Docker images to GHCR |
| `.github/workflows/deploy.yml` | CD — deploy from GHCR to production VM |
