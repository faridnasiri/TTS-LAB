---
name: 'deploy-image-lab'
description: 'Deploy Image Lab (9 engines: FLUX.2 Klein ×2, Ideogram 4, SANA, Boogu, Z-Image, Qwen-Image 2512, HiDream O1 via ComfyUI sidecar, ERNIE-Image) to the VM'
---

# Deploy Image Lab

## Quick Deploy
```powershell
.\scripts\deploy\deploy_image_lab.ps1
```

## Partial Deploy
```powershell
.\scripts\deploy\deploy_image_lab.ps1 -Phase 3   # Code + restart only
```

## Image Lab Details
- **Port:** 8002
- **Service:** `arthur-imglab.service`
- **Code path:** `/opt/arthur-img/`
- **Models:** `/opt/arthur-img-models/` (separate disk from TTS models)
- **HF cache:** `/opt/arthur-img-models/huggingface/`
- **Generated images:** `/opt/arthur-gen/images/`
- **HiDream sidecar:** ComfyUI in `/opt/arthur-img-comfy/`, venv `/opt/arthur-img-comfy-env/`,
  service `arthur-comfy.service`, port 8188 (see `docs/image-lab/HIDREAM_COMFYUI.md`)

## Engines

| Engine | VRAM | Notes |
|---|---|---|
| FLUX.2 Klein 4B | ~10 GB | GGUF, step-distilled, Apache 2.0 |
| FLUX.2 Klein 9B-KV | ~10 GB | GGUF Q6_K, I2I with KV cache |
| Ideogram 4 | 6-10 GB | NF4 / FP8 / BF16 quant options |
| SANA 1.6B | ~11 GB | Two variants on the `quant` field: `sprint-1.6b` (1–4 steps, no CFG) + `1.5-1.6b` (~20 steps, CFG 4.5) |
| Boogu Turbo | ~13 GB transient | fp8. **CPU-offload exception** (only engine allowed off GPU) — ~27 GB RAM |
| Z-Image Turbo | ~10-11 GB† | GGUF Q4_K_M transformer + in-repo Qwen3-4B encoder (encode→park), 8 steps, CFG 0.0 |
| Qwen-Image 2512 | ~14 GB† | GGUF Q4_K_M transformer + quantised VL encoder (stage+park), 20 steps, CFG 4.0 |
| HiDream O1-Dev | ~11-12 GB† | **ComfyUI sidecar process** (port 8188) — Dev fp8_scaled checkpoint, 28 steps, CFG 0.0 |
| ERNIE-Image-Turbo | ~10-11 GB† | NVFP4 (nunchaku-lite) + bnb4 text encoder, 8 steps, CFG 1.0 |

† = estimate — calibrate live (measured numbers in the T2I-swap session doc).

## Environment Variables
- `IDEOGRAM_API_KEY` — Ideogram API key (for remote inference fallback)
- `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` — Boogu (must precede any transformers import)
- `IMGLAB_COMFY_URL` — default `http://127.0.0.1:8188` (HiDream sidecar)

## Check Status
```bash
curl -s http://192.168.0.87:8002/status
curl -s http://127.0.0.1:8188/system_stats   # Comfy sidecar (HiDream)
```

## Check Logs
```bash
sudo journalctl -u arthur-imglab.service -n 100 --no-pager
sudo journalctl -u arthur-imglab.service -f   # Follow
sudo journalctl -u arthur-comfy.service -n 50 --no-pager   # HiDream sidecar
```

## Gallery
Web UI at `http://192.168.0.87:8002/`. Generated images stored at `/opt/arthur-gen/images/`.

## Common Issues
- **CUDA OOM:** Evict other engines first (heavy: qwenimage ~14 GB†, boogu ~13 GB transient, hidream ~11-12 GB† in its own process).
- **Ideogram 4 fails with BF16:** Use NF4 or FP8 quantization to fit in 16 GB.
- **Sprint 4-step requests fail:** server-side fix passes `intermediate_timesteps=None`
  for steps ≠ 2 (diffusers' SCM default 1.3 is only legal at 2 steps) — keep the fix
  in `_generate_sana` when editing the SANA loader.
- **Boogu all-black outputs:** never `torch.compile` the pipeline (documented).
- **Boogu load errors mentioning deepgemm/kernels:** confirm
  `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` is in the VM `.env` before transformers loads;
  the fp8 mllm also needs `kernels>=0.14,<0.15` in the venv (Phase 3).
- **HiDream gen fails / hangs:** comfy sidecar down — check `arthur-comfy.service`, then
  `curl -s http://127.0.0.1:8188/system_stats`; the bridge must run comfy's `/free`
  (`{"unload_models": true}`) after each gen so diffusers engines can load.
- **Qwen-Image encoder OOM:** the VL encoder is staged on GPU only during encode, then
  parked; if resident GGUF + encoder exceed the card, the TTS evict-all path must fire
  first — check the gate comment and `_ensure_encoder_headroom`.
