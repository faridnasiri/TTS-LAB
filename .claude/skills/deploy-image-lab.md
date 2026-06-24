---
name: 'deploy-image-lab'
description: 'Deploy Image Lab (FLUX.2, SD 3.5, Ideogram4, Wan2.2) to the VM'
---

# Deploy Image Lab

## Quick Deploy
```powershell
.\deploy_image_lab.ps1
```

## Partial Deploy
```powershell
.\deploy_image_lab.ps1 -Phase 3   # Code + restart only
```

## Image Lab Details
- **Port:** 8002
- **Service:** `arthur-imglab.service`
- **Code path:** `/opt/arthur-img/`
- **Models:** `/opt/arthur-img-models/` (separate disk from TTS models)
- **HF cache:** `/opt/arthur-img-models/huggingface/`
- **Generated images:** `/opt/arthur-gen/images/`

## Engines

| Engine | VRAM | Notes |
|---|---|---|
| FLUX.2 | ~16 GB | Full model, NVFP4 quantized |
| FLUX.2 Klein 4B | ~13 GB | Smaller variant |
| SD 3.5 Large | ~12 GB | Stable Diffusion |
| Wan2.2 | ~14 GB | NVFP4 quantized |
| Ideogram 4 | 6-10 GB | NF4 (~6 GB), FP8 (~10 GB), BF16 (~14 GB) |

## Environment Variables
- `IMGLAB_USE_COMFYUI=1` — Enable ComfyUI backend
- `IDEOGRAM_API_KEY` — Ideogram API key (for remote inference fallback)

## Check Status
```bash
curl -s http://192.168.0.87:8002/status
```

## Check Logs
```bash
sudo journalctl -u arthur-imglab.service -n 100 --no-pager
sudo journalctl -u arthur-imglab.service -f   # Follow
```

## Gallery
Web UI at `http://192.168.0.87:8002/`. Generated images stored at `/opt/arthur-gen/images/`.

## Common Issues
- **CUDA OOM:** Evict other engines first. FLUX.2 needs ~16 GB.
- **Ideogram 4 fails with BF16:** Use NF4 or FP8 quantization to fit in 16 GB.
- **NVFP4 not working:** Check `nvidia-smi` for Blackwell GPU — NVFP4 requires sm_120+.
