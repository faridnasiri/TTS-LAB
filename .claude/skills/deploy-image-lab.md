---
name: 'deploy-image-lab'
description: 'Deploy Image Lab (7 engines: FLUX.2 Klein ×2, SD 3.5, Wan2.2, Ideogram4, SANA, Boogu) to the VM'
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

## Engines

| Engine | VRAM | Notes |
|---|---|---|
| FLUX.2 Klein 4B | ~10 GB | GGUF, step-distilled, Apache 2.0 |
| FLUX.2 Klein 9B-KV | ~10 GB | GGUF Q6_K, I2I with KV cache |
| SD 3.5 Large | ~12 GB | GGUF/NF4 quantized |
| Wan2.2 | ~14 GB | T2V + I2V, NVFP4/GGUF |
| Ideogram 4 | 6-10 GB | NF4 / FP8 / BF16 quant options |
| SANA 1.6B | ~11 GB | Two variants on the `quant` field: `sprint-1.6b` (1–4 steps, no CFG) + `1.5-1.6b` (~20 steps, CFG 4.5) |
| Boogu Turbo | ~13 GB transient | fp8. **CPU-offload exception** (only engine allowed off GPU) — ~27 GB RAM |

## Environment Variables
- `IMGLAB_USE_COMFYUI=1` — Enable ComfyUI backend
- `IDEOGRAM_API_KEY` — Ideogram API key (for remote inference fallback)
- `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` — Boogu (must precede any transformers import)

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
- **CUDA OOM:** Evict other engines first (heavy: wan ~14 GB, boogu ~13 GB transient).
- **Ideogram 4 fails with BF16:** Use NF4 or FP8 quantization to fit in 16 GB.
- **Sprint 4-step requests fail:** server-side fix passes `intermediate_timesteps=None`
  for steps ≠ 2 (diffusers' SCM default 1.3 is only legal at 2 steps) — keep the fix
  in `_generate_sana` when editing the SANA loader.
- **Boogu all-black outputs:** never `torch.compile` the pipeline (documented).
- **NVFP4 not working:** Check `nvidia-smi` for Blackwell GPU — NVFP4 requires sm_120+.
- **Boogu load errors mentioning deepgemm/kernels:** confirm
  `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` is in the VM `.env` before transformers loads;
  the fp8 mllm also needs `kernels>=0.14,<0.15` in the venv (Phase 3).
