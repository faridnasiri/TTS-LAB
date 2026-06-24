---
name: 'deploy-lab'
description: 'Deploy TTS Lab to the Ubuntu VM — PowerShell phases, make deploy-engine, ansible, docker-compose'
---

# TTS Lab Deploy

## Quick Deploy (Code Only — Most Common)
```powershell
.\deploy_lab.ps1 -Phase 5
```
This SCPs `tts_lab_*.py` and patches to `/opt/arthur/`, then restarts the service. ~30 seconds.

## PowerShell Deploy Phases

| Phase | What It Does |
|---|---|
| 1 | apt packages + swap setup |
| 2 | PyTorch install |
| 3 | Engine pip packages |
| 4 | Model downloads |
| 5 | **Code deploy** (SCP tts_lab_*.py + patches + restart) |
| 6 | Re-patch + restart service |
| 7 | Restart service only |
| 8 | Verification (curl /status) |

### Partial Deploy Examples
```powershell
.\deploy_lab.ps1 -Phase 5                     # Code only (fastest)
.\deploy_lab.ps1 -Phase 6                     # Re-patch + restart
.\deploy_lab.ps1 -Phase 7                     # Restart service only
.\deploy_lab.ps1 -Phase 5 -SkipPhases "4"     # Code only, skip model download
.\deploy_lab.ps1 -GPU                         # Use CUDA PyTorch instead of CPU
.\deploy_lab.ps1 -Phase 1                     # Start from scratch
```

## Docker Deploy (on VM)
```bash
# Build + run a single engine container
make deploy-engine ENGINE=current

# Full chain rebuild (all 7 images, ~1-2 hours)
make rebuild

# Just build + run orchestrator
make deploy-orchestrator

# Docker Compose (preferred for container deployment)
docker compose up -d
docker compose --profile gpu up -d       # + Orpheus
docker compose --profile sglang up -d    # + SGLang engines
```

## Ansible Deploy
```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml
ansible-playbook -i ansible/inventory.yml ansible/site.yml --tags deploy
```

## Verification
```bash
# On the VM:
curl -s http://localhost:8001/status | python3 -m json.tool

# Or run the E2E test from dev machine:
.\scripts\test\e2e_test.ps1
```

## SSH Config
- **User:** arthur
- **Host:** 192.168.0.87
- **Key:** `~/.ssh/id_arthur_vm`

## Common Issues
- **Phase 3 pip fails:** VM may lack internet access. Retry Phase 2 first.
- **Phase 5 SCP fails:** SSH key permission issue — check `~/.ssh/id_arthur_vm` exists and is loaded.
- **Service won't start:** `sudo journalctl -u arthur-lab -n 50 --no-pager` for errors.
- **Engines show unavailable:** Patches may need re-applying (run Phase 6).
- **Import errors after deploy:** Check `python3 -c "import tts_lab_shims"` on VM — shims must load first.
