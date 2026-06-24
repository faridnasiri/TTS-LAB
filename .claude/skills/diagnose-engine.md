---
name: 'diagnose-engine'
description: 'Engine debugging workflow — check status, logs, GPU memory, imports, known failure patterns'
---

# Diagnosing Engine Issues

## Initial Checks

### 1. Check Engine Status
```bash
curl -s http://192.168.0.87:8001/status | python3 -m json.tool
```
Look for `"available": false` and the `reason` field for the failing engine.

### 2. Check Service Logs
```bash
sudo journalctl -u arthur-lab --no-pager -n 200 | grep -iE "error|exception|traceback|<engine_name>"
```

### 3. Check GPU Memory
```bash
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### 4. Check Python Imports (on VM)
```bash
source /opt/arthur-bench-env/bin/activate
python3 -c "import <engine_package>; print('OK')"
```

### 5. Check Models Exist
```bash
ls /opt/models/huggingface/ | grep -i <model_name>
```

## Common Failure Patterns

### "not available" — import fails
- Check the engine's pip package is installed: `pip list | grep <package>`
- Check for version conflicts with transformers/torch
- Check numpy version (< 2.0 required): `python3 -c "import numpy; print(numpy.__version__)"`

### "not available" — model file missing
- Check model path at `/opt/models/huggingface/` or `/opt/models/<engine>/`
- Run model download phase: `.\deploy_lab.ps1 -Phase 4`

### OOM / CUDA out of memory
- Check VRAM: `nvidia-smi` — other engines may be loaded
- Evict: restart the service `sudo systemctl restart arthur-lab`
- For containers, engine server auto-evicts between engines

### ChatTTS narrow() errors
- PyTorch 2.10 strict validation. Patches already applied in VM's gpt.py.
- If re-occurring after pip upgrade: re-run `.\deploy_lab.ps1 -Phase 6`

### transformers version mismatch
- Engine-current: transformers 5.12.1 (latest, sm_120)
- Engine-mid: transformers 4.51.3 (for qwen3tts compat)
- Engine-legacy: transformers 4.46 (for indextts, parler)

### torchcodec errors
- Symptom: "torchcodec.__spec__ is not set" or "torchcodec not found"
- Fix: Create the metadata stub (see `Dockerfile.stack.current` line 12) or re-apply shims

### inspect.getsourcefile crash
- Symptom: crash on `torch._dynamo` import chain (Python 3.11 + torch 2.10+)
- Fix: Already patched in `tts_lab_shims.py` — ensure shims are imported first

## Step-by-Step Diagnosis Flow

1. **Is the service running?** → `sudo systemctl status arthur-lab`
2. **Is the engine showing in /status?** → `curl -s localhost:8001/status | grep <engine>`
3. **Is it available?** → If "available": false, check the "reason" field
4. **Is there a recent error?** → `sudo journalctl -u arthur-lab -n 100`
5. **Is the dependency installed?** → SSH in and test the import
6. **Is the model downloaded?** → Check `/opt/models/huggingface/models--*`
7. **Are patches applied?** → Run `python3 /opt/arthur/patches/patch_transformers_stubs.py`
8. **Is GPU memory free?** → `nvidia-smi` — if full, `sudo systemctl restart arthur-lab`
