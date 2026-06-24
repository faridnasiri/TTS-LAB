---
name: 'add-engine'
description: 'Step-by-step guide for adding a new TTS engine to the lab'
---

# Adding a New TTS Engine

## Step 1: Add to MODEL_INFO (tts_lab_config.py)
Add an entry to the `MODEL_INFO` dict with:
```python
"your_engine": {
    "display": "Your Engine Name",
    "desc": "Short description of the engine",
    "size_mb": 2000,
    "params": ["voice", "speed"],
    "default_voice": "default_speaker",
    "loads_on": ["your_dependency_pkg"],
}
```

## Step 2: Add to MODEL_ORDER (tts_lab_config.py)
Insert the engine key into `MODEL_ORDER` list — position controls sidebar order.

## Step 3: Add _load_xxx / _synth_xxx (tts_lab_engines.py)
```python
def _load_your_engine():
    from your_package import YourModel
    return YourModel.from_pretrained("your/model")

def _synth_your_engine(inst, text, params):
    wav, sr = inst.synthesize(text, voice=params.get("voice"))
    return _to_wav(wav, sr), sr
```

## Step 4: Register in LOADERS and SYNTHERS (bottom of tts_lab_engines.py)
```python
LOADERS["your_engine"] = _load_your_engine
SYNTHERS["your_engine"] = _synth_your_engine
```

## Step 5: Add package to pkg_map (tts_lab_dispatch.py)
In `_check_available_local()`, add to `pkg_map`:
```python
pkg_map = { ..., "your_engine": "your_package", }
```

## Step 6: Engine server auto-registration
For container mode, the engine is auto-registered if in `MODEL_ORDER` and `LOADERS`/`SYNTHERS`. No changes needed to `tts_lab_engine_server.py`.

## Step 7: Update docs
- Add to `docs/engine_compatibility.yaml` — status, container, stack, vram_est_mb, notes
- Update `README.md` engine table
- Update `deploy_lab.ps1` Phase 3 pip install for the new package

## Step 8: Deploy
```powershell
.\deploy_lab.ps1 -Phase 5     # Code only (quick, ~30 sec)
# or for full install with pip packages:
.\deploy_lab.ps1              # Full (includes Phase 3 for pip install)
```

## Step 9: Test
```bash
curl -s http://localhost:8001/status | grep your_engine
# or run the quick smoke test:
bash /tmp/quick_test.sh
```

## Container Mode Notes
For the containerized architecture, also update:
- `docker-compose.yml` — add `{ENGINE}_URL` env var in orchestrator section if new container
- The appropriate engine Dockerfile if new pip packages are needed
