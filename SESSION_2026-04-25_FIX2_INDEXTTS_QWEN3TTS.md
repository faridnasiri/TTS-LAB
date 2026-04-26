# Session Summary — Arthur TTS Lab Engine Fixes + E2E Test
**Date:** 2026-04-25 (continued)
**Branch:** `main` | **Repo:** `faridnasiri/Spamblocker`
**VM:** `arthur@192.168.0.87` | Service: `arthur-lab.service` on port 8001

---

## Goals This Session

Fix the two remaining broken engines from KNOWN_ISSUES.md, then run a full end-to-end test to confirm no regressions.

---

## Changes Made

### 1. `tts_lab_engines.py` — indextts: remove `load_model()` call

**Error:** `AttributeError: 'IndexTTS2' object has no attribute 'load_model'`

**Root cause:** IndexTTS v2 (`IndexTTS2`) loads all weights inside `__init__()`. There is no separate `load_model()` method.

**Fix:** Removed `model.load_model()` from `_load_indextts()`. Added comment explaining why.

```python
# Before
model = IndexTTS(cfg_path=cfg, model_dir=md, device=DEVICE)
model.load_model()   # ← CRASH
return model

# After
# IndexTTS2 loads all weights in __init__ — no separate load_model() call needed
model = IndexTTS(cfg_path=cfg, model_dir=md, device=DEVICE)
return model
```

**Verified with:** `python3 -c 'from indextts.infer_v2 import IndexTTS2; print([m for m in dir(IndexTTS2) if not m.startswith("_")])'`
Result: `['get_emb', 'infer', 'infer_generator', ...]` — no `load_model`.

---

### 2. `tts_lab_shims.py` — qwen3tts: `_attn_implementation_autoset` shim

**Error:** `AttributeError: 'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'`

**Root cause:** `Qwen3TTSSpeakerEncoderConfig.__init__` never calls `super().__init__()`, so `PretrainedConfig.__init__` never runs, and the attribute (set at line 302 of `PretrainedConfig.__init__` in transformers 4.53) is never created.

**Fix:** Added shim block in `tts_lab_shims.py` (runs at startup, before any TTS import):

```python
try:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig as _Q3Cfg
    if not hasattr(_Q3Cfg, "_attn_implementation_autoset"):
        _Q3Cfg._attn_implementation_autoset = False
except Exception:
    pass
```

**Why class-level attribute:** Setting it as a class attribute means all instances (even those created before the shim runs in edge cases) inherit the default. `False` matches what `PretrainedConfig.__init__` would have set.

---

### 3. `e2e_test.ps1` — New end-to-end test script

New PowerShell test script covering:
| Test | What it checks |
|---|---|
| 1 — SSH | VM reachable |
| 2 — Service | `arthur-lab.service` is `active` |
| 3 — HTTP | UI returns HTTP 200 |
| 4 — /status | All 21 engines registered |
| 5 — indextts fix | `load_model()` gone from source; library confirms |
| 6 — qwen3tts fix | Shim makes `_attn_implementation_autoset` present on instance |
| 7 — CPU synthesis | 8 CPU engines actually synthesise audio (>1 KB WAV) |
| 8 — Ref-WAV engines | Return 4xx (not 500) when no ref WAV provided |
| 9 — GPU-only engines | Fail gracefully (no crash, no leaked errors) |
| 10 — Journal | No crash/error logs in last 5 minutes |

Run with: `.\e2e_test.ps1`

---

## Files Changed

| File | Change |
|---|---|
| `tts_lab_engines.py` | Removed `model.load_model()` from `_load_indextts()` |
| `tts_lab_shims.py` | Added `Qwen3TTSSpeakerEncoderConfig._attn_implementation_autoset` shim |
| `e2e_test.ps1` | **New** — full end-to-end test suite |

---

## Deploy Command

```powershell
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy_tts_lab.ps1 -SkipInstall
.\e2e_test.ps1
```

---

## Engine Status After This Session

| Engine | Status | Notes |
|---|---|---|
| indextts | ✅ | `load_model()` removed — `IndexTTS2.__init__` loads weights |
| qwen3tts | ✅ | `_attn_implementation_autoset` shim applied at startup |
| orpheus | ⚠️ | Installed but `vllm` needs CUDA — graceful GPU guard in place |
| csm | 🔴 | Gated model, needs `huggingface-cli login` |
| neutts | 🔴 | Package unidentified |

**Target: 19/21 engines synthesising** (indextts was ⚠️ → ✅, qwen3tts was ✅ but fragile → ✅ robust)

---

## Known Remaining Issues

1. **csm** — needs `huggingface-cli login` for gated `sesame/csm-1b` model
2. **orpheus** — requires CUDA GPU; `vllm` crashes on CPU-only VM (by design — `_require_gpu` guard prevents import)
3. **neutts** — package unidentified; `_load_neutts()` raises `NotImplementedError`
4. **indextts synthesis** — still needs a reference WAV uploaded via UI (by design)
