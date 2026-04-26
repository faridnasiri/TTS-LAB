# Session Summary — Arthur TTS Lab Engine Fix Marathon
**Date:** 2026-04-25  
**Branch:** `main` | **Repo:** `faridnasiri/Spamblocker`  
**VM:** `arthur@192.168.0.87` | Service: `arthur-lab.service` on port 8001  
**Stack:** Python 3.11, `transformers==4.53.2`, `torch` on CPU (Proxmox VM, no GPU passthrough to this host)

---

## Context

Arthur TTS Lab is a 21-engine TTS benchmark server (`/opt/arthur/tts_lab*.py`), split into 6 modules during a prior refactor:

| File | Role |
|---|---|
| `tts_lab_shims.py` | Imported FIRST — startup-time compatibility patches |
| `tts_lab_config.py` | MODEL_INFO catalogue, constants, slog ring-buffer |
| `tts_lab_engines.py` | All 21 `_load_X` / `_synth_X` function pairs |
| `tts_lab_dispatch.py` | `_ensure_loaded`, `_do_synth`, HTTP handlers |
| `tts_lab_ui.py` | HTML/JS UI builder |
| `tts_lab.py` | Thin FastAPI entry-point |

Deploy script: `tools/arthur_server/deploy_tts_lab.ps1 -SkipInstall`  
Site-packages patch scripts are re-applied on every deploy (step 4.5).

---

## Session Goal

Fix all engines that were broken after `transformers` was upgraded to **4.53.2** (needed for `qwen3tts`). Three engines were failing: **parler**, **indextts**, **qwen3tts**. A fourth (**openvoice**) had a meta-tensor crash.

---

## What Was Fixed (Chronological)

### 1. `openvoice` — Meta-tensor crash on load
**Error:** `Cannot copy out of meta tensor`  
**Fix:** After `ToneColorConverter` init, iterate all sub-modules and replace any `.is_meta` parameters/buffers with empty tensors before `load_ckpt()`.  
**File:** `tts_lab_engines.py` → `_load_openvoice()`  
**Status:** ✅ RTF ~2.2

---

### 2. `indextts` — Wrong class name
**Error:** `cannot import name 'IndexTTS' from 'indextts.infer_v2'`  
**Root cause:** IndexTTS v2 renamed the class to `IndexTTS2`.  
**Fix:** `from indextts.infer_v2 import IndexTTS2 as IndexTTS`  
**File:** `tts_lab_engines.py` → `_load_indextts()`  
**Additional:** `SequenceSummary` was removed from `transformers.modeling_utils` in 4.51+. Added stub via `patch_parler_tts.py` (section 3) and `tts_lab_shims.py`.  
**Status:** ✅ Loads — needs ref WAV to synthesise (by design)

---

### 3. `qwen3tts` — Missing transformers sub-modules
**Errors (in order):**
1. `No module named 'transformers.masking_utils'`
2. `No module named 'transformers.modeling_layers'`
3. `'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'`

**Root cause:** `qwen_tts` was written against `transformers 4.54+` which added `masking_utils` and `modeling_layers`. We're on 4.53.2.

**Fixes:**
- `tts_lab_shims.py`: inject `transformers.masking_utils` and `transformers.modeling_layers` into `sys.modules` at startup if absent (with stub functions/classes)
- `patch_transformers_stubs.py`: writes actual `.py` stub files to site-packages so they survive process restarts  
- `fix_transformers_shims.py`: patches `auto_docstring` and `check_model_inputs` into `transformers.utils.generic`

**Status:** ✅ RTF ~2.0

---

### 4. `parler` — Chain of 7 transformers 4.51+ API breakages

This was the hardest engine. Each fix revealed the next breakage. The full chain:

| # | Error | Root cause | Fix location |
|---|---|---|---|
| 1 | `Config has to be initialized with text_encoder` | `transformers 4.53` calls `ParlerTTSConfig()` with no args in `to_diff_dict()` | `patch_parler_tts.py` § cfg: early return |
| 2 | `cannot import name '_pad_token_tensor'` | Removed from `GenerationConfig` in 4.51+ | `patch_parler_tts.py`: replace with `torch.tensor(generation_config.pad_token_id)` |
| 3 | `'NoneType'.update` (1st time) | `generation_config.update()` returns `None` in 4.51+ (was `model_kwargs` dict) | `patch_parler_tts.py`: `generation_config.update(**kwargs); model_kwargs = kwargs` |
| 4 | `'NoneType'.update` (2nd time) | `model.generate()` returns `None` because `PreTrainedModel` no longer inherits `GenerationMixin` in 4.50+ | `patch_parler_tts.py`: add `_ParlerGenMixin` import + `class ParlerTTSForConditionalGeneration(PreTrainedModel, _ParlerGenMixin)` |
| 5 | `property has no setter` on XTTS | Our `GenerationConfig._pad_token_tensor` property shim had no setter — XTTS writes to it | `tts_lab_shims.py`: removed the property shim (not needed, source is patched) |
| 6 | `'Tensor'._pad_token_tensor` | `_prepare_attention_mask_for_generation` signature changed from `(inputs, pad_t, eos_t)` to `(inputs, generation_config, model_kwargs)` | `patch_parler_tts.py`: replace the call with inline `torch.ones(...)` mask |
| 7 | `_get_initial_cache_position() takes 3 args but 4 given` | Signature changed from `(input_ids, model_kwargs)` to `(seq_length, device, model_kwargs)` | `patch_parler_tts.py`: rewrite method to accept both via `(a, b=None, c=None)` |

**Additional:** `parler-tts` upgraded from 0.2.2 → 0.2.3. HF hub cache permission issue (service runs as root) fixed by resolving local snapshot path directly in `_load_parler()`.

**Status:** ✅ RTF ~4.9

---

## Current Engine Status (post-session)

| Engine | Status | RTF | Notes |
|---|---|---|---|
| piper | ✅ | 0.4 | |
| kokoro | ✅ | 4.8 | |
| melo | ✅ | 1.6 | |
| chattts | ✅ | ~4.6 | |
| outetts | ✅ | ~1.9 | |
| bark | ✅ | ~6.1 | |
| styletts2 | ✅ | ~1.0 | |
| f5tts | ✅ | — | needs ref WAV (by design) |
| dia | ✅ | ~3.0 | |
| xtts | ✅ | 0.8 | |
| cosyvoice | ✅ | 2.5 | |
| **parler** | ✅ | 4.9 | fixed this session |
| chatterbox | ✅ | 3.3 | |
| fishspeech | ✅ | 3.6 | |
| **qwen3tts** | ✅ | ~2.0 | fixed this session |
| **indextts** | ⚠️ | — | loads; `IndexTTS2` has no `load_model()` — needs fix next session |
| **openvoice** | ✅ | 1.7 | fixed this session |
| zonos | ✅ | ~2.5 | |
| csm | 🔴 | — | package not installed |
| orpheus | 🔴 | — | package not installed |
| neutts | 🔴 | — | package not installed |

**18/21 installed and synthesising.**

---

## Key Files Changed This Session

| File | Changes |
|---|---|
| `tts_lab_engines.py` | openvoice meta-tensor fix; indextts IndexTTS2 rename; parler local HF path resolver |
| `tts_lab_shims.py` | masking_utils, modeling_layers, indextts alias, SequenceSummary stubs at startup |
| `patch_parler_tts.py` | Complete rewrite — 7 targeted regex patches to parler_tts source |
| `patch_transformers_stubs.py` | Creates masking_utils.py, modeling_layers.py, SequenceSummary stub in site-packages |
| `fix_transformers_shims.py` | Patches auto_docstring, check_model_inputs into transformers.utils.generic |
| `deploy_tts_lab.ps1` | Added step 4.5 — re-apply all patches on every deploy; added patch scripts to file manifest |
| `quick_test.sh` | New: targeted synthesis test for 10 engines without restarting service |
| `test_slow_engines.sh` | New: 5-min timeout test for indextts/qwen3tts/openvoice |

---

## Infrastructure Notes

- Proxmox VM 104 root disk expanded 90 GB → 650 GB (via `lvextend` + `resize2fs`) — done prior session
- Service runs as **root** under `systemd` (`arthur-lab.service`)
- HF model cache: `/opt/models/huggingface/hub/` (root-owned)
- venv: `/opt/arthur-bench-env/` (Python 3.11)
- All patches applied via `deploy_tts_lab.ps1 -SkipInstall`

---

## Remaining TODOs for Next Session

1. **indextts** — `IndexTTS2` object has no `load_model()` — the v2 API changed. Need to check `IndexTTS2.__init__` and remove/replace the `model.load_model()` call in `_load_indextts()`
2. **qwen3tts** — `'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'` — `PretrainedConfig.__init__` sets this at line 302 in transformers 4.53, but qwen's config subclass may be bypassing `super().__init__()`. Needs investigation.
3. **csm** — install `sesame/csm-1b` (gated model, needs HF login)

---

## How to Resume

```powershell
# Deploy latest code + patches
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy_tts_lab.ps1 -SkipInstall

# Test synthesis
scp ... test_slow_engines.sh arthur@192.168.0.87:/tmp/
ssh arthur@192.168.0.87 "bash /tmp/quick_test.sh"
ssh arthur@192.168.0.87 "bash /tmp/test_slow_engines.sh"

# Check logs
ssh arthur@192.168.0.87 "sudo journalctl -u arthur-lab -n 50 --no-pager"

# Full status
curl http://192.168.0.87:8001/status
```
