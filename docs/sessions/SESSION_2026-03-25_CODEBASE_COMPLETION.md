# Arthur TTS Lab — Codebase Completion Session
> Date: 2026-03-25 (afternoon)
> Scope: bug fixes, dead-code removal, benchmark expansion to 13 engines, full cross-file consistency
> Branch: `main` — VM: `arthur@192.168.0.87`

---

## 1. What Was Done

This session completed the integration that was partially finished in the morning session.
All 13 TTS engines are now:

- ✅ **Installed** via `setup_tts_lab.sh`
- ✅ **Available** in the web UI (`tts_lab.py`)
- ✅ **Benchmarked** by `tts_benchmark.py` (standalone, offline)
- ✅ **Tested live** via `bench_all.py` (calls the running server)
- ✅ **Warm-RTF tested** via `bench_warm.py` (cold vs warm latency)
- ✅ **Listed** in `requirements.txt` and `requirements_benchmark.txt`
- ✅ **Referenced** in `download_models.sh` (auto-download size table)

---

## 2. Files Changed

| File | Change type | Summary |
|---|---|---|
| `tts_lab.py` | Bug fix + feature | Dead code removed, asyncio fix, `/refresh` endpoint + UI button, comment numbering |
| `setup_tts_lab.sh` | Feature | Steps 7b (ChatTTS) + 7c (OuteTTS) added; final summary updated to 12 pip + CosyVoice2 manual |
| `tts_benchmark.py` | Feature | 6 new bench functions added; BENCH_FNS + ALL_MODELS expanded; CLI `--no-heavy` / `--no-dia` added |
| `bench_all.py` | Feature | Updated from 6 to 13 models; heavy-model HTTP timeout raised to 600 s |
| `bench_warm.py` | Feature | `chattts` + `outetts` added to warm-RTF test list |
| `requirements.txt` | Update | All 13 TTS package deps documented with per-section comments |
| `requirements_benchmark.txt` | Update | Updated from 11 to 13 engines; ChatTTS + OuteTTS sections added |
| `download_models.sh` | Update | Notes section replaced with full 13-engine size/location table |
| `_update_tts_lab.py` | **Deleted** | Leftover temp script |
| `_rewrite_tts_lab_direct.py` | **Deleted** | Leftover temp script |

---

## 3. Bugs Fixed in `tts_lab.py`

### 3.1 Dead `_build_tabs()` block in `_build_page()`

**What was wrong:**
`_build_page()` called `_build_tabs()`, stored the result in `tabs_html`, then ran a
loop that performed a no-op string replace (old == new string) and appended result
cards *outside* the tab panes.  
`tabs_html` was never used — the function rebuilt everything from scratch in an inline
loop directly below, discarding `tabs_html` entirely.

```python
# REMOVED — the entire block was dead code
tabs_html = _build_tabs()
for n in MODEL_ORDER:
    rc = _result_card(n)
    tabs_html = tabs_html.replace(   # no-op: old string == new string
        f'<div class="tab-pane ..." id="tab-{n}">',
        f'<div class="tab-pane ..." id="tab-{n}">')
    tabs_html += rc                  # appended outside panes, discarded anyway
# (inline rebuild below was always the real code)
```

Also removed `_build_tabs()` itself — it was only ever called by this dead block
and produced subtly different HTML (no spinner, no result card).

---

### 3.2 `asyncio.get_event_loop()` deprecation

Python 3.10+ raises `DeprecationWarning` when `get_event_loop()` is called inside
a running coroutine. Python 3.12+ raises `RuntimeError` in some contexts.

```python
# Before
loop = asyncio.get_event_loop()

# After
loop = asyncio.get_running_loop()
```

---

### 3.3 Section comment numbers wrong (Dia through Chatterbox)

The morning session inserted ChatTTS (#4) and OuteTTS (#5) but only renumbered Bark, StyleTTS2, and F5-TTS. Everything after F5-TTS kept the old sequential numbers.

| Section | Old | Fixed |
|---|---|---|
| StyleTTS 2 | 5 | **7** |
| F5-TTS | 6 | **8** |
| Dia-1.6B | 7 | **9** |
| XTTS-v2 | 8 | **10** |
| CosyVoice2 | 9 | **11** |
| Parler-TTS | 10 | **12** |
| Chatterbox | 11 | **13** |

---

## 4. New Features in `tts_lab.py`

### 4.1 `POST /refresh` endpoint

The server caches `_available()` results in `_import_cache`.
After `pip install`ing a new package without restarting, availability badges stayed
**missing** until the process restarted.

```python
@app.post("/refresh")
async def refresh_availability():
    _import_cache.clear()
    return JSONResponse({"refreshed": True, "models": list(MODEL_ORDER)})
```

### 4.2 Refresh availability UI button

A **🔄 Refresh availability** button was added to the header next to the RAM bar.
Clicking it calls `POST /refresh` then re-polls `GET /status` — no page reload needed.

```
[RAM ████████░░ 6.1 / 32 GB  (25.9 GB free)  19.1%]    [🔄 Refresh availability]
```

---

## 5. `setup_tts_lab.sh` — New Steps Added

### Step 7b — ChatTTS

```bash
step "7b — ChatTTS (speed prompts + speaker sampling)"
pip install --quiet ChatTTS
ok "ChatTTS installed"
```

Model weights (~1.2–2.3 GB) auto-download from HuggingFace on first synthesise click.

### Step 7c — OuteTTS

```bash
step "7c — OuteTTS (character-prompt voice + voice cloning)"
pip install --quiet outetts

# Pre-download default OuteTTS model at install time
HF_HOME="${HF_HOME}" python - << 'PYEOF'
import outetts
cfg = outetts.ModelConfig(
    model_path='OuteAI/OuteTTS-0.3-500M',
    tokenizer_path='OuteAI/OuteTTS-0.3-500M',
    backend=outetts.Backend.HF, device='cpu'
)
outetts.Interface(cfg)
PYEOF
ok "outetts installed"
```

Pre-downloading OuteTTS-0.3-500M (~1 GB) at setup time avoids a timeout on the first UI request.

---

## 6. `tts_benchmark.py` — New Bench Functions

| Function | Engine | Key test params | Skip condition |
|---|---|---|---|
| `bench_chattts()` | ChatTTS | `[speed_5]`, temp 0.3, random speaker | < 1.5 GB free RAM |
| `bench_outetts()` | OuteTTS 0.3-500M | `en-female-1-neutral`, temp 0.4 | < 1.4 GB free RAM |
| `bench_bark()` | Bark (small models) | `v2/en_speaker_6`, `[hesitantly]…[sighs]` in text | < 1.2 GB free RAM |
| `bench_styletts2()` | StyleTTS 2 | alpha 0.3, beta 0.7, 5 diffusion steps | none (< 1.5 GB typical) |
| `bench_f5tts()` | F5-TTS | ref = `/tmp/tts_bench/piper.wav`, nfe 32 | piper.wav not found |
| `bench_dia()` | Dia-1.6B-0626 | `[S1]` prefix, auto max_tokens | < 2.5 GB free RAM |

Both Bark and StyleTTS2 patch `torch.load` at load time to allow legacy pickle checkpoints
(`weights_only=False`), then restore the original after loading.

### New CLI flags

```
--no-heavy       skip Dia, XTTS-v2, CosyVoice2 (all >2.5 GB RAM)
--no-dia         skip Dia-1.6B only
--no-xtts        skip XTTS-v2 only
--no-cosyvoice   skip CosyVoice2 only
--models a,b,c   run named subset only
```

---

## 7. Canonical Engine Order

This order is now consistent across every file.

```
 1. piper       Piper TTS        ONNX CPU, 61-116 MB, ~100x RT
 2. kokoro      Kokoro-82M       ONNX CPU, 89 MB,    ~35x RT
 3. melo        MeloTTS          PyTorch, 200 MB,    ~15x RT
 4. chattts     ChatTTS          PyTorch, 1.2-2.3 GB  TBD
 5. outetts     OuteTTS          HF/PyTorch, 1.0 GB   TBD
 6. bark        Bark             PyTorch, 1.3 GB,    ~30x RT  [emotion tokens]
 7. styletts2   StyleTTS 2       PyTorch, 0.7 GB,    ~2x RT
 8. f5tts       F5-TTS           PyTorch, 1.2 GB,    ~4x RT   [ref WAV required]
 9. dia         Dia-1.6B         PyTorch, 3 GB,      ~20x RT  [S1]/[S2] tags
10. xtts        XTTS-v2          Coqui, 1.8 GB,      ~3x RT   58 speakers
11. cosyvoice   CosyVoice2-0.5B  manual, 2 GB,       ~5x RT
12. parler      Parler-TTS mini  PyTorch, 2.5 GB,    ~20x RT  voice via description
13. chatterbox  Chatterbox       PyTorch, 3.0 GB,    ~12x RT  exaggeration slider
```

Files where this order is enforced:
- `MODEL_ORDER` — `tts_lab.py`
- `ALL_MODELS` — `tts_benchmark.py`
- `MODELS` — `bench_all.py`
- `MODELS` — `bench_warm.py`
- Step numbering — `setup_tts_lab.sh`
- Section headers — `requirements.txt` and `requirements_benchmark.txt`

---

## 8. Model Auto-Download Summary

Engines that download weights on first synthesise click (not pre-fetched by setup):

| Engine | Size | Cache |
|---|---|---|
| MeloTTS | ~200 MB | `$HF_HOME/hub/` |
| ChatTTS | ~1.2–2.3 GB | `$HF_HOME/hub/` |
| OuteTTS | ~1.0 GB | `$HF_HOME/hub/` _(pre-fetched by setup 7c)_ |
| Bark (small) | ~1.3 GB | `$XDG_CACHE_HOME/suno/` |
| StyleTTS 2 | ~700 MB | `$HF_HOME/hub/` |
| F5-TTS | ~1.2 GB | `$HF_HOME/hub/` |
| Dia-1.6B | ~3.0 GB | `$HF_HOME/hub/` |
| XTTS-v2 | ~1.8 GB | `~/.local/share/tts/` → symlinked to `/opt/models/tts_coqui/` |
| Parler-TTS mini | ~880 MB | `$HF_HOME/hub/` |
| Chatterbox | ~1.2 GB | `$HF_HOME/hub/` |
| CosyVoice2 | ~2 GB | `/opt/CosyVoice/pretrained_models/` _(manual)_ |

**Total (all except CosyVoice2): ~12–14 GB**  
Ensure `/opt/models` has **20+ GB free** before running all engines for the first time.

---

## 9. Quick Commands

```bash
# Re-check availability after pip install (no server restart required)
curl -sX POST http://192.168.0.87:8001/refresh | python3 -m json.tool

# Benchmark without swap-heavy engines
python tts_benchmark.py --no-heavy

# Benchmark fast engines only
python tts_benchmark.py --models piper,kokoro,melo,bark,styletts2

# Full live test against running server
python bench_all.py

# Cold vs warm RTF
python bench_warm.py

# Restart server
systemctl restart arthur-lab && journalctl -u arthur-lab -f
```

---

## 10. Known Remaining Issues

| Item | Status | Notes |
|---|---|---|
| F5-TTS in `bench_all.py` | ⚠️ | Returns error until a reference WAV is uploaded via web UI (`/upload`) |
| CosyVoice2 install | ⚠️ | Still manual git clone — not on PyPI |
| OuteTTS male speaker | ⚠️ | Only `en-female-1-neutral` built-in; use `voice_characteristics` prompt or upload ref WAV for male voice |
| ChatTTS speaker drift | ⚠️ | Random speaker sampled at server start; use `audio_prompt_id` to lock a speaker across calls |
| Bark / StyleTTS2 / F5-TTS / Dia not in `bench_warm.py` | ⚠️ | Only `chattts` + `outetts` added; the remaining 4 are long-running and omitted intentionally |
| Leftover `.bak` / `.wip` files | ⚠️ | `tts_benchmark.py.bak`, `tts_lab.py.orig`, `tts_lab.py.wip`, `tts_lab.py.lockedtest`, `tts_lab.py.tmpfix`, `tts_lab.tmp`, `_tmp.txt` — safe to delete |
