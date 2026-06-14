# Arthur TTS Lab — Model Expansion Reference (2026-03-25)

> Session scope: expand model disk, download new TTS families, integrate the safe ones into `tts_lab.py`, and record what is fully usable vs. only cached / still blocked.

---

## 1. Infrastructure changes made this session

| Item | Before | After |
|---|---:|---:|
| Model disk virtual size | 40 GB | 180 GB |
| `/opt/models` filesystem | 40 GB | ~177 GB |
| `/opt/models` free space after downloads | ~13 GB | ~120 GB |
| Experimental extra venv | none | `/opt/arthur-extra-env` |

### Live disk state after resize

```bash
/dev/sda1  -> /opt/models  ~177G total
/dev/sdb1  -> /            ~78G total
```

---

## 2. Models touched this session

### 2.1 Integrated into the main web UI (`tts_lab.py`)

| Model | Status | Cache size seen on disk | Notes |
|---|---|---:|---|
| `ChatTTS` | ✅ integrated | ~1.2–2.3 GB | Uses the bench env package already proven to load on CPU. |
| `OuteTTS` | ✅ integrated | ~971 MB / 1.2 GB / 2.4 GB | UI uses the `OuteTTS-0.3-500M` path by default. |

### 2.2 Downloaded / cached this session but **not** integrated into the main UI yet

| Model / repo | Status | Cache size seen on disk | Reason not in main UI yet |
|---|---|---:|---|
| `myshell-ai/OpenVoiceV2` | downloaded | ~126 MB | Runtime wiring still needed. |
| `fishaudio/fish-speech-1.5` | downloaded | ~1.4 GB | API / runtime wiring still needed. |
| `parler-tts/parler-tts-mini-expresso` | downloaded | ~2.5 GB | Main UI still defaults to `mini-v1`; expresso variant not yet exposed. |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | downloaded | ~2.4 GB | Current Python runtime still does not recognize `qwen3_tts`. |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | downloaded | ~4.3 GB | Same runtime blocker as above. |
| `IndexTeam/IndexTTS-2` | downloaded | ~5.5 GB | Runtime / API wiring still needed. |
| `Zyphra/Zonos-v0.1-transformer` | downloaded | ~3.1 GB | Installed in extra env; not wired into main lab. |

### 2.3 Visible on Hugging Face but **not fully usable** this session

| Model / repo | Status | Notes |
|---|---|---|
| `sesame/csm-1b` | partial / metadata only | Repo info is visible, but the local cache stayed tiny; no full working install was completed. |
| `canopylabs/orpheus-3b-0.1-ft` | blocked | Gated Hugging Face repo; requires token + approval. |

---

## 3. New UI-integrated models

## 3.1 ChatTTS

| Item | Value |
|---|---|
| Package | `ChatTTS` |
| Model cache | `models--2Noise--ChatTTS` / `models--2noise--ChatTTS` |
| Cache size seen | ~1.2 GB + ~2.3 GB variants in cache |
| UI status | ✅ integrated |
| Sample rate | 24 000 Hz |
| Arthur fit | ⭐⭐⭐⭐ |
| Main strength | Conversational cadence; easy CPU experimentation |

### Current UI/default settings used in `tts_lab.py`

| Param | Value |
|---|---|
| `prompt` | `[speed_5]` |
| `top_P` | `0.7` |
| `top_K` | `20` |
| `temperature` | `0.3` |
| `repetition_penalty` | `1.05` |
| `max_new_token` | `512` |
| `skip_refine_text` | `True` |
| speaker | sampled once via `sample_random_speaker()` |

### ChatTTS API notes confirmed live earlier in this session

- `ChatTTS.Chat().load(source="huggingface", device="cpu")` works
- `infer(...)` works and returns `numpy.ndarray` audio
- `sample_random_speaker()` works
- `sample_audio_speaker(wav)` exists for future UI reference-audio support

### Speed / performance

- **Not benchmarked yet in this session.**
- Expected to be slower than Piper / Melo, but lighter than the heaviest 3B-class models.

---

## 3.2 OuteTTS

| Item | Value |
|---|---|
| Package | `OuteTTS` / `outetts` |
| Models cached | `OuteTTS-0.3-500M`, `OuteTTS-1.0-0.6B`, `Llama-OuteTTS-1.0-1B` |
| Cache sizes seen | ~971 MB / ~1.2 GB / ~2.4 GB |
| UI status | ✅ integrated |
| Default UI model path | `OuteAI/OuteTTS-0.3-500M` |
| Default speaker | `en-female-1-neutral` |
| Output sample rate | 44 100 Hz (from `ModelOutput.sr`) |
| Arthur fit | ⭐⭐⭐⭐ |
| Main strength | Prompt/character-driven voice generation |

### Current UI/default settings used in `tts_lab.py`

| Param | Value |
|---|---|
| `model_path` | `OuteAI/OuteTTS-0.3-500M` |
| `speaker` | `en-female-1-neutral` |
| `temperature` | `0.4` |
| `repetition_penalty` | `1.1` |
| `top_k` | `40` |
| `top_p` | `0.9` |
| `min_p` | `0.05` |
| `max_length` | `32768` |

### OuteTTS API notes confirmed live earlier in this session

- `outetts.Interface(outetts.ModelConfig(...))` works on CPU
- `generate(outetts.GenerationConfig(...))` works
- `load_default_speaker("en-female-1-neutral")` works
- `create_speaker(audio_path, transcript=...)` exists for future upload integration

### Speed / performance

- **Not benchmarked yet in this session.**
- The `0.3-500M` model is the practical default for this CPU VM.

---

## 4. Main blockers still remaining

| Area | Current blocker |
|---|---|
| `Qwen3-TTS` | downloaded weights, but runtime still reports unrecognized `qwen3_tts` architecture |
| `Orpheus` | gated HF repo; requires token/access |
| `Sesame CSM` | full transport/download/runtime path still unresolved |
| `OpenVoiceV2` | package + checkpoints need clean runtime glue in the existing lab |
| `Fish Speech` | package installed in extra env, but not yet wired into the main FastAPI app |
| `IndexTTS-2` | model cached, but no UI/runtime integration yet |
| `Zonos` | extra-env import path works, but not integrated into the production lab service |

---

## 5. What changed in `tts_lab.py` this session

- Lab banner updated from **11** to **13** engines
- Added `ChatTTS` model entry and synthesis path
- Added `OuteTTS` model entry and synthesis path
- Updated model registry / availability checks to include both packages
- Updated page title and header counts to 13 engines

> Note: this was a **minimal safe integration**. The new tabs currently use stable defaults rather than exposing every package parameter in the web form.

---

## 6. Recommended next steps

### High-value next UI work

1. Expose the full parameter forms for `ChatTTS` and `OuteTTS`
2. Add `Parler Mini-Expresso` as a selectable `parler` model option
3. Wire `OpenVoiceV2` into the lab using uploaded reference WAVs
4. Decide whether `Fish Speech`, `Zonos`, and `IndexTTS-2` belong in the main bench env or stay isolated

### Benchmarking to run next

```bash
curl -s http://localhost:8001/status | python3 -m json.tool
```

Then benchmark the new safe models after deployment:

```bash
python3 /opt/arthur/bench_warm.py
```

Add dedicated cases for:

- `chattts`
- `outetts`

---

## 7. Bottom line

- Disk is **no longer the main blocker** for most models.
- The next blockers are now **runtime compatibility, gated repos, and clean UI wiring**.
- `ChatTTS` and `OuteTTS` are the two models from this session that are safe enough to be integrated immediately into the main lab UI.

