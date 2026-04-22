# TTS Lab — Session 2026-04-22: Qwen3-TTS Enabled
> Branch: `main` · Commits: `4a2c3b1` `bb669d6` `87c69da` `d6cfbe0` `8b5daf9` `ee3bdcd`

---

## Starting State
```
  ↷ qwen3tts       skipped (gated HF)
```
Engine existed in code but was permanently skipped — model ID didn't exist, availability probe was wrong.

## Ending State
```
  ▶ qwen3tts       PASS  dur=4560ms  rtf=4.75×  sr=24000Hz  load=11.25s  synth=21673ms
```

---

## What Was Done

### 1. Root Cause: Wrong Model ID
`Qwen/Qwen3-TTS` does not exist on HuggingFace. The actual released models are:

| Model ID | Size | Notes |
|----------|------|-------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ~1 GB | Voice cloning only — requires ref audio |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ~3 GB | Voice cloning only — requires ref audio |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~3 GB | **Built-in speakers, no ref audio needed** ✅ |

Default set to `CustomVoice` so it works out of the box.

### 2. Wrong Availability Probe
Old code probed `/api/models/Qwen/Qwen3-TTS` (metadata API) which returns 200 even for missing/gated repos.
Fixed to probe the actual API URL with the correct model ID.

### 3. Wrong Package: `AutoProcessor/AutoModel` → `qwen-tts`
`qwen3_tts` architecture is not yet merged into any released `transformers` version (not even 5.5.4 or HEAD as of this date).
Alibaba ships a dedicated `qwen-tts` package instead:

```bash
pip install -U qwen-tts
pip install transformers>=5.0   # qwen-tts requires 5.x
```

### 4. API Corrections Found by Testing

| Error | Fix |
|-------|-----|
| `generate()` doesn't exist on Base model | Use `generate_custom_voice()` or `generate_voice_clone()` |
| `speaker_name=` kwarg wrong | Correct kwarg is `speaker=` |
| Base model `get_supported_speakers()` returns `[]` | Use CustomVoice model which has 9 built-in speakers |

### 5. HF Token Setup
Token stored on VM via:
```bash
huggingface-cli login --token <TOKEN>
# Token saved to /home/arthur/.cache/huggingface/token
# HF_TOKEN also set in /etc/environment for service access
```
> ⚠️ **Rotate the token used during this session** at https://huggingface.co/settings/tokens

### 6. transformers Upgrade
Upgraded from `4.52.1` (pinned <5.0) to `5.6.0.dev0` (git HEAD) to satisfy `qwen-tts` requirements.
**Risk:** other engines (indextts, parler, coqui) used the `<5.0` pin to avoid breaking 5.x.
The existing global shims in `tts_lab.py` startup block already handle most 5.x removals — monitor other engines.

---

## Built-in Speakers (CustomVoice model)
`aiden` · `dylan` · `eric` · `ono_anna` · `ryan` · `serena` · `sohee` · `uncle_fu` · `vivian`

Pass via `voice` param in UI or API:
```json
{"text": "Hello world", "params": {"voice": "serena", "language": "english"}}
```

## Voice Clone Mode
Upload a WAV reference in the UI, then pass `ref_text` param with the transcript:
```json
{"text": "New text to speak", "params": {"ref_text": "Transcript of ref audio"}}
```

---

## Final Test Result
```
  ▶ qwen3tts       PASS  dur=4560ms  rtf=4.75×  sr=24000Hz  load=11.25s  synth=21673ms
```
RTF of ~4.75× is expected for a 1.7B model on this GPU without flash-attn.
Install `flash-attn` for faster inference:
```bash
pip install flash-attn --no-build-isolation
```
