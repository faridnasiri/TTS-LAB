# Arthur Server — Session Summary
> Chat sessions: 2026-03-23 → 2026-06-29
> Branch: `main`

---

## Session 2026-07-01 — Engine-Current Rebuild, Missing Deps, Final Verification

### Engine-Current Image Rebuild

Docker build on VM repeatedly hung (BuildKit, legacy builder). See ADHOC-LOG §12 for full details. Ended up building from intermediate image + manual pip installs + `docker commit`.

### Missing Dependencies Discovered

The intermediate image (step 14/41) was missing these packages that the full Dockerfile would have installed:

- **`resemble-perth`** (not `perth`!) — chatterbox needs `PerthImplicitWatermarker`
- **`s3tokenizer`**, `conformer`, `diffusers`, `pyloudnorm` — chatterbox deps
- **`transformers` from git main** — `HiggsAudioV2TokenizerModel` not in PyPI 5.12.1
- **`soxr`** — needed by transformers git for audio resampling
- **`python3-dev`, `gcc`** — triton JIT needs gcc to compile CUDA stubs

### Final Verified Engines (2026-07-01)

| Engine | Status |
|---|---|
| omnivoice + voice clone | ✅ sr=24000, rtf=7.9× |
| omnivoice basic | ✅ sr=24000, rtf=4.9× |
| chatterbox (persian) | ✅ sr=24000, rtf=37.7× |
| chatterboxturbo | ✅ sr=24000, rtf=13.9× |
| piper | ✅ sr=22050, rtf=1.1× |

### Documentation Updated

- `04-ADHOC-LOG.md` — §12: rebuild saga + missing deps table
- `06-STATE` — Updated to 2026-07-01 with test results + deps table
- `SESSION_SUMMARY.md` — This entry

---

## Session 2026-06-27/29 — Remote Engine Routing, OmniVoice Fix, LLM VRAM Coordination

### What Was Fixed

Three bugs were preventing remote TTS engines (particularly OmniVoice) from working through the orchestrator.

#### Bug 1: Missing Engine URL Env Vars
- **Symptom:** `POST /synthesize/omnivoice` → `"Not available: pip install omnivoice needed"`
- **Root cause:** Only 7 of 28 `{ENGINE}_URL` env vars were set in the orchestrator container. Missing engines fell through to local dispatch (no ML libs).
- **Fix:** Added 21 missing `-e {ENGINE}_URL=...` lines to Makefile `deploy-orchestrator`. Recreated container with all 28 URLs.
- **Files:** `Makefile`, `docker-compose.yml`

#### Bug 2: OmniVoice Voice Cloning — torchcodec Stub Conflict
- **Symptom:** `audio_prompt_id` → 500 error: `AttributeError: module 'torchcodec' has no attribute 'decoders'`
- **Root cause:** `torchcodec` v99.0.0 dummy stub installed for f5-tts compatibility. When ASR pipeline actually uses it (not just imports), `torchcodec.decoders` is inaccessible at runtime.
- **Fix:** Monkey-patched `is_torchcodec_available` → `False` in `tts_lab_shims.py`.
- **Files:** `tts_lab_shims.py`
- **Note:** Monkey patch. Proper fix requires container isolation (f5-tts vs omnivoice) or real torchcodec.

#### Bug 3: LLM VRAM Blocking Heavy TTS
- **Symptom:** CUDA OOM when loading heavy TTS engines (LLM using ~13.2 GB VRAM)
- **Root cause:** LLM→TTS eviction existed, but TTS→LLM eviction was missing.
- **Fix:** Mounted Docker socket in orchestrator. Added `_stop_llm_container()` / `_start_llm_container()` in dispatch. Heavy TTS stops LLM; LLM inference restarts it.
- **Files:** `tts_lab_dispatch.py`, `Makefile`, `docker-compose.yml`

### Verification

- 13 of 15 tested engines working (piper, kokoro, melo, chattts, f5tts, bark, outetts, chatterbox, fishspeech, zonos, qwen3tts, omnivoice, omnivoice+clone)
- 2 pre-existing failures: dia, styletts2 (hang >180s, not caused by these changes)
- 11 unavailable (expected — optional containers not running or blocked engines)

### Commit

`eb48b67` — fix: remote engine routing, omnivoice voice cloning, LLM VRAM eviction (4 files)

### Documentation Updated

- `docs/containerization/04-ADHOC-LOG.md` — Section 11: full incident report
- `docs/containerization/06-STATE-2026-06-29.md` — New state snapshot
- `docs/reference/KNOWN_ISSUES.md` — Rewritten with all fixed + open issues
- `docs/engine_compatibility.yaml` — Updated omnivoice notes
- `docs/sessions/SESSION_SUMMARY.md` — This entry

---

## Session 2026-04-20 — GPU Engine Fixes (RTX 5060 Ti)

### What Was Fixed

The VM received an RTX 5060 Ti (16 GB GDDR7). Several TTS engines were broken or not using GPU. This session fixed them all one by one with a full benchmark confirming results.

#### VM Infrastructure Changes
| Change | Detail |
|--------|--------|
| Pip cache | Moved to `/opt/models/pip-cache` (180 GB disk) — prevents root disk filling up |
| `onnxruntime-gpu` | Installed — CUDA EP + TensorRT EP now available |
| `libnvrtc.so.13` | Symlinked → `libnvrtc.so.12` for torchcodec ABI compat |
| ChatTTS `gpt.py` | Patched on VM: `narrow(1,-n,n)` guarded for `n=0` (PyTorch 2.10 strict validation) |
| `restart_server()` | Added `sudo` — bench restarts now actually work between engines |

#### Engine Fixes (bench4 — final confirmed results)

| Engine | Before | After | Root cause / Fix |
|--------|--------|-------|-----------------|
| **XTTS-v2** | RTF 3.85 (CPU) | **RTF 0.91** | `gpu=True` in coqui TTS constructor |
| **Chatterbox** | 500 error | **RTF 1.67** | `types.ModuleType` stub (not MagicMock); bypass `torchaudio.save()` → `_to_wav()` |
| **Zonos** | 500 error | **RTF 4.03** | `generate(prefix_conditioning=…)` keyword + `autoencoder.decode()` |
| **ChatTTS** | 500 error | **RTF 2.59** | Patched `gpt.py` narrow() + `empty_cache()` + fixed seed=2024 |
| **MeloTTS** | RTF 1.01 | **RTF 0.30** | Already GPU — confirmed 3.4× faster |
| **StyleTTS2** | RTF 1.52 | **RTF 0.35** | Already GPU — confirmed 4.3× faster |
| **Piper** | RTF 0.37 | RTF 0.36 | GPU EP was 40× slower — kept CPU ONNX (tiny model) |
| **Kokoro** | RTF 2.83 | RTF 2.77 | GPU EP no speedup for 82 MB model — kept CPU ONNX |
| **Parler** | 500 error | ⚠️ version-gated | Requires `transformers==4.46.1`; bench env has 4.57.6. Needs own venv. |
| **OpenVoice** | device mismatch | ⚠️ VAD edge case | Speaker embeddings moved to `DEVICE`; SE extractor null-guarded |
| **OuteTTS** | 500 error | ⚠️ needs GGUF | HF backend pre-encodes any text as ~15K tokens. Use LLAMACPP + `.gguf` file |
| **Orpheus** | 500 error | ⚠️ gated HF | Needs `huggingface-cli login` (canopylabs/orpheus-3b-0.1-ft) |

#### Root Causes Reference

| Error | Cause |
|-------|-------|
| `torchcodec.__spec__ is not set` | MagicMock stub — use `types.ModuleType` with proper `__spec__` |
| `torchaudio.save() TorchCodec required` | torchaudio 2.10 routes save through torchcodec — use `_to_wav()` directly |
| `Zonos: no .decode() found` | API changed: `generate(prefix_conditioning=)` + `autoencoder.decode()` |
| `ChatTTS narrow() length must be non-negative` | PyTorch 2.10 strict: `narrow(1,-n,n)` with `n=0` raises. Patch `gpt.py` line 215, 230, 239, 251 |
| `Parler Config has to be initialized…` | `transformers>=4.51` calls `__init__()` with no args; parler raises ValueError |
| `Parler 'NoneType' has no .update()` | `generation_config` is `None` in transformers 4.57 when no JSON exists |
| `OpenVoice tensor device mismatch` | Speaker SE tensors loaded on CPU, converter on CUDA — use `map_location=DEVICE` |
| `OuteTTS max_length < input_ids` | HF backend encodes any text as ~15K tokens — needs GGUF + LLAMACPP backend |
| `piper/kokoro GPU EP slower` | Tiny ONNX models: GPU memory transfer overhead exceeds GPU compute speedup |

#### Commits This Session

| Commit | Description |
|--------|-------------|
| `7e60c7c` | XTTS `gpu=True`, Zonos API, Chatterbox stub+synth, Piper/Kokoro CPU revert |
| `010fc48` | Chatterbox `ModuleType` stub, Parler version gate, OpenVoice GPU tensors |
| `a9c05b2` | Benchmark results table updated (final bench4 numbers) |
| `93847ae` | ChatTTS `gpt.py` patch, OuteTTS GGUF directive, OuteTTS CHUNKED mode |

---

## Project Goal

**SpamBlocker** is an Android scam-baiting app for Google Pixel running on the user's personal phone number.

When a scam call comes in, instead of hanging up, the app **wastes scammers' time** by deploying Arthur Henderson — a convincingly confused 78-year-old retired postal worker from Phoenix, Arizona — as an AI decoy. Arthur keeps scammers on the line as long as possible while extracting real operational intelligence that can be reported to law enforcement:

| Intelligence target | Why it matters |
|---|---|
| Callback number | VoIP provider can be subpoenaed for account owner |
| Website / URL | WHOIS + hosting company have registrant identity |
| AnyDesk / TeamViewer ID | Both companies cooperate with law enforcement |
| Crypto wallet addresses | Traceable via blockchain + exchange KYC |
| Badge / case numbers | Links to scam operation structure |

**Arthur's character:**  
Lives alone with his cat Mr. Whiskers since wife Martha passed. Son in Tucson calls Sundays. Slow, polite, always "about to comply", never hostile. Progresses through 4 frustration stages across the call to stay believable.

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ANDROID (Pixel 5 — user's personal number)                     │
│                                                                 │
│  BaiterScreeningService                                         │
│    ↓ unknown caller + high spam score                           │
│    → silence ringer, flag PendingAutoBait                       │
│                                                                 │
│  IncomingCallActivity  →  user taps [BAIT]                      │
│                                                                 │
│  BaiterInCallService  (InCallService)                           │
│    ├─ STT: Android SpeechRecognizer (on-device)                 │
│    ├─ ScamKeywordDetector  → scam alert overlay                 │
│    ├─ ScammerIntelExtractor → callback#, URLs, AnyDesk IDs      │
│    │                                                            │
│    └─ ConversationMode ─────────────────────────────────────┐  │
│         ├─ LocalGemini  → GeminiConversationEngine           │  │
│         │                  ↓ Gemini Flash API (LLM)          │  │
│         │                  ↓ Gemini TTS / ElevenLabs         │  │
│         │                  ↓ HardwareLoopbackPlayer          │  │
│         │                    (speaker → mic acoustic inject)  │  │
│         │                                                    │  │
│         ├─ VapiBridge  → PlaceCall(AiBridgeNumber)           │  │
│         │                  ↓ conference with scammer         │  │
│         │                  ↓ VAPI handles AI + voice         │  │
│         │                  ↓ VapiCallMonitor polls transcript │  │
│         │                                                    │  │
│         └─ HomeBridge  → PlaceCall(HomeBridgeNumber) ────────┘  │
│                            ↓ conference with scammer            │
└────────────────────────────┼────────────────────────────────────┘
                             │ PSTN / carrier
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  TWILIO  (+1 425-675-6272)                                     │
│    TwiML → MediaStream WebSocket                               │
│    ↓ 8kHz μ-law audio (bidirectional)                          │
└────────────────────────────┼───────────────────────────────────┘
                             │ WSS
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  CLOUDFLARE TUNNEL  (arthur.sys.tips)                          │
│    ↓ forwards to localhost:8000                                │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────┐
│  UBUNTU VM  192.168.0.87  (Hyper-V, 6 vCPU, Xeon D-1528)     │
│                                                                │
│  nginx → uvicorn → arthur_server.py                           │
│                                                                │
│  Per utterance:                                               │
│    μ-law chunks → PCM 16kHz                                   │
│    → faster-whisper base.en (local, RTF 0.35x)  → text        │
│    → Gemini Flash 2.0 (Arthur persona + stage)  → response    │
│    → Gemini 2.5 Flash TTS, voice=Gacrux         → PCM 24kHz  │
│    → resample → μ-law 8kHz → Twilio stream                   │
└────────────────────────────────────────────────────────────────┘
```

---

## What Was Built

### Home Bridge Mode (new conversation path)
Android app now has **3 conversation modes**:

| Mode | How it works |
|---|---|
| `LocalGemini` | Android on-device Gemini + Gemini TTS via TRRS loopback |
| `VapiBridge` | Outbound call to VAPI number → VAPI handles AI |
| **`HomeBridge`** ← **new** | Outbound call to Twilio DID → arthur.sys.tips → Ubuntu VM |

The Home Bridge flow:
```
Scammer calls Pixel 5
       ↓
Android conferences in a 2nd outbound call to Twilio DID
       ↓
Twilio TwiML → MediaStream WebSocket → arthur.sys.tips (Cloudflare Tunnel)
       ↓
Ubuntu VM (192.168.0.87): faster-whisper STT → Gemini Flash LLM → Gemini TTS
       ↓
PCM audio → μ-law → Twilio → back into the conference → scammer hears Arthur
```

---

## Ubuntu VM — arthur-server

**Host:** Hyper-V on WIN-ER1U9A7NKMI (192.168.0.153)  
**VM IP:** 192.168.0.87  
**SSH key:** `%USERPROFILE%\.ssh\id_arthur_vm`  
**User:** `arthur`

### Hardware
| Resource | Before | After |
|---|---|---|
| vCPUs | 2 | **6** (upgraded this session) |
| RAM | 3.8 GB | 3.8 GB |
| CPU | Intel Xeon D-1528 @ 1.90GHz | same |
| Swap | 0 | 0 ⚠️ add 2GB recommended |

### Project Services Running
| Service | Port | Purpose |
|---|---|---|
| `arthur.service` | 8000 | Python uvicorn server (arthur_server.py) |
| `cloudflared-arthur.service` | — | Exposes port 8000 as `arthur.sys.tips` |
| `nginx.service` | 443 | Reverse proxy / TLS termination |

### Files on VM
```
/opt/arthur/
  arthur_server.py     ← main server
/opt/arthur-env/       ← Python 3.11 venv
/root/.cache/huggingface/   ← faster-whisper model weights
```

---

## STT: faster-whisper

- **Package:** `faster-whisper 1.2.1` by SYSTRAN (NOT OpenAI's original)  
- **Same weights as OpenAI Whisper** — just 4x faster via CTranslate2 engine  
- **Runs 100% locally** — no API calls, no internet needed after model download  
- **Current model:** `base.en` (150MB, int8 quantized)

### Benchmark Results (real speech, espeak phrases)

| Model | vCPUs | Avg STT | Avg RTF | Verdict |
|---|---|---|---|---|
| `base.en` | 2 | 4.84s | 0.51x | ✅ Real-time |
| `small.en` | 2 | 15.26s | 1.59x | ❌ Too slow |
| `base.en` | **6** | **2.75s** | **0.35x** | ✅ Real-time (1.8x faster) |
| `small.en` | **6** | **7.76s** | **0.99x** | ✅ Borderline real-time |

**Current config:** `base.en` — 3x headroom, safe choice.  
**To switch to `small.en`:** Edit `WHISPER_MODEL` in `arthur_server.py` and run `deploy.ps1`. `small.en` has better accent accuracy (Indian/Asian scammer voices) and just fits at 6 vCPUs.

### Why RTF matters
- RTF < 1.0 = Whisper transcribes faster than audio was spoken ✅
- RTF > 1.0 = transcription lags behind real-time, Arthur's response is delayed ❌
- Real bottleneck in practice is **Gemini API (~500ms–2s)**, not Whisper

---

## arthur_server.py Architecture

```python
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
GEMINI_FLASH     = "gemini-2.0-flash"          # LLM
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS
GEMINI_TTS_VOICE = "Gacrux"                    # Mature male voice
WHISPER_MODEL    = "base.en"                   # STT
TWILIO_RATE      = 8000                        # μ-law 8kHz from Twilio
```

**Call flow per utterance:**
1. Twilio streams 8kHz μ-law audio chunks via WebSocket
2. Server accumulates chunks, detects silence via energy threshold
3. Converts μ-law → 16kHz PCM → faster-whisper → text
4. Text → Gemini Flash (with Arthur persona + stage prompts) → response text
5. Response text → Gemini TTS → 24kHz PCM → resample to 8kHz → μ-law → Twilio stream

**Arthur persona stages** (time-based, gets more confused):
- Stage 0 (0–3 min): Politely confused, asks clarifying questions
- Stage 1 (3–6 min): Slightly anxious, asks for repetition
- Stage 2 (6–9 min): Increasingly confused, circles back to earlier topics
- Stage 3 (9+ min): Maximum confusion, seemingly unable to proceed

---

## Local TTS Options (not yet integrated)

Discussed as a future improvement to replace Gemini TTS with fully local voice:

| Engine | Size | CPU Speed | Quality | Best Arthur voice |
|---|---|---|---|---|
| **Kokoro** | 82MB | ~35x RT | ⭐⭐⭐⭐⭐ | `bm_lewis` (British male) |
| **Piper** | 50–200MB | ~100x RT | ⭐⭐⭐ | `en_US-ryan-high` |
| **XTTS v2** | 1.8GB | ~3x RT | ⭐⭐⭐⭐⭐ | Voice cloning |
| **Parler TTS** | 880MB | ~5x RT | ⭐⭐⭐⭐ | Text-described voice |

**Recommended next step:** Integrate Kokoro `bm_lewis` as primary TTS in `arthur_server.py`, with Gemini TTS as fallback. Would reduce TTS latency from ~1–2s (cloud) to ~0.1s (local).

---

## Deploy Workflow

```powershell
# Deploy updated arthur_server.py to VM
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy.ps1
```

Or manually:
```powershell
$key = "$env:USERPROFILE\.ssh\id_arthur_vm"
scp -i $key arthur_server.py arthur@192.168.0.87:/tmp/
ssh -i $key arthur@192.168.0.87 "sudo cp /tmp/arthur_server.py /opt/arthur/ && sudo systemctl restart arthur"
```

Check live logs:
```powershell
ssh -i $key arthur@192.168.0.87 "sudo journalctl -u arthur -f"
```

---

## Cost Summary

| Component | Provider | Cost |
|---|---|---|
| STT (faster-whisper) | Local VM | $0 |
| LLM (Gemini Flash) | Google free tier | $0 |
| TTS (Gemini TTS) | Google free tier | $0 |
| Inbound call | Carrier | $0 (personal number) |
| Outbound to Twilio DID | Twilio | ~$0.008/min |
| **Total per call** | | **~$0.008/min** |

---

## Key Files Changed This Session

| File | Change |
|---|---|
| `Spamblocker/Services/BaiterInCallService.cs` | Added `InitiateHomeBridge()`, `_pendingHomeBridge`, `isHomeBridge` param to `MonitorBridgeCall()` |
| `Spamblocker/IncomingCallActivity.cs` | Added Home Bridge button wiring |
| `Spamblocker/Resources/layout/activity_bait_call.xml` | Added `btnBaitHomeBridge` button |
| `tools/arthur_server/arthur_server.py` | Full Python home bridge server (Twilio MediaStream) |
| `tools/arthur_server/deploy.ps1` | Windows deploy script |
| `tools/arthur_server/setup_vm.sh` | Ubuntu VM one-shot setup script |

---

## Next Chat — Suggested Topics

- [ ] Integrate Kokoro local TTS into `arthur_server.py`
- [ ] Add 2GB swap to Ubuntu VM (`fallocate -l 2G /swapfile`)
- [ ] Switch STT to `small.en` (better accent handling, safe now at 6 vCPUs)
- [ ] Add `small.en` model upgrade to `setup_vm.sh`
- [ ] Test end-to-end with a real scam call
