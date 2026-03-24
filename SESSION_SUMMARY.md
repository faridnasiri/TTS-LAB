# Arthur Server — Session Summary
> Chat session: 2026-03-23  
> Branch: `main` — commits `e188bbd`, `0a53f32`, `945d14a`

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
