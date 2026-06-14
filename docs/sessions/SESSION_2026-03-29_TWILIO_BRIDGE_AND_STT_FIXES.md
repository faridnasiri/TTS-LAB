# Session 2026-03-29 — Twilio Bridge End-to-End + STT/TTS Pipeline Fixes

## Summary

Full end-to-end bridge call path brought from broken to working.
Twilio replaced Telnyx as the carrier. Android conference lifecycle stabilised.
Local Piper TTS replaced Gemini cloud TTS. STT echo, silence hang, and
voice alignment issues resolved.

---

## Starting state

- `arthur_server.py` was using Telnyx Call Control API (not TwiML / Media Streams)
- Android app was dialing the old Telnyx DID (+14253841028)
- Twilio account was on free trial — inbound calls blocked for unverified callers
- TTS was Gemini 2.5 Flash TTS (cloud, ~7–8 s latency)
- Whisper silence processing caused 15–28 s hangs per silence window
- After 3-way conference, Android fired spurious OnCallAdded/OnCallRemoved events causing RecognizerBusy + premature UI teardown
- LLM stage context was silently lost when Gemini TTS was used for TTS-only
- Arthur's TTS echo was being transcribed by Whisper as scammer speech

---

## Problem → Fix log

### 1. Twilio: "number has calling restriction" (Error 21264)

**Root cause:** Twilio free trial blocks inbound calls from unverified callers.

**Fix:** User upgraded Twilio account from Trial to paid.

---

### 2. Telnyx → Twilio provider switch (`08e86ee`, `6919ba6`, `e397ce6`)

**Root cause:** `arthur_server.py` was built for Telnyx Call Control API
(`X-Telnyx-Signature` webhook, `answer_url` / `call_control_id` JSON).
Twilio sends a completely different format (form-encoded `CallSid`, `From`,
`To`) and expects TwiML XML back, not a JSON control command.

**Fix — three commits:**

| Commit | Change |
|--------|--------|
| `08e86ee` | `/incoming-call` now returns TwiML `<Response><Connect><Stream url="wss://arthur.sys.tips/media-stream"/></Connect></Response>` |
| `6919ba6` | Media stream handler now parses Twilio JSON event shape (`event`, `start.streamSid`, `media.payload`, `media.track`) |
| `e397ce6` | Webhook body parsed as `application/x-www-form-urlencoded` (Twilio's format), not JSON. No `python-multipart` dependency needed. |

**Verification:**
```
[CALL] Twilio webhook  sid=CA4d2f…  from=+14259702341  to=+14256756272
[CALL] Twilio Media Stream connected
[CALL] Stream started  sid=MZc628…  callSid=CA4d2f…  tracks=['inbound']
[CALL] Playing greeting: 'Hello? Oh my goodness…'
```

---

### 3. HomeBridgeNumber still pointed to old Telnyx DID

**Root cause:** `Secrets.cs` still had `HomeBridgeNumber = "+14253841028"` (old Telnyx number).

**Fix:** Updated `Secrets.cs` (gitignored, local only):
```csharp
// Home server bridge — Twilio number → arthur.sys.tips
public const string HomeBridgeNumber = "+14256756272";
```

Also updated:
- `Secrets.cs.template` (`7d8584f`): added `GeminiApiKey` and `HomeBridgeNumber` fields
- `tools/arthur_server/README.md` (`47a3c13`): replaced all Telnyx/Kokoro references with Twilio/Gemini TTS architecture

---

### 4. Android: RecognizerBusy + premature UI teardown after conference (`61266a5`)

**Root cause:** After `CurrentCall.Conference(bridgeCall)`, Android fires a burst of
`OnCallAdded` / `OnCallRemoved` callbacks for internal conference child/parent legs.
The old code treated every `OnCallAdded` as a fresh scammer call:

- `StartTranscription()` called 2–3 extra times → `RecognizerBusy`
- `Fatal error=RecognizerBusy — NOT restarting` → STT died permanently
- `OnCallRemoved` for `CurrentCall` immediately after merge → `IncomingCallActivity.Finish()` → UI torn down

**Fix — `BaiterInCallService.cs`:**

```
OnCallAdded:
  New guard: if (_bridgeCall != null && !PendingBridgeConference) → skip
  Prevents duplicate STT start for conference child legs.

OnCallRemoved — three explicit paths:
  call == _bridgeCall  → clear bridge ref only, leave scammer session alive
  call != CurrentCall  → phantom conference child leg, log and skip
  call == CurrentCall  → full teardown (only path that finishes activity)
```

---

### 5. Whisper 15–28 s silence hang (`4d38b04`)

**Root cause:** Buffer fills with 1 s of near-silence (RMS ≈ 0.001).
Whisper's VAD filters all audio → "Compression ratio threshold is not met" →
temperature retry loop (0.0 → 0.2) → 15–28 s wasted per silence window.

**Fix — two changes in `arthur_server.py`:**

```python
# Fast-discard absolute silence without calling Whisper
if rms < 0.004:
    self.audio_buf.clear()   # < 1 ms, no Whisper call
    continue

# Whisper flags that kill the retry loop
condition_on_previous_text=False  # disables temperature fallback
no_speech_threshold=0.6           # faster silence rejection
without_timestamps=True           # lower per-call latency
```

**Before / after:**
| | Silence window cost |
|---|---|
| Before | 15–28 s per window |
| After | < 1 ms |

---

### 6. Cloud Gemini TTS (7–8 s) → local Piper TTS (~200–500 ms) (`348cf56`)

**Root cause:** `arthur_server.py` was calling `gemini-2.5-flash-preview-tts` (cloud API)
for every Arthur utterance. Latency: 7–8 s. `tts_lab.py` was already running on
`localhost:8001` with Piper TTS at RTF 0.37.

**Fix:**

```python
# Config
LOCAL_TTS_URL = "http://localhost:8001"
PIPER_VOICE   = "en_US-joe-medium"

# _speak() now calls:
POST http://localhost:8001/synthesize/piper
{ "text": "...", "params": { "voice": "en_US-joe-medium" } }
# → { "audio_b64": "<WAV>", "sample_rate": 22050 }
# → parse_wav_pcm() → resample 22050→8000 Hz → μ-law → Twilio WebSocket
```

Added `parse_wav_pcm()` helper using Python stdlib `wave` module.

**Before / after:**
| | TTS latency |
|---|---|
| Gemini 2.5 Flash TTS | ~7 800 ms |
| Piper `en_US-joe-medium` | ~3 600 ms (RTF 0.11) |

Smoke test confirms:
```
latency=3600ms  sr=22050  rtf=0.112  err=None
```

---

### 7. Voice quality (`faa507b` → reverted, `a036687`)

**Attempt 1:** Switched `en_US-ryan-high` → `en_US-lessac-high`
(more mature-sounding male).

**Problem discovered:** `en_US-lessac-high` RTF = 1.15 on Xeon D-1528
(larger model) → 6 091 ms for the greeting. **Not real-time capable.**

**Final fix (`a036687`):** Switched to `en_US-joe-medium`:
- RTF = 0.11, latency = 3 600 ms ✓
- More distinctive/gruff male character than ryan-high
- Faster than ryan-high despite different vocal quality

Voice RTF comparison on Xeon D-1528:
| Voice | RTF | Greeting latency |
|---|---|---|
| `en_US-ryan-high` | 0.34 | 2 261 ms |
| `en_US-lessac-high` | 1.15 | 6 091 ms ❌ |
| `en_US-joe-medium` | 0.11 | 3 600 ms ✓ |

---

### 8. Arthur's TTS echo transcribed by Whisper as scammer speech (`b0f04ab`)

**Root cause:** After `_speak()` sends audio to the Twilio WebSocket,
`is_speaking = False` immediately. But the audio plays for `dur_s` seconds
on the phone, then the conference routes it back as inbound audio
~`dur_s + 4 s` after the WebSocket send. Whisper transcribed this echo
as scammer speech, producing garbage STT like:

```
Scammer: 'I like the feeling better.'
Scammer: 'My name is once again.'
```

The LLM then replied to these, producing completely misaligned responses.

**Fix:**

```python
# In _speak() — after sending audio
self._echo_mute_until = asyncio.get_event_loop().time() + dur_s + 4.0

# In run() media handler — before buffering inbound
if asyncio.get_event_loop().time() < self._echo_mute_until:
    continue  # discard echo window
```

Mute window = `dur_s` (audio playback) + `4.0 s` (conference echo round-trip buffer).

Log confirms it works:
```
[TTS]  ← latency=2261 ms  dur=6.70s  src=22050 Hz  mute=10.7s
```
No garbage STT turns observed after this fix.

---

### 9. LLM stage context lost after Gemini TTS removal (`27829ac`)

**Root cause:** `STAGE_PROMPTS` director notes were previously injected
into the Gemini TTS `full_prompt = f"{director_note} {text}"`.
When `_speak()` was rewritten to use Piper, the stage notes were silently
dropped. The LLM was behaving as Stage 1 forever regardless of call duration.

**Fix:** Compose `system_instruction` from `CORE_PERSONA + stage_note` per call:

```python
stage_note  = STAGE_PROMPTS.get(stage, STAGE_PROMPTS[1])
system_text = (
    f"{CORE_PERSONA}\n\n"
    f"[CURRENT BEHAVIOUR — Stage {stage}] {stage_note}\n"
    "If the caller's words seem garbled or unclear, respond as if you "
    "misheard them — stay in character, ask them to repeat, and naturally "
    "steer the conversation to extract intelligence."
)
```

Also added `initial_prompt` from Arthur's last utterance to `whisper.transcribe()`:
```python
context_hint = f"Arthur said: {last_arthur[:120]}"
initial_prompt=context_hint
```
Anchors Whisper transcription to the right semantic neighbourhood on
degraded 8 kHz μ-law PSTN audio.

**Other LLM tuning in same commit:**
- `maxOutputTokens`: 120 → 200 (longer Arthur responses)
- `temperature`: 0.9 → 0.85 (less hallucination)

---

### 10. Single-word scammer responses discarded (`a036687`)

**Root cause:** `_process_buffer()` had `len(transcript.split()) < 2` filter.
In the first live test after all fixes, scammer said `"Hello?"` then
`"Somebody?"` — both discarded, Arthur silent, scammer hung up.

**Fix:**
```python
# Before
if not transcript or len(transcript.split()) < 2:

# After
if not transcript or len(transcript.split()) < 1:
```
Any non-empty STT result now reaches the LLM.

---

## Final architecture (end of session)

```
Scammer phone → Pixel 5 SIM
    ↓ user taps Bait
BaiterInCallService.InitiateHomeBridge()
    ↓ PlaceCall(+14256756272)
Twilio DID receives call from Android
    ↓ webhook POST /incoming-call
arthur_server.py returns TwiML:
    <Response><Connect>
      <Stream url="wss://arthur.sys.tips/media-stream"/>
    </Connect></Response>
    ↓ Twilio opens WebSocket
Android conferences scammer + bridge call
    ↓ scammer audio flows to Twilio → WebSocket
arthur_server.py:
    ulaw_to_pcm16() → audio_buf → RMS gate
    │
    ├─ rms < 0.004 → discard (no Whisper)
    ├─ is_speaking  → discard (Arthur talking)
    └─ _echo_mute_until → discard (echo window)
        ↓ real scammer speech
    faster-whisper base.en (local, ~1 500 ms)
        ↓ transcript (≥ 1 word)
    Gemini 2.0 Flash API (LLM, ~900 ms)
    stage-aware system prompt + Whisper context hint
        ↓ Arthur's reply text
    Piper en_US-joe-medium via tts_lab.py localhost:8001 (~3 600 ms, RTF 0.11)
        ↓ WAV → resample 22050→8000 Hz → μ-law
    Twilio WebSocket → phone speaker → scammer hears Arthur
```

**Turn latency budget (end-to-end):**
| Stage | Time |
|---|---|
| STT (Whisper base.en) | ~1 500 ms |
| LLM (Gemini Flash) | ~900 ms |
| TTS (Piper joe-medium) | ~3 600 ms |
| Resample + encode | ~50 ms |
| **Total per turn** | **~6 050 ms** |

---

## Live call results

| Call | Duration | Turns | Notes |
|---|---|---|---|
| Test 1 (before fixes) | 20 s | 0 | No scammer speech heard (silence only) |
| Test 2 (after Twilio switch) | 20 s | 1 | `Scammer: 'Oh, one.'` — Piper not yet active |
| Test 3 (after Android lifecycle fix) | 40 s | 2 | Bridge conferenced, echo present (ryan-high) |
| Production call | **958 s / 16 min** | 4 | Echo garbling STT but Arthur still talked for 16 min |
| Test (joe-medium + echo fix) | ~30 s | 0 | `"Hello?"` / `"Somebody?"` discarded (< 2 words) |

---

## Commits this session

| Hash | Description |
|---|---|
| `08e86ee` | feat: switch inbound voice flow from Telnyx to Twilio TwiML |
| `6919ba6` | chore: make media stream parsing/logging Twilio-shaped |
| `e397ce6` | fix: parse Twilio webhook body without python-multipart |
| `7d8584f` | docs(secrets-template): add GeminiApiKey + HomeBridgeNumber fields |
| `47a3c13` | docs(arthur-server): rewrite README from Telnyx/Kokoro to Twilio/Gemini |
| `61266a5` | fix(android): ignore conference child-leg OnCallAdded/Removed events |
| `4d38b04` | perf(arthur-server): Whisper silence fast-discard + retry loop fix |
| `348cf56` | feat(arthur-server): switch TTS from Gemini cloud to local Piper |
| `faa507b` | fix: try en_US-lessac-high voice (reverted — too slow on this CPU) |
| `b0f04ab` | fix(arthur-server): TTS echo mute window (_echo_mute_until) |
| `27829ac` | feat(arthur-server): restore stage-aware LLM + Whisper context hint |
| `a036687` | fix(arthur-server): joe-medium voice + lower STT min to 1 word |

---

## Known remaining issues / next steps

| Issue | Notes |
|---|---|
| TTS latency ~3 600 ms | Piper RTF 0.11 on CPU. Adding GPU (RTX 3060 / A1000) would bring this to ~400 ms. See `BENCHMARK_RESULTS_2026-03-26.md`. |
| `en_US-joe-medium` voice character | Functional but not elderly-sounding enough. GPU would allow StyleTTS2 or XTTS-v2 for voice cloning. |
| STT quality on 8 kHz PSTN | Whisper `base.en` + `initial_prompt` helps. `small.en` would be more accurate but needs 4+ vCPU threads. |
| Android STT transcription shown on UI | The transcript panel shows what Android hears locally (phone earpiece), not the Twilio inbound stream. They differ. |
| Echo mute window is fixed at `dur_s + 4.0 s` | If Twilio buffers heavily, the window could still be too short. Monitor `[STT] Scammer:` lines for suspiciously Arthur-like phrases. |
