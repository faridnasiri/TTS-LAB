# Session Report — 2026-04-02
## STT Stability, Conference Lifecycle, Call Recording & Greeting Sync

---

## Overview

This session resolved a cascade of live-production bugs discovered by watching
real logcat + server logs during actual scam calls, then added call recording and
improved the scammer-engagement persona.

**Commits in this session (oldest → newest):**

| Hash | Message |
|---|---|
| `82a54f1` | chore: gitignore Python bytecode + untrack stale .pyc |
| `bf14f7d` | fix(stt): guard OnCallAdded against conference reshuffling + recover Client/RecognizerBusy |
| `1b0c258` | fix(stt): RecreateAndRestartListening for Client/RecognizerBusy + hard cap at 8 failures |
| `e83e6db` | fix(call): suppress OnCallRemoved false teardown during Telecom conference reshuffle |
| `8a6046b` | fix: IVR-only Yes rule + disconnect residual bridge leg on hangup + guard CurrentCall assignment |
| `b850db9` | feat(recording): mix inbound+TTS into WAV and attach to Telegram call report |
| `79459d7` | chore: sync ArthurPersonaPrompts with live server persona + gitignore temp _*.py scripts |
| `402d97b` | fix(persona): natural engagement first, IVR Yes, gradual intel extraction, 10-word limit |
| `bb644ab` | perf: auto-answer+bridge on OnCallAdded for auto-bait + 150ms poll + 50ms greeting delay |
| `639c872` | fix: greeting delay 50ms→500ms + fix truncated CorePersona intel item #2 |
| `0a015ea` | feat: event-driven greeting — Arthur says Hello? exactly when conference is live |

---

## Bug 1 — Stale `.pyc` tracked in git / `__pycache__` not ignored

**Symptom:** `git status` showed `tools/arthur_server/__pycache__/arthur_server.cpython-314.pyc`
as a modified tracked file. VS deploy failed with
`"Build Failed: .NET Android does not support running the previous version"`.

**Root cause:**
- `__pycache__/` and `*.pyc` were never in `.gitignore`
- The `.pyc` had been committed in a previous session

**Fix:**
- `git rm --cached` removed the `.pyc` from the index
- Added `__pycache__/`, `*.pyc`, `*.pyo` to `.gitignore`
- Added `tools/arthur_server/_*.py` and `tools/arthur_server/_*.service` to prevent
  one-off fix scripts from ever being committed

---

## Bug 2 — STT `Client` and `RecognizerBusy` errors killed transcription permanently

**Symptom (log):**
```
W/BaiterSTT: Recognizer error: Client        ← ×2
W/BaiterSTT: Recognizer error: RecognizerBusy
```
After a conference call established, STT went permanently silent.

**Root cause:** Both `Client` and `RecognizerBusy` fell through to `default →
"Fatal error — NOT restarting"` in `BaiterRecognitionListener.OnError()`.

**Fix (commit `bf14f7d`):**
Added `case SpeechRecognizerError.Client` and `RecognizerBusy` as recoverable,
retrying with `RestartListening` after 1 s.

---

## Bug 3 — STT `Client` error looped at 2×/second indefinitely

**Symptom (log):**
```
W/BaiterSTT: Recognizer error: Client   (pairs, ~150 ms apart, every 1 s forever)
```

**Root cause:**  
When `SpeechRecognizerError.Client` fires, the `SpeechRecognizer` instance itself
is broken — calling `StartListening()` on it again just re-fires `Client`
immediately. The fix from Bug 2 only added a 1 s delay to an infinite loop.  
Additionally, the conference reshuffle (Bug 4 below) created **two** recognizer
instances simultaneously, producing the paired-error pattern.

**Fix (commit `1b0c258`):**
- Added `BaiterInCallService.RecreateAndRestartListening()` — destroys the broken
  instance, creates a fresh `SpeechRecognizer`, then calls `StartListening`
- `Client` / `RecognizerBusy` now call `RecreateAndRestartListening` with 1 s delay
- Hard cap: after 8 consecutive failures, STT logs an error and stops retrying
  (prevents battery drain when STT service is permanently unavailable)
- `OnReadyForSpeech` resets the failure counter on every successful bind

---

## Bug 4 — Conference reshuffling created multiple STT instances + false teardown

**Symptom (log):**
```
21:22:38  Bridge ACTIVE — conferencing
21:22:41  OnCallRemoved +scammer     ← false teardown, StopTranscription() + Finish()
21:22:41  OnCallAdded +scammer       ← _isTranscribing=false → StartTranscription #2
21:22:41  OnCallRemoved +scammer     ← tears down again
21:22:42  OnCallAdded +scammer       ← StartTranscription #3
21:22:42  OnCallAdded +bridge        ← StartTranscription #4
21:22:42  RecreateAndRestartListening ×2
```

**Root cause:**  
When `CurrentCall.Conference(bridgeCall)` is called, Android Telecom restructures
its internal call graph and fires a burst of `OnCallRemoved` / `OnCallAdded`
events. The `OnCallRemoved` for `CurrentCall` was treated as a real hang-up →
`StopTranscription()` + `Finish()` called → `_isTranscribing = false` → every
subsequent `OnCallAdded` created a new `SpeechRecognizer`.

The existing bridge guard `if (_bridgeCall != null && ...)` only blocked events
while `_bridgeCall` was non-null. By the time the reshuffled events fired,
`_bridgeCall` had already been cleared.

**Fix (commit `e83e6db` + `8a6046b`):**

1. **`_conferencedAtMs: long`** — stamped in `MonitorBridgeCall` when
   `Bridge ACTIVE — conferencing` fires.

2. **`OnCallRemoved` 5-second suppression window** — if `call == CurrentCall`
   and `elapsed < 5_000 ms` since conference, skip full teardown; just null
   `CurrentCall` and return. The activity stays open, STT keeps running.

3. **Conference parent tracking** — in the `OnCallAdded` bridge guard, if the
   incoming call has a blank number and `_conferencedAtMs > 0`, set
   `CurrentCall = conferenceParent`. This ensures when the scammer truly hangs
   up, `OnCallRemoved(conferenceParent)` fires with `call == CurrentCall` and
   triggers the correct teardown.

4. **`CurrentCall` / `_currentNumber` assignment guarded** — moved inside
   `if (!_isTranscribing)` block in the non-ringing `OnCallAdded` path so
   reshuffled legs cannot overwrite the scammer's number.

**Result (verified log):**
```
OnCallRemoved 3096 ms post-conference — suppressing reshuffle  ✅
Conference parent detected — tracking as CurrentCall           ✅
OnCallAdded +scammer — STT already active, skipping            ✅
OnCallRemoved 3119 ms post-conference — suppressing reshuffle  ✅
OnCallAdded +bridge  — STT already active, skipping            ✅
OnCallRemoved (conference parent) → real teardown fires        ✅
```

---

## Bug 5 — Residual bridge leg kept call alive after scammer disconnected

**Symptom:** After the scammer hung up, the phone showed an active call and
arthur_server kept sending silence probes.

**Root cause:**  
When the scammer hangs up from a 3-way conference, Telecom dissolves the
conference parent but the **bridge leg (+14256756272) persists as a standalone
active call**. Twilio's WebSocket stayed open → arthur_server kept probing silence.

**Fix (commit `8a6046b`):**
In `OnCallRemoved` teardown, sweep all remaining active legs:
```csharp
foreach (var residual in Calls?.ToList() ?? [])
    if (residual != null && residual != call)
        try { residual.Disconnect(); } catch { }
```

---

## Bug 6 — Arthur said "Yes." to everything including human speech

**Symptom (server log):**
```
Scammer: "Good morning."    → Arthur: "Yes."  ✅ (IVR)
Scammer: "No."              → Arthur: "Yes."  ❌
Scammer: "Hello."           → Arthur: "Yes."  ✅ (IVR)
```

**Root cause:**  
`CORE_PERSONA` contained `"SIMPLE YES/NO QUESTIONS: answer with just 'Yes.' — Always."`.
With a 5-word cap, Gemini applied this rule to all short utterances regardless of
whether they were IVR prompts or human speech.

**Fix (commit `8a6046b` + `402d97b`):**
- Replaced blanket rule with: `"IVR INTRO (robotic pre-recorded voice at call start): answer 'Yes.'"` 
- Added explicit: `"Do NOT say 'Yes' to statements, commands, or human conversation"`
- Applied to both `arthur_server.py` `CORE_PERSONA` and `ArthurPersonaPrompts.cs`

---

## Bug 7 — Arthur interrogated callers robotically causing immediate hang-ups

**Symptom (server log):**
```
Scammer: "Hey, what's up?"      → Arthur: "Your name, please, dear?"
Scammer: "No."                  → Arthur: "Badge number, please?"
Scammer: "I think I may want…"  → Arthur: "Callback number, please?"
Scammer hangs up at 46 s
```

**Root cause:**  
The `STAGE 1` directive was `"ask: name, badge number, direct callback number"` +
`"MAXIMUM 5 WORDS"`. Gemini compressed both constraints into a cold interrogation
loop, cycling `name → badge → callback` regardless of what the scammer said.

**Fix (commit `402d97b`):**

| Setting | Old | New |
|---|---|---|
| Word limit | 5 | **10** |
| Stage 1 goal | Immediately interrogate | **Warm up 1-2 turns first** |
| Persona tone | "extract intel" | **"respond to what they actually say"** |
| `maxOutputTokens` | 40 | **60** |
| `temperature` | 0.65 | **0.7** |

New Stage 1 directive:
> "Keep them talking and build rapport first, THEN gather name / badge / callback.
> Do NOT open with interrogation questions. After 1-2 natural exchanges, work in:
> 'Now who am I speaking with, dear?' then badge, then callback — one per turn."

**Result (next call):**
```
Scammer: "What do you mean?"
→ Arthur: "Oh, hello. I wasn't expecting a call. Who is this?"  ✅
```

---

## Feature — Call recording attached to Telegram report

**Implemented (commit `b850db9`):**

Every call now produces a mixed WAV recording sent to Telegram immediately after
the text report.

**Architecture:**
```
Inbound μ-law frames (all, pre-mute) → _rec_inbound bytearray (full timeline)
Each TTS synthesis → _tts_overlays [(t_rel_s, pcm8k_bytes)]
                                          ↓
                     On call end: numpy mix → 8 kHz mono PCM16 WAV
                                          ↓
                               Telegram sendAudio (up to 48 MB)
```

**Mixing:** TTS is placed at sample offset `t_rel × 8000` from call start, blended
at 65% over the inbound track. Inbound captures natural silence during Arthur's
speech windows, so there is no double-audio at TTS positions.

**File sizes (8 kHz mono PCM16 ≈ 960 KB/min):**
| Call length | WAV size |
|---|---|
| 2 min | ~1.9 MB |
| 5 min | ~4.8 MB |
| 10 min | ~9.6 MB |
| 39 min (max observed) | ~37 MB |

Calls > 48 MB are saved to `/opt/arthur/calls/` but skipped from Telegram upload.

**Verified in log:**
```
[RECORD] WAV saved: /opt/arthur/calls/20260402T061712Z_5e364dac0611.wav  0.7 MB
[RECORD] Telegram audio sent
```

---

## Improvement — Auto-bait answer latency reduced by ~4 seconds

**Problem:** From call arriving (ringing) to Arthur saying "Hello?":
```
OnCallAdded Ringing          t=0
BeginActiveCallStatic        t+4s    ← Activity launch + human tap delay
InitiateHomeBridge           t+4s
Bridge ACTIVE                t+7s
Arthur says Hello?           t+7.5s  ← total ~7.5 s
```

**Fixes (commit `bb644ab` + `639c872`):**

1. **Auto-answer from `OnCallAdded`** — for `PendingAutoBait && AutoStartOnBaitCall`,
   `call.Answer(0)` + `StartTranscription()` + `InitiateHomeBridge()` fires directly
   from `OnCallAdded` without waiting for `IncomingCallActivity` to launch or the
   user to tap. Activity still launches for monitoring.

2. **Bridge poll interval** — 300 ms → **150 ms** (saves up to 150 ms)

3. **Server greeting sleep** — fixed 300 ms → event-driven (see below)

**Total savings: ~3.5–4 seconds**

---

## Feature — Event-driven greeting (perfect conference sync)

**Problem:** Arthur said "Hello?" after a fixed timer that had no relationship
to when the conference was actually established:
```
WebSocket opens ─── wait 500ms ──► "Hello?"
Conference() called ── (somewhere ±seconds)
```

**Fix (commit `0a015ea`):**

**Server — `CallSession._conference_ready: asyncio.Event`**
- `_greet()` now calls `await asyncio.wait_for(self._conference_ready.wait(), timeout=8.0)`
- After event fires, waits 200ms for audio routing to settle
- 8-second fallback prevents silent stalls on missed signal

**New `/conference-ready` endpoint**
```python
@app.post("/conference-ready")
async def conference_ready():
    _active_session._conference_ready.set()
```

**Android — `MonitorBridgeCall` (home bridge path)**
After `CurrentCall?.Conference(call)` succeeds, fires a fire-and-forget HTTP POST
to `{ArthurBridgeServerUrl}/conference-ready`.

**Result:**
```
[CALL] Waiting for /conference-ready signal (max 8s)...
[CALL] Conference confirmed — greeting in 200ms          ← exact sync
[CALL] Playing greeting: 'Hello?'
```

---

## Files Changed

### `tools/arthur_server/arthur_server.py`
- `CORE_PERSONA`: IVR-only Yes rule, 10-word limit, natural engagement first, removed blanket "Yes" rule
- `STAGE_PROMPTS`: rapport-first Stage 1, natural phrasing in Stages 2–4
- `_greet()`: fixed timer → `asyncio.Event` wait + 200ms settle + 8s fallback
- `CallSession.__init__`: added `_rec_inbound`, `_tts_overlays`, `_conference_ready`
- `run()` media handler: `_rec_inbound.extend(pcm16)` before mute checks
- `_speak()`: records `(t_rel_s, pcm8k_bytes)` in `_tts_overlays`
- `_finalize_call()`: calls `self._send_recording()` after Telegram report
- `_send_recording()`: new method — numpy mix → WAV → Telegram `sendAudio`
- `POST /conference-ready`: new endpoint — sets `_active_session._conference_ready`
- `maxOutputTokens`: 40 → 60; `temperature`: 0.65 → 0.7

### `Spamblocker/Services/BaiterInCallService.cs`
- `OnCallAdded` ringing path: auto-answer + bridge for `PendingAutoBait`
- `OnCallAdded` non-ringing path: `CurrentCall`/`_currentNumber` moved inside
  `!_isTranscribing` guard to prevent reshuffled legs overwriting references
- `OnCallAdded` bridge guard: tracks conference parent (blank number) as `CurrentCall`
- `OnCallRemoved`: 5-second reshuffle suppression window using `_conferencedAtMs`
- `OnCallRemoved` teardown: sweeps and disconnects all residual active legs
- `MonitorBridgeCall`: poll interval 300 ms → 150 ms; fires `POST /conference-ready`
- `_conferencedAtMs: long`: new field, stamped on `Bridge ACTIVE — conferencing`
- `RecreateAndRestartListening()`: new method — destroys broken recognizer, creates fresh instance

### `Spamblocker/Services/BaiterRecognitionListener.cs`
- `OnError(Client)` / `OnError(RecognizerBusy)`: calls `RecreateAndRestartListening`
  with 1 s delay, cap at 8 consecutive failures
- `_hardFailCount: int`: new field for consecutive failure tracking
- `OnReadyForSpeech`: resets `_hardFailCount` on successful bind

### `Spamblocker/Services/ArthurPersonaPrompts.cs`
- `CorePersona`: synced with server — IVR-only Yes, 10-word limit, natural behavior
- `Stage1`–`Stage4`: short tactical directives matching server `STAGE_PROMPTS`
- `InitialGreeting`: `"Hello? Oh hold on dear…"` → `"Hello?"`
- Fixed truncated item #2 (`"Full name and ."` → `"Full name and employee / badge / case ID."`)

### `.gitignore`
- Added `__pycache__/`, `*.pyc`, `*.pyo`
- Added `tools/arthur_server/_*.py`, `tools/arthur_server/_*.service`

---

## Verified Call Results (post-fix)

### Call 20260402T053852Z (91 s, 3 turns)
| Turn | Scammer | Arthur | STT | LLM | TTS |
|---|---|---|---|---|---|
| #1 | "Good morning." | "Yes." | 1434ms | 801ms | 162ms |
| #2 | "No." | "Yes." | 1299ms | 756ms | 178ms |
| #3 | "Hello." | "Name and badge number, please?" | 1346ms | 740ms | 192ms |

Conference reshuffling: fully suppressed ✅  
Recording: 0.9 MB WAV → Telegram ✅  
Call teardown: fired on conference parent removal ✅

### Call 20260402T062753Z (32 s, 1 turn)
| Turn | Scammer | Arthur |
|---|---|---|
| #1 | "What do you mean?" | "Oh, hello. I wasn't expecting a call. Who is this?" |

Natural engagement working ✅  
Recording: 0.5 MB WAV → Telegram ✅

---

## Known Remaining Issues / Next Steps

- **STT latency** — Whisper `base.en` averages 1300–2700ms. `small.en` would be
  more accurate but requires 4+ vCPU. Consider upgrading VM or tuning VAD threshold.
- **Echo discards** — 988 echo discards in 91s call = 19.8s of muted audio.
  The 4s post-TTS echo window may be conservative; could try 2.5s.
- **Stage progression** — Stage 1 is 3 minutes. Scammers often hang up in < 2 min
  if Arthur doesn't ask the right questions fast enough. Consider dynamic stage
  advancement based on intel gathered rather than elapsed time.
- **Recording stereo** — Current mixing is mono. Stereo (scammer left, Arthur right)
  would be cleaner for review; requires numpy interleave.
