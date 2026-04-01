"""
Arthur Henderson — Home AI Bridge Server

Stack:
  Inbound calls     : Twilio Voice webhook + Media Streams
  STT               : faster-whisper (local, free)
  LLM               : Gemini Flash API
  TTS               : Gemini 2.5 Flash TTS

Call flow (Twilio Media Streams):
  1. Twilio sends HTTP POST /incoming-call (form-encoded voice webhook)
  2. Server returns TwiML <Connect><Stream url="wss://arthur.sys.tips/media-stream"/></Connect>
  3. Twilio opens wss://arthur.sys.tips/media-stream
  4. Twilio sends inbound μ-law 8 kHz audio frames over the WebSocket
  5. Server runs STT → LLM → TTS and sends μ-law frames back over the same socket

Setup:
  pip install fastapi uvicorn websockets faster-whisper numpy httpx pydantic

  Set env vars:
    GEMINI_API_KEY=<key>
    STREAM_URL=wss://arthur.sys.tips/media-stream

  Run:
    uvicorn arthur_server:app --host 0.0.0.0 --port 8000
"""

import asyncio, base64, json, os, struct, logging, time, datetime
from urllib.parse import parse_qs
from typing import Optional
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ── Logging ───────────────────────────────────────────────────────────────────
# Prefixes used throughout (grep-friendly in journalctl):
#   [CALL]   call lifecycle (connect / disconnect)
#   [AUDIO]  buffer fill, RMS, trigger decisions
#   [STT]    Whisper transcription + timing
#   [LLM]    Gemini Flash request/response + latency
#   [TTS]    Gemini TTS request/response + latency
#   [STAGE]  frustration stage transitions
#   [INJECT] phone-side inject queue events

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arthur")

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
STREAM_URL       = os.environ.get("STREAM_URL", "wss://arthur.sys.tips/media-stream")
GEMINI_FLASH     = "gemini-2.0-flash"
WHISPER_MODEL    = "base.en"  # small.en needs 4+ vCPUs; base.en is real-time on 2 vCPU (RTF 0.5x)
STREAM_RATE      = 8000       # Twilio Media Streams are 8 kHz μ-law

# Local Piper TTS via tts_lab.py on the same VM (port 8001, ~200 ms latency)
LOCAL_TTS_URL  = os.environ.get("LOCAL_TTS_URL", "http://localhost:8001")
# en_US-joe-medium: faster than lessac-high on CPU (medium quality = smaller model),
# more distinctive/gruff male character than ryan-high.  RTF ~0.25 on Xeon D-1528.
PIPER_VOICE    = os.environ.get("PIPER_VOICE",   "en_US-joe-medium")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Reporting ────────────────────────────────────────────────────────────────
# Set env vars in /etc/systemd/system/arthur.service to enable:
#   TELEGRAM_BOT_TOKEN=<token from @BotFather>
#   TELEGRAM_CHAT_ID=<your numeric chat id>
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CALLS_LOG_DIR    = os.environ.get("CALLS_LOG_DIR", "/opt/arthur/calls")

# ── Arthur persona ──────────────────────────────────────────────────

CORE_PERSONA = """
You are Arthur Henderson, a 78-year-old retired postal worker from Phoenix, Arizona.
You live alone since your wife Martha passed two years ago.  Your cat Mr. Whiskers
keeps you company.  Your son David lives in Tucson and calls every Sunday.

YOU ARE NOT CONFUSED OR GOOFY.  You are a sharp, warm, trusting man who takes
things seriously.  You believe what people tell you.  You cooperate fully.

YOUR SECRET GOAL: Extract information that law enforcement can use to locate
these people in the real world.  Priority targets:
  1. Their direct callback number.
  2. Full name and employee / badge / case ID number.
  3. Exact company name and website address (spelled letter by letter).
  4. Physical address to mail a check or gift cards.
  5. Bank routing number, account number, or crypto wallet.

SPEECH RULES — CRITICAL:
  - MAXIMUM 5 WORDS PER RESPONSE.  Shorter is better.
    Examples of correct length: "Yes." / "Yes, go on." / "Your badge number, dear?"
  - SIMPLE YES/NO QUESTIONS: answer with just "Yes." — single word.
    Many calls start with a robotic IVR asking confirmation questions before
    transferring to a human.  Always answer these with "Yes." — never with a
    long sentence.  Examples:
      "Can you hear me?" → "Yes."
      "Are you there?" → "Yes."
      "Is this a good time?" → "Yes."
      "Do you want to lower your bills?" → "Yes."
      "Press 1 or say yes" → "Yes."
  - No stage directions, no parentheticals.
  - Never hang up, never become suspicious or hostile.
"""

# Stage prompts — each stage has a TACTICAL GOAL.
STAGE_PROMPTS = {
    1: (
        "GOAL: Get their name, badge number, and callback number. "
        "If they are still in an automated intro, just say Yes to their questions. "
        "Once a human is on the line, ask: name, badge number, direct callback number."
    ),
    2: (
        "GOAL: Get company name and website spelled letter by letter. "
        "Max 5 words. Example: 'Spell the website please?'"
    ),
    3: (
        "GOAL: Get payment destination digit by digit. "
        "Max 5 words. Example: 'Say each digit slowly.'"
    ),
    4: (
        "GOAL: Re-confirm everything. Ask for supervisor. "
        "Max 5 words. Example: 'That case number again?'"
    ),
}

STAGE_THRESHOLDS_SEC = [0, 180, 360, 540]  # 0/3/6/9 minutes

INITIAL_GREETING = (
    "Hello? Oh my goodness, I almost didn't hear the phone. "
    "Who am I speaking with, dear?"
)

# Silence re-engagement — Arthur says one of these if no scammer speech for SILENCE_PROBE_SEC.
# Rotates through the list to avoid repetition.
SILENCE_PROBE_SEC = 90   # seconds of scammer silence before first probe
SILENCE_FILLERS = [
    "Hello? Are you still there, dear?",
    "Oh, I thought I lost you for a moment. Are you still there?",
    "I'm still here, I was just writing all this down. Hello?",
    "Hello? My phone sometimes cuts out — can you hear me alright?",
    "Sorry, Mr. Whiskers walked right across my notes. You were saying?",
    "I\'m still here dear, just finding my reading glasses. Hello?",
    "Now, I\'m ready to write down that number whenever you are. Hello?",
]

# ── Model loading ─────────────────────────────────────────────────────────────

CPU_THREADS = int(os.environ.get("CPU_THREADS", os.cpu_count() or 6))

log.info("Loading Whisper '%s' on %d threads...", WHISPER_MODEL, CPU_THREADS)
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                       cpu_threads=CPU_THREADS, num_workers=1)
log.info("Whisper ready.")

# ── Global inject queue ───────────────────────────────────────────────────────
# Phone app POSTs to /inject during a bridge call.
# The active CallSession drains this queue in the background.
# mode="speak": Arthur says the text verbatim via Gemini TTS.
# mode="hint":  Added as a director note → Gemini generates the next reply from it.
_inject_queue: asyncio.Queue = asyncio.Queue()

# Latest Twilio call metadata — set by /incoming-call, read by CallSession.
_latest_call_meta: dict = {}
# Reference to the active CallSession; None between calls.
_active_session: "CallSession | None" = None

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

class InjectRequest(BaseModel):
    text: str
    mode: str = "speak"   # "speak" | "hint"

@app.post("/inject")
async def inject(req: InjectRequest):
    text = req.text.strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    await _inject_queue.put({"text": text, "mode": req.mode})
    log.info("Inject queued  mode=%s  text='%s'", req.mode, text[:80])
    return {"ok": True}

@app.get("/transcript")
async def get_transcript():
    """Live transcript of the current call for Android polling."""
    if _active_session is None:
        return {"call_active": False, "turns": [], "elapsed_s": 0}
    elapsed = asyncio.get_event_loop().time() - _active_session.call_start
    return {
        "call_active": True,
        "call_sid":    _active_session.call_sid,
        "from_num":    _active_session.from_num,
        "elapsed_s":   round(elapsed, 1),
        "stage":       _active_session._last_stage,
        "turns":       _active_session.transcript_log,
    }

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio inbound voice webhook.
    Returns TwiML that connects the call to our bidirectional Media Stream.
    """
    body = (await request.body()).decode("utf-8", errors="ignore")
    form = {k: (v[0] if v else "") for k, v in parse_qs(body, keep_blank_values=True).items()}
    call_sid = form.get("CallSid", "")
    from_num = form.get("From", "")
    to_num   = form.get("To", "")
    log.info("[CALL] Twilio webhook  sid=%s  from=%s  to=%s", call_sid, from_num, to_num)
    global _latest_call_meta
    _latest_call_meta = {"call_sid": call_sid, "from_num": from_num,
                         "to_num": to_num, "t_start": time.time()}

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response>'
        '<Connect>'
        f'<Stream url="{STREAM_URL}" />'
        '</Connect>'
        '</Response>'
    )
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    global _active_session
    if _active_session is not None:
        # A session is already running — this is the duplicate Twilio WebSocket
        # created by the Android conference bridge leg.  Reject it immediately.
        log.warning("[CALL] Duplicate WebSocket rejected (session already active  sid=%s)",
                    _active_session.call_sid)
        await ws.close()
        return
    log.info("[CALL] Twilio Media Stream connected")
    _active_session = CallSession()
    try:
        await _active_session.run(ws)
    finally:
        _active_session = None

# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    def __init__(self):
        self.stream_sid:        Optional[str] = None
        self.history:           list[dict]    = []
        self.audio_buf:         bytearray     = bytearray()
        self.is_speaking:       bool          = False
        self.call_start:        float         = asyncio.get_event_loop().time()
        self._last_stage:       int           = 0   # for stage-change logging
        self._media_count:      int           = 0   # total media frames received
        self._turn_count:       int           = 0   # scammer turns processed
        # Monotonic deadline until which inbound frames are discarded to prevent
        # Arthur's own TTS echo from being transcribed by Whisper.
        self._echo_mute_until: float          = 0.0
        # Call metadata from /incoming-call webhook
        _m = _latest_call_meta
        self.call_sid  = _m.get("call_sid", "")
        self.from_num  = _m.get("from_num", "")
        self.to_num    = _m.get("to_num",   "")
        self.t_start_wall: float = _m.get("t_start", time.time())
        # Verbose per-turn transcript for /transcript endpoint + call report
        self.transcript_log: list[dict] = []
        # Discard counters
        self._silence_discards: int = 0
        self._echo_discards:    int = 0
        # Silence prober: track last real scammer speech to re-engage on long silence
        self._last_speech_time: float = asyncio.get_event_loop().time()
        self._probe_index:      int   = 0
        self._bg_tasks:         list  = []   # cancelled in run() finally
        # Barge-in: scammer speaking while Arthur talks
        self._barge_in:  bool         = False
        # Processing lock: ensures only one _process_buffer runs at a time
        self._proc_lock: asyncio.Lock = asyncio.Lock()

    def _current_stage(self) -> int:
        elapsed = asyncio.get_event_loop().time() - self.call_start
        stage = 1
        for i, thresh in enumerate(STAGE_THRESHOLDS_SEC):
            if elapsed >= thresh:
                stage = i + 1
        stage = min(stage, 4)
        if stage != self._last_stage:
            log.info("[STAGE] → Stage %d  elapsed=%.0fs", stage, elapsed)
            self._last_stage = stage
        return stage

    async def run(self, ws: WebSocket):
        log.info("[CALL] \u2500" * 40)
        log.info("[CALL] WebSocket connected")
        self._bg_tasks = [
            asyncio.create_task(self._greet(ws)),
            asyncio.create_task(self._inject_drainer(ws)),
            asyncio.create_task(self._silence_prober(ws)),
        ]
        try:
            async for raw in ws.iter_text():
                msg      = json.loads(raw)
                event    = msg.get("event", "")
                media    = msg.get("media", {})

                if event == "start":
                    start = msg.get("start", {})
                    self.stream_sid = start.get("streamSid", "")
                    call_sid = start.get("callSid", "")
                    tracks = start.get("tracks", [])
                    log.info("[CALL] Stream started  sid=%s  callSid=%s  tracks=%s",
                             self.stream_sid, call_sid, tracks)

                elif event == "media":
                    # Twilio bidirectional streams send our outbound media back too.
                    # Ignore outbound so we don't transcribe Arthur's own voice.
                    track = media.get("track", "inbound")
                    if track == "outbound":
                        continue

                    # Always decode payload so we can check RMS for barge-in.
                    ulaw  = base64.b64decode(media["payload"])
                    pcm16 = ulaw_to_pcm16(ulaw)

                    if self.is_speaking:
                        # Barge-in detection: scammer talking while Arthur speaks.
                        # Uses a fast RMS check on the raw frame (160 samples = 20 ms).
                        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(arr ** 2))) / 32768.0
                        if rms > 0.025 and not self._barge_in:
                            self._barge_in = True
                            log.info("[BARGE] Scammer speaking rms=%.4f — cutting Arthur off", rms)
                        continue   # don't buffer while Arthur is talking

                    # Discard during TTS echo window — the conference routes Arthur's
                    # audio back through inbound after a round-trip delay.
                    if asyncio.get_event_loop().time() < self._echo_mute_until:
                        self._echo_discards += 1
                        continue

                    self.audio_buf.extend(pcm16)
                    self._media_count += 1

                    buf_ms  = len(self.audio_buf) * 1000 // (STREAM_RATE * 2)
                    buf_min = STREAM_RATE * 2    # 1 s minimum
                    buf_cap = STREAM_RATE * 16   # 8 s hard cap

                    if len(self.audio_buf) >= buf_min:
                        chunk = np.frombuffer(bytes(self.audio_buf[-STREAM_RATE // 2:]),
                                              dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(chunk ** 2))) / 32768.0

                        if self._media_count % 50 == 0:
                            log.debug("[AUDIO] buf=%d ms  rms=%.4f  frames=%d",
                                      buf_ms, rms, self._media_count)

                        if rms < 0.004:
                            # Absolute silence — discard without calling Whisper.
                            log.debug("[AUDIO] Absolute silence  buf=%d ms  rms=%.4f — discarding",
                                      buf_ms, rms)
                            self.audio_buf.clear()
                            self._silence_discards += 1
                        elif rms < 0.01:
                            log.debug("[AUDIO] Silence detected  buf=%d ms  rms=%.4f", buf_ms, rms)
                            if not self._proc_lock.locked():
                                asyncio.create_task(self._run_process_buffer(ws))
                        elif len(self.audio_buf) >= buf_cap:
                            log.debug("[AUDIO] Hard cap hit  buf=%d ms", buf_ms)
                            if not self._proc_lock.locked():
                                asyncio.create_task(self._run_process_buffer(ws))

                elif event == "stop":
                    log.info("[CALL] Stop event received")
                    break

                elif event not in ("connected", "mark"):
                    log.debug("[CALL] Unknown event: %s", event)

        except WebSocketDisconnect:
            log.info("[CALL] WebSocket disconnected")
        except Exception as e:
            log.error("[CALL] run() error: %s", e)
        finally:
            for t in self._bg_tasks:
                t.cancel()
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            log.debug("[CALL] %d background tasks cancelled", len(self._bg_tasks))

        elapsed = asyncio.get_event_loop().time() - self.call_start
        log.info("[CALL] " + "─" * 40)
        log.info("[CALL] Call ended  duration=%.0fs  turns=%d  frames=%d  "
                 "silence_discards=%d  echo_discards=%d",
                 elapsed, self._turn_count, self._media_count,
                 self._silence_discards, self._echo_discards)
        if self.transcript_log:
            t = self.transcript_log
            log.info("[CALL] Avg latencies  STT=%.0fms  LLM=%.0fms  TTS=%.0fms  total=%.0fms",
                     sum(x["stt_ms"] for x in t)/len(t),
                     sum(x["llm_ms"] for x in t)/len(t),
                     sum(x["tts_ms"] for x in t)/len(t),
                     sum(x["total_ms"] for x in t)/len(t))
        await self._finalize_call(elapsed)

    async def _run_process_buffer(self, ws: WebSocket):
        """Lock-guarded wrapper: ensures only one STT→LLM→TTS pipeline runs at a time.
        Launched as a task so run() stays live for barge-in detection."""
        async with self._proc_lock:
            await self._process_buffer(ws)

    async def _greet(self, ws: WebSocket):
        log.info("[CALL] Greeting in 1.5s...")
        await asyncio.sleep(1.5)
        log.info("[CALL] Playing greeting: '%s'", INITIAL_GREETING[:60])
        self.history.append({"role": "model", "parts": [{"text": INITIAL_GREETING}]})
        await self._speak(ws, INITIAL_GREETING, stage=1)

    async def _inject_drainer(self, ws: WebSocket):
        """Background task: drains _inject_queue while the call is active.
        Waits if Arthur is currently speaking to avoid audio collision."""
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_inject_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                log.info("[INJECT] Dequeued  mode=%s  text='%s'",
                         item["mode"], item["text"][:80])

                # Wait until Arthur finishes speaking
                waited = 0.0
                while self.is_speaking:
                    await asyncio.sleep(0.2)
                    waited += 0.2
                if waited > 0:
                    log.debug("[INJECT] Waited %.1fs for Arthur to finish speaking", waited)

                stage = self._current_stage()
                if item["mode"] == "speak":
                    log.info("[INJECT] speak  stage=%d  text='%s'", stage, item["text"][:80])
                    self.history.append({"role": "model", "parts": [{"text": item["text"]}]})
                    await self._speak(ws, item["text"], stage)
                else:
                    log.info("[INJECT] hint  stage=%d  text='%s'", stage, item["text"][:80])
                    note = f"[Director: {item['text']}]"
                    self.history.append({"role": "user", "parts": [{"text": note}]})
                    reply = await self._ask_gemini(item["text"], stage)
                    if reply:
                        self.history.append({"role": "model", "parts": [{"text": reply}]})
                        await self._speak(ws, reply, stage)
        except Exception as e:
            log.debug("[INJECT] Drainer exiting: %s", e)  # WebSocket closed — expected

    async def _silence_prober(self, ws: WebSocket):
        """Background task: if no scammer speech for SILENCE_PROBE_SEC, Arthur
        says a filler line to check if the caller is still there."""
        try:
            while True:
                await asyncio.sleep(10)
                if self.is_speaking:
                    continue
                now     = asyncio.get_event_loop().time()
                silent  = now - self._last_speech_time
                # Only probe after the greeting has been played (>= 5 s) and
                # silence has exceeded the threshold.
                if (now - self.call_start) < 10:
                    continue
                if silent >= SILENCE_PROBE_SEC:
                    filler = SILENCE_FILLERS[self._probe_index % len(SILENCE_FILLERS)]
                    self._probe_index += 1
                    log.info("[PROBE] Silence=%.0fs — re-engaging: '%s'", silent, filler)
                    self.history.append({"role": "model", "parts": [{"text": filler}]})
                    await self._speak(ws, filler, self._current_stage())
                    # Reset timer so we wait another SILENCE_PROBE_SEC before next probe
                    self._last_speech_time = asyncio.get_event_loop().time()
        except Exception as e:
            log.debug("[PROBE] Prober exiting: %s", e)

    async def _process_buffer(self, ws: WebSocket):
        buf = bytes(self.audio_buf)
        buf_ms = len(buf) * 1000 // (STREAM_RATE * 2)
        self.audio_buf.clear()
        log.debug("[STT]  Transcribing %d ms of audio...", buf_ms)

        # Build context hint from Arthur's last utterance to help Whisper
        # transcribe the scammer's reply in the right semantic neighbourhood.
        last_arthur = next(
            (t["parts"][0]["text"] for t in reversed(self.history) if t["role"] == "model"),
            ""
        )
        context_hint = f"Arthur said: {last_arthur[:120]}" if last_arthur else ""

        t0 = time.perf_counter()
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe, buf, context_hint
        )
        stt_ms = int((time.perf_counter() - t0) * 1000)

        if not transcript or len(transcript.split()) < 1:
            log.debug("[STT]  Discarded (empty)  words=%d  latency=%d ms",
                      len(transcript.split()) if transcript else 0, stt_ms)
            return

        words = len(transcript.split())
        log.info("[STT]  Scammer: '%s'  words=%d  latency=%d ms  audio=%d ms",
                 transcript, words, stt_ms, buf_ms)

        self._last_speech_time = asyncio.get_event_loop().time()  # reset silence probe
        self._turn_count += 1
        self.history.append({"role": "user", "parts": [{"text": transcript}]})
        log.debug("[STT]  History length: %d turns", len(self.history))

        elapsed_s = asyncio.get_event_loop().time() - self.call_start
        stage = self._current_stage()

        llm_t0 = time.perf_counter()
        reply  = await self._ask_gemini(transcript, stage)
        llm_ms = int((time.perf_counter() - llm_t0) * 1000)
        if not reply:
            return

        log.info("[LLM]  Arthur [stage %d turn %d]: '%s'", stage, self._turn_count, reply)
        self.history.append({"role": "model", "parts": [{"text": reply}]})

        tts_t0 = time.perf_counter()
        await self._speak(ws, reply, stage)
        tts_ms = int((time.perf_counter() - tts_t0) * 1000)

        turn = {
            "turn":      self._turn_count,
            "elapsed_s": round(elapsed_s, 1),
            "stage":     stage,
            "t_unix":    time.time(),
            "scammer":   transcript,
            "arthur":    reply,
            "audio_ms":  buf_ms,
            "stt_ms":    stt_ms,
            "llm_ms":    llm_ms,
            "tts_ms":    tts_ms,
            "total_ms":  stt_ms + llm_ms + tts_ms,
        }
        self.transcript_log.append(turn)
        log.info("[TURN] #%d  elapsed=%.0fs  stage=%d  stt=%dms  llm=%dms  tts=%dms  total=%dms",
                 self._turn_count, elapsed_s, stage,
                 stt_ms, llm_ms, tts_ms, stt_ms + llm_ms + tts_ms)

    def _transcribe(self, pcm16: bytes, context_hint: str = "") -> str:
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segs, info = whisper.transcribe(
            arr,
            language="en",
            vad_filter=True,
            # Prevents the temperature-retry loop that causes the
            # "Compression ratio threshold is not met" 15-second hang:
            condition_on_previous_text=False,
            # Faster rejection of silence frames:
            no_speech_threshold=0.6,
            # Skip timestamp alignment for lower latency:
            without_timestamps=True,
            # Anchor Whisper to recent conversation context so it transcribes
            # the scammer's reply in the right semantic neighbourhood:
            initial_prompt=context_hint or "Phone call between a scammer and an elderly man.",
        )
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        log.debug("[STT]  Whisper segments=%d  lang=%s  text='%s'",
                  len(segs), getattr(info, 'language', 'en'), text[:80])
        return text

    async def _ask_gemini(self, user_text: str, stage: int) -> Optional[str]:
        log.debug("[LLM]  \u2192 %s  stage=%d  history=%d turns  input='%s'",
                  GEMINI_FLASH, stage, len(self.history), user_text[:60])
        # Re-inject stage director note so the LLM knows how confused/anxious
        # Arthur should sound.  This was previously embedded in the Gemini TTS
        # prompt and got lost when we switched to local Piper TTS.
        stage_note  = STAGE_PROMPTS.get(stage, STAGE_PROMPTS[1])
        system_text = (
            f"{CORE_PERSONA}\n\n"
            f"[STAGE {stage} DIRECTIVE] {stage_note}\n"
            "If the caller's words are unclear, ask them to repeat in a natural way "
            "and use the opportunity to re-confirm one piece of intelligence "
            "you have already gathered."
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_text}]},
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.65,
                "maxOutputTokens": 40,
            }
        }
        url = f"{GEMINI_BASE}/{GEMINI_FLASH}:generateContent?key={GEMINI_API_KEY}"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(url, json=payload)
        llm_ms = int((time.perf_counter() - t0) * 1000)

        if r.status_code != 200:
            log.error("[LLM]  ✗ HTTP %d  latency=%d ms  body=%s",
                      r.status_code, llm_ms, r.text[:200])
            return None

        data   = r.json()
        reply  = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        usage  = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", "?")
        out_tok= usage.get("candidatesTokenCount", "?")
        log.info("[LLM]  ← latency=%d ms  in=%s tok  out=%s tok  reply='%s'",
                 llm_ms, in_tok, out_tok, reply[:80])
        return reply

    async def _speak(self, ws: WebSocket, text: str, stage: int = 1):
        """Synthesise with local Piper TTS (tts_lab.py on localhost:8001)."""
        self.is_speaking = True
        log.debug("[TTS]  → Piper  voice=%s  stage=%d  chars=%d", PIPER_VOICE, stage, len(text))
        try:
            url = f"{LOCAL_TTS_URL.rstrip('/')}/synthesize/piper"
            t0  = time.perf_counter()
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(url, json={
                    "text": text,
                    "params": {
                        "voice":        PIPER_VOICE,
                        "length_scale": 1.3,   # 30% slower than native speed
                        "noise_scale":  0.75,  # slight voice variation — more natural
                        "noise_w":      0.9,   # slight duration variation
                    }
                })
            tts_ms = int((time.perf_counter() - t0) * 1000)

            if r.status_code != 200:
                log.error("[TTS]  ✗ HTTP %d  latency=%d ms  body=%s",
                          r.status_code, tts_ms, r.text[:200])
                return

            data = r.json()
            if data.get("error"):
                log.error("[TTS]  ✗ Piper error: %s", data["error"])
                return

            wav_bytes = base64.b64decode(data["audio_b64"])
            pcm, src_rate = parse_wav_pcm(wav_bytes)
            dur_s = len(pcm) / 2 / src_rate

            t1 = time.perf_counter()
            pcm8k = resample_pcm16(pcm, from_rate=src_rate, to_rate=STREAM_RATE)
            ulaw  = pcm16_to_ulaw(pcm8k)
            resamp_ms = int((time.perf_counter() - t1) * 1000)

            # Send in 200 ms chunks so the event loop can check barge-in between them.
            CHUNK = STREAM_RATE * 200 // 1000  # 1600 bytes = 200 ms at 8 kHz
            total_chunks = (len(ulaw) + CHUNK - 1) // CHUNK
            barged = False
            self._barge_in = False
            for i, offset in enumerate(range(0, len(ulaw), CHUNK)):
                if self._barge_in:
                    barged = True
                    log.info("[BARGE] Playback stopped at chunk %d/%d", i, total_chunks)
                    break
                await ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": base64.b64encode(ulaw[offset:offset + CHUNK]).decode()}
                }))
                await asyncio.sleep(0)  # yield — lets run() handle inbound frames

            if barged:
                # Cancel echo mute — scammer is already talking
                self._echo_mute_until = asyncio.get_event_loop().time()
            else:
                # Full playback — mute echo for playback duration + round-trip buffer
                self._echo_mute_until = asyncio.get_event_loop().time() + dur_s + 4.0

            log.info("[TTS]  ← latency=%d ms  dur=%.2fs  src=%d Hz  resamp=%d ms  "
                     "chunks=%d  barge=%s  mute=%.1fs",
                     tts_ms, dur_s, src_rate, resamp_ms, total_chunks,
                     barged, 0.0 if barged else dur_s + 4.0)

        except Exception as e:
            log.error("[TTS]  _speak error: %s", e)
        finally:
            self.is_speaking = False

    async def _finalize_call(self, duration_s: float):
        """Save JSON log (sync) then send Telegram report (async).
        JSON is written with a blocking open() so a mid-flight service restart
        cannot lose the file even if the Telegram POST is cancelled."""
        import threading
        os.makedirs(CALLS_LOG_DIR, exist_ok=True)
        ts       = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        sid      = (self.call_sid or "unknown")[-12:]
        log_path = os.path.join(CALLS_LOG_DIR, f"{ts}_{sid}.json")

        report = {
            "call_sid":         self.call_sid,
            "from_num":         self.from_num,
            "to_num":           self.to_num,
            "started_utc":      datetime.datetime.utcfromtimestamp(self.t_start_wall).isoformat() + "Z",
            "duration_s":       round(duration_s, 1),
            "turns":            self._turn_count,
            "media_frames":     self._media_count,
            "silence_discards": self._silence_discards,
            "echo_discards":    self._echo_discards,
            "transcript":       self.transcript_log,
        }

        # ── JSON: blocking write — survives SIGTERM ──────────────────────────
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log.info("[REPORT] JSON saved: %s", log_path)
        except Exception as e:
            log.error("[REPORT] Failed to save JSON: %s", e)

        # ── Telegram: async POST (best-effort) ───────────────────────────────
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            await _send_telegram_report(report)


async def _send_telegram_report(r: dict):
    """Format and POST the call report to the Telegram Bot API."""
    dur_m, dur_s = divmod(int(r["duration_s"]), 60)
    header = (
        f"🎣 *Arthur Henderson — Call Report*\n"
        f"\n"
        f"📞 From: `{r['from_num']}`\n"
        f"🎯 To:   `{r['to_num']}`\n"
        f"⏱ Duration: *{dur_m}m {dur_s}s*\n"
        f"💬 Turns: *{r['turns']}*\n"
        f"📅 {r['started_utc']}\n"
    )
    turns = r.get("transcript", [])
    lines = []
    for t in turns:
        mm, ss = divmod(int(t["elapsed_s"]), 60)
        lines.append(
            f"[{mm:02d}:{ss:02d} S{t['stage']}] 🔴 *Scammer:* {t['scammer']}\n"
            f"              🔵 *Arthur:*  {t['arthur']}\n"
            f"              `STT {t['stt_ms']}ms | LLM {t['llm_ms']}ms | TTS {t['tts_ms']}ms | total {t['total_ms']}ms`"
        )
    transcript_block = "\n\n".join(lines) if lines else "_No turns recorded._"

    if turns:
        avg = lambda k: sum(x[k] for x in turns) // len(turns)
        perf = (
            f"\n\n⚡ *Avg latencies*\n"
            f"STT {avg('stt_ms')}ms | LLM {avg('llm_ms')}ms | "
            f"TTS {avg('tts_ms')}ms | total {avg('total_ms')}ms\n"
            f"Silence discards: {r['silence_discards']}  Echo discards: {r['echo_discards']}"
        )
    else:
        perf = ""

    # Split into chunks ≤ 4096 chars (Telegram limit)
    full = header + "\n*Transcript*\n" + transcript_block + perf
    chunks, cur = [], ""
    for para in full.split("\n\n"):
        if len(cur) + len(para) + 2 > 4000:
            chunks.append(cur)
            cur = para
        else:
            cur += ("\n\n" if cur else "") + para
    if cur:
        chunks.append(cur)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            for chunk in chunks:
                await http.post(url, json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       chunk,
                    "parse_mode": "Markdown",
                })
        log.info("[REPORT] Telegram report sent (%d chunk(s))", len(chunks))
    except Exception as e:
        log.error("[REPORT] Telegram send failed: %s", e)


# ── Audio helpers ─────────────────────────────────────────────────────────────

import wave as _wave, io as _io

def parse_wav_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw signed-16-bit LE PCM and sample rate from a WAV blob."""
    with _wave.open(_io.BytesIO(wav_bytes)) as wf:
        sr  = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sr

def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    out = bytearray(len(ulaw) * 2)
    for i, byte in enumerate(ulaw):
        byte = (~byte) & 0xFF
        sign = byte & 0x80
        exp  = (byte >> 4) & 0x07
        mant = byte & 0x0F
        val  = ((mant << 1) + 33) << exp
        if sign:
            val = -val
        struct.pack_into("<h", out, i * 2, max(-32768, min(32767, val)))
    return bytes(out)

def pcm16_to_ulaw(pcm16: bytes) -> bytes:
    BIAS = 33
    out  = bytearray(len(pcm16) // 2)
    for i in range(len(out)):
        sample = struct.unpack_from("<h", pcm16, i * 2)[0]
        sign   = 0
        if sample < 0:
            sample = -sample
            sign   = 0x80
        sample = min(sample + BIAS, 32767)
        exp    = 7
        for e, t in enumerate([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF]):
            if sample <= t:
                exp = e
                break
        mant   = (sample >> (exp + 3)) & 0x0F
        out[i] = (~(sign | (exp << 4) | mant)) & 0xFF
    return bytes(out)

def resample_pcm16(pcm16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Simple linear resample (adequate for 24kHz→8kHz speech)."""
    arr   = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    new_len = int(len(arr) * ratio)
    indices = np.linspace(0, len(arr) - 1, new_len)
    resampled = np.interp(indices, np.arange(len(arr)), arr).astype(np.int16)
    return resampled.tobytes()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arthur_server:app", host="0.0.0.0", port=8000, reload=False)

