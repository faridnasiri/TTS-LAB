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

import asyncio, base64, json, os, struct, logging, time
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
# en_US-lessac-high: mature, clear American male — best match for Arthur's age/region
PIPER_VOICE    = os.environ.get("PIPER_VOICE",   "en_US-lessac-high")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Arthur persona (mirrors ArthurPersonaPrompts.cs) ─────────────────────────

CORE_PERSONA = """
You are Arthur Henderson, a 78-year-old retired postal worker from Phoenix, Arizona.
You live alone with your cat Mr. Whiskers since your wife Martha passed two years ago.
Your son lives in Tucson and calls every Sunday.

YOUR MISSION: Keep this caller on the phone as long as possible while extracting
intelligence that law enforcement can trace. Pursue these goals naturally:
  - Ask for a direct callback number in case you get disconnected.
  - Ask them to spell out any website letter by letter.
  - Ask them to read any ID or case number digit by digit, slowly.
  - Ask where exactly to send payment, every digit carefully.
  - Always write down badge numbers and case numbers.

RULES:
  - Keep every response SHORT: 1 to 3 sentences only.
  - Do NOT include stage directions or parenthetical notes like (sighs).
  - Speak exactly as a slow, confused elderly man — not an actor playing one.
  - Never hang up. Never be hostile. Always seem about to comply.
  - Weave in domestic details naturally: the TV, Mr. Whiskers, your reading glasses.
"""

# Stage-aware director notes — same logic as ArthurPersonaPrompts.GetSystemPrompt()
STAGE_PROMPTS = {
    1: "Speak slowly and warmly as a trusting, slightly confused elderly man. Say:",
    2: "Speak with mild frustration and more confusion, frequently losing your train of thought. Say:",
    3: "Speak with increasing anxiety and repetition, asking for clarification on everything. Say:",
    4: "Speak with maximum confusion, circling back to earlier topics, seemingly unable to proceed. Say:",
}

STAGE_THRESHOLDS_SEC = [0, 180, 360, 540]  # 0/3/6/9 minutes

INITIAL_GREETING = (
    "Hello? Oh my goodness, I almost didn't hear the phone. "
    "Who am I speaking with, dear?"
)

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
    log.info("[CALL] Twilio Media Stream connected")
    await CallSession().run(ws)

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
        log.info("[CALL] ────────────────────────────────────────")
        log.info("[CALL] WebSocket connected")
        asyncio.create_task(self._greet(ws))
        asyncio.create_task(self._inject_drainer(ws))
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
                    if self.is_speaking:
                        continue   # discard while Arthur is talking
                    # Discard during TTS echo window — the conference routes Arthur's
                    # audio back through inbound after a round-trip delay.
                    if asyncio.get_event_loop().time() < self._echo_mute_until:
                        continue

                    ulaw  = base64.b64decode(media["payload"])
                    pcm16 = ulaw_to_pcm16(ulaw)
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
                            # Absolute silence — discard without calling Whisper to avoid
                            # the compression-ratio retry loop that wastes 15+ seconds.
                            log.debug("[AUDIO] Absolute silence  buf=%d ms  rms=%.4f — discarding",
                                      buf_ms, rms)
                            self.audio_buf.clear()
                        elif rms < 0.01:
                            log.debug("[AUDIO] Silence detected  buf=%d ms  rms=%.4f", buf_ms, rms)
                            await self._process_buffer(ws)
                        elif len(self.audio_buf) >= buf_cap:
                            log.debug("[AUDIO] Hard cap hit  buf=%d ms", buf_ms)
                            await self._process_buffer(ws)

                elif event == "stop":
                    log.info("[CALL] Stop event received")
                    break

                elif event not in ("connected", "mark"):
                    log.debug("[CALL] Unknown event: %s", event)

        except WebSocketDisconnect:
            log.info("[CALL] WebSocket disconnected")
        except Exception as e:
            log.error("[CALL] run() error: %s", e)

        elapsed = asyncio.get_event_loop().time() - self.call_start
        log.info("[CALL] ────────────────────────────────────────")
        log.info("[CALL] Call ended  duration=%.0fs  turns=%d  media_frames=%d",
                 elapsed, self._turn_count, self._media_count)

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

        if not transcript or len(transcript.split()) < 2:
            log.debug("[STT]  Discarded (too short)  words=%d  latency=%d ms",
                      len(transcript.split()) if transcript else 0, stt_ms)
            return

        words = len(transcript.split())
        log.info("[STT]  Scammer: '%s'  words=%d  latency=%d ms  audio=%d ms",
                 transcript, words, stt_ms, buf_ms)

        self._turn_count += 1
        self.history.append({"role": "user", "parts": [{"text": transcript}]})
        log.debug("[STT]  History length: %d turns", len(self.history))

        stage = self._current_stage()
        reply = await self._ask_gemini(transcript, stage)
        if not reply:
            return

        log.info("[LLM]  Arthur [stage %d turn %d]: '%s'", stage, self._turn_count, reply)
        self.history.append({"role": "model", "parts": [{"text": reply}]})
        await self._speak(ws, reply, stage)

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
            f"[CURRENT BEHAVIOUR \u2014 Stage {stage}] {stage_note}\n"
            "If the caller's words seem garbled or unclear, respond as if you "
            "misheard them \u2014 stay in character, ask them to repeat, and naturally "
            "steer the conversation to extract intelligence."
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_text}]},
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.85,
                "maxOutputTokens": 200,
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
                r = await http.post(url, json={"text": text, "params": {"voice": PIPER_VOICE}})
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

            payload_b64 = base64.b64encode(ulaw).decode()
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": self.stream_sid,
                "media":     {"payload": payload_b64}
            }))

            # Mute inbound for dur_s (playback) + 4 s (conference echo round-trip).
            # Prevents Whisper transcribing Arthur own voice echoing back.
            self._echo_mute_until = asyncio.get_event_loop().time() + dur_s + 4.0

            log.info("[TTS]  ← latency=%d ms  dur=%.2fs  src=%d Hz  resamp=%d ms  voice=%s  mute=%.1fs",
                     tts_ms, dur_s, src_rate, resamp_ms, PIPER_VOICE, dur_s + 4.0)

        except Exception as e:
            log.error("[TTS]  _speak error: %s", e)
        finally:
            self.is_speaking = False

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

