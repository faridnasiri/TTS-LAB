"""
Arthur Henderson — Home AI Bridge Server

Stack:
  Inbound calls     : Telnyx Call Control API (Programmable Voice)
  STT               : faster-whisper (local, free)
  LLM               : Gemini Flash API
  TTS               : Gemini 2.5 Flash TTS

Call flow (Call Control — NOT TeXML):
  1. Telnyx POST /incoming-call  { event_type: "call.initiated", payload: { call_control_id } }
  2. Server responds 200 OK {"result":"ok"}
  3. Server calls POST /v2/calls/{ccid}/actions/answer
  4. Server calls POST /v2/calls/{ccid}/actions/streaming_start  → WebSocket URL
  5. Telnyx opens wss://arthur.sys.tips/media-stream
  6. Bidirectional PCMU 8 kHz audio stream (same WebSocket protocol as TeXML)

Setup:
  pip install fastapi uvicorn websockets faster-whisper numpy httpx pydantic

  Set env vars:
    GEMINI_API_KEY=<key>
    TELNYX_API_KEY=<key from portal.telnyx.com → API Keys>

  Run:
    uvicorn arthur_server:app --host 0.0.0.0 --port 8000
"""

import asyncio, base64, json, os, struct, logging, time
from typing import Optional
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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
TELNYX_API_KEY   = os.environ.get("TELNYX_API_KEY", "")
TELNYX_API_BASE  = "https://api.telnyx.com/v2"
STREAM_URL       = os.environ.get("STREAM_URL", "wss://arthur.sys.tips/media-stream")
GEMINI_FLASH     = "gemini-2.0-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Gacrux"   # Mature — same as Android app default
WHISPER_MODEL    = "base.en"  # small.en needs 4+ vCPUs; base.en is real-time on 2 vCPU (RTF 0.5x)
TELNYX_RATE      = 8000       # Telnyx MediaStream is 8 kHz μ-law

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
    Telnyx Call Control webhook — receives JSON events, NOT XML.
    Responds 200 OK immediately, then issues answer + streaming_start
    via the Telnyx REST API in the background.
    """
    try:
        body = await request.json()
    except Exception:
        return {"result": "ok"}   # ignore malformed

    data     = body.get("data", {})
    ev_type  = data.get("event_type", "")
    payload  = data.get("payload", {})
    ccid     = payload.get("call_control_id", "")

    log.info("[CALL] Webhook  event=%s  ccid=%.20s…", ev_type, ccid or "?")

    if ev_type == "call.initiated":
        if not TELNYX_API_KEY:
            log.error("[CALL] TELNYX_API_KEY not set — cannot answer call")
        elif not ccid:
            log.error("[CALL] call_control_id missing in payload")
        else:
            asyncio.create_task(_answer_and_stream(ccid))

    # Always ACK immediately — Telnyx expects 200 within 5 s
    return {"result": "ok"}


async def _answer_and_stream(ccid: str):
    """Answer the call and start the WebSocket media stream."""
    headers = {
        "Authorization": f"Bearer {TELNYX_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10) as http:
        # 1. Answer
        r = await http.post(f"{TELNYX_API_BASE}/calls/{ccid}/actions/answer", headers=headers, json={})
        log.info("[CALL] answer → HTTP %d", r.status_code)
        if r.status_code not in (200, 201):
            log.error("[CALL] answer failed: %s", r.text[:200])
            return

        await asyncio.sleep(0.8)   # brief pause — call must be ACTIVE before streaming

        # 2. Start bidirectional media stream
        r = await http.post(
            f"{TELNYX_API_BASE}/calls/{ccid}/actions/streaming_start",
            headers=headers,
            json={
                "stream_url":   STREAM_URL,
                "stream_track": "inbound_track",   # scammer audio only → our STT
            },
        )
        log.info("[CALL] streaming_start → HTTP %d  url=%s", r.status_code, STREAM_URL)
        if r.status_code not in (200, 201):
            log.error("[CALL] streaming_start failed: %s", r.text[:200])

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Call connected")
    await CallSession().run(ws)

# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    def __init__(self):
        self.stream_sid:   Optional[str] = None
        self.history:      list[dict]    = []
        self.audio_buf:    bytearray     = bytearray()
        self.is_speaking:  bool          = False
        self.call_start:   float         = asyncio.get_event_loop().time()
        self._last_stage:  int           = 0   # for stage-change logging
        self._media_count: int           = 0   # total media frames received
        self._turn_count:  int           = 0   # scammer turns processed

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
                # Call Control uses "event_type"; TeXML used "event" — support both
                event    = msg.get("event_type") or msg.get("event", "")
                media    = msg.get("media", {})

                if event == "start":
                    start = msg.get("start", {})
                    # Call Control: stream_id   TeXML: streamSid
                    self.stream_sid = start.get("stream_id") or start.get("streamSid", "")
                    log.info("[CALL] Stream started  id=%s", self.stream_sid)

                elif event == "media":
                    # Call Control sends both inbound (scammer) and outbound (our TTS echo).
                    # Only process inbound so we don't transcribe our own audio.
                    track = media.get("track", "inbound")
                    if track == "outbound":
                        continue
                    if self.is_speaking:
                        continue   # discard while Arthur is talking

                    ulaw  = base64.b64decode(media["payload"])
                    pcm16 = ulaw_to_pcm16(ulaw)
                    self.audio_buf.extend(pcm16)
                    self._media_count += 1

                    buf_ms  = len(self.audio_buf) * 1000 // (TELNYX_RATE * 2)
                    buf_min = TELNYX_RATE * 2    # 1 s minimum
                    buf_cap = TELNYX_RATE * 16   # 8 s hard cap

                    if len(self.audio_buf) >= buf_min:
                        chunk = np.frombuffer(bytes(self.audio_buf[-TELNYX_RATE // 2:]),
                                              dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(chunk ** 2))) / 32768.0

                        if self._media_count % 50 == 0:
                            log.debug("[AUDIO] buf=%d ms  rms=%.4f  frames=%d",
                                      buf_ms, rms, self._media_count)

                        if rms < 0.01:
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
        buf_ms = len(buf) * 1000 // (TELNYX_RATE * 2)
        self.audio_buf.clear()
        log.debug("[STT]  Transcribing %d ms of audio...", buf_ms)

        t0 = time.perf_counter()
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe, buf
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

    def _transcribe(self, pcm16: bytes) -> str:
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segs, info = whisper.transcribe(arr, language="en", vad_filter=True)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        log.debug("[STT]  Whisper segments=%d  lang=%s  text='%s'",
                  len(segs), getattr(info, 'language', 'en'), text[:80])
        return text

    async def _ask_gemini(self, user_text: str, stage: int) -> Optional[str]:
        log.debug("[LLM]  → %s  stage=%d  history=%d turns  input='%s'",
                  GEMINI_FLASH, stage, len(self.history), user_text[:60])
        payload = {
            "system_instruction": {"parts": [{"text": CORE_PERSONA}]},
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 120,
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
        """Synthesise with Gemini TTS (Gacrux/Mature) + stage director's notes."""
        self.is_speaking = True
        log.debug("[TTS]  → %s  voice=%s  stage=%d  chars=%d",
                  GEMINI_TTS_MODEL, GEMINI_TTS_VOICE, stage, len(text))
        try:
            director_note = STAGE_PROMPTS.get(stage, STAGE_PROMPTS[1])
            full_prompt   = f"{director_note} {text}"

            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": GEMINI_TTS_VOICE}
                        }
                    }
                }
            }

            url = f"{GEMINI_BASE}/{GEMINI_TTS_MODEL}:generateContent?key={GEMINI_API_KEY}"
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(url, json=payload)
            tts_ms = int((time.perf_counter() - t0) * 1000)

            if r.status_code != 200:
                log.error("[TTS]  ✗ HTTP %d  latency=%d ms  body=%s",
                          r.status_code, tts_ms, r.text[:200])
                return

            data      = r.json()
            b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            pcm24k    = base64.b64decode(b64_audio)  # 16-bit PCM @ 24 kHz
            dur_s     = len(pcm24k) / 2 / 24000

            # Resample 24 kHz → 8 kHz for Telnyx
            t1    = time.perf_counter()
            pcm8k = resample_pcm16(pcm24k, from_rate=24000, to_rate=TELNYX_RATE)
            ulaw  = pcm16_to_ulaw(pcm8k)
            resamp_ms = int((time.perf_counter() - t1) * 1000)

            payload_b64 = base64.b64encode(ulaw).decode()
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": self.stream_sid,
                "media":     {"payload": payload_b64}
            }))

            log.info("[TTS]  ← latency=%d ms  dur=%.2fs  pcm24k=%d B  resamp=%d ms  voice=%s",
                     tts_ms, dur_s, len(pcm24k), resamp_ms, GEMINI_TTS_VOICE)

        except Exception as e:
            log.error("[TTS]  _speak error: %s", e)
        finally:
            self.is_speaking = False

# ── Audio helpers ─────────────────────────────────────────────────────────────

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

