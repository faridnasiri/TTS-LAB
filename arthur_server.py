"""
Arthur Henderson — Home AI Bridge Server
Replaces VAPI. Runs on home Hyper-V VM.

Stack:
  Inbound SIP/media : Telnyx TeXML + MediaStreams (WebSocket)
  STT               : faster-whisper (local, free)
  LLM               : Gemini Flash API (existing key, free tier)
  TTS               : Kokoro-82M (local, free)

Cost per call: ~$0.01–0.15 (only Telnyx per-minute + Gemini API tokens)
Compute      : $0 (home Hyper-V VM)

Setup:
  pip install fastapi uvicorn websockets faster-whisper kokoro-onnx \
              soundfile numpy google-generativeai

  Set env vars:
    GEMINI_API_KEY=<your key from Secrets.cs>
    TELNYX_API_KEY=<telnyx api key>  (optional, only for outbound calls)

  Run:
    uvicorn arthur_server:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem

  Or behind nginx/cloudflared (no SSL args needed).
"""

import asyncio, base64, json, os, struct, io, logging
from typing import Optional
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import google.generativeai as genai

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arthur")

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")   # paste key or set env var
GEMINI_MODEL   = "gemini-2.0-flash"
KOKORO_VOICE   = "am_michael"   # closest to elderly US male; alternatives: am_adam, am_eric
WHISPER_MODEL  = "base.en"      # "small.en" for better accuracy, ~500 MB RAM
SAMPLE_RATE    = 8000           # Telnyx MediaStream is 8kHz μ-law

genai.configure(api_key=GEMINI_API_KEY)

# ── Arthur system prompt (mirrors ArthurPersonaPrompts.cs Stage 1) ────────────

ARTHUR_SYSTEM_PROMPT = """
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
  - Do NOT include stage directions or parenthetical notes.
  - Speak exactly as a slow, confused elderly man — not an actor playing one.
  - Never hang up. Never be hostile. Always seem about to comply.
  - Weave in domestic details naturally: the TV, Mr. Whiskers, your reading glasses, your knee.
"""

INITIAL_GREETING = (
    "Hello? Oh my goodness, I almost didn't hear the phone. "
    "Who am I speaking with, dear?"
)

# ── Model loading (done once at startup) ──────────────────────────────────────

log.info("Loading Whisper model '%s'...", WHISPER_MODEL)
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
log.info("Whisper loaded.")

log.info("Loading Kokoro TTS...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")   # download from Kokoro repo
log.info("Kokoro loaded.")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

# ── Telnyx TeXML webhook — answers the call and starts media stream ───────────

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Telnyx hits this URL when the bridge number is dialed.
    Returns TeXML that connects the call to the WebSocket media stream.
    """
    host = request.headers.get("host", "localhost")
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/media-stream" />
  </Connect>
</Response>"""
    return HTMLResponse(content=texml, media_type="text/xml")

# ── WebSocket media stream — the main AI loop ─────────────────────────────────

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Media stream connected")

    session = CallSession()
    await session.run(ws)

# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    def __init__(self):
        self.stream_sid:  Optional[str] = None
        self.history:     list[dict]    = []
        self.audio_buf:   bytearray     = bytearray()
        self.silence_ms:  int           = 0
        self.is_speaking: bool          = False
        self.gemini      = genai.GenerativeModel(GEMINI_MODEL)

    async def run(self, ws: WebSocket):
        # Send greeting after 1.5 s
        asyncio.create_task(self._send_greeting_after_delay(ws))

        try:
            async for raw in ws.iter_text():
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    self.stream_sid = msg["start"]["streamSid"]
                    log.info("Stream started: %s", self.stream_sid)

                elif event == "media":
                    # μ-law audio from Telnyx, 8kHz, 20ms chunks
                    payload  = msg["media"]["payload"]
                    ulaw     = base64.b64decode(payload)
                    pcm16    = ulaw_to_pcm16(ulaw)
                    self.audio_buf.extend(pcm16)

                    # Accumulate ~1 s before transcribing
                    if len(self.audio_buf) >= SAMPLE_RATE * 2 and not self.is_speaking:
                        await self._process_audio_buffer(ws)

                elif event == "stop":
                    log.info("Stream stopped")
                    break

        except WebSocketDisconnect:
            log.info("WebSocket disconnected")

    async def _send_greeting_after_delay(self, ws: WebSocket):
        await asyncio.sleep(1.5)
        log.info("Sending greeting")
        self.history.append({"role": "model", "parts": [{"text": INITIAL_GREETING}]})
        await self._speak(ws, INITIAL_GREETING)

    async def _process_audio_buffer(self, ws: WebSocket):
        buf = bytes(self.audio_buf)
        self.audio_buf.clear()

        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe, buf
        )

        if not transcript or len(transcript.split()) < 2:
            return

        log.info("Scammer said: %s", transcript)
        self.history.append({"role": "user", "parts": [{"text": transcript}]})

        reply = await self._ask_gemini(transcript)
        if not reply:
            return

        log.info("Arthur replies: %s", reply)
        self.history.append({"role": "model", "parts": [{"text": reply}]})
        await self._speak(ws, reply)

    def _transcribe(self, pcm16_bytes: bytes) -> str:
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = whisper.transcribe(arr, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    async def _ask_gemini(self, user_text: str) -> Optional[str]:
        try:
            # Build chat history for context
            chat_history = []
            if not self.history:
                chat_history.append({"role": "user", "parts": [{"text": "Begin."}]})
            else:
                chat_history = self.history[:-1]  # exclude the latest user turn

            chat = self.gemini.start_chat(history=chat_history)
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chat.send_message(
                    user_text,
                    generation_config=genai.GenerationConfig(
                        system_instruction=ARTHUR_SYSTEM_PROMPT,
                        temperature=0.9,
                        max_output_tokens=120,
                    )
                )
            )
            return resp.text.strip()
        except Exception as e:
            log.error("Gemini error: %s", e)
            return None

    async def _speak(self, ws: WebSocket, text: str):
        self.is_speaking = True
        try:
            # TTS → PCM → μ-law → base64 → Telnyx
            pcm, sr = await asyncio.get_event_loop().run_in_executor(
                None, lambda: kokoro.create(text, voice=KOKORO_VOICE, speed=0.85, lang="en-us")
            )

            # Resample to 8kHz if needed
            if sr != SAMPLE_RATE:
                import resampy
                pcm = resampy.resample(pcm, sr, SAMPLE_RATE)

            pcm_int16 = (pcm * 32767).astype(np.int16)
            ulaw_bytes = pcm16_to_ulaw(pcm_int16.tobytes())
            payload    = base64.b64encode(ulaw_bytes).decode()

            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload}
            }))
        except Exception as e:
            log.error("TTS/send error: %s", e)
        finally:
            self.is_speaking = False

# ── μ-law codec (G.711) ───────────────────────────────────────────────────────

def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """Decode G.711 μ-law to 16-bit PCM."""
    out = bytearray(len(ulaw) * 2)
    for i, byte in enumerate(ulaw):
        byte  = (~byte) & 0xFF
        sign  = byte & 0x80
        exp   = (byte >> 4) & 0x07
        mant  = byte & 0x0F
        val   = ((mant << 1) + 33) << exp
        if sign:
            val = -val
        struct.pack_into("<h", out, i * 2, max(-32768, min(32767, val)))
    return bytes(out)

def pcm16_to_ulaw(pcm16: bytes) -> bytes:
    """Encode 16-bit PCM to G.711 μ-law."""
    ULAW_MAX = 0x1FFF
    BIAS     = 33
    out      = bytearray(len(pcm16) // 2)
    for i in range(len(out)):
        sample = struct.unpack_from("<h", pcm16, i * 2)[0]
        sign   = 0
        if sample < 0:
            sample = -sample
            sign   = 0x80
        sample = min(sample + BIAS, 32767)
        exp    = 7
        for e, thresh in enumerate([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF]):
            if sample <= thresh:
                exp = e
                break
        mant    = (sample >> (exp + 3)) & 0x0F
        out[i]  = (~(sign | (exp << 4) | mant)) & 0xFF
    return bytes(out)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arthur_server:app", host="0.0.0.0", port=8000, reload=False)
