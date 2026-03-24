"""
Arthur Henderson — Home AI Bridge Server
Replaces VAPI. Runs on home Hyper-V VM.

Stack:
  Inbound SIP/media : Telnyx TeXML + MediaStreams (WebSocket)
  STT               : faster-whisper (local, free)
  LLM               : Gemini Flash API (existing key, free tier)
  TTS               : Gemini 2.5 Flash TTS (same key, best quality for Arthur)

Why Gemini TTS instead of local Kokoro:
  - Director's notes system makes Arthur sound genuinely confused/elderly
  - Stage-aware prompting (same ArthurPersonaPrompts logic)
  - Gacrux "Mature" voice is already proven in the Android app
  - Free tier: 15 RPM / 1M tokens/day — more than enough for scam baiting
  - Kokoro sounds like a professional reader, not an 78-year-old retiree

Cost per call: ~$0.08 (Telnyx per-minute only — all AI is free tier)
Compute      : $0 (home Hyper-V VM)

Setup:
  pip install fastapi uvicorn websockets faster-whisper \
              soundfile numpy httpx

  Set env vars:
    GEMINI_API_KEY=<your key from Secrets.cs>

  Run:
    uvicorn arthur_server:app --host 0.0.0.0 --port 8000
"""

import asyncio, base64, json, os, struct, logging
from typing import Optional
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arthur")

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
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

log.info("Loading Whisper '%s'...", WHISPER_MODEL)
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
log.info("Whisper ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host", "localhost")
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/media-stream" />
  </Connect>
</Response>"""
    return HTMLResponse(content=texml, media_type="text/xml")

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Call connected")
    await CallSession().run(ws)

# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    def __init__(self):
        self.stream_sid:  Optional[str] = None
        self.history:     list[dict]    = []
        self.audio_buf:   bytearray     = bytearray()
        self.is_speaking: bool          = False
        self.call_start:  float         = asyncio.get_event_loop().time()

    def _current_stage(self) -> int:
        elapsed = asyncio.get_event_loop().time() - self.call_start
        stage = 1
        for i, thresh in enumerate(STAGE_THRESHOLDS_SEC):
            if elapsed >= thresh:
                stage = i + 1
        return min(stage, 4)

    async def run(self, ws: WebSocket):
        asyncio.create_task(self._greet(ws))
        try:
            async for raw in ws.iter_text():
                msg   = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    self.stream_sid = msg["start"]["streamSid"]

                elif event == "media" and not self.is_speaking:
                    ulaw  = base64.b64decode(msg["media"]["payload"])
                    pcm16 = ulaw_to_pcm16(ulaw)
                    self.audio_buf.extend(pcm16)

                    # Wait for a natural pause (500ms silence) before transcribing.
                    # This ensures Arthur hears a complete sentence, not a fragment.
                    # Minimum 1s of audio before triggering to avoid single-word noise.
                    if len(self.audio_buf) >= TELNYX_RATE * 2:  # at least 1s buffered
                        chunk = np.frombuffer(bytes(self.audio_buf[-TELNYX_RATE // 2:]),
                                              dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(chunk ** 2))) / 32768.0
                        # Silence threshold: RMS < 0.01 = ~quiet for last 500ms
                        if rms < 0.01 and len(self.audio_buf) >= TELNYX_RATE * 1:
                            await self._process_buffer(ws)
                        # Hard cap: process after 8s regardless (prevents infinite wait)
                        elif len(self.audio_buf) >= TELNYX_RATE * 16:
                            await self._process_buffer(ws)

                elif event == "stop":
                    break
        except WebSocketDisconnect:
            pass
        log.info("Call ended")

    async def _greet(self, ws: WebSocket):
        await asyncio.sleep(1.5)
        self.history.append({"role": "model", "parts": [{"text": INITIAL_GREETING}]})
        await self._speak(ws, INITIAL_GREETING, stage=1)

    async def _process_buffer(self, ws: WebSocket):
        buf = bytes(self.audio_buf)
        self.audio_buf.clear()

        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe, buf
        )
        if not transcript or len(transcript.split()) < 2:
            return

        log.info("Scammer: %s", transcript)
        self.history.append({"role": "user", "parts": [{"text": transcript}]})

        stage = self._current_stage()
        reply = await self._ask_gemini(transcript, stage)
        if not reply:
            return

        log.info("Arthur [stage %d]: %s", stage, reply)
        self.history.append({"role": "model", "parts": [{"text": reply}]})
        await self._speak(ws, reply, stage)

    def _transcribe(self, pcm16: bytes) -> str:
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = whisper.transcribe(arr, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()

    async def _ask_gemini(self, user_text: str, stage: int) -> Optional[str]:
        payload = {
            "system_instruction": {"parts": [{"text": CORE_PERSONA}]},
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 120,
            }
        }
        url = f"{GEMINI_BASE}/{GEMINI_FLASH}:generateContent?key={GEMINI_API_KEY}"
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(url, json=payload)
        if r.status_code != 200:
            log.error("Gemini LLM %d: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _speak(self, ws: WebSocket, text: str, stage: int = 1):
        """Synthesise with Gemini TTS (Gacrux/Mature) + stage director's notes."""
        self.is_speaking = True
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
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(url, json=payload)

            if r.status_code != 200:
                log.error("Gemini TTS %d: %s", r.status_code, r.text[:200])
                return

            data      = r.json()
            b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            pcm24k    = base64.b64decode(b64_audio)  # 16-bit PCM @ 24 kHz

            # Resample 24 kHz → 8 kHz for Telnyx
            pcm8k  = resample_pcm16(pcm24k, from_rate=24000, to_rate=TELNYX_RATE)
            ulaw   = pcm16_to_ulaw(pcm8k)
            payload_b64 = base64.b64encode(ulaw).decode()

            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": self.stream_sid,
                "media":     {"payload": payload_b64}
            }))
            log.info("Spoke %d bytes PCM (stage %d, %s)", len(pcm24k), stage, GEMINI_TTS_VOICE)

        except Exception as e:
            log.error("_speak error: %s", e)
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

