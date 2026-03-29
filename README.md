# Arthur Server — Home Hyper-V Deployment

## What this is
A self-hosted replacement for VAPI that runs Arthur Henderson on your home server.
Zero compute cost. Twilio phone number + media streaming carries the call; Gemini handles LLM + TTS.

## Architecture
```
Scammer → Pixel 5 SIM → BaiterInCallService.InitiateHomeBridge()
                              ↓ dials Twilio DID
                         Twilio voice webhook → this server
                              ↓
                    Twilio Media Stream (WebSocket)
                              ↓
              faster-whisper (STT, local)
                              ↓
              Gemini Flash API (LLM, free tier)
                              ↓
         Gemini 2.5 Flash TTS (natural elderly voice)
                              ↓
              Audio back through Twilio → bridged call
                              ↓
                    Scammer hears Arthur
```

## VM setup (Ubuntu 22.04 on Hyper-V)

```sh
sudo apt update && sudo apt install -y python3.11 python3-pip ffmpeg git nginx certbot
pip install -r requirements.txt
```

No local TTS model download is required for the bridge server. `arthur_server.py`
uses Gemini 2.5 Flash TTS directly.

## Environment variables

```sh
export GEMINI_API_KEY="REDACTED_GOOGLE_API_KEY"   # already in Secrets.cs
```

## Expose to internet

### Option A — Cloudflare Tunnel (recommended, no port forwarding)
```sh
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cf.deb
sudo dpkg -i cf.deb
cloudflared tunnel login
cloudflared tunnel create arthur
cloudflared tunnel route dns arthur arthur.YOURDOMAIN.com
cloudflared tunnel run --url http://localhost:8000 arthur
```

### Option B — nginx reverse proxy + Let's Encrypt
```sh
# Point your domain / DDNS to your home IP, then:
sudo certbot --nginx -d arthur.YOURDOMAIN.com
# nginx proxies wss://arthur.YOURDOMAIN.com → localhost:8000
```

## Run the server

```sh
cd tools/arthur_server
uvicorn arthur_server:app --host 0.0.0.0 --port 8000
# Server is now at http://localhost:8000
# Cloudflare/nginx makes it wss://arthur.YOURDOMAIN.com
```

## Twilio setup

1. twilio.com → Phone Numbers → Buy a voice-capable number
2. On the number, set **Voice** to:
   - **Webhook**
   - **HTTP POST**
   - URL: `https://arthur.YOURDOMAIN.com/incoming-call`
3. No TwiML App is required if the number points directly to the webhook
4. Twilio will read the TwiML returned by `/incoming-call` and open:
   - `wss://arthur.YOURDOMAIN.com/media-stream`

## Android app change — one line

In `Spamblocker/Secrets.cs`:
```csharp
public const string HomeBridgeNumber = "+1YOUR_TWILIO_NUMBER";
```

That is the only change needed in the Android app.

## Cost breakdown

| Item                  | Monthly cost       |
|-----------------------|-------------------:|
| Twilio DID            | provider-dependent |
| Twilio inbound min    | provider-dependent |
| Gemini Flash          | $0 (free tier)     |
| Gemini 2.5 Flash TTS  | $0 (free tier)     |
| faster-whisper        | $0 (local)         |
| Hyper-V VM compute    | $0 (your hardware) |
| Cloudflare Tunnel     | $0 (free)    |
| **Total**             | **~$1–3/month** |

15-minute bait call cost: **~$0.08** (just Telnyx per-minute)
