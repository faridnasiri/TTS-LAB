# Arthur Server — Home Hyper-V Deployment

## What this is
A self-hosted replacement for VAPI that runs Arthur Henderson on your home server.
Zero compute cost. Telnyx phone number ~$1/month.

## Architecture
```
Scammer → Pixel 5 SIM → BaiterInCallService.InitiateBridge()
                              ↓ dials Telnyx DID
                         Telnyx webhook → this server
                              ↓
                    WebSocket MediaStream
                              ↓
              faster-whisper (STT, local)
                              ↓
              Gemini Flash API (LLM, free tier)
                              ↓
              Kokoro-82M TTS (local, free)
                              ↓
              Audio back through Telnyx → bridged call
                              ↓
                    Scammer hears Arthur
```

## VM setup (Ubuntu 22.04 on Hyper-V)

```sh
sudo apt update && sudo apt install -y python3.11 python3-pip ffmpeg git nginx certbot
pip install -r requirements.txt
```

## Download Kokoro model files

```sh
mkdir -p models && cd models
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin
```

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

## Telnyx setup (~$1/month)

1. telnyx.com → Numbers → Buy a number
2. Voice → SIP Connections → Create new connection
3. Webhook URL: `https://arthur.YOURDOMAIN.com/incoming-call`
4. TeXML enabled
5. Media Streaming: ON, Stream URL: `wss://arthur.YOURDOMAIN.com/media-stream`

## Android app change — one line

In `Spamblocker/Secrets.cs`:
```csharp
public const string AiBridgeNumber = "+1YOUR_TELNYX_NUMBER";
```

That is the only change needed in the Android app.

## Cost breakdown

| Item                  | Monthly cost  |
|-----------------------|--------------|
| Telnyx DID            | ~$1.00       |
| Telnyx inbound min    | ~$0.005/min  |
| Gemini Flash          | $0 (free tier) |
| Kokoro TTS            | $0 (local)   |
| faster-whisper        | $0 (local)   |
| Hyper-V VM compute    | $0 (your hardware) |
| Cloudflare Tunnel     | $0 (free)    |
| **Total**             | **~$1–3/month** |

15-minute bait call cost: **~$0.08** (just Telnyx per-minute)
