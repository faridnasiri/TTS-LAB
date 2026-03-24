#!/usr/bin/env bash
# Arthur Server — Ubuntu VM Setup Script
# Run this ON the VM (192.168.0.153) as root or sudo user:
#   scp tools/arthur_server/setup_vm.sh user@192.168.0.153:~/
#   ssh user@192.168.0.153
#   chmod +x setup_vm.sh && sudo ./setup_vm.sh

set -e

echo "=== 1. System packages ==="
apt update && apt install -y python3.11 python3.11-venv python3-pip ffmpeg git nginx certbot python3-certbot-nginx curl wget

echo "=== 2. Python venv ==="
cd /opt
python3.11 -m venv arthur-env
source arthur-env/bin/activate

echo "=== 3. Python dependencies ==="
pip install --upgrade pip
pip install fastapi uvicorn websockets faster-whisper soundfile numpy httpx

echo "=== 4. Copy server files ==="
mkdir -p /opt/arthur
# Files will be placed here — copy from your Windows box:
#   scp tools/arthur_server/arthur_server.py user@192.168.0.153:/opt/arthur/
echo "  -> Copy arthur_server.py to /opt/arthur/ manually (see below)"

echo "=== 5. Whisper model pre-download ==="
source /opt/arthur-env/bin/activate
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu')"
echo "  -> Whisper base.en downloaded"

echo "=== 6. Cloudflare Tunnel (no port forwarding needed) ==="
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cf.deb
dpkg -i /tmp/cf.deb
echo "  -> Run manually: cloudflared tunnel login"
echo "  -> Then:         cloudflared tunnel create arthur"
echo "  -> Then:         cloudflared tunnel route dns arthur arthur.YOURDOMAIN.com"

echo "=== 7. Systemd service ==="
cat > /etc/systemd/system/arthur.service << 'EOF'
[Unit]
Description=Arthur Henderson AI Bridge Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur
Environment="GEMINI_API_KEY=REDACTED_GOOGLE_API_KEY"
ExecStart=/opt/arthur-env/bin/uvicorn arthur_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable arthur
echo "  -> Start with: systemctl start arthur"
echo "  -> Logs with:  journalctl -u arthur -f"

echo "=== 8. Cloudflare tunnel service ==="
cat > /etc/systemd/system/cloudflared-arthur.service << 'EOF'
[Unit]
Description=Cloudflare Tunnel for Arthur
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/cloudflared tunnel run arthur
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cloudflared-arthur
echo "  -> Start with: systemctl start cloudflared-arthur (AFTER tunnel login)"

echo ""
echo "=== DONE — Manual steps remaining ==="
echo "1. scp tools/arthur_server/arthur_server.py user@192.168.0.153:/opt/arthur/"
echo "2. ssh into VM and run: cloudflared tunnel login"
echo "3. cloudflared tunnel create arthur"
echo "4. cloudflared tunnel route dns arthur arthur.YOURDOMAIN.com"
echo "5. systemctl start arthur && systemctl start cloudflared-arthur"
echo "6. Set Telnyx webhook to: https://arthur.YOURDOMAIN.com/incoming-call"
echo "7. Update Secrets.cs: AiBridgeNumber = \"+1YOUR_TELNYX_NUMBER\""
