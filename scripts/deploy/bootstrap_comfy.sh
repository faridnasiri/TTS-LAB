#!/bin/bash
set -uo pipefail
# arthur-comfy sidecar bootstrap — 2026-09-07 Phase D (HiDream O1-Dev engine)
# Run: ssh arthur@192.168.0.87 'bash -s' < bootstrap_comfy.sh
COMFY_DIR=/opt/arthur-img-comfy
VENV=/opt/arthur-img-comfy-env
CKPT_NAME=hidream_o1_image_dev_fp8_scaled.safetensors
CKPT_DIR=$COMFY_DIR/models/checkpoints

step() { echo "[$1/6] $2"; }

step 1 "python3.11 venv"
sudo mkdir -p $VENV $COMFY_DIR
sudo chown -R arthur:arthur $VENV $COMFY_DIR 2>/dev/null || true
if [ ! -x $VENV/bin/python ]; then
  sudo /usr/bin/python3.11 -m venv $VENV
  sudo chown -R arthur:arthur $VENV
fi
$VENV/bin/pip install -q --upgrade pip wheel || true

step 2 "torch 2.11.0+cu128 (sm_120 Blackwell)"
$VENV/bin/pip install -q torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# torchaudio MUST come from the same cu128 index — the PyPI wheel links
# libcudart.so.13 (cu130) and comfy's startup import chain (lightricks
# audio_vae) crashes with libcudart.so.13 missing (2026-09-07 incident).
# Install it BEFORE requirements.txt so the unpinned PyPI line is satisfied
# instead of upgraded.
$VENV/bin/pip install -q torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

step 3 "ComfyUI source (origin/master)"
if [ ! -d $COMFY_DIR/.git ]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI.git $COMFY_DIR
fi
sudo chown -R arthur:arthur $COMFY_DIR
cd $COMFY_DIR
git fetch -q origin
git checkout -q origin/master
git rev-parse --short HEAD > /opt/arthur-img-comfy-commit.txt
$VENV/bin/pip install -q -r requirements.txt
$VENV/bin/pip install -q 'huggingface_hub>=1.16'

step 4 "HiDream Dev fp8_scaled checkpoint (7.5 GB)"
if [ ! -f $CKPT_DIR/$CKPT_NAME ]; then
  sudo mkdir -p $CKPT_DIR
  sudo chown -R arthur:arthur $CKPT_DIR
  HF_HOME=/opt/arthur-img-models/huggingface $VENV/bin/python - <<'PY'
import os, shutil
os.environ.setdefault('HF_HOME', '/opt/arthur-img-models/huggingface')
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='Comfy-Org/HiDream-O1-Image',
                    filename='checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors')
shutil.copyfile(p, '/opt/arthur-img-comfy/models/checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors')
print('checkpoint copied from', p)
PY
else
  echo "checkpoint already present"
fi

step 5 "systemd unit arthur-comfy.service"
sudo tee /etc/systemd/system/arthur-comfy.service >/dev/null <<'UNIT'
[Unit]
Description=Arthur ComfyUI sidecar (HiDream O1-Dev engine, port 8188)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur-img-comfy
ExecStart=/opt/arthur-img-comfy-env/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=/opt/arthur-img-models/huggingface
EnvironmentFile=-/opt/arthur-img-comfy/.env

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable arthur-comfy.service >/dev/null 2>&1
sudo systemctl restart arthur-comfy.service

step 6 "wait for /system_stats"
for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "comfy up after ~$((i*2))s"
    curl -s http://127.0.0.1:8188/system_stats | head -c 200
    echo
    exit 0
  fi
  sleep 2
done
echo "comfy did not come up — check: journalctl -u arthur-comfy -n 50" >&2
exit 1
