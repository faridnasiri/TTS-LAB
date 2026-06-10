#!/opt/arthur-img-env/bin/python3
"""Pre-download Qwen3-VL to HF cache so HF_HUB_OFFLINE works."""
import os
os.environ['HF_HOME'] = '/opt/arthur-img-models/huggingface'

# Read HF_TOKEN from .env
env_path = '/opt/arthur-img/.env'
token = ''
if os.path.exists(env_path):
    for line in open(env_path):
        if line.startswith('HF_TOKEN='):
            token = line.split('=', 1)[1].strip()
            break

print(f"Downloading Qwen3-VL 8B (gated, using HF token)...")
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    'Qwen/Qwen3-VL-8B',
    trust_remote_code=True,
    torch_dtype='auto',
    token=token,
)
tok = AutoTokenizer.from_pretrained(
    'Qwen/Qwen3-VL-8B',
    trust_remote_code=True,
    token=token,
)

cache_path = os.path.join(os.environ['HF_HOME'], 'hub', 'models--Qwen--Qwen3-VL-8B')
print(f"Qwen3-VL cached: {os.path.exists(cache_path)}")
print("Done — offline mode will work now.")
