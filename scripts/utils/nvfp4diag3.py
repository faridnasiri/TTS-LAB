import sys, os
sys.path.insert(0, '/opt/arthur-img')
os.environ.setdefault('HF_HOME', '/opt/arthur-img-models/huggingface')

import torch

shard1 = '/opt/arthur-img-models/nvfp4/sd35/transformer/diffusion_pytorch_model-00001-of-00002.bin'
print('Loading shard 1...', flush=True)
try:
    sd = torch.load(shard1, map_location='cpu', weights_only=False)
    print('Keys:', len(sd), flush=True)
    for k, v in list(sd.items())[:5]:
        print(f'  {k}: type={type(v).__name__}, shape={getattr(v,"shape","?")}, is_meta={getattr(v,"is_meta",False)}', flush=True)
    # Check if any tensors are meta
    meta_keys = [k for k, v in sd.items() if hasattr(v, 'is_meta') and v.is_meta]
    print(f'Meta tensors: {len(meta_keys)}/{len(sd)}', flush=True)
    if meta_keys:
        print('Sample meta keys:', meta_keys[:3], flush=True)
    # Check total actual data
    real_keys = [k for k, v in sd.items() if hasattr(v, 'numel') and callable(v.numel)]
    print(f'Tensors with numel: {len(real_keys)}', flush=True)
except Exception as e:
    import traceback
    print('FAIL:', e, flush=True)
    traceback.print_exc()

print('DONE', flush=True)
