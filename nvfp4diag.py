import sys, os
sys.path.insert(0, '/opt/arthur-img')
os.environ.setdefault('HF_HOME', '/opt/arthur-img-models/huggingface')
import torch
from diffusers import AutoModel

path = '/opt/arthur-img-models/nvfp4/sd35/transformer'

print('=== Test 1: quant_config + device_map=cuda ===', flush=True)
try:
    from diffusers import TorchAoConfig
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    qc = TorchAoConfig(NVFP4WeightOnlyConfig())
    m = AutoModel.from_pretrained(path, quantization_config=qc,
        torch_dtype=torch.bfloat16, device_map='cuda', use_safetensors=False)
    print('OK type:', type(m).__name__, flush=True)
    del m; torch.cuda.empty_cache()
except Exception as e:
    print('FAIL1:', e, flush=True)

print('=== Test 2: low_cpu_mem_usage=False ===', flush=True)
try:
    m = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, use_safetensors=False)
    print('CPU ok, moving to cuda...', flush=True)
    m = m.cuda()
    print('OK CUDA', flush=True)
    del m; torch.cuda.empty_cache()
except Exception as e:
    print('FAIL2:', e, flush=True)

print('DONE', flush=True)
