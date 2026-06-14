import torch
sd = torch.load("/opt/arthur-img-models/nvfp4/sd35/transformer/diffusion_pytorch_model-00001-of-00002.bin",
                map_location="cpu", weights_only=False)
keys = list(sd.keys())
print(f"Total keys: {len(keys)}")
for k in keys[:5]:
    v = sd[k]
    dtype = v.dtype if hasattr(v, 'dtype') else 'no dtype'
    is_nvfp4 = 'NVFP4' in type(v).__name__ or 'MX' in type(v).__name__
    print(f"  {k}: {type(v).__name__}, dtype={dtype}, nvfp4={is_nvfp4}")
