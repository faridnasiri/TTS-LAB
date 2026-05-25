"""Check wan-t2v nvfp4 shards for meta tensors."""
import torch, glob, os

base = "/opt/arthur-img-models/nvfp4/wan-t2v/transformer"
shards = sorted(glob.glob(os.path.join(base, "*.bin")))
print(f"Shards found: {len(shards)}")

total_keys = 0
total_meta = 0
for shard in shards:
    print(f"\nLoading {os.path.basename(shard)} ...")
    sd = torch.load(shard, map_location="cpu", weights_only=False)
    keys = list(sd.keys())
    meta = [k for k, v in sd.items() if hasattr(v, "is_meta") and v.is_meta]
    print(f"  keys={len(keys)}, meta={len(meta)}")
    total_keys += len(keys)
    total_meta += len(meta)

print(f"\nTotal: keys={total_keys}, meta={total_meta}")
print("DONE")
