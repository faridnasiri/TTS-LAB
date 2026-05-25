#!/usr/bin/env python3
"""Quick sanity check: can we wrap NVFP4WeightOnlyConfig in TorchAoConfig?"""
import sys

print("Step 1: import NVFP4WeightOnlyConfig")
try:
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    print("  OK:", NVFP4WeightOnlyConfig)
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

print("Step 2: import TorchAoConfig")
try:
    from diffusers import TorchAoConfig
    print("  OK:", TorchAoConfig)
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

print("Step 3: wrap in TorchAoConfig")
try:
    cfg = TorchAoConfig(NVFP4WeightOnlyConfig())
    print("  OK:", cfg.quant_type)
except Exception as e:
    print("  FAIL:", type(e).__name__, e)
    sys.exit(1)

print("ALL OK - NVFP4WeightOnlyConfig works with TorchAoConfig")
