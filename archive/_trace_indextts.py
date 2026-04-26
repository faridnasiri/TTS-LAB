#!/usr/bin/env python3
# Trace which module causes numpy.core.multiarray to fail when indextts loads
import sys, importlib, traceback

# Monkey-patch numpy.core.multiarray to raise if called from wrong numpy
import numpy
print(f"numpy version: {numpy.__version__}")

# Try importing indextts piece by piece
steps = [
    "indextts",
    "indextts.gpt",
    "indextts.gpt.transformers_gpt2",
    "indextts.gpt.transformers_generation_utils",
    "indextts.gpt.model_v2",
    "indextts.infer",
]
for mod in steps:
    try:
        importlib.import_module(mod)
        print(f"OK  {mod}")
    except Exception as e:
        print(f"FAIL {mod}: {e}")
        traceback.print_exc()
        break
