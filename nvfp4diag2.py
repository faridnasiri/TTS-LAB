import sys
sys.path.insert(0, '/opt/arthur-img')

# First check if we can import the C extension directly
print('--- Direct import test ---', flush=True)
try:
    import torchao._C_mxfp8 as _mxfp8
    print('_C_mxfp8 OK:', dir(_mxfp8)[:5], flush=True)
except Exception as e:
    print('_C_mxfp8 FAIL:', e, flush=True)

try:
    import torchao._C_cutlass_90a as _cutlass
    print('_C_cutlass_90a OK', flush=True)
except Exception as e:
    print('_C_cutlass_90a FAIL:', e, flush=True)

# Check where the warning comes from
print('\n--- torchao import trace ---', flush=True)
import warnings, traceback
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import torchao
    if w:
        for warning in w:
            print('WARNING:', warning.message, flush=True)
            traceback.print_stack()

# Try the mx_formats import
print('\n--- NVFP4WeightOnlyConfig import ---', flush=True)
try:
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    print('NVFP4WeightOnlyConfig OK', flush=True)
except Exception as e:
    print('NVFP4WeightOnlyConfig FAIL:', e, flush=True)
    import traceback; traceback.print_exc()

print('DONE', flush=True)
