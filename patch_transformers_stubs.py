"""
Creates stub modules for transformers internals required by qwen_tts but absent
in transformers 4.51.x:
  - transformers.masking_utils  (create_causal_mask, create_sliding_window_causal_mask)
  - transformers.modeling_layers (GradientCheckpointingLayer)
  - transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS  (if missing)
"""
import sys, ast
import torch
import torch.nn as nn

TRANSFORMERS = '/opt/arthur-bench-env/lib/python3.11/site-packages/transformers'

stubs = {
    f'{TRANSFORMERS}/masking_utils.py': '''
import torch

def create_causal_mask(input_ids=None, attention_mask=None, **kwargs):
    """Stub causal mask — returns None (transformers will handle internally)."""
    return None

def create_sliding_window_causal_mask(input_ids=None, attention_mask=None,
                                       sliding_window=None, **kwargs):
    """Stub sliding-window causal mask."""
    return None

def create_causal_4d_mask(*args, **kwargs):
    return None

def prepare_4d_causal_attention_mask(*args, **kwargs):
    return None
''',
    f'{TRANSFORMERS}/modeling_layers.py': '''
import torch.nn as nn

class GradientCheckpointingLayer(nn.Module):
    """Stub GradientCheckpointingLayer — passes through to forward() normally."""
    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
''',
}

for path, code in stubs.items():
    import os
    if os.path.exists(path):
        print(f'EXISTS (skip): {path}')
        continue
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f'SYNTAX ERROR in stub {path}: {e}')
        sys.exit(1)
    with open(path, 'w') as f:
        f.write(code.lstrip())
    print(f'CREATED: {path}')

# Also ensure ALL_ATTENTION_FUNCTIONS is importable from modeling_utils
mu_path = f'{TRANSFORMERS}/modeling_utils.py'
with open(mu_path) as f:
    mu = f.read()
appends = []
if 'ALL_ATTENTION_FUNCTIONS' not in mu:
    appends.append('ALL_ATTENTION_FUNCTIONS = {}')
if 'class SequenceSummary' not in mu:
    appends.append(
        'class SequenceSummary:\n'
        '    """Stub removed in transformers 4.51+. IndexTTS multiple_choice_head only."""\n'
        '    def __init__(self, config=None, **kwargs): pass\n'
        '    def __call__(self, *a, **kw): return None\n'
    )
if appends:
    with open(mu_path, 'a') as f:
        f.write('\n' + '\n'.join(appends) + '\n')
    print(f'Added to modeling_utils: {appends}')
else:
    print('modeling_utils.py already has all required symbols')
