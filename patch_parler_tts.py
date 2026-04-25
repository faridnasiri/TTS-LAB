"""
Patches parler_tts on the server to be compatible with transformers 4.51+:

1. modeling_parler_tts.py: replace _pad/_bos/_eos_token_tensor (removed in
   transformers 4.51) with inline torch.tensor() calls.

2. configuration_parler_tts.py: skip the text_encoder/audio_encoder/decoder
   validation when __init__ is called with no args (transformers 4.53+
   calls ParlerTTSConfig() with no args inside to_diff_dict() to get defaults).
"""
import re, sys, ast

# ── 1. modeling_parler_tts.py ─────────────────────────────────────────────────
model_path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/modeling_parler_tts.py'

with open(model_path) as f:
    src = f.read()

if 'parler_tts_patched_token_tensors' not in src:
    replacements = [
        (r'generation_config\._pad_token_tensor', 'torch.tensor(generation_config.pad_token_id)'),
        (r'generation_config\._bos_token_tensor', 'torch.tensor(generation_config.bos_token_id)'),
        (r'generation_config\._eos_token_tensor', 'torch.tensor(generation_config.eos_token_id)'),
    ]
    new_src = src
    for pattern, repl in replacements:
        new_src = re.sub(pattern, repl, new_src)
    new_src = new_src.replace('import torch', 'import torch  # parler_tts_patched_token_tensors', 1)
    ast.parse(new_src)
    with open(model_path, 'w') as f:
        f.write(new_src)
    count = sum(len(re.findall(p, src)) for p, _ in replacements)
    print(f'PATCHED modeling ({count} token_tensor replacements)')
else:
    print('modeling already patched')

# ── 2. configuration_parler_tts.py ───────────────────────────────────────────
cfg_path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/configuration_parler_tts.py'

with open(cfg_path) as f:
    cfg_src = f.read()

old = '        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:\n            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")'
new = (
    '        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:\n'
    '            # Allow no-arg instantiation (transformers 4.51+ calls __init__() with no\n'
    '            # args inside to_diff_dict() to introspect defaults). Return early with a\n'
    '            # minimal valid state so repr/logging does not crash.\n'
    '            self.vocab_size = vocab_size\n'
    '            self.prompt_cross_attention = prompt_cross_attention\n'
    '            return'
)

if old in cfg_src:
    new_cfg = cfg_src.replace(old, new, 1)
    ast.parse(new_cfg)
    with open(cfg_path, 'w') as f:
        f.write(new_cfg)
    print(f'PATCHED configuration (no-arg init guard)')
elif 'Return early with a' in cfg_src:
    print('configuration already patched')
else:
    print(f'ERROR: expected string not found in {cfg_path}')
    sys.exit(1)
