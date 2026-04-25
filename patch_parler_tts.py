"""
Patches parler_tts/modeling_parler_tts.py to replace _pad_token_tensor /
_bos_token_tensor / _eos_token_tensor (removed in transformers 4.51+) with
inline torch.tensor() calls using the plain int attributes.
"""
import re, sys

path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/modeling_parler_tts.py'

with open(path) as f:
    src = f.read()

if 'parler_tts_patched_token_tensors' in src:
    print('Already patched.')
    sys.exit(0)

import ast
replacements = [
    # generation_config._pad_token_tensor  ->  torch.tensor(generation_config.pad_token_id)
    (r'generation_config\._pad_token_tensor',
     'torch.tensor(generation_config.pad_token_id)'),
    (r'generation_config\._bos_token_tensor',
     'torch.tensor(generation_config.bos_token_id)'),
    (r'generation_config\._eos_token_tensor',
     'torch.tensor(generation_config.eos_token_id)'),
]

new_src = src
for pattern, repl in replacements:
    new_src = re.sub(pattern, repl, new_src)

# Mark as patched
new_src = new_src.replace(
    'import torch',
    'import torch  # parler_tts_patched_token_tensors',
    1
)

try:
    ast.parse(new_src)
except SyntaxError as e:
    print(f'SYNTAX ERROR after patch: {e}')
    sys.exit(1)

with open(path, 'w') as f:
    f.write(new_src)

count = sum(len(re.findall(p, src)) for p, _ in replacements)
print(f'PATCHED {count} occurrences in {path}')
