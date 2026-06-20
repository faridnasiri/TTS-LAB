"""
Patches parler_tts 0.2.3 for transformers 4.51+ compatibility.
Safe to re-run -- guarded by 'parler_tts_patched_token_tensors' marker.
"""
import re, sys, ast

# ?? 1. modeling_parler_tts.py ?????????????????????????????????????????????????
model_path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/modeling_parler_tts.py'

with open(model_path) as f:
    src = f.read()

if 'parler_tts_patched_token_tensors' not in src:
    replacements = [
        # _pad/_bos/_eos_token_tensor removed from GenerationConfig in 4.51+
        (r'generation_config\._pad_token_tensor',
         'torch.tensor(generation_config.pad_token_id)'),
        (r'generation_config\._bos_token_tensor',
         'torch.tensor(generation_config.bos_token_id)'),
        (r'generation_config\._eos_token_tensor',
         'torch.tensor(generation_config.eos_token_id)'),
        # generation_config.update() returns None in 4.51+
        (r'model_kwargs = generation_config\.update\(\*\*kwargs\)',
         'generation_config.update(**kwargs); model_kwargs = kwargs'),
        # _prepare_attention_mask_for_generation signature changed -- replace with inline mask
        (r'model_kwargs\["attention_mask"\] = self\._prepare_attention_mask_for_generation\(\s*'
         r'inputs_tensor,\s*torch\.tensor\(generation_config\.pad_token_id\),\s*'
         r'torch\.tensor\(generation_config\.eos_token_id\)\s*\)',
         'model_kwargs["attention_mask"] = torch.ones('
         'inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)'),
        # _get_initial_cache_position: make it accept both old (input_ids, model_kwargs)
        # AND new (seq_length, device, model_kwargs) signatures from GenerationMixin
        (r'    def _get_initial_cache_position\(self, input_ids, model_kwargs\):',
         '    def _get_initial_cache_position(self, a, b=None, c=None):\n'
         '        import torch as _tgc\n'
         '        if c is None:\n'
         '            input_ids, model_kwargs = a, b\n'
         '        else:\n'
         '            input_ids = _tgc.zeros((1, int(a)), dtype=_tgc.long, device=b)\n'
         '            model_kwargs = c'),
        # Add GenerationMixin so generate() works in transformers 4.50+
        (r'from transformers\.modeling_utils import PreTrainedModel\n',
         'from transformers.modeling_utils import PreTrainedModel\n'
         'try:\n'
         '    from transformers import GenerationMixin as _ParlerGenMixin\n'
         'except ImportError:\n'
         '    from transformers.generation.utils import GenerationMixin as _ParlerGenMixin\n'),
        (r'class ParlerTTSForConditionalGeneration\(PreTrainedModel\):',
         'class ParlerTTSForConditionalGeneration(PreTrainedModel, _ParlerGenMixin):'),
        # tie_weights: transformers 5.x passes recompute_mapping=True
        # which parler 0.2.3 doesn't accept.  Add **kwargs to swallow it.
        (r'def tie_weights\(self\):',
         'def tie_weights(self, **kwargs):'),
    ]
    new_src = src
    for pattern, repl in replacements:
        new_src = re.sub(pattern, repl, new_src)
    new_src = new_src.replace('import torch', 'import torch  # parler_tts_patched_token_tensors', 1)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f'SYNTAX ERROR: {e}')
        sys.exit(1)
    with open(model_path, 'w') as f:
        f.write(new_src)
    count = sum(len(re.findall(p, src)) for p, _ in replacements)
    print(f'PATCHED modeling ({count} replacements)')
else:
    print('modeling already patched')

# ?? 2. configuration_parler_tts.py ???????????????????????????????????????????
cfg_path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/configuration_parler_tts.py'

with open(cfg_path) as f:
    cfg_src = f.read()

OLD_CFG = (
    '        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs'
    ' or "decoder" not in kwargs:\n'
    '            raise ValueError("Config has to be initialized with'
    ' text_encoder, audio_encoder and decoder config")'
)
NEW_CFG = (
    '        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs'
    ' or "decoder" not in kwargs:\n'
    '            self.vocab_size = vocab_size\n'
    '            self.prompt_cross_attention = prompt_cross_attention\n'
    '            return  # transformers 4.51+: called empty in to_diff_dict()'
)

if OLD_CFG in cfg_src:
    new_cfg = cfg_src.replace(OLD_CFG, NEW_CFG, 1)
    ast.parse(new_cfg)
    with open(cfg_path, 'w') as f:
        f.write(new_cfg)
    print('PATCHED configuration')
elif 'to_diff_dict()' in cfg_src:
    print('configuration already patched')
else:
    print(f'ERROR: expected string not found in {cfg_path}')
    sys.exit(1)

# ?? 3. modeling_utils stubs ???????????????????????????????????????????????????
mu_path = '/opt/arthur-bench-env/lib/python3.11/site-packages/transformers/modeling_utils.py'
with open(mu_path) as f:
    mu = f.read()
appends = []
if 'ALL_ATTENTION_FUNCTIONS' not in mu:
    appends.append('ALL_ATTENTION_FUNCTIONS = {}')
if 'class SequenceSummary' not in mu:
    appends.append(
        'class SequenceSummary:\n'
        '    def __init__(self, config=None, **kw): pass\n'
        '    def __call__(self, *a, **kw): return None\n'
    )
if appends:
    with open(mu_path, 'a') as f:
        f.write('\n' + '\n'.join(appends) + '\n')
    print(f'PATCHED modeling_utils ({len(appends)} stubs)')
else:
    print('modeling_utils already has all stubs')
