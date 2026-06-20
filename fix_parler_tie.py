"""Fix parler-tts tie_weights for transformers 5.x compat."""
path = '/opt/arthur-bench-env/lib/python3.11/site-packages/parler_tts/modeling_parler_tts.py'
with open(path) as f:
    src = f.read()
old = 'def tie_weights(self):'
new = 'def tie_weights(self, **kwargs):'
if new not in src:
    src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('PATCHED tie_weights')
else:
    print('already patched')
