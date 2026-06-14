"""Fix mangled auto_docstring shim in transformers utils __init__ and generic."""
import sys

for path in [
    '/opt/arthur-bench-env/lib/python3.11/site-packages/transformers/utils/__init__.py',
    '/opt/arthur-bench-env/lib/python3.11/site-packages/transformers/utils/generic.py',
]:
    with open(path) as f:
        src = f.read()

    # Strip everything from first occurrence of 'def auto_docstring' or 'def check_model_inputs'
    cut = min(
        (src.find(f'\ndef {fn}') for fn in ['auto_docstring', 'check_model_inputs']
         if src.find(f'\ndef {fn}') != -1),
        default=-1
    )
    base = src[:cut].rstrip() if cut != -1 else src.rstrip()

    shim = """

def auto_docstring(*args, **kwargs):
    \"\"\"Shim for transformers>=4.54 auto_docstring decorator.\"\"\"
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda fn: fn

def check_model_inputs(*args, **kwargs):
    \"\"\"Shim for transformers>=4.54 check_model_inputs decorator.\"\"\"
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda fn: fn

from collections.abc import MutableMapping

class GeneralInterface(MutableMapping):
    \"\"\"Full MutableMapping implementation restoring what truncation removed.

    AttentionInterface and AttentionMaskInterface inherit from this class
    and rely on __getitem__, __contains__, register(), valid_keys(), etc.
    The original generic.py truncation replaced this with an empty stub,
    breaking every transformer model that uses attention/masking interfaces.
    \"\"\"
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        self._local_mapping[key] = value

    def __delitem__(self, key):
        if key in self._local_mapping:
            del self._local_mapping[key]

    def __iter__(self):
        merged = {**self._global_mapping, **self._local_mapping}
        return iter(merged)

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value):
        cls._global_mapping.update({key: value})

    def valid_keys(self):
        return list(self.keys())
"""
    with open(path, 'w') as f:
        f.write(base + shim)

    # Verify syntax
    import ast
    try:
        ast.parse(base + shim)
        print(f'OK: {path}')
    except SyntaxError as e:
        print(f'SYNTAX ERROR in {path}: {e}')
        sys.exit(1)
