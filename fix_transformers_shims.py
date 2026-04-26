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

class GeneralInterface:
    \"\"\"Shim for transformers>=4.54 GeneralInterface base class used by masking_utils.\"\"\"
    pass
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
