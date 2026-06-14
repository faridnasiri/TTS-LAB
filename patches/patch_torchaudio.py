"""
Rewrites torchaudio/_torchcodec.py load_with_torchcodec to raise ImportError
(not RuntimeError) when torchcodec shared libs fail to load, so torchaudio
__init__.py can catch it and fall back to soundfile.
"""
import sys, re

path = '/opt/arthur-bench-env/lib/python3.11/site-packages/torchaudio/_torchcodec.py'

with open(path) as f:
    src = f.read()

old_pattern = re.compile(
    r'(    # Import torchcodec here to provide clear error if not available\n)'
    r'.*?'
    r'(    audio_decoder = AudioDecoder)',
    re.DOTALL
)

replacement = (
    r'\g<1>'
    '    try:\n'
    '        from torchcodec.decoders import AudioDecoder\n'
    '    except (ImportError, RuntimeError) as _tc_err:\n'
    '        raise ImportError(\n'
    '            f"torchcodec unavailable ({_tc_err}); torchaudio will use soundfile fallback"\n'
    '        ) from _tc_err\n'
    r'\g<2>'
)

new_src, n = old_pattern.subn(replacement, src)
if n == 0:
    print('ERROR: pattern not found — showing context around AudioDecoder:')
    idx = src.find('AudioDecoder')
    print(repr(src[max(0,idx-300):idx+100]))
    sys.exit(1)

with open(path, 'w') as f:
    f.write(new_src)
print(f'PATCHED ({n} replacement(s)): {path}')
