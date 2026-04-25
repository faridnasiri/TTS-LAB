"""
Patches torchaudio/__init__.py load() to fall back to soundfile when
torchcodec is unavailable. Handles both file paths and BytesIO objects.
"""
import sys

path = '/opt/arthur-bench-env/lib/python3.11/site-packages/torchaudio/__init__.py'

with open(path) as f:
    src = f.read()

if 'soundfile_load_fallback' in src:
    print('Already patched — nothing to do.')
    sys.exit(0)

old = '''    return load_with_torchcodec(
        uri,
        frame_offset=frame_offset,
        num_frames=num_frames,
        normalize=normalize,
        channels_first=channels_first,
        format=format,
        buffer_size=buffer_size,
        backend=backend,
    )'''

new = '''    # soundfile_load_fallback
    try:
        return load_with_torchcodec(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
            buffer_size=buffer_size,
            backend=backend,
        )
    except ImportError:
        import io as _io
        import soundfile as _sf
        import torch as _torch
        # soundfile accepts file paths and file-like objects (BytesIO) natively
        data, sr = _sf.read(uri, dtype="float32", always_2d=True)
        # data shape: [frames, channels]
        t = _torch.from_numpy(data.T if channels_first else data)
        if frame_offset:
            t = t[..., frame_offset:] if channels_first else t[frame_offset:]
        if num_frames > 0:
            t = t[..., :num_frames] if channels_first else t[:num_frames]
        return t, sr'''

if old not in src:
    print(f'ERROR: expected string not found in {path}')
    # show surrounding context for debugging
    idx = src.find('load_with_torchcodec')
    print(src[max(0,idx-100):idx+200])
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, 'w') as f:
    f.write(src)
print(f'PATCHED: {path}')
