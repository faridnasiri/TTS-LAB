import sys; sys.path.insert(0, '/opt/arthur')
from tts_lab_engines import _load_chatterbox
import soundfile as sf, numpy as np

print('=== Persian model ===')
inst = _load_chatterbox(model='persian')
print('Char map needed:', getattr(inst, '_needs_persian_char_map', False))

text = 'سلام حال شما چطور است'
audio = inst.generate(text)
arr = audio.cpu().numpy().squeeze()
sf.write('/tmp/cb_p_raw.wav', arr.astype(np.float32), 24000)
print(f'Raw: {len(arr)/24000:.1f}s peak={np.abs(arr).max():.3f} rms={np.sqrt(np.mean(arr**2)):.4f} nz={(np.abs(arr)>0.01).mean()*100:.1f}%')

print('=== Default model ===')
inst_en = _load_chatterbox(model='default')
audio2 = inst_en.generate(text)
arr2 = audio2.cpu().numpy().squeeze()
sf.write('/tmp/cb_p_default.wav', arr2.astype(np.float32), 24000)
print(f'Default: {len(arr2)/24000:.1f}s peak={np.abs(arr2).max():.3f} rms={np.sqrt(np.mean(arr2**2)):.4f} nz={(np.abs(arr2)>0.01).mean()*100:.1f}%')
print('Done')
