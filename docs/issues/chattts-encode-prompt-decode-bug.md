## Title: `sample_audio_speaker` + `infer` crashes with LZMAError — encode_prompt / _decode format mismatch

### Description

`sample_audio_speaker()` produces speaker embeddings that cannot be used by `infer()`. The call chain crashes with `_lzma.LZMAError: Corrupt input data` because `encode_prompt()` and `_decode()` use incompatible data formats.

### Environment
- ChatTTS version: 0.2.5
- Python: 3.10.12
- OS: Ubuntu 22.04

### Steps to Reproduce

```python
import ChatTTS
import torchaudio
import torch

# Load model
chat = ChatTTS.Chat()
chat.load(source="huggingface", device="cuda")

# Load a reference WAV (any 5-15s speech clip)
wav, sr = torchaudio.load("reference_voice.wav")

# Create speaker embedding from reference audio
spk_emb = chat.sample_audio_speaker(wav)  # ← works fine

# Try to synthesize with it
out = chat.infer(
    "Hello world.",
    skip_refine_text=True,
    params_infer_code=chat.InferCodeParams(
        spk_emb=spk_emb,   # ← crashes
        prompt="[speed_5]",
    )
)
```

### Error

```
_lzma.LZMAError: Corrupt input data

  File "ChatTTS/core.py", line 638, in _infer_code
    self.speaker.apply(
  File "ChatTTS/model/speaker.py", line 32, in apply
    spk_emb_tensor = torch.from_numpy(self._decode(spk_emb))
  File "ChatTTS/model/speaker.py", line 148, in _decode
    lzma.decompress(
_lzma.LZMAError: Corrupt input data
```

### Root Cause

Two functions use **incompatible wire formats** for the same LZMA-compressed data:

**`encode_prompt()`** (line 133-143 in speaker.py) **prepends a 4-byte shape header:**
```python
s = b14.encode_to_string(
    np.array(shp, dtype="<u2").tobytes()          # ← 4-byte shape prefix
    + lzma.compress(arr.astype("<u2").tobytes(),   # ← LZMA payload
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
)
```

**`_decode()`** (line 146-155) **expects raw LZMA with no prefix:**
```python
return np.frombuffer(
    lzma.decompress(
        b14.decode_from_string(spk_emb),           # ← tries to decompress shape bytes as LZMA
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
    ),
    dtype=np.float16,
).copy()
```

The 4 shape bytes (`[width_lo, width_hi, height_lo, height_hi]` in little-endian uint16) are fed to the LZMA decompressor, which rejects them as corrupt input.

Meanwhile, **`_encode()`** (line 133-143) writes raw LZMA **without** a shape prefix — so `_encode()` → `_decode()` is a clean round-trip, which is why `sample_random_speaker()` works fine.

There is a matching **`decode_prompt()`** (line 147-161) that correctly strips the shape prefix, but `apply()` never calls it — `apply()` always uses `_decode()` regardless of where the embedding came from.

### The Inconsistency

| Function pair | Format | Works? |
|---|---|---|
| `_encode()` → `_decode()` | Raw LZMA | ✅ |
| `encode_prompt()` → `decode_prompt()` | Shape + LZMA | ✅ |
| `encode_prompt()` → `_decode()` | Mixed | ❌ **This is what `sample_audio_speaker` → `infer` → `apply` calls** |

### Suggested Fix

Either:

**A)** Have `apply()` (or `_infer_code()`) detect the format and dispatch to the correct decoder:
```python
# In speaker.apply():
if isinstance(spk_emb, str):
    try:
        spk_emb_tensor = torch.from_numpy(self._decode(spk_emb))
    except Exception:
        spk_emb_tensor = self.decode_prompt(spk_emb)  # try shape-prefixed format
```

**B)** Change `encode_prompt()` to use `_encode()` instead of its own LZMA compression, so both paths produce the same format:
```python
@staticmethod
def encode_prompt(prompt: torch.Tensor) -> str:
    return Speaker._encode(prompt)  # already handles 2D tensors
```

Option B is cleaner — there should be only one encoding path.

### Workaround

For anyone hitting this: call `sample_random_speaker()` instead of `sample_audio_speaker()`. You lose voice cloning, but synthesis works.

### Note

The `sample_random_speaker()` path works because `_sample_random()` → `_encode()` → `_decode()` all use the same raw-LZMA format. The bug only affects embeddings from `encode_prompt()` (used by `sample_audio_speaker()` and anywhere else `speaker.encode_prompt()` is called directly).
