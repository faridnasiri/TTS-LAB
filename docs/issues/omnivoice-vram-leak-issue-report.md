# OmniVoice VRAM Leak — GitHub Issue Report

> To be posted at: https://github.com/k2-fsa/OmniVoice/issues/new

**Title:** VRAM leak: GPU memory grows with each `generate()` call, eventually OOMs on longer text

---

## Environment

- **GPU:** NVIDIA RTX 5060 Ti 16 GB (Blackwell sm_120)
- **CUDA:** 12.8
- **Torch:** 2.12.0.dev nightly (cu128)
- **Python:** 3.11
- **omnivoice:** latest pip (`pip install omnivoice`)
- **Deployment:** Docker container, single GPU, one engine at a time in VRAM

## Summary

Each call to `OmniVoice.generate()` leaks GPU memory that `torch.cuda.empty_cache()` does not fully recover. After a handful of successive calls (same instance, no reload), available VRAM drops enough that longer input texts (~180+ chars) hit **CUDA out of memory** during the diffusion forward pass. Short texts (~60 chars) continue to work because their attention tensors are smaller and fit in the remaining fragmented VRAM.

This is a **slow leak** — not a one-shot OOM. A freshly loaded model works for all text lengths. It's only after ~3–5 prior syntheses that the leak accumulates past the threshold.

The leak is likely in the diffusion decoding loop (`_generate_iterative`): intermediate tensors held by the CUDA caching allocator across timesteps that aren't freed before the next call.

## Reproduction

```python
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")

# Short text — always works
model.generate(text="Hello world.", language="en")

# Long text — works on a fresh load
model.generate(
    text="This is a substantially longer passage designed to stress "
         "the diffusion decoder memory allocator across multiple "
         "timesteps. We need enough tokens to observe the leak.",
    language="en",
)

# After 3-5 more calls with any text length, the long text starts
# failing with CUDA OOM. Short text still works because its
# attention tensors are smaller.
for i in range(5):
    model.generate(text=f"Call {i}. Some moderate length text here.", language="en")

# This now OOMs — VRAM leaked from prior calls + long-text tensors > 16 GB
model.generate(
    text="A passage long enough to need substantial target tokens...",
    language="en",
)
```

## Symptoms in Production

In our TTS service ([TTS-Lab](https://github.com/farid-nasiri/TTS-LAB)), the Omnivoice engine runs in a Docker container behind an HTTP API. We observed:

- **Intermittent 500 errors** from the engine container
- Short texts (~60 chars): always succeed
- Longer texts (~166-190 chars): fail after a few prior syntheses
- Container logs showed the model was being reused ("already loaded — reusing"), skipping the eviction/`empty_cache()` path that would have masked the leak
- Direct testing confirmed: 7 rapid successive calls with ~190-char text all succeed *after* adding `empty_cache()` before/after each `generate()`; without the fix, call 4+ fails

## Workaround (applied downstream)

We call `torch.cuda.empty_cache()` **before and after** every `generate()` call, plus an explicit `.cpu()` on the output tensor before converting to numpy:

```python
import torch

torch.cuda.empty_cache()                    # clear before
audio = model.generate(text=text, **kwargs)
if hasattr(audio[0], "cpu"):
    audio = [a.cpu() for a in audio]        # force off GPU
result = audio[0].numpy()
torch.cuda.empty_cache()                    # clear after
```

This works reliably in our tests — 7 successive calls with ~190-char text all succeed with stable VRAM (~7 GB used / ~8.8 GB free on 16 GB card). Without the `empty_cache()` calls, call 4+ fails on long text.

## Related

May be related to the existing open issue [#180](https://github.com/k2-fsa/OmniVoice/issues/180) ("显存泄露问题" / VRAM memory leak), though that report describes a Whisper + multi-GPU scenario. This report is single-GPU, no ASR involved, pure `generate()` leak.

## Suggested Fix Direction

The diffusion loop in `_generate_iterative` builds a 2× batch (cond + uncond) per step and runs `self.forward()` for each timestep. Tensors allocated during intermediate steps may not be freed until the entire loop exits. Consider:

- Wrapping each timestep iteration in `with torch.no_grad():` (if not already)
- Explicitly `del`-ing intermediate logit/prob tensors within the loop
- Adding a `torch.cuda.empty_cache()` at the end of `generate()` itself (cheapest, most reliable fix)

---

*Report drafted 2026-06-25 by the TTS-Lab team. Fix deployed and verified in production.*
