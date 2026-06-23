# VibeVoice — Text-Only Inference Report for Upstream

> **Target:** Microsoft VibeVoice team (vibevoice pip package, microsoft/VibeVoice-1.5B)
> **Date:** 2026-06-23
> **Environment:** Python 3.10, torch 2.12 nightly CUDA 12.8, transformers 4.51.3, vibevoice 0.0.1

---

## Summary

We attempted to run VibeVoice-1.5B for text-to-speech locally (without SGLang) using the published `vibevoice` pip package. The model loads correctly and the tokenizer works after a minor patch, but the inference pipeline cannot produce audio without SGLang — the `generate()` method's speech-processing path is never initialized for text-only input, and `speech_outputs` returns `[None]`.

This report documents the full investigation, the specific code paths involved, and two concrete suggestions for enabling standalone text-to-speech.

---

## What We Tried

We used the `VibeVoiceForConditionalGenerationInference` class from `vibevoice.modular.modeling_vibevoice_inference` with direct imports (bypassing AutoConfig, since `vibevoice` is not in `CONFIG_MAPPING`). We used the Qwen2 tokenizer vocabulary files since the VibeVoice HF repo contains no tokenizer config.

### Step 1: Model loading — WORKS

```python
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

cfg = VibeVoiceConfig.from_pretrained("microsoft/VibeVoice-1.5B")
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-1.5B", config=cfg,
    device_map="cuda", torch_dtype=torch.bfloat16)
```

Result: Model loads in 3-8 seconds, 5.4 GB VRAM on an RTX 5060 Ti. No state dict errors, no OOM.

### Step 2: Tokenizer — WORKS (with patch)

The `VibeVoiceTextTokenizer.__init__()` passes `add_special_tokens=True` to `PreTrainedTokenizerBase.__init__()`. In transformers 4.51.3, `add_special_tokens` is both a constructor parameter and a method on the base class — causing an `AttributeError` collision.

**Workaround:**
```python
import transformers.tokenization_utils_base as tub
_orig_init = tub.PreTrainedTokenizerBase.__init__
def _patched_init(self, **kwargs):
    kwargs.pop("add_special_tokens", None)
    return _orig_init(self, **kwargs)
tub.PreTrainedTokenizerBase.__init__ = _patched_init
```

Since the HF repo has no tokenizer files, we used Qwen2's vocabulary:
```python
vocab_file = hf_hub_download("Qwen/Qwen2-0.5B", "vocab.json")
merges_file = hf_hub_download("Qwen/Qwen2-0.5B", "merges.txt")
tok = VibeVoiceTextTokenizer(vocab_file=vocab_file, merges_file=merges_file)
```

Result: Tokenizer constructs and tokenizes correctly.

### Step 3: Inference — FAILS (two issues)

#### Issue A: Prefill crashes on None speech tensors

`generate()` at line 467 unconditionally calls `.to()` on `speech_tensors`, `speech_masks`, and `speech_input_mask`, all of which default to `None` for text-only calls:

```python
# modeling_vibevoice_inference.py, lines 466-470 (vibevoice 0.0.1)
if is_prefill:
    prefill_inputs = {
        "speech_tensors": speech_tensors.to(device=device),      # ← None.to() crashes
        "speech_masks": speech_masks.to(device),                  # ← None.to() crashes
        "speech_input_mask": speech_input_mask.to(device),        # ← None.to() crashes
    }
```

However, `forward()` at line 221 **already** guards against None:
```python
# modeling_vibevoice_inference.py, line 221 (vibevoice 0.0.1)
if speech_tensors is not None and speech_masks is not None:
    acoustic_features, speech_embeds = self._process_speech_inputs(...)
```

So only the prefill section is missing the guard — `forward()` handles it correctly.

**Temporary fix applied (proof of concept):**
```python
if is_prefill:
    if speech_tensors is not None:
        prefill_inputs = {
            "speech_tensors": speech_tensors.to(device=device),
            "speech_masks": speech_masks.to(device),
            "speech_input_mask": speech_input_mask.to(device),
        }
    else:
        prefill_inputs = {}
```

#### Issue B: speech_outputs is [None] even after the patch

With the prefill guard in place, `generate()` runs without crashing, but produces:

```python
VibeVoiceGenerationOutput(
    sequences=tensor([[9707, 1879, 13, 647, 23549, 51167, 23172, 1273, 13, 151643]]),
    speech_outputs=[None],                         # ← no audio generated
    reach_max_step_sample=tensor([False])
)
```

Text tokens are generated (shape `[1, 10]`), but no audio waveform is produced because the speech processing path (acoustic tokenizer, connector, semantic tokenizer) is never initialized. The model needs the speech-prefill step to set up its multimodal state even for text-to-speech.

---

## Request

Two concrete asks, in priority order:

### 1. Fix the prefill None guard (low effort)

Add a None check around line 467 of `modeling_vibevoice_inference.py`:

```python
if is_prefill:
    if speech_tensors is not None:
        prefill_inputs = {
            "speech_tensors": speech_tensors.to(device=device),
            "speech_masks": speech_masks.to(device),
            "speech_input_mask": speech_input_mask.to(device),
        }
    else:
        prefill_inputs = {}
```

This prevents the `AttributeError: 'NoneType' object has no attribute 'to'` crash for text-only calls. `forward()` already handles None correctly — this just lets it reach `forward()`.

### 2. Support text-only generation (higher effort, higher value)

Enable `generate()` to produce audio without speech-prefill initialization. If the speech path requires a minimal initialization even for text-to-speech (e.g., a zero-length or silence-based acoustic encoding pass), having that path documented or handled internally would make local/standalone inference viable without SGLang.

Alternatively, if text-only generation is not intended to be supported by the vibevoice package, documenting this explicitly would help downstream users understand the expected serving path (SGLang-Omni).

---

## Environment Details

| Component | Version |
|-----------|---------|
| vibevoice (pip) | 0.0.1 |
| transformers | 4.51.3 (also tested with 4.57.3 — same behavior) |
| torch | 2.12.0.dev20260408+cu128 / 2.14.0.dev20260622+cu130 |
| CUDA | 12.8 / 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Python | 3.10.12 |
| Driver | 580.159.03 |

## Tokenizer Files

The HF repo `microsoft/VibeVoice-1.5B` does not include tokenizer configuration files (`tokenizer_config.json`, `vocab.json`, `merges.txt`). We used Qwen2-0.5B's tokenizer files as a substitute since `VibeVoiceTextTokenizer` extends `Qwen2Tokenizer`. If the intended tokenizer source is different, documenting it would help.

---

## Threads / Issues

GitHub repo: https://github.com/microsoft/VibeVoice
HF model: https://huggingface.co/microsoft/VibeVoice-1.5B
