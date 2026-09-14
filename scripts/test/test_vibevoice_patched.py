#!/usr/bin/env python3
"""VibeVoice — tokenizer patch + prefill guard (source-patched in container)."""
import torch, time, soundfile as sf
from huggingface_hub import hf_hub_download

# ---- Patch: strip add_special_tokens from PreTrainedTokenizerBase ----
import transformers.tokenization_utils_base as tub
_orig_init = tub.PreTrainedTokenizerBase.__init__
def _patched_init(self, **kwargs):
    kwargs.pop('add_special_tokens', None)
    return _orig_init(self, **kwargs)
tub.PreTrainedTokenizerBase.__init__ = _patched_init
# (prefill guard for None speech_tensors was patched directly in the vibevoice source)

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference, VibeVoiceTextTokenizer
)
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

print("Tokenizer...")
vocab_file = hf_hub_download("Qwen/Qwen2-0.5B", "vocab.json")
merges_file = hf_hub_download("Qwen/Qwen2-0.5B", "merges.txt")
tok = VibeVoiceTextTokenizer(vocab_file=vocab_file, merges_file=merges_file)
text = "Hello world. VibeVoice deployment test."
inputs = tok(text, return_tensors="pt")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to("cuda")

print("Model...")
t0 = time.time()
cfg = VibeVoiceConfig.from_pretrained("microsoft/VibeVoice-1.5B")
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-1.5B", config=cfg,
    device_map="cuda", torch_dtype=torch.bfloat16)
print(f"  {time.time()-t0:.0f}s  {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("Generating...")
t1 = time.time()
output = model.generate(**inputs, tokenizer=tok, max_new_tokens=500)
gen_time = time.time() - t1

print(f"  Output type: {type(output).__name__}")
for attr in sorted(dir(output)):
    if not attr.startswith("_"):
        v = getattr(output, attr)
        if hasattr(v, "shape"):
            print(f"    {attr}: shape={v.shape}")
        elif isinstance(v, torch.Tensor):
            print(f"    {attr}: Tensor shape={v.shape}")
        else:
            print(f"    {attr}: {type(v).__name__} = {v}")

if hasattr(output, "audio_values") and output.audio_values is not None:
    audio = output.audio_values.cpu().float().numpy().squeeze()
    dur = len(audio) / 24000
    sf.write("/tmp/vibevoice_test.wav", audio, 24000)
    print(f"  Audio: {len(audio)} samples, {dur:.1f}s, sr=24000")
    print("VIBEVOICE LOCAL INFERENCE CONFIRMED")
elif hasattr(output, "sequences"):
    print(f"  Sequences shape: {output.sequences.shape}")
    print("  (token output only — speech decoding may need post-processing)")
