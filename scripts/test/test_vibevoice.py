#!/usr/bin/env python3
"""VibeVoice inference — download Qwen2 tokenizer, load model, generate audio."""
import torch, time, soundfile as sf
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference, VibeVoiceTextTokenizer
)
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from huggingface_hub import hf_hub_download

# Download Qwen2 tokenizer files (VibeVoice uses Qwen2-based tokenizer)
print("Downloading Qwen2 tokenizer files...")
vocab_file = hf_hub_download("Qwen/Qwen2-0.5B", "vocab.json")
merges_file = hf_hub_download("Qwen/Qwen2-0.5B", "merges.txt")
print(f"  vocab: {vocab_file}")
print(f"  merges: {merges_file}")

# Create tokenizer
print("Creating tokenizer...")
tok = VibeVoiceTextTokenizer(vocab_file=vocab_file, merges_file=merges_file)
text = "Hello world. This is a VibeVoice deployment test."
inputs = tok(text, return_tensors="pt")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to("cuda")
print(f"  Tokenized: input_ids shape={inputs['input_ids'].shape}")

# Load model
t0 = time.time()
print("Loading model...")
cfg = VibeVoiceConfig.from_pretrained("microsoft/VibeVoice-1.5B")
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-1.5B", config=cfg,
    device_map="cuda", torch_dtype=torch.bfloat16)
load_time = time.time() - t0
vram = torch.cuda.memory_allocated() / 1e9
print(f"  Loaded: {load_time:.0f}s  VRAM: {vram:.1f} GB")

# Generate
print("Generating audio...")
t1 = time.time()
output = model.generate(**inputs, max_new_tokens=500)
gen_time = time.time() - t1
print(f"  Generated in: {gen_time:.0f}s")

if hasattr(output, "audio_values"):
    audio = output.audio_values.cpu().float().numpy().squeeze()
    dur = len(audio) / 24000
    sf.write("/tmp/vibevoice_test.wav", audio, 24000)
    print(f"  Audio: {len(audio)} samples, {dur:.1f}s, sr=24000")
    rtf = (load_time + gen_time) / dur if dur > 0 else 0
    print(f"  RTF: {rtf:.1f}x")
    print("STEP 4 PASSED — VibeVoice local inference confirmed!")
else:
    print(f"  Output type: {type(output).__name__}")
    if hasattr(output, "keys"):
        print(f"  Keys: {list(output.keys())}")
