#!/usr/bin/env python3
import sys, os, torch
sys.path.insert(0, "/opt/models/fish-speech")
os.environ["HF_HOME"] = "/opt/models/huggingface"

MODEL_DIR = "/opt/models/huggingface/hub/models--fishaudio--fish-speech-1.5/snapshots/275a984d33c33659e39eed41ff5bcd6e67517f4c"
LLAMA_PTH = f"{MODEL_DIR}/model.pth"
CODEC_PTH = f"{MODEL_DIR}/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

print("Testing LLAMA load...")
try:
    from fish_speech.models.text2semantic.inference import load_model
    llm = load_model(checkpoint_path=LLAMA_PTH, device="cuda",
                     precision=torch.bfloat16, compile=False)
    print("LLAMA OK:", type(llm))
except Exception as e:
    import traceback; traceback.print_exc()
    print("LLAMA FAIL:", e)

print("\nTesting VQ-GAN load...")
try:
    from fish_speech.models.vqgan.inference import load_model as load_codec
    dec = load_codec(config_name="firefly_gan_vq", checkpoint_path=CODEC_PTH, device="cuda")
    print("VQ-GAN OK:", type(dec))
except Exception as e:
    import traceback; traceback.print_exc()
    print("VQ-GAN FAIL:", e)
