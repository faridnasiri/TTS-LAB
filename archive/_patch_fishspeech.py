#!/usr/bin/env python3
"""Patch _load_fishspeech and _synth_fishspeech in tts_lab.py for fish-speech 1.5.1."""
import re, sys

path = "/opt/arthur/tts_lab.py"
src = open(path).read()

OLD = r'''def _load_fishspeech\(model_id="fishaudio/fish-speech-1\.5"\):.*?(?=\n# -- 15\. Sesame CSM)'''
NEW = '''def _load_fishspeech(model_id="fishaudio/fish-speech-1.5"):
    """Fish Speech 1.5.1 — text2semantic (LLAMA) + VQ-GAN codec.
    Source: /opt/models/fish-speech  (git clone --branch v1.5.1).
    Weights: fishaudio/fish-speech-1.5 (auto-cached in HF_HOME).
    """
    import sys as _sys, torch
    from pathlib import Path as _P

    # Ensure source root on sys.path (.pth may not be active in all envs)
    _fs_root = "/opt/models/fish-speech"
    if _fs_root not in _sys.path:
        _sys.path.insert(0, _fs_root)

    try:
        from fish_speech.models.text2semantic.inference import load_model as _load_llm
        from fish_speech.models.vqgan.inference import load_model as _load_codec
    except ImportError as e:
        raise ImportError(
            f"Fish Speech 1.5.1 code not found: {e}\\n"
            f"Expected source at {_fs_root}.\\n"
            f"Run: sudo git clone --depth=1 --branch v1.5.1 "
            f"https://github.com/fishaudio/fish-speech {_fs_root}"
        ) from e

    from huggingface_hub import snapshot_download as _dl
    model_dir = _P(_dl(model_id, ignore_patterns=["*.md", "*.txt", "*.gitignore"]))

    llama_pth = next((p for p in model_dir.glob("model*.pth")), None)
    if llama_pth is None:
        raise FileNotFoundError(f"No model.pth in {model_dir}")
    codec_pth = next((p for p in model_dir.glob("firefly-gan*.pth")), None)
    if codec_pth is None:
        raise FileNotFoundError(f"No firefly-gan*.pth in {model_dir}")

    _precision = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    llm     = _load_llm(checkpoint_path=str(llama_pth), device=DEVICE,
                        precision=_precision, compile=False)
    decoder = _load_codec(config_name="firefly_gan_vq",
                          checkpoint_path=str(codec_pth), device=DEVICE)
    return {"llm": llm, "decoder": decoder, "precision": _precision}

def _synth_fishspeech(inst, text, params):
    """Fish Speech 1.5.1 — LLAMA token generation + VQ-GAN decode."""
    import torch
    from fish_speech.models.text2semantic.inference import (
        GenerateRequest, WrappedGenerateResponse,
    )
    llm     = inst["llm"]
    decoder = inst["decoder"]

    req = GenerateRequest(
        device=DEVICE,
        text=text,
        prompt_text=None,
        prompt_tokens=None,
        max_new_tokens=int(float(params.get("max_new_tokens", 1024))),
        top_p=float(params.get("top_p", 0.7)),
        repetition_penalty=float(params.get("rep_penalty", 1.5)),
        temperature=float(params.get("temperature", 0.7)),
        compile=False,
        iterative_prompt=True,
        chunk_length=100,
    )
    codes = []
    for resp in llm(req):
        if isinstance(resp, WrappedGenerateResponse):
            if resp.status == "error":
                raise RuntimeError(f"Fish Speech LLAMA error: {resp.response}")
            if resp.response and hasattr(resp.response, "codes"):
                codes.append(resp.response.codes)

    if not codes:
        raise RuntimeError("Fish Speech: LLAMA produced no codes")

    codes_t = torch.cat(codes, dim=1)
    with torch.no_grad():
        audio = decoder.decode(indices=codes_t.unsqueeze(0).to(DEVICE))[0, 0].cpu().float().numpy()

    sr = getattr(getattr(decoder, "spec_transform", None), "sample_rate", 24000)
    return _to_wav(audio.astype(np.float32), int(sr)), int(sr)

'''

match = re.search(OLD, src, re.DOTALL)
if not match:
    print("ERROR: pattern not found")
    sys.exit(1)

patched = src[:match.start()] + NEW + src[match.end():]
open(path, 'w').write(patched)
print(f"Patched {match.end() - match.start()} chars → {len(NEW)} chars")
