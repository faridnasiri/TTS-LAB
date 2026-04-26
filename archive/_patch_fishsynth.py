#!/usr/bin/env python3
"""Replace _synth_fishspeech with correct 1.5.1 generate_long call."""
import re, sys

path = "/opt/arthur/tts_lab.py"
src = open(path).read()

OLD = r'def _synth_fishspeech\(inst, text, params\):.*?(?=\n# -- 15\.)'

NEW = '''def _synth_fishspeech(inst, text, params):
    """Fish Speech 1.5.1 — direct generate_long + VQ-GAN decode."""
    import torch
    from fish_speech.models.text2semantic.inference import generate_long, GenerateResponse

    model, decode_one_token = inst["llm"]   # load_model returns (model, decode_one_token)
    decoder = inst["decoder"]

    chunks = list(generate_long(
        model=model,
        device=DEVICE,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=int(float(params.get("max_new_tokens", 1024))),
        top_p=float(params.get("top_p", 0.7)),
        repetition_penalty=float(params.get("rep_penalty", 1.5)),
        temperature=float(params.get("temperature", 0.7)),
        iterative_prompt=True,
        chunk_length=100,
    ))

    codes = [c.codes for c in chunks if isinstance(c, GenerateResponse) and c.codes is not None]
    if not codes:
        raise RuntimeError("Fish Speech: generate_long produced no codes")

    codes_t = torch.cat(codes, dim=1)          # (n_codebooks, T)
    with torch.no_grad():
        # VQ-GAN decode: indices shape (batch, n_codebooks, T)
        audio_t = decoder.decode(
            indices=codes_t.unsqueeze(0).to(DEVICE),
        )
    audio = audio_t[0, 0].cpu().float().numpy()
    sr = getattr(getattr(decoder, "spec_transform", None), "sample_rate", 24000)
    return _to_wav(audio.astype(np.float32), int(sr)), int(sr)

'''

match = re.search(OLD, src, re.DOTALL)
if not match:
    print("ERROR: pattern not found"); sys.exit(1)

patched = src[:match.start()] + NEW + src[match.end():]
open(path, 'w').write(patched)
print(f"Patched _synth_fishspeech ({match.end()-match.start()} → {len(NEW)} chars)")
