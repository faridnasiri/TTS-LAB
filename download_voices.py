#!/usr/bin/env python3
"""Download Persian voices from HuggingFace datasets into the voice library."""
import io, json, os, sys, wave
from pathlib import Path
import numpy as np

sys.path.insert(0, "/opt/arthur")
from voice_library import add_voice, get_voice, VOICES_DIR, _ensure_dirs

def download_from_manatts(target=20, min_dur=3.0, max_dur=12.0):
    """Download voices from MahtaFetrat/Mana-TTS (single female speaker, 114h)."""
    from datasets import load_dataset
    print(f"Loading MahtaFetrat/Mana-TTS (streaming)...")
    ds = load_dataset("MahtaFetrat/Mana-TTS", split="train",
                      streaming=True, trust_remote_code=False)

    _ensure_dirs()
    downloaded = 0
    for i, row in enumerate(ds):
        audio_list = row.get("audio")
        if not isinstance(audio_list, list):
            continue
        dur = float(row.get("duration", 0))
        if dur < min_dur or dur > max_dur:
            continue

        sr = row.get("samplerate", 44100)
        sentence = str(row.get("transcript", ""))
        voice_id = f"fa_female_mana_{downloaded+1:03d}"

        if get_voice(voice_id):
            downloaded += 1
            if downloaded >= target:
                break
            continue

        try:
            audio = np.array(audio_list, dtype=np.float32)
            mx = np.abs(audio).max()
            if mx > 0:
                audio /= mx
            audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_i16.tobytes())

            add_voice(voice_id, buf.getvalue(), {
                "speaker_gender": "female",
                "speaker_age": "",
                "transcription": sentence,
                "source": "MahtaFetrat/Mana-TTS",
                "quality_score": 0.85,
                "language": "fa",
            })
            print(f"  [{downloaded+1}/{target}] {voice_id} d={dur:.1f}s '{sentence[:50]}'")
            downloaded += 1
        except Exception as e:
            print(f"  [{downloaded+1}/{target}] ✗ {e}")

        if downloaded >= target:
            break

    print(f"Done: {downloaded} voices")
    return downloaded


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    args = ap.parse_args()
    download_from_manatts(target=args.target, min_dur=args.min_dur, max_dur=args.max_dur)
