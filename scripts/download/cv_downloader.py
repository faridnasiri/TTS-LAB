#!/usr/bin/env python3
"""
cv_downloader.py — Stream Common Voice Persian 25.0 from HuggingFace,
filter, and import into the Voice Library.

Uses Reza2kn/Common-Voice-25-Persian-Cleaned (community mirror) which
contains only validated clips with audio arrays — no tar.gz needed.

Usage:
  python3 cv_downloader.py                  # import 200 diverse voices
  python3 cv_downloader.py --target 500     # import 500 voices
  python3 cv_downloader.py --female-only    # only female voices
"""
import io, json, os, sys, time, wave
from pathlib import Path
import numpy as np

VOICE_LIBRARY = Path("/opt/arthur/voice_library")
VOICES_DIR = VOICE_LIBRARY / "voices"
INDEX_PATH = VOICE_LIBRARY / "index.json"

# ── helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dirs():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    (VOICE_LIBRARY / "embeddings" / "ge2e").mkdir(parents=True, exist_ok=True)
    (VOICE_LIBRARY / "embeddings" / "campp").mkdir(parents=True, exist_ok=True)

def voice_exists(vid):
    return (VOICES_DIR / vid / "sample.wav").exists()

def voice_count():
    try:
        return len([d for d in VOICES_DIR.iterdir() if d.is_dir()])
    except Exception:
        return 0

def load_index():
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {"voices": {}, "last_updated": None}

def save_index(idx):
    idx["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    INDEX_PATH.write_text(json.dumps(idx, indent=2, ensure_ascii=False))

def add_voice(voice_id, wav_data, meta):
    vdir = VOICES_DIR / voice_id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "sample.wav").write_bytes(wav_data)
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        meta["sample_rate"] = wf.getframerate()
        meta["duration_s"] = round(wf.getnframes() / wf.getframerate(), 3)
    meta["added"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["has_ge2e_embedding"] = False
    meta["has_campp_embedding"] = False
    with open(vdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    idx = load_index()
    idx.setdefault("voices", {})[voice_id] = meta
    save_index(idx)

# ── main import ──────────────────────────────────────────────────────────────
def import_from_cv(
    target=200,
    min_dur=3.0,
    max_dur=12.0,
    female_ratio=0.5,
    female_only=False,
):
    """
    Stream Common Voice Persian 25.0 from HF community mirror.
    Filters by duration, speaker diversity, quality, and gender.
    Imports directly without needing local storage.
    """
    from datasets import load_dataset

    ensure_dirs()
    log(f"Loading Reza2kn/Common-Voice-25-Persian-Cleaned (streaming)...")

    ds = load_dataset(
        "Reza2kn/Common-Voice-25-Persian-Cleaned",
        split="validated",
        streaming=True,
    )

    male_target = 0 if female_only else int(target * (1 - female_ratio))
    female_target = target if female_only else target - male_target

    seen_speakers = {}
    imported = 0
    errors = 0
    skipped_dur = 0
    skipped_gender = 0
    male_count = 0
    female_count = 0

    # First pass: count total for progress estimate
    log(f"Streaming clips (target={target}, {min_dur}-{max_dur}s, "
        f"female_only={female_only})...")

    for i, row in enumerate(ds):
        # ── Parse audio ──
        audio = row.get("audio", {})
        if not isinstance(audio, dict):
            continue
        audio_array = audio.get("array")
        if audio_array is None:
            continue
        sr = audio.get("sampling_rate", 48000)

        # Duration
        if "duration" in audio:
            dur = audio["duration"] / 1000  # ms → s
        else:
            dur = len(audio_array) / sr

        # ── Filter duration ──
        if dur < min_dur or dur > max_dur:
            skipped_dur += 1
            continue

        # ── Filter by quality ──
        up = int(row.get("up_votes", 0))
        down = int(row.get("down_votes", 0))
        if up < 1:
            continue

        # ── Parse metadata ──
        gender = str(row.get("gender", "")).lower().strip()
        if gender in ("", "other", "do_not_wish_to_say"):
            # Assign alternating gender for diversity
            gender = "female" if female_count < male_count else "male"
        else:
            gender = gender.replace(" ", "_")

        # ── Gender quota ──
        if gender in ("female", "feminine"):
            if female_count >= female_target:
                skipped_gender += 1
                continue
        else:
            if male_count >= male_target:
                skipped_gender += 1
                continue

        # ── Speaker diversity: max 3 clips per speaker ──
        client_id = str(row.get("client_id", ""))
        cc = seen_speakers.get(client_id, 0)
        if cc >= 3:
            continue
        seen_speakers[client_id] = cc + 1

        # ── Track counts ──
        if gender in ("female", "feminine"):
            female_count += 1
        else:
            male_count += 1

        # ── Voice ID and check duplicate ──
        g = "female" if gender in ("female", "feminine") else "male"
        voice_id = f"cv_fa_{g}_{imported+1:04d}"
        if voice_exists(voice_id):
            imported += 1
            if imported >= target:
                break
            continue

        # ── Convert audio and import ──
        try:
            audio_f32 = np.array(audio_array, dtype=np.float32)
            mx = np.abs(audio_f32).max()
            if mx > 0:
                audio_f32 /= mx
            audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_i16.tobytes())
            wav_data = buf.getvalue()

            if len(wav_data) < 2000:
                errors += 1
                continue

            quality = round(up / max(up + down, 1), 3)
            add_voice(voice_id, wav_data, {
                "speaker_gender": g,
                "speaker_age": str(row.get("age", "")),
                "transcription": str(row.get("sentence", "")),
                "source": "common_voice_25.0_cleaned",
                "quality_score": quality,
                "up_votes": up,
                "down_votes": down,
                "client_id": client_id,
                "language": "fa",
            })
            imported += 1

            if imported % 20 == 0:
                log(f"  [{imported}/{target}] f={female_count} m={male_count} "
                    f"dur={dur:.1f}s q={quality:.2f} "
                    f"({skipped_dur} dur-skip, {skipped_gender} gender-skip)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                log(f"  Error: {e}")

        if imported >= target:
            break

    log(f"\n{'='*60}")
    log(f"  Speakers seen: {len(seen_speakers)}")
    log(f"  Imported: {imported}  (f={female_count}, m={male_count})")
    log(f"  Skipped: {skipped_dur} duration, {skipped_gender} gender quota")
    log(f"  Errors: {errors}")
    log(f"  Library total: {voice_count()} voices")
    log(f"{'='*60}")

    return imported


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Import Common Voice Persian voices into Voice Library")
    ap.add_argument("--target", type=int, default=200,
                    help="Number of voices to import (default: 200)")
    ap.add_argument("--female-only", action="store_true",
                    help="Only import female voices")
    ap.add_argument("--female-ratio", type=float, default=0.5,
                    help="Target female ratio (default: 0.5)")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    args = ap.parse_args()

    import_from_cv(
        target=args.target,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        female_ratio=args.female_ratio,
        female_only=args.female_only,
    )
