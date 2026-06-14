#!/usr/bin/env python3
"""
import_elevenlabs_voices.py — Import ElevenLabs voice samples into TTS Lab

Converts MP3→WAV, extracts display names, adds to Voice Library + UPLOAD_DIR
so they appear in both the Voice Library browse page AND every engine's
reference WAV dropdown (Chatterbox, ManaTTS, F5-TTS, etc.).

Usage (on the arthur server):
  python3 import_elevenlabs_voices.py /path/to/voices/
  python3 import_elevenlabs_voices.py /path/to/voices/ --clear-first

The source dir can contain .mp3 or .wav files.
"""
import argparse, json, os, re, shutil, subprocess, sys, time, wave
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────
VOICE_LIBRARY = Path(os.environ.get("VOICE_LIBRARY_DIR", "/opt/arthur/voice_library"))
VOICES_DIR    = VOICE_LIBRARY / "voices"
INDEX_PATH    = VOICE_LIBRARY / "index.json"
UPLOAD_DIR    = Path(os.environ.get("UPLOAD_DIR", "/tmp/tts_uploads"))

SOURCE        = "elevenlabs"          # metadata tag
TIMESTAMP_RE  = re.compile(
    r'^ElevenLabs_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_(.+)'
)
SUFFIX_RE     = re.compile(r'[._](pvc|pre)_sp\d+_s\d+_sb\d+_v\d+$')

# ── helpers ────────────────────────────────────────────────────────────────────

def slog(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)

def ensure_dirs():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    (VOICE_LIBRARY / "embeddings" / "ge2e").mkdir(parents=True, exist_ok=True)
    (VOICE_LIBRARY / "embeddings" / "campp").mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def load_index():
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {"voices": {}, "last_updated": None}

def save_index(idx):
    idx["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    INDEX_PATH.write_text(json.dumps(idx, indent=2, ensure_ascii=False))

def clear_all():
    """Remove ALL voices from the library and index."""
    if VOICES_DIR.exists():
        shutil.rmtree(VOICES_DIR)
    # Also clear UPLOAD_DIR files that came from voice library
    if UPLOAD_DIR.exists():
        for wav in UPLOAD_DIR.glob("*.wav"):
            try:
                wav.unlink(missing_ok=True)
            except PermissionError:
                subprocess.run(["sudo", "rm", "-f", str(wav)], capture_output=True)
                slog("WARN", f"  Used sudo to remove root-owned: {wav.name}")
    ensure_dirs()
    save_index({"voices": {}, "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    slog("CLEAR", "All voices removed from library and uploads")

def extract_name(filename: str) -> tuple[str, str]:
    """
    Extract display name and ISO timestamp from ElevenLabs filename.

    Input:  ElevenLabs_2026-06-14T03_41_23_Liam - Energetic, Social Media Creator_pre_sp100_s50_sb75_v3.wav
    Output: ("Liam - Energetic, Social Media Creator", "2026-06-14T03:41:23")

    Returns (raw_name, None) if pattern doesn't match.
    """
    stem = Path(filename).stem
    m = TIMESTAMP_RE.match(stem)
    if not m:
        return (stem, None)

    ts_raw = m.group(1)  # "2026-06-14T03_41_23"
    rest   = m.group(2)  # "Liam - ..._pvc_sp100_s50_sb75_v3"

    # Convert timestamp to ISO format
    ts_iso = ts_raw.replace("_", ":", 1).replace("_", ":", 1).replace("_", " ", 1)
    ts_iso = ts_iso.replace("T", "T", 1)  # keep T

    # Strip the _pvc_... or _pre_... suffix from the name
    rest = re.sub(r'\.(mp3|wav)$', '', rest)  # in case stem still has extension
    name = SUFFIX_RE.sub('', rest) if SUFFIX_RE.search(rest) else rest

    # Clean: replace underscores with spaces (from our conversion of apostrophes)
    name = name.replace('_', ' ').strip()

    return (name, ts_raw)

def make_voice_id(name: str) -> str:
    """Generate a clean voice_id from display name."""
    vid = name.lower()
    vid = re.sub(r'[^a-z0-9\s-]', '', vid)
    vid = re.sub(r'\s+', '_', vid)
    vid = re.sub(r'-+', '_', vid)
    vid = re.sub(r'_+', '_', vid)
    vid = vid.strip('_')[:50]
    return f"el_{vid}"

def read_wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        return {
            "sample_rate": wf.getframerate(),
            "duration_s": round(wf.getnframes() / wf.getframerate(), 3),
            "num_samples": wf.getnframes(),
            "channels": wf.getnchannels(),
            "bit_depth": wf.getsampwidth() * 8,
        }

def convert_to_wav(src: Path, dst: Path) -> bool:
    """
    Convert MP3 (or any format) to 16-bit mono 22050 Hz WAV via ffmpeg.
    Returns True on success.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "22050", "-sample_fmt", "s16",
        str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except subprocess.CalledProcessError as e:
        slog("ERROR", f"ffmpeg failed for {src.name}: {e.stderr.decode()[:200]}")
        return False
    except FileNotFoundError:
        slog("FATAL", "ffmpeg not found — install it: sudo apt install ffmpeg")
        sys.exit(1)

# ── main ────────────────────────────────────────────────────────────────────────

def import_voices(source_dir: str, clear_first: bool = False):
    src_path = Path(source_dir)
    if not src_path.is_dir():
        slog("FATAL", f"Source directory not found: {src_path}")
        sys.exit(1)

    if clear_first:
        clear_all()
    else:
        ensure_dirs()

    # Collect all audio files
    audio_files = sorted(
        [f for f in src_path.iterdir() if f.suffix.lower() in ('.mp3', '.wav')]
    )
    slog("SCAN", f"Found {len(audio_files)} audio files in {src_path}")

    idx = load_index()
    imported = 0
    skipped  = 0
    staging  = Path("/tmp/el_voice_import")
    staging.mkdir(parents=True, exist_ok=True)

    for af in audio_files:
        name, ts = extract_name(af.name)
        vid = make_voice_id(name)

        slog("IMPORT", f"  {af.name[:60]}... → [{vid}] \"{name}\"")

        # ── Convert to WAV if needed ──
        if af.suffix.lower() == '.mp3':
            wav_staging = staging / f"{vid}.wav"
            if not convert_to_wav(af, wav_staging):
                skipped += 1
                continue
            wav_src = wav_staging
        else:
            wav_src = af

        try:
            wav_bytes = wav_src.read_bytes()
            wav_info = read_wav_info(wav_src)
        except Exception as e:
            slog("ERROR", f"  Cannot read WAV: {e}")
            skipped += 1
            continue

        if len(wav_bytes) < 2000:
            slog("SKIP", f"  WAV too small ({len(wav_bytes)} bytes)")
            skipped += 1
            continue

        # ── Determine gender from name (heuristic: male voices in this set) ──
        gender = "male"  # all ElevenLabs voices here are male

        # ── Add to Voice Library ──
        voice_dir = VOICES_DIR / vid
        voice_dir.mkdir(parents=True, exist_ok=True)
        (voice_dir / "sample.wav").write_bytes(wav_bytes)

        quality = 1.0  # ElevenLabs studio quality

        entry = {
            "speaker_gender": gender,
            "speaker_name": name,
            "transcription": "",
            "source": SOURCE,
            "quality_score": quality,
            "language": "en",
            **wav_info,
            "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "has_ge2e_embedding": False,
            "has_campp_embedding": False,
        }

        idx["voices"][vid] = entry
        with open(voice_dir / "metadata.json", "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        # ── Copy to UPLOAD_DIR for engine dropdown visibility ──
        upload_dest = UPLOAD_DIR / f"{vid}.wav"
        shutil.copy2(wav_src, upload_dest)

        imported += 1
        slog("OK", f"  ✓ {vid}  d={wav_info['duration_s']}s  sr={wav_info['sample_rate']}Hz  → library + dropdowns")

    # ── Save index ──
    save_index(idx)

    # ── Cleanup staging ──
    if staging.exists():
        shutil.rmtree(staging)

    # ── Summary ──
    total = len(idx["voices"])
    total_dur = sum(v.get("duration_s", 0) for v in idx["voices"].values())
    slog("DONE", f"Imported {imported} voices ({skipped} skipped). Library total: {total}, {total_dur/60:.1f} min")
    slog("DONE", f"Voice Library dir: {VOICES_DIR}")
    slog("DONE", f"Upload/dropdown dir: {UPLOAD_DIR}")
    slog("DONE", "Refresh the UI — voices appear in Browse page + all engine ref dropdowns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Import ElevenLabs voice samples into TTS Lab Voice Library")
    ap.add_argument("source_dir", help="Directory containing .mp3 or .wav files")
    ap.add_argument("--clear-first", action="store_true",
                    help="Remove ALL existing voices before importing")
    args = ap.parse_args()
    import_voices(args.source_dir, clear_first=args.clear_first)
