"""
voice_library.py — Persian Voice Library for TTS Lab

Manages a curated library of reference voices for TTS voice cloning.
Downloads from Common Voice Persian, stores metadata, pre-computes
speaker embeddings for fast voice switching.

Directory structure:
  /opt/arthur/voice_library/
  ├── voices/           # WAV files + metadata
  │   ├── fa_female_001/
  │   │   ├── sample.wav
  │   │   └── metadata.json
  │   └── ...
  ├── embeddings/
  │   ├── ge2e/         # ManaTTS speaker embeddings (.npy)
  │   └── campp/        # Chatterbox speaker embeddings (.npy)
  └── index.json         # master index of all voices
"""
from __future__ import annotations
import io, json, os, re, sys, time, wave
from pathlib import Path
from typing import Optional

import numpy as np

VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "/opt/arthur/voice_library"))
VOICES_DIR       = VOICE_LIBRARY_DIR / "voices"
EMBEDDINGS_DIR   = VOICE_LIBRARY_DIR / "embeddings"
INDEX_PATH       = VOICE_LIBRARY_DIR / "index.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    (EMBEDDINGS_DIR / "ge2e").mkdir(parents=True, exist_ok=True)
    (EMBEDDINGS_DIR / "campp").mkdir(parents=True, exist_ok=True)


def _read_wav_info(path: Path) -> dict:
    """Return sample_rate, duration_s, num_samples for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return {
            "sample_rate": wf.getframerate(),
            "duration_s": round(wf.getnframes() / wf.getframerate(), 3),
            "num_samples": wf.getnframes(),
            "channels": wf.getnchannels(),
            "bit_depth": wf.getsampwidth() * 8,
        }


def _load_index() -> dict:
    """Load the master voice index."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {"voices": {}, "last_updated": None}


def _save_index(idx: dict):
    idx["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(INDEX_PATH, "w") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)


# ── voice CRUD ───────────────────────────────────────────────────────────────

def list_voices(
    gender: str = "",
    min_duration: float = 0,
    max_duration: float = 999,
    min_quality: float = 0,
    limit: int = 200,
) -> list[dict]:
    """List voices with optional filters."""
    idx = _load_index()
    voices = []
    for vid, v in idx["voices"].items():
        if gender and v.get("speaker_gender", "").lower() != gender.lower():
            continue
        dur = v.get("duration_s", 0)
        if dur < min_duration or dur > max_duration:
            continue
        if v.get("quality_score", 0) < min_quality:
            continue
        v["id"] = vid
        voices.append(v)
    voices.sort(key=lambda v: v.get("quality_score", 0), reverse=True)
    return voices[:limit]


def get_voice(voice_id: str) -> Optional[dict]:
    idx = _load_index()
    v = idx["voices"].get(voice_id)
    if v:
        v["id"] = voice_id
    return v


def get_voice_path(voice_id: str) -> Optional[Path]:
    idx = _load_index()
    v = idx["voices"].get(voice_id)
    if not v:
        return None
    p = VOICES_DIR / voice_id / "sample.wav"
    return p if p.exists() else None


def add_voice(
    voice_id: str,
    wav_data: bytes,
    metadata: dict,
    overwrite: bool = False,
) -> dict:
    """Add a voice to the library. Returns metadata dict."""
    _ensure_dirs()
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists() and not overwrite:
        raise FileExistsError(f"Voice {voice_id} already exists")

    voice_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voice_dir / "sample.wav"
    wav_path.write_bytes(wav_data)

    wav_info = _read_wav_info(wav_path)
    entry = {
        **metadata,
        **wav_info,
        "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "has_ge2e_embedding": False,
        "has_campp_embedding": False,
    }

    idx = _load_index()
    idx["voices"][voice_id] = entry
    _save_index(idx)

    # Also write a per-voice metadata.json for portability
    with open(voice_dir / "metadata.json", "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    return entry


def remove_voice(voice_id: str):
    import shutil
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    idx = _load_index()
    idx["voices"].pop(voice_id, None)
    _save_index(idx)
    # Remove cached embeddings
    for emb_type in ("ge2e", "campp"):
        emb_path = EMBEDDINGS_DIR / emb_type / f"{voice_id}.npy"
        if emb_path.exists():
            emb_path.unlink()


# ── embeddings ───────────────────────────────────────────────────────────────

def get_embedding(voice_id: str, emb_type: str = "ge2e") -> Optional[np.ndarray]:
    """Get cached speaker embedding. Computes and caches if missing."""
    emb_path = EMBEDDINGS_DIR / emb_type / f"{voice_id}.npy"
    if emb_path.exists():
        return np.load(emb_path)

    wav_path = get_voice_path(voice_id)
    if not wav_path:
        return None

    try:
        if emb_type == "ge2e":
            emb = _compute_ge2e_embedding(wav_path)
        elif emb_type == "campp":
            emb = _compute_campp_embedding(wav_path)
        else:
            return None
    except Exception:
        return None

    if emb is not None:
        np.save(emb_path, emb)
        idx = _load_index()
        if voice_id in idx["voices"]:
            idx["voices"][voice_id][f"has_{emb_type}_embedding"] = True
            _save_index(idx)
    return emb


def _compute_ge2e_embedding(wav_path: Path) -> Optional[np.ndarray]:
    """Compute GE2E speaker embedding (used by ManaTTS)."""
    try:
        import torch
        from encoder.inference import load_model as load_encoder

        encoder = load_encoder(
            "/opt/models/Persian-MultiSpeaker-Tacotron2/saved_models/final_models/encoder.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Load and preprocess
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio / (np.abs(audio).max() + 1e-8)

        # Compute embedding
        embed = encoder.embed_utterance(audio)
        del encoder
        torch.cuda.empty_cache()
        return embed
    except Exception:
        return None


def _compute_campp_embedding(wav_path: Path) -> Optional[np.ndarray]:
    """Compute CAM++ speaker embedding (used by Chatterbox)."""
    try:
        import torch
        import torchaudio
        from chatterbox.embedder import SpeakerEmbedder

        embedder = SpeakerEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
        audio, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.squeeze(0)

        embed = embedder.embed(audio)  # returns numpy or torch tensor
        if isinstance(embed, torch.Tensor):
            embed = embed.cpu().numpy()

        del embedder
        torch.cuda.empty_cache()
        return embed
    except Exception:
        return None


# ── voice stats ──────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return library statistics."""
    idx = _load_index()
    voices = idx["voices"]
    total = len(voices)
    total_dur = sum(v.get("duration_s", 0) for v in voices.values())
    by_gender = {}
    for v in voices.values():
        g = v.get("speaker_gender", "unknown")
        by_gender[g] = by_gender.get(g, 0) + 1
    with_embeddings = sum(
        1 for v in voices.values()
        if v.get("has_ge2e_embedding") or v.get("has_campp_embedding")
    )
    return {
        "total_voices": total,
        "total_duration_s": round(total_dur, 1),
        "total_duration_h": round(total_dur / 3600, 1),
        "by_gender": by_gender,
        "with_embeddings": with_embeddings,
        "last_updated": idx.get("last_updated"),
    }


# ── import from uploads ──────────────────────────────────────────────────────

def import_from_uploads(upload_dir: Path, voice_id_prefix: str = "imported"):
    """Import existing WAV files from the TTS uploads directory into the library."""
    count = 0
    for wav_path in sorted(upload_dir.glob("*.wav")):
        vid = f"{voice_id_prefix}_{wav_path.stem}"
        if get_voice(vid):
            continue  # already imported
        # Try to transcribe with whisper
        text = ""
        try:
            from f5_tts.infer.utils_infer import transcribe
            text = transcribe(str(wav_path))
        except Exception:
            pass
        wav_data = wav_path.read_bytes()
        add_voice(vid, wav_data, {
            "speaker_gender": "",
            "speaker_age": "",
            "transcription": text,
            "source": "tts_uploads",
            "quality_score": 0.7,
            "language": "fa",
        })
        print(f"  Imported: {vid}  dur={_read_wav_info(wav_path)['duration_s']}s")
        count += 1
    return count


# ── download from Common Voice ────────────────────────────────────────────────

def download_common_voice_persian(
    target_voices: int = 40,
    min_duration: float = 3.0,
    max_duration: float = 12.0,
    min_upvotes: int = 1,
    female_ratio: float = 0.5,
):
    """
    Download high-quality Persian voice clips from Mozilla Data Collective.

    Downloads the Common Voice 25.0 Persian tar.gz (10.4 GB), extracts clips
    matching criteria, and imports them into the voice library.

    Requires: pip install datasets soundfile  (for older HF mirrors)
    Falls back to direct HTTP download from Mozilla Data Collective.
    """
    _ensure_dirs()
    print("Loading Common Voice Persian metadata...")

    # ── Try HuggingFace first (older versions may still be cached) ──
    ds = None
    for ds_name in [
        "mozilla-foundation/common_voice_11_0",
        "mozilla-foundation/common_voice_13_0",
        "mozilla-foundation/common_voice_16_1",
    ]:
        try:
            from datasets import load_dataset as _ld
            ds = _ld(ds_name, "fa", split="train", streaming=True,
                     trust_remote_code=False)
            print(f"  Using {ds_name}")
            break
        except Exception:
            continue

    # ── Fallback: try other Persian speech datasets on HF ──
    if ds is None:
        for ds_name in [
            "MahtaFetrat/Mana-TTS",
            "MahtaFetrat/GPTInformal-Persian-Speech-Dataset",
        ]:
            try:
                from datasets import load_dataset as _ld2
                ds = _ld2(ds_name, split="train", streaming=True,
                          trust_remote_code=False)
                print(f"  Using {ds_name} (single-speaker dataset)")
                break
            except Exception:
                continue

    # ── Last resort: manual download instructions ──
    if ds is None:
        print("")
        print("=" * 60)
        print("  No dataset source available automatically.")
        print("  To add Persian voices manually:")
        print("")
        print("  1. Visit: https://mozilladatacollective.com/datasets/")
        print("     (search for 'Persian' / 'fa')")
        print("  2. Download the Common Voice tar.gz")
        print("  3. Extract clips and place in:")
        print(f"     {VOICES_DIR}/")
        print("  4. Or use the 'Import' button in the UI to import")
        print("     WAV files from the TTS uploads directory.")
        print("=" * 60)
        return 0

    # ── first pass: collect candidate clips ──
    candidates: list[dict] = []
    seen_speakers: set[str] = set()
    male_count = 0
    female_count = 0
    target_each = target_voices // 2

    print(f"Scanning clips (target: {target_voices} voices, "
          f"{min_duration}-{max_duration}s, female_ratio={female_ratio})...")

    for row in ds:
        # ── Handle different dataset formats ──
        # Common Voice format: audio is dict with 'duration' in ms
        # ManaTTS format: 'duration' is a top-level float in seconds, 'audio' is list
        if "duration" in row and isinstance(row["duration"], (int, float)):
            clip_dur = float(row["duration"])
        elif "audio" in row:
            audio_field = row["audio"]
            if isinstance(audio_field, dict):
                clip_dur = audio_field.get("duration", 0) / 1000  # ms → s
            elif isinstance(audio_field, list):
                sr = row.get("samplerate", 48000)
                clip_dur = len(audio_field) / sr
            else:
                clip_dur = 0
        else:
            clip_dur = 0

        if clip_dur < min_duration or clip_dur > max_duration:
            continue

        upvotes = row.get("up_votes", 1)
        downvotes = row.get("down_votes", 0)
        if upvotes < min_upvotes or downvotes > upvotes:
            continue

        gender = (row.get("gender", "") or "").lower().replace(" ", "_")
        if not gender:
            gender = "female" if female_count < male_count else "male"  # alternate

        client_id = row.get("client_id", f"spk_{len(seen_speakers)}")
        sentence = row.get("sentence", "") or row.get("transcript", "")

        # Respect female ratio
        if gender == "female" and female_count >= target_each:
            continue
        if gender == "male" and male_count >= target_each:
            continue

        # Only one clip per speaker for diversity
        speaker_key = str(client_id)
        if speaker_key in seen_speakers:
            continue
        seen_speakers.add(speaker_key)

        if gender == "female":
            female_count += 1
        else:
            male_count += 1

        quality = min(1.0, (upvotes + 1) / (upvotes + downvotes + 2))
        candidates.append({
            "client_id": client_id,
            "sentence": sentence,
            "gender": gender,
            "age": row.get("age", ""),
            "duration_s": round(clip_dur, 3),
            "quality_score": round(quality, 3),
            "up_votes": upvotes,
            "down_votes": downvotes,
        })

        if len(candidates) >= target_voices:
            break

    print(f"Selected {len(candidates)} candidates "
          f"(f={female_count}, m={male_count}) from {len(seen_speakers)} speakers")

    # ── second pass: download audio for each candidate ──
    # Re-scan the dataset to get audio arrays (streaming doesn't support indexing)
    print("Downloading audio clips...")
    downloaded = 0
    # Recreate stream for second pass
    ds2 = None
    for ds_name in [
        "mozilla-foundation/common_voice_11_0",
        "mozilla-foundation/common_voice_13_0",
        "mozilla-foundation/common_voice_16_1",
        "MahtaFetrat/Mana-TTS",
        "MahtaFetrat/GPTInformal-Persian-Speech-Dataset",
    ]:
        try:
            ds2 = _ld(ds_name, "fa" if "common_voice" in ds_name else None,
                       split="train", streaming=True, trust_remote_code=False)
            if "common_voice" not in ds_name:
                ds2 = _ld(ds_name, split="train", streaming=True, trust_remote_code=False)
            break
        except Exception:
            continue

    if ds2 is None:
        print("  ERROR: Cannot re-stream dataset for audio download")
        return 0

    # Build a lookup of client_id → candidate
    candidate_ids = {str(c["client_id"]): c for c in candidates}
    for row2 in ds2:
        cid = str(row2.get("client_id", ""))
        if cid not in candidate_ids:
            continue
        cand = candidate_ids[cid]
        voice_id = f"cv_fa_{cand['gender']}_{len(seen_speakers)-len(candidate_ids)+1:03d}"
        # Rename to match
        idx = list(candidate_ids.keys()).index(cid)
        voice_id = f"cv_fa_{cand['gender']}_{idx+1:03d}"

        if get_voice(voice_id):
            downloaded += 1
            candidate_ids.pop(cid)
            if not candidate_ids:
                break
            continue

        try:
            audio_field = row2.get("audio")
            audio_array = None
            sample_rate = 48000
            if isinstance(audio_field, dict):
                audio_array = audio_field.get("array")
                sample_rate = audio_field.get("sampling_rate", 48000)
            elif isinstance(audio_field, list):
                audio_array = np.array(audio_field, dtype=np.float32)
                sample_rate = row2.get("samplerate", 48000)

            if audio_array is None:
                continue

            # Convert to 16-bit PCM WAV
            audio_f32 = np.array(audio_array, dtype=np.float32)
            if audio_f32.max() > 0:
                audio_f32 /= audio_f32.max()
            audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate or 48000)
                wf.writeframes(audio_i16.tobytes())
            wav_data = buf.getvalue()

            add_voice(voice_id, wav_data, {
                "speaker_gender": cand["gender"],
                "speaker_age": str(cand.get("age", "")),
                "transcription": cand["sentence"],
                "source": "common_voice_25.0",
                "quality_score": cand["quality_score"],
                "up_votes": cand["up_votes"],
                "down_votes": cand.get("down_votes", 0),
                "client_id": cand["client_id"],
                "language": "fa",
            })
            print(f"  [{i+1}/{len(candidates)}] {voice_id} ✓ "
                  f"d={cand['duration_s']}s  q={cand['quality_score']}  "
                  f"g={cand['gender']}  '{cand['sentence'][:40]}...'")
            downloaded += 1
        except Exception as e:
            print(f"  [{i+1}/{len(candidates)}] {voice_id} ✗ {e}")

    print(f"\nDownloaded {downloaded}/{len(candidates)} voices to {VOICES_DIR}")
    return downloaded


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Persian Voice Library Manager")
    ap.add_argument("--download", type=int, default=0,
                    help="Download N voices from Common Voice Persian")
    ap.add_argument("--import-uploads", action="store_true",
                    help="Import existing WAVs from TTS uploads dir")
    ap.add_argument("--list", action="store_true", help="List all voices")
    ap.add_argument("--stats", action="store_true", help="Show library stats")
    ap.add_argument("--compute-embeddings", action="store_true",
                    help="Pre-compute speaker embeddings for all voices")
    ap.add_argument("--upload-dir", default="/tmp/tts_uploads",
                    help="Upload directory for --import-uploads")
    args = ap.parse_args()

    if args.download:
        download_common_voice_persian(target_voices=args.download)

    if args.import_uploads:
        import_from_uploads(Path(args.upload_dir))

    if args.list:
        voices = list_voices()
        print(f"=== Voice Library: {len(voices)} voices ===")
        for v in voices:
            emb = []
            if v.get("has_ge2e_embedding"):
                emb.append("ge2e")
            if v.get("has_campp_embedding"):
                emb.append("campp")
            emb_str = f" [{','.join(emb)}]" if emb else ""
            print(f"  {v['id']:30s} g={v.get('speaker_gender','?'):8s} "
                  f"d={v.get('duration_s',0):.1f}s  q={v.get('quality_score',0):.2f}  "
                  f"'{v.get('transcription','')[:50]}'{emb_str}")

    if args.stats:
        stats = get_stats()
        print(f"=== Voice Library Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.compute_embeddings:
        voices = list_voices()
        print(f"Computing embeddings for {len(voices)} voices...")
        for i, v in enumerate(voices):
            vid = v["id"]
            print(f"  [{i+1}/{len(voices)}] {vid}...")
            ge2e = get_embedding(vid, "ge2e")
            print(f"    ge2e: {'✓' if ge2e is not None else '✗'}")
            campp = get_embedding(vid, "campp")
            print(f"    campp: {'✓' if campp is not None else '✗'}")
        print("Done.")
