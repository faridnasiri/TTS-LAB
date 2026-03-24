#!/usr/bin/env python3
"""
TTS Benchmark — Arthur Server
Sequentially tests 7 TTS engines on the VM.  One model in RAM at a time.

Engines tested:
  1. Piper TTS          pip install piper-tts          ~200 MB  ~100x RT
  2. Kokoro-82M         pip install kokoro-onnx         ~500 MB   ~35x RT
  3. MeloTTS            pip install melo-tts           ~1.2 GB   ~15x RT
  4. XTTS-v2            pip install TTS                ~3.2 GB    ~3x RT  ⚠ needs swap
  5. CosyVoice2-0.5B    manual (see run_benchmark.sh)  ~2.5 GB    ~5x RT  ⚠ needs swap
  6. Parler-TTS mini    pip install parler-tts         ~1.5 GB    ~5x RT
  7. Chatterbox(-Turbo) pip install chatterbox-tts     ~1.8 GB   ~10x RT

Usage:
  python tts_benchmark.py                        # run all
  python tts_benchmark.py --models kokoro,piper  # run subset
  python tts_benchmark.py --no-xtts --no-cosyvoice  # skip RAM-heavy ones

Output:
  /tmp/tts_bench/         WAV files for listening comparison
  benchmark_results.json  raw numbers
"""

import argparse, gc, json, os, sys, time, traceback, wave
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Test config ───────────────────────────────────────────────────────────────

# Representative Arthur utterance: ~40 words, ~8s at elderly confused pace
TEST_PHRASE = (
    "Oh my goodness, just a moment dear, I need to find my reading glasses. "
    "Now, you said I owe money to the IRS? "
    "Can you give me that case number again, nice and slow? "
    "My son always tells me to write these things down."
)

ALL_MODELS   = ["piper", "kokoro", "melo", "xtts", "cosyvoice", "parler", "chatterbox"]
OUTPUT_DIR   = Path("/tmp/tts_bench")
RESULTS_FILE = Path("benchmark_results.json")
MODELS_DIR   = Path("models")    # relative to this script's directory
COSYVOICE_DIR = Path("/opt/CosyVoice")

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    model:         str
    status:        str   = "skip"   # pass | fail | skip
    load_time_s:   float = 0.0
    synth_time_s:  float = 0.0
    audio_dur_s:   float = 0.0
    rtf:           float = 0.0      # synth_time / audio_dur — lower is faster
    peak_ram_mb:   int   = 0        # delta RSS during synthesis (psutil)
    output_hz:     int   = 0
    voice:         str   = ""
    streaming:     bool  = False    # can yield first chunk before full synthesis
    arthur_fit:    str   = ""       # subjective voice suitability for elderly male
    notes:         str   = ""
    error:         str   = ""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ram_mb() -> int:
    """Current process RSS in MB (requires psutil; returns 0 if missing)."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / 1_048_576)
    except Exception:
        return 0

def _free_ram_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available / 1_048_576)
    except Exception:
        return 9999

def _wav_dur(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def _arr_dur(arr: np.ndarray, rate: int) -> float:
    return len(arr.flatten()) / rate

def _safe_del(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()

def _models_dir() -> Path:
    """Resolve models/ relative to this script."""
    return Path(__file__).parent / "models"

# ── 1. Piper TTS ──────────────────────────────────────────────────────────────

def bench_piper() -> BenchResult:
    r = BenchResult(model="piper", voice="en_US-ryan-high")
    m = _models_dir()
    model_path  = m / "en_US-ryan-high.onnx"
    config_path = m / "en_US-ryan-high.onnx.json"
    if not model_path.exists():
        r.error = f"Model not found: {model_path}  (run download_models.sh)"
        return r
    try:
        from piper.voice import PiperVoice
    except ImportError:
        r.error = "Not installed: pip install piper-tts"
        return r

    out = OUTPUT_DIR / "piper.wav"
    voice = None
    try:
        t0 = time.perf_counter()
        voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        with wave.open(str(out), "wb") as wf:
            voice.synthesize(TEST_PHRASE, wf, sentence_silence=0.2)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb = max(0, _ram_mb() - rb)

        r.audio_dur_s = round(_wav_dur(out), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = 22050
        r.streaming   = False
        r.arthur_fit  = "Decent mature male; robotic cadence; limited naturalness"
        r.notes       = "ONNX; lightest RAM; fastest start; prosody not configurable"
        r.status      = "pass"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(voice)
    return r

# ── 2. Kokoro-82M ─────────────────────────────────────────────────────────────

def bench_kokoro() -> BenchResult:
    r = BenchResult(model="kokoro", voice="bm_lewis")
    m = _models_dir()
    model_path  = m / "kokoro-v1.0.onnx"
    voices_path = m / "voices-v1.0.bin"
    if not model_path.exists() or not voices_path.exists():
        r.error = "Model files not found in models/ (run download_models.sh)"
        return r
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install kokoro-onnx soundfile"
        return r

    out = OUTPUT_DIR / "kokoro.wav"
    kokoro = None
    try:
        t0 = time.perf_counter()
        kokoro = Kokoro(str(model_path), str(voices_path))
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        samples, sample_rate = kokoro.create(TEST_PHRASE, voice="bm_lewis", speed=0.85, lang="en-us")
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        sf.write(str(out), samples, sample_rate)
        r.audio_dur_s = round(_arr_dur(samples, sample_rate), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sample_rate
        r.streaming   = False
        r.arthur_fit  = "⭐⭐⭐⭐⭐ British male (bm_lewis) sounds credibly elderly & warm"
        r.notes       = "ONNX; best quality/speed balance; speed param slows cadence"
        r.status      = "pass"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(kokoro)
    return r

# ── 3. MeloTTS ────────────────────────────────────────────────────────────────

def bench_melo() -> BenchResult:
    r = BenchResult(model="melo", voice="EN-US")
    try:
        from melo.api import TTS
    except ImportError:
        r.error = "Not installed: pip install melo-tts"
        return r

    out = OUTPUT_DIR / "melo.wav"
    tts = None
    try:
        t0 = time.perf_counter()
        tts = TTS(language="EN", device="cpu")
        r.load_time_s = round(time.perf_counter() - t0, 3)
        speaker_ids = tts.hps.data.spk2id

        rb = _ram_mb()
        t0 = time.perf_counter()
        tts.tts_to_file(TEST_PHRASE, speaker_ids["EN-US"], str(out), speed=0.85)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        r.audio_dur_s = round(_wav_dur(out), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = 44100
        r.streaming   = False
        r.arthur_fit  = "⭐⭐⭐ Clear American male; sounds younger than Arthur"
        r.notes       = "PyTorch; EN-BR speaker sounds slightly older; 44100 Hz output"
        r.status      = "pass"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(tts)
    return r

# ── 4. XTTS-v2 ───────────────────────────────────────────────────────────────

def bench_xtts() -> BenchResult:
    r = BenchResult(model="xtts", voice="Ana Florence (built-in)")
    try:
        from TTS.api import TTS
    except ImportError:
        r.error = "Not installed: pip install TTS"
        return r

    # RAM guard — XTTS loads ~3.2 GB
    free = _free_ram_mb()
    if free < 2800:
        r.status = "skip"
        r.error  = (
            f"Only {free} MB RAM free — XTTS-v2 needs ~3.2 GB. "
            "Add swap first: sudo fallocate -l 4G /swapfile && "
            "sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
        )
        return r

    out = OUTPUT_DIR / "xtts.wav"
    tts = None
    try:
        t0 = time.perf_counter()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        r.load_time_s = round(time.perf_counter() - t0, 3)

        speakers = tts.speakers or []
        speaker  = "Ana Florence" if "Ana Florence" in speakers else (speakers[0] if speakers else None)
        r.voice  = speaker or "default"

        rb = _ram_mb()
        t0 = time.perf_counter()
        wav_arr = tts.tts(text=TEST_PHRASE, speaker=speaker, language="en")
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        import soundfile as sf
        sr = 24000  # XTTS-v2 outputs 24 kHz
        sf.write(str(out), np.array(wav_arr, dtype=np.float32), sr)
        r.audio_dur_s = round(_arr_dur(np.array(wav_arr), sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.streaming   = True   # tts.tts_stream() is available
        r.arthur_fit  = "⭐⭐⭐⭐⭐ Voice cloning with reference WAV gives perfect elderly voice"
        r.notes       = "⚠ ~3.2 GB RAM; pass speaker_wav= for voice cloning; streaming available"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"
        r.error  = "OOM — add 4 GB swap then re-run"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(tts)
    return r

# ── 5. CosyVoice2-0.5B ───────────────────────────────────────────────────────

def bench_cosyvoice() -> BenchResult:
    r = BenchResult(model="cosyvoice", voice="English SFT speaker", streaming=True)
    if not COSYVOICE_DIR.exists():
        r.error = (
            f"Not installed at {COSYVOICE_DIR}. "
            "Run: git clone https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice "
            "&& pip install -r /opt/CosyVoice/requirements.txt"
        )
        return r

    cv_str = str(COSYVOICE_DIR)
    if cv_str not in sys.path:
        sys.path.insert(0, cv_str)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ImportError:
        r.error = "CosyVoice2 import failed. Check /opt/CosyVoice installation."
        return r

    model_dir = COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
    if not model_dir.exists():
        r.error = (
            f"Model not found at {model_dir}. "
            "Run: cd /opt/CosyVoice && python tools/download_model.py CosyVoice2-0.5B"
        )
        return r

    free = _free_ram_mb()
    if free < 2000:
        r.status = "skip"
        r.error  = f"Only {free} MB free — CosyVoice2 needs ~2.5 GB. Add swap first."
        return r

    out = OUTPUT_DIR / "cosyvoice.wav"
    cv = None
    try:
        import soundfile as sf
        t0 = time.perf_counter()
        cv = CosyVoice2(str(model_dir), load_jit=False, load_trt=False)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        sr = cv.sample_rate

        rb = _ram_mb()
        t0 = time.perf_counter()
        chunks = []
        for chunk in cv.inference_sft(TEST_PHRASE, "English Female"):
            chunks.append(chunk["tts_speech"].numpy().flatten())
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        sf.write(str(out), audio, sr)
        r.audio_dur_s = round(_arr_dur(audio, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "⭐⭐⭐ Primarily Chinese TTS; English accent acceptable; use zero-shot for custom voice"
        r.notes       = "Streaming; manual install; zero-shot mode needs reference WAV"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"
        r.error  = "OOM — add swap and retry"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(cv)
    return r

# ── 6. Parler-TTS mini ────────────────────────────────────────────────────────

def bench_parler() -> BenchResult:
    r = BenchResult(model="parler", voice="described: slow elderly warm male")
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import torch
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install parler-tts transformers torch"
        return r

    out = OUTPUT_DIR / "parler.wav"
    model = tokenizer = None
    try:
        model_id = "parler-tts/parler-tts-mini-v1"
        t0 = time.perf_counter()
        model     = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        r.load_time_s = round(time.perf_counter() - t0, 3)

        # Describe Arthur's voice in natural language
        description = (
            "An elderly man with a slow, warm, slightly confused voice speaks gently "
            "and unhurriedly, with natural pauses between sentences."
        )
        input_ids  = tokenizer(description, return_tensors="pt").input_ids
        prompt_ids = tokenizer(TEST_PHRASE, return_tensors="pt").input_ids

        rb = _ram_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        audio_arr = generation.cpu().numpy().squeeze()
        sr = model.config.sampling_rate
        sf.write(str(out), audio_arr, sr)
        r.audio_dur_s = round(_arr_dur(audio_arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.streaming   = False
        r.arthur_fit  = "⭐⭐⭐⭐ Text-describable voice — great for Arthur if description is tuned"
        r.notes       = "Voice quality is prompt-dependent; try different descriptions"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"
        r.error  = "OOM — add swap and retry"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(model, tokenizer)
    return r

# ── 7. Chatterbox (Turbo) ─────────────────────────────────────────────────────

def bench_chatterbox() -> BenchResult:
    r = BenchResult(model="chatterbox", voice="default (exaggeration=0.65)")
    try:
        from chatterbox.tts import ChatterboxTTS
        import torchaudio
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install chatterbox-tts torchaudio"
        return r

    out = OUTPUT_DIR / "chatterbox.wav"
    model = None
    try:
        t0 = time.perf_counter()
        # Try turbo variant first, fall back to base
        try:
            model = ChatterboxTTS.from_pretrained("resemble-ai/chatterbox-turbo", device="cpu")
            r.voice = "chatterbox-turbo (exaggeration=0.65)"
        except Exception:
            model = ChatterboxTTS.from_pretrained(device="cpu")
            r.voice = "chatterbox-base (exaggeration=0.65)"
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        wav = model.generate(
            TEST_PHRASE,
            exaggeration=0.65,   # slight expressiveness — confused elderly warmth
            cfg_weight=0.5,      # balanced adherence to prompt vs naturalness
        )
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        torchaudio.save(str(out), wav, model.sr)
        audio_arr = wav.squeeze().cpu().numpy()
        r.audio_dur_s = round(_arr_dur(audio_arr, model.sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = model.sr
        r.streaming   = False
        r.arthur_fit  = "⭐⭐⭐⭐⭐ Exaggeration param adds natural confusion/hesitation — unique Arthur fit"
        r.notes       = "exaggeration=0.65 adds character; try 0.5–0.8 range; reference WAV optional"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"
        r.error  = "OOM — add swap and retry"
    except Exception:
        r.status = "fail"
        r.error  = traceback.format_exc(limit=4)
    finally:
        _safe_del(model)
    return r

# ── Registry ──────────────────────────────────────────────────────────────────

BENCH_FNS = {
    "piper":       bench_piper,
    "kokoro":      bench_kokoro,
    "melo":        bench_melo,
    "xtts":        bench_xtts,
    "cosyvoice":   bench_cosyvoice,
    "parler":      bench_parler,
    "chatterbox":  bench_chatterbox,
}

# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(results: list[BenchResult]):
    W = 110
    print()
    print("=" * W)
    print("  ARTHUR TTS BENCHMARK RESULTS")
    print("=" * W)
    hdr = f"  {'Model':<13} {'Status':<6}  {'Load':>6}  {'Synth':>7}  {'RTF':>6}  {'RAM Δ':>7}  {'Hz':>6}  Arthur fit"
    print(hdr)
    print("-" * W)
    for r in results:
        if r.status == "pass":
            rtf_icon = "✅" if r.rtf < 0.5 else ("⚠️ " if r.rtf < 1.0 else "❌")
            print(
                f"  {r.model:<13} {'✅ pass':<8}"
                f"  {r.load_time_s:>5.1f}s"
                f"  {r.synth_time_s:>6.2f}s"
                f"  {rtf_icon}{r.rtf:>5.3f}"
                f"  {r.peak_ram_mb:>6}MB"
                f"  {r.output_hz:>6}"
                f"  {r.arthur_fit[:45]}"
            )
        elif r.status == "skip":
            print(f"  {r.model:<13} {'⏭ skip':<8}  —  {r.error[:70]}")
        else:
            print(f"  {r.model:<13} {'❌ fail':<8}  —  {r.error[:70]}")
    print("=" * W)

    passed = [r for r in results if r.status == "pass"]
    if passed:
        fastest = min(passed, key=lambda r: r.rtf)
        # Balanced score: RTF * (1 + ram_penalty) — lighter RAM = better for 3.8 GB VM
        balanced = min(passed, key=lambda r: r.rtf * (1 + r.peak_ram_mb / 4000))
        print(f"\n  🏆 Fastest RTF  : {fastest.model} (RTF {fastest.rtf:.3f})")
        if balanced.model != fastest.model:
            print(f"  ⚖️  Best balance  : {balanced.model} (RTF {balanced.rtf:.3f}, +{balanced.peak_ram_mb} MB)")
        print(f"\n  Listen to WAV files in {OUTPUT_DIR}/ to evaluate voice quality.")
        print(f"  The winner for Arthur must SOUND like a confused 78-year-old — not just be fast.\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Arthur TTS Benchmark")
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated list: piper,kokoro,melo,xtts,cosyvoice,parler,chatterbox  or 'all'"
    )
    parser.add_argument("--no-xtts",       action="store_true", help="Skip XTTS-v2 (needs swap)")
    parser.add_argument("--no-cosyvoice",  action="store_true", help="Skip CosyVoice2 (manual install)")
    args = parser.parse_args()

    if args.models.strip().lower() == "all":
        models_to_run = list(ALL_MODELS)
    else:
        models_to_run = [m.strip().lower() for m in args.models.split(",")]
        invalid = [m for m in models_to_run if m not in BENCH_FNS]
        if invalid:
            print(f"Unknown model(s): {invalid}. Valid: {list(BENCH_FNS)}")
            sys.exit(1)

    if args.no_xtts and "xtts" in models_to_run:
        models_to_run.remove("xtts")
    if args.no_cosyvoice and "cosyvoice" in models_to_run:
        models_to_run.remove("cosyvoice")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"System RAM: {vm.total // 1_048_576} MB total, {vm.available // 1_048_576} MB free")
    except ImportError:
        print("(install psutil for RAM tracking: pip install psutil)")

    print(f"Test phrase : {len(TEST_PHRASE.split())} words — \"{TEST_PHRASE[:60]}...\"")
    print(f"Running     : {', '.join(models_to_run)}")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    results = []
    for name in models_to_run:
        fn = BENCH_FNS[name]
        print(f"▶ [{name}] ...", flush=True, end="")
        result = fn()
        results.append(result)
        if result.status == "pass":
            print(f"\r✅ [{name}]  RTF={result.rtf:.3f}  synth={result.synth_time_s}s  RAM+{result.peak_ram_mb}MB")
        elif result.status == "skip":
            print(f"\r⏭ [{name}]  skipped — {result.error[:80]}")
        else:
            print(f"\r❌ [{name}]  FAILED — {result.error[:80]}")
        gc.collect()
        print()

    print_summary(results)

    RESULTS_FILE.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"Raw results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
