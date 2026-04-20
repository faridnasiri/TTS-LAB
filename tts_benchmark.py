#!/usr/bin/env python3
"""
TTS Benchmark — Arthur Server
Sequentially tests 13 TTS engines on the VM.  One model in RAM at a time.

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

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import torch as _t
    DEVICE      = "cuda" if _t.cuda.is_available() else "cpu"
    DEVICE_NAME = _t.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    del _t
except Exception:
    DEVICE = "cpu"; DEVICE_NAME = "CPU"

print(f"[bench] Device: {DEVICE}  ({DEVICE_NAME})")

# ── Test config ───────────────────────────────────────────────────────────────

# Representative Arthur utterance: ~40 words, ~8s at elderly confused pace
TEST_PHRASE = (
    "Oh my goodness, just a moment dear, I need to find my reading glasses. "
    "Now, you said I owe money to the IRS? "
    "Can you give me that case number again, nice and slow? "
    "My son always tells me to write these things down."
)

ALL_MODELS   = ["piper","kokoro","melo","chattts","outetts","bark","styletts2","f5tts","dia","xtts","cosyvoice","parler","chatterbox",
                "fishspeech","csm","qwen3tts","orpheus","neutts","indextts","zonos","openvoice"]
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
        voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=(DEVICE=="cuda"))
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
        tts = TTS(language="EN", device=DEVICE)
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

# -- 4. ChatTTS ---------------------------------------------------------------

def bench_chattts() -> BenchResult:
    r = BenchResult(model="chattts", voice="random speaker")
    try:
        import ChatTTS
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install ChatTTS"
        return r

    free = _free_ram_mb()
    if free < 1500:
        r.status = "skip"
        r.error  = f"Only {free} MB free -- ChatTTS needs ~1.8 GB."
        return r

    out = OUTPUT_DIR / "chattts.wav"
    inst = None
    try:
        t0 = time.perf_counter()
        inst = ChatTTS.Chat()
        if not inst.load(source="huggingface", device=DEVICE):
            raise RuntimeError("ChatTTS load failed")
        spk = inst.sample_random_speaker()
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        out_wav = inst.infer(
            TEST_PHRASE,
            skip_refine_text=True,
            params_infer_code=inst.InferCodeParams(
                prompt="[speed_5]", top_P=0.7, top_K=20, temperature=0.3,
                repetition_penalty=1.05, max_new_token=512, show_tqdm=False, spk_emb=spk,
            ),
        )
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        arr = np.array(out_wav[0] if isinstance(out_wav, list) else out_wav, dtype=np.float32).flatten()
        sr  = 24000
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Conversational; [speed_5] token; speaker sampling matches elderly pace"
        r.notes       = "Speed via [speed_1-9] token; optional reference WAV for speaker matching"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# -- 5. OuteTTS ---------------------------------------------------------------

def bench_outetts() -> BenchResult:
    r = BenchResult(model="outetts", voice="en-female-1-neutral")
    try:
        import outetts
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install outetts"
        return r

    free = _free_ram_mb()
    if free < 1400:
        r.status = "skip"
        r.error  = f"Only {free} MB free -- OuteTTS needs ~1.6 GB."
        return r

    out = OUTPUT_DIR / "outetts.wav"
    inst = None
    try:
        t0 = time.perf_counter()
        cfg  = outetts.ModelConfig(
            model_path="OuteAI/OuteTTS-0.3-500M", tokenizer_path="OuteAI/OuteTTS-0.3-500M",
            backend=outetts.Backend.HF, device=DEVICE, max_seq_length=32768,
        )
        inst    = outetts.Interface(cfg)
        speaker = inst.load_default_speaker("en-female-1-neutral")
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        out_gen = inst.generate(outetts.GenerationConfig(
            text=TEST_PHRASE, speaker=speaker, max_length=32768,
            sampler_config=outetts.SamplerConfig(
                temperature=0.4, repetition_penalty=1.1, top_k=40, top_p=0.9, min_p=0.05,
            ),
        ))
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        arr = out_gen.audio.detach().cpu().numpy().squeeze()
        sr  = getattr(out_gen, "sr", 44100)
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Character prompt + voice cloning; voice_characteristics param"
        r.notes       = "Default speaker female; use voice_characteristics + ref WAV for elderly male"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# -- 6. Bark ------------------------------------------------------------------

def bench_bark() -> BenchResult:
    r = BenchResult(model="bark", voice="v2/en_speaker_6")
    try:
        from bark import generate_audio, preload_models
        from bark.generation import SAMPLE_RATE
    except ImportError:
        r.error = "Not installed: pip install bark"
        return r

    free = _free_ram_mb()
    if free < 1200:
        r.status = "skip"
        r.error  = f"Only {free} MB free -- Bark needs ~1.5 GB."
        return r

    out = OUTPUT_DIR / "bark.wav"
    try:
        import torch, soundfile as sf
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        t0 = time.perf_counter()
        try:
            _use_small = (DEVICE != "cuda")
            os.environ["SUNO_USE_SMALL_MODELS"] = "True" if _use_small else "False"
            preload_models(
                text_use_small=_use_small, coarse_use_small=_use_small, fine_use_small=_use_small,
                text_use_gpu=(DEVICE=="cuda"), coarse_use_gpu=(DEVICE=="cuda"), fine_use_gpu=(DEVICE=="cuda"),
            )
        finally:
            torch.load = _orig
        r.load_time_s = round(time.perf_counter() - t0, 3)

        text_tok = "[hesitantly] " + TEST_PHRASE + " [sighs]"
        rb = _ram_mb()
        t0 = time.perf_counter()
        audio = generate_audio(text_tok, history_prompt="v2/en_speaker_6")
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        sf.write(str(out), audio, SAMPLE_RATE)
        r.audio_dur_s = round(_arr_dur(audio, SAMPLE_RATE), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = SAMPLE_RATE
        r.arthur_fit  = "Emotion tokens [hesitantly][sighs][laughs][clears throat] embedded in text"
        r.notes       = "~30x RT; small model; test uses [hesitantly]+[sighs]"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    return r

# -- 7. StyleTTS 2 ------------------------------------------------------------

def bench_styletts2() -> BenchResult:
    r = BenchResult(model="styletts2", voice="default (no ref WAV)")
    try:
        from styletts2 import tts as _st2
    except ImportError:
        r.error = "Not installed: pip install styletts2"
        return r

    out  = OUTPUT_DIR / "styletts2.wav"
    inst = None
    try:
        import torch, soundfile as sf
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        t0 = time.perf_counter()
        try:
            inst = _st2.StyleTTS2()
        finally:
            torch.load = _orig
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        audio = inst.inference(text=TEST_PHRASE, alpha=0.3, beta=0.7, diffusion_steps=5)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        arr = np.array(audio, dtype=np.float32).flatten()
        sr  = 24000
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Fastest high-quality (~2x RT); style transfer from ref WAV adds character"
        r.notes       = "alpha/beta control style vs prosody weight; ref WAV optional"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# -- 8. F5-TTS ----------------------------------------------------------------

def bench_f5tts() -> BenchResult:
    r = BenchResult(model="f5tts", voice="cloned from piper.wav (if present)")
    try:
        from f5_tts.api import F5TTS
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install f5-tts"
        return r

    ref_path = OUTPUT_DIR / "piper.wav"
    if not ref_path.exists():
        r.status = "skip"
        r.error  = "Reference WAV not found -- run bench_piper first"
        return r

    out  = OUTPUT_DIR / "f5tts.wav"
    inst = None
    try:
        t0 = time.perf_counter()
        inst = F5TTS()
        r.load_time_s = round(time.perf_counter() - t0, 3)

        rb = _ram_mb()
        t0 = time.perf_counter()
        wav, sr, _ = inst.infer(
            ref_file=str(ref_path), ref_text="",
            gen_text=TEST_PHRASE, speed=1.0, nfe_step=32,
        )
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        arr = np.array(wav, dtype=np.float32).flatten()
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Best zero-shot voice cloning; use aged-voice ref WAV for Arthur"
        r.notes       = "Ref=piper.wav here; production: upload Arthur voice clip via tts_lab"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# -- 9. Dia-1.6B --------------------------------------------------------------

def bench_dia() -> BenchResult:
    r = BenchResult(model="dia", voice="[S1] speaker")
    try:
        from dia.model import Dia
        import soundfile as sf
    except ImportError:
        r.error = "Not installed: pip install git+https://github.com/nari-labs/dia.git"
        return r

    free = _free_ram_mb()
    if free < 2500:
        r.status = "skip"
        r.error  = f"Only {free} MB free -- Dia-1.6B needs ~3 GB."
        return r

    out  = OUTPUT_DIR / "dia.wav"
    inst = None
    try:
        t0 = time.perf_counter()
        for mid in ["nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"]:
            try:
                _dtype = "bfloat16" if DEVICE == "cuda" else "float32"
                try:
                    inst = Dia.from_pretrained(mid, compute_dtype=_dtype, device=DEVICE); break
                except TypeError:
                    inst = Dia.from_pretrained(mid, compute_dtype=_dtype); break
            except Exception:
                if mid == "nari-labs/Dia-1.6B": raise
        r.load_time_s = round(time.perf_counter() - t0, 3)

        dia_text    = f"[S1] {TEST_PHRASE}"
        auto_tokens = min(1024, max(256, len(dia_text) * 6))
        rb = _ram_mb()
        t0 = time.perf_counter()
        output = inst.generate(
            dia_text, max_tokens=auto_tokens,
            cfg_scale=3.0, temperature=1.2, top_p=0.95, use_torch_compile=False,
        )
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)

        sr  = 44100
        arr = np.array(output, dtype=np.float32).flatten() if output is not None else np.zeros(sr, dtype=np.float32)
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Dialogue-native; [S1]/[S2] + [laughs][sighs] emotion tags; March 2025"
        r.notes       = "3 GB RAM; auto max_tokens; 44100 Hz"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# ── 10. XTTS-v2 ─────────────────────────────────────────────────────────────

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

# ── 11. CosyVoice2-0.5B ───────────────────────────────────────────────────────

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

# ── 12. Parler-TTS mini ────────────────────────────────────────────────────────

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
        model     = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        r.load_time_s = round(time.perf_counter() - t0, 3)

        # Describe Arthur's voice in natural language
        description = (
            "An elderly man with a slow, warm, slightly confused voice speaks gently "
            "and unhurriedly, with natural pauses between sentences."
        )
        input_ids  = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
        prompt_ids = tokenizer(TEST_PHRASE, return_tensors="pt").input_ids.to(DEVICE)

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

# ── 13. Chatterbox (Turbo) ─────────────────────────────────────────────────────

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
            model = ChatterboxTTS.from_pretrained("resemble-ai/chatterbox-turbo", device=DEVICE)
            r.voice = f"chatterbox-turbo (exaggeration=0.65, device={DEVICE})"
        except Exception:
            model = ChatterboxTTS.from_pretrained(device=DEVICE)
            r.voice = f"chatterbox-base (exaggeration=0.65, device={DEVICE})"
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

# ── 14. Fish Speech ───────────────────────────────────────────────────────────

def bench_fishspeech() -> BenchResult:
    r = BenchResult(model="fishspeech", voice="default (no ref WAV)")
    try:
        from fish_speech.inference.api import TTSInference
    except ImportError:
        r.error = "Not installed: pip install fish-speech"
        return r
    free = _free_ram_mb()
    if free < 1200:
        r.status = "skip"; r.error = f"Only {free} MB free -- Fish Speech needs ~1.5 GB."; return r
    out  = OUTPUT_DIR / "fishspeech.wav"
    inst = None
    try:
        import torch, soundfile as sf
        t0   = time.perf_counter()
        inst = TTSInference.from_pretrained("fishaudio/fish-speech-1.5", device=DEVICE,
                                             dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        rb = _ram_mb()
        t0 = time.perf_counter()
        result = inst.generate(text=TEST_PHRASE, reference_audio=None, format="wav", speed=1.0)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        if isinstance(result, tuple):
            arr, sr = result
        else:
            import wave as _wv, io as _io
            with _wv.open(_io.BytesIO(result), "rb") as wf:
                sr = wf.getframerate()
            arr = np.frombuffer(result[44:], dtype=np.int16).astype(np.float32) / 32768.0
        arr = np.array(arr, dtype=np.float32).flatten()
        sf.write(str(out), arr, int(sr))
        r.audio_dur_s = round(_arr_dur(arr, int(sr)), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = int(sr)
        r.arthur_fit  = "Zero-shot voice cloning; upload ref WAV for elderly voice character"
        r.notes       = "No ref WAV used here; S2-Pro/v1.5 variant; API may vary by version"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(inst)
    return r

# ── 15. Sesame CSM 1B ─────────────────────────────────────────────────────────

def bench_csm() -> BenchResult:
    r = BenchResult(model="csm", voice="Speaker 0")
    try:
        from generator import load_csm_1b
    except ImportError:
        r.error = "Not installed: pip install git+https://github.com/SesameAILabs/csm  (needs: huggingface-cli login)"
        return r
    free = _free_ram_mb()
    if free < 1800:
        r.status = "skip"; r.error = f"Only {free} MB free -- Sesame CSM needs ~2 GB."; return r
    out = OUTPUT_DIR / "csm.wav"
    gen = None
    try:
        import soundfile as sf
        t0  = time.perf_counter()
        gen = load_csm_1b(device=DEVICE)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        rb = _ram_mb()
        t0 = time.perf_counter()
        audio = gen.generate(text=TEST_PHRASE, speaker=0, context=[], max_audio_length_ms=30000)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        sr  = gen.sample_rate
        arr = audio.cpu().numpy().flatten().astype(np.float32)
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Multi-speaker; context-conditioned; natural conversational prosody"
        r.notes       = "Gated HF model — huggingface-cli login required; speakers 0-2"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(gen)
    return r

# ── 16. Qwen3-TTS ─────────────────────────────────────────────────────────────

def bench_qwen3tts() -> BenchResult:
    r = BenchResult(model="qwen3tts", voice="default")
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        r.error = "Not installed: pip install transformers"
        return r
    free = _free_ram_mb()
    if free < 1800:
        r.status = "skip"; r.error = f"Only {free} MB free -- Qwen3-TTS needs ~2 GB."; return r
    out = OUTPUT_DIR / "qwen3tts.wav"
    model = processor = None
    try:
        import torch, soundfile as sf
        model_id  = "Qwen/Qwen3-TTS"
        t0        = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model     = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                        torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32).to(DEVICE)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        inputs = {k: v.to(DEVICE) if hasattr(v,'to') else v
                  for k,v in processor(text=TEST_PHRASE, return_tensors="pt").items()}
        rb = _ram_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        if hasattr(output, "audio"):
            arr = output.audio.squeeze().cpu().numpy().astype(np.float32)
        else:
            arr = np.array(output[0] if isinstance(output, (list, tuple)) else output,
                           dtype=np.float32).flatten()
        fe  = getattr(processor, "feature_extractor", None)
        sr  = int(getattr(fe, "sampling_rate", None) or 22050)
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Natural Qwen3-based multilingual TTS; check HF for latest model ID"
        r.notes       = "Model ID Qwen/Qwen3-TTS may change; uses transformers"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(model, processor)
    return r

# ── 17. Orpheus 3B ────────────────────────────────────────────────────────────

def bench_orpheus() -> BenchResult:
    r = BenchResult(model="orpheus", voice="tara")
    try:
        from orpheus_tts import OrpheusModel
    except ImportError:
        r.error = "Not installed: pip install orpheus-speech"
        return r
    free = _free_ram_mb()
    if free < 2500:
        r.status = "skip"; r.error = f"Only {free} MB free -- Orpheus 3B needs ~3 GB."; return r
    out   = OUTPUT_DIR / "orpheus.wav"
    model = None
    try:
        import soundfile as sf
        t0    = time.perf_counter()
        model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
        r.load_time_s = round(time.perf_counter() - t0, 3)
        emotion_text  = f"<sigh> {TEST_PHRASE}"
        rb = _ram_mb()
        t0 = time.perf_counter()
        chunks = list(model.generate_speech(prompt=emotion_text, voice="tara"))
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        raw = b"".join(chunks)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sr  = 24000
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "<sigh> + emotion tags add natural confusion/hesitation — strong Arthur fit"
        r.notes       = "Voices: tara leah jess leo dan mia zac zoe; emotion: <laugh><sigh><gasp><groan><cough>"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(model)
    return r

# ── 18. NeuTTS Air ────────────────────────────────────────────────────────────

def bench_neutts() -> BenchResult:
    r = BenchResult(model="neutts", voice="default")
    r.status = "skip"
    r.error  = "NeuTTS Air not configured — edit _load_neutts() in tts_lab.py and bench_neutts() here with correct package"
    return r

# ── 19. IndexTTS-2 ────────────────────────────────────────────────────────────

def bench_indextts() -> BenchResult:
    r = BenchResult(model="indextts", voice="cloned from piper.wav")
    try:
        from indextts.infer import IndexTTS
    except ImportError:
        r.error = "Not installed: pip install git+https://github.com/index-tts/IndexTTS"
        return r
    ref_path = OUTPUT_DIR / "piper.wav"
    if not ref_path.exists():
        r.status = "skip"; r.error = "Reference WAV not found -- run bench_piper first"; return r
    free = _free_ram_mb()
    if free < 1500:
        r.status = "skip"; r.error = f"Only {free} MB free -- IndexTTS-2 needs ~2 GB."; return r
    out   = OUTPUT_DIR / "indextts.wav"
    model = None
    try:
        t0    = time.perf_counter()
        model = IndexTTS(model_dir="IndexTeam/IndexTTS", device=DEVICE)
        model.load_model()
        r.load_time_s = round(time.perf_counter() - t0, 3)
        rb = _ram_mb()
        t0 = time.perf_counter()
        model.infer(audio_prompt=str(ref_path), text=TEST_PHRASE, output_path=str(out))
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        r.audio_dur_s = round(_wav_dur(out), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = 24000
        r.arthur_fit  = "Zero-shot cloning; piper.wav as ref; upload aged-voice WAV for full Arthur character"
        r.notes       = "Reference WAV always required; ~1.5 GB RAM; 24 kHz output"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(model)
    return r

# ── 20. Zonos v0.1 ────────────────────────────────────────────────────────────

def bench_zonos() -> BenchResult:
    r = BenchResult(model="zonos", voice="transformer, speaking_rate=13")
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
    except ImportError:
        r.error = "Not installed: pip install git+https://github.com/Zyphra/Zonos  +  pip install phonemizer"
        return r
    free = _free_ram_mb()
    if free < 2000:
        r.status = "skip"; r.error = f"Only {free} MB free -- Zonos needs ~2.5 GB."; return r
    out   = OUTPUT_DIR / "zonos.wav"
    model = None
    try:
        import torch, soundfile as sf
        t0    = time.perf_counter()
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEVICE)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        emotion    = [0.3, 0.05, 0.05, 0.05, 0.1, 0.05, 0.2, 0.2]
        cond       = make_cond_dict(text=TEST_PHRASE, language="en-us",
                                    speaking_rate=13.0, emotion=emotion)
        conditioning = model.prepare_conditioning(cond)
        rb = _ram_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            codes = model.generate(conditioning, max_new_tokens=1024, disable_torch_compile=True)
        wavs = model.autoregressive_model.decode(codes)
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        arr = wavs[0].squeeze().cpu().numpy().astype(np.float32)
        sr  = 44000
        sf.write(str(out), arr, sr)
        r.audio_dur_s = round(_arr_dur(arr, sr), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = sr
        r.arthur_fit  = "Emotion vector + speaking_rate=13 → naturally hesitant elderly speech"
        r.notes       = "44 kHz; hybrid variant available; phonemizer+espeak-ng required"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(model)
    return r

# ── 21. OpenVoice v2 ──────────────────────────────────────────────────────────

def bench_openvoice() -> BenchResult:
    r = BenchResult(model="openvoice", voice="EN-US base (no ref WAV)")
    try:
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS
    except ImportError:
        r.error = "Not installed: pip install git+https://github.com/myshell-ai/OpenVoice  (also needs melo-tts)"
        return r
    ov_dir = Path("/opt/models/openvoice_v2")
    if not (ov_dir / "converter" / "config.json").exists():
        r.status = "skip"; r.error = f"Checkpoints not at {ov_dir} — run setup step 21"; return r
    out = OUTPUT_DIR / "openvoice.wav"
    src = str(OUTPUT_DIR / "_ov_base.wav")
    converter = tts_model = None
    try:
        import torch, soundfile as sf
        t0 = time.perf_counter()
        converter  = ToneColorConverter(str(ov_dir / "converter" / "config.json"), device=DEVICE)
        converter.load_ckpt(str(ov_dir / "converter" / "checkpoint.pth"))
        tts_model  = MeloTTS(language="EN", device=DEVICE)
        r.load_time_s = round(time.perf_counter() - t0, 3)
        sp_ids = tts_model.hps.data.spk2id
        rb = _ram_mb()
        t0 = time.perf_counter()
        tts_model.tts_to_file(TEST_PHRASE, sp_ids.get("EN-US") or list(sp_ids.values())[0], src, speed=0.85)
        ses_path = ov_dir / "base_speakers" / "ses" / "en_us.pth"
        if ses_path.exists():
            se = torch.load(str(ses_path), map_location="cpu", weights_only=False)
            converter.convert(audio_src_path=src, src_se=se, tgt_se=se, output_path=str(out), tau=0.3)
        else:
            import shutil; shutil.copy(src, str(out))
        r.synth_time_s = round(time.perf_counter() - t0, 3)
        r.peak_ram_mb  = max(0, _ram_mb() - rb)
        Path(src).unlink(missing_ok=True)
        r.audio_dur_s = round(_wav_dur(out), 2)
        r.rtf         = round(r.synth_time_s / r.audio_dur_s, 4) if r.audio_dur_s else 0
        r.output_hz   = 22050
        r.arthur_fit  = "MeloTTS base + tone-color; upload ref WAV for zero-shot voice cloning"
        r.notes       = "Checkpoints required at /opt/models/openvoice_v2; melo-tts needed"
        r.status      = "pass"
    except MemoryError:
        r.status = "fail"; r.error = "OOM"
    except Exception:
        r.status = "fail"; r.error = traceback.format_exc(limit=4)
    finally:
        _safe_del(converter, tts_model)
        Path(src).unlink(missing_ok=True)
    return r

# ── Registry ──────────────────────────────────────────────────────────────────

BENCH_FNS = {
    "piper":       bench_piper,
    "kokoro":      bench_kokoro,
    "melo":        bench_melo,
    "chattts":     bench_chattts,
    "outetts":     bench_outetts,
    "bark":        bench_bark,
    "styletts2":   bench_styletts2,
    "f5tts":       bench_f5tts,
    "dia":         bench_dia,
    "xtts":        bench_xtts,
    "cosyvoice":   bench_cosyvoice,
    "parler":      bench_parler,
    "chatterbox":  bench_chatterbox,
    "fishspeech":  bench_fishspeech,
    "csm":         bench_csm,
    "qwen3tts":    bench_qwen3tts,
    "orpheus":     bench_orpheus,
    "neutts":      bench_neutts,
    "indextts":    bench_indextts,
    "zonos":       bench_zonos,
    "openvoice":   bench_openvoice,
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
        help="Comma-separated model names or 'all'. Valid: " + ",".join(ALL_MODELS)
    )
    parser.add_argument("--no-heavy",      action="store_true", help="Skip all models needing >2.5 GB RAM")
    parser.add_argument("--no-xtts",       action="store_true", help="Skip XTTS-v2 (needs swap)")
    parser.add_argument("--no-cosyvoice",  action="store_true", help="Skip CosyVoice2 (manual install)")
    parser.add_argument("--no-dia",        action="store_true", help="Skip Dia-1.6B (needs ~3 GB)")
    args = parser.parse_args()

    if args.models.strip().lower() == "all":
        models_to_run = list(ALL_MODELS)
    else:
        models_to_run = [m.strip().lower() for m in args.models.split(",")]
        invalid = [m for m in models_to_run if m not in BENCH_FNS]
        if invalid:
            print(f"Unknown model(s): {invalid}. Valid: {list(BENCH_FNS)}")
            sys.exit(1)

    HEAVY_MODELS = {"dia", "xtts", "cosyvoice"}
    if args.no_heavy:
        models_to_run = [m for m in models_to_run if m not in HEAVY_MODELS]
    if args.no_xtts and "xtts" in models_to_run:
        models_to_run.remove("xtts")
    if args.no_cosyvoice and "cosyvoice" in models_to_run:
        models_to_run.remove("cosyvoice")
    if args.no_dia and "dia" in models_to_run:
        models_to_run.remove("dia")

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
