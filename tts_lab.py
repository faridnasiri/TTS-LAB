#!/usr/bin/env python3
"""
Arthur TTS Lab -- 21-Engine Edition
Piper  Kokoro  MeloTTS  ChatTTS  OuteTTS  Bark  StyleTTS2
F5-TTS  Dia-1.6B  XTTS-v2  CosyVoice2  Parler-TTS  Chatterbox
FishSpeech  Sesame-CSM  Qwen3-TTS  Orpheus  NeuTTS-Air  IndexTTS  Zonos  OpenVoice

Port  : 8001
Open  : http://192.168.0.87:8001
"""
from __future__ import annotations
import asyncio, base64, gc, io, json, os, shutil, sys, tempfile, threading, time, traceback, uuid, wave
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# -- Thread-pool pinning (before torch/ort) --
_N_CORES = os.cpu_count() or 6
os.environ.setdefault("OMP_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("MKL_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(_N_CORES))
os.environ.setdefault("ORT_NUM_THREADS",      str(_N_CORES))
# Bark model cache on data disk
os.environ.setdefault("XDG_CACHE_HOME", "/opt/models/cache")
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "False")   # full Bark models — we have 16 GB VRAM
try:
    import torch
    torch.set_num_threads(_N_CORES)
    torch.set_num_interop_threads(max(1, _N_CORES // 2))
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    VRAM_TOTAL_MB = int(torch.cuda.get_device_properties(0).total_memory / 1048576) if DEVICE == "cuda" else 0
except Exception:
    DEVICE = "cpu"; DEVICE_NAME = "CPU"; VRAM_TOTAL_MB = 0

# -- Paths --
MODELS_DIR    = Path(__file__).parent / "models"
COSYVOICE_DIR = Path("/opt/CosyVoice")
UPLOAD_DIR    = Path("/tmp/tts_uploads")
MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# -- Kokoro voice catalogue (54 voices) --
ALL_KOKORO_VOICES = [
    "bm_daniel","bm_fable","bm_george","bm_lewis",
    "bf_alice","bf_emma","bf_isabella","bf_lily",
    "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa",
    "af_alloy","af_aoede","af_bella","af_heart","af_jessica","af_kore",
    "af_nicole","af_nova","af_river","af_sarah","af_sky",
    "ef_dora","em_alex","em_santa","ff_siwis",
    "hf_alpha","hf_beta","hm_omega","hm_psi",
    "if_sara","im_nicola",
    "jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro","jm_kumo",
    "pf_dora","pm_alex","pm_santa",
    "zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi",
    "zm_yunjian","zm_yunxi","zm_yunxia","zm_yunyang",
]
KOKORO_LANG_MAP = {
    "am":"en-us","af":"en-us","bm":"en-gb","bf":"en-gb",
    "ef":"es","em":"es","ff":"fr","hf":"hi","hm":"hi",
    "if":"it","im":"it","jf":"ja","jm":"ja","pf":"pt-br","pm":"pt-br","zf":"zh","zm":"zh",
}

# -- XTTS catalogues --
ALL_XTTS_SPEAKERS = [
    "Aaron Dreschner","Abrahan Mack","Adde Michal","Alexandra Hisakawa","Alison Dietlinde",
    "Alma Maria","Ana Florence","Andrew Chipper","Annmarie Nele","Asya Anara",
    "Badr Odhiambo","Baldur Sanjin","Barbora MacLean","Brenda Stern","Camilla Holmstrom",
    "Chandra MacFarland","Claribel Dervla","Craig Gutsy","Daisy Studious","Damien Black",
    "Damjan Chapman","Dionisio Schuyler","Eugenio Mataracı","Ferran Simen","Filip Traverse",
    "Gilberto Mathias","Gitta Nikolina","Gracie Wise","Henriette Usha","Ige Behringer",
    "Ilkin Urbano","Kazuhiko Atallah","Kumar Dahl","Lidiya Szekeres","Lilya Stainthorpe",
    "Ludvig Milivoj","Luis Moray","Maja Ruoho","Marcos Rudaski","Narelle Moon",
    "Nova Hogarth","Rosemary Okafor","Royston Min","Sofia Hellen","Suad Qasim",
    "Szofi Granger","Tammie Ema","Tammy Grit","Tanja Adelina","Torcull Diarmuid",
    "Uta Obando","Viktor Eka","Viktor Menelaos","Vjollca Johnnie","Wulf Carlevaro",
    "Xavier Hayasaka","Zacharie Aimilios","Zofija Kendrick",
]
XTTS_LANGUAGES = {
    "en":"English","fr":"French","de":"German","es":"Spanish","it":"Italian",
    "pt":"Portuguese","pl":"Polish","cs":"Czech","ar":"Arabic","hi":"Hindi",
    "hu":"Hungarian","ja":"Japanese","ko":"Korean","nl":"Dutch","ru":"Russian",
    "tr":"Turkish","zh-cn":"Chinese",
}

# -- Bark voice presets --
BARK_PRESETS = [
    ("v2/en_speaker_6", "en_speaker_6 — male, measured (best Arthur)"),
    ("v2/en_speaker_9", "en_speaker_9 — male, older"),
    ("v2/en_speaker_0", "en_speaker_0 — male, deep"),
    ("v2/en_speaker_1", "en_speaker_1 — male, warm"),
    ("v2/en_speaker_7", "en_speaker_7 — male, elderly"),
    ("v2/en_speaker_3", "en_speaker_3 — male, gravelly"),
    ("v2/en_speaker_4", "en_speaker_4 — male, neutral"),
    ("v2/en_speaker_2", "en_speaker_2 — female"),
    ("v2/en_speaker_5", "en_speaker_5 — female"),
    ("v2/en_speaker_8", "en_speaker_8 — female, soft"),
]

CHATTTS_SPEEDS = [(f"[speed_{i}]", f"speed_{i}") for i in range(1, 10)]
OUTETTS_MODELS = [
    ("OuteAI/OuteTTS-0.3-500M", "OuteTTS 0.3 500M (default)"),
    ("OuteAI/OuteTTS-1.0-0.6B", "OuteTTS 1.0 0.6B"),
    ("OuteAI/Llama-OuteTTS-1.0-1B", "OuteTTS 1.0 1B"),
]
OUTETTS_SPEAKERS = [("en-female-1-neutral", "en-female-1-neutral")]
PARLER_MODELS = [
    ("parler-tts/parler-tts-mini-v1", "Mini v1"),
    ("parler-tts/parler-tts-mini-expresso", "Mini Expresso"),
]

ORPHEUS_VOICES       = [("tara","tara"),("leah","leah"),("jess","jess"),("leo","leo"),
                        ("dan","dan"),("mia","mia"),("zac","zac"),("zoe","zoe")]
ZONOS_VARIANTS       = [("transformer","Transformer (quality, ~1.2 GB)"),
                        ("hybrid","Hybrid (faster, ~1.5 GB)")]
CSM_SPEAKERS         = [(str(i), f"Speaker {i}") for i in range(3)]
OPENVOICE_MODELS_DIR = Path("/opt/models/openvoice_v2")
INDEXTTS_DIR         = Path("/opt/models/indextts")

# -- Model registry --
MODEL_INFO = {
    "piper":     {"label":"Piper TTS",    "size":"61-116 MB","rtf_est":"RTF 0.08 (GPU)", "ram_est_mb":200,  "heavy":False,"notes":"6 voices. ONNX CPU-only (GPU: CUDA EP). Real-time on any hardware. Best for production.","arthur_fit":2},
    "kokoro":    {"label":"Kokoro-82M",   "size":"89 MB",    "rtf_est":"RTF 0.06 (GPU)","ram_est_mb":500,  "heavy":False,"notes":"54 voices, 9 languages. bm_lewis is the best Arthur voice. GPU: real-time at RTF ~0.06.","arthur_fit":5},
    "melo":      {"label":"MeloTTS",      "size":"200 MB",   "rtf_est":"RTF 0.05 (GPU)","ram_est_mb":1200, "heavy":False,"notes":"5 English accents. EN-BR sounds slightly older. GPU: blazing fast RTF ~0.05.","arthur_fit":3},
    "chattts":   {"label":"ChatTTS",      "size":"1.2-2.3 GB","rtf_est":"RTF ~0.18 (GPU)","ram_est_mb":1800, "heavy":True, "notes":"Conversational TTS with speed prompts, speaker sampling, and optional reference-speaker extraction. GPU: real-time.","arthur_fit":4},
    "outetts":   {"label":"OuteTTS",      "size":"1.0-2.4 GB","rtf_est":"RTF ~0.38 (GPU)","ram_est_mb":1600, "heavy":True, "notes":"Prompt-controlled character voice. GPU: real-time (~0.38 RTF).","arthur_fit":4},
    "bark":      {"label":"Bark",         "size":"2.5 GB (full)","rtf_est":"RTF ~0.50 (GPU)","ram_est_mb":3000, "heavy":True, "notes":"Full-size Bark models (16 GB VRAM). Unique emotion tokens: [laughs] [sighs] [clears throat] [hesitantly]. GPU required.","arthur_fit":5},
    "styletts2": {"label":"StyleTTS 2",   "size":"0.7 GB",   "rtf_est":"RTF 0.06 (GPU)","ram_est_mb":1500, "heavy":True, "notes":"Fastest high-quality neural TTS. Style transfer from reference WAV. Alpha/beta control. GPU: very fast.","arthur_fit":4},
    "f5tts":     {"label":"F5-TTS",       "size":"1.2 GB",   "rtf_est":"RTF ~0.08 (GPU)","ram_est_mb":2000, "heavy":True, "notes":"Best zero-shot voice cloning. Flow matching. Upload 5-15s reference WAV. GPU: real-time.","arthur_fit":4},
    "dia":       {"label":"Dia-1.6B",     "size":"3 GB",     "rtf_est":"RTF ~0.95 (GPU)","ram_est_mb":3000, "heavy":True, "notes":"Dialogue-native. [S1]/[S2] speakers + [laughs] [sighs] emotion tags. GPU: borderline real-time.","arthur_fit":5},
    "xtts":      {"label":"XTTS-v2",      "size":"1.8 GB",   "rtf_est":"RTF ~0.12 (GPU)","ram_est_mb":3200, "heavy":True, "notes":"58 speakers, 17 languages. Voice cloning. Best multi-speaker quality. GPU: real-time.","arthur_fit":5},
    "cosyvoice": {"label":"CosyVoice2",   "size":"2 GB",     "rtf_est":"RTF ~0.28 (GPU)","ram_est_mb":2500, "heavy":True, "notes":"Chinese-first with English zero-shot support. GPU: real-time.","arthur_fit":3},
    "parler":    {"label":"Parler-TTS",   "size":"2.5-3.3 GB","rtf_est":"RTF ~0.42 (GPU)","ram_est_mb":1500, "heavy":True, "notes":"Voice controlled entirely by natural language description. GPU: real-time.","arthur_fit":4},
    "chatterbox": {"label":"Chatterbox",  "size":"3.0 GB",   "rtf_est":"RTF ~0.38 (GPU)","ram_est_mb":1800, "heavy":True, "notes":"Exaggeration slider + voice cloning. Most controllable confusion. GPU: real-time.","arthur_fit":5},
    "fishspeech": {"label":"Fish Speech",  "size":"~1.1 GB",  "rtf_est":"RTF ~0.14 (GPU)","ram_est_mb":1500, "heavy":True, "notes":"Zero-shot voice cloning (VQ-VAE codec). Upload 5-30s reference WAV. GPU: real-time.","arthur_fit":4},
    "csm":        {"label":"Sesame CSM 1B","size":"~2 GB",    "rtf_est":"RTF ~0.08 (GPU)","ram_est_mb":2000, "heavy":True, "notes":"Conversational Speech Model 1B. Multi-speaker. Context-conditioned. HF login required. GPU: real-time.","arthur_fit":4},
    "qwen3tts":   {"label":"Qwen3-TTS",   "size":"~1-3 GB",  "rtf_est":"RTF ~0.32 (GPU)","ram_est_mb":2000, "heavy":True, "notes":"Alibaba Qwen3-based TTS. GPU: real-time.","arthur_fit":3},
    "orpheus":    {"label":"Orpheus 3B",   "size":"~3 GB",    "rtf_est":"RTF ~0.78 (GPU)","ram_est_mb":3000, "heavy":True, "notes":"LLaMA-3B-based TTS. Emotion tags: <laugh> <sigh> <chuckle> <gasp>. 8 voices. GPU required (vllm).","arthur_fit":5},
    "neutts":     {"label":"NeuTTS Air",   "size":"TBD",      "rtf_est":"TBD",           "ram_est_mb":1000, "heavy":True, "notes":"Not yet configured — edit _load_neutts() with the correct package import + install.","arthur_fit":3},
    "indextts":   {"label":"IndexTTS-2",   "size":"~1.5 GB",  "rtf_est":"RTF ~0.10 (GPU)","ram_est_mb":2000, "heavy":True, "notes":"Zero-shot voice cloning from IndexTeam. Reference WAV required. GPU: real-time.","arthur_fit":4},
    "zonos":      {"label":"Zonos v0.1",   "size":"~1.2 GB",  "rtf_est":"RTF ~0.06 (GPU)","ram_est_mb":2500, "heavy":True, "notes":"Hybrid/Transformer from Zyphra. Emotion vector + speaking-rate control. 44 kHz output. GPU: very fast.","arthur_fit":4},
    "openvoice":  {"label":"OpenVoice v2", "size":"~600 MB",  "rtf_est":"RTF ~0.10 (GPU)","ram_est_mb":1500, "heavy":True, "notes":"MeloTTS base + tone-color conversion. Zero-shot voice cloning. GPU: real-time.","arthur_fit":3},
}

MODEL_ORDER = ["piper","kokoro","melo","chattts","outetts","bark","styletts2","f5tts","dia","xtts","cosyvoice","parler","chatterbox",
               "fishspeech","csm","qwen3tts","orpheus","neutts","indextts","zonos","openvoice"]

ARTHUR_PRESETS = [
    ("Greeting",   "Hello? Oh my goodness, who is this? I almost didn't hear the phone. Let me turn the TV down a moment."),
    ("Confused",   "Now where did I put that... oh, I'm sorry dear, I've gone and confused myself again. What was that number you gave me?"),
    ("Intel hunt", "Can you give me that badge number one more time, nice and slow? And the website, could you spell that out letter by letter for me?"),
    ("Stage 4",    "I've been very patient but I can't find my reading glasses and Mr. Whiskers knocked the paper off the table and I keep losing the case number."),
    ("Short",      "Oh, just a moment dear."),
]

BARK_ARTHUR_PRESETS = [
    ("[sighs] Hello? Oh my goodness, who is this? [clears throat] Let me turn the TV down.",
     "Bark: Greeting with sighs"),
    ("[hesitantly] Now where did I put that... [long pause] oh I'm sorry dear, I've gone and confused myself. [laughs nervously]",
     "Bark: Confused with tokens"),
    ("Can you give me that badge number one more time [pause] nice and slow? I keep writing it down and losing it. [sighs]",
     "Bark: Intel hunt"),
]

HEAVY = {n for n, i in MODEL_INFO.items() if i["heavy"]}

_state = {
    n: {"instance":None,"status":"unloaded","lock":threading.Lock(),
        "error":"","load_time_s":0.0,"loaded_voice":None,"loaded_model":None}
    for n in MODEL_ORDER
}

# -- Utilities --
def _ram_mb():
    try:
        import psutil; v = psutil.virtual_memory()
        return v.total//1048576, v.used//1048576, v.available//1048576
    except Exception:
        return 16384, 0, 16384

def _to_wav(audio, sr):
    arr = np.array(audio, dtype=np.float64).flatten()
    if arr.dtype != np.int16:
        arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr))
        wf.writeframes(arr.tobytes())
    return buf.getvalue()

def _wav_dur(wav):
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def _safe_del(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect()

def _evict_heavy(keep):
    for n in HEAVY:
        if n != keep and _state[n]["instance"] is not None:
            _safe_del(_state[n]["instance"])
            _state[n]["instance"] = None
            _state[n]["status"]   = "unloaded"

def _piper_voices():
    return sorted(p.stem for p in MODELS_DIR.glob("*.onnx") if "kokoro" not in p.name)

def _read_wav_mono_f32(path: Path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        fr         = wf.getframerate()
        frames     = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise RuntimeError("Reference WAV must be 16-bit PCM.")
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, fr

def _require_gpu(engine: str):
    """Raise immediately if no CUDA GPU is present.
    Call this at the TOP of loaders that are known to be impossible on CPU.
    Fails in milliseconds instead of hanging for minutes.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{engine} requires a CUDA GPU and will not run on CPU.\n"
                "Benchmarked result: timeout / error on CPU.\n"
                "Add a GPU and restart the server, then try again."
            )
    except ImportError:
        pass  # if torch isn't importable yet, let the loader handle it


# ============================================================
# LOADERS + SYNTH FUNCTIONS
# ============================================================

# -- 1. Piper --
def _load_piper(voice="en_US-ryan-high"):
    import onnxruntime as ort
    from piper.voice import PiperVoice
    mp = MODELS_DIR / f"{voice}.onnx"; cp = MODELS_DIR / f"{voice}.onnx.json"
    if not mp.exists(): raise FileNotFoundError(f"Piper voice not found: {mp}")
    # Piper is a tiny ONNX model (~50 MB); CUDA EP overhead > GPU benefit — keep CPU.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _N_CORES; opts.inter_op_num_threads = max(1, _N_CORES // 2)
    try:    return PiperVoice.load(str(mp), config_path=str(cp) if cp.exists() else None, use_cuda=False, sess_options=opts)
    except TypeError: return PiperVoice.load(str(mp), config_path=str(cp) if cp.exists() else None, use_cuda=False)

def _synth_piper(inst, text, params):
    sr = inst.config.sample_rate; raw = bytearray()
    spd = float(params.get("speed", 1.0))
    try:
        from piper.config import SynthesisConfig
        cfg = SynthesisConfig(
            length_scale=float(params.get("length_scale", 1.0/spd if spd!=1.0 else 1.0)),
            noise_scale=float(params.get("noise_scale", 0.667)),
            noise_w=float(params.get("noise_w", 0.8)))
        for chunk in inst.synthesize(text, cfg): raw.extend(chunk.audio_int16_bytes); sr=chunk.sample_rate
    except (ImportError, TypeError):
        for chunk in inst.synthesize(text): raw.extend(chunk.audio_int16_bytes); sr=chunk.sample_rate
    return _to_wav(np.frombuffer(bytes(raw), dtype=np.int16).astype(np.float32)/32767, sr), sr

# -- 2. Kokoro --
def _load_kokoro():
    # kokoro-onnx 0.5.x calls EspeakWrapper.set_data_path() but phonemizer 3.3+
    # renamed it.  Also point espeak at system data (espeakng-loader bundled path
    # is a CI build path that doesn't exist on the VM).
    ESPEAK_DATA = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper as _EW
        if not hasattr(_EW, "set_data_path"):
            _EW.set_data_path = classmethod(lambda cls, p: None)
        import os; os.environ.setdefault("ESPEAK_DATA_PATH", ESPEAK_DATA)
    except Exception:
        pass
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    mp = MODELS_DIR/"kokoro-v1.0.onnx"; vp = MODELS_DIR/"voices-v1.0.bin"
    if not mp.exists(): raise FileNotFoundError("kokoro-v1.0.onnx missing")
    # Kokoro-ONNX (82 MB): CUDA EP latency > CPU for this model size — use CPU.
    # True GPU speedup requires the PyTorch kokoro package (different install).
    opts = ort.SessionOptions(); opts.intra_op_num_threads = _N_CORES
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    try:
        from kokoro_onnx.config import EspeakConfig
        return Kokoro(str(mp), str(vp), espeak_config=EspeakConfig(data_path=ESPEAK_DATA))
    except (ImportError, TypeError, AttributeError):
        pass
    try:    return Kokoro(str(mp), str(vp), sess_options=opts)
    except TypeError: return Kokoro(str(mp), str(vp))

def _synth_kokoro(inst, text, params):
    voice = params.get("voice", "bm_lewis")
    lang  = params.get("lang") or KOKORO_LANG_MAP.get(voice[:2], "en-us")
    samples, sr = inst.create(text, voice=voice, speed=float(params.get("speed", 0.85)), lang=lang)
    return _to_wav(np.array(samples, dtype=np.float32), sr), sr

# -- 3. MeloTTS --
def _load_melo():
    from melo.api import TTS
    return TTS(language="EN", device=DEVICE)

def _synth_melo(inst, text, params):
    sp_ids = dict(inst.hps.data.spk2id)
    sp = params.get("speaker","EN-US").replace("-","_").replace("EN_US","EN-US").replace("EN_BR","EN-BR").replace("EN_AU","EN-AU")
    sp_id = sp_ids.get(sp) or sp_ids.get("EN-US") or list(sp_ids.values())[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: tmp = f.name
    inst.tts_to_file(text, sp_id, tmp, speed=float(params.get("speed", 0.85)))
    wav = Path(tmp).read_bytes(); Path(tmp).unlink(missing_ok=True)
    with wave.open(io.BytesIO(wav),"rb") as wf: sr = wf.getframerate()
    return wav, sr

# -- 4. ChatTTS --
def _load_chattts():
    import ChatTTS
    inst = ChatTTS.Chat()
    if not inst.load(source="huggingface", device=DEVICE):
        raise RuntimeError("ChatTTS load failed")
    try:
        inst._arthur_spk = inst.sample_random_speaker()
    except Exception:
        inst._arthur_spk = None
    return inst

def _synth_chattts(inst, text, params):
    import torch
    spk_emb = getattr(inst, "_arthur_spk", None)
    prompt_id = params.get("audio_prompt_id", "")
    if prompt_id:
        prompt_path = UPLOAD_DIR / f"{prompt_id}.wav"
        if prompt_path.exists():
            prompt_wav, _ = _read_wav_mono_f32(prompt_path)
            spk_emb = inst.sample_audio_speaker(prompt_wav)
    # Use a fixed default seed — random seeds cause narrow() with certain
    # GPU memory layouts (ChatTTS 0.2.4 + PyTorch 2.10 incompatibility).
    seed = int(float(params.get("seed", 0))) or 2024
    infer_kw = dict(
        prompt=params.get("prompt", "[speed_5]"),
        top_P=float(params.get("top_p", 0.7)),
        top_K=int(float(params.get("top_k", 20))),
        temperature=float(params.get("temperature", 0.3)),
        repetition_penalty=float(params.get("repetition_penalty", 1.05)),
        max_new_token=int(float(params.get("max_new_token", 512))),
        show_tqdm=False,
        spk_emb=spk_emb,
        manual_seed=seed,
    )
    skip = str(params.get("skip_refine_text", "true")).lower() in ("1", "true", "yes", "on")
    # GPU memory fragmentation after other model loads causes 0-token generation
    # which triggers narrow() in the DVAE. Clear cache before inference.
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _last_err = None
    for _bump in [0, 7, 42, 100]:
        _kw = dict(infer_kw, manual_seed=seed + _bump)
        try:
            out = inst.infer(text, skip_refine_text=skip,
                             params_infer_code=inst.InferCodeParams(**_kw))
            _last_err = None
            break
        except RuntimeError as _e:
            if "narrow" in str(_e) or "length must be non-negative" in str(_e):
                _last_err = _e
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    if _last_err is not None:
        raise RuntimeError(
            f"ChatTTS narrow() after 4 attempts — GPU state issue: {_last_err}\n"
            "Try reloading the engine (DELETE /models/chattts) then retrying."
        ) from _last_err
    arr = np.array(out[0] if isinstance(out, list) else out, dtype=np.float32)
    return _to_wav(arr, 24000), 24000

# -- 5. OuteTTS --
def _load_outetts(model_path="OuteAI/OuteTTS-0.3-500M"):
    import outetts
    # OuteTTS HF backend pre-encodes text as ~15 K positional tokens regardless
    # of text length, always exceeding max_seq_length → generation fails.
    # Fix: use GGUF model with LLAMACPP backend.  Download once:
    #   huggingface-cli download OuteAI/OuteTTS-0.3-500M-GGUF \
    #       --local-dir /opt/models/outetts-gguf --include "*.gguf"
    # Then pass model_path="/opt/models/outetts-gguf/model.gguf"
    gguf_path = model_path  # caller can pass an explicit .gguf path
    if not gguf_path.endswith(".gguf"):
        raise RuntimeError(
            "OuteTTS-0.3-500M HF backend is broken: pre-encodes text as ~15K tokens "
            "exceeding any max_seq_length.\n"
            "Use GGUF: huggingface-cli download OuteAI/OuteTTS-0.3-500M-GGUF "
            "--local-dir /opt/models/outetts-gguf --include '*.gguf'\n"
            "Then POST params: {\"model_path\":\"/opt/models/outetts-gguf/model.gguf\"}"
        )
    cfg = outetts.ModelConfig(
        model_path=gguf_path,
        tokenizer_path="OuteAI/OuteTTS-0.3-500M",
        backend=outetts.Backend.LLAMACPP,
        device=DEVICE,
    )
    return outetts.Interface(cfg)

def _synth_outetts(inst, text, params):
    import outetts
    speaker = None
    prompt_id = params.get("audio_prompt_id", "")
    if prompt_id:
        prompt_path = UPLOAD_DIR / f"{prompt_id}.wav"
        if prompt_path.exists():
            speaker = inst.create_speaker(str(prompt_path), transcript=params.get("transcript", "") or None)
    if speaker is None:
        speaker = inst.load_default_speaker(params.get("speaker", "en-female-1-neutral"))
    # OuteTTS HF backend pre-encodes text+audio vocab → even "Hello" = 14K tokens.
    # CHUNKED mode splits text into small pieces and generates each within the
    # model window. Do NOT pass max_length with CHUNKED (outetts manages sizing).
    try:
        gen_type = outetts.GenerationType.CHUNKED
        gen_cfg_kw = dict(
            text=text,
            voice_characteristics=params.get("voice_characteristics") or None,
            speaker=speaker,
            generation_type=gen_type,
        )
    except AttributeError:
        # Older outetts without CHUNKED — use REGULAR with a large max_length
        gen_cfg_kw = dict(
            text=text,
            voice_characteristics=params.get("voice_characteristics") or None,
            speaker=speaker,
            max_length=8192,
        )
    gen_cfg_kw["sampler_config"] = outetts.SamplerConfig(
        temperature=float(params.get("temperature", 0.4)),
        repetition_penalty=float(params.get("repetition_penalty", 1.1)),
        top_k=int(float(params.get("top_k", 40))),
        top_p=float(params.get("top_p", 0.9)),
        min_p=float(params.get("min_p", 0.05)),
    )
    out = inst.generate(outetts.GenerationConfig(**gen_cfg_kw))
    arr = out.audio.detach().cpu().numpy().squeeze()
    return _to_wav(arr, getattr(out, "sr", 44100)), getattr(out, "sr", 44100)

# -- 6. Bark --
def _load_bark():
    import torch
    # Bark checkpoints contain numpy scalars not whitelisted in PyTorch 2.6+ weights_only mode.
    # Patch torch.load to allow legacy pickles only during preload, then restore.
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    try:
        # Full-size models on GPU (16 GB VRAM); small models only as CPU fallback
        _use_small = (DEVICE != "cuda")
        os.environ["SUNO_USE_SMALL_MODELS"] = "True" if _use_small else "False"
        from bark import preload_models
        preload_models(
            text_use_small=_use_small, coarse_use_small=_use_small,
            fine_use_small=_use_small,
            text_use_gpu=(DEVICE == "cuda"), coarse_use_gpu=(DEVICE == "cuda"),
            fine_use_gpu=(DEVICE == "cuda"),
        )
        from bark.generation import SAMPLE_RATE
    finally:
        torch.load = _orig
    return {"sr": SAMPLE_RATE}

def _synth_bark(inst, text, params):
    from bark import generate_audio
    from bark.generation import SAMPLE_RATE
    preset = params.get("voice_preset", "v2/en_speaker_6")
    history = preset if preset != "none" else None
    audio = generate_audio(text, history_prompt=history)
    return _to_wav(audio.astype(np.float32), SAMPLE_RATE), SAMPLE_RATE

# -- 7. StyleTTS 2 --
def _load_styletts2():
    import torch
    # StyleTTS2 checkpoints use builtins.getattr not whitelisted in PyTorch 2.6+.
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    try:
        from styletts2 import tts
        result = tts.StyleTTS2()
    finally:
        torch.load = _orig
    return result

def _synth_styletts2(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
    if ref_path and not Path(ref_path).exists(): ref_path = None
    out = inst.inference(
        text=text,
        target_voice_path=ref_path,
        alpha=float(params.get("alpha", 0.3)),
        beta=float(params.get("beta", 0.7)),
        diffusion_steps=int(float(params.get("diffusion_steps", 5))),
        embedding_scale=float(params.get("embedding_scale", 1.0)),
    )
    return _to_wav(np.array(out, dtype=np.float32), 24000), 24000

# -- 8. F5-TTS --
def _load_f5tts():
    from f5_tts.api import F5TTS
    return F5TTS()

def _synth_f5tts(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = UPLOAD_DIR / f"{ref_id}.wav" if ref_id else None
    if not ref_path or not ref_path.exists():
        raise RuntimeError("F5-TTS requires a reference audio clip. Upload a 5-15s WAV first.")
    ref_text = params.get("ref_text", "")
    speed    = float(params.get("speed", 1.0))
    nfe      = int(float(params.get("nfe_step", 32)))
    wav, sr, _ = inst.infer(
        ref_file=str(ref_path), ref_text=ref_text,
        gen_text=text, speed=speed, nfe_step=nfe,
    )
    return _to_wav(np.array(wav, dtype=np.float32).flatten(), sr), sr

# -- 9. Dia-1.6B --
def _load_dia():
    from dia.model import Dia
    # Use bfloat16 on GPU for speed; float32 on CPU for compatibility
    _dtype = "bfloat16" if DEVICE == "cuda" else "float32"
    # Dia-1.6B-0626 has updated config schema matching current package;
    # fall back to Dia-1.6B for cached weights
    for mid in ["nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"]:
        try:
            return Dia.from_pretrained(mid, compute_dtype=_dtype, device=DEVICE)
        except TypeError:
            # older Dia versions don't accept device= kwarg
            try:
                return Dia.from_pretrained(mid, compute_dtype=_dtype)
            except Exception as e:
                if mid == "nari-labs/Dia-1.6B": raise
                last_err = e; continue
        except Exception as e:
            if mid == "nari-labs/Dia-1.6B": raise
            last_err = e; continue

def _synth_dia(inst, text, params):
    if "[S1]" not in text and "[S2]" not in text:
        text = f"[S1] {text}"
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
    # Auto-estimate max_tokens: ~86 Dia tokens per second of audio,
    # rough estimate is 15 chars ≈ 1 second of speech → cap at 1024 unless overridden
    auto_tokens = min(1024, max(256, len(text) * 6))
    ui_val = int(float(params.get("max_tokens", 0)))
    kw = dict(
        max_tokens=ui_val if ui_val > 0 else auto_tokens,
        cfg_scale=float(params.get("cfg_scale", 3.0)),
        temperature=float(params.get("temperature", 1.2)),
        top_p=float(params.get("top_p", 0.95)),
        use_torch_compile=False,
    )
    if ref_path and Path(ref_path).exists():
        kw["audio_prompt_path"] = ref_path
    output = inst.generate(text, **kw)
    sr = 44100
    arr = np.array(output, dtype=np.float32).flatten() if output is not None else np.zeros(sr, dtype=np.float32)
    return _to_wav(arr, sr), sr

# -- 10. XTTS-v2 --
def _patch_transformers_for_coqui():
    try:
        import transformers.utils.import_utils as iu
        if not hasattr(iu,"is_torch_greater_or_equal"):
            from packaging.version import Version; import torch
            iu.is_torch_greater_or_equal = lambda v: Version(torch.__version__)>=Version(v)
        if not hasattr(iu,"is_torchcodec_available"):
            iu.is_torchcodec_available = lambda: False
    except Exception: pass
    try:
        import transformers.pytorch_utils as pu
        if not hasattr(pu,"isin_mps_friendly"):
            import torch; pu.isin_mps_friendly = lambda e,t: torch.isin(e,t)
    except Exception: pass

def _load_xtts():
    _patch_transformers_for_coqui()
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2",
               progress_bar=False, gpu=(DEVICE == "cuda"))

def _synth_xtts(inst, text, params):
    kw = dict(text=text, speaker=params.get("speaker","Torcull Diarmuid"), language=params.get("language","en"))
    if params.get("temperature"): kw["temperature"] = float(params["temperature"])
    arr = inst.tts(**kw)
    return _to_wav(np.array(arr, dtype=np.float32), 24000), 24000

# -- 11. CosyVoice2 --
def _load_cosyvoice():
    # hyperpyyaml is a CosyVoice dependency not always installed by default
    import importlib.util as _ilu
    if not _ilu.find_spec("hyperpyyaml"):
        raise ImportError(
            "CosyVoice2 requires hyperpyyaml — run:  pip install hyperpyyaml\n"
            "Then restart the server."
        )
    for p in [str(COSYVOICE_DIR), str(COSYVOICE_DIR/"third_party"/"Matcha-TTS")]:
        if p not in sys.path: sys.path.insert(0, p)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    md = COSYVOICE_DIR/"pretrained_models"/"CosyVoice2-0.5B"
    return CosyVoice2(str(md), load_jit=False, load_trt=False)

def _synth_cosyvoice(inst, text, params):
    chunks = [c["tts_speech"].numpy().flatten() for c in inst.inference_sft(text, params.get("speaker","English Female"))]
    sr = inst.sample_rate
    return _to_wav(np.concatenate(chunks) if chunks else np.zeros(sr,np.float32), sr), sr

# -- 12. Parler-TTS --
def _load_parler(model_id="parler-tts/parler-tts-mini-v1"):
    # parler-tts 0.2.x requires transformers==4.46.1 (hard pin in setup.py).
    # The bench env uses transformers 4.57.6 (needed for vllm/Orpheus).
    # These are fundamentally incompatible — parler needs its own venv.
    # To use parler: create a separate venv with transformers==4.46.1 and
    # run a dedicated parler server, or wait for a parler-tts release that
    # supports transformers >=4.51.
    import transformers
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tv >= (4, 51):
        raise RuntimeError(
            f"Parler-TTS 0.2.x requires transformers<=4.46.1 "
            f"(installed: {transformers.__version__}). "
            "Install in a separate venv: pip install transformers==4.46.1"
        )
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    mdl = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(model_id)
    return (mdl, tok)

def _synth_parler(inst, text, params):
    import torch
    model, tok = inst
    desc = params.get("description","An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly.")
    iids = tok(desc, return_tensors="pt").input_ids.to(DEVICE)
    pids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    kw = dict(input_ids=iids, prompt_input_ids=pids)
    if params.get("temperature"): kw["temperature"]=float(params["temperature"]); kw["do_sample"]=True
    if params.get("max_new_tokens"): kw["max_new_tokens"]=int(float(params["max_new_tokens"]))
    with torch.no_grad(): gen = model.generate(**kw)
    return _to_wav(gen.cpu().numpy().squeeze().astype(np.float32), model.config.sampling_rate), model.config.sampling_rate

# -- 13. Chatterbox --
def _load_chatterbox():
    import sys, types, importlib.machinery
    # torchcodec 0.11 was built for CUDA 13.x; we have CUDA 12.8.
    # torchaudio 2.10 detects torchcodec via __spec__ — MagicMock breaks that check.
    # Use real types.ModuleType stubs so __spec__ is a proper ModuleSpec.
    for _tc in ["torchcodec", "torchcodec._C", "torchcodec.decoders",
                "torchcodec.decoders._core", "torchcodec.decoders.video_decoder",
                "torchcodec.encoders"]:
        if _tc not in sys.modules:
            _m = types.ModuleType(_tc)
            _is_pkg = not "." in _tc.split("torchcodec.")[-1] or _tc == "torchcodec"
            _m.__spec__ = importlib.machinery.ModuleSpec(
                _tc, loader=None, is_package=_is_pkg)
            _m.__path__ = []     # marks it as a package to submodule imports
            sys.modules[_tc] = _m
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=DEVICE)

def _synth_chatterbox(inst, text, params):
    kw = dict(exaggeration=float(params.get("exaggeration",0.65)), cfg_weight=float(params.get("cfg_weight",0.5)))
    seed = params.get("seed")
    if seed and int(float(seed)) != 0: kw["seed"] = int(float(seed))
    pid = params.get("audio_prompt_id")
    if pid:
        p = UPLOAD_DIR / f"{pid}.wav"
        if p.exists(): kw["audio_prompt_path"] = str(p)
    wav = inst.generate(text, **kw)
    # Avoid torchaudio.save() — it attempts to import torchcodec.encoders which
    # fails when the torchcodec stub is active (libnvrtc.so.13 missing).
    arr = wav.squeeze().cpu().numpy().astype(np.float32)
    return _to_wav(arr, inst.sr), inst.sr

# -- 14. Fish Speech --
def _load_fishspeech(model_id="fishaudio/fish-speech-1.5"):
    """Fish Speech — uses fish_speech.inference_engine.TTSInferenceEngine.
    PyPI package provides the engine; full model-loading code lives in the GitHub repo.
    Install (full): cd /tmp && git clone https://github.com/fishaudio/fish-speech
                    pip install -e /tmp/fish-speech/
    Weights auto-download from HuggingFace: fishaudio/fish-speech-1.5
    """
    import torch
    try:
        from fish_speech.inference_engine import TTSInferenceEngine  # noqa: F401
        from fish_speech.utils.schema import ServeTTSRequest          # noqa: F401
    except ImportError as e:
        raise ImportError(f"pip install fish-speech (or git clone the full repo): {e}") from e
    try:
        from fish_speech.models.vqgan.inference import load_model as _load_codec
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue as _llm
    except ImportError:
        raise ImportError(
            "Fish Speech model-loading code missing.\n"
            "The PyPI package only ships the inference engine framework.\n"
            "Full install: cd /tmp && git clone https://github.com/fishaudio/fish-speech\n"
            "              pip install -e /tmp/fish-speech/"
        )
    from huggingface_hub import snapshot_download as _dl
    from pathlib import Path as _P
    model_dir = _P(_dl(model_id, ignore_patterns=["*.md", "*.txt", "*.gitignore"]))
    codec_pth = next((p for pat in ["firefly-gan*.pth", "*.pth"]
                      for p in model_dir.glob(pat)), None)
    if codec_pth is None:
        raise FileNotFoundError(f"No .pth codec file in {model_dir}")
    _precision = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    decoder  = _load_codec(config_name="firefly_gan_vq", checkpoint_path=str(codec_pth), device=DEVICE)
    llama_q  = _llm(checkpoint_path=str(model_dir), device=DEVICE,
                    precision=_precision, compile=False)
    return TTSInferenceEngine(llama_queue=llama_q, decoder_model=decoder,
                               precision=torch.float32, compile=False)

def _synth_fishspeech(inst, text, params):
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    refs   = []
    ref_id = params.get("audio_prompt_id", "")
    if ref_id:
        p = UPLOAD_DIR / f"{ref_id}.wav"
        if p.exists():
            refs = [ServeReferenceAudio(audio=p.read_bytes(), text="")]
    req   = ServeTTSRequest(text=text, references=refs)
    final = None
    for result in inst.inference(req):
        if result.code == "final":
            final = result
        elif result.code == "error" and result.error:
            raise RuntimeError(f"Fish Speech error: {result.error}")
    if final is None or final.audio is None:
        raise RuntimeError("Fish Speech: no audio generated")
    sr, audio_np = final.audio
    return _to_wav(audio_np.astype(np.float32), int(sr)), int(sr)

# -- 15. Sesame CSM 1B --
def _load_csm():
    """Sesame Conversational Speech Model (1B).
    Install: pip install git+https://github.com/SesameAILabs/csm
    NOTE: Model is gated on HuggingFace — run: huggingface-cli login
    """
    from generator import load_csm_1b
    return load_csm_1b(device=DEVICE)

def _synth_csm(inst, text, params):
    speaker = int(float(params.get("speaker_id", 0)))
    max_ms   = int(float(params.get("max_audio_length_ms", 30000)))
    audio    = inst.generate(text=text, speaker=speaker, context=[], max_audio_length_ms=max_ms)
    sr       = inst.sample_rate
    arr      = audio.cpu().numpy().flatten().astype(np.float32)
    return _to_wav(arr, sr), sr

# -- 16. Qwen3-TTS --
def _load_qwen3tts(model_id="Qwen/Qwen3-TTS"):
    """Qwen3-TTS — Alibaba Qwen3-based TTS via transformers.
    Install: transformers already installed (dep of parler-tts).
    Model auto-downloads from HuggingFace on first load.
    Check https://huggingface.co/Qwen for latest model ID.
    NOTE: Qwen/Qwen3-TTS was not yet public at initial benchmarking (March 2026).
    """
    import torch
    from transformers import AutoProcessor, AutoModel
    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        mdl  = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=_dtype).to(DEVICE)
    except Exception as _e:
        _s = str(_e).lower()
        if "not found" in _s or "404" in _s or "repository" in _s or "does not exist" in _s:
            raise RuntimeError(
                f"Qwen3-TTS model '{model_id}' not found on HuggingFace.\n"
                "The model may not be public yet, or the ID has changed.\n"
                "Check https://huggingface.co/Qwen for the current model name.\n"
                f"Original error: {_e}"
            ) from _e
        raise
    return (mdl, proc)

def _synth_qwen3tts(inst, text, params):
    import torch
    model, proc = inst
    inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in proc(text=text, return_tensors="pt").items()}
    ref_id = params.get("audio_prompt_id", "")
    if ref_id and (UPLOAD_DIR / f"{ref_id}.wav").exists():
        inputs["reference_audio"] = str(UPLOAD_DIR / f"{ref_id}.wav")
    with torch.no_grad():
        output = model.generate(**inputs)
    if hasattr(output, "audio"):
        arr = output.audio.squeeze().cpu().numpy().astype(np.float32)
    elif isinstance(output, (list, tuple)):
        arr = np.array(output[0], dtype=np.float32).flatten()
    else:
        arr = output.cpu().numpy().flatten().astype(np.float32)
    fe  = getattr(proc, "feature_extractor", None)
    sr  = getattr(fe, "sampling_rate", None) or 22050
    return _to_wav(arr, sr), sr

# -- 17. Orpheus 3B --
def _load_orpheus(model_name="canopylabs/orpheus-3b-0.1-ft"):
    """Orpheus TTS 3B — LLaMA-3B-based TTS with emotion tags.
    Install: pip install orpheus-speech
    Emotion tags (embed in text): <laugh> <sigh> <chuckle> <gasp> <cough> <sniffle> <groan> <yawn>
    Voices: tara leah jess leo dan mia zac zoe
    NOTE: orpheus_tts/decoder.py ships with snac_device=\"cuda\" hardcoded.
          We patch it to \"cpu\" on install (see _remote_install_new_engines.sh).
    """
    _require_gpu("Orpheus 3B")   # vllm requires CUDA — refuses immediately on CPU
    from orpheus_tts import OrpheusModel
    try:
        return OrpheusModel(model_name=model_name)
    except Exception as _e:
        _s = str(_e).lower()
        if "device" in _s or "cuda" in _s or "vllm" in _s or "empty" in _s:
            raise RuntimeError(
                "Orpheus 3B requires a CUDA GPU — vllm cannot run on CPU.\n"
                "Add a GPU and ensure CUDA drivers are installed, then restart.\n"
                f"Original error: {_e}"
            ) from _e
        raise

def _synth_orpheus(inst, text, params):
    voice  = params.get("voice", "tara")
    chunks = list(inst.generate_speech(prompt=text, voice=voice))
    if not chunks:
        return _to_wav(np.zeros(24000, np.float32), 24000), 24000
    raw = b"".join(chunks)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return _to_wav(arr, 24000), 24000

# -- 18. NeuTTS Air --
def _load_neutts():
    """NeuTTS Air — placeholder. Edit _load_neutts() once the package is identified.

    Template:
        from neutts_package import NeuTTSAir
        return NeuTTSAir.from_pretrained("model-id", device="cpu")
    """
    raise NotImplementedError(
        "NeuTTS Air: package not yet configured.\n"
        "Edit _load_neutts() in tts_lab.py with the correct import after installing.\n"
        "See setup_tts_lab.sh step 18 for notes."
    )

def _synth_neutts(inst, text, params):
    raise NotImplementedError("NeuTTS Air: configure _load_neutts() first.")

# -- 19. IndexTTS-2 --
def _load_indextts(model_dir=None):
    """IndexTTS-2 — zero-shot voice cloning from IndexTeam/Bilibili.
    Install: pip install git+https://github.com/index-tts/IndexTTS
    Model auto-downloads to HF cache on first load.
    """
    from indextts.infer import IndexTTS
    md = model_dir or (str(INDEXTTS_DIR) if INDEXTTS_DIR.exists() else "IndexTeam/IndexTTS")
    model = IndexTTS(model_dir=md, device=DEVICE)
    model.load_model()
    return model

def _synth_indextts(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
    if not ref_path or not Path(ref_path).exists():
        raise RuntimeError(
            "IndexTTS-2 requires a reference WAV (voice prompt). "
            "Upload a 5-30s WAV clip first, then synthesise."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        inst.infer(audio_prompt=ref_path, text=text, output_path=tmp)
        wav = Path(tmp).read_bytes()
    finally:
        Path(tmp).unlink(missing_ok=True)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        sr = wf.getframerate()
    return wav, sr

# -- 20. Zonos v0.1 --
def _load_zonos(variant="transformer"):
    """Zonos v0.1 — Hybrid/Transformer TTS from Zyphra.
    Install: pip install git+https://github.com/Zyphra/Zonos  +  pip install phonemizer
    System:  sudo apt install espeak-ng  (already installed by setup_tts_lab.sh)
    """
    from zonos.model import Zonos
    return Zonos.from_pretrained(f"Zyphra/Zonos-v0.1-{variant}", device=DEVICE)

def _synth_zonos(inst, text, params):
    import torch
    from zonos.conditioning import make_cond_dict

    speaker = None
    ref_id  = params.get("audio_prompt_id", "")
    if ref_id:
        ref_path = UPLOAD_DIR / f"{ref_id}.wav"
        if ref_path.exists():
            try:
                import torchaudio
                wav_t, sr_r = torchaudio.load(str(ref_path))
                speaker = inst.make_speaker_embedding(wav_t, sr_r)
            except Exception:
                pass

    emotion = [
        float(params.get("happiness", 0.3)),
        float(params.get("sadness",   0.05)),
        float(params.get("disgust",   0.05)),
        float(params.get("fear",      0.05)),
        float(params.get("surprise",  0.1)),
        float(params.get("anger",     0.05)),
        float(params.get("other",     0.2)),
        float(params.get("neutral",   0.2)),
    ]
    cond = make_cond_dict(
        text=text,
        language=params.get("language", "en-us"),
        speaker=speaker,
        speaking_rate=float(params.get("speaking_rate", 13.0)),
        emotion=emotion,
    )
    conditioning = inst.prepare_conditioning(cond)
    with torch.no_grad():
        codes = inst.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=int(float(params.get("max_new_tokens", 1024))),
            cfg_scale=float(params.get("cfg_scale", 2.0)),
            disable_torch_compile=True,
        )
    # Decode via autoencoder (Zonos 0.1.0 API)
    wav_t    = inst.autoencoder.decode(codes)
    sr       = inst.autoencoder.sampling_rate
    arr      = wav_t[0].squeeze().cpu().numpy().astype(np.float32)
    return _to_wav(arr, sr), sr

# -- 21. OpenVoice v2 --
def _load_openvoice():
    """OpenVoice v2 — MeloTTS base + tone-color conversion. MyShell AI.
    Install: pip install git+https://github.com/myshell-ai/OpenVoice
    Checkpoints (v1 or v2 layout):
      sudo ln -sfn /opt/models/huggingface/hub/models--myshell-ai--OpenVoice/snapshots/<hash>/checkpoints \\
                   /opt/models/openvoice_v2
    Supports both v1 (base_speakers/EN/) and v2 (base_speakers/ses/) layouts.
    """
    import sys
    # wavmark is an optional audio-watermarking dep that openvoice imports at module level.
    # It is not needed for inference; stub it so CPU-only installs don't abort.
    if "wavmark" not in sys.modules:
        from unittest.mock import MagicMock as _MM
        sys.modules["wavmark"] = _MM()
    import torch
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS

    ckpt_dir = OPENVOICE_MODELS_DIR / "converter"
    if not (ckpt_dir / "config.json").exists():
        raise FileNotFoundError(
            f"OpenVoice checkpoints missing at {OPENVOICE_MODELS_DIR}.\n"
            f"Run: sudo ln -sfn <hf_snapshot>/checkpoints /opt/models/openvoice_v2"
        )
    converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=DEVICE)
    converter.load_ckpt(str(ckpt_dir / "checkpoint.pth"))
    base_tts  = MeloTTS(language="EN", device=DEVICE)

    base_se = {}
    ses_dir = OPENVOICE_MODELS_DIR / "base_speakers" / "ses"  # v2 layout
    en_dir  = OPENVOICE_MODELS_DIR / "base_speakers" / "EN"   # v1 layout
    if ses_dir.exists():
        for p in ses_dir.glob("*.pth"):
            t = torch.load(str(p), map_location=DEVICE, weights_only=False)
            base_se[p.stem] = t.to(DEVICE) if hasattr(t, "to") else t
    elif en_dir.exists():
        for fname, key in [("en_default_se.pth", "en_default"), ("en_style_se.pth", "en_style")]:
            p = en_dir / fname
            if p.exists():
                t = torch.load(str(p), map_location=DEVICE, weights_only=False)
                base_se[key] = t.to(DEVICE) if hasattr(t, "to") else t
    return {"converter": converter, "base_tts": base_tts, "base_se": base_se}

def _synth_openvoice(inst, text, params):
    import torch
    converter = inst["converter"]
    base_tts  = inst["base_tts"]
    base_se   = inst["base_se"]

    spk_key = params.get("speaker", "EN-US")
    sp_ids  = dict(base_tts.hps.data.spk2id)
    sp_id   = sp_ids.get(spk_key) or sp_ids.get("EN-US") or list(sp_ids.values())[0]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: src_tmp = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: out_tmp = f.name
    try:
        base_tts.tts_to_file(text, sp_id, src_tmp, speed=float(params.get("speed", 0.85)))
        se_key = spk_key.lower().replace("-", "_")
        src_se = base_se.get(se_key) or base_se.get("en_us") or (list(base_se.values())[0] if base_se else None)
        ref_id = params.get("audio_prompt_id", "")
        ref_path = UPLOAD_DIR / f"{ref_id}.wav" if ref_id else None
        target_se = src_se   # default: identity conversion (same speaker)
        if ref_path and ref_path.exists():
            try:
                from openvoice import se_extractor
                _se, _ = se_extractor.get_se(str(ref_path), converter, vad=True)
                if _se is not None and _se.numel() > 0:
                    target_se = _se
            except Exception:
                pass   # VAD found no speech — fall back to identity
        if src_se is not None and target_se is not None:
            converter.convert(
                audio_src_path=src_tmp, src_se=src_se, tgt_se=target_se,
                output_path=out_tmp, tau=float(params.get("tau", 0.3)),
            )
            wav = Path(out_tmp).read_bytes()
        else:
            wav = Path(src_tmp).read_bytes()
        with wave.open(io.BytesIO(wav), "rb") as wf:
            sr = wf.getframerate()
        return wav, sr
    finally:
        Path(src_tmp).unlink(missing_ok=True)
        Path(out_tmp).unlink(missing_ok=True)

# ============================================================
# REGISTRY + AVAILABILITY + DISPATCH
# ============================================================
LOADERS  = {"piper":_load_piper,"kokoro":_load_kokoro,"melo":_load_melo,
            "chattts":_load_chattts,"outetts":_load_outetts,"bark":_load_bark,
            "styletts2":_load_styletts2,"f5tts":_load_f5tts,"dia":_load_dia,
            "xtts":_load_xtts,"cosyvoice":_load_cosyvoice,"parler":_load_parler,
            "chatterbox":_load_chatterbox,
            "fishspeech":_load_fishspeech,"csm":_load_csm,"qwen3tts":_load_qwen3tts,
            "orpheus":_load_orpheus,"neutts":_load_neutts,"indextts":_load_indextts,
            "zonos":_load_zonos,"openvoice":_load_openvoice}
SYNTHERS = {"piper":_synth_piper,"kokoro":_synth_kokoro,"melo":_synth_melo,
            "chattts":_synth_chattts,"outetts":_synth_outetts,"bark":_synth_bark,
            "styletts2":_synth_styletts2,"f5tts":_synth_f5tts,"dia":_synth_dia,
            "xtts":_synth_xtts,"cosyvoice":_synth_cosyvoice,"parler":_synth_parler,
            "chatterbox":_synth_chatterbox,
            "fishspeech":_synth_fishspeech,"csm":_synth_csm,"qwen3tts":_synth_qwen3tts,
            "orpheus":_synth_orpheus,"neutts":_synth_neutts,"indextts":_synth_indextts,
            "zonos":_synth_zonos,"openvoice":_synth_openvoice}

_import_cache: Dict[str, Tuple[bool, str]] = {}
_import_cache_lock = threading.Lock()

def _available(name: str) -> Tuple[bool, str]:
    """Return (available, reason) for *name*, running the check in a thread-pool
    worker so the event-loop thread is never blocked by heavy C-extension imports."""
    with _import_cache_lock:
        if name in _import_cache:
            return _import_cache[name]
    # Not cached yet — compute synchronously (called from executor thread at startup,
    # or on first synthesise click before the sweep has reached this engine).
    result = _check_available(name)
    with _import_cache_lock:
        _import_cache[name] = result
    return result

def _check_available(name: str) -> Tuple[bool, str]:
    """Synchronous availability probe — uses find_spec + fs checks only.
    No exec() / actual imports so the sweep is thread-safe and instant."""
    import importlib.util as ilu
    pkg_map = {
        "piper":"piper","kokoro":"kokoro_onnx","melo":"melo",
        "chattts":"ChatTTS","outetts":"outetts","bark":"bark","styletts2":"styletts2","f5tts":"f5_tts",
        "dia":"dia","xtts":"TTS","cosyvoice":None,"parler":"parler_tts","chatterbox":"chatterbox",
        "fishspeech":"fish_speech","csm":None,"qwen3tts":"transformers",
        "orpheus":"orpheus_tts","neutts":None,"indextts":"indextts",
        "zonos":"zonos","openvoice":"openvoice",
    }
    # ── 1. Quick package-present check (find_spec — no import, no C-ext init) ──
    pkg = pkg_map.get(name)
    if pkg and not ilu.find_spec(pkg):
        return False, f"pip install {pkg} needed"
    # ── 2. GPU-required engines — check CUDA before doing anything else ────────
    _GPU_REQUIRED = {"outetts", "bark", "orpheus"}
    if name in _GPU_REQUIRED:
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA GPU required — not available on this machine"
        except ImportError:
            pass  # torch not installed yet; loader will handle it
    # ── 3. Engine-specific file / directory checks ─────────────────────────────
    if name == "piper":
        if not _piper_voices(): return False, "No .onnx voice found in models/"
    elif name == "kokoro":
        if not (MODELS_DIR/"kokoro-v1.0.onnx").exists(): return False, "kokoro-v1.0.onnx missing"
    elif name == "cosyvoice":
        if not COSYVOICE_DIR.exists(): return False, "git clone FunAudioLLM/CosyVoice /opt/CosyVoice"
        if not (COSYVOICE_DIR/"pretrained_models"/"CosyVoice2-0.5B").exists():
            return False, "CosyVoice2-0.5B model not downloaded"
        if not ilu.find_spec("hyperpyyaml"):
            return False, "pip install hyperpyyaml  (CosyVoice2 dependency)"
    elif name == "fishspeech":
        # find_spec('fish_speech') passes for the PyPI package, but models.vqgan
        # only exists in the full git-clone install.
        if not ilu.find_spec("fish_speech.models.vqgan"):
            return False, (
                "Full repo needed: "
                "cd /tmp && git clone https://github.com/fishaudio/fish-speech "
                "&& pip install -e /tmp/fish-speech/"
            )
    elif name == "neutts":
        return False, "NeuTTS Air: not configured — edit _load_neutts() in tts_lab.py"
    elif name == "openvoice":
        if not (OPENVOICE_MODELS_DIR/"converter"/"config.json").exists():
            return False, f"Checkpoints missing at {OPENVOICE_MODELS_DIR}"
    elif name == "csm":
        # Sesame CSM is from GitHub (generator.py), NOT the PyPI 'csm' package
        if not ilu.find_spec("generator"):
            return False, "Sesame CSM needs: pip install git+https://github.com/SesameAILabs/csm + huggingface-cli login"
    elif name == "indextts":
        if not ilu.find_spec("indextts"):
            return False, "pip install git+https://github.com/index-tts/IndexTTS"
    return True, ""

def _do_synth(name, text, params):
    st = _state[name]
    with st["lock"]:
        if name == "piper":
            wanted = params.get("voice", "en_US-ryan-high")
            if st["instance"] and st.get("loaded_voice") != wanted:
                _safe_del(st["instance"]); st["instance"] = None
        if name == "outetts":
            wanted = params.get("model_path", "OuteAI/OuteTTS-0.3-500M")
            if st["instance"] and st.get("loaded_model") != wanted:
                _safe_del(st["instance"]); st["instance"] = None
        if name == "parler":
            wanted = params.get("model_id", "parler-tts/parler-tts-mini-v1")
            if st["instance"] and st.get("loaded_model") != wanted:
                _safe_del(st["instance"]); st["instance"] = None
        if name == "zonos":
            wanted = params.get("variant", "transformer")
            if st["instance"] and st.get("loaded_model") != wanted:
                _safe_del(st["instance"]); st["instance"] = None
        if st["instance"] is None:
            ok, reason = _available(name)
            if not ok: raise RuntimeError(f"Not available: {reason}")
            if MODEL_INFO[name]["heavy"]: _evict_heavy(keep=name)
            st["status"] = "loading"; t0 = time.perf_counter()
            try:
                if name == "piper":
                    st["instance"] = _load_piper(params.get("voice", "en_US-ryan-high"))
                elif name == "outetts":
                    st["instance"] = _load_outetts(params.get("model_path", "OuteAI/OuteTTS-0.3-500M"))
                elif name == "parler":
                    st["instance"] = _load_parler(params.get("model_id", "parler-tts/parler-tts-mini-v1"))
                elif name == "zonos":
                    st["instance"] = _load_zonos(params.get("variant", "transformer"))
                else:
                    st["instance"] = LOADERS[name]()
                st["load_time_s"] = round(time.perf_counter()-t0, 2)
                st["status"] = "loaded"; st["error"] = ""
                if name == "piper":   st["loaded_voice"] = params.get("voice", "en_US-ryan-high")
                if name == "outetts": st["loaded_model"] = params.get("model_path", "OuteAI/OuteTTS-0.3-500M")
                if name == "parler":  st["loaded_model"] = params.get("model_id", "parler-tts/parler-tts-mini-v1")
                if name == "zonos":   st["loaded_model"] = params.get("variant", "transformer")
            except Exception as e:
                st["status"] = "error"; st["error"] = str(e); raise
    t0 = time.perf_counter()
    wav, sr = SYNTHERS[name](st["instance"], text, params)
    synth_s = time.perf_counter() - t0
    dur = _wav_dur(wav)
    return {"audio_b64": base64.b64encode(wav).decode(), "sample_rate": sr,
            "synth_time_ms": int(synth_s*1000), "audio_dur_ms": int(dur*1000),
            "rtf": round(synth_s/dur, 4) if dur > 0 else 0,
            "load_time_s": st["load_time_s"]}

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Arthur TTS Lab")

class SynthReq(BaseModel):
    text:   str
    params: dict = {}

# Pre-warm the availability cache in a background thread so the event loop is
# never blocked by heavy C-extension imports (outetts→vllm, fish_speech→vllm, etc.)
_sweep_done = threading.Event()

def _sweep_availability():
    """Run once at startup: probe every engine and populate _import_cache."""
    for n in MODEL_ORDER:
        try:
            _available(n)
        except Exception:
            pass
    _sweep_done.set()

@app.on_event("startup")
async def _startup():
    t = threading.Thread(target=_sweep_availability, name="avail-sweep", daemon=True)
    t.start()

@app.get("/", response_class=HTMLResponse)
async def index():
    # Serve the page immediately; badges update via /status polling
    return HTMLResponse(_build_page())

@app.get("/status")
async def status():
    models = {}
    sweep_running = not _sweep_done.is_set()
    for n in MODEL_ORDER:
        ok, reason = _available(n); st = _state[n]
        models[n] = {**MODEL_INFO[n], "available":ok, "reason":reason,
                     "status":st["status"], "load_time_s":st["load_time_s"], "error":st["error"]}
        if sweep_running and n not in _import_cache:
            models[n]["available"] = False
            models[n]["reason"]    = "checking..."
    tot, used, free = _ram_mb()
    gpu_info = {}
    if DEVICE == "cuda":
        try:
            import torch
            gp = torch.cuda.memory_stats(0)
            gpu_info = {
                "name":       DEVICE_NAME,
                "vram_total": VRAM_TOTAL_MB,
                "vram_used":  int(torch.cuda.memory_allocated(0) / 1048576),
                "vram_free":  int((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1048576),
            }
        except Exception:
            gpu_info = {"name": DEVICE_NAME, "vram_total": VRAM_TOTAL_MB}
    return JSONResponse({"models":models,
                         "system":{"total":tot,"used":used,"free":free},
                         "gpu": gpu_info,
                         "device": DEVICE})

@app.get("/voices/{model}")
async def voices(model):
    vmap = {
        "piper":     _piper_voices() or ["en_US-ryan-high"],
        "kokoro":    ALL_KOKORO_VOICES,
        "melo":      ["EN-Default","EN-US","EN-BR","EN-AU","EN_INDIA"],
        "outetts":   [v for v,_ in OUTETTS_SPEAKERS],
        "bark":      [v for v,_ in BARK_PRESETS],
        "xtts":      ALL_XTTS_SPEAKERS,
        "cosyvoice": ["English Female","English Male"],
    }
    return JSONResponse({"voices": vmap.get(model, [])})

# Per-engine synthesis timeout (seconds). With RTX 5060 Ti all engines are fast.
_SYNTH_TIMEOUT = {
    "orpheus":    240,   # LLM-based (vllm) — 3B model, still needs headroom
    "dia":        180,   # 1.6B autoregressive, borderline real-time on GPU
    "bark":       180,   # transformer TTS, full-size models on GPU
    "qwen3tts":   180,
    "outetts":    120,
    "f5tts":      120,
    "chattts":     90,
}
_DEFAULT_SYNTH_TIMEOUT = 300   # 5 min hard cap for all other engines

@app.post("/synthesize/{model}")
async def synthesize(model, req: SynthReq):
    if model not in MODEL_ORDER: return JSONResponse({"error":f"Unknown: {model}"}, status_code=400)
    if not req.text.strip(): return JSONResponse({"error":"Empty text"}, status_code=400)
    timeout = _SYNTH_TIMEOUT.get(model, _DEFAULT_SYNTH_TIMEOUT)
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _do_synth, model, req.text, req.params),
            timeout=float(timeout)
        )
        return JSONResponse(result)
    except asyncio.TimeoutError:
        return JSONResponse({
            "error": f"⏱ Synthesis timeout after {timeout}s — '{model}' is too slow on CPU. "
                     f"This engine requires a GPU. Add a GPU and restart the server."
        }, status_code=408)
    except Exception as e:
        return JSONResponse({"error":str(e),"trace":traceback.format_exc(limit=4)}, status_code=500)

@app.delete("/models/{model}")
async def unload_model(model):
    st = _state.get(model)
    if st and st["instance"] is not None:
        _safe_del(st["instance"]); st["instance"] = None; st["status"] = "unloaded"
    return {"unloaded": model}

@app.post("/refresh")
async def refresh_availability():
    # Clear the cache and re-sweep in background
    with _import_cache_lock:
        _import_cache.clear()
    _sweep_done.clear()
    t = threading.Thread(target=_sweep_availability, name="avail-resweep", daemon=True)
    t.start()
    return JSONResponse({"refreshed": True, "models": list(MODEL_ORDER),
                         "note": "sweep running in background — poll /status in ~60 s"})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    dest = UPLOAD_DIR / f"{uid}.wav"
    with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
    return JSONResponse({"id":uid, "filename":file.filename, "size":dest.stat().st_size})

# ============================================================
# HTML BUILDERS
# ============================================================
def _stars(n): return "&#9733;"*n + "&#9734;"*(5-n)

def _sel(param, opts, cur=None):
    o = "".join(f'<option value="{v}"{" selected" if v==cur else ""}>{l}</option>' for v,l in opts)
    return f'<select class="form-select form-select-sm bg-dark text-light border-secondary" data-param="{param}">{o}</select>'

def _rng(param, lo, hi, step, val, note=""):
    return (f'<input type="range" class="form-range" data-param="{param}" '
            f'min="{lo}" max="{hi}" step="{step}" value="{val}" oninput="rangeUpdate(this)">'
            +(f'<small class="text-muted">{note}</small>' if note else ""))

def _grp(label, ctrl): return f'<div class="param-group"><label>{label}</label>{ctrl}</div>'
def _row(*cols): return f'<div class="param-row">{"".join(cols)}</div>'

def _upload_widget(prompt_file_id, prompt_status_id, prompt_hidden_id, label="Reference audio WAV (5-30s)"):
    return (f'<div class="param-group" style="max-width:100%;width:100%"><label>{label}</label>'
            f'<div class="d-flex gap-2 align-items-center">'
            f'<input type="file" id="{prompt_file_id}" class="form-control form-control-sm bg-dark text-light border-secondary" accept="audio/wav,audio/*" style="max-width:320px">'
            f'<button class="btn btn-sm btn-outline-info" onclick="uploadPrompt(\'{prompt_file_id}\',\'{prompt_status_id}\',\'{prompt_hidden_id}\')">Upload</button>'
            f'<span id="{prompt_status_id}" class="text-muted small"></span>'
            f'<input type="hidden" data-param="audio_prompt_id" id="{prompt_hidden_id}">'
            f'</div></div>')

def _build_params(name):

    if name == "piper":
        voices = _piper_voices() or ["en_US-ryan-high"]
        vopts = "\n".join(
            f'<option value="{v}">{"[GB]" if "GB" in v else "[US]"} {v}{"  default" if "ryan-high" in v else ""}</option>'
            for v in voices)
        sel = f'<select class="form-select form-select-sm bg-dark text-light border-secondary" data-param="voice">{vopts}</select>'
        return (_row(_grp("Voice (6 downloaded)", sel),
                     _grp('Speed <span class="range-val">1.0</span>', _rng("speed","0.5","2.0","0.05","1.0")))
               +_row(_grp('Length scale <span class="range-val">1.0</span>', _rng("length_scale","0.5","2.0","0.05","1.0","higher=slower")),
                     _grp('Noise scale <span class="range-val">0.667</span>', _rng("noise_scale","0.1","1.5","0.05","0.667","voice variation")),
                     _grp('Noise-W <span class="range-val">0.8</span>', _rng("noise_w","0.1","1.5","0.05","0.8","duration variation"))))

    if name == "kokoro":
        grps = [
            ("British Male (Arthur pick)", [v for v in ALL_KOKORO_VOICES if v.startswith("bm_")]),
            ("British Female",             [v for v in ALL_KOKORO_VOICES if v.startswith("bf_")]),
            ("American Male",              [v for v in ALL_KOKORO_VOICES if v.startswith("am_")]),
            ("American Female",            [v for v in ALL_KOKORO_VOICES if v.startswith("af_")]),
            ("Spanish",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("ef_","em_"))]),
            ("French",                     [v for v in ALL_KOKORO_VOICES if v.startswith("ff_")]),
            ("Hindi",                      [v for v in ALL_KOKORO_VOICES if v.startswith(("hf_","hm_"))]),
            ("Italian",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("if_","im_"))]),
            ("Japanese",                   [v for v in ALL_KOKORO_VOICES if v.startswith(("jf_","jm_"))]),
            ("Portuguese",                 [v for v in ALL_KOKORO_VOICES if v.startswith(("pf_","pm_"))]),
            ("Chinese",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("zf_","zm_"))]),
        ]
        opts = "".join(
            f'<optgroup label="{gl}">{"".join(f"""<option value="{v}"{" selected" if v=="bm_lewis" else ""}>{v}{"  (Arthur pick)" if v=="bm_lewis" else ""}</option>""" for v in vl)}</optgroup>'
            for gl,vl in grps if vl)
        sel = f'<select class="form-select form-select-sm bg-dark text-light border-secondary" data-param="voice">{opts}</select>'
        return _row(_grp("Voice (54, grouped by language)", sel),
                    _grp('Speed <span class="range-val">0.85</span>', _rng("speed","0.5","1.5","0.05","0.85")))

    if name == "melo":
        sp = [("EN-Default","EN-Default"),("EN-US","EN-US American"),
              ("EN-BR","EN-BR British"),("EN-AU","EN-AU Australian"),("EN_INDIA","EN_INDIA Indian")]
        return _row(_grp("Speaker (5 accents)", _sel("speaker",sp,"EN-US")),
                    _grp('Speed <span class="range-val">0.85</span>', _rng("speed","0.5","1.5","0.05","0.85")))

    if name == "chattts":
        return (_row(
            _grp("Prompt speed token", _sel("prompt", CHATTTS_SPEEDS, "[speed_5]")),
            _grp('Temperature <span class="range-val">0.3</span>', _rng("temperature","0.1","1.5","0.05","0.3")),
            _grp('Top-P <span class="range-val">0.7</span>', _rng("top_p","0.1","1.0","0.01","0.7")),
            _grp('Top-K <span class="range-val">20</span>', _rng("top_k","1","100","1","20")),
        )+_row(
            _grp('Repetition penalty <span class="range-val">1.05</span>', _rng("repetition_penalty","1.0","2.0","0.01","1.05")),
            _grp('Max new tokens <span class="range-val">512</span>', _rng("max_new_token","128","2048","64","512")),
            _grp('Seed <span class="range-val">0</span>', _rng("seed","0","9999","1","0","0=random")),
        )+_row(
            _grp("Skip refine text", _sel("skip_refine_text", [("true","true"),("false","false")], "true")),
        )+f'<div class="param-row">{_upload_widget("ct-file","ct-status","ct-prompt-id","Reference WAV (optional — derive ChatTTS speaker embedding)")}</div>')

    if name == "outetts":
        vc = ('<textarea class="form-control form-control-sm bg-dark text-light border-secondary" data-param="voice_characteristics" rows="3" placeholder="Optional character description, e.g. elderly man, warm, raspy, hesitant"></textarea>')
        transcript = ('<input type="text" class="form-control form-control-sm bg-dark text-light border-secondary" data-param="transcript" placeholder="Optional transcript of uploaded reference WAV">')
        return (_row(
            _grp("Model", _sel("model_path", OUTETTS_MODELS, "OuteAI/OuteTTS-0.3-500M")),
            _grp("Default speaker", _sel("speaker", OUTETTS_SPEAKERS, "en-female-1-neutral")),
        )+_row(
            _grp('Temperature <span class="range-val">0.4</span>', _rng("temperature","0.1","1.5","0.05","0.4")),
            _grp('Repetition penalty <span class="range-val">1.1</span>', _rng("repetition_penalty","1.0","2.0","0.01","1.1")),
            _grp('Top-K <span class="range-val">40</span>', _rng("top_k","1","100","1","40")),
            _grp('Top-P <span class="range-val">0.9</span>', _rng("top_p","0.1","1.0","0.01","0.9")),
            _grp('Min-P <span class="range-val">0.05</span>', _rng("min_p","0.0","0.5","0.01","0.05")),
        )+_row(_grp('Max tokens <span class="range-val">0 (auto)</span>', _rng("max_length","0","4096","128","0","0=auto from text length (~30 tok/word)")))
        +f'<div class="param-row">{_upload_widget("ot-file","ot-status","ot-prompt-id","Reference WAV (optional — create OuteTTS speaker)")}</div>'
        +'<div class="param-row">{_grp("Reference transcript", transcript)}</div>'
        +f'<div class="param-row" style="flex-direction:column"><div class="param-group" style="max-width:100%;width:100%"><label>Voice characteristics</label>{vc}</div></div>')

    if name == "bark":
        preset_opts = [(v,l) for v,l in BARK_PRESETS]
        token_hint = ('<div class="alert alert-info py-2 small mt-2 mb-0">'
                      '<strong>Emotion tokens you can embed in text:</strong><br>'
                      '<code>[laughs]</code> &nbsp; <code>[sighs]</code> &nbsp; <code>[clears throat]</code> &nbsp; '
                      '<code>[hesitantly]</code> &nbsp; <code>[gasps]</code> &nbsp; <code>[long pause]</code> &nbsp; '
                      '<code>[nervously]</code> &nbsp; <code>[quietly]</code> &nbsp; <code>[MAN]</code> &nbsp; <code>[WOMAN]</code><br>'
                      '<em>Example: "Hello? [sighs] Oh my goodness, just a moment dear. [clears throat]"</em>'
                      '</div>')
        bark_presets_html = "".join(
            f'<button class="btn btn-sm btn-outline-secondary mb-1" onclick="setPreset(this.dataset.txt)" '
            f'data-txt="{t.replace(chr(34),chr(39))}" style="font-size:.72rem">{l}</button>'
            for t,l in BARK_ARTHUR_PRESETS)
        cpu_warn = ('<div class="alert alert-danger py-2 small mt-2 mb-0">'
                    '⚠ <strong>Bark is extremely slow on CPU</strong> — benchmarked at timeout (&gt;480s) for 40 words. '
                    'Will be killed after 180s. <strong>Requires GPU for practical use.</strong></div>')
        return (_row(_grp("Voice preset", _sel("voice_preset", preset_opts, "v2/en_speaker_6")))
               +f'<div class="mt-2">{bark_presets_html}</div>'
               +token_hint)

    if name == "styletts2":
        return (_row(
            _grp('Alpha (style weight) <span class="range-val">0.3</span>', _rng("alpha","0.0","1.0","0.05","0.3","0=copy ref style exactly")),
            _grp('Beta (prosody weight) <span class="range-val">0.7</span>', _rng("beta","0.0","1.0","0.05","0.7","0=copy ref prosody")),
            _grp('Diffusion steps <span class="range-val">5</span>', _rng("diffusion_steps","3","15","1","5","more=better+slower")),
            _grp('Embedding scale <span class="range-val">1.0</span>', _rng("embedding_scale","0.5","3.0","0.1","1.0")),
        )+f'<div class="param-row">{_upload_widget("sty-file","sty-status","sty-prompt-id","Style reference WAV (optional — sets voice timbre)")}</div>'
        +'<p class="text-muted small mt-1">Without reference: uses built-in neutral voice. With reference: clones timbre/style.</p>')

    if name == "f5tts":
        ref_box = ('<div class="param-group" style="max-width:100%;width:100%">'
                   '<label>Reference text <small class="text-muted">(what the speaker says in the reference WAV)</small></label>'
                   '<input type="text" class="form-control form-control-sm bg-dark text-light border-secondary" '
                   'data-param="ref_text" placeholder="Exact words spoken in the reference WAV clip...">'
                   '</div>')
        return (f'<div class="param-row">{_upload_widget("f5-file","f5-status","f5-prompt-id","Reference WAV (REQUIRED — 5-15s of the target voice)")}</div>'
               +f'<div class="param-row">{ref_box}</div>'
               +_row(_grp('Speed <span class="range-val">1.0</span>', _rng("speed","0.5","2.0","0.1","1.0")),
                     _grp('NFE steps <span class="range-val">32</span>', _rng("nfe_step","8","64","4","32","more=better quality+slower")))
               +'<p class="text-muted small mt-1">F5-TTS REQUIRES a reference WAV — upload a 5-15s clip of any voice you want to clone.</p>')

    if name == "dia":
        tags_hint = ('<div class="alert alert-info py-2 small mt-2 mb-0">'
                     '<strong>Dia text format:</strong><br>'
                     '<code>[S1]</code> and <code>[S2]</code> = speaker turns &nbsp;|&nbsp; '
                     '<code>[laughs]</code> <code>[sighs]</code> <code>[coughs]</code> <code>[groans]</code> '
                     '<code>[gasps]</code> <code>[sobs]</code> <code>[clears throat]</code><br>'
                     '<em>Auto-prefixes [S1] if no speaker tag found. Upload a voice WAV to clone a speaker.</em>'
                     '</div>')
        return (tags_hint
               +_row(_grp('CFG scale <span class="range-val">3.0</span>', _rng("cfg_scale","1.0","5.0","0.1","3.0","guidance strength")),
                     _grp('Temperature <span class="range-val">1.2</span>', _rng("temperature","0.5","2.0","0.1","1.2")),
                     _grp('Top-P <span class="range-val">0.95</span>', _rng("top_p","0.5","1.0","0.01","0.95")))
               +_row(_grp('Max tokens <span class="range-val">auto</span>', _rng("max_tokens","128","1024","64","0","0=auto from text length")))
               +f'<div class="param-row">{_upload_widget("dia-file","dia-status","dia-prompt-id","Voice reference WAV (optional — speaker cloning via audio_prompt_path)")}</div>')

    if name == "xtts":
        sp_opts = [(s, s+("  [Arthur pick]" if s=="Torcull Diarmuid" else "")) for s in ALL_XTTS_SPEAKERS]
        lang_opts = [(k,f"{k}  {v}") for k,v in XTTS_LANGUAGES.items()]
        return ('<div class="alert alert-warning py-2 small mb-2">3.2 GB RAM — evicts other heavy models.</div>'
               +_row(_grp("Speaker (58 total)", _sel("speaker",sp_opts,"Torcull Diarmuid")),
                     _grp("Language (17 total)", _sel("language",lang_opts,"en")))
               +_row(_grp('Temperature <span class="range-val">0.3</span>', _rng("temperature","0.01","1.0","0.01","0.3","lower=more stable"))))

    if name == "cosyvoice":
        sp = [("English Female","English Female"),("English Male","English Male")]
        return ('<div class="alert alert-secondary py-2 small mb-2">SFT mode: English Female/Male pre-trained speakers.</div>'
               +_row(_grp("Speaker", _sel("speaker",sp))))

    if name == "parler":
        presets = [
            "An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly.",
            "A tired old man with a Southern American accent speaks very slowly, stumbling over words.",
            "An elderly gentleman with a British accent speaks politely and hesitantly.",
            "A friendly 78-year-old man speaks in a clear American accent, with long natural pauses.",
        ]
        preset_btns = "".join(
            f'<button class="btn btn-sm btn-outline-secondary mb-1" '
            f'onclick="document.querySelector(\'#tab-parler [data-param=description]\').value=this.dataset.txt" '
            f'data-txt="{p}" style="font-size:.72rem">{p[:42]}...</button>' for p in presets)
        ta = (f'<textarea class="form-control form-control-sm bg-dark text-light border-secondary"'
              f' data-param="description" rows="3">{presets[0]}</textarea>')
        return (_row(_grp("Model", _sel("model_id", PARLER_MODELS, "parler-tts/parler-tts-mini-v1")))
               +f'<div class="param-row" style="flex-direction:column">'
                f'<div class="param-group" style="max-width:100%;width:100%"><label>Voice description</label>{ta}</div>'
                f'<div class="d-flex gap-1 flex-wrap mt-1">{preset_btns}</div></div>'
               +_row(_grp('Temperature <span class="range-val">1.0</span>', _rng("temperature","0.1","2.0","0.1","1.0")),
                     _grp('Max tokens <span class="range-val">1000</span>', _rng("max_new_tokens","200","2000","50","1000"))))

    if name == "chatterbox":
        return (_row(
            _grp('Exaggeration <span class="range-val">0.65</span>', _rng("exaggeration","0.0","1.0","0.05","0.65","0=flat, 1=expressive")),
            _grp('CFG weight <span class="range-val">0.5</span>', _rng("cfg_weight","0.1","1.0","0.05","0.5","lower=natural")),
            _grp('Seed <span class="range-val">0</span>', _rng("seed","0","9999","1","0","0=random")),
        )+f'<div class="param-row">{_upload_widget("cb-file","cb-status","cb-prompt-id","Voice cloning reference WAV (optional)")}</div>')

    # -- 14. Fish Speech --
    if name == "fishspeech":
        return (f'<div class="param-row">{_upload_widget("fs2-file","fs2-status","fs2-prompt-id","Reference WAV (optional — enables voice cloning)")}</div>'
               +_row(_grp('Speed <span class="range-val">1.0</span>', _rng("speed","0.5","2.0","0.1","1.0")))
               +'<p class="text-muted small mt-1">Without reference: synthesises in default voice. Upload a 5-30s WAV to clone any voice.</p>')

    # -- 15. Sesame CSM 1B --
    if name == "csm":
        hint = ('<div class="alert alert-info py-2 small mt-2 mb-0">'
                'Gated model — run <code>huggingface-cli login</code> before first load. '
                'Speaker 0 is male, 1-2 are alternatives.</div>')
        return (_row(_grp("Speaker", _sel("speaker_id", CSM_SPEAKERS, "0")),
                     _grp('Max audio (ms) <span class="range-val">30000</span>', _rng("max_audio_length_ms","5000","60000","1000","30000")))
               +hint)

    # -- 16. Qwen3-TTS --
    if name == "qwen3tts":
        return (f'<div class="param-row">{_upload_widget("q3-file","q3-status","q3-prompt-id","Reference WAV (optional — for voice conditioning)")}</div>'
               +'<div class="alert alert-warning py-2 small mt-2 mb-0">'
               +'⚠ <strong>Qwen/Qwen3-TTS may not be public on HuggingFace yet.</strong> '
               +'Load will fail with a 404 if the model is gated or the ID has changed. '
               +'Check <a href="https://huggingface.co/Qwen" target="_blank" class="alert-link">huggingface.co/Qwen</a> for the current model name.</div>')

    # -- 17. Orpheus 3B --
    if name == "orpheus":
        emotion_hint = ('<div class="alert alert-info py-2 small mt-2 mb-0">'
                        '<strong>Emotion tags (embed in text):</strong><br>'
                        '<code>&lt;laugh&gt;</code> &nbsp; <code>&lt;chuckle&gt;</code> &nbsp; <code>&lt;sigh&gt;</code> &nbsp; '
                        '<code>&lt;cough&gt;</code> &nbsp; <code>&lt;sniffle&gt;</code> &nbsp; <code>&lt;groan&gt;</code> &nbsp; '
                        '<code>&lt;yawn&gt;</code> &nbsp; <code>&lt;gasp&gt;</code><br>'
                        '<em>Example: "Oh my goodness &lt;sigh&gt; just a moment dear &lt;cough&gt; I need to find my glasses."</em>'
                        '</div>')
        gpu_warn = ('<div class="alert alert-danger py-2 small mt-2 mb-0">'
                    '⚠ <strong>Orpheus 3B requires a CUDA GPU</strong> — vllm will not run on CPU. '
                    'Load will fail immediately without a GPU. Add GPU and restart the server.</div>')
        return (_row(_grp("Voice", _sel("voice", ORPHEUS_VOICES, "tara")))
               +emotion_hint
               +gpu_warn)

    # -- 18. NeuTTS Air --
    if name == "neutts":
        return ('<div class="alert alert-warning py-2 small">'
                '⚠ <strong>NeuTTS Air is not yet configured.</strong><br>'
                'Edit <code>_load_neutts()</code> and <code>_synth_neutts()</code> in <code>tts_lab.py</code> '
                'with the correct package import once installed.</div>')

    # -- 19. IndexTTS-2 --
    if name == "indextts":
        return (f'<div class="param-row">{_upload_widget("idx-file","idx-status","idx-prompt-id","Reference WAV (REQUIRED — 5-30s of target voice)")}</div>'
               +'<p class="text-muted small mt-1">IndexTTS-2 requires a reference WAV for every synthesis call. '
               'Upload a clip of the voice you want to clone, then click Synthesise.</p>')

    # -- 20. Zonos v0.1 --
    if name == "zonos":
        lang_opts = [("en-us","English US"),("en-gb","English GB"),("de","German"),("fr","French"),
                     ("ja","Japanese"),("ko","Korean"),("zh","Chinese"),("es","Spanish")]
        emotion_info = ('<div class="alert alert-info py-2 small mt-2 mb-0">'
                        '<strong>Emotion vector</strong> — sliders below control the 8-dim emotion blend. '
                        'Higher <em>neutral</em> + low rest = calm elderly speech. '
                        'Raise <em>other</em> for natural variation.</div>')
        return (_row(_grp("Variant", _sel("variant", ZONOS_VARIANTS, "transformer")),
                     _grp("Language", _sel("language", lang_opts, "en-us")),
                     _grp('Speaking rate <span class="range-val">13.0</span>', _rng("speaking_rate","5.0","25.0","0.5","13.0","words/sec")),
                     _grp('Max tokens <span class="range-val">1024</span>', _rng("max_new_tokens","256","2048","64","1024")))
               +emotion_info
               +_row(_grp('Happiness <span class="range-val">0.3</span>', _rng("happiness","0.0","1.0","0.05","0.3")),
                     _grp('Sadness <span class="range-val">0.05</span>', _rng("sadness","0.0","1.0","0.05","0.05")),
                     _grp('Surprise <span class="range-val">0.1</span>', _rng("surprise","0.0","1.0","0.05","0.1")),
                     _grp('Neutral <span class="range-val">0.2</span>', _rng("neutral","0.0","1.0","0.05","0.2")),
                     _grp('Other <span class="range-val">0.2</span>', _rng("other","0.0","1.0","0.05","0.2")))
               +f'<div class="param-row">{_upload_widget("zn-file","zn-status","zn-prompt-id","Reference WAV (optional — speaker voice cloning)")}</div>')

    # -- 21. OpenVoice v2 --
    if name == "openvoice":
        sp = [("EN-US","EN-US American"),("EN-BR","EN-BR British"),("EN-AU","EN-AU Australian")]
        return ('<div class="alert alert-secondary py-2 small mb-2">MeloTTS synthesises; tone-color conversion adapts timbre. '
                'Without reference WAV: uses the selected base speaker identity directly.</div>'
               +_row(_grp("Base speaker", _sel("speaker", sp, "EN-US")),
                     _grp('Speed <span class="range-val">0.85</span>', _rng("speed","0.5","1.5","0.05","0.85")),
                     _grp('Tau (blend) <span class="range-val">0.3</span>', _rng("tau","0.0","1.0","0.05","0.3","0=original, 1=full clone")))
               +f'<div class="param-row">{_upload_widget("ov-file","ov-status","ov-prompt-id","Reference WAV (optional — voice to clone via tone-color conversion)")}</div>')

    return ""


def _build_presets():
    btns = " ".join(
        f'<button class="btn btn-sm btn-outline-secondary" onclick="setPreset(this.dataset.txt)" data-txt="{t}">{l}</button>'
        for l,t in ARTHUR_PRESETS)
    return f'<div class="preset-bar">{btns}</div>'

def _build_page():
    CSS = """
<style>
:root{--bs-body-bg:#1a1a2e;--bs-body-color:#e0e0e0;}
body{background:#1a1a2e;color:#e0e0e0;font-family:system-ui,sans-serif;}
.container-fluid{max-width:1400px;margin:auto;}
h2{color:#7eb8f7;font-weight:700;}

.nav-tabs{border-bottom:1px solid #333;gap:2px;margin-bottom:0;}
.nav-link{color:#aaa;border:1px solid #333;background:#1e2235;border-radius:6px 6px 0 0;font-size:.8rem;padding:5px 10px;transition:all .15s;}
.nav-link:hover{color:#fff;background:#2a3050;}
.nav-link.active{background:#2d3561;color:#7eb8f7;border-color:#4a5580;}
.tab-content{border:1px solid #333;border-top:none;border-radius:0 0 10px 10px;padding:16px;background:#1e2235;}

.model-header{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;margin-bottom:12px;}
.model-title{font-size:1.15rem;font-weight:700;color:#7eb8f7;}
.rtf-badge{background:#2a3561;color:#7eb8f7;border:1px solid #4a5580;border-radius:4px;padding:2px 7px;font-size:.75rem;margin-left:6px;}

.params-area{display:flex;flex-direction:column;gap:8px;}
.param-row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;}
.param-group{display:flex;flex-direction:column;gap:4px;min-width:160px;}
.param-group label{font-size:.78rem;color:#aaa;margin:0;}
.range-val{font-weight:700;color:#7eb8f7;margin-left:4px;}
.form-range::-webkit-slider-thumb{background:#7eb8f7;}

.text-box{width:100%;height:120px;background:#141428;color:#e0e0e0;border:1px solid #444;border-radius:8px;padding:12px;font-size:.95rem;resize:vertical;}
.preset-bar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}

.result-card{background:#141428;border:1px solid #333;border-radius:8px;padding:16px;margin-top:16px;display:none;}
.metric-pill{background:#2a3561;border:1px solid #4a5580;border-radius:20px;padding:4px 14px;font-size:.82rem;color:#7eb8f7;}
.metric-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}

.status-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin-bottom:20px;}
.status-card{background:#1e2235;border:1px solid #333;border-radius:8px;padding:12px 14px;font-size:.82rem;}
.status-card .name{font-weight:700;color:#7eb8f7;}
.dot-ok{color:#4caf50;} .dot-err{color:#f44336;} .dot-load{color:#ff9800;} .dot-off{color:#666;}

.ram-bar-wrap{background:#2a3050;border-radius:8px;overflow:hidden;height:16px;width:100%;max-width:400px;}
.ram-bar{background:linear-gradient(90deg,#4caf50,#7eb8f7);height:100%;transition:width .6s;}

.alert-info{background:#1a2a3a;border-color:#4a6a8a;color:#8ab8e8;}
.alert-warning{background:#2a2010;border-color:#6a5010;color:#d0a060;}

.btn-synth{background:#3d5af1;border-color:#3d5af1;font-weight:700;letter-spacing:.03em;}
.btn-synth:hover{background:#2d4ae1;}

audio{width:100%;margin-top:8px;}
.spinner{display:none;width:1.2rem;height:1.2rem;border:2px solid #7eb8f7;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-left:8px;}
@keyframes spin{to{transform:rotate(360deg)}}

code{background:#2a3050;padding:1px 5px;border-radius:4px;font-size:.85em;}
</style>"""

    JS = r"""
<script>
const API = '';
let lastAudio = null;

// Range display update
document.addEventListener('input', e => {
  if (e.target.type === 'range') {
    const span = e.target.closest('.param-group')?.querySelector('.range-val');
    if (span) span.textContent = parseFloat(e.target.value).toFixed(
      e.target.step < 0.1 ? 3 : e.target.step < 1 ? 2 : 0);
  }
});
function rangeUpdate(el) {}  // handled by above

function getParams(modelId) {
  const pane = document.getElementById('tab-' + modelId);
  const params = {};
  pane.querySelectorAll('[data-param]').forEach(el => {
    if (el.id && el.id.endsWith('-prompt-id') && !el.value) return;
    params[el.dataset.param] = el.value;
  });
  return params;
}

function setPreset(text) {
  document.getElementById('text-input').value = text;
}

async function synth(model) {
  const text = document.getElementById('text-input').value.trim();
  if (!text) { alert('Enter some text first'); return; }
  const btn = document.getElementById('btn-' + model) || document.querySelector(`#tab-${model} .btn-synth`);
  const spin = document.getElementById('spin-' + model) || document.createElement('span');
  const card = document.getElementById('result-' + model);
  if (btn) { btn.disabled = true; }
  if (spin) { spin.style.display = 'inline-block'; }
  const t0 = performance.now();
  try {
    const res = await fetch(`${API}/synthesize/${model}`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text, params: getParams(model)})
    });
    const data = await res.json();
    if (data.error) { showError(model, data.error + (data.trace ? '\n\n' + data.trace : '')); return; }
    const blob = new Blob([Uint8Array.from(atob(data.audio_b64), c => c.charCodeAt(0))], {type:'audio/wav'});
    const url = URL.createObjectURL(blob);
    if (card) {
      card.style.display = 'block';
      card.querySelector('.audio-player').src = url;
      card.querySelector('.audio-player').load();
      card.querySelector('.m-synth').textContent  = data.synth_time_ms + ' ms';
      card.querySelector('.m-dur').textContent    = data.audio_dur_ms  + ' ms';
      card.querySelector('.m-rtf').textContent    = data.rtf + '×';
      card.querySelector('.m-load').textContent   = data.load_time_s   + ' s';
      card.querySelector('.m-sr').textContent     = data.sample_rate   + ' Hz';
      const rtf = parseFloat(data.rtf);
      card.querySelector('.m-rtf').style.color = rtf <= 1 ? '#4caf50' : rtf <= 5 ? '#ff9800' : '#f44336';
    }
  } catch(e) { showError(model, e.toString()); }
  finally {
    if (btn) btn.disabled = false;
    if (spin) spin.style.display = 'none';
  }
}

function showError(model, msg) {
  const card = document.getElementById('result-' + model);
  if (card) {
    card.style.display = 'block';
    card.querySelector('.error-msg').textContent = msg;
  } else { alert(msg); }
}

async function unload(model) {
  await fetch(`${API}/models/${model}`, {method:'DELETE'});
  await refreshStatus();
}

async function uploadPrompt(fileId, statusId, hiddenId) {
  const input = document.getElementById(fileId);
  const status = document.getElementById(statusId);
  if (!input.files[0]) { status.textContent = 'No file selected'; return; }
  status.textContent = 'Uploading...';
  const fd = new FormData(); fd.append('file', input.files[0]);
  try {
    const r = await fetch(`${API}/upload`, {method:'POST', body:fd});
    const d = await r.json();
    document.getElementById(hiddenId).value = d.id;
    status.textContent = `✅ Uploaded (id:${d.id}, ${(d.size/1024).toFixed(0)} KB)`;
  } catch(e) { status.textContent = '❌ ' + e; }
}

async function refreshStatus() {
  const d = await (await fetch(`${API}/status`)).json();
  // RAM bar
  const {total, used, free} = d.system;
  const pct = (used/total*100).toFixed(1);
  document.getElementById('ram-bar').style.width = pct + '%';
  document.getElementById('ram-text').textContent =
    `RAM: ${used} / ${total} MB  (${free} MB free)  ${pct}%`;
  // GPU bar
  if (d.gpu && d.gpu.vram_total) {
    const gUsed = d.gpu.vram_used || 0;
    const gTot  = d.gpu.vram_total;
    const gPct  = (gUsed/gTot*100).toFixed(1);
    const gFree = d.gpu.vram_free ?? (gTot - gUsed);
    document.getElementById('vram-bar').style.width = gPct + '%';
    document.getElementById('vram-text').textContent =
      `VRAM: ${gUsed} / ${gTot} MB  (${gFree} MB free)  ${gPct}%`;
  }
  // Cards
  const grid = document.getElementById('status-grid');
  grid.innerHTML = Object.entries(d.models).map(([n,m]) => {
    const dot = m.status==='loaded' ? '🟢' : m.status==='loading' ? '🟡' : m.status==='error' ? '🔴' : '⚫';
    const av  = m.available ? `<span style="color:#4caf50">✓ available</span>` : `<span style="color:#f44336">✗ missing</span>`;
    const err = m.error ? `<div style="color:#f44336;font-size:.75rem">${m.error.substring(0,80)}</div>` : '';
    return `<div class="status-card">${dot} <span class="name">${m.label}</span> ${av}
    <div class="text-muted" style="font-size:.75rem">${m.size} · ${m.rtf_est} · ${m.ram_est_mb} MB RAM · ${m.status}</div>
    ${m.load_time_s>0?`<div class="text-muted" style="font-size:.75rem">Loaded in ${m.load_time_s}s</div>`:''}
    ${err}</div>`;
  }).join('');
}

setInterval(refreshStatus, 6000);
window.addEventListener('load', refreshStatus);

async function refreshAvailability() {
  const btn = document.getElementById('btn-refresh');
  if (btn) btn.disabled = true;
  try {
    await fetch(`${API}/refresh`, {method: 'POST'});
    await refreshStatus();
  } finally {
    if (btn) btn.disabled = false;
  }
}
</script>"""

    def _result_card(n):
        return (f'<div class="result-card" id="result-{n}">'
                f'<div class="metric-row">'
                f'<span class="metric-pill">⏱ Synth: <b class="m-synth">—</b></span>'
                f'<span class="metric-pill">🔊 Audio: <b class="m-dur">—</b></span>'
                f'<span class="metric-pill">RTF: <b class="m-rtf">—</b></span>'
                f'<span class="metric-pill">⬇ Load: <b class="m-load">—</b></span>'
                f'<span class="metric-pill">SR: <b class="m-sr">—</b></span>'
                f'</div>'
                f'<audio class="audio-player" controls preload="none"></audio>'
                f'<pre class="error-msg text-danger small mt-2"></pre>'
                f'</div>')

    tabs = []; panes = []
    for n in MODEL_ORDER:
        info = MODEL_INFO[n]
        ok, reason = _available(n)
        stars = _stars(info["arthur_fit"])
        badge = ('<span class="badge bg-success ms-1">available</span>' if ok else
                 f'<span class="badge bg-danger ms-1">missing</span>')
        tabs.append(f'<button class="nav-link{" active" if n==MODEL_ORDER[0] else ""}" '
                    f'data-bs-toggle="tab" data-bs-target="#tab-{n}" type="button">'
                    f'{info["label"]}{badge}</button>')
        panes.append(
            f'<div class="tab-pane fade{" show active" if n==MODEL_ORDER[0] else ""}" id="tab-{n}">'
            f'<div class="model-header">'
            f'<div><span class="model-title">{info["label"]}</span>'
            f' <span class="rtf-badge">{info["rtf_est"]}</span> {stars}</div>'
            f'<div class="model-meta text-muted small">Weights: <b>{info["size"]}</b> &nbsp;'
            f' RAM: <b>~{info["ram_est_mb"]} MB</b> &nbsp; Arthur fit: {stars}</div>'
            f'</div>'
            f'<div class="params-area">{_build_params(n)}</div>'
            +(f'<p class="text-warning mt-2">⚠ Not available: {reason}</p>' if not ok else '')
            +f'<div class="d-flex gap-2 align-items-center mt-3">'
            +f'<button id="btn-{n}" class="btn btn-primary btn-synth" onclick="synth(\'{n}\')">▶ Synthesise</button>'
            +f'<button class="btn btn-outline-secondary btn-sm" onclick="unload(\'{n}\')">⏏ Unload</button>'
            +f'<span id="spin-{n}" class="spinner"></span>'
            +f'</div>'
            +_result_card(n)
            +f'</div>')

    gpu_badge = (f'<span class="badge ms-2 px-2 py-1" style="background:#1e3a1e;color:#4caf50;border:1px solid #4caf50;font-size:.75rem">'
                 f'🟢 GPU: {DEVICE_NAME} · {VRAM_TOTAL_MB} MB VRAM</span>'
                 if DEVICE == "cuda" else
                 '<span class="badge ms-2 px-2 py-1" style="background:#3a1e1e;color:#f44336;border:1px solid #f44336;font-size:.75rem">🔴 CPU only</span>')

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arthur TTS Lab — {len(MODEL_ORDER)} Engines</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
{CSS}</head><body>
<div class="container-fluid py-3">
<h2 class="mb-1">🎙 Arthur TTS Lab <small class="text-muted fs-6">{len(MODEL_ORDER)} Engines</small>{gpu_badge}</h2>
<div class="mb-2 d-flex align-items-center gap-3 flex-wrap">
  <div>
    <div class="ram-bar-wrap"><div class="ram-bar" id="ram-bar" style="width:0%"></div></div>
    <small id="ram-text" class="text-muted">Loading RAM info...</small>
  </div>
  <div>
    <div class="ram-bar-wrap"><div class="ram-bar" id="vram-bar" style="width:0%;background:linear-gradient(90deg,#4caf50,#39c0c0)"></div></div>
    <small id="vram-text" class="text-muted">Loading VRAM info...</small>
  </div>
  <button id="btn-refresh" class="btn btn-sm btn-outline-secondary" onclick="refreshAvailability()" title="Re-check which packages are installed">🔄 Refresh availability</button>
</div>
<div class="status-grid" id="status-grid"></div>

<div class="mb-2">
  <label class="form-label fw-bold">Arthur text <small class="text-muted">(shared across all models)</small></label>
  {_build_presets()}
  <textarea id="text-input" class="text-box">{ARTHUR_PRESETS[0][1]}</textarea>
</div>

<div class="nav nav-tabs flex-wrap" id="model-tabs">{"".join(tabs)}</div>
<div class="tab-content">{"".join(panes)}</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
{JS}</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_lab:app", host="0.0.0.0", port=8001, reload=False, workers=1)