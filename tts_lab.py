#!/usr/bin/env python3
"""
Arthur TTS Lab -- 11-Engine Edition
Piper  Kokoro  MeloTTS  Bark  StyleTTS2  F5-TTS  Dia-1.6B
XTTS-v2  CosyVoice2  Parler-TTS  Chatterbox

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
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "True")
try:
    import torch
    torch.set_num_threads(_N_CORES)
    torch.set_num_interop_threads(max(1, _N_CORES // 2))
except Exception:
    pass

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

# -- Model registry --
MODEL_INFO = {
    "piper":     {"label":"Piper TTS",    "size":"61-116 MB","rtf_est":"~100x RT","ram_est_mb":200,  "heavy":False,"notes":"6 voices. ONNX CPU-only. Real-time. Best for production.","arthur_fit":2},
    "kokoro":    {"label":"Kokoro-82M",   "size":"89 MB",    "rtf_est":"~35x RT", "ram_est_mb":500,  "heavy":False,"notes":"54 voices, 9 languages. bm_lewis is the best Arthur voice.","arthur_fit":5},
    "melo":      {"label":"MeloTTS",      "size":"200 MB",   "rtf_est":"~15x RT", "ram_est_mb":1200, "heavy":False,"notes":"5 English accents. EN-BR sounds slightly older.","arthur_fit":3},
    "bark":      {"label":"Bark",         "size":"1.3 GB",   "rtf_est":"~30x RT", "ram_est_mb":1500, "heavy":True, "notes":"Unique emotion tokens: [laughs] [sighs] [clears throat] [hesitantly] in text.","arthur_fit":5},
    "styletts2": {"label":"StyleTTS 2",   "size":"0.7 GB",   "rtf_est":"~2x RT",  "ram_est_mb":1500, "heavy":True, "notes":"Fastest high-quality neural TTS. Style transfer from reference WAV. Alpha/beta control.","arthur_fit":4},
    "f5tts":     {"label":"F5-TTS",       "size":"1.2 GB",   "rtf_est":"~4x RT",  "ram_est_mb":2000, "heavy":True, "notes":"Best zero-shot voice cloning. Flow matching. Upload 5-15s reference WAV.","arthur_fit":4},
    "dia":       {"label":"Dia-1.6B",     "size":"3 GB",     "rtf_est":"~20x RT", "ram_est_mb":3000, "heavy":True, "notes":"Dialogue-native. [S1]/[S2] speakers + [laughs] [sighs] emotion tags. March 2025.","arthur_fit":5},
    "xtts":      {"label":"XTTS-v2",      "size":"1.8 GB",   "rtf_est":"~3x RT",  "ram_est_mb":3200, "heavy":True, "notes":"58 speakers, 17 languages. Voice cloning. Best multi-speaker quality.","arthur_fit":5},
    "cosyvoice": {"label":"CosyVoice2",   "size":"2 GB",     "rtf_est":"~5x RT",  "ram_est_mb":2500, "heavy":True, "notes":"Chinese-first with English zero-shot support.","arthur_fit":3},
    "parler":    {"label":"Parler-TTS",   "size":"3.3 GB",   "rtf_est":"~20x RT", "ram_est_mb":1500, "heavy":True, "notes":"Voice controlled entirely by natural language description.","arthur_fit":4},
    "chatterbox":{"label":"Chatterbox",   "size":"3.0 GB",   "rtf_est":"~12x RT", "ram_est_mb":1800, "heavy":True, "notes":"Exaggeration slider + voice cloning. Most controllable confusion.","arthur_fit":5},
}

MODEL_ORDER = ["piper","kokoro","melo","bark","styletts2","f5tts","dia","xtts","cosyvoice","parler","chatterbox"]

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
        "error":"","load_time_s":0.0,"loaded_voice":None}
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

# ============================================================
# LOADERS + SYNTH FUNCTIONS
# ============================================================

# -- 1. Piper --
def _load_piper(voice="en_US-ryan-high"):
    import onnxruntime as ort
    from piper.voice import PiperVoice
    mp = MODELS_DIR / f"{voice}.onnx"; cp = MODELS_DIR / f"{voice}.onnx.json"
    if not mp.exists(): raise FileNotFoundError(f"Piper voice not found: {mp}")
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
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    mp = MODELS_DIR/"kokoro-v1.0.onnx"; vp = MODELS_DIR/"voices-v1.0.bin"
    if not mp.exists(): raise FileNotFoundError(f"kokoro-v1.0.onnx missing")
    opts = ort.SessionOptions(); opts.intra_op_num_threads = _N_CORES
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
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
    return TTS(language="EN", device="cpu")

def _synth_melo(inst, text, params):
    sp_ids = dict(inst.hps.data.spk2id)
    sp = params.get("speaker","EN-US").replace("-","_").replace("EN_US","EN-US").replace("EN_BR","EN-BR").replace("EN_AU","EN-AU")
    sp_id = sp_ids.get(sp) or sp_ids.get("EN-US") or list(sp_ids.values())[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: tmp = f.name
    inst.tts_to_file(text, sp_id, tmp, speed=float(params.get("speed", 0.85)))
    wav = Path(tmp).read_bytes(); Path(tmp).unlink(missing_ok=True)
    with wave.open(io.BytesIO(wav),"rb") as wf: sr = wf.getframerate()
    return wav, sr

# -- 4. Bark --
def _load_bark():
    import torch
    # Bark checkpoints contain numpy scalars not whitelisted in PyTorch 2.6+ weights_only mode.
    # Patch torch.load to allow legacy pickles only during preload, then restore.
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    try:
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"
        from bark import preload_models
        preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)
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

# -- 5. StyleTTS 2 --
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

# -- 6. F5-TTS --
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

# -- 7. Dia-1.6B --
def _load_dia():
    from dia.model import Dia
    # Dia-1.6B-0626 has updated config schema matching current package;
    # fall back to Dia-1.6B for cached weights
    for mid in ["nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"]:
        try:
            return Dia.from_pretrained(mid, compute_dtype="float32")
        except Exception as e:
            if mid == "nari-labs/Dia-1.6B":
                raise
            last_err = e
            continue

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

# -- 8. XTTS-v2 --
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
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

def _synth_xtts(inst, text, params):
    kw = dict(text=text, speaker=params.get("speaker","Torcull Diarmuid"), language=params.get("language","en"))
    if params.get("temperature"): kw["temperature"] = float(params["temperature"])
    arr = inst.tts(**kw)
    return _to_wav(np.array(arr, dtype=np.float32), 24000), 24000

# -- 9. CosyVoice2 --
def _load_cosyvoice():
    for p in [str(COSYVOICE_DIR), str(COSYVOICE_DIR/"third_party"/"Matcha-TTS")]:
        if p not in sys.path: sys.path.insert(0, p)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    md = COSYVOICE_DIR/"pretrained_models"/"CosyVoice2-0.5B"
    return CosyVoice2(str(md), load_jit=False, load_trt=False)

def _synth_cosyvoice(inst, text, params):
    chunks = [c["tts_speech"].numpy().flatten() for c in inst.inference_sft(text, params.get("speaker","English Female"))]
    sr = inst.sample_rate
    return _to_wav(np.concatenate(chunks) if chunks else np.zeros(sr,np.float32), sr), sr

# -- 10. Parler-TTS --
def _load_parler():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    mid = "parler-tts/parler-tts-mini-v1"
    return (ParlerTTSForConditionalGeneration.from_pretrained(mid), AutoTokenizer.from_pretrained(mid))

def _synth_parler(inst, text, params):
    import torch
    model, tok = inst
    desc = params.get("description","An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly.")
    iids = tok(desc, return_tensors="pt").input_ids
    pids = tok(text, return_tensors="pt").input_ids
    kw = dict(input_ids=iids, prompt_input_ids=pids)
    if params.get("temperature"): kw["temperature"]=float(params["temperature"]); kw["do_sample"]=True
    if params.get("max_new_tokens"): kw["max_new_tokens"]=int(float(params["max_new_tokens"]))
    with torch.no_grad(): gen = model.generate(**kw)
    return _to_wav(gen.cpu().numpy().squeeze().astype(np.float32), model.config.sampling_rate), model.config.sampling_rate

# -- 11. Chatterbox --
def _load_chatterbox():
    import perth
    # perth 1.0.0 ships PerthImplicitWatermarker=None (proprietary stub).
    # Chatterbox calls it in __init__; patch with DummyWatermarker so it loads.
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device="cpu")

def _synth_chatterbox(inst, text, params):
    import torchaudio as ta
    kw = dict(exaggeration=float(params.get("exaggeration",0.65)), cfg_weight=float(params.get("cfg_weight",0.5)))
    seed = params.get("seed")
    if seed and int(float(seed)) != 0: kw["seed"] = int(float(seed))
    pid = params.get("audio_prompt_id")
    if pid:
        p = UPLOAD_DIR / f"{pid}.wav"
        if p.exists(): kw["audio_prompt_path"] = str(p)
    wav = inst.generate(text, **kw)
    buf = io.BytesIO(); ta.save(buf, wav, inst.sr, format="wav"); buf.seek(0)
    return buf.read(), inst.sr

# ============================================================
# REGISTRY + AVAILABILITY + DISPATCH
# ============================================================
LOADERS  = {"piper":_load_piper,"kokoro":_load_kokoro,"melo":_load_melo,
            "bark":_load_bark,"styletts2":_load_styletts2,"f5tts":_load_f5tts,
            "dia":_load_dia,"xtts":_load_xtts,"cosyvoice":_load_cosyvoice,
            "parler":_load_parler,"chatterbox":_load_chatterbox}
SYNTHERS = {"piper":_synth_piper,"kokoro":_synth_kokoro,"melo":_synth_melo,
            "bark":_synth_bark,"styletts2":_synth_styletts2,"f5tts":_synth_f5tts,
            "dia":_synth_dia,"xtts":_synth_xtts,"cosyvoice":_synth_cosyvoice,
            "parler":_synth_parler,"chatterbox":_synth_chatterbox}

_import_cache = {}
def _available(name):
    if name in _import_cache: return _import_cache[name]
    def _check():
        import importlib.util as ilu
        pkg_map = {
            "piper":"piper","kokoro":"kokoro_onnx","melo":"melo",
            "bark":"bark","styletts2":"styletts2","f5tts":"f5_tts",
            "dia":"dia","xtts":"TTS","cosyvoice":None,"parler":"parler_tts","chatterbox":"chatterbox",
        }
        pkg = pkg_map.get(name)
        if pkg and not ilu.find_spec(pkg): return False, f"pip install {pkg} needed"
        if name == "piper" and not _piper_voices(): return False, "No .onnx voice found"
        if name == "kokoro" and not (MODELS_DIR/"kokoro-v1.0.onnx").exists(): return False, "kokoro-v1.0.onnx missing"
        if name == "cosyvoice":
            if not COSYVOICE_DIR.exists(): return False, "git clone FunAudioLLM/CosyVoice /opt/CosyVoice"
            if not (COSYVOICE_DIR/"pretrained_models"/"CosyVoice2-0.5B").exists(): return False, "Model not downloaded"
        stmts = {
            "piper":     "from piper.voice import PiperVoice",
            "kokoro":    "from kokoro_onnx import Kokoro",
            "melo":      "from melo.api import TTS as _M",
            "bark":      "from bark import generate_audio",
            "styletts2": "from styletts2 import tts as _st2",
            "f5tts":     "from f5_tts.api import F5TTS",
            "dia":       "from dia.model import Dia",
            "xtts":      "from TTS.api import TTS as _X",
            "cosyvoice": "from cosyvoice.cli.cosyvoice import CosyVoice2",
            "parler":    "from parler_tts import ParlerTTSForConditionalGeneration",
            "chatterbox":"from chatterbox.tts import ChatterboxTTS",
        }
        stmt = stmts.get(name, "")
        if stmt:
            if name == "xtts": _patch_transformers_for_coqui()
            if name == "cosyvoice":
                for p in [str(COSYVOICE_DIR), str(COSYVOICE_DIR/"third_party"/"Matcha-TTS")]:
                    if p not in sys.path: sys.path.insert(0, p)
            try: exec(stmt, {})
            except ImportError as e: return False, f"Import error: {e}"
        return True, ""
    r = _check()
    _import_cache[name] = r
    return r

def _do_synth(name, text, params):
    st = _state[name]
    with st["lock"]:
        if name == "piper":
            wanted = params.get("voice", "en_US-ryan-high")
            if st["instance"] and st.get("loaded_voice") != wanted:
                _safe_del(st["instance"]); st["instance"] = None
        if st["instance"] is None:
            ok, reason = _available(name)
            if not ok: raise RuntimeError(f"Not available: {reason}")
            if MODEL_INFO[name]["heavy"]: _evict_heavy(keep=name)
            st["status"] = "loading"; t0 = time.perf_counter()
            try:
                st["instance"] = (_load_piper(params.get("voice","en_US-ryan-high"))
                                  if name == "piper" else LOADERS[name]())
                st["load_time_s"] = round(time.perf_counter()-t0, 2)
                st["status"] = "loaded"; st["error"] = ""
                if name == "piper": st["loaded_voice"] = params.get("voice","en_US-ryan-high")
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

@app.get("/", response_class=HTMLResponse)
async def index(): return HTMLResponse(_build_page())

@app.get("/status")
async def status():
    models = {}
    for n in MODEL_ORDER:
        ok, reason = _available(n); st = _state[n]
        models[n] = {**MODEL_INFO[n], "available":ok, "reason":reason,
                     "status":st["status"], "load_time_s":st["load_time_s"], "error":st["error"]}
    tot, used, free = _ram_mb()
    return JSONResponse({"models":models, "system":{"total":tot,"used":used,"free":free}})

@app.get("/voices/{model}")
async def voices(model):
    vmap = {
        "piper":     _piper_voices() or ["en_US-ryan-high"],
        "kokoro":    ALL_KOKORO_VOICES,
        "melo":      ["EN-Default","EN-US","EN-BR","EN-AU","EN_INDIA"],
        "bark":      [v for v,_ in BARK_PRESETS],
        "xtts":      ALL_XTTS_SPEAKERS,
        "cosyvoice": ["English Female","English Male"],
    }
    return JSONResponse({"voices": vmap.get(model, [])})

@app.post("/synthesize/{model}")
async def synthesize(model, req: SynthReq):
    if model not in MODEL_ORDER: return JSONResponse({"error":f"Unknown: {model}"}, status_code=400)
    if not req.text.strip(): return JSONResponse({"error":"Empty text"}, status_code=400)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _do_synth, model, req.text, req.params)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error":str(e),"trace":traceback.format_exc(limit=4)}, status_code=500)

@app.delete("/models/{model}")
async def unload_model(model):
    st = _state.get(model)
    if st and st["instance"] is not None:
        _safe_del(st["instance"]); st["instance"] = None; st["status"] = "unloaded"
    return {"unloaded": model}

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
        return (f'<div class="param-row" style="flex-direction:column">'
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

    return ""

def _build_tabs():
    tabs = []; panes = []
    for n in MODEL_ORDER:
        info  = MODEL_INFO[n]
        ok, reason = _available(n)
        stars = _stars(info["arthur_fit"])
        badge = ('<span class="badge bg-success ms-1">available</span>' if ok else
                 f'<span class="badge bg-danger ms-1" title="{reason}">missing</span>')
        note_icon = f'<span class="text-muted" style="font-size:.75rem" title="{info["notes"]}">ℹ️</span>'
        rtf_badge = f'<span class="rtf-badge">{info["rtf_est"]}</span>'
        tabs.append(f'<button class="nav-link{" active" if n==MODEL_ORDER[0] else ""}" '
                    f'id="tab-btn-{n}" data-bs-toggle="tab" data-bs-target="#tab-{n}" type="button">'
                    f'{info["label"]}{badge}</button>')
        params_html = _build_params(n)
        panes.append(
            f'<div class="tab-pane fade{" show active" if n==MODEL_ORDER[0] else ""}" id="tab-{n}">'
            f'<div class="model-header">'
            f'<div><span class="model-title">{info["label"]}</span>'
            f' {rtf_badge} {stars} {note_icon}</div>'
            f'<div class="model-meta text-muted small">'
            f'Weights: <b>{info["size"]}</b> &nbsp; RAM: <b>~{info["ram_est_mb"]} MB</b> &nbsp; '
            f'Est. RTF: <b>{info["rtf_est"]}</b>'
            f'{"  <span class=badge+bg-warning+text-dark>heavy</span>".replace("+", " ") if info["heavy"] else ""}'
            f'</div></div>'
            f'<div class="params-area">{params_html}</div>'
            f'{"<hr>" if not ok else ""}'
            f'{("<p class=text-warning>⚠ " + reason + "</p>") if not ok else ""}'
            f'<button class="btn btn-primary btn-synth mt-2" onclick="synth(\'{n}\')">'
            f'▶ Synthesise with {info["label"]}</button>'
            f'<button class="btn btn-outline-secondary btn-sm ms-2 mt-2" onclick="unload(\'{n}\')">'
            f'⏏ Unload</button>'
            f'</div>')
    nav  = f'<div class="nav nav-tabs flex-wrap" id="model-tabs">{"".join(tabs)}</div>'
    pane = f'<div class="tab-content" id="model-tab-content">{"".join(panes)}</div>'
    return nav + pane

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

    tabs_html = _build_tabs()
    # Inject spinner + result card into each tab pane
    for n in MODEL_ORDER:
        rc = _result_card(n)
        tabs_html = tabs_html.replace(
            f'<div class="tab-pane fade{" show active" if n==MODEL_ORDER[0] else ""}" id="tab-{n}">',
            f'<div class="tab-pane fade{" show active" if n==MODEL_ORDER[0] else ""}" id="tab-{n}">')
        tabs_html += rc  # appended below

    # Re-build with result cards properly inside panes
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

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arthur TTS Lab — 11 Engines</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
{CSS}</head><body>
<div class="container-fluid py-3">
<h2 class="mb-1">🎙 Arthur TTS Lab <small class="text-muted fs-6">11 Engines · {len(MODEL_ORDER)} models</small></h2>
<div class="mb-2">
  <div class="ram-bar-wrap"><div class="ram-bar" id="ram-bar" style="width:0%"></div></div>
  <small id="ram-text" class="text-muted">Loading RAM info...</small>
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