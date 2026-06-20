"""
tts_lab_engines.py — all 21 TTS engine loader + synth pairs.

Each engine exposes:
    _load_<name>([params])  →  instance
    _synth_<name>(inst, text, params)  →  (wav_bytes, sample_rate)

Bottom of file: LOADERS and SYNTHERS dicts used by tts_lab_dispatch.
"""
from __future__ import annotations
import io, os, re, sys, tempfile, time, wave
import numpy as np
from pathlib import Path

from tts_lab_shims  import _N_CORES, DEVICE
from tts_lab_config import (
    MODELS_DIR, COSYVOICE_DIR, UPLOAD_DIR, INDEXTTS_DIR, OPENVOICE_MODELS_DIR,
    OUTETTS_DEFAULT_GGUF, OUTETTS_DEFAULT_TOKENIZER, QWEN3TTS_MODEL_ID,
    slog,
)
from tts_lab_utils import _to_wav, _wav_dur, _read_wav_mono_f32, _require_gpu


# ── 1. Piper ──────────────────────────────────────────────────────────────────
def _load_piper(voice="en_US-ryan-high"):
    import onnxruntime as ort
    from piper.voice import PiperVoice
    mp = MODELS_DIR / f"{voice}.onnx"
    cp = MODELS_DIR / f"{voice}.onnx.json"
    if not mp.exists():
        raise FileNotFoundError(f"Piper voice not found: {mp}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _N_CORES
    opts.inter_op_num_threads = max(1, _N_CORES // 2)
    try:
        return PiperVoice.load(str(mp), config_path=str(cp) if cp.exists() else None,
                               use_cuda=False, sess_options=opts)
    except TypeError:
        return PiperVoice.load(str(mp), config_path=str(cp) if cp.exists() else None,
                               use_cuda=False)

def _synth_piper(inst, text, params):
    sr = inst.config.sample_rate
    raw = bytearray()
    spd = float(params.get("speed", 1.0))
    try:
        from piper.config import SynthesisConfig
        cfg = SynthesisConfig(
            length_scale=float(params.get("length_scale", 1.0 / spd if spd != 1.0 else 1.0)),
            noise_scale=float(params.get("noise_scale", 0.667)),
            noise_w=float(params.get("noise_w", 0.8)),
        )
        for chunk in inst.synthesize(text, cfg):
            raw.extend(chunk.audio_int16_bytes)
            sr = chunk.sample_rate
    except (ImportError, TypeError):
        for chunk in inst.synthesize(text):
            raw.extend(chunk.audio_int16_bytes)
            sr = chunk.sample_rate
    return _to_wav(np.frombuffer(bytes(raw), dtype=np.int16).astype(np.float32) / 32767, sr), sr


# ── 2. Kokoro ─────────────────────────────────────────────────────────────────
def _load_kokoro():
    from tts_lab_config import KOKORO_LANG_MAP
    ESPEAK_DATA = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper as _EW
        if not hasattr(_EW, "set_data_path"):
            _EW.set_data_path = classmethod(lambda cls, p: None)
        os.environ.setdefault("ESPEAK_DATA_PATH", ESPEAK_DATA)
    except Exception:
        pass
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    mp = MODELS_DIR / "kokoro-v1.0.onnx"
    vp = MODELS_DIR / "voices-v1.0.bin"
    if not mp.exists():
        raise FileNotFoundError("kokoro-v1.0.onnx missing")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _N_CORES
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    try:
        from kokoro_onnx.config import EspeakConfig
        return Kokoro(str(mp), str(vp), espeak_config=EspeakConfig(data_path=ESPEAK_DATA))
    except (ImportError, TypeError, AttributeError):
        pass
    try:
        return Kokoro(str(mp), str(vp), sess_options=opts)
    except TypeError:
        return Kokoro(str(mp), str(vp))

def _synth_kokoro(inst, text, params):
    from tts_lab_config import KOKORO_LANG_MAP
    voice = params.get("voice", "bm_lewis")
    lang  = params.get("lang") or KOKORO_LANG_MAP.get(voice[:2], "en-us")
    samples, sr = inst.create(text, voice=voice, speed=float(params.get("speed", 0.85)), lang=lang)
    return _to_wav(np.array(samples, dtype=np.float32), sr), sr


# ── 3. MeloTTS ────────────────────────────────────────────────────────────────
def _load_melo():
    from melo.api import TTS
    return TTS(language="EN", device=DEVICE)

def _synth_melo(inst, text, params):
    sp_ids = dict(inst.hps.data.spk2id)
    sp = (params.get("speaker", "EN-US")
          .replace("-", "_").replace("EN_US", "EN-US")
          .replace("EN_BR", "EN-BR").replace("EN_AU", "EN-AU"))
    sp_id = sp_ids.get(sp) or sp_ids.get("EN-US") or list(sp_ids.values())[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    inst.tts_to_file(text, sp_id, tmp, speed=float(params.get("speed", 0.85)))
    wav = Path(tmp).read_bytes()
    Path(tmp).unlink(missing_ok=True)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        sr = wf.getframerate()
    return wav, sr


# ── 4. ChatTTS ────────────────────────────────────────────────────────────────
def _load_chattts():
    try:
        from transformers import BertTokenizer as _BT
        if not hasattr(_BT, "encode_plus"):
            def _encode_plus(self, text, **kwargs):
                return self(text, **kwargs)
            _BT.encode_plus = _encode_plus
    except Exception:
        pass
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


# ── 5. OuteTTS ────────────────────────────────────────────────────────────────
#
#   HF backend is permanently broken: pre-encodes text as ~15K positional tokens.
#   MUST use GGUF via LLAMACPP backend.
#   Default: /opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf
#   POST params: {"model_path":"/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf"}

def _load_outetts(model_path=OUTETTS_DEFAULT_GGUF):
    import outetts
    gguf_path = model_path
    if not gguf_path.endswith(".gguf"):
        # HF backend is permanently broken (pre-encodes text as ~15K tokens).
        # Silently fall back to the default GGUF instead of raising.
        gguf_path = OUTETTS_DEFAULT_GGUF
    # Pick the matching HF tokenizer for each GGUF variant
    if "0.3" in gguf_path:
        tokenizer = "OuteAI/OuteTTS-0.3-500M"
    else:
        tokenizer = OUTETTS_DEFAULT_TOKENIZER   # "OuteAI/OuteTTS-1.0-0.6B" covers both Q4 and Q8
    cfg = outetts.ModelConfig(
        model_path=gguf_path,
        tokenizer_path=tokenizer,
        backend=outetts.Backend.LLAMACPP,
        device=DEVICE,
        max_seq_length=32768,
        n_gpu_layers=99,
    )
    return outetts.Interface(cfg)

def _synth_outetts(inst, text, params):
    import outetts
    speaker = None
    prompt_id = params.get("audio_prompt_id", "")
    if prompt_id:
        prompt_path = UPLOAD_DIR / f"{prompt_id}.wav"
        if prompt_path.exists():
            speaker = inst.create_speaker(str(prompt_path),
                                          transcript=params.get("transcript", "") or None)
    if speaker is None:
        speaker = inst.load_default_speaker(params.get("speaker", "en-female-1-neutral"))
    try:
        gen_type = outetts.GenerationType.CHUNKED
        gen_cfg_kw = dict(
            text=text,
            voice_characteristics=params.get("voice_characteristics") or None,
            speaker=speaker,
            generation_type=gen_type,
        )
    except AttributeError:
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


# ── 6. Bark ───────────────────────────────────────────────────────────────────
def _load_bark():
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        _use_small = (DEVICE != "cuda")
        os.environ["SUNO_USE_SMALL_MODELS"] = "True" if _use_small else "False"
        from bark import preload_models
        preload_models(
            text_use_small=_use_small, coarse_use_small=_use_small, fine_use_small=_use_small,
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
    preset  = params.get("voice_preset", "v2/en_speaker_6")
    history = preset if preset != "none" else None
    audio   = generate_audio(text, history_prompt=history)
    return _to_wav(audio.astype(np.float32), SAMPLE_RATE), SAMPLE_RATE


# ── 7. StyleTTS 2 ─────────────────────────────────────────────────────────────
def _load_styletts2():
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        from styletts2 import tts
        result = tts.StyleTTS2()
    finally:
        torch.load = _orig
    return result

def _synth_styletts2(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
    if ref_path and not Path(ref_path).exists():
        ref_path = None
    out = inst.inference(
        text=text,
        target_voice_path=ref_path,
        alpha=float(params.get("alpha", 0.3)),
        beta=float(params.get("beta", 0.7)),
        diffusion_steps=int(float(params.get("diffusion_steps", 5))),
        embedding_scale=float(params.get("embedding_scale", 1.0)),
    )
    return _to_wav(np.array(out, dtype=np.float32), 24000), 24000


# ── 8. F5-TTS ─────────────────────────────────────────────────────────────────
def _load_f5tts():
    from f5_tts.api import F5TTS
    return F5TTS(device=DEVICE)

def _synth_f5tts(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = UPLOAD_DIR / f"{ref_id}.wav" if ref_id else None
    if not ref_path or not ref_path.exists():
        raise RuntimeError("F5-TTS requires a reference audio clip. Upload a 5-15s WAV first.")
    ref_text = params.get("ref_text", "")
    speed    = float(params.get("speed", 1.0))
    nfe      = int(float(params.get("nfe_step", 32)))

    # F5-TTS v1 Base vocab has only 25 Arabic chars: أبةتجحدرزسصطعقكلمنهويَُِْ
    # Missing 17 Persian chars (اآپثچخذژشضظغفکگیئ) — all map to space token (idx 0),
    # producing silence/garbled output.  Map to closest in-vocab Arabic equivalents.
    # NOTE: mappings like پ→ب, چ→ج, ف→ب, ش→س, گ→ق lose phonemic distinctions.
    # This makes speech understandable but heavily accented.
    _F5TTS_PERSIAN_MAP = str.maketrans({
        "ا": "أ",  # ALEF → ALEF HAMZA (lossless)
        "آ": "أ",  # ALEF MADDA → ALEF HAMZA
        "پ": "ب",  # PE → BE (labial → labial)
        "ث": "س",  # SE → SIN (fricative → fricative)
        "چ": "ج",  # CHE → JEEM
        "خ": "ح",  # KHE → HE (guttural → guttural)
        "ذ": "ز",  # ZAL → ZE (dental → dental)
        "ژ": "ز",  # ZHE → ZE
        "ش": "س",  # SHIN → SIN (sibilant → sibilant)
        "ض": "ص",  # ZAD → SAD (emphatic → emphatic)
        "ظ": "ط",  # ZA → TA (emphatic → emphatic)
        "غ": "ع",  # GHAYN → AYN (pharyngeal → pharyngeal)
        "ف": "ب",  # FE → BE (only labial in vocab; "F" sound lost)
        "ک": "ك",  # Persian KAF → Arabic KAF (lossless)
        "گ": "ق",  # GAF → QAF
        "ی": "ي",  # Persian YE → Arabic YE (lossless)
        "ئ": "ي",  # YE HAMZA → YE
    })
    text = text.translate(_F5TTS_PERSIAN_MAP)
    if ref_text:
        ref_text = ref_text.translate(_F5TTS_PERSIAN_MAP)

    wav, sr, _ = inst.infer(
        ref_file=str(ref_path), ref_text=ref_text,
        gen_text=text, speed=speed, nfe_step=nfe,
    )
    return _to_wav(np.array(wav, dtype=np.float32).flatten(), sr), sr


# ── 9. Dia-1.6B ───────────────────────────────────────────────────────────────
def _load_dia():
    from dia.model import Dia
    _dtype = "bfloat16" if DEVICE == "cuda" else "float32"
    for mid in ["nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"]:
        try:
            return Dia.from_pretrained(mid, compute_dtype=_dtype, device=DEVICE)
        except TypeError:
            try:
                return Dia.from_pretrained(mid, compute_dtype=_dtype)
            except Exception as e:
                if mid == "nari-labs/Dia-1.6B":
                    raise
                continue
        except Exception as e:
            if mid == "nari-labs/Dia-1.6B":
                raise
            continue

def _synth_dia(inst, text, params):
    if "[S1]" not in text and "[S2]" not in text:
        text = f"[S1] {text}"
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
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
    sr  = 44100
    arr = np.array(output, dtype=np.float32).flatten() if output is not None else np.zeros(sr, dtype=np.float32)
    return _to_wav(arr, sr), sr


# ── 10. XTTS-v2 ───────────────────────────────────────────────────────────────
def _load_xtts():
    # All transformers 5.x shims applied at startup in tts_lab_shims.py
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2",
               progress_bar=False, gpu=(DEVICE == "cuda"))

def _synth_xtts(inst, text, params):
    kw = dict(text=text,
              speaker=params.get("speaker", "Torcull Diarmuid"),
              language=params.get("language", "en"))
    if params.get("temperature"):
        kw["temperature"] = float(params["temperature"])
    arr = inst.tts(**kw)
    return _to_wav(np.array(arr, dtype=np.float32), 24000), 24000


# ── 11. CosyVoice2 ────────────────────────────────────────────────────────────
def _load_cosyvoice():
    import importlib.util as _ilu
    if not _ilu.find_spec("hyperpyyaml"):
        raise ImportError(
            "CosyVoice2 requires hyperpyyaml — run:  pip install hyperpyyaml\n"
            "Then restart the server."
        )
    for p in [str(COSYVOICE_DIR), str(COSYVOICE_DIR / "third_party" / "Matcha-TTS")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    md = COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
    return CosyVoice2(str(md), load_jit=False, load_trt=False)

def _synth_cosyvoice(inst, text, params):
    import soundfile as _sf
    import tempfile, os

    def _read_wav_16k(path):
        """Read WAV as mono float32 tensor at 16 kHz via soundfile (no torchcodec)."""
        import torch as _t
        data, sr = _sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            import torchaudio as _ta
            t = _t.from_numpy(data).unsqueeze(0)
            data = _ta.functional.resample(t, sr, 16000).squeeze(0).numpy()
        return data  # numpy float32 mono

    prompt_id   = params.get("audio_prompt_id", "")
    prompt_path = UPLOAD_DIR / f"{prompt_id}.wav" if prompt_id else None

    if prompt_path and prompt_path.exists():
        ref_numpy = _read_wav_16k(prompt_path)
        ref_text  = params.get("transcript", "") or ""
    else:
        ref_numpy = _read_wav_16k(COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav")
        ref_text  = "And then he said, the excitement in his voice unmistakable."

    # CosyVoice's load_wav() expects a file path — write tensor to a temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _sf.write(tmp_path, ref_numpy, 16000)
        chunks = [c["tts_speech"].numpy().flatten()
                  for c in inst.inference_zero_shot(text, ref_text, tmp_path)]
    finally:
        os.unlink(tmp_path)

    sr = inst.sample_rate
    return _to_wav(np.concatenate(chunks) if chunks else np.zeros(sr, np.float32), sr), sr


# ── 12. Parler-TTS ────────────────────────────────────────────────────────────
def _load_parler(model_id="parler-tts/parler-tts-mini-v1"):
    import os
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    # Resolve local cached path to avoid HF hub permission issues
    _hf_home = os.environ.get("HF_HOME", "/opt/models/huggingface")
    _slug = model_id.replace("/", "--")
    _snap_dir = os.path.join(_hf_home, "hub", f"models--{_slug}", "snapshots")
    if os.path.isdir(_snap_dir):
        snaps = sorted(os.listdir(_snap_dir))
        if snaps:
            model_id = os.path.join(_snap_dir, snaps[-1])
    mdl = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(model_id)
    return (mdl, tok)

def _synth_parler(inst, text, params):
    import torch
    model, tok = inst
    desc = params.get("description", "An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly.")
    iids = tok(desc, return_tensors="pt").input_ids.to(DEVICE)
    pids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    kw = dict(input_ids=iids, prompt_input_ids=pids)
    if params.get("temperature"):
        kw["temperature"] = float(params["temperature"])
        kw["do_sample"]   = True
    if params.get("max_new_tokens"):
        kw["max_new_tokens"] = int(float(params["max_new_tokens"]))
    with torch.no_grad():
        gen = model.generate(**kw)
    return _to_wav(gen.cpu().numpy().squeeze().astype(np.float32),
                   model.config.sampling_rate), model.config.sampling_rate


# ── 13. Chatterbox ────────────────────────────────────────────────────────────
def _load_chatterbox(model="default"):
    """Load Chatterbox TTS.

    Models:
      default — English-only (0.5B, 16 layers, 704 tokens)
      persian — Persian fine-tune (0.5B, 30 layers, 2454 tokens) from hootan09
      v3      — Multilingual v3 (1.0B, 30 layers, 2454 tokens, new vocoder) from ResembleAI
    """
    import types, importlib.machinery, shutil
    for _tc in ["torchcodec", "torchcodec._C", "torchcodec.decoders",
                "torchcodec.decoders._core", "torchcodec.decoders.video_decoder",
                "torchcodec.encoders"]:
        if _tc not in sys.modules:
            _m = types.ModuleType(_tc)
            _is_pkg = "." not in _tc.split("torchcodec.")[-1] or _tc == "torchcodec"
            _m.__spec__ = importlib.machinery.ModuleSpec(_tc, loader=None, is_package=_is_pkg)
            _m.__path__ = []
            sys.modules[_tc] = _m
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS

    if model == "persian":
        from huggingface_hub import snapshot_download
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.t3 import T3
        from chatterbox.models.tokenizers import EnTokenizer
        from safetensors.torch import load_file
        fa_dir = Path(snapshot_download(
            "hootan09/ChatterBox",
            allow_patterns=["t3_fa.safetensors", "mtl_tokenizer.json"],
        ))
        inst = ChatterboxTTS.from_pretrained(device="cpu")
        t3_new = T3(hp=T3Config.multilingual())
        t3_state = load_file(str(fa_dir / "t3_fa.safetensors"))
        t3_new.load_state_dict(t3_state, strict=True)
        inst.t3 = t3_new.to(DEVICE).eval()
        inst.ve = inst.ve.to(DEVICE)
        inst.s3gen = inst.s3gen.to(DEVICE)
        inst.device = DEVICE
        inst.tokenizer = EnTokenizer(str(fa_dir / "mtl_tokenizer.json"))
        inst._needs_persian_char_map = True  # mTL tokenizer missing آ/أ/إ — text-level mapping in synth
        return inst

    if model == "v3":
        from huggingface_hub import snapshot_download
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.t3 import T3
        from chatterbox.models.tokenizers import EnTokenizer
        from safetensors.torch import load_file
        v3_dir = Path(snapshot_download(
            "ResembleAI/chatterbox",
            allow_patterns=["t3_mtl23ls_v3.safetensors", "s3gen_v3.safetensors",
                          "grapheme_mtl_merged_expanded_v1.json", "ve.safetensors"],
        ))
        slog("LOAD", "chatterbox", "Loading v3 (30-layer, 2GB, 2454-token grapheme)...")
        inst = ChatterboxTTS.from_pretrained(device="cpu")
        t3_new = T3(hp=T3Config.multilingual())
        t3_state = load_file(str(v3_dir / "t3_mtl23ls_v3.safetensors"))
        t3_new.load_state_dict(t3_state, strict=True)
        inst.t3 = t3_new.to(DEVICE).eval()
        # v3 has its own vocoder (s3gen_v3)
        inst.s3gen.load_state_dict(load_file(str(v3_dir / "s3gen_v3.safetensors")), strict=False)
        inst.s3gen = inst.s3gen.to(DEVICE)
        inst.ve = inst.ve.to(DEVICE)
        inst.device = DEVICE
        # v3 uses grapheme tokenizer (2454 tokens) — matches text_emb weight shape
        inst.tokenizer = EnTokenizer(str(v3_dir / "grapheme_mtl_merged_expanded_v1.json"))
        return inst

    return ChatterboxTTS.from_pretrained(device=DEVICE)

# ── Sentence splitting for long-form synthesis ──────────────────────────────────
_SENTENCE_RE = re.compile(r'(?<=[.!?۔！？])\s+')

def _split_for_tts(text: str, max_chars: int = 250) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Splits at sentence boundaries, grouping 1-3 sentences per chunk up to
    max_chars.  Short chunks are left intact; long sentences that exceed
    max_chars are further split at commas/semicolons.
    """
    raw = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    buf = ""
    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        # Long single sentence → split at weaker boundaries
        if len(sent) > max_chars:
            if buf.strip():
                chunks.append(buf.strip())
                buf = ""
            subs = re.split(r'(?<=[,،；;])\s+', sent)
            sub_buf = ""
            for sub in subs:
                if len(sub_buf) + len(sub) > max_chars and sub_buf:
                    chunks.append(sub_buf.strip())
                    sub_buf = sub
                else:
                    sub_buf += (" " + sub) if sub_buf else sub
            if sub_buf.strip():
                chunks.append(sub_buf.strip())
            continue
        if len(buf) + len(sent) > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf += (" " + sent) if buf else sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _synth_chatterbox(inst, text, params):
    import traceback as _tb

    kw = dict(
        exaggeration=float(params.get("exaggeration", 0.65)),
        cfg_weight=float(params.get("cfg_weight", 0.5)),
        repetition_penalty=float(params.get("repetition_penalty", 1.5)),
    )
    import torch as _torch
    _seed = int(float(params.get("seed", "0") or "0"))
    if _seed != 0:
        _torch.manual_seed(_seed)
        _torch.cuda.manual_seed_all(_seed)
        np.random.seed(_seed)
        slog("SEED", "chatterbox", f"Seed set to {_seed}")
        # CUDA determinism — makes repeated runs with the same seed
        # produce identical audio. Restored per-chunk in _generate_one.
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    pid = params.get("audio_prompt_id")
    if pid:
        p = UPLOAD_DIR / f"{pid}.wav"
        if p.exists():
            kw["audio_prompt_path"] = str(p)

    # ── Text preprocessing (applied once, before splitting) ──
    provider = params.get("use_g2p", "persian_phonemizer")
    text = _process_persian_text(text, provider)

    # ── Persian text normalization (non-destructive, no spacing changes) ──
    # Persian → Western digits: the mTL tokenizer has both, but Western
    # digits are at rank ~265 vs ~1655 — the model knows them 6× better.
    _PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    # Arabic → Persian character variants (unify to Persian script)
    _ARABIC_VARIANTS = str.maketrans("يك", "یک")
    text = text.translate(_PERSIAN_DIGITS)
    text = text.translate(_ARABIC_VARIANTS)

    # Decompose Persian characters missing from the mTL tokenizer (2352 tokens)
    if getattr(inst, "_needs_persian_char_map", False):
        text = text.replace("آ", "آ")  # ALEF MADDA → ALEF + MADDAH ABOVE
        text = text.replace("أ", "ا")   # ALEF HAMZA ABOVE → ALEF
        text = text.replace("إ", "ا")   # ALEF HAMZA BELOW → ALEF

    # ── Chunk long text at sentence boundaries ──
    # Persian fine-tune was trained on short utterances (avg 5-8s).
    # The alignment-stream-analyzer FORCES EOS when it detects token
    # repetition (2× same token).  Longer chunks cause the model to
    # wander into repetition territory → early termination.  Keep each
    # chunk small so the model stays within its training distribution.
    _is_persian = getattr(inst, "_needs_persian_char_map", False)
    _CHUNK_CHARS = 80 if _is_persian else 150
    chunks = _split_for_tts(text, max_chars=_CHUNK_CHARS)
    # Safety net: force-split ANY chunk that exceeds _CHUNK_CHARS × 1.5.
    # This catches comma-split sub-parts that are still too long, and
    # texts with no sentence breaks.  Without this, the model hits
    # token repetition → EOS forced → chunk silently truncated.
    _max_chunk = int(_CHUNK_CHARS * 1.25)  # 100 for persian, 187 for others
    _clean: list[str] = []
    for _ch in chunks:
        if len(_ch) > _max_chunk:
            for _j in range(0, len(_ch), _CHUNK_CHARS):
                _clean.append(_ch[_j:_j + _CHUNK_CHARS])
        else:
            _clean.append(_ch)
    if len(_clean) != len(chunks):
        slog("CHUNK", "chatterbox",
             f"Force-split: {len(chunks)} → {len(_clean)} chunks "
             f"(sizes: {[len(c) for c in _clean]})")
    chunks = _clean
    silence_ms = float(params.get("chunk_silence_ms", 350))
    if silence_ms < 0:
        silence_ms = 0

    # ── Core synthesis helper ──
    def _generate_one(chunk_text: str):
        """Synthesise a single chunk of text. Returns (float32_numpy, sr)."""
        _kw = dict(kw)  # copies exaggeration, cfg_weight, audio_prompt_path
        # Truncate if a single chunk somehow exceeds the token limit
        test_tokens = inst.tokenizer.text_to_tokens(chunk_text)
        if test_tokens.size(0) > 2048:
            test_tokens = test_tokens[:2048]
            ratio = 2048 / test_tokens.size(0)
            chunk_text = chunk_text[:max(1, int(len(chunk_text) * ratio * 1.1))]
        # Patch max_new_tokens (chatterbox hardcodes 1000 internally)
        max_tokens = int(float(params.get("max_length", "20000")))
        _orig_infer = inst.t3.inference
        _old_deterministic = _torch.backends.cudnn.deterministic
        _old_benchmark = _torch.backends.cudnn.benchmark
        def _patched_infer(*a, **_kw2):
            _kw2["max_new_tokens"] = max_tokens
            return _orig_infer(*a, **_kw2)
        inst.t3.inference = _patched_infer
        try:
            if _seed != 0:
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
            wav = inst.generate(chunk_text, **_kw)
        except Exception:
            raise RuntimeError(f"chatterbox generate() failed:\n{_tb.format_exc()}")
        finally:
            _torch.backends.cudnn.deterministic = _old_deterministic
            _torch.backends.cudnn.benchmark = _old_benchmark
            inst.t3.inference = _orig_infer
        return wav.squeeze().cpu().numpy().astype(np.float32), inst.sr

    # ── Single chunk — fast path ──
    if len(chunks) <= 1:
        arr, sr = _generate_one(chunks[0] if chunks else text)
        return _to_wav(arr, sr), sr

    # ── Multi-chunk — synthesise and stitch ──
    slog("CHUNK", "chatterbox",
         f"Splitting {len(text)} chars → {len(chunks)} chunks "
         f"(avg {sum(len(c) for c in chunks)//len(chunks)} chars, {silence_ms}ms gap)")
    all_audio = []
    sr = inst.sr
    _t_start = time.perf_counter()
    _vram_fn = None
    try:
        if DEVICE == "cuda":
            import torch.cuda as _cuda
            _vram_fn = lambda: _cuda.memory_allocated() / (1024**3)
    except Exception:
        pass
    _failures = 0
    _first_error = None
    for i, chunk in enumerate(chunks):
        if _seed != 0:
            _torch.manual_seed(_seed + i)
            _torch.cuda.manual_seed_all(_seed + i)
            np.random.seed(_seed + i)
        _t0 = time.perf_counter()
        _vram_before = _vram_fn() if _vram_fn else 0
        try:
            arr, sr_chunk = _generate_one(chunk)
        except Exception:
            _failures += 1
            if _first_error is None:
                _first_error = _tb.format_exc()
            slog("CHUNK", "chatterbox",
                 f"  [{i+1}/{len(chunks)}] FAILED — {_tb.format_exc()[-150:]}")
            continue
        _t1 = time.perf_counter()
        _vram_after = _vram_fn() if _vram_fn else 0
        sr = sr_chunk or sr
        all_audio.append(arr)
        # Silence gap between chunks (except after the last)
        if i < len(chunks) - 1 and silence_ms > 0:
            gap = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
            all_audio.append(gap)
        _chunk_s = _t1 - _t0
        _chunk_dur = len(arr) / sr
        _elapsed = _t1 - _t_start
        _eta = (_elapsed / (i + 1)) * (len(chunks) - i - 1)
        slog("CHUNK", "chatterbox",
             f"  [{i+1}/{len(chunks)}] {_chunk_s:.1f}s gen → {_chunk_dur:.1f}s audio  "
             f"RTF {_chunk_s/_chunk_dur:.1f}×  "
             f"VRAM {_vram_after:.2f}GB  "
             f"elapsed {_elapsed:.0f}s  ETA {_eta:.0f}s")
    if not all_audio:
        raise RuntimeError(
            f"All {_failures}/{len(chunks)} chunks failed.\n"
            f"First error:\n{_first_error}")
    if _failures:
        slog("CHUNK", "chatterbox",
             f"  ⚠ {_failures}/{len(chunks)} chunks failed — kept {len(all_audio)//2+1} survivors")
    combined = np.concatenate(all_audio)
    total_dur = len(combined) / sr
    slog("CHUNK", "chatterbox",
         f"  Stitched {len(chunks)} chunks → {total_dur:.1f}s ({len(combined)} samples)")
    return _to_wav(combined, sr), sr


# ── 14. Fish Speech ───────────────────────────────────────────────────────────
def _load_fishspeech(model_id="fishaudio/fish-speech-1.5"):
    import torch
    _fs_root = "/opt/models/fish-speech"
    if _fs_root not in sys.path:
        sys.path.insert(0, _fs_root)
    try:
        from fish_speech.models.text2semantic.inference import load_model as _load_llm
        from fish_speech.models.vqgan.inference import load_model as _load_codec
    except ImportError as e:
        raise ImportError(
            f"Fish Speech 1.5.1 code not found: {e}\n"
            f"Run: sudo git clone --depth=1 --branch v1.5.1 "
            f"https://github.com/fishaudio/fish-speech {_fs_root}"
        ) from e
    from huggingface_hub import snapshot_download as _dl
    model_dir = Path(_dl(model_id, ignore_patterns=["*.md", "*.txt", "*.gitignore"]))
    llama_pth = next((p for p in model_dir.glob("model*.pth")), None)
    if llama_pth is None:
        raise FileNotFoundError(f"No model.pth in {model_dir}")
    codec_pth = next((p for p in model_dir.glob("firefly-gan*.pth")), None)
    if codec_pth is None:
        raise FileNotFoundError(f"No firefly-gan*.pth in {model_dir}")
    _precision = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    llm = _load_llm(checkpoint_path=str(model_dir), device=DEVICE,
                    precision=_precision, compile=False)
    _model, _ = llm
    with torch.device(DEVICE):
        _model.setup_caches(max_batch_size=1, max_seq_len=_model.config.max_seq_len,
                            dtype=next(_model.parameters()).dtype)
    decoder = _load_codec(config_name="firefly_gan_vq",
                          checkpoint_path=str(codec_pth), device=DEVICE)
    return {"llm": llm, "decoder": decoder, "precision": _precision}

def _synth_fishspeech(inst, text, params):
    import torch
    from fish_speech.models.text2semantic.inference import generate_long, GenerateResponse
    model, decode_one_token = inst["llm"]
    decoder = inst["decoder"]
    chunks = list(generate_long(
        model=model, device=DEVICE, decode_one_token=decode_one_token,
        text=text, num_samples=1,
        max_new_tokens=int(float(params.get("max_new_tokens", 256))),
        top_p=float(params.get("top_p", 0.7)),
        repetition_penalty=float(params.get("rep_penalty", 1.5)),
        temperature=float(params.get("temperature", 0.7)),
        iterative_prompt=True, chunk_length=100,
    ))
    codes = [c.codes for c in chunks if isinstance(c, GenerateResponse) and c.codes is not None]
    if not codes:
        raise RuntimeError("Fish Speech: generate_long produced no codes")
    codes_t = torch.cat(codes, dim=1)
    indices = codes_t.unsqueeze(0).to(DEVICE)
    feature_lengths = torch.tensor([indices.shape[2]], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        audio_t = decoder.decode(indices=indices, feature_lengths=feature_lengths)
    audio_tensor, _ = audio_t
    audio = audio_tensor[0, 0].cpu().float().numpy()
    sr = getattr(getattr(decoder, "spec_transform", None), "sample_rate", 24000)
    return _to_wav(audio.astype(np.float32), int(sr)), int(sr)


# ── 15. Sesame CSM 1B ─────────────────────────────────────────────────────────
def _load_csm():
    _csm_dir = "/opt/models/csm"
    if _csm_dir not in sys.path:
        sys.path.insert(0, _csm_dir)
    from generator import load_csm_1b
    try:
        return load_csm_1b(device=DEVICE)
    except TypeError as _e:
        if "config" not in str(_e):
            raise
        try:
            import torch
            from generator import Generator
            from models import Model
            _model = Model.from_pretrained("sesame/csm-1b")
            _model = _model.to(device=DEVICE, dtype=torch.bfloat16)
            return Generator(_model)
        except Exception as _e2:
            raise RuntimeError(
                f"CSM load failed with both APIs.\nOriginal: {_e}\nFallback: {_e2}\n"
                "Ensure /opt/models/csm is the latest SesameAILabs/csm clone."
            ) from _e2

def _synth_csm(inst, text, params):
    speaker = int(float(params.get("speaker_id", 0)))
    max_ms  = int(float(params.get("max_audio_length_ms", 30000)))
    audio   = inst.generate(text=text, speaker=speaker, context=[], max_audio_length_ms=max_ms)
    sr      = inst.sample_rate
    arr     = audio.cpu().numpy().flatten().astype(np.float32)
    return _to_wav(arr, sr), sr


# ── 16. Qwen3-TTS ─────────────────────────────────────────────────────────────
def _load_qwen3tts(model_id=QWEN3TTS_MODEL_ID):
    import torch
    from qwen_tts import Qwen3TTSModel
    # qwen_tts forwards **kwargs into AutoModel.from_pretrained — use torch_dtype
    # (the standard transformers key), not dtype, which leaks into __init__.
    # attn_implementation is also not supported by Qwen3TTSForConditionalGeneration.__init__,
    # so omit it and let transformers pick the default (eager/sdpa auto-selected).
    _dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    return Qwen3TTSModel.from_pretrained(model_id, device_map=DEVICE, torch_dtype=_dtype)

def _synth_qwen3tts(inst, text, params):
    ref_id  = params.get("audio_prompt_id", "")
    ref_wav = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id and (UPLOAD_DIR / f"{ref_id}.wav").exists() else None
    ref_txt = params.get("ref_text", "")

    def _float(key, default):
        try:
            v = float(params.get(key, default))
            return v if v != default else None
        except Exception:
            return None

    def _int(key, default):
        try:
            v = int(params.get(key, default))
            return v if v != default else None
        except Exception:
            return None

    gen_kwargs: dict = {}
    for k, d in [("temperature", 0.9), ("top_p", 1.0), ("repetition_penalty", 1.05),
                 ("subtalker_temperature", 0.9), ("subtalker_top_p", 1.0)]:
        v = _float(k, d)
        if v is not None:
            gen_kwargs[k] = v
    for k, d in [("top_k", 50), ("subtalker_top_k", 50), ("max_new_tokens", 2048)]:
        v = _int(k, d)
        if v is not None:
            gen_kwargs[k] = v

    if ref_wav and ref_txt:
        wavs, sr = inst.generate_voice_clone(
            text=text, language=params.get("language", "english"),
            ref_audio=ref_wav, ref_text=ref_txt, **gen_kwargs)
    else:
        instruct = params.get("instruct", "").strip() or None
        wavs, sr = inst.generate_custom_voice(
            text=text, language=params.get("language", "english"),
            speaker=params.get("voice", "aiden"), instruct=instruct, **gen_kwargs)
    return _to_wav(np.array(wavs[0], dtype=np.float32), sr), sr


# ── 17. Orpheus 3B ────────────────────────────────────────────────────────────
def _load_orpheus(model_name="canopylabs/orpheus-3b-0.1-ft"):
    _require_gpu("Orpheus 3B")
    from orpheus_tts import OrpheusModel
    try:
        return OrpheusModel(model_name=model_name)
    except Exception as _e:
        _s = str(_e).lower()
        if "device" in _s or "cuda" in _s or "vllm" in _s or "empty" in _s:
            raise RuntimeError(
                "Orpheus 3B requires a CUDA GPU — vllm cannot run on CPU.\n"
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


# ── 18. NeuTTS Air ────────────────────────────────────────────────────────────
def _load_neutts():
    raise NotImplementedError(
        "NeuTTS Air: package not yet configured.\n"
        "Edit _load_neutts() in tts_lab_engines.py with the correct import after installing."
    )

def _synth_neutts(inst, text, params):
    raise NotImplementedError("NeuTTS Air: configure _load_neutts() first.")


# ── 19. IndexTTS-2 ────────────────────────────────────────────────────────────
def _load_indextts(model_dir=None):
    # Fix ExtensionsTrie removed in transformers 5.x — indextts imports it transitively
    try:
        import transformers.tokenization_utils as _tku, transformers as _tf
        for _cls_name in ["ExtensionsTrie", "AddedToken"]:
            if not hasattr(_tku, _cls_name) or not isinstance(getattr(_tku, _cls_name, None), type):
                _stub = type(_cls_name, (), {"__init__": lambda s,*a,**kw: None})
                setattr(_tku, _cls_name, _stub)
                setattr(_tf, _cls_name, _stub)
    except Exception:
        pass

    _REMOVED_CACHE_CLASSES = [
        "OffloadedCache", "QuantizedCacheConfig", "QuantizedCache",
        "QuantoQuantizedCache", "HQQQuantizedCache", "SlidingWindowCache", "StaticCacheConfig",
    ]
    try:
        import transformers.cache_utils as _cu, transformers as _tf
        for _cls_name in _REMOVED_CACHE_CLASSES:
            if not hasattr(_cu, _cls_name):
                _stub = type(_cls_name, (), {})
                setattr(_cu, _cls_name, _stub)
            if not hasattr(_tf, _cls_name):
                setattr(_tf, _cls_name, getattr(_cu, _cls_name))
    except Exception:
        pass
    from indextts.infer_v2 import IndexTTS2 as IndexTTS
    from huggingface_hub import snapshot_download as _dl
    if model_dir:
        md = model_dir
    elif INDEXTTS_DIR.exists():
        md = str(INDEXTTS_DIR)
    else:
        md = _dl("IndexTeam/IndexTTS-2", ignore_patterns=["*.md", "*.txt"])
    cfg = str(Path(md) / "config.yaml")
    # IndexTTS2 loads all weights in __init__ — no separate load_model() call needed
    model = IndexTTS(cfg_path=cfg, model_dir=md, device=DEVICE)
    return model

def _synth_indextts(inst, text, params):
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id else None
    if not ref_path or not Path(ref_path).exists():
        raise RuntimeError(
            "IndexTTS-2 requires a reference WAV. Upload a 5-30s clip first."
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


# ── 20. Zonos v0.1 ────────────────────────────────────────────────────────────
def _load_zonos(variant="transformer"):
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
    wav_t = inst.autoencoder.decode(codes)
    sr    = inst.autoencoder.sampling_rate
    arr   = wav_t[0].squeeze().cpu().numpy().astype(np.float32)
    return _to_wav(arr, sr), sr


# ── 21. OpenVoice v2 ─────────────────────────────────────────────────────────
def _load_openvoice():
    import torch
    if "wavmark" not in sys.modules:
        import types as _t
        _wm = _t.ModuleType("wavmark")
        _wm.load_model = lambda: type("_NoopWM", (), {"to": lambda s, d: None})()
        sys.modules["wavmark"] = _wm
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS
    ckpt_dir = OPENVOICE_MODELS_DIR / "converter"
    if not (ckpt_dir / "config.json").exists():
        raise FileNotFoundError(
            f"OpenVoice checkpoints missing at {OPENVOICE_MODELS_DIR}.\n"
            f"Run: sudo ln -sfn <hf_snapshot>/checkpoints /opt/models/openvoice_v2"
        )
    converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=DEVICE)
    converter.watermark_model = None
    # Some OpenVoice builds initialise sub-modules on the "meta" device then
    # call .to(device) which raises "Cannot copy out of meta tensor".
    # Migrate any meta parameters to empty tensors first.
    if hasattr(converter, "model") and converter.model is not None:
        for m in converter.model.modules():
            for name, param in list(m.named_parameters(recurse=False)):
                if param.is_meta:
                    setattr(m, name, torch.nn.Parameter(
                        torch.empty_like(param, device=DEVICE)))
            for name, buf in list(m.named_buffers(recurse=False)):
                if buf.is_meta:
                    m.register_buffer(name, torch.empty_like(buf, device=DEVICE))
    converter.load_ckpt(str(ckpt_dir / "checkpoint.pth"))
    base_tts = MeloTTS(language="EN", device=DEVICE)
    base_se: dict = {}
    ses_dir = OPENVOICE_MODELS_DIR / "base_speakers" / "ses"
    en_dir  = OPENVOICE_MODELS_DIR / "base_speakers" / "EN"
    if ses_dir.exists():
        for p in ses_dir.glob("*.pth"):
            t = torch.load(str(p), map_location=DEVICE, weights_only=False)
            base_se[p.stem] = t.to(DEVICE) if hasattr(t, "to") else t
    elif en_dir.exists():
        for fname, key in [("en_default_se.pth","en_default"), ("en_style_se.pth","en_style")]:
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
    spk_key   = params.get("speaker", "EN-US")
    sp_ids    = dict(base_tts.hps.data.spk2id)
    sp_id     = sp_ids.get(spk_key) or sp_ids.get("EN-US") or list(sp_ids.values())[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        src_tmp = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_tmp = f.name
    try:
        try:
            base_tts.tts_to_file(text, sp_id, src_tmp, speed=float(params.get("speed", 0.85)))
        except (ValueError, LookupError) as _e:
            if any(k in str(_e) for k in ("averaged_perceptron_tagger", "broadcast", "nltk")):
                raise RuntimeError(
                    "OpenVoice/MeloTTS: NLTK tagger missing for root user.\n"
                    "Fix: sudo python3 -c \"import nltk; "
                    "nltk.download('averaged_perceptron_tagger_eng',"
                    "download_dir='/usr/share/nltk_data')\""
                ) from _e
            raise
        src_size = Path(src_tmp).stat().st_size
        if src_size <= 44:
            raise RuntimeError(
                f"OpenVoice base TTS produced no audio ({src_size}B). "
                "Try longer text or a different speaker."
            )
        se_key = spk_key.lower().replace("-", "_")
        src_se = base_se.get(se_key) or base_se.get("en_us") or (list(base_se.values())[0] if base_se else None)
        ref_id   = params.get("audio_prompt_id", "")
        ref_path = UPLOAD_DIR / f"{ref_id}.wav" if ref_id else None
        target_se = src_se
        if ref_path and ref_path.exists():
            try:
                from openvoice import se_extractor
                _se, _ = se_extractor.get_se(str(ref_path), converter, vad=True)
                if _se is not None and _se.numel() > 0:
                    target_se = _se
            except Exception:
                pass
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


# ── 22. Matcha-TTS ────────────────────────────────────────────────────────────
# Fast flow-matching TTS via sherpa-onnx. Persian + English, 2 voices.
# Models auto-download from HuggingFace on first load.

def _load_matcha(voice="khadijah"):
    import sherpa_onnx
    from huggingface_hub import snapshot_download as _dl
    from tts_lab_config import MATCHA_MODEL_REPOS, MATCHA_VOCODER_REPO, MATCHA_VOCODER_FILE

    if voice not in MATCHA_MODEL_REPOS:
        raise ValueError(
            f"Unknown matcha voice: {voice!r}. Options: {list(MATCHA_MODEL_REPOS)}"
        )

    repo_id     = MATCHA_MODEL_REPOS[voice]
    model_dir   = Path(_dl(repo_id))
    model_path  = str(model_dir / "model.onnx")
    tokens_path = str(model_dir / "tokens.txt")
    data_dir    = str(model_dir / "espeak-ng-data")

    # HiFi-GAN vocoder (separate, shared across voices)
    vocoder_dir  = Path(_dl(MATCHA_VOCODER_REPO))
    vocoder_path = str(vocoder_dir / MATCHA_VOCODER_FILE)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"model.onnx not found in {model_dir}")
    if not Path(vocoder_path).exists():
        raise FileNotFoundError(f"{MATCHA_VOCODER_FILE} not found in {vocoder_dir}")

    # sherpa-onnx Matcha config
    matcha_cfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
        acoustic_model=model_path,
        vocoder=vocoder_path,
        tokens=tokens_path,
        data_dir=data_dir,
        noise_scale=0.333,
        length_scale=1.0,
    )
    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        matcha=matcha_cfg,
        num_threads=_N_CORES,
        provider="cuda" if DEVICE == "cuda" else "cpu",
        debug=False,
    )
    tts_cfg = sherpa_onnx.OfflineTtsConfig(
        model=model_cfg,
        max_num_sentences=1,
    )
    if not tts_cfg.validate():
        raise RuntimeError("Matcha-TTS: OfflineTtsConfig validation failed")
    return sherpa_onnx.OfflineTts(tts_cfg)


def _synth_matcha(inst, text, params):
    speed       = float(params.get("speed", 1.0))
    temperature = float(params.get("temperature", 0.333))

    # Rebuild engine if temperature differs from what was loaded
    # (noise_scale is fixed at construction time in sherpa-onnx).
    # _ensure_loaded handles the eviction — we just generate here.
    result = inst.generate(
        text,
        sid=0,
        speed=speed,
    )
    # sherpa-onnx returns GeneratedAudio with .samples (np.array) and .sample_rate
    return _to_wav(result.samples, int(result.sample_rate)), int(result.sample_rate)


# ── 23. ManaTTS ───────────────────────────────────────────────────────────────
# Tacotron v1 (SV2TTS pipeline) + HiFi-GAN vocoder. Persian-only.
# Requires reference WAV for speaker embedding extraction.

# ── Persian text processing (G2P + normalization) ───────────────────────────
# Cache singletons — created on first use
_g2p_cache: dict = {}

def _process_persian_text(text: str, provider: str) -> str:
    """Process Persian text with the selected provider.

    Providers:
      persian_phonemizer — G2P: adds vowel marks (dictionary + neural)
      hazm              — Normalize only: Arabic→Persian chars, ZWNJ, spacing (NO vowel marks)
      parsivar          — Normalize only: chars + digits→ASCII + spacing (NO vowel marks)
      none              — Raw text, no processing
    """
    provider = (provider or "none").strip().lower()

    # Map legacy boolean values
    if provider in ("1", "true", "yes", "on"):
        provider = "persian_phonemizer"
    elif provider in ("0", "false", "no", "off", ""):
        provider = "none"

    if provider == "none":
        return text

    if provider == "persian_phonemizer":
        try:
            from persian_phonemizer import Phonemizer
            if "phonemizer" not in _g2p_cache:
                _g2p_cache["phonemizer"] = Phonemizer(output_format='eraab')
            return _g2p_cache["phonemizer"].phonemize(text)
        except Exception:
            return text

    if provider == "hazm":
        try:
            from hazm import Normalizer
            if "hazm" not in _g2p_cache:
                _g2p_cache["hazm"] = Normalizer()
            return _g2p_cache["hazm"].normalize(text)
        except Exception:
            return text

    if provider == "parsivar":
        try:
            from parsivar import Normalizer
            if "parsivar" not in _g2p_cache:
                _g2p_cache["parsivar"] = Normalizer(
                    statistical_space_correction=True,
                    date_normalizing_needed=True,
                )
            return _g2p_cache["parsivar"].normalize(text)
        except Exception:
            return text

    return text


def _normalize_persian_text(text: str) -> str:
    """Normalize Persian text for Tacotron2's limited character vocabulary.

    Converts Persian/Arabic digits to ASCII, Arabic chars to Persian equivalents,
    and strips characters not in the model's symbol set.
    """
    # Persian digits → ASCII
    persian_digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    text = text.translate(persian_digits)

    # Arabic chars → Persian equivalents
    arabic_to_persian = str.maketrans({
        "ك": "ک",  # Arabic kaf → Persian kaf
        "ي": "ی",  # Arabic ye → Persian ye
        "ة": "ه",  # Arabic ta marbuta → Persian he
        "ؤ": "و",  # Arabic waw with hamza → vav
        "أ": "ا",  # Arabic alef with hamza → alef
        "إ": "ا",  # Arabic alef with hamza below → alef
        "ئ": "ی",  # Arabic ye with hamza → ye
        "ء": "",   # Remove hamza
    })
    text = text.translate(arabic_to_persian)

    # Strip characters outside the model's known symbol set
    from synthesizer.persian_utils.symbols import _characters
    allowed = set(_characters + " _pad_eos~")
    text = "".join(c for c in text if c in allowed or c.isspace())

    return text.strip()


def _split_persian_text(text: str, max_chars: int = 200) -> list:
    """Split Persian text into chunks respecting sentence boundaries."""
    import re
    text = _normalize_persian_text(text)
    if not text:
        return []
    sentences = re.split(r'(?<=[.?!؟])\s+|(?<=[.?!؟])$', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks or [text.strip()]


def _load_manatts():
    from huggingface_hub import snapshot_download as _dl
    from tts_lab_config import MANATTS_REPO_DIR, MANATTS_MODEL_REPO

    # 1. Verify implementation repo
    if not MANATTS_REPO_DIR.exists():
        raise FileNotFoundError(
            f"ManaTTS repo not found at {MANATTS_REPO_DIR}.\n"
            "Clone: git clone https://github.com/MahtaFetrat/Persian-MultiSpeaker-Tacotron2 "
            f"{MANATTS_REPO_DIR}"
        )
    if str(MANATTS_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(MANATTS_REPO_DIR))

    # 2. Import the cloned repo's inference modules
    from encoder import inference as encoder_mod
    from synthesizer.inference import Synthesizer

    # 3. Download model checkpoints from HuggingFace
    model_dir = Path(_dl(MANATTS_MODEL_REPO, allow_patterns=["*.pt", "*.pth"]))

    # 4. Load encoder
    encoder_path = model_dir / "encoder.pt"
    if not encoder_path.exists():
        # encoder.pt may be bundled in the cloned repo
        encoder_path = MANATTS_REPO_DIR / "saved_models" / "default" / "encoder.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(
            "encoder.pt not found. Download it and place in the models directory."
        )
    encoder_mod.load_model(str(encoder_path))

    # 5. Load synthesizer
    synth_path = model_dir / "synthesizer.pt"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"synthesizer.pt not found in {model_dir}.\n"
            f"Download from: https://huggingface.co/{MANATTS_MODEL_REPO}"
        )
    synthesizer = Synthesizer(str(synth_path))

    # 6. Load HiFi-GAN vocoder (916MB, downloaded separately)
    vocoder = None
    vocoder_paths = [
        Path("/opt/models/manatts-vocoder/vctk_hifigan.v1/checkpoint-2500000steps.pkl"),
        model_dir / "vocoder_HiFiGAN.pkl",
    ]
    for vp in vocoder_paths:
        if vp.exists():
            try:
                from parallel_wavegan.utils import load_model as load_pwg
                vocoder = load_pwg(str(vp))
                vocoder.remove_weight_norm()
                vocoder = vocoder.eval().to(DEVICE)
                break
            except Exception:
                continue

    return {
        "synthesizer": synthesizer,
        "encoder": encoder_mod,
        "vocoder": vocoder,
        "sr": int(Synthesizer.sample_rate),
    }


def _synth_manatts(inst, text, params):
    import torch
    from tts_lab_config import UPLOAD_DIR, MANATTS_MAX_CHARS
    from synthesizer.inference import Synthesizer

    # 1. Text processing: G2P diacritics or normalization (hazm/parsivar)
    text = _process_persian_text(text, params.get("use_g2p", "persian_phonemizer"))

    # 2. Validate reference audio
    ref_id   = params.get("audio_prompt_id", "")
    ref_path = UPLOAD_DIR / f"{ref_id}.wav" if ref_id else None
    if not ref_path or not ref_path.exists():
        raise RuntimeError(
            "ManaTTS requires a reference WAV for speaker embedding.\n"
            "Upload a 3-10s WAV of the target speaker first."
        )

    encoder    = inst["encoder"]
    synthesizer = inst["synthesizer"]
    vocoder    = inst["vocoder"]

    # 2. Load and preprocess reference WAV
    wav = Synthesizer.load_preprocess_wav(str(ref_path))
    encoder_wav = encoder.preprocess_wav(wav)
    embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # 3. Split text and synthesize
    chunks = _split_persian_text(text, max_chars=MANATTS_MAX_CHARS)
    embeds = [embed] * len(chunks)
    specs = synthesizer.synthesize_spectrograms(chunks, embeds)
    spec  = np.concatenate(specs, axis=1)

    # 4. Vocode
    if vocoder is not None:
        x = torch.from_numpy(spec.T).to(DEVICE)
        with torch.no_grad():
            wav_out = vocoder.inference(x)
        wav_out = wav_out.squeeze().cpu().numpy().astype(np.float32)
    else:
        import librosa
        wav_out = librosa.feature.inverse.mel_to_audio(
            spec, sr=inst["sr"]
        ).astype(np.float32)

    wav_out = wav_out / np.abs(wav_out).max() * 0.97
    return _to_wav(wav_out, inst["sr"]), inst["sr"]


# ── 24. Chatterbox-Turbo ───────────────────────────────────────────────────────
def _load_chatterboxturbo(model="default"):
    """Load Chatterbox-Turbo TTS (350M distilled one-step model).

    Same package as Chatterbox but uses ChatterboxTurboTTS class.
    Models:
      default — English Turbo (350M, one-step decoder)
      turbo    — Same as default (alias)
    """
    import types, importlib.machinery
    for _tc in ["torchcodec", "torchcodec._C", "torchcodec.decoders",
                "torchcodec.decoders._core", "torchcodec.decoders.video_decoder",
                "torchcodec.encoders"]:
        if _tc not in sys.modules:
            _m = types.ModuleType(_tc)
            _is_pkg = "." not in _tc.split("torchcodec.")[-1] or _tc == "torchcodec"
            _m.__spec__ = importlib.machinery.ModuleSpec(_tc, loader=None, is_package=_is_pkg)
            _m.__path__ = []
            sys.modules[_tc] = _m
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    slog("LOAD", "chatterboxturbo", f"Loading ChatterboxTurboTTS (350M, one-step)...")
    return ChatterboxTurboTTS.from_pretrained(device=DEVICE)


def _synth_chatterboxturbo(inst, text, params):
    import torch

    # Core generation parameters
    kw = dict(
        exaggeration=float(params.get("exaggeration", 0.5)),
        cfg_weight=float(params.get("cfg_weight", 0.5)),
    )

    # Optional generation parameters
    def _maybe_float(key, default=None):
        v = params.get(key, "")
        if v != "" and v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
        return default

    def _maybe_int(key, default=None):
        v = params.get(key, "")
        if v != "" and v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
        return default

    temperature = _maybe_float("temperature")
    if temperature is not None:
        kw["temperature"] = temperature

    top_p = _maybe_float("top_p")
    if top_p is not None:
        kw["top_p"] = top_p

    top_k = _maybe_int("top_k")
    if top_k is not None:
        kw["top_k"] = top_k

    repetition_penalty = _maybe_float("repetition_penalty")
    if repetition_penalty is not None:
        kw["repetition_penalty"] = repetition_penalty

    min_p = _maybe_float("min_p")
    if min_p is not None:
        kw["min_p"] = min_p

    norm_loudness = params.get("norm_loudness", "true").lower()
    if norm_loudness in ("false", "0", "no"):
        kw["norm_loudness"] = False

    # Voice cloning reference WAV
    pid = params.get("audio_prompt_id")
    if pid:
        p = UPLOAD_DIR / f"{pid}.wav"
        if p.exists():
            kw["audio_prompt_path"] = str(p)

    # Seed (via torch global state — not a generate() parameter)
    _seed = int(float(params.get("seed", "0") or "0"))
    if _seed != 0:
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        np.random.seed(_seed)

    # Persian text processing (reuse from chatterbox)
    provider = params.get("use_g2p", "none")
    text = _process_persian_text(text, provider)

    wav = inst.generate(text, **kw)
    arr = wav.squeeze().cpu().numpy().astype(np.float32)
    return _to_wav(arr, inst.sr), inst.sr


# ── 25. Microsoft VibeVoice-1.5B ───────────────────────────────────────────────
def _load_vibevoice():
    """Load Microsoft VibeVoice-1.5B (next-token diffusion, 3B params).

    English + Chinese only. Built-in AI disclaimer + watermark.

    Serving: VibeVoice has no Python code in its HF repo (no trust_remote_code),
    and transformers 5.x doesn't natively support 'vibevoice' architecture yet.
    Use SGLang-Omni server — the recommended serving path per the model card.

    Setup:
      1. docker pull lmsysorg/sglang-omni:dev
      2. docker run --gpus all -p 8000:8000 lmsysorg/sglang-omni:dev \\
           --model microsoft/VibeVoice-1.5B
      3. Set VIBEVOICE_SGLANG_URL=http://host:8000/v1/audio/speech
    """
    slog("LOAD", "vibevoice", "VibeVoice-1.5B — using SGLang-Omni endpoint")
    return {
        "sglang_url": os.environ.get("VIBEVOICE_SGLANG_URL", "http://localhost:8000/v1/audio/speech"),
        "sr": 24000,
    }


def _synth_vibevoice(inst, text, params):
    """Synthesise via VibeVoice SGLang-Omni server."""
    import requests
    url = inst["sglang_url"]
    payload = {"input": text}
    try:
        resp = requests.post(url, json=payload, timeout=float(params.get("timeout", 300)))
        resp.raise_for_status()
        wav_bytes = resp.content
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _to_wav(arr, sr), sr
    except Exception as e:
        raise RuntimeError(
            f"VibeVoice synthesis failed — is SGLang-Omni server running at {url}?\n"
            f"Start: docker run --gpus all -p 8000:8000 lmsysorg/sglang-omni:dev "
            f"--model microsoft/VibeVoice-1.5B\n"
            f"Or set VIBEVOICE_SGLANG_URL env var.\n"
            f"Error: {e}"
        )


# ── 26. BosonAI Higgs Audio v3 (4B) ────────────────────────────────────────────
def _load_higgs():
    """Load BosonAI Higgs Audio v3 TTS (4B AR decoder, 100+ languages).

    Supports voice cloning, control tokens, 102 languages including Persian.

    Serving: Higgs has no Python code in its HF repo (no trust_remote_code),
    and transformers 5.x doesn't natively support 'higgs_multimodal_qwen3'.
    Use SGLang-Omni or vLLM-Omni — the recommended serving paths.

    Setup (SGLang-Omni):
      1. docker pull lmsysorg/sglang-omni:dev
      2. docker run --gpus all -p 8000:8000 lmsysorg/sglang-omni:dev \\
           --model bosonai/higgs-audio-v3-tts-4b
      3. Set HIGGS_SGLANG_URL=http://host:8000/v1/audio/speech

    Setup (vLLM-Omni):
      1. pip install vllm-omni
      2. vllm-omni serve bosonai/higgs-audio-v3-tts-4b
    """
    slog("LOAD", "higgs", "Higgs Audio v3 — using SGLang-Omni endpoint")
    return {
        "sglang_url": os.environ.get("HIGGS_SGLANG_URL", "http://localhost:8000/v1/audio/speech"),
        "sr": 24000,
    }


def _synth_higgs(inst, text, params):
    """Synthesise via Higgs SGLang-Omni server.

    Supports inline control tokens: <|emotion:happy|>, <|sfx:laugh|laugh|>, etc.
    """
    import requests
    url = inst["sglang_url"]
    payload = {
        "input": text,
        "temperature": float(params.get("temperature", 0.8)),
        "top_k": int(params.get("top_k", 50)),
        "max_new_tokens": int(params.get("max_new_tokens", 1024)),
    }
    # Optional voice cloning references
    ref_id = params.get("audio_prompt_id", "")
    if ref_id and (UPLOAD_DIR / f"{ref_id}.wav").exists():
        ref_obj = {"audio_path": str(UPLOAD_DIR / f"{ref_id}.wav")}
        ref_txt = params.get("ref_text", "").strip()
        if ref_txt:
            ref_obj["text"] = ref_txt
        payload["references"] = [ref_obj]

    try:
        resp = requests.post(url, json=payload, timeout=float(params.get("timeout", 300)))
        resp.raise_for_status()
        wav_bytes = resp.content
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _to_wav(arr, sr), sr
    except Exception as e:
        raise RuntimeError(
            f"Higgs synthesis failed — is SGLang-Omni server running at {url}?\n"
            f"Start: docker run --gpus all -p 8000:8000 lmsysorg/sglang-omni:dev "
            f"--model bosonai/higgs-audio-v3-tts-4b\n"
            f"Or set HIGGS_SGLANG_URL env var.\n"
            f"Error: {e}"
        )


# ── 27. K2-FSA OmniVoice (0.6B) ───────────────────────────────────────────────
def _load_omnivoice():
    """Load K2-FSA OmniVoice (0.6B diffusion LM, 600+ languages).

    Zero-shot voice cloning + voice design.
    Apache-2.0 license.
    """
    import torch as _torch
    from omnivoice import OmniVoice
    _dtype = _torch.bfloat16 if DEVICE == "cuda" else _torch.float32
    slog("LOAD", "omnivoice", "Loading OmniVoice (~1.2 GB, 0.6B params, 600+ langs)...")
    return OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=DEVICE if DEVICE == "cuda" else "cpu",
        dtype=_dtype,
    )


def _synth_omnivoice(inst, text, params):
    # Language — critical for Persian/Farsi support (600+ languages)
    language = params.get("language", "").strip()
    gen_kw = {}
    if language:
        gen_kw["language"] = language

    # Voice cloning: reference audio + optional transcript
    ref_id = params.get("audio_prompt_id", "")
    ref_wav = str(UPLOAD_DIR / f"{ref_id}.wav") if ref_id and (UPLOAD_DIR / f"{ref_id}.wav").exists() else None
    ref_txt = params.get("ref_text", "").strip() or None

    if ref_wav:
        gen_kw["ref_audio"] = ref_wav
        if ref_txt:
            gen_kw["ref_text"] = ref_txt
    elif ref_txt:
        # voice_clone_prompt: text-only clone (no reference audio)
        gen_kw["voice_clone_prompt"] = ref_txt

    # Voice design / style instruction (no reference audio needed)
    instruct = params.get("instruct", "").strip()
    if instruct:
        gen_kw["instruct"] = instruct

    # Speed control
    speed_str = params.get("speed", "").strip()
    if speed_str:
        try:
            gen_kw["speed"] = float(speed_str)
        except (ValueError, TypeError):
            pass

    # Duration hint (seconds)
    dur_str = params.get("duration", "").strip()
    if dur_str:
        try:
            gen_kw["duration"] = float(dur_str)
        except (ValueError, TypeError):
            pass

    audio = inst.generate(text=text, **gen_kw)
    # audio is list of np.ndarray with shape (T,) at 24 kHz
    if isinstance(audio, list):
        arr = audio[0]
    else:
        arr = audio
    arr = np.array(arr, dtype=np.float32).flatten()
    return _to_wav(arr, 24000), 24000


# ── 28. Fish Audio S2-Pro (5B) ────────────────────────────────────────────────
def _load_s2pro():
    """Fish Audio S2-Pro (Dual-AR 5B, 80+ languages).

    S2-Pro requires SGLang for streaming inference.
    Setup instructions:
      1. Install SGLang: pip install sglang[all]
      2. Launch server: python -m sglang.launch_server \\
           --model fishaudio/s2-pro --tp 1
      3. Then use the OpenAI-compatible /v1/audio/speech endpoint.

    This stub returns a placeholder; edit _load_s2pro() and _synth_s2pro()
    to point at your SGLang server.
    """
    slog("LOAD", "s2pro", "S2-Pro loaded as stub — configure SGLang server endpoint")
    return {
        "sglang_url": os.environ.get("S2PRO_SGLANG_URL", "http://localhost:8000/v1/audio/speech"),
        "sr": 44100,
    }


def _synth_s2pro(inst, text, params):
    """Synthesise via S2-Pro SGLang server (OpenAI-compatible /v1/audio/speech).

    Set S2PRO_SGLANG_URL env var to point at your SGLang server.
    """
    import requests, base64 as _b64
    url = inst["sglang_url"]
    payload = {
        "input": text,
        "voice": params.get("voice", "default"),
    }
    # Optional: add control tags via inline [tag] syntax in text
    try:
        resp = requests.post(url, json=payload, timeout=float(params.get("timeout", 300)))
        resp.raise_for_status()
        # Response is raw WAV bytes
        wav_bytes = resp.content
        # Parse WAV to get numpy array + sr
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _to_wav(arr, sr), sr
    except Exception as e:
        raise RuntimeError(
            f"S2-Pro synthesis failed — is SGLang server running at {url}?\n"
            f"Start it with: python -m sglang.launch_server --model fishaudio/s2-pro\n"
            f"Error: {e}"
        )


# ── Dispatch tables ───────────────────────────────────────────────────────────
LOADERS: dict = {
    "piper":      _load_piper,    "kokoro":    _load_kokoro,
    "melo":       _load_melo,     "matcha":    _load_matcha,
    "chattts":    _load_chattts,  "outetts":   _load_outetts,
    "bark":       _load_bark,     "styletts2": _load_styletts2,
    "f5tts":      _load_f5tts,    "dia":       _load_dia,
    "xtts":       _load_xtts,     "cosyvoice": _load_cosyvoice,
    "parler":     _load_parler,   "chatterbox":_load_chatterbox,
    "chatterboxturbo": _load_chatterboxturbo,
    "fishspeech": _load_fishspeech,"csm":      _load_csm,
    "qwen3tts":   _load_qwen3tts, "orpheus":   _load_orpheus,
    "neutts":     _load_neutts,   "indextts":  _load_indextts,
    "manatts":    _load_manatts,  "zonos":     _load_zonos,
    "openvoice":  _load_openvoice,
    "vibevoice":  _load_vibevoice,"higgs":     _load_higgs,
    "omnivoice":  _load_omnivoice,"s2pro":     _load_s2pro,
}
SYNTHERS: dict = {
    "piper":      _synth_piper,    "kokoro":    _synth_kokoro,
    "melo":       _synth_melo,     "matcha":    _synth_matcha,
    "chattts":    _synth_chattts,  "outetts":   _synth_outetts,
    "bark":       _synth_bark,     "styletts2": _synth_styletts2,
    "f5tts":      _synth_f5tts,    "dia":       _synth_dia,
    "xtts":       _synth_xtts,     "cosyvoice": _synth_cosyvoice,
    "parler":     _synth_parler,   "chatterbox":_synth_chatterbox,
    "chatterboxturbo": _synth_chatterboxturbo,
    "fishspeech": _synth_fishspeech,"csm":      _synth_csm,
    "qwen3tts":   _synth_qwen3tts, "orpheus":   _synth_orpheus,
    "neutts":     _synth_neutts,   "indextts":  _synth_indextts,
    "manatts":    _synth_manatts,  "zonos":     _synth_zonos,
    "openvoice":  _synth_openvoice,
    "vibevoice":  _synth_vibevoice,"higgs":     _synth_higgs,
    "omnivoice":  _synth_omnivoice,"s2pro":     _synth_s2pro,
}
