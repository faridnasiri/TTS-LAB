"""
tts_lab_engines.py — all 21 TTS engine loader + synth pairs.

Each engine exposes:
    _load_<name>([params])  →  instance
    _synth_<name>(inst, text, params)  →  (wav_bytes, sample_rate)

Bottom of file: LOADERS and SYNTHERS dicts used by tts_lab_dispatch.
"""
from __future__ import annotations
import io, os, sys, tempfile, wave
import numpy as np
from pathlib import Path

from tts_lab_shims  import _N_CORES, DEVICE
from tts_lab_config import (
    MODELS_DIR, COSYVOICE_DIR, UPLOAD_DIR, INDEXTTS_DIR, OPENVOICE_MODELS_DIR,
    OUTETTS_DEFAULT_GGUF, OUTETTS_DEFAULT_TOKENIZER, QWEN3TTS_MODEL_ID,
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
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    from transformers.generation.configuration_utils import GenerationConfig
    # parler_tts uses _pad_token_tensor/_bos_token_tensor/_eos_token_tensor which were
    # removed from transformers GenerationConfig after 4.46. Shim them back as properties.
    if not hasattr(GenerationConfig, "_pad_token_tensor"):
        def _make_tensor_prop(id_attr):
            def _prop(self):
                val = getattr(self, id_attr, None)
                return torch.tensor(val) if val is not None else None
            return property(_prop)
        GenerationConfig._pad_token_tensor = _make_tensor_prop("pad_token_id")
        GenerationConfig._bos_token_tensor = _make_tensor_prop("bos_token_id")
        GenerationConfig._eos_token_tensor = _make_tensor_prop("eos_token_id")
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
def _load_chatterbox():
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
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=DEVICE)

def _synth_chatterbox(inst, text, params):
    kw = dict(
        exaggeration=float(params.get("exaggeration", 0.65)),
        cfg_weight=float(params.get("cfg_weight", 0.5)),
    )
    seed = params.get("seed")
    if seed and int(float(seed)) != 0:
        kw["seed"] = int(float(seed))
    pid = params.get("audio_prompt_id")
    if pid:
        p = UPLOAD_DIR / f"{pid}.wav"
        if p.exists():
            kw["audio_prompt_path"] = str(p)
    wav = inst.generate(text, **kw)
    arr = wav.squeeze().cpu().numpy().astype(np.float32)
    return _to_wav(arr, inst.sr), inst.sr


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
        max_new_tokens=int(float(params.get("max_new_tokens", 1024))),
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
    _dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    _attn  = "sdpa" if DEVICE == "cuda" else "eager"
    return Qwen3TTSModel.from_pretrained(model_id, device_map=DEVICE, dtype=_dtype,
                                         attn_implementation=_attn)

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
    from indextts.infer_v2 import IndexTTS
    from huggingface_hub import snapshot_download as _dl
    if model_dir:
        md = model_dir
    elif INDEXTTS_DIR.exists():
        md = str(INDEXTTS_DIR)
    else:
        md = _dl("IndexTeam/IndexTTS-2", ignore_patterns=["*.md", "*.txt"])
    cfg = str(Path(md) / "config.yaml")
    model = IndexTTS(cfg_path=cfg, model_dir=md, device=DEVICE)
    model.load_model()
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


# ── Dispatch tables ───────────────────────────────────────────────────────────
LOADERS: dict = {
    "piper":      _load_piper,    "kokoro":    _load_kokoro,
    "melo":       _load_melo,     "chattts":   _load_chattts,
    "outetts":    _load_outetts,  "bark":      _load_bark,
    "styletts2":  _load_styletts2,"f5tts":     _load_f5tts,
    "dia":        _load_dia,      "xtts":      _load_xtts,
    "cosyvoice":  _load_cosyvoice,"parler":    _load_parler,
    "chatterbox": _load_chatterbox,
    "fishspeech": _load_fishspeech,"csm":      _load_csm,
    "qwen3tts":   _load_qwen3tts, "orpheus":   _load_orpheus,
    "neutts":     _load_neutts,   "indextts":  _load_indextts,
    "zonos":      _load_zonos,    "openvoice": _load_openvoice,
}
SYNTHERS: dict = {
    "piper":      _synth_piper,    "kokoro":    _synth_kokoro,
    "melo":       _synth_melo,     "chattts":   _synth_chattts,
    "outetts":    _synth_outetts,  "bark":      _synth_bark,
    "styletts2":  _synth_styletts2,"f5tts":     _synth_f5tts,
    "dia":        _synth_dia,      "xtts":      _synth_xtts,
    "cosyvoice":  _synth_cosyvoice,"parler":    _synth_parler,
    "chatterbox": _synth_chatterbox,
    "fishspeech": _synth_fishspeech,"csm":      _synth_csm,
    "qwen3tts":   _synth_qwen3tts, "orpheus":   _synth_orpheus,
    "neutts":     _synth_neutts,   "indextts":  _synth_indextts,
    "zonos":      _synth_zonos,    "openvoice": _synth_openvoice,
}
