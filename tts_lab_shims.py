"""
tts_lab_shims.py — startup-time compatibility patches.

Must be imported FIRST in tts_lab.py, before any ML library.
Exports: _N_CORES, DEVICE, DEVICE_NAME, VRAM_TOTAL_MB
"""
from __future__ import annotations
import os, sys, types

# ── Thread-pool pinning (before torch / ORT load so the env vars take effect) ─
_N_CORES = os.cpu_count() or 6
os.environ.setdefault("OMP_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("MKL_NUM_THREADS",      str(_N_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(_N_CORES))
os.environ.setdefault("ORT_NUM_THREADS",      str(_N_CORES))
# Bark model cache on data disk
os.environ.setdefault("XDG_CACHE_HOME",       "/opt/models/cache")
os.environ.setdefault("SUNO_USE_SMALL_MODELS","False")   # full Bark — 16 GB VRAM

# ── Torch + DEVICE ────────────────────────────────────────────────────────────
try:
    import torch
    torch.set_num_threads(_N_CORES)
    torch.set_num_interop_threads(max(1, _N_CORES // 2))
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME   = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    VRAM_TOTAL_MB = int(torch.cuda.get_device_properties(0).total_memory / 1048576) if DEVICE == "cuda" else 0
except Exception:
    DEVICE = "cpu"; DEVICE_NAME = "CPU"; VRAM_TOTAL_MB = 0

# ── Transformers 5.x compatibility shims ─────────────────────────────────────
# Applied at startup before any TTS import so every engine sees a clean namespace.
try:
    import transformers.pytorch_utils as _tpu
    import transformers as _tf
    import torch as _torch

    # isin_mps_friendly: removed from pytorch_utils in transformers 5.x
    if not hasattr(_tpu, "isin_mps_friendly"):
        def _isin_mps_friendly(*args, **kwargs):
            if "elements" in kwargs:
                kwargs["input"] = kwargs.pop("elements")
            return _torch.isin(*args, **kwargs)
        _tpu.isin_mps_friendly = _isin_mps_friendly
        _tf.pytorch_utils.isin_mps_friendly = _isin_mps_friendly

    # is_torch_greater_or_equal: moved out of import_utils in some 5.x builds
    import transformers.utils.import_utils as _tiu
    if not hasattr(_tiu, "is_torch_greater_or_equal"):
        from packaging.version import Version as _V
        _tiu.is_torch_greater_or_equal = lambda v: _V(_torch.__version__) >= _V(v)
    if not hasattr(_tiu, "is_torchcodec_available"):
        _tiu.is_torchcodec_available = lambda: False

    # ExtensionsTrie + AddedToken removed in transformers 5.x
    try:
        import transformers.tokenization_utils as _tku
        for _cls_name in ["ExtensionsTrie", "AddedToken"]:
            if not hasattr(_tku, _cls_name):
                _stub_cls = type(_cls_name, (), {
                    "__init__": lambda self, *a, **kw: None,
                    "__doc__": f"Removed in transformers 5.x — stubbed.",
                })
                setattr(_tku, _cls_name, _stub_cls)
                if not hasattr(_tf, _cls_name):
                    setattr(_tf, _cls_name, _stub_cls)
    except Exception:
        pass

    # Sub-modules removed in transformers 5.x — inject empty stubs
    import importlib as _il2
    import types as _types
    _REMOVED_SUBMODULES = [
        "transformers.generation.beam_constraints",
        "transformers.generation.beam_search",
        "transformers.generation.logits_process",
        "transformers.generation.stopping_criteria",
    ]
    for _mod_name in _REMOVED_SUBMODULES:
        if _mod_name not in sys.modules:
            try:
                __import__(_mod_name)
            except ImportError:
                _stub_mod = _types.ModuleType(_mod_name)
                _stub_mod.__doc__ = f"Removed in transformers 5.x — stubbed."
                _stub_mod.__getattr__ = lambda name: type(name, (), {
                    "__init__": lambda self, *a, **kw: None,
                    "__call__": lambda self, *a, **kw: None,
                })
                sys.modules[_mod_name] = _stub_mod

    # Missing symbols in still-present transformers.generation modules
    _GENERATION_MODULE_STUBS = {
        "transformers.generation.candidate_generator": {
            "_crop_past_key_values": lambda model, past_key_values, max_length: past_key_values,
        },
        "transformers.generation.configuration_utils": {
            "NEED_SETUP_CACHE_CLASSES_MAPPING": {},
            "QUANT_BACKEND_CLASSES_MAPPING": {},
            "ALL_CACHE_IMPLEMENTATIONS": [],
        },
        "transformers.generation.utils": {
            "GenerateOutput": type("GenerateOutput", (object,), {}),
        },
        "transformers.modeling_utils": {
            "SequenceSummary": type("SequenceSummary", (object,), {
                "__init__": lambda self, *a, **kw: None,
                "__call__": lambda self, *a, **kw: None,
            }),
        },
    }
    for _mod_path, _stubs in _GENERATION_MODULE_STUBS.items():
        try:
            _mod = _il2.import_module(_mod_path)
            for _sym, _val in _stubs.items():
                if not hasattr(_mod, _sym):
                    setattr(_mod, _sym, _val)
        except Exception:
            pass

except Exception:
    pass

# ── transformers cache_utils stubs (before any indextts import) ───────────────
try:
    import transformers.cache_utils as _cu
    import transformers as _tf2
    for _cls_name in [
        "QuantizedCacheConfig", "QuantizedCache", "QuantoQuantizedCache",
        "HQQQuantizedCache", "OffloadedCache", "SlidingWindowCache", "StaticCacheConfig",
    ]:
        _stub = type(_cls_name, (object,), {"__init__": lambda self, *a, **kw: None})
        for _target in (_cu, _tf2):
            if not hasattr(_target, _cls_name):
                setattr(_target, _cls_name, _stub)
except Exception:
    pass

# ── torchaudio: list_audio_backends removed in 2.x (fish-speech calls it) ────
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except Exception:
    pass

# ── Fish Speech source path ───────────────────────────────────────────────────
try:
    import importlib.util as _ilu2
    if not _ilu2.find_spec("fish_speech"):
        _fs_path = "/opt/models/fish-speech"
        if _fs_path not in sys.path:
            sys.path.insert(0, _fs_path)
except Exception:
    pass

# ── transformers.masking_utils (added in 4.54, needed by qwen_tts) ───────────
try:
    import transformers.masking_utils  # noqa — exists on 4.54+
except ImportError:
    _mu = types.ModuleType("transformers.masking_utils")
    _mu.create_causal_mask = lambda *a, **kw: None
    _mu.create_sliding_window_causal_mask = lambda *a, **kw: None
    _mu.create_causal_4d_mask = lambda *a, **kw: None
    _mu.prepare_4d_causal_attention_mask = lambda *a, **kw: None
    sys.modules["transformers.masking_utils"] = _mu
    try:
        import transformers as _tf3
        _tf3.masking_utils = _mu
    except Exception:
        pass

# ── transformers.modeling_layers (added in 4.54, needed by qwen_tts) ─────────
try:
    import transformers.modeling_layers  # noqa
except ImportError:
    import torch.nn as _nn2
    _ml = types.ModuleType("transformers.modeling_layers")
    class _GradCkptLayer(_nn2.Module):
        _supports_gradient_checkpointing = True
    _ml.GradientCheckpointingLayer = _GradCkptLayer
    sys.modules["transformers.modeling_layers"] = _ml

# ── qwen3tts: Qwen3TTSSpeakerEncoderConfig missing _attn_implementation_* attrs
# Its __init__ never calls super().__init__(), so PretrainedConfig never sets:
#   _attn_implementation_autoset  (set to False in PretrainedConfig.__init__)
#   _attn_implementation_internal (set to None via kwargs.pop in PretrainedConfig.__init__)
# Fix: inject both as class-level defaults before any qwen_tts model import.
try:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig as _Q3Cfg
    if not hasattr(_Q3Cfg, "_attn_implementation_autoset"):
        _Q3Cfg._attn_implementation_autoset = False
    if not hasattr(_Q3Cfg, "_attn_implementation_internal"):
        _Q3Cfg._attn_implementation_internal = None
except Exception:
    pass

# ── qwen3tts: _merge_generate_kwargs passes temperature/do_sample/max_new_tokens
# directly to model.generate() as model_kwargs, but transformers 4.53 requires
# them to be in a GenerationConfig object, not forwarded as model forward kwargs.
# Fix: monkeypatch _merge_generate_kwargs to wrap generation-only params into a
# GenerationConfig and return it as a "generation_config" key so the underlying
# Qwen3TTSForConditionalGeneration.generate() receives them correctly.
try:
    from qwen_tts import Qwen3TTSModel as _Q3M
    from transformers import GenerationConfig as _GenCfg

    _GEN_PARAM_KEYS = frozenset([
        "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
        "max_new_tokens",
    ])

    _orig_merge = _Q3M._merge_generate_kwargs

    def _patched_merge(self, **kwargs):
        merged = _orig_merge(self, **kwargs)
        # Split: generation-config params → GenerationConfig; rest stays as model kwargs
        gen_params = {k: merged.pop(k) for k in list(merged) if k in _GEN_PARAM_KEYS}
        if gen_params:
            merged["generation_config"] = _GenCfg(**gen_params)
        return merged

    _Q3M._merge_generate_kwargs = _patched_merge
except Exception:
    pass

# ── indextts.infer_v2: alias IndexTTS -> IndexTTS2 ───────────────────────────
try:
    import indextts.infer_v2 as _iv2
    if not hasattr(_iv2, "IndexTTS") and hasattr(_iv2, "IndexTTS2"):
        _iv2.IndexTTS = _iv2.IndexTTS2
except Exception:
    pass

# ── parler_tts: _pad/bos/eos_token_tensor are patched directly in
# patch_parler_tts.py — no runtime shim needed here. ─────────────────────────

# ── parler_tts compatibility shims ────────────────────────────────────────────
# parler_tts 0.2.3 has its own complete generate() method. No MRO or
# attention-mask shims needed — all API fixes are in patch_parler_tts.py.
