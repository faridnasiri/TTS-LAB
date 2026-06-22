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
# torch 2.10 + Python 3.11 have an inspect incompatibility: some modules
# end up with __file__ set to a type object instead of a string, which
# crashes inspect.getsourcefile.  The crash propagates through lazy_loader
# → librosa → perth → chatterbox.  Fix by hardening one function minimally:
# patch getsourcefile to return a safe string when it would crash.
import inspect as _inspect
import importlib as _importlib, types as _types

# Increase recursion limit — the inspect chain can get deep when
# lazy_loader calls inspect.stack() during module imports.
import sys as _sys
_sys.setrecursionlimit(10000)

_orig_getsourcefile = _inspect.getsourcefile
def _patched_getsourcefile(obj):
    try:
        return _orig_getsourcefile(obj)
    except (TypeError, AttributeError):
        return "/dev/null"
_inspect.getsourcefile = _patched_getsourcefile

# Pre-stub torch._dynamo._trace_wrapped_higher_order_op to prevent
# the corrupting import chain that starts from transformers.masking_utils.
# This module's import triggers torch._dynamo → torch.distributed.tensor
# → _collective_utils → @register_fake → inspect crash, which corrupts
# module __file__ attributes.  Stubbing it early stops the chain.
for _early_stub in [
    "torch._dynamo._trace_wrapped_higher_order_op",
]:
    if _early_stub not in _sys.modules:
        _m = _types.ModuleType(_early_stub)
        _m.TransformGetItemToIndex = type("TransformGetItemToIndex", (), {})
        _m.trace_wrapped = lambda fn, *a, **kw: fn
        _m.__file__ = "<stub>"
        _sys.modules[_early_stub] = _m

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
#
# CRITICAL: do all transformers patching FIRST, before the try block below
# that may trigger torch._dynamo → tensor → _collective_utils import chains
# which corrupt module __file__ attributes (torch 2.10 + Python 3.11 bug).

# --- ExtensionsTrie + AddedToken (removed in transformers 5.x) ---
try:
    import transformers.tokenization_utils as _tku
    import transformers as _tf_early
    for _cls_name in ["ExtensionsTrie", "AddedToken"]:
        if not hasattr(_tku, _cls_name) or not isinstance(getattr(_tku, _cls_name, None), type):
            _stub_cls = type(_cls_name, (), {
                "__init__": lambda self, *a, **kw: None,
                "__doc__": f"Removed in transformers 5.x — stubbed.",
            })
            setattr(_tku, _cls_name, _stub_cls)
            if not hasattr(_tf_early, _cls_name) or not isinstance(getattr(_tf_early, _cls_name, None), type):
                setattr(_tf_early, _cls_name, _stub_cls)
except Exception:
    pass

# --- check_model_inputs compat — transformers 5.12 is too strict ---
# The check_model_inputs decorator in tf 5.12 rejects kwargs that older
# engine code passes (inputs_embeds, attention_mask, position_ids).
# Replace with a pass-through: engines pass correct kwargs for their models.
try:
    import transformers.utils.generic as _tug
    _orig_cmi = _tug.check_model_inputs
    def _pass_thru_cmi(func=None):
        return func if func is not None else _orig_cmi
    _tug.check_model_inputs = _pass_thru_cmi
except Exception:
    pass

# --- parler-tts config defaults (transformers 5.x removed attrs) ---
# ParlerTTSConfig lacks tie_encoder_decoder which transformers 5.x checks.
try:
    import parler_tts.configuration_parler_tts as _pcfg
    if not hasattr(_pcfg.ParlerTTSConfig, "tie_encoder_decoder"):
        _pcfg.ParlerTTSConfig.tie_encoder_decoder = False
except Exception:
    pass

# --- qwen_tts config defaults ---
# Qwen3TTSTalkerConfig lacks pad_token_id which transformers 5.x needs.
try:
    import qwen_tts
    for _cfg_cls_name in ["Qwen3TTSTalkerConfig"]:
        _cfg_cls = getattr(qwen_tts, _cfg_cls_name, None)
        if _cfg_cls is not None and not hasattr(_cfg_cls, "pad_token_id"):
            _cfg_cls.pad_token_id = 0
except Exception:
    pass

# Set default config attributes that transformers 5.x expects
# (pad_token_id for qwen_tts, tie_encoder_decoder for parler, etc.)
try:
    import transformers.configuration_utils as _tcfg
    _orig_cfg_init = _tcfg.PretrainedConfig.__init__
    def _patched_cfg_init(self, *a, **kw):
        _orig_cfg_init(self, *a, **kw)
        for _attr, _default in [
            ("pad_token_id", 0),
            ("tie_encoder_decoder", False),
            ("tie_word_embeddings", True),
            ("is_encoder_decoder", False),
        ]:
            if not hasattr(self, _attr):
                setattr(self, _attr, _default)
    _tcfg.PretrainedConfig.__init__ = _patched_cfg_init
except Exception:
    pass

# --- Main compat block ---
try:
    import transformers.pytorch_utils as _tpu
    import transformers as _tf
    import torch as _torch

    # isin_mps_friendly: removed from pytorch_utils in transformers 5.x
    def _isin_mps_friendly(*args, **kwargs):
        if "elements" in kwargs:
            kwargs["input"] = kwargs.pop("elements")
        return _torch.isin(*args, **kwargs)
    _tpu.isin_mps_friendly = _isin_mps_friendly
    _tf.pytorch_utils.isin_mps_friendly = _isin_mps_friendly

    # find_pruneable_heads_and_indices + prune_conv1d_layer: removed in transformers 5.x
    for _fn_name in ["find_pruneable_heads_and_indices", "prune_conv1d_layer", "prune_layer"]:
        if not hasattr(_tpu, _fn_name):
            _stub_fn = (lambda: (lambda *a, **kw: ([], {})))() if "find" in _fn_name else (lambda *a, **kw: None)
            setattr(_tpu, _fn_name, _stub_fn)
            setattr(_tf.pytorch_utils, _fn_name, _stub_fn)

    # torch.isin compat: CoquiTTS calls torch.isin(inp, test_elements)
    # which torch 2.10 rejects with keyword args.  Save original and wrap.
    _orig_isin = _torch.isin
    def _patched_isin(*args, **kwargs):
        # Normalise: CoquiTTS passes input= or elements= as keyword
        if len(args) >= 1:
            return _orig_isin(*args, **kwargs)
        # All kwargs — convert to positional
        elements = kwargs.pop("elements", kwargs.pop("input", None))
        test_elements = kwargs.pop("test_elements", None)
        if elements is not None and test_elements is not None:
            return _orig_isin(elements, test_elements, **kwargs)
        return _orig_isin(*args, **kwargs)
    _torch.isin = _patched_isin

    # is_torch_greater_or_equal: moved out of import_utils in some 5.x builds
    import transformers.utils.import_utils as _tiu
    if not hasattr(_tiu, "is_torch_greater_or_equal"):
        from packaging.version import Version as _V
        _tiu.is_torch_greater_or_equal = lambda v: _V(_torch.__version__) >= _V(v)
    if not hasattr(_tiu, "is_torchcodec_available"):
        _tiu.is_torchcodec_available = lambda: False

    # Fix corrupted module metadata — torch 2.10 + Python 3.11 can leave
    # modules with non-string __file__ (type objects, None, etc).
    for _mod_name, _mod in list(sys.modules.items()):
        try:
            _f = getattr(_mod, "__file__", None)
            if _f is not None and not isinstance(_f, str):
                if hasattr(_mod, "__spec__") and _mod.__spec__ is not None:
                    _mod.__file__ = _mod.__spec__.origin or f"<{_mod_name}>"
                else:
                    _mod.__file__ = f"<{_mod_name}>"
        except Exception:
            pass

    # Pre-stub commonly removed transformers 5.x items.
    # Format: (module_path, name, value_or_type)
    # value_or_type can be a type (for classes) or any value (for constants)
    _REMOVED_TF_ITEMS = [
        # logits_process classes (indextts)
        ("transformers.generation.logits_process", "HammingDiversityLogitsProcessor", type),
        ("transformers.generation.logits_process", "HammingDiversityLogitsWarper", type),
        # pytorch_utils (indextts, parler, xtts — already handled above)
        # utils constants (indextts)
        ("transformers.utils", "FLAX_WEIGHTS_NAME", None),
        ("transformers.utils", "TF2_WEIGHTS_NAME", None),
        ("transformers.utils", "TF_WEIGHTS_NAME", None),
        ("transformers.utils", "WEIGHTS_NAME", "pytorch_model.bin"),
        ("transformers.utils", "WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json"),
        ("transformers.utils", "SAFE_WEIGHTS_NAME", "model.safetensors"),
        ("transformers.utils", "SAFE_WEIGHTS_INDEX_NAME", "model.safetensors.index.json"),
        ("transformers.utils", "CONFIG_NAME", "config.json"),
    ]
    for _mod_path, _name, _kind in _REMOVED_TF_ITEMS:
        try:
            _mod = _importlib.import_module(_mod_path)
            if not hasattr(_mod, _name):
                if _kind is type:
                    _val = type(_name, (), {"__init__": lambda s,*a,**kw: None})
                else:
                    _val = _kind
                setattr(_mod, _name, _val)
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

# ── GeneralInterface MutableMapping shim ────────────────────────────────────────
# fix_transformers_shims.py truncates generic.py just before class
# GeneralInterface(MutableMapping), replacing it with an empty stub.
# All of AttentionInterface, AttentionMaskInterface (and any other subclass)
# inherit from GeneralInterface and lose __getitem__, __contains__, register(),
# valid_keys(), etc.
#
# Restore the full MutableMapping contract on GeneralInterface at the class
# level so EVERY subclass picks it up.
try:
    from transformers.utils.generic import GeneralInterface  # noqa: F811

    if not hasattr(GeneralInterface, "__getitem__"):
        def _gi_getitem(self, key):
            local = getattr(self, "_local_mapping", None)
            if local is not None and key in local:
                return local[key]
            return self._global_mapping[key]
        GeneralInterface.__getitem__ = _gi_getitem

    if not hasattr(GeneralInterface, "__setitem__"):
        def _gi_setitem(self, key, value):
            if not hasattr(self, "_local_mapping"):
                self._local_mapping = {}
            self._local_mapping[key] = value
        GeneralInterface.__setitem__ = _gi_setitem

    if not hasattr(GeneralInterface, "__delitem__"):
        def _gi_delitem(self, key):
            if hasattr(self, "_local_mapping") and key in self._local_mapping:
                del self._local_mapping[key]
        GeneralInterface.__delitem__ = _gi_delitem

    if not hasattr(GeneralInterface, "__iter__"):
        def _gi_iter(self):
            local = getattr(self, "_local_mapping", {})
            merged = {**self._global_mapping, **local}
            return iter(merged)
        GeneralInterface.__iter__ = _gi_iter

    if not hasattr(GeneralInterface, "__len__"):
        def _gi_len(self):
            local = getattr(self, "_local_mapping", {})
            return len(self._global_mapping.keys() | local.keys())
        GeneralInterface.__len__ = _gi_len

    if not hasattr(GeneralInterface, "register"):
        @classmethod
        def _gi_register(cls, key, value):
            cls._global_mapping.update({key: value})
        GeneralInterface.register = _gi_register

    if not hasattr(GeneralInterface, "valid_keys"):
        def _gi_valid_keys(self):
            local = getattr(self, "_local_mapping", {})
            merged = {**self._global_mapping, **local}
            return list(merged.keys())
        GeneralInterface.valid_keys = _gi_valid_keys

    if not hasattr(GeneralInterface, "keys"):
        def _gi_keys(self):
            local = getattr(self, "_local_mapping", {})
            merged = {**self._global_mapping, **local}
            return merged.keys()
        GeneralInterface.keys = _gi_keys

except Exception:
    pass

# ── chatterbox T3 hidden_states fix (transformers 4.57.6 compat) ──────────────
# LlamaModel.forward() in transformers 4.57.6 no longer collects all hidden
# states — it returns BaseModelOutputWithPast with hidden_states=None.
# chatterbox's T3HuggingfaceBackend does tfmr_out.hidden_states[-1], which
# raises TypeError: 'NoneType' object is not subscriptable.
# Fix: patch LlamaModel.forward to fill hidden_states when it's None.
# The result is a single-element tuple (last_hidden_state, post-norm).
# chatterbox only accesses [-1] so this is sufficient.  Other models that
# need per-layer hidden states will need a more complete fix later.
try:
    from transformers.models.llama.modeling_llama import LlamaModel

    _orig_llama_forward = LlamaModel.forward

    def _patched_llama_forward(self, **kwargs):
        # transformers 5.12 check_model_inputs rejects extra kwargs.
        # Strip any kwargs not accepted by the undecorated inner forward.
        _inner = getattr(_orig_llama_forward, "__wrapped__", _orig_llama_forward)
        if _inner is _orig_llama_forward:
            # No wrapper — just call with all kwargs
            out = _orig_llama_forward(self, **kwargs)
        else:
            # Inner is the raw forward — check what it accepts
            import inspect as _inspect
            _valid = set(_inspect.signature(_inner).parameters.keys())
            _clean = {k: v for k, v in kwargs.items() if k in _valid}
            # Always include 'self' equivalent
            out = _inner(self, **_clean)
        if out.hidden_states is None and hasattr(out, "last_hidden_state"):
            out.hidden_states = (out.last_hidden_state,)
        return out

    LlamaModel.forward = _patched_llama_forward
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
# Also replace load_with_torchcodec — torchaudio 2.10 ships it but raises
# RuntimeError when torchcodec is not installed. Using _ta.load() as fallback
# causes infinite recursion (torchaudio.load calls load_with_torchcodec internally
# in 2.10). Instead fall back directly to soundfile which has no such loop.
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
    # Replace unconditionally — avoids RuntimeError and recursion
    def _load_with_torchcodec_fallback(path, *args, **kwargs):
        import soundfile as _sf
        import torch as _torch
        import io as _io
        # soundfile.read() can't handle BytesIO via str() — use the file object directly
        if isinstance(path, _io.BytesIO):
            path.seek(0)
            data, sr = _sf.read(path, dtype="float32", always_2d=True)
        elif isinstance(path, (str, bytes)):
            data, sr = _sf.read(path, dtype="float32", always_2d=True)
        else:
            # Try as file-like object, fall back to str()
            try:
                data, sr = _sf.read(path, dtype="float32", always_2d=True)
            except Exception:
                data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) — transpose to (channels, samples)
        return _torch.from_numpy(data.T), sr
    _ta.load_with_torchcodec = _load_with_torchcodec_fallback
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
except Exception:
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
# Wrapped broadly — the import chain pulls in torchvision which may be
# incompatible with torch nightly (torchvision::nms operator missing).
try:
    import transformers.modeling_layers  # noqa
except (ImportError, RuntimeError, OSError, ModuleNotFoundError):
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

# ── qwen3tts: Qwen3TTSTokenizerV1/V2Model.config_class is None
# transformers auto_factory.py line 619 does:
#   model_class.config_class.__name__  → AttributeError: 'NoneType' has no '__name__'
# Fix: set config_class to the matching config type on each tokenizer model class.
try:
    from qwen_tts.inference.qwen3_tts_tokenizer import (
        Qwen3TTSTokenizerV1Model  as _TokV1M,
        Qwen3TTSTokenizerV1Config as _TokV1C,
        Qwen3TTSTokenizerV2Model  as _TokV2M,
        Qwen3TTSTokenizerV2Config as _TokV2C,
    )
    if _TokV1M.config_class is None:
        _TokV1M.config_class = _TokV1C
    if _TokV2M.config_class is None:
        _TokV2M.config_class = _TokV2C
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

# ── Chatterbox SDPA fix: AlignmentStreamAnalyzer needs output_attentions=True
# which sdpa doesn't support. Force eager on the specific llama layers that
# the spy hooks into.  DO NOT restore _attn_implementation afterward — the
# hooks fire during the forward pass (later), not during registration (now).
# Restoring "sdpa" here would cause the hooked layers to return None for
# attention weights, crashing torch.stack(self.last_aligned_attns).
try:
    from chatterbox.models.t3.inference.alignment_stream_analyzer import (
        AlignmentStreamAnalyzer as _ASA,
    )
    _orig_add_spy = _ASA._add_attention_spy
    def _patched_add_spy(self, tfmr, i, layer_idx, head_idx):
        if hasattr(tfmr, "config"):
            tfmr.config._attn_implementation = "eager"
        return _orig_add_spy(self, tfmr, i, layer_idx, head_idx)
    _ASA._add_attention_spy = _patched_add_spy
except Exception:
    pass

# ── scipy.signal.kaiser compat (removed in scipy 1.14+, needed by parallel_wavegan)
try:
    import scipy.signal as _sig
    if not hasattr(_sig, "kaiser"):
        from scipy.signal.windows import kaiser as _kaiser
        _sig.kaiser = _kaiser
except Exception:
    pass
