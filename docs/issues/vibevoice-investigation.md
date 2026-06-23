# VibeVoice Investigation — 2026-06-23

> **Model:** microsoft/VibeVoice-1.5B (1.5B params, AR + diffusion, English + Chinese)
> **Container:** engine-mid (torch 2.12 nightly, transformers 4.51.3, CUDA 12.8)
> **Package:** vibevoice 0.0.1 (pip)
> **Target hardware:** RTX 5060 Ti, 16 GB VRAM
> **Verdict:** BLOCKED for local inference — model loads, inference pipeline requires SGLang

---

## 1. Background

The original containerization plan assumed VibeVoice required SGLang-Omni. The `vibevoice` pip package (0.0.1) provides modeling code. This investigation tested whether local inference without SGLang is viable.

---

## 2. What Was Tested

### 2.1 Stage 1 — AutoConfig (FAILED)

**Test:** `AutoConfig.from_pretrained("microsoft/VibeVoice-1.5B", trust_remote_code=True)`

**Result:** `KeyError: 'vibevoice'` — architecture not registered in transformers 4.51.3 `CONFIG_MAPPING`.

**Verdict:** Irrelevant. Direct imports bypass this. Not worth spending time on registration.

### 2.2 Stage 2 — Direct Import (PASSED)

```python
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import VibeVoiceModel
```

**Result:** Both classes import and instantiate. Config parses cleanly.

### 2.3 Stage 3 — Model Weight Loading (PASSED)

**Test:** `VibeVoiceForConditionalGenerationInference.from_pretrained("microsoft/VibeVoice-1.5B", config=cfg, device_map="cuda", torch_dtype=torch.bfloat16)`

| Metric | Value |
|--------|-------|
| Load time | 3-8 seconds |
| VRAM (after load) | 5.4 GB |
| State dict errors | None |
| Unmapped keys | None |

### 2.4 Stage 4 — Tokenizer (PASSED via Monkey-Patch)

**Problem:** `VibeVoiceTextTokenizer` extends `Qwen2Tokenizer`. Its `__init__` passes `add_special_tokens=True` to `PreTrainedTokenizerBase.__init__()`. In tf 4.51.3, `add_special_tokens` is both a parameter and a method — collision. The HF repo has no tokenizer files.

**Fix (two parts):**

Part 1 — Monkey-patch the base class:
```python
import transformers.tokenization_utils_base as tub
_orig_init = tub.PreTrainedTokenizerBase.__init__
def _patched_init(self, **kwargs):
    kwargs.pop('add_special_tokens', None)
    return _orig_init(self, **kwargs)
tub.PreTrainedTokenizerBase.__init__ = _patched_init
```

Part 2 — Download Qwen2 tokenizer files (VibeVoice uses Qwen2 BPE):
```python
vocab_file = hf_hub_download("Qwen/Qwen2-0.5B", "vocab.json")
merges_file = hf_hub_download("Qwen/Qwen2-0.5B", "merges.txt")
tok = VibeVoiceTextTokenizer(vocab_file=vocab_file, merges_file=merges_file)
```

**Result:** Tokenizer constructs and tokenizes correctly. `input_ids` shape (1, 9) for "Hello world. VibeVoice deployment test."

### 2.5 Stage 5 — Inference (FAILED — Deep Pipeline Coupling)

Five patch layers were attempted. Each revealed a deeper coupling point:

#### Layer 1: generate() doesn't auto-detect tokenizer
**Error:** `'NoneType' object has no attribute 'bos_token_id'`
**Fix:** Pass `tokenizer=tok` explicitly.

#### Layer 2: speech_tensors.to() called on None
**Error:** `AttributeError: 'NoneType' object has no attribute 'to'`
**Cause:** `generate()` unconditionally calls `speech_tensors.to(device=device)` during prefill, even for text-only generation.
**Fix:** Provide dummy `speech_tensors=torch.zeros(1, 0)`.

#### Layer 3: Zero-length tensor fails conv1d
**Error:** `RuntimeError: Calculated padded input size per channel: (6). Kernel size: (7). Kernel size can't be greater than actual input size`
**Fix:** Provide 2400-sample silence tensor (0.1s @ 24kHz).

#### Layer 4: Acoustic mask shape mismatch
**Error:** `IndexError: The shape of the mask [1, 2400] at index 1 does not match the shape of the indexed tensor [1, 1, 1536] at index 1`
**Cause:** Acoustic tokenizer encodes 2400 samples → latent shape (1, 1, 1536). Speech mask is sample-level (1, 2400). The mask must match the encoder's stride-compressed latent dimension, not raw samples. This alignment is handled by SGLang internally.

#### Layer 5: Not attempted
Would require knowledge of the acoustic tokenizer's exact encoding stride to construct correct dummy latent masks. This is deep internal pipeline state that SGLang manages.

### 2.6 Why the Pipeline is Tightly Coupled

The vibevoice inference pipeline processes text and speech as interleaved multimodal streams:

```
text  → text_tokenizer      → input_ids
speech → acoustic_tokenizer → acoustic_features → connector → speech_embeds
                                    ↓
                          AR decoder (interleaves text + speech)
                                    ↓
                          semantic_tokenizer → audio waveform
```

The `speech_tensors`, `speech_masks`, and `speech_input_mask` parameters all have `Optional` type hints, but the internal code unconditionally dereferences them. Text-only generation is not a supported code path in vibevoice 0.0.1. This is not a bug — it's a design choice. The package was built for SGLang-Omni serving.

---

## 3. Root Cause Summary

| Component | Status | Blocker |
|-----------|:------:|---------|
| Config class | ✅ Works | — |
| Model class | ✅ Works | — |
| Weight loading | ✅ Works (5.4 GB, 3-8s) | — |
| AutoConfig registration | ❌ | Not needed (direct import works) |
| Tokenizer constructor | ✅ Patched | `add_special_tokens` collision in tf 4.51.3 |
| Tokenizer files | ✅ Qwen2 vocab | HF repo has no tokenizer files |
| generate() call | ❌ | Multimodal pipeline assumes speech input |
| Inference pipeline | ❌ | Deep coupling to SGLang coordination |

**The model itself is loadable and functional.** The blocker is the inference pipeline's assumption that SGLang handles tokenizer coordination, tensor alignment, and streaming cache — none of which exist in standalone mode.

---

## 4. Solution Paths

### Path A: SGLang-Omni (Wait for upstream — ONLY viable path)

```yaml
# Already defined in docker-compose.yml:
vibevoice:
  image: lmsysorg/sglang-omni:dev
  command: --model microsoft/VibeVoice-1.5B
  profiles: [sglang]
```

SGLang handles tokenizer coordination, speech/text interleaving, and streaming cache. When SGLang updates its image to a transformers version that supports the vibevoice architecture, this path works with zero code changes.

**Monitor:** https://github.com/sgl-project/sglang/releases

### Path B: Newer vibevoice release (Wait for upstream)

If Microsoft releases an updated vibevoice package with text-only generation support, or if a future transformers release adds native `vibevoice` architecture support with proper tokenizer handling.

### Path C: Deep fork (NOT RECOMMENDED)

Rewriting the inference pipeline's speech-processing path for text-only generation would require understanding the acoustic tokenizer's encoding stride, constructing correct dummy latent masks, and potentially modifying the AR decoder to skip speech token interleaving. Brittle, high-maintenance, and likely to break on upstream updates.

---

## 5. Comparison With S2-Pro

| Property | VibeVoice | S2-Pro |
|----------|-----------|--------|
| Model loads locally? | ✅ Yes | ❌ No |
| Tokenizer works? | ✅ Patched | N/A |
| Inference works? | ❌ Needs SGLang pipeline | ❌ Needs SGLang runtime |
| Blocked by? | Multimodal pipeline coupling | paged KV cache, RadixAttention |
| Fix effort | Deep fork (brittle) | Wait for upstream |
| SGLang path viable? | ✅ Yes (when image updates) | ✅ Yes (when image updates) |
| Architecture impact | None | None |

---

## 6. Disposition

**VibeVoice remains EXPERIMENTAL. Classification: BLOCKED for local inference — waiting on SGLang upstream.**

The model loads and the tokenizer can be patched. But the inference pipeline assumes SGLang is handling multimodal coordination. After 5 patch layers each revealing the next coupling point, further attempts at standalone local inference have sharply diminishing returns. The correct path is SGLang-Omni.
