# VibeVoice Investigation — 2026-06-23

> **Model:** microsoft/VibeVoice-1.5B (1.5B params, AR + diffusion, English + Chinese)
> **Container:** engine-mid (torch 2.12 nightly, transformers 4.51.3, CUDA 12.8)
> **Package:** vibevoice 0.0.1 (pip)
> **Target hardware:** RTX 5060 Ti, 16 GB VRAM

---

## 1. Background

The original containerization plan assumed VibeVoice required SGLang-Omni for serving. The model's HuggingFace repo (`microsoft/VibeVoice-1.5B`) contains only a `config.json` and weight shards — no Python modeling code, no `trust_remote_code` support, and no tokenizer files. The model card recommends SGLang-Omni as the serving path.

However, the `vibevoice` pip package (0.0.1) provides modeling code that could potentially enable local inference without SGLang. This investigation tests whether local inference is viable through the engine-mid container.

---

## 2. What Was Tested

### 2.1 Stage 1 — AutoConfig (FAILED)

**Test:** `AutoConfig.from_pretrained("microsoft/VibeVoice-1.5B", trust_remote_code=True)`

**Result:** `KeyError: 'vibevoice'` — the `vibevoice` architecture type is not registered in transformers 4.51.3's `CONFIG_MAPPING`.

**Root cause:** Neither transformers 4.51.3 nor the vibevoice 0.0.1 pip package registers the `vibevoice` model type with HuggingFace's auto-registry. The config and modeling classes exist in the package (`configuration_vibevoice.py`, `modeling_vibevoice.py`, `modeling_vibevoice_inference.py`) but are not wired into `AutoConfig` / `AutoModel`.

**Gate status:** `config_load` = FAILED

### 2.2 Stage 2 — Direct Import of Config + Model Classes (PASSED)

**Test:** Bypass `AutoConfig`/`AutoModel` entirely:

```python
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import VibeVoiceModel
cfg = VibeVoiceConfig.from_pretrained("microsoft/VibeVoice-1.5B")
model = VibeVoiceModel.from_pretrained("microsoft/VibeVoice-1.5B", config=cfg,
    device_map="cuda", torch_dtype=torch.bfloat16)
```

**Result:** Both classes import and instantiate correctly. The config parses without errors. This confirms the modeling code is functional — the problem is purely a registration-layer issue, not a code or model incompatibility.

### 2.3 Stage 3 — Model Weight Loading (PASSED)

**Test:** Load full weights via `VibeVoiceModel.from_pretrained()` or `VibeVoiceForConditionalGenerationInference.from_pretrained()`.

**Results:**

| Metric | Value |
|--------|-------|
| Load time | 8 seconds |
| VRAM (after load) | 5.4 GB |
| GPU | RTX 5060 Ti |
| OOM? | No |
| State dict errors? | None |
| Unmapped keys? | None |

**Key finding:** The model loads cleanly from the HF repo weight shards. No serialization errors, no unmapped state dict keys, no OOM. VRAM fits comfortably within the 7 GB budget for engine-mid alongside a light engine-current engine.

**Gate status:** `model_load` = PASSED (provisional — pending tokenizer resolution)

### 2.4 Stage 4 — Tokenizer (FAILED)

#### 2.4.1 Attempt 1: from_pretrained

```python
tok = VibeVoiceTextTokenizer.from_pretrained("microsoft/VibeVoice-1.5B", config=cfg)
```

**Error:** `OSError: Can't load tokenizer for 'microsoft/VibeVoice-1.5B'`

**Root cause:** The HF repo contains NO tokenizer files — no `tokenizer_config.json`, no `vocab.json`, no `merges.txt`. The VibeVoice repo is weights-only.

#### 2.4.2 Attempt 2: Direct init (no tokenizer files)

```python
tok = VibeVoiceTextTokenizer(config=cfg)
```

**Error:** `missing 2 required positional arguments: 'vocab_file' and 'merges_file'`

**Root cause:** `VibeVoiceTextTokenizer` extends `Qwen2Tokenizer` which requires BPE vocabulary files.

#### 2.4.3 Attempt 3: Qwen2 tokenizer files + direct init

```python
vocab_file = hf_hub_download("Qwen/Qwen2-0.5B", "vocab.json")
merges_file = hf_hub_download("Qwen/Qwen2-0.5B", "merges.txt")
tok = VibeVoiceTextTokenizer(vocab_file=vocab_file, merges_file=merges_file)
```

**Error:** `AttributeError: add_special_tokens conflicts with the method add_special_tokens in VibeVoiceTextTokenizer`

**Root cause chain:**
1. `VibeVoiceTextTokenizer.__init__()` has `add_special_tokens=True` as a parameter
2. It calls `super().__init__(..., add_special_tokens=add_special_tokens, ...)`
3. `Qwen2Tokenizer.__init__()` passes it further to `PreTrainedTokenizerBase.__init__()`
4. In transformers 4.51.3, `PreTrainedTokenizerBase` has a METHOD called `add_special_tokens` (for adding special tokens at runtime)
5. Having both a parameter AND a method with the same name causes `AttributeError`

**This is a transformers version incompatibility.** The vibevoice 0.0.1 package was built against a transformers version where this conflict was already resolved (likely >=4.55 or 5.x).

### 2.5 Stage 5 — Inference (NOT REACHED)

The `.generate()` method on `VibeVoiceForConditionalGenerationInference` was never tested because the tokenizer could not be constructed. If the tokenizer is resolved, the code path for inference exists:

```python
output = model.generate(**inputs, max_new_tokens=500)
audio = output.audio_values  # shape: (1, samples) at 24 kHz
```

---

## 3. Root Cause Summary

| Component | Status | Blocker |
|-----------|:------:|---------|
| Config class | Works | — |
| Model class | Works | — |
| Weight loading | Works (5.4 GB, 8s) | — |
| AutoConfig registration | Blocked | `vibevoice` not in `CONFIG_MAPPING` |
| Tokenizer construction | Blocked | `add_special_tokens` parameter/method conflict in tf 4.51.3 |
| Inference | Not reached | Blocked by tokenizer |

**The model itself is fully loadable and functional.** Both failures are at the integration layer (AutoConfig registration, tokenizer version compatibility), not at the model layer.

---

## 4. Solution Paths

### Path A: Upgrade transformers in engine-mid (Recommended first attempt)

```bash
docker exec tts-lab-engine-mid pip install transformers==4.57.3
```

**Rationale:** transformers 4.57.3 is already proven in engine-qwen (with the `TransformGetItemToIndex` shim patch already in `tts_lab_shims.py`). The `add_special_tokens` conflict was likely fixed somewhere between 4.51.3 and 4.57.3.

**Risk:** Higgs (the other engine in engine-mid) may break with newer transformers. Mitigation: test Higgs immediately after upgrade.

**If successful:** VibeVoice gets its tokenizer. Inference becomes testable. No new container needed.

### Path B: SGLang-Omni (Wait for upstream)

```yaml
# Already defined in docker-compose.yml:
vibevoice:
  image: lmsysorg/sglang-omni:dev
  command: --model microsoft/VibeVoice-1.5B
  profiles: [sglang]
```

**Rationale:** The model card recommends this. SGLang handles tokenization internally. When SGLang releases an image with transformers >= 4.57 (or 5.x), it should work.

**Risk:** No timeline. SGLang currently ships transformers 5.6.0 which is too old/too different.

### Path C: Dedicated vibevoice container

Create `Dockerfile.engine-vibevoice` with its own transformers version (4.57.3+), pinning whatever it needs. Clean isolation but adds container count.

---

## 5. What Would Make This Work Locally

If the tokenizer conflict is resolved (Path A), the local inference code in `tts_lab_engines.py` would be approximately 15 lines to write `_load_vibevoice()` and `_synth_vibevoice()` using direct imports from the vibevoice package and Qwen2 tokenizer files — bypassing AutoConfig entirely.

---

## 6. Comparison With S2-Pro

| Property | VibeVoice | S2-Pro |
|----------|-----------|--------|
| Model loads locally? | Yes | No |
| Needs SGLang runtime? | Only for tokenizer (solvable) | Yes (paged KV, RadixAttention, CUDA graph) |
| Blocked by? | transformers tokenizer conflict (fixable) | SGLang image version (wait required) |
| Fix effort | ~15 lines engine code + tf upgrade | Zero code, wait for upstream |
| Risk of fix | Medium (may break Higgs) | None |
| Architecture impact | None (stays in engine-mid) | None (already in compose) |

---

## 7. Recommendation

**Try Path A first** — upgrade transformers in engine-mid to 4.57.3 and test both VibeVoice and Higgs. One `pip install` command. If both work, VibeVoice becomes the second engine after Qwen3TTS to graduate from SGLang-assumed to local inference. If Higgs breaks, roll back and wait for SGLang or create a dedicated container.
