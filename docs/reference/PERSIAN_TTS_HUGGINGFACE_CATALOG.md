# Persian/Farsi TTS Models on Hugging Face — Complete Catalog & Integration Plan

> **Date:** 2026-06-23
> **Status:** Research complete — integration plan ready for review
> **Related:** [PERSIAN_TTS_MODELS.md](PERSIAN_TTS_MODELS.md) — existing Persian engines in the lab
> **Related:** [../engine_compatibility.yaml](../engine_compatibility.yaml) — machine-readable SSOT
> **Related:** [../containerization/01-ARCHITECTURE.md](../containerization/01-ARCHITECTURE.md) — container topology

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Methodology](#2-research-methodology)
3. [The Complete Landscape](#3-the-complete-landscape)
4. [Already in the Lab](#4-already-in-the-lab)
5. [Top 10 New Models — Ranked & Justified](#5-top-10-new-models--ranked--justified)
6. [Quality Tiers](#6-quality-tiers)
7. [Models NOT Recommended](#7-models-not-recommended)
8. [Integration Plan](#8-integration-plan)
9. [Container Placement Strategy](#9-container-placement-strategy)
10. [VRAM Budget Analysis](#10-vram-budget-analysis)
11. [Implementation Phases](#11-implementation-phases)
12. [Engine Catalog Entries](#12-engine-catalog-entries)

---

## 1. Executive Summary

After exhaustive search of Hugging Face, there are approximately **20 dedicated Persian TTS model repositories** and **4 multilingual models with production-quality Persian support**. The community landscape is dominated by VITS-based models from 2-3 authors (Saillab, Kamtera, karim23657), trained on Common Voice or Azure synthetic data.

Of these, **10 are genuinely worth integrating** into the Arthur TTS Lab. The rest are checkpoint variations of the same VITS architecture or have quality issues. The single highest-impact addition is **facebook/mms-tts-fas** (Meta's official Persian TTS), followed by **MOSS-TTS-Nano-100M** (brand new LLM-based architecture, April 2026).

**Key Finding:** There are no "production-grade" dedicated Persian TTS models on Hugging Face that surpass what's already in the lab (Chatterbox Persian T3). The models recommended here add architecture diversity, voice variety, and benchmark coverage — not a step-change in Persian audio quality.

### Quick Stats

| Metric | Count |
|--------|-------|
| Total dedicated Persian TTS repos found | ~20 |
| Truly distinct models (different architecture/author/data) | ~12 |
| Models already in Arthur TTS Lab | 6 |
| Models recommended for integration | 10 |
| Models that fit engine-current stack | 8 |
| Models that need dedicated containers | 2 |
| Total additional VRAM at full load | ~4-5 GB |

---

## 2. Research Methodology

Searches conducted across Hugging Face, Google, and arXiv (2026-06-23):

- `huggingface "persian" OR "farsi" OR "fa" TTS model`
- `huggingface.co/models?language=fa&task=text-to-speech`
- `site:huggingface.co "persian-tts" OR "farsi-tts"`
- Cross-referenced with arXiv papers (ParsVoice, ZabanZad, MOSS-TTS)
- Traced model references in ZabanZad PoC Space source code
- Checked karim23657's Persian TTS Collection on HF

**Limitations:** Some models may be gated/private (mhrahmani/persian-tts-vits-0). Some Saillab models returned 401 on direct access. Model quality assessments are based on paper-reported metrics (MOS, WER) where available; otherwise based on architecture, training data quality, and community feedback.

---

## 3. The Complete Landscape

### 3.1 Dedicated Persian TTS Models (20 repos)

#### VITS-based (Community) — 11 repos

| # | Model | Author | Speakers | Training Data | Downloads/Mo |
|---|-------|--------|----------|---------------|-------------|
| 1 | `Kamtera/persian-tts-female-vits` | Kamtera | 1 Female | Persian TTS Dataset (Kaggle) | 104 |
| 2 | `Kamtera/persian-tts-male-vits` | Kamtera | 1 Male | Fine-tuned from female VITS | ~50 |
| 3 | `karim23657/persian-tts-female-GPTInformal-Persian-vits` | karim23657 | 1 Female | GPTInformal-Persian (colloquial) | ~20 |
| 4 | `saillab/persian-tts-azure-grapheme-60K` | Saillab | 1 Speaker | Azure TTS synthetic | ~30 |
| 5 | `saillab/persian-tts-cv15-reduct-grapheme-multispeaker` | Saillab | Multi | Common Voice 15 (reduced) | ~25 |
| 6 | `saillab/persian-tts-grapheme-arm24-finetuned-on1` | Saillab | 1 Speaker | ARM24 + fine-tuned on 1 speaker | ~15 |
| 7 | `saillab/multi_speaker` | Saillab | Multi | Common Voice 15 (90K steps) | ~20 |
| 8 | `saillab/female_cv_azure_male_azure_female` | Saillab | 1 Female | CV + Azure male + Azure female mixed | ~15 |
| 9 | `saillab/Multi_Speaker_Cv_plus_Azure_female_in_one_set` | Saillab | Multi | CV + Azure female combined | ~10 |
| 10 | `mhrahmani/persian-tts-vits-0` | mhrahmani | 1 Speaker | Unknown (57K steps) | gated |
| 11 | `SeyedAli/Persian-Speech-synthesis-MMS` | SeyedAli | 1 Speaker | Mirror of MMS-TTS-FAS | ~275 |

#### Official / Research Models — 5 repos

| # | Model | Author | Architecture | Size |
|---|-------|--------|-------------|------|
| 12 | `facebook/mms-tts-fas` | Meta | VITS | 36.3M |
| 13 | `MahtaFetrat/Persian-Tacotron2-on-ManaTTS` | MahtaFetrat | Tacotron2 | 371 MB |
| 14 | `mah92/Khadijah-FA_EN-Matcha-TTS-Model` | mah92 | Matcha-TTS | ~200 MB |
| 15 | `Hamid20/speecht5_tts_persian` | Hamid20 | SpeechT5 | ~0.1B |
| 16 | `MahtaFetrat/Mana-Persian-Piper` | MahtaFetrat | Piper ONNX | ~50 MB |

#### XTTS / Chatterbox Fine-tunes — 2 repos

| # | Model | Author | Architecture | Quality |
|---|-------|--------|-------------|---------|
| 17 | `saillab/xtts_v2_fa` | Saillab | XTTS v2 | MOS 3.6 / SMOS 4.0 |
| 18 | `Thomcles/Chatterbox-TTS-Persian-Farsi` | Thomcles | Chatterbox | Poor (community reports) |

#### Piper Variants — 2 repos

| # | Model | Author | Voices |
|---|-------|--------|--------|
| 19 | `rhasspy/piper-voices` fa_IR/gyro-* | rhasspy | gyro-low, gyro-medium, gyro-high |
| 20 | `SadeghK/persian-text-to-speech` | SadeghK | amir (base checkpoint) |

### 3.2 Multilingual Models with Persian Support — 4 repos

| # | Model | Author | Architecture | Persian Tier |
|---|-------|--------|-------------|-------------|
| 21 | `bosonai/higgs-audio-v3-tts-4b` | BosonAI | Higgs Audio (4B) | Production (WER/CER <5%) |
| 22 | `FenomAI/MOSS-TTS-Nano-100M` | OpenMOSS | LLM (0.1B) | Experimental (20 languages) |
| 23 | `hootan09/ChatterBox` | hootan09/ResembleAI | Chatterbox (0.5B) | 23 languages incl. Persian |
| 24 | `Qwen/Qwen3-TTS` | Alibaba | Qwen3-TTS (1.7B) | Experimental Persian |

---

## 4. Already in the Lab

These models are already integrated and serving in Arthur TTS Lab:

| Engine Key | HF Model | Architecture | Status | Persian Quality |
|-----------|----------|-------------|--------|----------------|
| `chatterbox` | `hootan09/ChatterBox` + Persian T3 | Chatterbox (0.5B) | ✅ Supported | **Best** — Persian fine-tuned T3 |
| `matcha` | `csukuangfj/matcha-tts-fa_en-{khadijah,musa}` | Matcha-TTS | ✅ Supported | Good — bilingual FA-EN |
| `manatts` | `MahtaFetrat/Persian-Tacotron2-on-ManaTTS` | Tacotron2 | ⚠️ Experimental | Good — single speaker |
| `piper` | `MahtaFetrat/Mana-Persian-Piper` | Piper ONNX | ✅ Supported | Good — real-time |
| `fishspeech` | (multilingual) | Fish Speech | ✅ Supported | Decent — zero-shot |
| `f5tts` | (multilingual) | F5-TTS | ✅ Supported | Decent — zero-shot w/ ref WAV |
| `higgs` | `bosonai/higgs-audio-v3-tts-4b` | Higgs (4B) | ⚠️ Experimental | Unknown — not yet validated |
| `qwen3tts` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Qwen3-TTS | ⚠️ Experimental | Unknown — experimental Persian |
| `xtts` | (base Coqui XTTS) | XTTS v2 | ✅ Supported | **Poor** — base model lacks Persian |

> **Note:** The base `xtts` engine in engine-current is the generic Coqui XTTS v2 model. It does NOT include the Persian fine-tune (`saillab/xtts_v2_fa`). This is a key gap — the Persian XTTS fine-tune would significantly improve Persian quality for the XTTS engine.

---

## 5. Top 10 New Models — Ranked & Justified

### Tier 1: Must-Add (unique architecture, official source, or significant capability)

#### #1 🥇 [facebook/mms-tts-fas](https://huggingface.co/facebook/mms-tts-fas) — Meta Official Persian TTS

| Property | Value |
|----------|-------|
| **Architecture** | VITS (36.3M params) |
| **VRAM** | ~200 MB |
| **Disk** | ~150 MB |
| **API** | `VitsModel.from_pretrained("facebook/mms-tts-fas")` |
| **License** | CC-BY-NC 4.0 |
| **Why #1** | Only Persian TTS from a major AI lab (Meta). Reference-quality baseline. Works via standard `transformers` API — 5 lines of code to integrate. Non-deterministic (variational) for natural variation. |
| **Integration effort** | Trivial — fits in engine-current, no new dependencies |
| **Container** | `engine-current` (stack: current) |

#### #2 🥈 [FenomAI/MOSS-TTS-Nano-100M](https://huggingface.co/FenomAI/MOSS-TTS-Nano-100M) — Newest LLM Architecture (April 2026)

| Property | Value |
|----------|-------|
| **Architecture** | Pure autoregressive Audio Tokenizer + LLM (0.1B params) |
| **VRAM** | ~300 MB |
| **Disk** | ~200 MB |
| **API** | CLI: `moss-tts-nano generate --prompt-speech ref.wav --text "..."` |
| **License** | Apache 2.0 ✅ |
| **Why #2** | Brand new (released April 10, 2026). Completely different architecture from anything in the lab — LLM-based, not encoder-decoder or flow. 48kHz stereo, streaming inference, long-text support with chunked voice cloning. 20 languages. Apache 2.0 license is the most permissive. |
| **Integration effort** | Medium — needs `moss-tts-nano` pip package, reference audio handling |
| **Container** | `engine-current` (stack: current) or dedicated if pip conflicts arise |

#### #3 🥉 [saillab/xtts_v2_fa](https://huggingface.co/saillab/xtts_v2_fa) — Persian XTTS v2 Fine-tune (ZabanZad)

| Property | Value |
|----------|-------|
| **Architecture** | XTTS v2 (GPT-based voice cloning, ~1.6B params) |
| **VRAM** | ~2 GB |
| **Disk** | ~1.6 GB (checkpoint: `best_model_110880.pth`) |
| **API** | Coqui TTS `Synthesizer` or `TTS` API |
| **License** | Coqui TTS license (CPML) |
| **Why #3** | Your existing `xtts` engine uses the base XTTS v2 model which has poor Persian quality. This is the dedicated Persian fine-tune from the ZabanZad project. MOS 3.6 naturalness, MOS 4.0 speaker similarity (validated in ParsVoice paper). Voice cloning for Persian — uniquely valuable. |
| **Integration effort** | Medium — Coqui TTS already installed in engine-current; needs model download + config |
| **Container** | `engine-current` (stack: current) — coqui-tts already installed |

### Tier 2: Best Community Voices

#### #4 [Kamtera/persian-tts-female-vits](https://huggingface.co/Kamtera/persian-tts-female-vits) — Most Popular Female Persian Voice

| Property | Value |
|----------|-------|
| **Architecture** | VITS (Coqui TTS framework) |
| **VRAM** | ~500 MB |
| **Disk** | ~200 MB |
| **API** | Coqui TTS: `tts --model_path model.pth --config_path config.json` |
| **Why #4** | Most widely-used community Persian female TTS (104 downloads/month, 27 Spaces). Good benchmark for community-VITS quality ceiling against your existing Matcha-TTS Khadijah. |
| **Integration effort** | Low — Coqui TTS already installed |
| **Container** | `engine-current` |

#### #5 [Kamtera/persian-tts-male-vits](https://huggingface.co/Kamtera/persian-tts-male-vits) — Male Persian Voice

| Property | Value |
|----------|-------|
| **Architecture** | VITS (Coqui TTS framework) |
| **VRAM** | ~500 MB |
| **Disk** | ~200 MB |
| **API** | Coqui TTS |
| **Why #5** | Male counterpart to the female VITS. Most of your existing Persian voices are female (Khadijah, Mana-Piper, ManaTTS). This fills the male voice gap. |
| **Integration effort** | Low — Coqui TTS already installed |
| **Container** | `engine-current` |

### Tier 3: Specialized Value

#### #6 [saillab/persian-tts-azure-grapheme-60K](https://huggingface.co/saillab/persian-tts-azure-grapheme-60K) — Best Studio-Quality VITS

| Property | Value |
|----------|-------|
| **Architecture** | VITS (grapheme-based, no phonemizer) |
| **VRAM** | ~500 MB |
| **Disk** | ~200 MB |
| **Why #6** | Trained on Azure TTS data — higher recording quality than Common Voice models. Grapheme-based (no phonemizer needed) = simpler pipeline, fewer failure modes. Best reference for "clean studio" Persian VITS quality. |
| **Integration effort** | Low |
| **Container** | `engine-current` |

#### #7 [saillab/persian-tts-cv15-reduct-grapheme-multispeaker](https://huggingface.co/saillab/persian-tts-cv15-reduct-grapheme-multispeaker) — Only Multi-Speaker Persian Model

| Property | Value |
|----------|-------|
| **Architecture** | VITS (multi-speaker, grapheme-based) |
| **VRAM** | ~500 MB |
| **Disk** | ~250 MB (includes `speakers.pth`) |
| **Why #7** | Only Persian TTS model with multiple built-in speaker identities. No reference audio needed to switch voices — just change speaker ID. Unique capability for A/B voice comparison. |
| **Integration effort** | Low |
| **Container** | `engine-current` |

#### #8 [karim23657/persian-tts-female-GPTInformal-Persian-vits](https://huggingface.co/karim23657/persian-tts-female-GPTInformal-Persian-vits) — Colloquial Persian

| Property | Value |
|----------|-------|
| **Architecture** | VITS |
| **VRAM** | ~500 MB |
| **Disk** | ~200 MB |
| **Why #8** | **Unique:** trained on informal/colloquial Persian (GPTInformal dataset). Handles slang, casual expressions, and everyday speech patterns that formal TTS models struggle with. Complements your existing formal-Persian engines. |
| **Integration effort** | Low |
| **Container** | `engine-current` |

### Tier 4: Architecture Diversity & Alternatives

#### #9 [Hamid20/speecht5_tts_persian](https://huggingface.co/Hamid20/speecht5_tts_persian) — SpeechT5 Architecture

| Property | Value |
|----------|-------|
| **Architecture** | SpeechT5 (encoder-decoder transformer, ~0.1B params) |
| **VRAM** | ~500 MB |
| **Disk** | ~400 MB |
| **API** | `SpeechT5ForTextToSpeech.from_pretrained()` + HiFi-GAN vocoder |
| **Why #9** | Only SpeechT5 Persian fine-tune. Different architecture from VITS — encoder-decoder transformer vs. variational. Trained 4000 steps, validation loss 0.5369. Good for architecture diversity in benchmarks. |
| **Integration effort** | Low — transformers already installed |
| **Container** | `engine-current` |

#### #10 [rhasspy/piper-voices fa_IR/gyro-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/fa_IR) — Alternative Piper Voice

| Property | Value |
|----------|-------|
| **Architecture** | Piper ONNX (~50 MB) |
| **VRAM** | ~50 MB (runs CPU ONNX) |
| **Disk** | ~50 MB per variant |
| **Why #10** | Different voice from your existing Mana-Persian-Piper. Available in 3 quality tiers (low/medium/high). Ultra-fast (RTF ~0.2×). Gives a second Persian Piper voice for comparison. |
| **Integration effort** | Trivial — Piper already installed, just download ONNX file |
| **Container** | `engine-current` |

---

## 6. Quality Tiers

### Production-Grade (validated quality)

| Model | Validation |
|-------|-----------|
| `facebook/mms-tts-fas` | Meta official, trained on MMS corpus |
| `saillab/xtts_v2_fa` | MOS 3.6, SMOS 4.0 (ParsVoice paper) |
| `hootan09/ChatterBox` + Persian T3 | ✅ Already in lab, your best Persian engine |

### Good Community Quality

| Model | Notes |
|-------|-------|
| `Kamtera/persian-tts-female-vits` | Most popular community model |
| `saillab/persian-tts-azure-grapheme-60K` | Azure data > Common Voice quality |
| `mah92/Khadijah-FA_EN-Matcha-TTS-Model` | ✅ Already in lab, good bilingual |

### Experimental / Novel

| Model | Notes |
|-------|-------|
| `MOSS-TTS-Nano-100M` | Newest architecture, untested Persian quality |
| `Hamid20/speecht5_tts_persian` | 4000 training steps only, validation loss 0.5369 |
| `karim23657/persian-tts-female-GPTInformal-Persian-vits` | Informal domain, quality may vary |

### Benchmark-Only

| Model | Notes |
|-------|-------|
| `saillab/persian-tts-cv15-reduct-grapheme-multispeaker` | CV15 data = moderate quality |
| `Kamtera/persian-tts-male-vits` | Fine-tuned from female, may have artifacts |
| `rhasspy/piper-voices fa_IR/gyro` | Piper quality ceiling is moderate |

---

## 7. Models NOT Recommended

| Model | Reason to Skip |
|-------|---------------|
| `saillab/persian-tts-grapheme-arm24-finetuned-on1` | Checkpoint variant of azure-grapheme-60K; pick the best one |
| `saillab/multi_speaker` (90K) | Older than cv15-reduct multispeaker; pick the newer one |
| `saillab/female_cv_azure_male_azure_female` | Inferior data mix vs. pure Azure model |
| `saillab/Multi_Speaker_Cv_plus_Azure_female_in_one_set` | Redundant with cv15-reduct |
| `SeyedAli/Persian-Speech-synthesis-MMS` | Mirror of facebook/mms-tts-fas — use the original |
| `Thomcles/Chatterbox-TTS-Persian-Farsi` | Community reports: "poor quality, doesn't sound natural" |
| `Kamtera/persian-tts-female-glow_tts` | Glow-TTS (2019) superseded by VITS (2021); only add if benchmarking architecture |
| `mhrahmani/persian-tts-vits-0` | Gated/private, only 57K steps |
| `SadeghK/persian-text-to-speech` | Base Piper checkpoints, superseded by Mana-Persian-Piper |

---

## 8. Integration Plan

### 8.1 Integration Pattern

Each new engine follows the standard Arthur TTS Lab pattern:

```
1. tts_lab_config.py:    Add MODEL_INFO entry + MODEL_ORDER entry
2. tts_lab_engines.py:   Add _load_<engine>() + _synth_<engine>()
3. docker-compose.yml:   Add {ENGINE}_URL env var in orchestrator
4. Dockerfile:           Add pip install + model download (as needed)
5. docs/engine_compatibility.yaml: Add engine entry
```

### 8.2 Model-by-Model Integration Details

#### #1 facebook/mms-tts-fas — TRIVIAL

```python
# _load_mmsfas():
from transformers import VitsModel, AutoTokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-fas")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
return (model, tokenizer)

# _synth_mmsfas(inst, text, params):
model, tokenizer = inst
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform
return _to_wav(output.numpy().squeeze(), model.config.sampling_rate)
```

**Dependencies:** None new — `transformers` already installed in all stacks.
**Model download:** Automatic via HF cache on first load.
**Container:** `engine-current` (stack: current).

#### #2 MOSS-TTS-Nano-100M — MEDIUM

```bash
# Dockerfile addition:
RUN pip install --no-cache-dir moss-tts-nano
```

```python
# _load_mosstts():
# MOSS-TTS-Nano can be used via Python API or subprocess CLI
# The Python API approach:
import moss_tts_nano
model = moss_tts_nano.MOSSTTSNano.from_pretrained("FenomAI/MOSS-TTS-Nano-100M")
return model

# _synth_mosstts(inst, text, params):
# Reference audio for voice cloning
ref_audio = params.get("reference_audio")  # path to WAV
output = inst.generate(text=text, prompt_speech=ref_audio)
return _to_wav(output, inst.sample_rate)
```

**Dependencies:** `moss-tts-nano` pip package. Check for conflicts with engine-current stack (torch nightly, tf 5.12).
**Container:** `engine-current` initially. If pip conflicts arise, consider dedicated container or engine-mid.
**Risk:** MOSS-TTS-Nano is very new (April 2026). The pip package may have unstable dependencies.

#### #3 saillab/xtts_v2_fa — MEDIUM

```python
# _load_xtts():
# If loading the Persian fine-tune specifically:
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/opt/models/xtts_v2_fa/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path="/opt/models/xtts_v2_fa/best_model_110880.pth")
model.cuda()
return model

# _synth_xtts(inst, text, params):
# XTTS uses speaker reference for cloning
ref_path = params.get("reference_audio")
outputs = inst.synthesize(text, config, speaker_wav=ref_path, language="fa")
return _to_wav(outputs["wav"], 24000)
```

**Alternative:** Add as a voice preset within the existing `xtts` engine — when `voice="fa"` or `language="fa"`, load the Persian checkpoint instead of the base one. This avoids adding a new engine key.

**Dependencies:** Coqui TTS already installed in engine-current (line 43 of Dockerfile.engine-current). Model download needed.
**Container:** `engine-current`.

#### #4-8 VITS Models — LOW

All Kamtera and Saillab VITS models share the same integration pattern via Coqui TTS:

```python
# _load_persian_vits(model_path, config_path):
from TTS.utils.synthesizer import Synthesizer
synthesizer = Synthesizer(
    model_path=str(MODELS_DIR / model_path),
    config_path=str(MODELS_DIR / config_path),
    use_cuda=True,
)
return synthesizer

# _synth_persian_vits(inst, text, params):
wavs = inst.tts(text)
return _to_wav(np.array(wavs), inst.output_sample_rate)
```

**Dependencies:** Coqui TTS already installed.
**Container:** `engine-current`.

#### #9 Hamid20/speecht5_tts_persian — LOW

```python
# _load_speecht5():
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
model = SpeechT5ForTextToSpeech.from_pretrained("Hamid20/speecht5_tts_persian")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
return (model, processor, vocoder)

# _synth_speecht5(inst, text, params):
model, processor, vocoder = inst
inputs = processor(text=text, return_tensors="pt")
# Needs speaker embedding from params or default
speaker_embedding = params.get("speaker_embedding")
speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
return _to_wav(speech.numpy(), 16000)
```

**Dependencies:** `transformers` already installed.
**Container:** `engine-current`.

#### #10 rhasspy/piper-voices fa_IR/gyro — TRIVIAL

Just download the ONNX model files and add a voice option to the existing `piper` engine:

```python
# In _load_piper() — add voice preset:
"fa_IR-gyro-medium": "fa_IR-gyro-medium"  # Auto-discovered from MODELS_DIR
```

No code changes needed beyond model download and voice catalog entry.

---

## 9. Container Placement Strategy

### Decision Matrix

| Model | Stack Req. | Fits engine-current? | Fits engine-mid? | Dedicated? |
|-------|-----------|---------------------|------------------|------------|
| mms-tts-fas | torch ≥2.0, tf ≥4.33 | ✅ Yes | ✅ Yes | No |
| MOSS-TTS-Nano | torch ≥2.0 | ✅ Yes (check) | ✅ Yes | Maybe¹ |
| xtts_v2_fa | coqui-tts | ✅ Yes | ❌ No coqui | No |
| Kamtera VITS (×3) | coqui-tts | ✅ Yes | ❌ No coqui | No |
| Saillab VITS (×2) | coqui-tts | ✅ Yes | ❌ No coqui | No |
| speecht5_fa | tf ≥4.36 | ✅ Yes | ✅ Yes | No |
| piper-gyro | piper-tts | ✅ Yes | ❌ No piper | No |

> ¹ **MOSS-TTS-Nano risk:** The `moss-tts-nano` pip package may pin specific torch/transformers versions. If it conflicts with engine-current (torch nightly + tf 5.12), place it in a **new dedicated container** based on `stack-current` but with its own pip constraints.

### Recommended Placement

**Option A: All in engine-current (simplest)**
```
engine-current (port 8101)
  ├── [existing] 21 engines
  └── [+new] 8 Persian models
      ├── mmsfas       (~200 MB VRAM)
      ├── mosstts      (~300 MB VRAM)
      ├── xtts_fa      (~2 GB VRAM, replaces base xtts for Persian)
      ├── kamtera_f    (~500 MB VRAM)
      ├── kamtera_m    (~500 MB VRAM)
      ├── saillab_az   (~500 MB VRAM)
      ├── saillab_cv   (~500 MB VRAM)
      └── speecht5_fa  (~500 MB VRAM)
```

**Pros:** No new containers. Reuses existing infrastructure. Single `docker compose up -d`.
**Cons:** Single engine at a time (VRAM eviction). All models compete for 16 GB.

**Option B: Dedicated Persian container (cleanest)**

Create a new `engine-persian` container for Persian-only models:

```
engine-persian (port 8105, profile: persian)
  FROM tts-lab-stack-current:latest  ← reuse stack
  ├── mmsfas, mosstts, xtts_fa
  ├── Kamtera female, male
  ├── Saillab Azure, CV multi-speaker
  ├── speecht5_fa, piper-gyro
  ├── [existing Persian engines could also route here]
  └── Total VRAM: ~5 GB (one at a time via eviction)
```

**Pros:** Clean separation. Persian models can load/unload independently of main engine-current. Can run simultaneously with other containers. Easier to manage Persian-specific dependencies.
**Cons:** New container to build and maintain. Additional port allocation. New Dockerfile.

### Recommendation: Option A (all in engine-current) for phase 1, with Option B as a fast-follow if VRAM pressure or dependency conflicts arise.

---

## 10. VRAM Budget Analysis

### RTX 5060 Ti 16 GB GDDR7 — Current VRAM Allocation

| Container | Peak VRAM | Notes |
|-----------|-----------|-------|
| engine-current | ~12 GB | Bark engine at full load |
| engine-mid | ~9 GB | Higgs at full load (INT8: ~5-6 GB) |
| engine-qwen | ~3 GB | Qwen3TTS 1.7B |
| engine-legacy | ~4 GB | IndexTTS |
| **Total concurrent** | **~28 GB** | ❌ Exceeds 16 GB |

> **Important:** The lab uses single-engine-at-a-time loading with VRAM eviction (`_evict_current()`). Only ONE engine is loaded in each container at any time. New models don't increase peak VRAM — they add more options that fit within the existing per-container budget.

### New Models — Per-Model VRAM

| Model | VRAM (loaded) | Fits in 16 GB? |
|-------|--------------|----------------|
| mms-tts-fas | ~200 MB | ✅ Trivial |
| MOSS-TTS-Nano | ~300 MB | ✅ Trivial |
| xtts_v2_fa | ~2 GB | ✅ Yes |
| Kamtera female VITS | ~500 MB | ✅ Yes |
| Kamtera male VITS | ~500 MB | ✅ Yes |
| Saillab Azure VITS | ~500 MB | ✅ Yes |
| Saillab CV multi-speaker | ~500 MB | ✅ Yes |
| GPTInformal VITS | ~500 MB | ✅ Yes |
| speecht5_fa | ~500 MB | ✅ Yes |
| piper-gyro | ~50 MB | ✅ Trivial |

**Worst-case concurrent (same container):** If the eviction mechanism fails and two models load simultaneously, max additional VRAM is ~2.5 GB (xtts_v2_fa + one VITS). Still well within budget.

---

## 11. Implementation Phases

### Phase 1: Quick Wins (1-2 hours)

1. **facebook/mms-tts-fas** — Add to engine-current. Trivial integration. Immediately gives you a Meta reference baseline.
2. **rhasspy/piper-voices fa_IR/gyro** — Download ONNX files, add voice preset to existing piper engine.
3. **Hamid20/speecht5_tts_persian** — Add to engine-current. Standard transformers API.

**Deliverable:** 3 new Persian engines running. Zero new dependencies.

### Phase 2: Community Voices (2-4 hours)

4. **Kamtera/persian-tts-female-vits** — Coqui TTS integration.
5. **Kamtera/persian-tts-male-vits** — Same pattern.
6. **saillab/persian-tts-azure-grapheme-60K** — Same pattern.
7. **saillab/persian-tts-cv15-reduct-grapheme-multispeaker** — Same pattern, plus speaker selection UI.
8. **karim23657/persian-tts-female-GPTInformal-Persian-vits** — Same pattern.

**Deliverable:** 5 additional Persian voices. Full community VITS coverage.

### Phase 3: Architecture Gems (4-6 hours)

9. **MOSS-TTS-Nano-100M** — Evaluate pip compatibility. If clean, add to engine-current. If conflicts, create `Dockerfile.engine-persian`.
10. **saillab/xtts_v2_fa** — Replace or augment existing xtts engine with Persian checkpoint. Update UI to show "Persian" as a language option for XTTS.

**Deliverable:** The two most architecturally interesting additions. Complete top-10 coverage.

### Phase 4: Production Polish (2-4 hours, optional)

- Add Persian-specific text preprocessing (already have `_process_persian_text()` in tts_lab_engines.py)
- Add Persian voice library entries for the new engines
- Run Persian synthesis sweep across all 16 Persian-capable engines
- Update `docs/engine_compatibility.yaml` with validation results
- Update `docs/reference/PERSIAN_TTS_MODELS.md`

---

## 12. Engine Catalog Entries

### Proposed MODEL_INFO entries (for tts_lab_config.py)

```python
"mmsfas": {
    "label": "MMS Persian (Meta)", "size": "~150 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 200, "heavy": False,
    "notes": "Meta MMS-TTS Persian VITS. Reference baseline. Non-deterministic.",
    "arthur_fit": 5,
},
"mosstts": {
    "label": "MOSS-TTS Nano", "size": "~200 MB", "rtf_est": "RTF ~1.0×",
    "ram_est_mb": 300, "heavy": False,
    "notes": "LLM-based (0.1B params). 20 languages. Streaming. Apache 2.0.",
    "arthur_fit": 4,
},
"kamtera_f": {
    "label": "Kamtera Persian ♀", "size": "~200 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "Community VITS female Persian voice. Most popular (104 dl/mo).",
    "arthur_fit": 5,
},
"kamtera_m": {
    "label": "Kamtera Persian ♂", "size": "~200 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "Community VITS male Persian voice.",
    "arthur_fit": 5,
},
"saillab_az": {
    "label": "Saillab Azure FA", "size": "~200 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "VITS trained on Azure TTS Persian data. Grapheme-based, no phonemizer.",
    "arthur_fit": 5,
},
"saillab_cv": {
    "label": "Saillab CV Multi-FA", "size": "~250 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "Multi-speaker VITS (Common Voice 15). Built-in speaker selection.",
    "arthur_fit": 4,
},
"gptinf_fa": {
    "label": "GPTInformal Persian", "size": "~200 MB", "rtf_est": "RTF ~0.5×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "Trained on informal/colloquial Persian. Handles slang and casual speech.",
    "arthur_fit": 3,
},
"speecht5_fa": {
    "label": "SpeechT5 Persian", "size": "~400 MB", "rtf_est": "RTF ~1.0×",
    "ram_est_mb": 500, "heavy": False,
    "notes": "Fine-tuned Microsoft SpeechT5 for Persian. Encoder-decoder transformer.",
    "arthur_fit": 4,
},
"piper_gyro": {
    "label": "Piper Gyro Persian", "size": "~50 MB", "rtf_est": "RTF ~0.2×",
    "ram_est_mb": 50, "heavy": False,
    "notes": "Alternative Piper ONNX Persian voice (gyro-medium). Ultra-fast CPU.",
    "arthur_fit": 5,
},
```

### Proposed MODEL_ORDER additions

```python
# Insert after existing Persian engines in MODEL_ORDER:
MODEL_ORDER = [
    # ... existing engines ...
    "mmsfas", "mosstts",
    "kamtera_f", "kamtera_m",
    "saillab_az", "saillab_cv",
    "gptinf_fa", "speecht5_fa",
    # ... rest of existing engines ...
]
```

---

## Appendix A: Persian TTS Quality Hierarchy (Combined — Existing + New)

After integrating all 10 new models, the Persian quality ranking would be:

| Rank | Engine | Type | Quality |
|------|--------|------|---------|
| 1 | chatterbox (Persian T3) | Chatterbox 0.5B | ⭐⭐⭐⭐⭐ Best overall |
| 2 | higgs (INT8) | Higgs 4B | ⭐⭐⭐⭐ (pending validation) |
| 3 | saillab xtts_v2_fa | XTTS v2 fine-tune | ⭐⭐⭐⭐ MOS 3.6, SMOS 4.0 |
| 4 | matcha (Khadijah) | Matcha-TTS FA-EN | ⭐⭐⭐⭐ Good bilingual |
| 5 | mms-tts-fas | VITS (Meta) | ⭐⭐⭐ Reference baseline |
| 6 | manatts | Tacotron2 | ⭐⭐⭐ Good single-speaker |
| 7 | fishspeech | Fish Speech | ⭐⭐⭐ Decent zero-shot |
| 8 | f5tts | F5-TTS | ⭐⭐⭐ Good with ref audio |
| 9 | Kamtera female VITS | VITS | ⭐⭐½ Best community |
| 10 | Saillab Azure VITS | VITS | ⭐⭐½ Clean studio sound |
| 11 | Mana-Persian-Piper | Piper ONNX | ⭐⭐½ Fast, reliable |
| 12 | Piper Gyro | Piper ONNX | ⭐⭐½ Alternative voice |
| 13 | Saillab CV Multi | VITS | ⭐⭐ Multi-speaker |
| 14 | Kamtera male VITS | VITS | ⭐⭐ Male option |
| 15 | GPTInformal VITS | VITS | ⭐⭐ Colloquial domain |
| 16 | MOSS-TTS-Nano | LLM 0.1B | ⭐⭐ (untested Persian) |
| 17 | SpeechT5 Persian | SpeechT5 | ⭐½ (4000 steps only) |
| 18 | Qwen3TTS | Qwen3 1.7B | ⭐½ (experimental) |
| 19 | XTTS (base) | XTTS v2 generic | ⭐ Poor Persian |
| 20 | CosyVoice | CosyVoice 3 | ❌ No Persian support |
| 21 | Bark | Bark | ❌ No Persian support |

---

## Appendix B: Sources

- [facebook/mms-tts-fas](https://huggingface.co/facebook/mms-tts-fas) — Meta MMS Persian TTS
- [FenomAI/MOSS-TTS-Nano-100M](https://huggingface.co/FenomAI/MOSS-TTS-Nano-100M) — MOSS-TTS Nano
- [saillab/xtts_v2_fa](https://huggingface.co/saillab/xtts_v2_fa) — Persian XTTS v2 fine-tune
- [Kamtera/persian-tts-female-vits](https://huggingface.co/Kamtera/persian-tts-female-vits) — Female Persian VITS
- [Kamtera/persian-tts-male-vits](https://huggingface.co/Kamtera/persian-tts-male-vits) — Male Persian VITS
- [saillab/persian-tts-azure-grapheme-60K](https://huggingface.co/saillab/persian-tts-azure-grapheme-60K) — Azure VITS
- [saillab/persian-tts-cv15-reduct-grapheme-multispeaker](https://huggingface.co/saillab/persian-tts-cv15-reduct-grapheme-multispeaker) — Multi-speaker VITS
- [karim23657/persian-tts-female-GPTInformal-Persian-vits](https://huggingface.co/karim23657/persian-tts-female-GPTInformal-Persian-vits) — Informal Persian VITS
- [Hamid20/speecht5_tts_persian](https://huggingface.co/Hamid20/speecht5_tts_persian) — SpeechT5 Persian
- [rhasspy/piper-voices fa_IR](https://huggingface.co/rhasspy/piper-voices/tree/main/fa_IR) — Piper Persian voices
- [saillab/ZabanZad_PoC](https://huggingface.co/spaces/saillab/ZabanZad_PoC) — Persian TTS comparison space
- [karim23657 Persian TTS Collection](https://huggingface.co/collections/karim23657/persian-tts-text-to-speech)
- ParsVoice paper: arXiv 2510.10774 — MOS 3.6 for XTTS on Persian
- [MahtaFetrat/Mana-Persian-Piper](https://huggingface.co/MahtaFetrat/Mana-Persian-Piper) — Mana Persian Piper
- [mah92/Khadijah-FA_EN-Matcha-TTS-Model](https://huggingface.co/mah92/Khadijah-FA_EN-Matcha-TTS-Model) — Khadijah Matcha-TTS
