# TTS Engine Deep-Comparison Reference
> Live benchmark: Xeon D-1528 @ 1.90 GHz, 12 vCores, AVX2, no AVX-512, Ubuntu 22.04
> Test phrase (long): *"Oh my goodness, just a moment dear, let me find my reading glasses. Now you said I owe money to the IRS? Can you give me that case number again, nice and slow?"*  (~11–14s audio
> All RTF numbers measured warm (model already resident in RAM) unless noted.
> Updated: 2026-04-23  Added: Qwen3-TTS 1.7B measured GPU results, UI redesign notes

---

## 1. Master Comparison Table

| Engine | Installed | Warm RTF ▲ | Audio dur | Load time | RAM | Real-time? | Voice clone | Emotion tags | Arthur fit |
|---|---|---|---|---|---|---|---|---|---|
| **Piper TTS** | ✅ | **0.08×** | 11 017 ms | 4.4s | 200 MB | ✅ | ❌ | ❌ | ⭐⭐ |
| **MeloTTS** | ✅ | **1.08×** | 9 892 ms | 2.5s | 1 200 MB | ✅* | ❌ | ❌ | ⭐⭐⭐ |
| **StyleTTS 2** | ✅ | **1.67×** | 11 247 ms | 9.5s | 1 500 MB | ❌ | ✅ ref-WAV | ❌ | ⭐⭐⭐⭐ |
| **Kokoro-82M** | ✅ | **3.07×** | 12 032 ms | 1.8s | 500 MB | ❌ | ❌ | ❌ | ⭐⭐⭐⭐⭐ |
| **XTTS-v2** | ✅ | **4.74×** | 11 916 ms | 26.1s | 3 200 MB | ❌ | ✅ ref-WAV | ❌ | ⭐⭐⭐⭐⭐ |
| **Qwen3-TTS 1.7B** | ✅ | **4.41×** | 4 400 ms | 6.7s | 2 000 MB | ❌ | ✅ ref-WAV | ✅ instruct | ⭐⭐⭐ |
| **F5-TTS** | ✅ | **~5×** est | — | 15s | 2 000 MB | ❌ | ✅ REQUIRED | ❌ | ⭐⭐⭐⭐ |
| **Chatterbox** | ✅ | **11.7×** | 7 000 ms | 20.3s | 1 800 MB | ❌ | ✅ optional | ❌ | ⭐⭐⭐⭐⭐ |
| **Bark** | ✅ | **20.3×** | 14 906 ms | 16.3s | 1 500 MB | ❌ | ❌ | ✅ BEST | ⭐⭐⭐⭐⭐ |
| **Parler-TTS** | ✅ | **23.4×** | 14 338 ms | 24.0s | 1 500 MB | ❌ | ❌ | ❌ | ⭐⭐⭐⭐ |
| **Dia-1.6B** | ✅ | **~55×** est | — | 22s | 3 000 MB | ❌ | ❌ | ✅ | ⭐⭐⭐⭐⭐ |
| **CosyVoice2** | ❌ | N/A | — | — | 2 500 MB | ❌ | ✅ | ❌ | ⭐⭐⭐ |

*MeloTTS degrades to ~1.8× when other heavy models are RAM-resident simultaneously.
RTF < 1.0 = faster than real-time. RTF > 1.0 = slower than real-time (cannot be used in live phone calls).

---

## 2. Voice Quality & Character Analysis

| Engine | Naturalness | Robotic feel | Prosody variation | Expressiveness | Consistency | Elderly Arthur fit |
|---|---|---|---|---|---|---|
| **Piper** | ⭐⭐ | 🤖🤖🤖🤖 | Very flat | None | ✅ Very consistent | Poor — monotone robotic |
| **Kokoro** | ⭐⭐⭐ | 🤖🤖🤖 | Moderate | None | ✅ Consistent | Good accent (bm_lewis), still robotic cadence |
| **MeloTTS** | ⭐⭐⭐ | 🤖🤖 | Moderate | None | ✅ Consistent | Better — EN-BR sounds older |
| **StyleTTS 2** | ⭐⭐⭐⭐ | 🤖 | Good | None | ✅ Good | Very natural — matches reference style |
| **XTTS-v2** | ⭐⭐⭐⭐ | 🤖 | Good | None | ⚠️ Slight variance | Very natural multi-speaker |
| **F5-TTS** | ⭐⭐⭐⭐⭐ | None | Excellent | None | ⚠️ Varies w/ ref | Best naturalness (needs ref WAV) |
| **Chatterbox** | ⭐⭐⭐⭐ | None | Good | ✅ Exaggeration | ⚠️ Slight variance | Best confusion/hesitation |
| **Bark** | ⭐⭐⭐ | None | Very good | ✅✅ Best tokens | ⚠️ Inconsistent | Best emotion tokens, occasional distortion |
| **Parler-TTS** | ⭐⭐⭐⭐ | None | Very good | ✅ Via prompt | ⚠️ Prompt-sensitive | Prompt fully controls character |
| **Dia-1.6B** | ⭐⭐⭐⭐⭐ | None | Excellent | ✅ Best dialogue | ✅ Good | Best two-person dialogue, most realistic |
| **CosyVoice2** | ⭐⭐⭐⭐ | None | Good | None | ✅ Good | Not installed |

---

## 3. Technical Architecture

| Engine | Inference backend | Precision | Quantised? | AVX-512 needed? | Multi-thread benefit |
|---|---|---|---|---|---|
| **Piper** | ONNX Runtime | INT8 (VITS) | ✅ | ❌ | Moderate (ONNX parallel) |
| **Kokoro** | ONNX Runtime | INT8 | ✅ | ❌ | Moderate |
| **MeloTTS** | PyTorch | FP32 | ❌ | ❌ | Good (MKL BLAS) |
| **StyleTTS 2** | PyTorch (diffusion) | FP32 | ❌ | ❌ | Good |
| **XTTS-v2** | PyTorch (GPT-2 AR) | FP32 | ❌ | ❌ | Poor (autoregressive) |
| **F5-TTS** | PyTorch (flow match) | FP32 | ❌ | ❌ | Moderate |
| **Chatterbox** | PyTorch (AR codec) | FP32 | ❌ | ❌ | Poor (autoregressive) |
| **Bark** | PyTorch (GPT AR × 3) | FP32 | ❌ | ❌ | Poor (3 serial AR passes) |
| **Parler-TTS** | PyTorch (T5+codec AR) | FP32 | ❌ | ❌ | Poor (autoregressive) |
| **Dia-1.6B** | PyTorch (1.6B AR) | FP32 | ❌ | ❌ | Very poor (large AR) |
| **CosyVoice2** | PyTorch | FP32 | ❌ | ❌ | — |

**Why autoregressive models are slow on this CPU:**
Each token is generated sequentially — threads cannot help with the serial dependency chain.
Only the weight matrix multiplications within each token step benefit from parallelism.
Piper and Kokoro use ONNX feedforward (all tokens in parallel) — hence their speed advantage.

---

## 4. Voice Options & Customisation

| Engine | # voices / speakers | Selection method | Languages | Custom voice? |
|---|---|---|---|---|
| **Piper** | 6 on disk (900+ online) | `.onnx` file per voice | 30+ languages | Download .onnx |
| **Kokoro** | 54 in voices.bin | Dropdown (prefix-based) | 9 languages | Not supported |
| **MeloTTS** | 5 accents | Dropdown | English only | Not supported |
| **StyleTTS 2** | 1 default + reference | Upload WAV → style | English | ✅ Upload reference |
| **XTTS-v2** | 58 built-in | Dropdown | 17 languages | ✅ Upload reference |
| **F5-TTS** | ∞ (zero-shot) | Upload WAV | EN + multilingual | ✅ Required |
| **Chatterbox** | 1 default | Exaggeration slider | English | ✅ Upload reference |
| **Bark** | 10 presets | Dropdown | EN + multilingual | ❌ (preset only) |
| **Parler-TTS** | ∞ (text-driven) | Free text description | English | ❌ (prompt only) |
| **Dia-1.6B** | 2 speakers (S1/S2) | Tag in text | English | ❌ |
| **CosyVoice2** | Several presets | Dropdown | Chinese-first | ✅ Upload reference |

---

## 5. Control Parameters

### Piper TTS
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `voice` | select | 6 on disk | `en_GB-alan-medium` | Voice file to use |
| `speed` | slider | 0.5–2.0 | 1.0 | Speaking rate |
| `length_scale` | slider | 0.5–2.0 | 1.0 | Duration (inverse speed) |
| `noise_scale` | slider | 0.1–1.5 | 0.667 | Voice variation / naturalness |
| `noise_w` | slider | 0.1–1.5 | 0.8 | Duration variation |

### Kokoro-82M
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `voice` | select | 54 voices | `bm_lewis` | Voice embedding |
| `speed` | slider | 0.5–1.5 | 0.85 | Speaking rate |
| `lang` | auto | from prefix | `en-gb` | Auto-set from voice name |

### MeloTTS
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `speaker` | select | 5 accents | `EN-US` | Accent variant |
| `speed` | slider | 0.5–1.5 | 0.85 | Speaking rate |

### StyleTTS 2
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `alpha` | slider | 0–1 | 0.3 | Style weight: 0=copy ref exactly, 1=default style |
| `beta` | slider | 0–1 | 0.7 | Prosody weight: 0=copy ref prosody, 1=default |
| `diffusion_steps` | slider | 3–15 | 5 | Quality (more=better+slower) |
| `embedding_scale` | slider | 0.5–3 | 1.0 | Voice embedding strength |
| reference WAV | upload | optional | none | Sets voice timbre & style |

### XTTS-v2
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `speaker` | select | 58 built-in | `Torcull Diarmuid` | Built-in speaker embedding |
| `language` | select | 17 languages | `en` | Target language |
| reference WAV | upload | optional | none | Override built-in voice |

### F5-TTS
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| reference WAV | upload | **REQUIRED** | — | Source voice to clone |
| `ref_text` | text | string | `""` | Exact transcript of reference clip |
| `speed` | slider | 0.5–2.0 | 1.0 | Speaking rate |
| `nfe_step` | slider | 8–64 | 32 | Diffusion steps (quality vs speed) |

### Bark
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `voice_preset` | select | 10 EN presets | `v2/en_speaker_6` | Speaker identity |
| Emotion tokens in text | inline | see table | none | `[laughs]` `[sighs]` etc. |
| `temperature` | slider | 0.5–1.0 | 0.7 | Randomness / expressiveness |

### Chatterbox
| Parameter | Type | Range | Default | Arthur sweet spot |
|---|---|---|---|---|
| `exaggeration` | slider | 0.0–1.0 | 0.65 | **0.5–0.7** (confused elderly) |
| `cfg_weight` | slider | 0.1–1.0 | 0.5 | Lower = more natural |
| `seed` | number | 0–9999 | 0 (random) | Fix for reproducibility |
| reference WAV | upload | optional | none | Voice cloning |

### Parler-TTS
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `description` | textarea | free text | *default prompt* | Controls ALL voice characteristics |
| `temperature` | slider | 0.5–2.0 | 1.0 | Randomness |
| `max_new_tokens` | slider | 256–2048 | 1024 | Output length cap |

### Dia-1.6B
| Parameter | Type | Range | Default | Effect |
|---|---|---|---|---|
| `[S1]`/`[S2]` in text | inline | tag | — | Speaker turn |
| `[laughs]`/`[sighs]` etc. | inline | tag | — | Emotion sound |
| `cfg_scale` | slider | 1.0–5.0 | 3.0 | Guidance strength |
| `temperature` | slider | 0.5–2.0 | 1.2 | Randomness |
| `top_p` | slider | 0.5–1.0 | 0.95 | Nucleus sampling |
| `max_tokens` | slider | 0=auto | auto | 0 = estimate from text length |

---

## 6. Emotion & Expressiveness Tokens

Only Bark and Dia support non-verbal sounds embedded directly in the transcript.

### Bark tokens
| Token | Effect | Reliability |
|---|---|---|
| `[laughs]` | Natural laugh inserted | ✅ Reliable |
| `[sighs]` | Audible sigh | ✅ Reliable |
| `[clears throat]` | Throat clearing | ✅ Reliable |
| `[hesitantly]` | Hesitant tone | ⚠️ Subtle |
| `[gasps]` | Short gasp | ✅ Reliable |
| `[long pause]` | Extended pause | ✅ Reliable |
| `[nervously]` | Nervous tone | ⚠️ Subtle |
| `[quietly]` | Lower volume/whisper | ✅ Reliable |
| `[MAN]` / `[WOMAN]` | Force gender | ✅ Reliable |
| `[music]` | Background music | ⚠️ Unpredictable |

**Arthur example with Bark:**
```
Hello? [sighs] Oh my goodness, who is this dear? [clears throat]
You said I owe money to the IRS? [hesitantly] Just a moment, let me find my glasses.
[long pause] Now you said... the case number was...? Can you say that again, nice and slow?
```

### Dia-1.6B tokens
| Token | Effect | Reliability |
|---|---|---|
| `[S1]` / `[S2]` | Speaker turn | ✅ Core feature |
| `[laughs]` | Laugh | ✅ Reliable |
| `[sighs]` | Sigh | ✅ Reliable |
| `[coughs]` | Cough | ✅ Reliable |
| `[groans]` | Groan | ✅ Reliable |
| `[gasps]` | Gasp | ✅ Reliable |
| `[sobs]` | Sob | ✅ Reliable |
| `[clears throat]` | Throat clear | ✅ Reliable |

**Arthur example with Dia (dialogue mode):**
```
[S1] Hello? [sighs] Oh my goodness.
[S2] Good afternoon, this is the IRS fraud division.
[S1] Oh dear. [gasps] The IRS? I've been expecting your call. [laughs nervously]
[S2] We need to verify your social security number immediately.
[S1] Just a moment dear, let me find my reading glasses. [clears throat] Now... you said the case number was?
```

---

## 7. Disk & RAM Footprint

| Engine | Model files | Model disk | Runtime RAM | RAM/quality ratio |
|---|---|---|---|---|
| **Piper** | 1 ONNX per voice | 25–116 MB/voice | **200 MB** | 🏆 Extremely efficient |
| **Kokoro** | 1 ONNX + 1 voices.bin | **116 MB total** for all 54 | **500 MB** | 🏆 Exceptionally efficient |
| **MeloTTS** | HF cache | 199 MB | **1 200 MB** | ✅ Good |
| **StyleTTS 2** | HF cache | ~700 MB | **1 500 MB** | ✅ Good |
| **F5-TTS** | HF cache | 1 300 MB | **2 000 MB** | ✅ Good |
| **Bark** | Local cache | 4 400 MB | **1 500 MB** | ⚠️ Large download, good RAM |
| **Chatterbox** | HF cache | 3 000 MB | **1 800 MB** | ⚠️ Heavy download |
| **Parler-TTS** | HF cache | 3 300 MB | **1 500 MB** | ⚠️ Heavy |
| **XTTS-v2** | Local cache | 1 800 MB | **3 200 MB** | ❌ RAM hungry |
| **Dia-1.6B** | HF cache | 6 100 MB | **3 000+ MB** | ❌ Very heavy |
| **CosyVoice2** | Manual clone | ~2 000 MB | **2 500 MB** | — not installed |

---

## 8. Latency Breakdown

For each engine, where does time actually go on this CPU?

| Engine | Tokenisation | Model forward | Vocoder / codec | Total bottleneck |
|---|---|---|---|---|
| **Piper** | <1 ms | ~80 ms/s audio (ONNX, parallel) | built-in | Forward pass — parallelises well |
| **Kokoro** | <1 ms | ~400 ms/s audio (ONNX) | built-in | Forward pass — ONNX overhead |
| **MeloTTS** | <5 ms | ~400 ms/s audio (PyTorch) | built-in | Forward pass — MKL helps |
| **StyleTTS 2** | <5 ms | ~200 ms/s audio (diffusion) | built-in | Diffusion steps — reduce for speed |
| **XTTS-v2** | ~100 ms | ~500 ms/token (AR GPT) | EnCodec | Each AR token sequential |
| **F5-TTS** | <5 ms | ~300 ms/s audio (flow) | built-in | Flow matching — diffusion-like |
| **Chatterbox** | ~100 ms | ~500 ms/token (AR) | EnCodec | Sequential codec token gen |
| **Bark** | ~50 ms | ~600 ms/token × 3 passes | EnCodec | Three serial AR stages |
| **Parler-TTS** | ~50 ms | ~800 ms/token (T5+codec AR) | built-in | Long token sequence |
| **Dia-1.6B** | <5 ms | ~1000 ms/token (1.6B AR) | DAC | 1.6B params per token step |

---

## 9. Arthur Persona Fit — Detailed

Arthur is a **confused elderly British man** who is easily confused, rambles, takes pauses, asks questions to repeat themselves, occasionally sighs or clears his throat.

| Engine | Best voice/setting | Why it fits | What's missing |
|---|---|---|---|
| **Piper `en_GB-alan-medium`** | Alan, speed=0.85 | British accent, fast | Monotone, zero variation, robotic |
| **Kokoro `bm_lewis`** | bm_lewis, speed=0.85 | Best British male, smooth | Still synthetic cadence |
| **MeloTTS `EN-BR`** | British, speed=0.80 | Slightly older sound | Flat affect, no confusion |
| **StyleTTS 2** | alpha=0.1, beta=0.3, steps=10 | Most natural prosody | Needs reference WAV for old voice |
| **XTTS-v2 `Torcull Diarmuid`** | lang=en, speed=0.9 | Elderly quality to voice | No emotion control |
| **F5-TTS** | Upload elderly male WAV | Clones any voice exactly | Requires high quality reference |
| **Chatterbox** | exaggeration=0.6, cfg=0.4 | Hesitation and confusion baked in | Slow; occasional unnatural pauses |
| **Bark `v2/en_speaker_7`** | en_speaker_7, tokens in text | `[sighs]` `[clears throat]` real sounds | Inconsistent; occasional garbling |
| **Parler-TTS** | Long description | Fully controlled by description | Slow; prompt engineering needed |
| **Dia-1.6B** | `[S1]` turn + emotion tags | Native dialogue + `[laughs nervously]` | Extremely slow; 55× RTF |

**Best Arthur description for Parler-TTS:**
```
An elderly British man in his mid-seventies with a warm, slightly confused, meandering voice.
He speaks slowly and hesitatingly, with gentle pauses as if searching for words.
His tone is friendly but bewildered. He sometimes repeats himself slightly.
The recording quality is slightly warm, as if from an older telephone handset.
```

**Best Bark preset + tokens for Arthur:**
- Voice: `v2/en_speaker_7` (male, elderly)
- Text format: include `[sighs]`, `[clears throat]`, `[hesitantly]` liberally
- Temperature: 0.6 (less random = more consistent)

**Best Chatterbox settings for Arthur:**
- `exaggeration = 0.60` — adds confused hesitation without sounding theatrical
- `cfg_weight = 0.40` — keeps delivery natural
- Upload a 10–15s clip of an elderly British male voice for maximum effect

---

## 10. Production Feasibility

| Engine | Phone call use | Best use case | Blocker |
|---|---|---|---|
| **Piper** | ✅ YES | Production TTS today | Robotic — test other voices |
| **MeloTTS** | ✅ YES* | Backup / fallback | RAM pressure degrades RTF |
| **StyleTTS 2** | ❌ 1.67× too slow | Voice style evaluation | 1.67× RTF on long text |
| **Kokoro** | ❌ 3.07× | Voice quality reference | Clock-speed bound |
| **XTTS-v2** | ❌ 4.74× | Multi-speaker evaluation | Autoregressive, 3 GB RAM |
| **F5-TTS** | ❌ ~5× | Voice cloning research | Requires reference WAV |
| **Chatterbox** | ❌ 11.7× | Emotion exploration | Slow |
| **Bark** | ❌ 20.3× | Emotion token evaluation | 3 AR passes |
| **Parler-TTS** | ❌ 23.4× | Description tuning | Slowest practical model |
| **Dia-1.6B** | ❌ ~55× | Dialogue naturalness research | 1.6B params on CPU |

*MeloTTS marginal — keep only Piper+Melo loaded, no other models.

**GPU upgrade impact (GTX 1060 6 GB):**
All models would become real-time or better.
Piper: ~0.01×  Kokoro: ~0.1×  StyleTTS2: ~0.05×  XTTS: ~0.15×  Bark: ~0.5×

---

## 11. All 11 Known Bugs Fixed

| # | Engine | Bug | Fix |
|---|---|---|---|
| 1–2 | deploy.ps1 | Wrong VM IP (.153); plink/password auth | Fixed IP; SSH key auth |
| 3–4 | .sh files | CRLF line endings; UTF-8 BOM | PowerShell LF+no-BOM conversion |
| 5–6 | setup_tts_lab.sh | set -e killed script; wrong pip name `melo-tts` | Removed set -e; corrected name |
| 7 | XTTS | `TTS` package abandoned | Changed to `coqui-tts 0.27.5` |
| 8–9 | Kokoro | 404 URL; wget 0-byte files | Fixed release tag; switched to curl -L |
| 10 | MeloTTS | MeCab error on import | `sudo python -m unidic download` |
| 11–12 | Piper | synthesize() yields AudioChunk; sample_rate moved | Rewrote to iterate chunks; flat attr |
| 13–14 | MeloTTS | spk2id is HParams not dict; EN_INDIA underscore | dict() wrap; fixed key |
| 15 | Parler | transformers 5.x removed SlidingWindowCache | Pinned transformers==4.46.1 |
| 16–17 | XTTS | 3 missing symbols; COQUI_TOS interactive EOF | Monkey-patch; env var |
| 18–19 | Threading | PyTorch 6 threads; Whisper 0 threads | set_num_threads(12); cpu_threads=12 |
| 20 | Availability | importlib.find_spec passes on broken imports | Replaced with live exec() test |
| 21–23 | protobuf | Missing builder module; old transformers; tilde dirs | Upgraded to 6.x; cleaned dirs |
| 24–25 | Bark/StyleTTS2 | weights_only=True blocks numpy/getattr globals | Patch torch.load before load only |
| 26 | StyleTTS2 | ref_audio= renamed to target_voice_path= | Fixed param name |
| 27–28 | Dia | speed_factor removed; max_tokens=3072 hangs 35min | Removed; auto-estimate |
| 28b | Dia | Dia-1.6B config missing encoder_config | Try Dia-1.6B-0626 first |
| 29 | StyleTTS2 | NLTK punkt_tab for arthur user, runs as root | sudo nltk.download |
| **30** | **Chatterbox** | **perth.PerthImplicitWatermarker=None → TypeError** | **perth.PerthImplicitWatermarker = perth.DummyWatermarker** |

---

## 12. What to Listen to First

To find the best Arthur voice, synthesise this exact text with each engine:

```
Hello? [sighs] Oh, my goodness dear. Who is this? Just a moment, let me find my glasses.
You said I owe money to the IRS? Oh dear. Can you give me that case number again — nice and slow?
```

**Priority order for evaluation:**
1. 🥇 **Kokoro `bm_lewis`** — British, natural, 54 voices, 116 MB total, pick in <5min
2. 🥈 **Chatterbox exaggeration=0.6** — most confused/elderly feel
3. 🥉 **Bark `v2/en_speaker_7`** + `[sighs]` `[clears throat]` — most expressive
4. **StyleTTS2** with uploaded elderly reference WAV
5. **XTTS `Torcull Diarmuid`** — good elderly quality
6. **Parler-TTS** with the long description above
7. **Dia** — when you want full two-person dialogue
8. **Piper `en_GB-alan-medium`** — fastest, most robotic, good as fallback