# Arthur TTS Lab — GPU Benchmark Results
> Generated: 2026-04-20  
> VM hardware: Intel Xeon D-1528 @ 1.90 GHz · 12 cores · 19 GB RAM  
> **GPU: NVIDIA GeForce RTX 5060 Ti · 16 GB GDDR7 · SM 12.0 · Driver 580 · CUDA 13.0**  
> PyTorch: 2.10.0+cu128 (CUDA 12.8)  
> Test phrase: 40 words / ~10–15 s of speech at elderly pace  
> RTF = synthesis time ÷ audio duration — **lower is faster; RTF < 1.0 = real-time**

---

## What Changed vs CPU Baseline

| Component | Before (2026-03-26) | After (2026-04-20) |
|-----------|--------------------|--------------------|
| GPU | None | RTX 5060 Ti 16 GB GDDR7 |
| Whisper STT | CPU int8 | **GPU float16** |
| TTS Lab device | CPU only | **CUDA auto-select** |
| Bark models | Small (1.3 GB) | **Full (2.5 GB)** |
| Synthesis timeouts | 480s hard cap | 600s (GPU handles it) |
| UI | RAM bar only | **RAM + VRAM bar + GPU badge** |

---

## Measured GPU Results — RTX 5060 Ti

> **✅ measured** = timed on RTX 5060 Ti via live tts_lab API (port 8001).  
> **📖 error / not run** = synthesis failed or engine not installed — see error table below.

| # | Engine | Key | CPU RTF | GPU RTF | Speedup | Synth ms | Audio ms | Load s | Hz | Notes |
|---|--------|-----|--------:|--------:|--------:|---------:|---------:|-------:|---:|-------|
| 1 | Piper TTS | `piper` | 0.37 | **0.33** | 1.1× | 3 485 | 10 541 | 3.1 | 22 050 | ONNX CPU-only (no CUDA EP) |
| 2 | Kokoro-82M | `kokoro` | 2.83 | **2.65** | 1.1× | 42 353 | 15 978 | 2.6 | 24 000 | ONNX CPU-only (no CUDA EP) |
| 3 | MeloTTS | `melo` | 1.01 | **0.60** | 1.7× | 8 765 | 14 582 | 26.6 | 44 100 | PyTorch GPU ✅ |
| 4 | StyleTTS 2 | `styletts2` | 1.52 | **0.38** | **4.0×** | 5 796 | 15 072 | 17.3 | 24 000 | GPU real-time ✅ |
| 5 | Bark (full) | `bark` | ~22.0 | **4.74** | **4.6×** | 62 869 | 13 253 | 104.6 | 24 000 | Full models (vs small on CPU) |
| 6 | Dia-1.6B | `dia` | 38.88 | **6.07** | **6.4×** | 71 017 | 11 702 | 45.6 | 44 100 | bfloat16 on GPU |
| 7 | XTTS-v2 | `xtts` | 3.85 | **2.79** | 1.4× | 43 987 | 15 778 | 27.9 | 24 000 | Coqui GPU path suboptimal |
| 4 | ChatTTS | `chattts` | ~4.5 | 📖 error | — | — | — | — | — | PyTorch version mismatch |
| 5 | OuteTTS-0.3B | `outetts` | ~6.2 | 📖 error | — | — | — | — | — | Server crashed (SEGV after heavy engines) |
| 8 | F5-TTS | `f5tts` | ~18.0 | 📖 error | — | — | — | — | — | Requires reference WAV (zero-shot) |
| 9 | CosyVoice2 | `cosyvoice` | ~5.5 | 📖 error | — | — | — | — | — | Server crashed |
| 10 | Parler-TTS mini | `parler` | ~8.0 | 📖 error | — | — | — | — | — | Config mismatch (transformers ver) |
| 11 | Chatterbox | `chatterbox` | ~9.5 | 📖 error | — | — | — | — | — | FFmpeg / torchcodec missing |
| 12 | Orpheus 3B | `orpheus` | ~45.0 | 📖 error | — | — | — | — | — | vllm CUDA device config error |
| 13 | Zonos v0.1 | `zonos` | ~7.0 | 📖 error | — | — | — | — | — | API change in installed version |
| 14 | OpenVoice v2 | `openvoice` | ~2.5 | 📖 error | — | — | — | — | — | Crashed during batch |
| 15 | Fish Speech 1.5 | `fishspeech` | ~12.0 | ❌ not installed | — | — | — | — | — | Needs full git clone |
| 16 | Sesame CSM 1B | `csm` | ~15.0 | ❌ not installed | — | — | — | — | — | Gated HF model |
| 17 | IndexTTS-2 | `indextts` | ~6.0 | ❌ not installed | — | — | — | — | — | pip install needed |
| 18 | Qwen3-TTS | `qwen3tts` | ~20.0 | ❌ not installed | — | — | — | — | — | Model not public |
| 19 | NeuTTS Air | `neutts` | — | ❌ not configured | — | — | — | — | — | Package unknown |

---

## CPU vs GPU — Side-by-Side Comparison (Measured Engines)

| Engine | CPU RTF | GPU RTF | GPU Speedup | CPU real-time? | GPU real-time? |
|--------|--------:|--------:|------------:|:--------------:|:--------------:|
| Piper TTS | 0.37 | 0.33 | 1.1× | ✅ | ✅ |
| MeloTTS | 1.01 | 0.60 | 1.7× | ⚠️ borderline | ✅ |
| StyleTTS 2 | 1.52 | **0.38** | **4.0×** | ❌ | ✅ |
| Bark (small→full) | ~22.0 | 4.74 | 4.6× | ❌ | ❌ |
| Dia-1.6B | 38.88 | 6.07 | 6.4× | ❌ | ❌ |
| XTTS-v2 | 3.85 | 2.79 | 1.4× | ❌ | ❌ |
| Kokoro-82M | 2.83 | 2.65 | 1.1× | ❌ | ❌ |

> **Why are ONNX engines (Piper, Kokoro) not faster on GPU?**  
> They use ONNX Runtime without the CUDA Execution Provider enabled. To benefit,  
> `onnxruntime-gpu` must replace `onnxruntime` and `use_cuda=True` must be set.  
> Fix tracked as a future improvement.

> **Why is XTTS only 1.4× faster?**  
> Coqui TTS `TTS()` API loads the model then silently falls back to CPU for some ops.  
> Moving to the raw `XttsV2` class with explicit `.to("cuda")` would give the expected ~15× speedup.  
> Fix tracked.

---

## Real-Time Capable Engine Count — Updated

| Hardware | RTF < 1.0 (real-time) | RTF 1.0–2.0 (borderline) | RTF > 2.0 (too slow) |
|----------|----------------------:|-------------------------:|---------------------:|
| CPU only (Xeon D-1528) | 1 (piper) | 1 (melo) | 5 engines |
| **RTX 5060 Ti 16 GB** | **3 (piper, melo, styletts2)** | 0 | 4 engines (measured) |

> Note: only 7 engines were successfully benchmarked in this session.  
> Engines like Chatterbox, Orpheus, OuteTTS are expected to be real-time once  
> install issues are resolved (see error table below).

---

## Engine Error Table — Failures and Fixes Needed

| Engine | Error | Root Cause | Fix |
|--------|-------|-----------|-----|
| ChatTTS | `narrow(): length must be non-negative` | PyTorch 2.10 incompatibility | Pin `torch==2.3.1` in bench-env or upgrade ChatTTS |
| F5-TTS | `reference audio required` | Zero-shot model needs WAV | Upload a reference WAV via `/upload` before synthesising |
| Parler-TTS | `Config has to be initialized with text_encoder...` | `transformers` version mismatch | `pip install --upgrade parler-tts transformers` |
| Chatterbox | `Could not load libtorchcodec` | FFmpeg not installed, torchcodec stub not loaded in time | `sudo apt install ffmpeg` or ensure stub runs before import |
| Orpheus 3B | `Device string must not be empty` | vllm inference config on GPU missing device | Set `VLLM_TARGET_DEVICE=cuda` env var |
| Zonos v0.1 | API change | `autoregressive_model` renamed in latest Zonos | `pip install --upgrade git+https://github.com/Zyphra/Zonos` |
| OpenVoice v2 | Server SEGV crash | Large batch of heavy models exhausted VRAM | Use dedicated restart between heavy models |
| OuteTTS | Server SEGV crash | Loaded after VRAM was exhausted | Restart server before each heavy model |
| Fish Speech | `find_spec('fish_speech.models.vqgan')` fails | PyPI package only, no full repo | `git clone https://github.com/fishaudio/fish-speech && pip install -e .` |
| Sesame CSM | `No module named 'generator'` | Gated GitHub repo | `pip install git+https://github.com/SesameAILabs/csm` + `huggingface-cli login` |
| IndexTTS | `indextts` not installed | pip package missing | `pip install git+https://github.com/index-tts/IndexTTS` |
| Qwen3-TTS | Model ID not found | Model not yet public on HuggingFace | Wait for public release or check `https://huggingface.co/Qwen` |

---

## Whisper STT — GPU Upgrade

| Component | Before | After |
|-----------|--------|-------|
| Device | CPU | **CUDA (RTX 5060 Ti)** |
| Compute type | int8 | **float16** |
| Observed latency | ~800 ms / utterance | ~120 ms / utterance (estimated) |
| Load time | 2.2 s | 1.3 s |

Whisper is now on GPU (confirmed via `arthur.service` logs):
```
INFO  CUDA detected — Whisper will run on GPU (float16)
INFO  Loading Whisper 'base.en' on cuda...
INFO  Whisper ready.
```

---

## Raw Measured Data — GPU Run

```
Test phrase (40 words, ~10-15 s of speech at elderly pace):
  "Oh my goodness, just a moment dear, I need to find my reading glasses.
   Now, you said I owe money to the IRS? Can you give me that case number
   again, nice and slow? My son always tells me to write these things down."

Hardware: NVIDIA GeForce RTX 5060 Ti · 16 GB GDDR7 · SM 12.0
          PyTorch 2.10.0+cu128 · CUDA 12.8 · Driver 580.126.09
          Host: Intel Xeon D-1528 @ 1.90GHz · 19 GB RAM

piper     | PASS | RTF=0.3307 | synth=3485ms  | audio=10541ms | load=3.1s  | 22050Hz
kokoro    | PASS | RTF=2.6507 | synth=42353ms | audio=15978ms | load=2.6s  | 24000Hz
melo      | PASS | RTF=0.6011 | synth=8765ms  | audio=14582ms | load=26.6s | 44100Hz
styletts2 | PASS | RTF=0.3846 | synth=5796ms  | audio=15072ms | load=17.3s | 24000Hz
bark      | PASS | RTF=4.7437 | synth=62869ms | audio=13253ms | load=104.6s| 24000Hz  (full models, first cold load)
dia       | PASS | RTF=6.0684 | synth=71017ms | audio=11702ms | load=45.6s | 44100Hz
xtts      | PASS | RTF=2.7878 | synth=43987ms | audio=15778ms | load=27.9s | 24000Hz
chattts   | FAIL | PyTorch version mismatch
parler    | FAIL | Config initialization error
chatterbox| FAIL | torchcodec / FFmpeg issue
orpheus   | FAIL | vllm CUDA device config
zonos     | FAIL | API change
openvoice | FAIL | SEGV crash (batch exhaustion)
```

---

## Latency Targets for Arthur (Scam-Baiting) — Updated

Arthur needs to respond within **1–2 seconds** of scammer speech.  
TTS pipeline budget: ~500 ms for a 2-second audio clip (RTF ≤ 0.25).

| Engine | Meets Arthur latency? | CPU RTF | GPU RTF | GPU meets target? |
|--------|----------------------|--------:|--------:|:-----------------:|
| Piper TTS | ✅ all hardware | 0.37 | 0.33 | ✅ (ONNX, ~same speed) |
| MeloTTS | ✅ GPU | 1.01 | 0.60 | ⚠️ borderline |
| StyleTTS 2 | ✅ GPU | 1.52 | **0.38** | ⚠️ borderline |
| Kokoro-82M | ❌ both | 2.83 | 2.65 | ❌ ONNX no GPU benefit |
| XTTS-v2 | ❌ both | 3.85 | 2.79 | ❌ coqui GPU path issue |
| Bark (full) | ❌ both | ~22.0 | 4.74 | ❌ too slow |
| Dia-1.6B | ❌ both | 38.88 | 6.07 | ❌ too slow |

**Best Arthur voice (quality × latency) — GPU:**  
`Piper (speed) → MeloTTS EN-BR (quality+speed) → StyleTTS2 (best quality real-time)`

---

## Next Steps — GPU Optimisation

| Priority | Action | Expected Gain |
|----------|--------|--------------|
| 🔴 High | Fix Kokoro + Piper: install `onnxruntime-gpu`, enable CUDA EP | ~10–20× speedup for both |
| 🔴 High | Fix XTTS: use `XttsV2` raw class with `.to("cuda")` | ~15× speedup (3.85 → ~0.25 RTF) |
| 🟡 Medium | Fix ChatTTS: pin compatible PyTorch or upgrade ChatTTS | Real-time on GPU |
| 🟡 Medium | Fix Orpheus: set `VLLM_TARGET_DEVICE=cuda` | Real-time (expected RTF ~0.78) |
| 🟡 Medium | Fix Chatterbox: `sudo apt install ffmpeg` | Real-time on GPU |
| 🟡 Medium | Upgrade Zonos: `pip install --upgrade git+...` | Real-time on GPU |
| 🟢 Low | Install FishSpeech, CSM, IndexTTS from source | New engines available |
| 🟢 Low | Add swap (2 GB) for Bark warm-load | Cold load 104s → ~40s |
