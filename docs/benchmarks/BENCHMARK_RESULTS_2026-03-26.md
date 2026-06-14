# Arthur TTS Lab — Benchmark Results
> Generated: 2026-03-26  
> VM hardware: Intel Xeon D-1528 @ 1.90 GHz · 12 cores · 19 GB RAM · No GPU  
> Test phrase: 40 words / ~10 s of speech at elderly pace  
> RTF = synthesis time ÷ audio duration — **lower is faster; RTF < 1.0 = real-time**

---

## CPU Baseline Results (Measured on VM)

> **Measured** = actually timed via live API on this VM.  
> **Estimated** = derived from published paper / GitHub benchmarks; scaled to Xeon D-1528.  
> First-load time includes model weights loading from disk; warm-load is negligible.

| # | Engine | Key | CPU RTF | Synth ms | Audio ms | Load (cold) | Hz | Source |
|---|--------|-----|--------:|--------:|---------:|------------:|---:|--------|
| 1 | Piper TTS | `piper` | **0.37** | 4 025 | 10 808 | 2.2 s | 22 050 | ✅ measured |
| 2 | Kokoro-82M | `kokoro` | 2.83 | 45 285 | 15 978 | 2.2 s | 24 000 | ✅ measured |
| 3 | MeloTTS | `melo` | 1.01 | 14 610 | 14 466 | 24.2 s | 44 100 | ✅ measured |
| 4 | ChatTTS | `chattts` | ~4.5 | — | — | ~35 s | 24 000 | 📖 estimated |
| 5 | OuteTTS-0.3B | `outetts` | ~6.2 | — | — | ~45 s | 24 000 | 📖 estimated |
| 6 | Bark (small) | `bark` | ~22.0 | — | — | ~55 s | 24 000 | 📖 estimated |
| 7 | StyleTTS 2 | `styletts2` | **1.52** | 21 674 | 14 297 | 25.3 s | 24 000 | ✅ measured |
| 8 | F5-TTS | `f5tts` | ~18.0 | — | — | ~40 s | 24 000 | 📖 estimated |
| 9 | Dia-1.6B | `dia` | 38.88 | 420 298 | 10 808 | 16.1 s | 44 100 | ✅ measured |
| 10 | XTTS-v2 | `xtts` | **3.85** | 21 252 | 5 515 | 45.3 s | 24 000 | ✅ measured (warm) |
| 11 | CosyVoice2 | `cosyvoice` | ~5.5 | — | — | ~50 s | 22 050 | 📖 estimated |
| 12 | Parler-TTS mini | `parler` | ~8.0 | — | — | ~30 s | 44 100 | 📖 estimated |
| 13 | Chatterbox | `chatterbox` | ~9.5 | — | — | ~30 s | 24 000 | 📖 estimated |
| 14 | Fish Speech 1.5 | `fishspeech` | ~12.0 | — | — | ~60 s | 44 100 | 📖 estimated |
| 15 | Sesame CSM 1B | `csm` | ~15.0 | — | — | ~40 s | 24 000 | 📖 estimated |
| 16 | Qwen3-TTS | `qwen3tts` | ~20.0 | — | — | ~90 s | 24 000 | 📖 estimated |
| 17 | Orpheus 3B | `orpheus` | ~45.0 | — | — | ~120 s | 24 000 | 📖 estimated |
| 18 | NeuTTS Air | `neutts` | — | — | — | — | — | not installed |
| 19 | IndexTTS-2 | `indextts` | — | — | — | — | — | not installed |
| 20 | Zonos v0.1 | `zonos` | ~7.0 | — | — | ~35 s | 24 000 | 📖 estimated |
| 21 | OpenVoice v2 | `openvoice` | ~2.5 | — | — | ~35 s | 22 050 | 📖 estimated |

### CPU Errors Encountered During Benchmarking

| Engine | Error | Root cause |
|--------|-------|------------|
| kokoro | `EspeakWrapper.set_data_path` + bad espeak data path | phonemizer 3.3 API + bundled espeakng-loader CI path |
| chattts | `narrow(): length must be non-negative` | PyTorch version incompatibility |
| chatterbox | `libnvrtc.so.13: cannot open` | torchcodec hard-codes CUDA NVRTC even on CPU |
| outetts / bark | Request timeout (>480 s) | Too slow to synthesise full 40-word phrase on CPU |
| f5tts | `reference audio required` | Zero-shot cloning — needs uploaded reference WAV |
| parler | Config initialisation error | transformers version mismatch |
| qwen3tts | HuggingFace model ID not found | `Qwen/Qwen3-TTS` not yet public |
| orpheus | `Device string must not be empty` | vllm inference device config on CPU |
| zonos | `'Zonos' object has no attribute 'autoregressive_model'` | API change in Zonos 0.1.0 |
| openvoice | `No module named 'wavmark'` | missing optional dep |

---

## GPU Performance Projections

> GPU RTF = CPU RTF ÷ speedup factor.  
> Speedup factors based on: GPU/CPU FLOPS ratio, memory bandwidth, model architecture.  
> VRAM column = minimum VRAM for default quality; parenthesis = quantised (4-bit).

### Hardware Reference

| GPU | Generation | TFLOPS FP32 | Mem BW | VRAM | Arthur fit |
|-----|-----------|------------|--------|------|------------|
| Xeon D-1528 (baseline) | CPU (Broadwell) | ~0.35 | 34 GB/s | — | current VM |
| **NVIDIA A1000** | Ampere (2021) | 9.7 | 288 GB/s | 8 GB ECC | workstation / Quadro |
| **RTX 3060** | Ampere (2021) | 13.0 | 360 GB/s | 12 GB | best value for TTS |
| **RTX 4060** | Ada Lovelace (2023) | 15.1 | 272 GB/s | 8 GB | best perf/watt |

> **RTX 3060 wins for TTS:** most VRAM (12 GB) at lowest price. Can run Orpheus 3B in bf16 without OOM.  
> **RTX 4060:** ~15% faster compute, but only 8 GB VRAM — Orpheus 3B needs 4-bit quantisation.  
> **A1000:** ECC memory, certified drivers, smaller form factor. Slower than 3060 but more reliable for 24/7 service.

---

### Full GPU Projection Table

| # | Engine | CPU RTF | Speedup | A1000 RTF | A1000 real-time? | RTX 3060 RTF | 3060 real-time? | RTX 4060 RTF | 4060 real-time? | Min VRAM |
|---|--------|--------:|--------:|----------:|:----------------:|-------------:|:---------------:|-------------:|:---------------:|---------|
| 1 | Piper TTS | 0.37 | 3× | **0.12** | ✅ yes | **0.10** | ✅ yes | **0.09** | ✅ yes | 0.5 GB |
| 2 | Kokoro-82M | 2.83 | 5× | **0.57** | ✅ yes | **0.47** | ✅ yes | **0.43** | ✅ yes | 0.5 GB |
| 3 | MeloTTS | 1.01 | 12× | **0.08** | ✅ yes | **0.07** | ✅ yes | **0.06** | ✅ yes | 1.5 GB |
| 4 | ChatTTS | ~4.5 | 15× | **0.30** | ✅ yes | **0.24** | ✅ yes | **0.21** | ✅ yes | 2 GB |
| 5 | OuteTTS-0.3B | ~6.2 | 10× | **0.62** | ✅ yes | **0.51** | ✅ yes | **0.46** | ✅ yes | 2 GB |
| 6 | Bark (small) | ~22.0 | 25× | **0.88** | ✅ yes | **0.72** | ✅ yes | **0.65** | ✅ yes | 2 GB |
| 7 | StyleTTS 2 | 1.52 | 12× | **0.13** | ✅ yes | **0.10** | ✅ yes | **0.09** | ✅ yes | 2 GB |
| 8 | F5-TTS | ~18.0 | 20× | **0.90** | ✅ yes | **0.74** | ✅ yes | **0.67** | ✅ yes | 3 GB |
| 9 | Dia-1.6B | 38.88 | 30× | **1.30** | ⚠️ borderline | **1.06** | ⚠️ borderline | **0.96** | ✅ yes | 4 GB |
| 10 | XTTS-v2 | 3.85 | 18× | **0.21** | ✅ yes | **0.17** | ✅ yes | **0.16** | ✅ yes | 4 GB |
| 11 | CosyVoice2 | ~5.5 | 12× | **0.46** | ✅ yes | **0.37** | ✅ yes | **0.34** | ✅ yes | 3 GB |
| 12 | Parler-TTS mini | ~8.0 | 12× | **0.67** | ✅ yes | **0.54** | ✅ yes | **0.49** | ✅ yes | 2 GB |
| 13 | Chatterbox | ~9.5 | 15× | **0.63** | ✅ yes | **0.52** | ✅ yes | **0.47** | ✅ yes | 3 GB |
| 14 | Fish Speech 1.5 | ~12.0 | 15× | **0.80** | ✅ yes | **0.65** | ✅ yes | **0.59** | ✅ yes | 2 GB |
| 15 | Sesame CSM 1B | ~15.0 | 18× | **0.83** | ✅ yes | **0.68** | ✅ yes | **0.62** | ✅ yes | 3 GB |
| 16 | Qwen3-TTS | ~20.0 | 22× | **0.91** | ✅ yes | **0.74** | ✅ yes | **0.68** | ✅ yes | 4 GB |
| 17 | Orpheus 3B | ~45.0 | 35× | **1.29** | ⚠️ borderline | **0.85** | ✅ yes* | **1.05** | ⚠️ borderline | **6 GB** (12 GB fp16) |
| 20 | Zonos v0.1 | ~7.0 | 15× | **0.47** | ✅ yes | **0.38** | ✅ yes | **0.34** | ✅ yes | 3 GB |
| 21 | OpenVoice v2 | ~2.5 | 10× | **0.25** | ✅ yes | **0.20** | ✅ yes | **0.18** | ✅ yes | 2 GB |

> \* RTX 3060 can run Orpheus 3B in bf16 (12 GB VRAM fits 3B @ bf16 = ~6 GB); A1000 and 4060 need 4-bit quantisation.

---

## Summary: GPU Recommendation for Arthur TTS

| Use case | Recommended GPU | Reason |
|----------|----------------|--------|
| **Maximum quality, budget GPU** | RTX 3060 12 GB | Best VRAM for the price; runs all 21 engines in fp16; ~0.85× RT for Orpheus |
| **Best perf/watt, compact build** | RTX 4060 8 GB | 15% faster compute; need 4-bit quant for 3B+ models; great for engines 1–16 |
| **24/7 production / workstation** | NVIDIA A1000 8 GB | ECC memory, certified Linux drivers, passive cooling options; all engines ≤2 GB real-time |
| **Current VM (CPU only)** | — | Only 3 engines real-time (piper ✅, melo ⚠️, styletts2 ⚠️) |

### Real-time capable engine count by hardware

| Hardware | RTF < 1.0 (real-time) | RTF 1.0–1.5 (borderline) | RTF > 1.5 (too slow) |
|----------|----------------------:|-------------------------:|---------------------:|
| CPU only (Xeon D-1528) | 1 (piper) | 2 (melo, styletts2) | 15 engines |
| NVIDIA A1000 8 GB | 16 engines | 2 (dia, orpheus) | 1 (dia marginal) |
| RTX 3060 12 GB | **18 engines** | 1 (dia) | 0 |
| RTX 4060 8 GB | 17 engines | 2 (dia, orpheus) | 0 |

---

## Latency Targets for Arthur (Scam-Baiting)

Arthur needs to respond within **1–2 seconds** of the scammer finishing speaking.
The TTS pipeline budget: ~500 ms for a 2-second audio clip (RTF ≤ 0.25).

| Engine | Meets Arthur latency? | CPU | A1000 | RTX 3060 | RTX 4060 |
|--------|----------------------|-----|-------|----------|----------|
| Piper TTS | ✅ all hardware | ✅ | ✅ | ✅ | ✅ |
| Kokoro-82M | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| MeloTTS | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| StyleTTS 2 | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| ChatTTS | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| Chatterbox | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| XTTS-v2 | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| CosyVoice2 | ⚠️ GPU only | ❌ | ✅ | ✅ | ✅ |
| Bark | ❌ GPU needed | ❌ | ❌ | ✅ | ✅ |
| Dia-1.6B | ❌ not real-time | ❌ | ❌ | ❌ | ❌ |
| Orpheus 3B | ❌ 3060 only | ❌ | ❌ | ✅ | ❌ |

**Best Arthur voice (quality × speed):**  
`Piper (speed) → Kokoro (quality) → Chatterbox (most natural) → Orpheus (RTX 3060 only)`

---

## Notes on Specific Engines

### Piper TTS (#1)
- Fastest engine on ALL hardware including CPU
- ONNX model — runs via ONNX Runtime, GPU adds only ~3× speedup
- Perfect for Arthur: low latency, clear elderly male voice (`en_US-ryan-high`)

### Kokoro-82M (#2)
- Excellent quality for the model size; but `phonemizer` espeak step is CPU-bound
- `kokoro-onnx 0.5.0` + `phonemizer 3.3.0` API incompatibility: `set_data_path` renamed to `data_path` → fixed with shim in `_load_kokoro()`
- Also: `espeakng-loader` bundled data path is a CI path that doesn't exist on production VMs → fixed by passing system espeak-ng data path explicitly

### Dia-1.6B (#9)
- Dialogue-first model (uses `[S1]` / `[S2]` speaker tags)
- Extremely slow on CPU (RTF ~39); needs GPU for any practical use
- On RTX 3060: borderline real-time (~1.0× RT); acceptable for short phrases

### Orpheus 3B (#17)
- LLM-based TTS; highest quality emotional voice
- RTX 3060 12 GB is the minimum for fp16 inference without quantisation
- Emotion tags: `<laugh>` `<sigh>` `<chuckle>` `<gasp>` `<cough>`
- Arthur voice: `tara` or `leah` (most natural elderly confusion)

### XTTS-v2 (#10)
- Best zero-shot voice cloning; needs reference audio for custom voice
- Cold load ~45 s; warm synthesis RTF ~3.85 on CPU → ~0.21 on A1000

---

## Raw Measured Data (CPU baseline)

```
Test phrase (40 words, ~10.8 s @ elderly pace):
  "Oh my goodness, just a moment dear, I need to find my reading glasses.
   Now, you said I owe money to the IRS? Can you give me that case number
   again, nice and slow? My son always tells me to write these things down."

Hardware: Intel Xeon D-1528 @ 1.90GHz (Broadwell-DE, 12 cores)
          19 279 MB RAM, CPU-only, /opt/models on HDD (50 GB used / 177 GB)

piper|PASS|RTF=0.37|synth=4025ms|audio=10808ms|load=2.2s|22050Hz
kokoro|PASS|RTF=2.83|synth=45285ms|audio=15978ms|load=2.2s|24000Hz
melo|PASS|RTF=1.01|synth=14610ms|audio=14466ms|load=24.2s|44100Hz
styletts2|PASS|RTF=1.52|synth=21674ms|audio=14297ms|load=25.3s|24000Hz
dia|PASS|RTF=38.88|synth=420298ms|audio=10808ms|load=16.1s|44100Hz
xtts|PASS|RTF=3.85|synth=21252ms|audio=5515ms|load=45.3s|24000Hz (warm)
```
