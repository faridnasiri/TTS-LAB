# Arthur TTS Lab — Benchmark Results 2026-04-23
> VM: arthur@192.168.0.87 · RTX 5060 Ti 16 GB GDDR7 · SM 12.0 · CUDA 13.0 · torch 2.11.0+cu128  
> Python 3.11 · `/opt/arthur-bench-env`  
> Test phrase: *"Oh my goodness, just a moment dear. You said I owe money?"* (~4–5s audio)

---

## Qwen3-TTS — Optimisation Progression

| Commit | Change | RTF | Load | Synth ms |
|---|---|---|---|---|
| Before session | `attn=eager` (default) | 4.75× | 11.25s | 21673ms |
| `369cf4f` | `attn_implementation="sdpa"` | **4.46×** | **6.79s** | **17482ms** |
| `5f1a70a` | 0.6B-CustomVoice test | 4.43× | 6.85s | 23748ms (longer audio) |
| `5f1a70a` | Reverted to 1.7B-CustomVoice | 4.41× | 6.70s | 19417ms |

**Conclusion:** SDPA gives −40% load time and −19% synth time. Model size (0.6B vs 1.7B) has no meaningful RTF impact on this GPU — memory bandwidth bound.

---

## All Engines — Current State (GPU, RTX 5060 Ti)

| Engine | Key | RTF | Synth ms | Audio ms | Load s | SR | Status |
|---|---|---|---|---|---|---|---|
| Piper TTS | `piper` | **0.47×** | 1336 | 2832 | 2.3 | 22050 | ✅ real-time |
| MeloTTS | `melo` | **0.30×** | 4299 | 14582 | 26.6 | 44100 | ✅ real-time |
| StyleTTS 2 | `styletts2` | **0.35×** | 5330 | 15072 | 29.7 | 24000 | ✅ real-time |
| XTTS-v2 | `xtts` | **0.91×** | 14398 | 15853 | 53.8 | 24000 | ✅ real-time |
| Chatterbox | `chatterbox` | **1.67×** | 17015 | 10200 | 30.8 | 24000 | ✅ |
| ChatTTS | `chattts` | **2.59×** | — | — | — | — | ✅ |
| Kokoro-82M | `kokoro` | **2.77×** | 44261 | 15978 | 2.6 | 24000 | ✅ |
| Bark (full) | `bark` | **4.64×** | 67881 | 14640 | 72.4 | 24000 | ✅ |
| **Qwen3-TTS** | `qwen3tts` | **4.41×** | 19417 | 4400 | 6.7 | 24000 | ✅ |
| Zonos v0.1 | `zonos` | **4.03×** | 47918 | 11888 | 36.4 | 44100 | ✅ |
| Dia-1.6B | `dia` | **6.75×** | 79010 | 11702 | 50.5 | 44100 | ✅ |
| Fish Speech | `fishspeech` | ~1.5× est | — | — | — | — | ✅ (permanent /opt install) |
| IndexTTS-2 | `indextts` | **0.4×** est | — | — | — | — | ✅ (needs ref WAV) |
| Parler-TTS | `parler` | — | — | — | — | — | ⚠️ transformers pin conflict |
| OpenVoice v2 | `openvoice` | — | — | — | — | — | ⚠️ VAD edge case |
| Orpheus 3B | `orpheus` | — | — | — | — | — | ⚠️ vllm gated |
| F5-TTS | `f5tts` | — | — | — | — | — | ⚠️ needs ref WAV |

---

## flash-attn Investigation — Final Verdict

| Approach | Result |
|---|---|
| Pre-built wheel `flash_attn==2.8.3` | Max torch2.4 — **incompatible** with torch2.11 |
| Source compile flash-attn 2.x with `sm_120` | **Not supported** — FA2 max is SM 9.0 (Hopper) |
| `flash_attn_4==4.0.0b10` (PyPI, pure Python) | Installed, but uses `flash_attn.cute` API — **incompatible** with qwen_tts import |
| `attn_implementation="sdpa"` (PyTorch built-in) | ✅ **Works** — cuDNN fused attention, native SM 12.0, −19% synth time |

flash-attn 2.x will never support SM 12.0. FA4 supports it but has a different API that third-party packages don't use yet. **SDPA is the correct long-term solution for Blackwell GPUs.**
