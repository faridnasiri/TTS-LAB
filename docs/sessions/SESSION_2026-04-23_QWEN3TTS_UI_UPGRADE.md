# TTS Lab — Session 2026-04-23: Qwen3-TTS Fine-Tuning + Full UI Redesign
> Branch: `main`
> Commits: `369cf4f` `5f1a70a` `67a146a` `791876a`

---

## Starting State
```
  ▶ qwen3tts   PASS  dur=3920ms  rtf=4.46×  sr=24000Hz  load=6.79s  synth=17482ms
```
- UI: horizontal tab strip (21 tabs, no filtering, hard to navigate)
- Qwen3-TTS panel: only speaker + language + optional ref WAV — no sampling controls
- flash-attn investigation was abandoned last session (SM 12.0 not supported by FA2)

---

## Part 1 — flash-attn Final Verdict

### Why flash-attn 2.x Cannot Work on RTX 5060 Ti
| Fact | Detail |
|---|---|
| GPU compute capability | SM **12.0** (Blackwell GB206) |
| flash-attn 2.x max SM | SM **9.0** (Hopper) — Blackwell not supported |
| Pre-built wheels available | Only up to `torch2.4+cu12` — **no torch2.11 wheel** |
| Source compile | Would fail at NVCC with unsupported arch even with 32 GB RAM |
| Flash Attention 4 (beta) | Published as `flash_attn_4==4.0.0b10` on PyPI — pure Python wheel |
| FA4 API | Uses `flash_attn.cute` namespace — **incompatible** with qwen_tts's `flash_attn.flash_attn_interface` import |

### Solution: `attn_implementation="sdpa"`
PyTorch's built-in `scaled_dot_product_attention` (SDPA) uses cuDNN's fused attention kernel on SM 12.0. Same mathematical output as flash-attn, natively supported.

**Change in `_load_qwen3tts()`:**
```python
_attn = "sdpa" if DEVICE == "cuda" else "eager"
mdl = Qwen3TTSModel.from_pretrained(model_id, device_map=DEVICE, dtype=_dtype,
                                    attn_implementation=_attn)
```

**Result:**
| Metric | Before (no flag) | After (`sdpa`) |
|---|---|---|
| Load time | 11.25s | **6.79s (−40%)** |
| Synth time | 21673ms | **17482ms (−19%)** |
| RTF | 4.75× | **4.46×** |

Commit: `369cf4f`

---

## Part 2 — Model Comparison: 1.7B vs 0.6B

All available Qwen3-TTS models as of 2026-04-23:

| Model | Size | Built-in speakers | Downloads |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~3 GB | ✅ 9 voices | 1.59M ← **current** |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~3 GB | ❌ clone only | 1.40M |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~3 GB | ❌ text-instruct style | 524K |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~1 GB | ✅ fewer voices | 253K |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~1 GB | ❌ clone only | 692K |

### Benchmark: 0.6B-CustomVoice vs 1.7B-CustomVoice

| Model | RTF | Load | Synth ms | Audio dur |
|---|---|---|---|---|
| 1.7B-CustomVoice | **4.46×** | 6.79s | 17482ms | 3920ms |
| 0.6B-CustomVoice | 4.43× | 6.85s | 23748ms | 5360ms |

**Finding:** RTF is virtually identical. The 12Hz autoregressive decoder is the bottleneck — it generates 12 codec tokens/second of audio sequentially. Model parameter count affects per-token compute time, but on the RTX 5060 Ti the bottleneck is KV-cache memory bandwidth, not matrix multiply throughput. Halving parameters does not halve RTF.

**Decision:** Keep `1.7B-CustomVoice` — same speed, better quality.

Commit: `5f1a70a`

---

## Part 3 — Full UI Redesign

### Problem with Old UI
- 21 horizontal tabs — needed to scroll left/right, active tab not always visible
- No way to filter/search engines
- Status grid (engine dots) was a separate area above the tabs — disconnected from engine panel
- No full-height use of screen — content area scrolled independently

### New Layout: Sidebar + Detail Panel

```
┌──────────────────────────────────────────────────────────────────┐
│ 🎙 Arthur TTS Lab  21 engines  🟢 RTX 5060 Ti  RAM ████  VRAM ██ │  ← sticky header
├──────────────────┬───────────────────────────────────────────────┤
│ TTS Engines      │  Qwen3-TTS                    ~2-4×  ✓ avail  │
│ 🔍 Filter…      │  💾 ~3 GB   🧠 ~2000 MB   🎭 ★★★★☆           │
│ ────────────────│                                                 │
│ 🟢 Piper   0.4× │  [Style instruction text field]                │
│ 🟢 Kokoro  2.8× │                                                 │
│ 🟢 MeloTTS 0.3× │  ── Main talker ──                             │
│ 🟢 ChatTTS 2.6× │  Temp ──●──  Top-p ──●──  Top-k ──●──  Rep ●  │
│ 🟢 Bark    4.6× │                                                 │
│ 🟢 XTTS    0.9× │  ── Sub-talker ──                              │
│ 🟢 Qwen3   4.4× │  Sub-temp ●──  Sub-top-p ●──  Sub-top-k ●──   │
│ ⚫ Orpheus       │                                                 │
│ ⚫ Parler        │  ── Generation ──                              │
│ ...              │  Max tokens ────────────●──────────            │
│                  │                                                 │
│                  │  [▶ Synthesise]  [⬇ Preload]  [⏏ Unload]  ⟳  │
│                  │                                                 │
│                  │  ⏱ 17482ms  🔊 3920ms  RTF 4.46×  ⬇ 6.7s     │
│                  │  ▶────────────────────────────────────────     │
└──────────────────┴───────────────────────────────────────────────┘
            Arthur's text + preset buttons  (always visible)
```

### Key UX Improvements
| Feature | Old | New |
|---|---|---|
| Engine navigation | 21 horizontal tabs | Vertical sidebar, always visible |
| Engine status | Separate grid above tabs | Live 🟢🟡🔴⚫ dot inline in sidebar |
| Engine filtering | None | 🔍 text filter, instant |
| Layout | Single scroll | Full-height split — sidebar + content each scroll independently |
| Header | Inline with content | Sticky top bar — always visible while scrolling |
| Mobile | Horizontal scroll tabs | Sidebar collapses to horizontal engine row |

### CSS Architecture
- CSS custom properties (`--bg`, `--panel`, `--card`, `--border`, `--accent`) for consistent theming
- No Bootstrap tab machinery — plain `display:block/none` JS switching, simpler and faster
- Custom scrollbar styling
- Smooth animations on bar fills and hover states

Commit: `67a146a`

---

## Part 4 — Qwen3-TTS: All Parameters Exposed

### Full Parameter Discovery
Sourced from `/opt/arthur-bench-env/lib/python3.11/site-packages/qwen_tts/inference/qwen3_tts_model.py` — `_merge_generate_kwargs()`.

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `temperature` | 0.9 | 0.1–2.0 | Main talker randomness. Lower = stable/robotic. Higher = expressive/variable. |
| `top_p` | 1.0 | 0.1–1.0 | Nucleus sampling. Lower = more conservative vocab choices. |
| `top_k` | 50 | 1–200 | Vocab cutoff per token step. |
| `repetition_penalty` | 1.05 | 1.0–1.5 | >1.0 penalises repeated codec tokens. Reduces stuttering/looping. |
| `subtalker_temperature` | 0.9 | 0.1–2.0 | Sub-talker (tokenizer-v2, 1.7B model) — fine acoustic detail head. |
| `subtalker_top_p` | 1.0 | 0.1–1.0 | Sub-talker nucleus sampling. |
| `subtalker_top_k` | 50 | 1–200 | Sub-talker vocab cutoff. |
| `max_new_tokens` | 2048 | 256–4096 | Max codec tokens. At 12 Hz ≈ ~170 tokens/sec audio. 2048 ≈ ~12s. |
| `instruct` | `""` | free text | Natural-language style instruction (CustomVoice 1.7B only, ignored on 0.6B). |
| `speaker` | `"aiden"` | 9 choices | Built-in voice identity. |
| `language` | `"english"` | 8 choices | Output language. |

### Instruct Examples (for Arthur Henderson character)
```
"speak slowly and gently like a confused elderly man"
"speak with a warm, slightly shaky elderly voice, slightly breathless"
"speak at a slow pace with natural pauses, as if searching for words"
"use a soft, polite tone with occasional hesitation"
```

### _synth_qwen3tts Logic
```python
# Only non-default values are forwarded (avoids overriding model's generate_config.json)
gen_kwargs = {}
for k, d in [("temperature", 0.9), ("top_p", 1.0), ("repetition_penalty", 1.05), ...]:
    v = _float(k, d)
    if v is not None: gen_kwargs[k] = v

# Voice clone takes priority if ref_wav + ref_txt both provided
if ref_wav and ref_txt:
    wavs, sr = inst.generate_voice_clone(..., **gen_kwargs)
else:
    instruct = params.get("instruct", "").strip() or None
    wavs, sr = inst.generate_custom_voice(..., instruct=instruct, **gen_kwargs)
```

Commit: `791876a`

---

## Final State

```
  ▶ qwen3tts   PASS  dur=4400ms  rtf=4.41×  sr=24000Hz  load=6.7s  synth=19417ms
```

### What's Live at `http://192.168.0.87:8001`
- New sidebar UI — all 21 engines, live status dots, searchable
- Qwen3-TTS panel: speaker, language, instruct prompt, 9 sampling sliders, max-tokens, voice clone upload
- All other engines unchanged and working

### Why RTF Won't Improve Further Without Major Changes
The 12Hz autoregressive token generation is sequential by nature. Each token depends on all previous tokens (causal attention). Options to improve:
1. **INT4/INT8 quantization** — `qwen-tts` does not yet support GPTQ/AWQ/bitsandbytes
2. **Speculative decoding** — not implemented in `qwen-tts`
3. **Smaller model** — 0.6B tested, same RTF (bandwidth-bound not compute-bound)
4. **Wait** — Alibaba may ship an optimized inference backend

---

## Files Changed
| File | Change |
|---|---|
| `tools/arthur_server/tts_lab.py` | `attn_implementation=sdpa`, full UI rewrite, qwen3tts full params |
| `tools/arthur_server/SESSION_2026-04-23_QWEN3TTS_UI_UPGRADE.md` | This file |
