# Handoff — S1-Mini engine + quantization feasibility (2026-08-23)

> **Read first:** [CLAUDE.md](../../CLAUDE.md) and the auto-memory index at `C:\Users\farid\.claude\projects\c--repos-TTS-LAB\memory\MEMORY.md`. Continuity record for the "reduce VRAM for S2-Pro / EditX" request.

## TL;DR — verdicts on the four ideas

| Idea from request | Verdict | Why |
|---|---|---|
| **OpenAudio S1-Mini (0.5B)** — light Fish model | ✅ **IMPLEMENTED** (new `s1mini` engine, in-process, engine-current) | Real model, fish-speech architecture, ~5 GB VRAM vs S2-Pro ~11 GB. |
| **Fish Speech 1.5 (~500M)** | ✅ Already in the lab | The `fishspeech` engine (fishaudio/fish-speech-1.5, ~1.1 GB). |
| **S2-Pro INT4/AWQ/GGUF** (14 GB → 5-6 GB) | ❌ **Not feasible** | sgl-omni 0.1.3's s2-pro pipeline has no quantization support — no `--quantization` flag, no quantized request param, no official quantized checkpoints; GGUF is only community llama.cpp ports sgl-omni cannot serve. |
| **EditX INT4/GPTQ/AWQ** (8 GB → 3-4 GB) | ✅ Already done | The lab runs Step-Audio-EditX **AWQ-4bit** already (that IS the quantization; ~12.8 GiB resident incl. vLLM + CosyVoice). |

**"Option in UI to choose, also as API parameter":** the light/heavy choice is engine selection — `s1mini` appears in the sidebar next to `fishspeech`, and remote calls use `POST /synthesize/s1mini` with the same params as the UI sends. There is no variant param on s2pro/editx — no second variant exists for either.

## What was delivered — `s1mini` engine

- **Model:** `fishaudio/s1-mini` (0.5B distilled DualAR + DAC codec, 13 languages) — ⚠ **gated** (accept license on HF, `HF_TOKEN` required) and **CC-BY-NC-SA-4.0 non-commercial** (fine for this personal lab).
- **Codebase:** fish-speech **main** at `/opt/models/fish-speech-s1` — the v1.5.1 checkout used by `fishspeech` has **no** `modded_dac_vq` codec config or `TTSInferenceEngine`. Clone: `sudo git clone https://github.com/fishaudio/fish-speech /opt/models/fish-speech-s1` (host, mounted via `/opt/models`).
- **Inference:** `launch_thread_safe_queue` + `load_decoder_model(config_name="modded_dac_vq", checkpoint_path=codec.pth)` + `TTSInferenceEngine`; synth via `ServeTTSRequest(text, references=[ServeReferenceAudio(audio=ref_wav_bytes, text=transcript)])` — cloning works from the same ref WAV + transcript flow as editx/s2pro (sidecar transcript fallback included).
- **Files touched:**
  - `tts_lab_config.py` — `S1MINI_REPO_DIR`/`S1MINI_MODEL_ID`, `MODEL_INFO["s1mini"]` (heavy, ~5 GB), `MODEL_ORDER` (after fishspeech), `SYNTH_TIMEOUT["s1mini"]=600`.
  - `tts_lab_engines.py` — `_purge_fish_speech_modules()` + `_load_s1mini()`/`_synth_s1mini()` + LOADERS/SYNTHERS.
  - `tts_lab_dispatch.py` — availability probe (checkout dir + gated-repo token check, qwen3tts style).
  - `tts_lab_ui.py` — params panel (tokens/chunk/top-p/temp/rep-penalty + clone refs) + license/quality warnings.
  - `docker-compose.yml` — `S1MINI_URL: http://engine-current:8101` (needed for orchestrator routing).
- **HEAVY:** yes — the orchestrator stops the s2pro container (11 GB) before s1mini loads, and s2pro's own path evicts all engine containers first. Both directions are safe on 16 GB.

## ⚠️ Two traps to know

1. **Two fish-speech checkouts, one `fish_speech` package.** `_purge_fish_speech_modules()` (sys.modules purge) runs at the top of BOTH `_load_fishspeech` and `_load_s1mini`. Without it the second engine silently imports the first checkout's API (no `modded_dac_vq`, no `TTSInferenceEngine`). Never remove the purge. Safe because the engine server evicts before every load and synth re-imports lazily.
2. **transformers pin skew (unvalidated).** fish-speech main pins `transformers<=4.57.3`; engine-current runs 5.12.1. The inference modules have no transformers imports except `AutoTokenizer` (stable API), so it *should* work — but it is **unvalidated on the VM GPU**. Status is `experimental` in `docs/engine_compatibility.yaml`. If load fails under tf 5.x, the fallback is a dedicated container (editx-pattern: python 3.12 + torch 2.13.0+cu130 + tf 4.57.3).

## VM-side steps (not yet done)

```bash
# 1. Clone fish-speech main (host, under the /opt/models mount)
sudo git clone https://github.com/fishaudio/fish-speech /opt/models/fish-speech-s1
# 2. Accept the s1-mini license + ensure HF_TOKEN is set in engine-current (compose already passes ${HF_TOKEN:-})
# 3. Deploy code (Phase 5) and restart the orchestrator + engine-current
.\scripts\deploy\deploy_lab.ps1 -Phase 5
docker restart tts-lab-orchestrator tts-lab-engine-current
# 4. Verify: status shows s1mini available; POST /synthesize/s1mini with a ref WAV
curl -s http://192.168.0.87:8009/status | python -m json.tool | grep -A3 s1mini
```

First load downloads ~3.6 GB (gated) + loads; allowance 600 s.

---

## ✅ Deployed + verified 2026-08-23 (evening)

License accepted, new HF_TOKEN persisted (`/opt/arthur-tts-lab/.env`, chmod 600 — compose reads it; `/etc/environment` updated for bare-metal SSH tests). Engine live on engine-current, `/status` available, first-load download + synthesis **SUCCESS**.

### Two checkout patches (both in `patches/patch_fish_speech_s1.py`, idempotent, `sudo python3` on the VM)

1. **`fish_speech/tokenizer.py` — FishTokenizer tiktoken fallback.** transformers **5.15.0** (the engine-current image has newer than the doc's 5.12.1) removed the tiktoken-reading slow tokenizer classes 4.x used. s1-mini ships only `tokenizer.tiktoken` + `special_tokens.json` (no `tokenizer_config.json`, no `tokenizer.json`) → `AutoTokenizer.from_pretrained` raises → `model.tokenizer = None` → every synth dies with `'NoneType' object has no attribute 'encode'` at `content_sequence.py:196`. Fix: try AutoTokenizer, on failure load via the `tiktoken` lib directly (`_TiktokenTokenizer` — Qwen-family pat_str, `allowed_special="all"`). Verified: semantic range 151658–155753, vocab 155754.
2. **`fish_speech/models/dac/inference.py:20` — idempotent `OmegaConf.register_new_resolver("eval", eval)`.** Registry is global; after the sys.modules purge (engine reload after evict / switching fish checkouts) the module re-import crashes with `resolver 'eval' is already registered` → load 500. Fix: try/except ValueError.

### Synthesis quirks (verified behaviour)

- **Text-only input is degenerate**: no reference → ~5 semantic tokens → 0.19 s clip ("Generated 5 tokens"). S1-Mini **requires an audio prompt** — voice clone only. UI already has the ref-upload panel; API: `params: {"audio_prompt_id": "en-leo2", "ref_text": "…transcript…"}` (sidecar transcript fallback works). Verified clone: en-leo2 → 18.76 s @ 44.1 kHz in ~124 s.
- **HEAVY eviction confirmed live**: loading s1mini SIGKILLed the s2pro container (137) — 16 GB VRAM; they cannot coexist (as designed).
- Curl gotcha: `POST /synthesize/{engine}` needs `-H 'Content-Type: application/json'` (SynthReq JSON body); bare `-d` (urlencoded) 422s.
- First-load timing: download ~3.6 GB (gated, cached in `/opt/models/huggingface/hub/`), load 38–113 s, then ~2 s per short synth; clone synth ~2 min.

### Status

`docs/engine_compatibility.yaml`: s1mini `experimental` → **`verified`**. The transformers-5.x risk that kept it experimental is resolved (patches above). VM repo `/opt/arthur-tts-lab` is current; local repo commit includes patches + this doc + yaml update.
