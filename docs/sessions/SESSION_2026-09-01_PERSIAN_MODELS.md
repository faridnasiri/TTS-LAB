# Session 2026-09-01 — Persian (Farsi) TTS models

## Goal

Add the four recommended open-source Farsi TTS options without touching
existing stacks: Piper Persian voices, PerTTS (Piper-based — same files),
ParsVoice-XTTS fine-tune (own container), Tihu (rejected).

## Verdicts

| Model | Outcome | Why |
|---|---|---|
| Piper fa_IR voices (gyro + mana) | ✅ Added (Phase 4 downloads) | Zero code — engine auto-discovers `.onnx` files |
| PerTTS voices (amir/ganji…) | ✅ Free | PerTTS **is** Piper — its voices are `fa_IR-*-medium.onnx`, same drop-in |
| XTTS-v2 ParsVoice fine-tune | ⚠️ Parked — **weights do not exist** | Repo is README-only (verified 2026-09-01); engine-fa infra ready and waiting |
| Tihu | ❌ Rejected | C++ / Python-2-era build, 2020, robotic; lab neural engines dominate |
| hazm/parsivar normalizer | ✅ Already present (Chatterbox path only) | `_process_persian_text` — digits, char map, persian_phonemizer G2P |

"ParsiGoo" does **not** exist as a checkpoint — it's a training dataset.
The real fine-tune is **ParsVoice-XTTS** (`MohammadJRanjbar/ParsVoice-XTTS`),
fine-tuned on the 2,200-hour ParsVoice corpus (1,800+ speakers, U. Tehran) —
but as of **2026-09-01 the HF repo contains only `README.md` + `.gitattributes`**:
no `config.json`, no `.pth` weights, 0 downloads.

### ⚠️ Verified 2026-09-01 — no Persian XTTS weights exist anywhere on HF

Checked with the VM's HF token (`set -a; . /opt/arthur-tts-lab/.env`):

| Repo | Result |
|---|---|
| `MohammadJRanjbar/ParsVoice-XTTS` | ✅ exists — **README only** (2 siblings). The user's gate acceptance is moot: nothing is gated to download |
| `MohammadJRanjbar/parsvoice-xtts-v2` (the README's own example id) | 404 — repo does not exist |
| `alikhabazian/XTTS_Persian` | 33 dl — `best_model.pth` etc. all **0 bytes** (LFS pointers never pushed) |
| `alikhabazian/Xtts_persian_v2` | README only |
| `Fahd1199/xtts-darija` + others | Arabic/darija or XTTS-v2 mirrors — not Persian |

`xttsfa` is **parked**: loader tries the repos each load (so weights auto-appear
if/when uploaded) and otherwise raises a clear "weights not published" error;
UI panel and MODEL_INFO say PARKED. The engine-fa container, compose profile,
and all wiring stay — deleting them would just be re-work when the author
uploads (paper is EMNLP 2026 Main Conference; README has "FILL IN" training
placeholders — clearly work-in-progress).

## Changes

1. **`scripts/deploy/deploy_lab.ps1`** Phase 4 — download `fa_IR-gyro-medium`
   (rhasspy/piper-voices) + `fa_IR-mana-medium` (MahtaFetrat/Mana-Persian-Piper,
   ~1000-epoch Mana-TTS fine-tune — the better Persian voice) into
   `/opt/models/tts/`. Bind-mounted + symlinked into every container
   (`Dockerfile.base` line 111) → Piper engine + UI auto-discover, no code.
2. **`docker/Dockerfile.engine-fa`** (new) — port 8106, `--stack fa`.
   - `nvidia/cuda:12.8.2-runtime-ubuntu24.04` + python 3.12
   - `torch==2.13.0+cu130` stable (sm_120 — same pin as editx)
   - `coqui-tts` 0.27.x — **no `codec` extra** → torchcodec never installed;
     coqui ≥0.27.4 bundles no torch, so the nightly conflict class is gone
   - Smoke test at build: torch 2.13 + `import TTS`
3. **`docker-compose.yml`** — `engine-fa` service (profile `fa`, GPU,
   model-volume, HF_TOKEN) + `XTTSFA_URL` in the orchestrator.
4. **`tts_lab_config.py`** — `MODEL_INFO["xttsfa"]` (heavy, 3.2 GB est),
   `MODEL_ORDER` (after `xtts`), `SYNTH_TIMEOUT["xttsfa"]=300` (first load:
   ~1.8 GB gated HF download).
5. **`tts_lab_engines.py`** — `_load_xttsfa` (tries both repo-id spellings,
   raises with gated-access guidance) + `_synth_xttsfa` (language fixed `fa`,
   zero-shot clone via `audio_prompt_id` → `speaker_wav`; no ref = model
   default voice). Registered in LOADERS/SYNTHERS.
6. **`tts_lab_dispatch.py`** — `pkg_map["xttsfa"]="TTS"`, `_GPU_CONTAINERS`
   += `tts-lab-engine-fa`.
7. **`tts_lab_ui.py`** + **`tts_lab.py`** — params panel (clone upload,
   temperature, gated-model alert) + `/voices` entry.
8. **`Makefile`** — ENGINE list += `fa`, PORT 8106.
9. **`docs/engine_compatibility.yaml`** — `fa` stack + `xttsfa` engine
   (experimental). Also quoted one pre-existing unparseable s2pro
   `description:` scalar (bare `Persian: ` inside plain scalar) so the file
   actually parses.
10. **`tts_lab_engine_server.py` + `tts_lab_engines.py` + `tts_lab_dispatch.py`**
    — **piper voice never reached the loader in container mode.** The engine
    server called `LOADERS[name]()` with zero args, so every container synth
    used the default `en_US-ryan-high` — an English espeak-ng phonemizer on
    Arabic script spells each character by Unicode name ("Arabic seen, Arabic
    alef…") instead of reading the sentence. Fix: shared `_engine_load_arg()`
    helper (piper `voice`, matcha `voice`, chatterbox `model`, outetts
    `model_path`, parler `model_id`, zonos `variant`), `req.params` plumbed
    through both `/synthesize` call sites, and an arg-keyed load cache
    (`_current_arg` — same engine+arg reuses, arg change reloads).
    Same bug class affected matcha/chatterbox/outetts/parler/zonos voices.
11. **`tts_lab_engine_server.py`** — recycle-on-load-failure gate narrowed
    `if _HAS_VLLM:` → `if _HAS_VLLM and name == "editx":`. vllm is installed
    in engine-current but hosts no vLLM engines, so ANY failed load (e.g. a
    bogus piper voice) made the container exit 0 → crash-loop. Now only a
    real editx failure recycles.

## Piper Farsi fix — verified 2026-09-01 (end-to-end via orchestrator)

| Test | Result |
|---|---|
| Persian + `fa_IR-mana-medium` (the broken case) | ✅ **1.71 s** real speech (was 9.3 s of character-spelling) |
| Persian + mana, longer sentence | ✅ 3.23 s — proportional, reads the sentence |
| English + en voice (regression) | ✅ 1.38 s |
| Back-to-back en → mana → en | ✅ all synthesize; `arg change … — reloading` only on switch, same-arg calls `reusing` |
| Bogus voice → clean 500, container survives | ✅ `restarts=0`, healthy |

Engine logs confirm the plumbing: `Loading piper ... arg='fa_IR-mana-medium'`.
Deployed to VM container via docker cp (container's engines.py was older than
HEAD — upgraded to HEAD + fixes). `fa_IR-gyro-medium` still missing on VM
(only mana downloaded — gyro is on the Phase 4 list, ~120 MB).

## Blocker (user action, now moot)

`MohammadJRanjbar/ParsVoice-XTTS` shows a gate and is CPML-licensed — but the
gate gates **nothing**: the repo has no model files. The user accepted access
on 2026-09-01; the README is fetchable with the VM's token, and there is
simply nothing to download. **No further user action helps** — only the
author publishing weights does.

## Deploy

```bash
# 1. Piper Persian voices (host, ~120 MB) — THE working deliverable
#    — run deploy_lab.ps1 Phase 4, or download manually:
mkdir -p /opt/models/tts
wget -O /opt/models/tts/fa_IR-gyro-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx
wget -O /opt/models/tts/fa_IR-gyro-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx.json
wget -O /opt/models/tts/fa_IR-mana-medium.onnx https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx
wget -O /opt/models/tts/fa_IR-mana-medium.onnx.json https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx.json

# 2. engine-fa image — build NOW or when weights land; image is ready either way
make build-engine ENGINE=fa

# 3. Container (compose is canonical — host networking breaks URL routing)
docker compose --profile fa up -d          # starts engine-fa only
# or full fleet: docker compose --profile gpu --profile sglang --profile editx --profile fa up -d

# 4. When the author publishes weights: first xttsfa synth downloads them —
#    allow 300 s. Until then xttsfa shows the PARKED error.
```

## Open risks

- **ParsVoice-XTTS may stay unpublished** — nothing we can do; parked engine
  fails with a clear message (never a traceback). The Piper fa voices are the
  Persian win that works *today* (plus existing MMS-Fa / Kokoro-fa engines).
- **coqui-tts 0.27.x under torch 2.13.0 stable** (if weights appear) — the
  designed pairing, but unverified until a real synth runs; fall back to
  `coqui-tts==0.27.5` pin check or the `codec-cuda` extra.
