# Arthur TTS Lab — Known Issues & Next Steps

## Housekeeping — 2026-09-14 stability pass ✅

- **`arthur.service` (legacy “Arthur Henderson AI Bridge”, :8000) DISABLED.** Its venv
  `/opt/arthur-env` and app `/opt/arthur/arthur_server.py` were removed during the container
  migration (code archived at `archive/arthur_server.py`), but the unit stayed `enabled` with
  `Restart=always` / `RestartSec=5`. It had attempted a restart **453,690 times** — roughly one
  every 5 seconds since ~2026-08-19 — emitting ~4,800 journal lines/hour of pure noise. Fixed with
  `systemctl disable --now arthur.service`; re-enable only if that runtime is restored.
  *It was invisible to `systemctl --failed`, because a unit in auto-restart is `activating`, not
  `failed`. “No failed units” is not “nothing is failing” — also check
  `systemctl list-units --state=activating`.*
- **Reclaimed 18 GB of disk** (523 → 505 GB used; 83% → 81%) by deleting one unreferenced
  dangling image (`a01e7be6fcf5`, built 2026-08-22 — 32.3 GB apparent, 18 GB actually freed once
  shared layers were accounted for). Docker's reclaimable total dropped 29 GB → 2 GB. Worth
  re-checking periodically: this box has had a disk-full incident before.
- **Deliberately not touched:** the load average of ~12–16 on 12 cores is the operator's own
  *niced* Picocrypt volume brute-force (`~/picocrypt-venv/bin/recover.py`), not lab
  infrastructure. It yields to normal-priority work, so it slows diagnosis without starving the lab.
- **Diagnostic note — engine containers publish no host ports.** `engine-current` has no `ports:`
  in compose, so a failing `curl localhost:8101/health` is *expected*, not a fault: the orchestrator
  reaches it at `engine-current:8101` over the docker network. Only :8009 (orchestrator) and :8002
  (image lab) listen on the host.

### Reboot to kernel 6.8.0-138 — done 2026-09-14 ✅

The `reboot-required` flag (set since Sep 10 for kernel `6.8.0-138` + `libc6` security fixes) is now
cleared. Pre-flight mattered here, because booting the wrong kernel would have meant **no GPU at
all** — DKMS has an NVIDIA module only for 6.8.0-136/138, not for the 5.15.0-x entries also present:

- GRUB's default entry verified to point at `/boot/vmlinuz-6.8.0-138-generic` ✅
- `Filesystem state: clean` — the Sep-2026 boot hang was an *unclean* ext4 plus a refused `fsck -p`,
  so this check was not ceremonial.
- Every running service audited for reboot survival (only `static` units listed — nothing lost).
- The 6 stopped, profile-gated engines pinned to `--restart=no` so they could not all self-start and
  fight over the 16 GB GPU during boot.

Post-reboot: kernel `6.8.0-138`; module **and** userspace both `580.178.04` (no skew); module loaded
from `/lib/modules/6.8.0-138-generic/updates/dkms/nvidia.ko`; CDI spec regenerated with 89 refs to
`580.178.04`; **a newly-started GPU container works**; both health checks exit 0; zero failed units
and zero units in auto-restart. The system clock had drifted ~7 h behind and NTP corrected it at
boot (now `synchronized: yes`) — worth knowing when reading older journal timestamps.

### New: host health check (`tts-lab-host-health.timer`, every 30 min)

`scripts/utils/check_host_health.sh` (deployed to `/opt/tts-lab-ops/`) covers precisely the two
classes that hid from us: units stuck in `auto-restart` (invisible to `systemctl --failed`) and
disk/upgrade conditions. Read-only. `--quiet` suppresses routine OK lines but **never** problems or
warnings. `--install` writes its own unit + timer.

### ⚠️ 37 packages are pending — do NOT blanket-upgrade them

`apt-get upgrade` currently wants 37 packages, including a **major Docker/containerd jump**
(`docker-ce` 29.6.0, `containerd.io` 2.2.5, `docker-compose-plugin` 5.1.4), **Grafana 13**, the
**CUDA 12.8 toolkit**, and an NVIDIA driver *repackaging*. The repackaging is subtle: the NVIDIA CUDA
repo is pinned at priority **600** vs Ubuntu's **500**, so apt wants to move the driver from
`580.178.04-0ubuntu0.22.04.1` (Ubuntu) to `580.178.04-1ubuntu1` (NVIDIA) — the **same** driver
version, only different packaging provenance. No functional gain, non-trivial risk.

Apply these deliberately and one concern at a time, with a reboot wherever a driver is involved.
`check_host_health.sh` reports them as a `DELIBERATE-UPGRADE set` for this reason, and
`unattended-upgrades` is blacklisted from `nvidia-`/`libnvidia-` (see the 2026-09-13 entry).

---

## Latest incident — 2026-09-13 NVIDIA driver skew + stale CDI spec ✅ FIXED (hardened)

- **Was:** `nvidia-smi` on the VM reported `Failed to initialize NVML: Driver/library version
  mismatch`. The report was accurate — but it described only one of **three** faults, and the
  missing two were the ones that actually threatened the lab.
- **Root causes (all three had to line up):**
  1. **Driver upgraded without a reboot.** `unattended-upgrades` installed nvidia `580.178.04` on
     2026-09-11 06:48; the module resident in RAM stayed `580.173.02`. `nvidia-smi`, host
     CuPy/PyTorch, and every *new* host CUDA process failed (error 804 `forward compatibility was
     attempted on non supported HW`).
  2. **The CDI refresh died with status 127.** The same upgrade triggered
     `nvidia-cdi-refresh.service`, whose guard is `nvidia-smi -L || /usr/sbin/nvidia-smi -L ||
     /usr/lib/wsl/lib/nvidia-smi -L`. Ubuntu has no `/usr/lib/wsl/...`; with `nvidia-smi` failing
     *for reason 1* the shell chain ended on a missing binary and exited **127 “command not
     found”** — which reads as a broken unit rather than a broken driver. So
     `/var/run/cdi/nvidia.yaml` was never regenerated and kept **89 references to `580.173.02`**
     libraries `apt` had deleted.
  3. **No retry.** The unit sets `Restart=on-failure`, but it is `Type=oneshot` and systemd
     **ignores `Restart=` for oneshot services**. It stayed failed for two days, unnoticed.
- **Impact — wider than the host fault:** the container toolkit bind-mounts the libraries named in
  that spec into every GPU container, so **every GPU container start failed** with
  `failed to fulfil mount request: open .../libEGL_nvidia.so.580.173.02`. That covers restarts by
  deploy, by `systemctl`, **and by a reboot**. The lab only *looked* healthy because
  `tts-lab-engine-current` had been up since 2026-09-08 — before the upgrade. A reboot (the obvious
  fix for the host symptom) would have taken the whole lab down with no way back short of
  regenerating the spec by hand.
- **Fix applied:** stopped the GPU consumers, `modprobe -r`/`modprobe nvidia` (the on-disk `.ko`
  was already `580.178.04` for the running kernel 6.8.0-136 *and* the pending 6.8.0-138, so no
  reboot was required), then `nvidia-ctk cdi generate` to regenerate the spec. Verified: host
  `nvidia-smi` → `580.178.04`; a **newly started** GPU container gets the GPU; in-container
  `torch.cuda.is_available()` True with a real GPU matmul; image lab :8002 and orchestrator :8009
  both 200.
- **Hardening (so it cannot recur silently):** systemd drop-in replacing the brittle guard with
  `ExecStartPre=nvidia-smi -L` + `ExecStart=nvidia-ctk cdi generate`; a 10-min
  `nvidia-cdi-refresh.timer` so a failed refresh self-heals; a 15-min `tts-lab-gpu-health.timer`
  reporting module/user-space/CDI drift; and an `unattended-upgrades` blacklist for
  `nvidia-`/`libnvidia-` so a driver bump is a deliberate act.
- **`tts-lab-gpu-probe` is NOT a broken container.** It had shown "unhealthy" for days and was the
  session's first false trail, but it is deliberate infrastructure — `tts_lab_dispatch.py::_gpu_probe_exec`
  creates and reuses it (`tail -f /dev/null`, host PID namespace, GPU request) to run `nvidia-smi`
  in-container. Deleting it is futile (the next probe recreates it). What was wrong was its health
  state: it inherits the engine image's `HEALTHCHECK` (`curl :8105/health`), which cannot pass in a
  container running only `tail`, so `docker ps` always showed one unhealthy container on a healthy
  GPU host. Fixed in `tts_lab_dispatch.py` via `"Healthcheck": {"Test": ["NONE"]}` — effective on
  the next dispatch rebuild.
- **Files:** `scripts/utils/fix_nvidia_driver_mismatch.sh` (repair, idempotent, `--dry-run`),
  `scripts/utils/check_gpu_stack.sh` (deployed to `/opt/tts-lab-ops/`, drift detector),
  `scripts/utils/harden_nvidia_driver.sh` (applies the hardening).
- **Detail:** [session log 2026-09-13 NVIDIA driver + CDI](../sessions/SESSION_2026-09-13_NVIDIA_DRIVER_CDI.md)

---

## Latest incident — 2026-08-24 EditX garbage voices ✅ FIXED (see session log)

- **Was:** "editx tts produces garbage voices only" — all EditX clones came out as
  gibberish / near-silence / garbled speech regardless of ref or text.
- **Root causes (three, all verified with token dumps + whisper round-trip):**
  1. **Interleave rotation** — the model sometimes prepends spurious vq06 token(s),
     shifting the `[02,02,06,06,06]` frame the CosyVoice vocoder parses positionally
     → every frame decodes to garbage. Fixed in `tts.py` `_generate` (Dockerfile.engine-editx
     patch #3, v2): drop the leading offset with the best 5-chunk alignment, then
     **decode only the longest fully-valid chunk prefix** (also strips the tail
     leak). A fully garbage draw (text ids in audio slots) previously **crashed the
     container** (CUDA device-side assert in the flow decoder poisons the context →
     onnxruntime terminate, 2026-08-24 21:59); the v2 block raises a "bad draw —
     retry with a different seed" error instead, before any CUDA is touched.
  2. **No reliable EOS** — greedy argmax loops forever on a near-silent 8-token
     attractor (67 s ramble measured); the model only stops when sampling randomly
     hits `<|EOT|>`. Unseeded requests (vLLM seed 0) land on the attractor. Fixed by
     defaults in `_synth_editx`: temperature **0.5** + repetition_penalty **1.1** +
     text-scaled **max_tokens cap** (leak-killer) + deterministic crc32 seed
     (reproducible; UI 🎲 re-rolls a bad draw). ~90%+ clean clones per draw.
  3. **Persian unsupported** — Step-Audio-EditX was trained on EN/ZH/JA/KO only;
     the tokenizer has Persian chars but the model cannot produce Persian speech
     (verified with en/zh refs + fa refs across 8 combos). UI + MODEL_INFO now state
     this. Other engines (qwen3tts, etc.) remain the Persian path.
- **Second incident same day (22:19, high temp):** the bad-draw gate 500'd
  cleanly, but the engine server's generic-error path auto-evicted + reloaded —
  the old vLLM EngineCore still pinned its 12 GiB → new core init failed
  ("0.87/15.48 GiB free") → cascade → container recycled. Fixed: `_synth_editx`
  converts the bad-draw error into `SynthParamError` → 400 without evict/reload
  (the server's own comment at `tts_lab_engine_server.py:338` warns editx reload
  fails outright). Bad draws now cost a 🎲 click, never a container death.
- **Files:** `docker/Dockerfile.engine-editx` (patch #3), `tts_lab_engines.py`
  (`_synth_editx` defaults + bad-draw→400), `tts_lab_ui.py`, `tts_lab_config.py`.
- **Detail:** [session log 2026-08-24 editx](../sessions/EDITX-GARBAGE-2026-08-24.md)

---

## Latest incident — 2026-08-17 boot hang (recovered, see report)

- Guest hung ~8h in initramfs shell after forced restart — root fs carried ext4 error bit from unclean shutdown; boot `fsck -p` refused auto-repair. Fixed with `e2fsck -fy` from the shell; zero data loss.
- **`arthur-lab.service` (port 8001, bare-metal) RETIRED** (masked) — venv `/opt/arthur-bench-env` and lab code at `/opt/arthur/` were removed during container migration; service crash-looped since Aug 12. Orchestrator (8009) is the lab.
- **Full report + recovery runbook:** [incident-2026-08-17-boot-hang-initramfs.md](../issues/incident-2026-08-17-boot-hang-initramfs.md)

---

## Engines fixed (session 2026-06-27/29)

### omnivoice — remote routing ✅ FIXED
- **Was:** `"Not available: pip install omnivoice needed"` when called via orchestrator
- **Root cause:** `OMNIVOICE_URL` env var was not set in the orchestrator container — only 7 of 28 `_URL` vars were configured. The orchestrator tried to load omnivoice locally (no ML libs).
- **Fix applied:** Added all 28 `{ENGINE}_URL` env vars to Makefile `deploy-orchestrator` target. Recreated orchestrator container.
- **Files:** `Makefile`, `docker-compose.yml`

### omnivoice — voice cloning (torchcodec) ✅ FIXED
- **Was:** `AttributeError: module 'torchcodec' has no attribute 'decoders'` when using `audio_prompt_id`
- **Root cause:** `torchcodec` v99.0.0 is a **dummy stub** (`class AudioDecoder: pass`) installed to satisfy `f5-tts`'s pip dependency. When OmniVoice transcribes reference audio via transformers' ASR pipeline, `isinstance(inputs, torchcodec.decoders.AudioDecoder)` fails at runtime even though the attribute exists when checked directly.
- **Fix applied:** Monkey-patched `is_torchcodec_available` → `False` in `tts_lab_shims.py`. This forces the ASR pipeline to use its default preprocessing path.
- **Files:** `tts_lab_shims.py`
- **Note:** This is a monkey patch. Proper fix: isolate f5-tts and omnivoice into separate containers so the dummy torchcodec isn't needed, or install a real torchcodec version.

### LLM-TTS VRAM coordination ✅ FIXED
- **Was:** Heavy TTS engines (omnivoice, etc.) hit CUDA OOM because the LLM was using ~13.2 GB VRAM. No mechanism existed to evict the LLM before heavy TTS synthesis.
- **Root cause:** The LLM→TTS eviction protocol (`_evict_all_tts_engines()` before LLM inference) existed, but the reverse (TTS→LLM) was missing.
- **Fix applied:** Mounted Docker socket in orchestrator. Added `_stop_llm_container()` / `_start_llm_container()` in `tts_lab_dispatch.py`. Heavy TTS engines auto-stop the LLM container; LLM requests auto-restart it.
- **Files:** `tts_lab_dispatch.py`, `Makefile`, `docker-compose.yml`

---

## Engines fixed (as of 2026-04-25 session 2)

### indextts — ✅ FIXED
- **Was:** `AttributeError: 'IndexTTS2' object has no attribute 'load_model'`
- **Fix applied:** Removed `model.load_model()` from `_load_indextts()` in `tts_lab_engines.py`
- **Verified:** `IndexTTS2` has no `load_model` method; `__init__` loads all weights

### qwen3tts — ✅ FIXED (shim hardened)
- **Was:** `'Qwen3TTSSpeakerEncoderConfig' object has no attribute '_attn_implementation_autoset'`
- **Fix applied:** Added class-level shim in `tts_lab_shims.py` — sets `_attn_implementation_autoset = False` on `Qwen3TTSSpeakerEncoderConfig` at startup
- **Root cause:** Config `__init__` never calls `super().__init__()`, so `PretrainedConfig` never sets the attribute

---

## Open Issues

| Engine | Symptom | Notes |
|---|---|---|
| dia | Hangs >180s | Pre-existing. Engine loads but synthesis never completes. Not caused by 2026-06-27 changes. |
| styletts2 | Hangs >180s | Pre-existing. Same symptom as dia. Not caused by 2026-06-27 changes. |
| cosyvoice | Not installed | Needs `git clone FunAudioLLM/CosyVoice /opt/CosyVoice` |
| manatts | Not installed | Needs `pip install parallel-wavegan` |
| openvoice | Not installed | Needs `pip install openvoice` |
| neutts | Not configured | Needs `_load_neutts()` implementation in `tts_lab_engines.py` |
| orpheus | Gated HF model | Needs `huggingface-cli login` |
| csm | Gated HF model | Needs `huggingface-cli login` |
| parler | Version-gated | Requires `transformers==4.46.1` in engine-legacy container |
| higgs/vibevoice/s2pro | Containers not running | SGLang-based engines — `docker compose --profile sglang up -d` |
| indextts/parler | engine-legacy not running | `docker compose --profile legacy up -d` |

## Infra notes
- transformers pinning: pinned to `4.53.2` in requirements.txt
- All site-packages patches re-applied on every deploy via step 4.5 in `deploy_tts_lab.ps1`
- VM root disk expanded to 650 GB — no storage pressure
- E2E test script: `e2e_test.ps1` — run after every deploy to verify all fixes hold
- **Docker socket mounted in orchestrator** — enables LLM container start/stop for VRAM coordination
- **torchcodec v99.0.0 is a dummy stub** — do NOT upgrade it; if removed, f5-tts breaks. If kept as-is, ASR pipelines crash. The shim (`is_torchcodec_available → False`) bridges the gap.
