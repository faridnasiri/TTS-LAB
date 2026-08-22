# S2-Pro Investigation — 2026-06-23

> **Model:** fishaudio/s2-pro (Dual-AR 5B, 80+ languages)
> **Approach:** Custom SGLang container built from pip (tts-lab-sglang:latest)
> **Target hardware:** RTX 5060 Ti, 16 GB VRAM
> **Verdict:** ~~BLOCKED~~ **UNBLOCKED 2026-08-22** — sglang-omni 0.1.3 (PyPI) ships day-0 S2-Pro support, sm_120-validated upstream

## 0. Resolution (2026-08-22) — supersedes the BLOCKED verdict below

Upstream released **SGLang-Omni 0.1.3** (`pip install --pre sglang-omni==0.1.3`), which has first-class S2-Pro support (`config_cls: S2ProPipelineConfig`, served via `sgl-omni serve`). The whole dependency graph is **CUDA-13-aligned by design** (flashinfer[cu13], nixl-cu13, mooncake-cuda13, cutlass-dsl 4.6 → protobuf 6.30.2) — the same cu130 world as the lab's torch nightly. The old two-step force-reinstall (torch nightly onto a cu128 sglang[all] image) is dead; the new image installs sglang-omni in **one pip invocation** and then force-reinstalls torch nightly cu130 (pip proceeds past the stale `torch==2.11.0` pin with a warning).

Key facts for the current deployment:

| Item | Detail |
|---|---|
| Serving | `sgl-omni serve --model-path fishaudio/s2-pro --config /opt/s2pro_tts.yaml --port 8000` (ENTRYPOINT/CMD in `docker/Dockerfile.sglang`) |
| API | OpenAI-compatible `POST /v1/audio/speech`; **response body is raw WAV bytes** (not base64 JSON) |
| Cloning | `references: [{audio_path, text}]` — 10-30 s clip + transcript (ref_audio/ref_text shorthand also works) |
| Image | `tts-lab-sglang-omni:latest` (NOT `tts-lab-sglang:latest` — that tag is still the vibevoice/higgs POC `launch_server` image) |
| Base | `nvidia/cuda:12.8.2-devel-ubuntu22.04` — devel because flashinfer 0.6.14 JIT-compiles sm_120 kernels at first start (no prebuilt cache) and flash-attn-4 beta has no wheels; nvcc needed for both |
| Codec | `descript-audiotools 0.7.2` + `descript-audio-codec 1.0.0` installed `--no-deps` (audiotools pins protobuf<3.20 metadata-only — would break cutlass-dsl's protobuf 6.x) |
| JIT cache | First start compiles sm_120 kernels (~minutes). Persisted via `/opt/models/flashinfer-jit` bind mount so orchestrator stop/start cycles don't recompile |
| VRAM | ~11 GB always-resident while container runs — orchestrator stops/starts the container around EditX/LLM loads |

Deploy status: container built and validated on the VM 2026-08-22 — see `docs/engine_compatibility.yaml` (s2pro → `experimental`, validation checks currently pending).

---

## 1. Why S2-Pro Cannot Run Locally (original analysis — kept for history)



## 1. Why S2-Pro Cannot Run Locally

Unlike VibeVoice (which has a pip package with modeling code), S2-Pro is a weights-only HF repo. The entire inference stack lives inside SGLang:

- **Paged KV Cache** — 5B Dual-AR model with long context requires virtual-memory-style KV cache paging
- **RadixAttention** — shared prefix caching across the text+audio token streams
- **CUDA Graph Replay** — fused GPU execution graphs for the Dual-AR decoder

None of these are Python libraries you can `pip install`. They are runtime-level modifications to how the model executes. S2-Pro was designed exclusively for SGLang serving.

---

## 2. Attempted: Custom SGLang Container

Built `tts-lab-sglang:latest` from pip (`sglang[all]`) to get a newer transformers than the pre-built `lmsysorg/sglang-omni:dev` Docker image.

### 2.1 Build Result

| Component | Version |
|-----------|---------|
| sglang | 0.5.13.post1 |
| torch (installed by sglang) | 2.11.0 (stable) |
| transformers | 5.8.1 |
| flash-attn | 4.0.0b18 |
| CUDA (base image) | 12.8.2 |

Build succeeded in ~16 minutes. Image size: ~20 GB.

### 2.2 Launch Attempt: torch stable → sm_120 incompatible

```bash
docker run --rm --gpus all tts-lab-sglang:latest \
  --model fishaudio/s2-pro --host 0.0.0.0 --port 8000
```

**Result:** `RuntimeError: CUDA capability sm_120 is not compatible with the current PyTorch installation.`

**Root cause:** SGLang 0.5.13 pins torch 2.11.0 stable (CUDA 12.8), which supports up to sm_90 (H100). The RTX 5060 Ti is Blackwell sm_120 — requires torch 2.12+ nightly.

### 2.3 Force torch nightly → dependency conflict

```dockerfile
RUN pip install --no-cache-dir --force-reinstall --pre \
    torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

**Result:** `ERROR: No matching distribution found for nvidia-nvjitlink==13.0.88.*`

**Root cause:** SGLang's CUDA toolkit packages are pinned to exact versions matching torch 2.11 stable + CUDA 12.8. Nightly torch (CUDA 13.0) requires different nvidia-* package versions. The dependency graph cannot be resolved.

---

## 3. Blocker Summary

| Blocker | Detail | Dependency |
|---------|--------|:----------:|
| sm_120 (Blackwell) | torch stable doesn't support RTX 5060 Ti | SGLang must adopt torch nightly or 2.13+ |
| Transformers version | S2-Pro architecture needs transformers > 5.6.0 | SGLang 0.5.13 has 5.8.1 ✅ |
| CUDA toolkit pins | SGLang pins exact nvidia-* versions | Breaks when torch is upgraded |

**The sm_120 blocker is the root cause.** Even if we resolved the CUDA toolkit pins, SGLang's bundled flash-attn, triton kernels, and custom CUDA extensions are compiled for sm_90 max. They would need recompilation for sm_120.

---

## 4. Path Forward

### Only viable path: Wait for upstream SGLang

Two conditions must both be met:

1. **SGLang releases a Blackwell-compatible build** (torch >= 2.12 nightly or torch 2.13+ stable with sm_120 support)
2. **SGLang bundles transformers >= 5.8.0** (already met in 0.5.13)

When both are true, S2-Pro is runnable via the existing docker-compose definition:

```bash
docker compose --profile sglang up -d s2pro
```

The orchestrator already routes `S2PRO_SGLANG_URL=http://s2pro:8000/v1/audio/speech`.

### Monitor

- https://github.com/sgl-project/sglang/releases
- https://github.com/sgl-project/sglang/issues (search: sm_120, Blackwell)

### What Not To Do

- Do not attempt local inference (no modeling code exists)
- Do not fork S2-Pro model code (the architecture IS the SGLang runtime)
- Do not try to patch CUDA toolkit pins (flash-attn, triton kernels also need sm_120 recompilation)

---

## 5. Comparison with VibeVoice

| | VibeVoice | S2-Pro |
|---|:---:|:---:|
| Model loads locally? | ✅ Yes | ❌ No (no loader code) |
| pip package with modeling code? | ✅ vibevoice 0.0.1 | ❌ None |
| Blocker type | Inference pipeline (multimodal coupling) | Runtime (sm_120 + SGLang) |
| Fixable without upstream? | ❌ speech_outputs always None | ❌ No inference code at all |
| SGLang path viable? | ✅ When SGLang updates transformers | ✅ When SGLang supports sm_120 |
| Dockerfile ready? | ✅ (SGLang profile) | ✅ (Dockerfile.sglang + compose) |

---

## 6. Disposition

**S2-Pro: BLOCKED. Waiting on SGLang Blackwell (sm_120) support.**

The model requires SGLang at the architectural level — there is no local inference code to patch or bypass. The custom SGLang container we built gets closer (transformers 5.8.1 vs the pre-built image's 5.6.0) but hits the sm_120 wall. When SGLang releases a Blackwell-compatible version, S2-Pro is unblocked with zero code changes.
