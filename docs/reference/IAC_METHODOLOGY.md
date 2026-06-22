# TTS Lab — IaC Methodology, Issues & Lessons Learned

> **Date:** 2026-06-21
> **Status:** Living document — update as new issues are discovered
> **Related:**
> - [IAC_REWRITE_PLAN.md](IAC_REWRITE_PLAN.md) — Architecture and container design
> - [IAC_FIX_PLAN.md](IAC_FIX_PLAN.md) — Remaining work (5 engines)
> - [CONTAINERIZATION_ADHOC_REFERENCE.md](CONTAINERIZATION_ADHOC_REFERENCE.md) — Day-by-day ad-hoc log

---

## 1. The Two Approaches

### 1.1 Ad-Hoc Patching (What We Started With)

Ad-hoc patching means modifying a running container directly, without updating the Dockerfile or committing changes to version control.

**Workflow:**
```
1. Find a bug (engine crashes with 500)
2. SSH into VM
3. docker exec into container
4. pip install missing-package
5. docker cp fixed-file.py container:/opt/arthur/
6. docker restart container
7. Test — works!
8. Move on to next bug
```

**What's wrong with this:**
- The Docker image doesn't have the fix. If the container is recreated from the image, all fixes are lost.
- No version history. You can't `git blame` to see why a change was made.
- The fixes are invisible to CI/CD. GitHub Actions builds images without them.
- Python `.pyc` bytecode cache masks source file updates. You change the `.py` file but Python keeps running the old compiled version.
- Disk fills up with untracked pip installs and build cache.
- Rebuilding from scratch requires manually re-applying every fix.

**When ad-hoc is acceptable:**
- Emergency hotfix in production while the proper fix is being prepared
- One-time debugging to understand a problem
- Testing a hypothesis before committing to code changes

**When ad-hoc is NOT acceptable:**
- As the primary development methodology
- For anything that needs to survive a container restart
- For fixes that took more than 5 minutes to discover

### 1.2 IaC (Infrastructure as Code — What We Moved To)

Every fix is committed to git, baked into a Dockerfile, and the image is rebuilt from a clean build context. Nothing is done directly to a running container.

**Workflow:**
```
1. Edit code in local repo
2. git commit && git push
3. On VM: cd /opt/tts-lab-docker && sudo git pull
4. sudo docker build -f docker/Dockerfile.<target> -t <image>:latest .
5. sudo docker stop <container> && sudo docker rm <container>
6. sudo docker run ... <image>:latest
7. Test via curl
```

**Why this matters:**
- Every fix is version-controlled. `git log` shows the complete history.
- Images are reproducible. Anyone can clone the repo and build the same image.
- CI/CD (GitHub Actions) automatically builds and pushes images on push.
- Rollback is trivial: `git revert` + rebuild.
- No `.pyc` cache issues because Docker builds start from a clean filesystem.
- No disk bloat from ad-hoc pip installs inside running containers.

---

## 2. The IaC Implementation Pattern

### 2.1 Docker Image Hierarchy (Tiered Architecture)

```
Layer 1: tts-lab-base           (~7 GB)   Shared by ALL containers
Layer 2: tts-lab-stack-current  (~19 GB)  torch nightly + transformers 5.12
Layer 2: tts-lab-stack-mid      (~16 GB)  torch 2.10 stable + transformers 4.x
Layer 3: tts-lab-engine-current (~60 GB)  15 engines (all fixes baked in)
Layer 3: tts-lab-engine-mid     (~17 GB)  VibeVoice + Higgs
Layer 3: tts-lab-engine-qwen    (~8 GB)   Qwen3-TTS (hf-hub < 1.0)
Layer 1: tts-lab-orchestrator   (~7 GB)   Web UI + HTTP dispatch
```

Every layer is a separate `docker build` step. Changes to a layer only require rebuilding that layer and the ones above it. Docker caches unchanged layers.

### 2.2 The Build Context Must Be Git-Cloned

**Critical discovery:** The VM must have a `git clone` of the repo as the build context. If you manually copy files with `scp`, you will eventually miss one and the Docker image will have stale code.

```bash
# One-time setup on VM:
sudo rm -rf /opt/tts-lab-docker
sudo git clone https://github.com/faridnasiri/TTS-LAB.git /opt/tts-lab-docker

# For every rebuild:
cd /opt/tts-lab-docker && sudo git pull
sudo docker build -f docker/Dockerfile.<target> -t <image>:latest .
```

### 2.3 BuildKit Cache — The Silent Saboteur

Docker BuildKit maintains its own cache separate from Docker image layers. Even with `--no-cache`, BuildKit can serve stale file contents from its cache.

**Symptom:** You change a file in git, push, git pull on VM, rebuild with `--no-cache`, but the running container STILL has the old code.

**Root cause:** BuildKit has a separate cache (`docker builder prune` shows it). This cache is keyed by file content hash — but if BuildKit cached the file before your change, it might serve the old version even after `git pull`.

**Fix:**
```bash
# Before rebuilding after code changes:
sudo docker builder prune -af   # Clear BuildKit cache
sudo docker build --no-cache -f docker/Dockerfile.<target> -t <image>:latest .
```

**Lesson:** If a `--no-cache` rebuild doesn't pick up your code changes, the BuildKit cache is the culprit. Clear it.

### 2.4 Which Files Get Copied? — The COPY Layer Problem

**Critical discovery:** Engine Dockerfiles only copied `tts_lab_shims.py` and `tts_lab_engine_server.py`. The `tts_lab_engines.py` and `tts_lab_dispatch.py` files were inherited from the base image (Dockerfile.base has `COPY tts_lab_engines.py .`). When we changed engine code, the engine images didn't pick it up because the base image (built weeks ago) had the old version.

**Fix:** Engine Dockerfiles must COPY all four files:
```dockerfile
COPY tts_lab_shims.py /opt/arthur/
COPY tts_lab_engine_server.py /opt/arthur/
COPY tts_lab_engines.py /opt/arthur/
COPY tts_lab_dispatch.py /opt/arthur/
```

**Lesson:** Always check which files are inherited from parent images vs copied in the current Dockerfile.

---

## 3. Issues Encountered & Resolved

### 3.1 Torch Nightly Volatility

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| CUDA library mismatch | `libcudart.so.13: cannot open` | Nightly index moved from CUDA 12.8 to CUDA 13 toolchain | Added `cuda-nvrtc-13-0` package to base image + ldconfig |
| torchvision crash | `operator torchvision::nms does not exist` | Torch nightly lacks torchvision operator | Catch `RuntimeError` in shims import |
| `--no-deps` breaks deps | `libcupti.so.12: undefined symbol` | `--force-reinstall --no-deps` skips CUDA runtime libraries | Use `--force-reinstall` WITHOUT `--no-deps` |

### 3.2 Disk Space

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| Build fails | `no space left on device` | Docker images + BuildKit cache = 200+ GB | `docker builder prune -af` freed 128 GB |
| Repeated issue | Every few builds, disk fills | BuildKit accumulates cache for every layer | Add `docker builder prune -af` to pre-build cleanup |

### 3.3 Dependency Conflicts Between Engines

| Conflict | Engines | Resolution |
|----------|---------|------------|
| `huggingface-hub` version | f5tts needs ≥ 1.0, qwen3tts needs < 1.0 | Separate container (engine-qwen) |
| `transformers` version | 15 engines need 5.x, 3 engines need 4.x | Separate container (engine-mid) |
| `torch` version | 15 engines need nightly (sm_120), orpheus needs stable | Separate container (orpheus — blocked) |

### 3.4 Engine-Specific Issues

| Engine | Issue | Root Cause | IaC Fix |
|--------|-------|-----------|---------|
| **ChatTTS** | `LZMAError: Corrupt input data` | Library bug: `encode_prompt` format ≠ `_decode` format | Fall back to random speaker when ref voice requested |
| **Dia** | KV cache 128 vs 540 | `Dia-1.6B` (no suffix) config format incompatible | Delete broken model, use `Dia-1.6B-0626` only |
| **Dia** | `libnvrtc-builtins.so.13.0` missing | Torch nightly links CUDA 13 | Added `cuda-nvrtc-13-0` + ldconfig |
| **OuteTTS** | `soundfile.LibsndfileError: BytesIO` | Shim's `torchaudio.load` fallback called `str(path)` on BytesIO | Added `isinstance(path, io.BytesIO)` check |
| **OuteTTS** | 55s for "Hello world" | Default `max_length=8192` regardless of text | Auto-cap at `min(max(len*50, 2048), 4096)` |
| **OuteTTS** | `libnvrtc-builtins.so.13.0` missing | Same as Dia | Same fix |
| **f5tts** | `requires reference audio clip` | No default ref WAV | Auto-select first available WAV from `/tmp/tts_uploads/` |
| **qwen3tts** | `KeyError: 'default'` | ROPE_INIT_FUNCTIONS removed in transformers 5.x | Register `"default"` → `"llama3"` mapping |
| **qwen3tts** | `huggingface-hub>=0.34,<1.0 required` | f5tts needs ≥ 1.0 | Separate container (engine-qwen) |
| **qwen3tts** | `unexpected keyword 'inputs_embeds'` | qwen_tts 0.1.1 incompatible with transformers 5.12 | UNRESOLVED — needs transformers 5.0-5.11 |

### 3.5 General IaC Issues

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| Stale build context | Rebuilt images have old code | VM had manually-copied files, not git clone | `git clone` the repo as build context |
| BuildKit cache | `--no-cache` doesn't pick up changes | BuildKit has separate cache | `docker builder prune -af` before rebuild |
| Missing COPY layers | Code changes don't appear in container | Engine Dockerfiles only copied 2 of 4 .py files | COPY all four .py files in engine Dockerfiles |
| `.pyc` bytecode | `docker cp` fixes don't take effect | Python loads stale `.pyc` | Not an issue in IaC (clean build), but MUST document |

---

## 4. When to Add a New Container

The decision tree for adding a new container vs fixing in an existing one:

```
Does the engine have a dependency that CONFLICTS with existing engines?
├── YES → New container
│   ├── Different transformers major version? → New container
│   ├── Different huggingface-hub version range? → New container
│   ├── Different torch version? → New container (or new stack)
│   └── Needs SGLang/vllm? → New container (pre-built image)
└── NO → Fix in existing container
    ├── Missing pip package? → Add to Dockerfile RUN
    ├── Missing system library? → Add to Dockerfile.base apt-get
    ├── Code bug? → Fix in .py file, COPY in Dockerfile
    └── Wrong default params? → Fix in _synth_* function
```

**Rule of thumb:** If fixing the engine would break another engine, it needs its own container. If not, fix it in the existing container.

---

## 5. The IaC Checklist

Before starting any engine fix:

- [ ] Is the fix committed to git? (not just `docker cp`)
- [ ] Is the fix in the correct Dockerfile? (not applied to running container)
- [ ] Are all required `.py` files COPY'd in that Dockerfile?
- [ ] Have I pushed to GitHub? (`git push`)
- [ ] On VM: `git pull` in `/opt/tts-lab-docker/`
- [ ] On VM: `docker builder prune -af` (clear BuildKit cache)
- [ ] On VM: `docker build -f docker/Dockerfile.<target> -t <image>:latest .`
- [ ] Did the build actually rebuild the changed layers? (check output for "CACHED" vs new layer IDs)
- [ ] `docker stop` + `docker rm` old container
- [ ] `docker run` new container
- [ ] Test with `curl` — does it pass?
- [ ] Run full sweep: `python3 sweep.py`

---

## 6. Container Inventory

| # | Container | Port | Stack | Engines | Status |
|---|-----------|:----:|-------|---------|:------:|
| 1 | `tts-lab-orchestrator` | 8009 | base | Web UI | ✅ |
| 2 | `tts-lab-engine-current` | 8101 | current | 15 (bark→zonos + f5tts + qwen3tts) | ✅ |
| 3 | `tts-lab-engine-mid` | 8103 | mid | VibeVoice, Higgs | 🔧 |
| 4 | `tts-lab-engine-qwen` | 8104 | current | Qwen3-TTS | 🔧 |
| 5 | `tts-lab-engine-legacy` | 8102 | legacy | indextts, parler | 🔧 Deferred |
| 6 | `tts-lab-orpheus` | 8002 | cuda | Orpheus 3B | ❌ Blocked |
| 7 | `tts-lab-s2pro` | 8005 | SGLang | S2-Pro | ❌ Blocked |
