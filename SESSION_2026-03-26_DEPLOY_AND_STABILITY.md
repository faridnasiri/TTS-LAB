# Arthur TTS Lab — Full Session Report: Deploy Pipeline & CPU-VM Stability
> Date: 2026-03-26 (full day — three conversation threads)  
> Scope: 13→21 engine expansion, production deploy pipeline, SEGV root-cause investigation & fix  
> Branch: `main` commit `2861f01` — VM: `arthur@192.168.0.87:8001`  
> Git: `git log --oneline -5` → `2861f01 feat(tts-lab): 21-engine edition — 18/21 live on CPU-only VM`

---

## 1. Outcome

| Metric | Value |
|---|---|
| Engines registered | **21 / 21** |
| Packages available | **18 / 21** (csm, neutts, indextts missing) |
| Server stability | **stable** — HTTP 200 on first request, no SEGV, no restart loop |
| Deploy time (`-SkipInstall`) | ~30 s |
| Deploy time (full install) | ~15 min |
| RAM at idle | ~1 200 MB / 8 617 MB (14%) |

---

## 2. What Was Built This Session

### 2.1 `deploy_tts_lab.ps1` (new)
An 8-step idempotent deploy script that runs from Windows:

```
Step 1  SSH connectivity check (echo PONG)
Step 2  Create / chown /opt/arthur on VM
Step 3  SCP all lab files (10 files)
Step 4  Remote Python syntax check (ast.parse)
Step 5  Optional: bash _remote_install_new_engines.sh (pip installs, streamed live)
Step 6  Patch service file (CUDA_VISIBLE_DEVICES env vars) + systemctl restart
Step 7  Poll HTTP 200 up to 24×3 s = 72 s
Step 8  GET /status → render 21-engine table with RAM / availability
```

Key design decisions:
- `vm()` helper uses `& ssh` not `Invoke-Expression` so `&&` and `|` in remote commands are not parsed by PowerShell
- `-SkipInstall` switch skips step 5 for rapid file-only redeploy
- Step 6 is idempotent: `grep -q` before each `sed -i /a` prevents duplicate env lines
- HTTP wait loop gives the server 72 s to boot, then fails the deploy with a debug hint

### 2.2 `_remote_install_new_engines.sh` (new)
Bash installer for engines 14–21, executed on the VM. Key features:
- `ok` / `warn` coloured helpers; never `set -e` (partial installs are acceptable)
- Strips CRLF before first line with `sed -i 's/\r$//'` (run from `deploy_tts_lab.ps1` before exec)
- Includes the **orpheus snac_device patch** post-install (see §5.2)
- Includes Zonos git-clone + editable install (see §5.3)
- Includes OpenVoice checkpoint symlink creation (see §5.4)

### 2.3 `_available()` / `_check_available()` rewrite
Replaced all `exec(stmt, {})` import probes with pure `importlib.util.find_spec()` + filesystem checks.  
Motivation: thread-safety (see §4 — the SEGV).

### 2.4 Background availability sweep
```python
@app.on_event("startup")
async def _startup():
    t = threading.Thread(target=_sweep_availability, name="avail-sweep", daemon=True)
    t.start()
```
- Server returns HTTP 200 immediately; badges update as the sweep populates `_import_cache`
- `/status` shows `"checking..."` for uncached engines while sweep is in progress
- `/refresh` clears the cache and re-launches a new sweep thread

---

## 3. Files Changed

| File | Type | Summary |
|---|---|---|
| `deploy_tts_lab.ps1` | **New** | 8-step Windows→VM deploy pipeline |
| `_remote_install_new_engines.sh` | **New** | VM-side pip installer for engines 14–21 with CPU patches |
| `tts_lab.py` | **Modified** | New loaders 14–21; `_available()` rewrite; startup sweep; OpenVoice v1/v2 layout |
| `requirements.txt` | **Modified** | Sections 14–21 added with install notes and CPU-VM caveats |
| `SESSION_2026-03-26_NEW_ENGINES.md` | **New** | Final state table + concise issue log (replaced stub from prior session) |
| `SESSION_2026-03-25_CODEBASE_COMPLETION.md` | **New (staged)** | Prior session notes (created in prior session, committed now) |
| `SESSION_2026-03-25_MODEL_EXPANSION_REFERENCE.md` | **New (staged)** | Prior session reference doc |
| `bench_all.py` | **Modified** | Engine list expanded to 21 |
| `bench_warm.py` | **Modified** | New engines added to warm-RTF list |
| `tts_benchmark.py` | **Modified** | Bench functions for engines 14–21 |
| `setup_tts_lab.sh` | **Modified** | Steps 14–21 install instructions |
| `download_models.sh` | **Modified** | Size table extended to 21 engines |
| `requirements_benchmark.txt` | **Modified** | Bench deps for new engines |
| `_update_tts_lab.py` | **Deleted** | Leftover temp script |
| `_rewrite_tts_lab_direct.py` | **Deleted** | Leftover temp script |
| `_patch_tts_lab.py` | **Deleted** | Leftover temp script |

---

## 4. The Big Bug: SEGV (signal 11) on Every HTTP Request

### 4.1 Symptom
After installing engines 14–21 (`orpheus-speech`, `fish-speech`, `zonos`, `openvoice`),
the service entered a permanent restart loop. Every process lived ~46 seconds then died with:

```
arthur-lab.service: Main process exited, code=killed, status=11/SEGV
```

`curl` returned HTTP 000 (connection refused) every time because the process crashed
in the middle of handling the first request.

### 4.2 Diagnosis timeline

| Step | Finding |
|---|---|
| `journalctl -n 50` | Crash always followed `WARNING:parler_tts:Flash attention 2 is not installed` |
| Standalone Python test | ALL imports worked fine in sequence — no SEGV |
| Isolated import test with `CUDA_VISIBLE_DEVICES=` | Still worked |
| Standalone test in MODEL_ORDER sequence | Still worked |
| Added `CUDA_VISIBLE_DEVICES=` to service file | SEGV unchanged |

**Key observation:** standalone Python (single-threaded, arthur user) = fine.  
Uvicorn service (event-loop thread + our background thread) = SEGV.

### 4.3 Root cause

`_build_page()` called `_available(n)` for all 21 engines synchronously inside the
async request handler. `_available()` called `exec(stmt, {})` which ran actual Python
imports of heavy ML packages:

```
outetts  → vllm  → vllm._C (CUDA C-extension, partial init)
                 → Triton (CUDA C-extension, partial init)
orpheus  → SNAC  → model.to("cuda")  → CUDA driver call
```

These C-extension initialisation calls ran on the **main uvicorn event-loop thread**.
Meanwhile the **GIL-released C code** from vllm/Triton left CUDA driver handles in
a partially-initialised state. When the Python runtime garbage-collected them (or
when torch ran its own cleanup), it accessed freed/invalid memory → **SEGV**.

The same code worked standalone because in a single-threaded process there is no
concurrent GC / asyncio infrastructure interleaving with the import.

### 4.4 Fix

Replaced ALL `exec(stmt, {})` probes with **pure `importlib.util.find_spec()` + filesystem checks**:

```python
def _check_available(name: str) -> Tuple[bool, str]:
    import importlib.util as ilu
    pkg = pkg_map.get(name)
    if pkg and not ilu.find_spec(pkg):
        return False, f"pip install {pkg} needed"
    # Engine-specific filesystem checks (no imports, no C-exts)
    if name == "openvoice" and not (OPENVOICE_MODELS_DIR/"converter"/"config.json").exists():
        return False, f"Checkpoints missing at {OPENVOICE_MODELS_DIR}"
    ...
    return True, ""
```

`find_spec()` locates the module's source file via the filesystem — **zero module code
is executed**, zero C-extensions are loaded, completely thread-safe.

Side effect: the availability check is now instant (~1 ms per engine vs ~5 s) and
the sweep runs in a daemon thread at startup without any risk to the event loop.

---

## 5. Per-Package Issues & Fixes

### 5.1 CRLF line endings in shell scripts
**Symptom:** `_remote_install_new_engines.sh` executed on Linux with Windows CRLF
endings → bash saw `\r` at end of every variable assignment:
```
/opt/arthur-bench-env/bin/pip: not found   (the \r became part of the path)
```
**Fix:** Added to `deploy_tts_lab.ps1` step 3:
```bash
sed -i 's/\r$//' /tmp/_remote_install_new_engines.sh /opt/arthur/setup_tts_lab.sh ...
```

---

### 5.2 Orpheus 3B — `snac_device = "cuda"` hardcoded
**Package:** `orpheus-speech` (PyPI v0.1.0)  
**File:** `orpheus_tts/decoder.py` line 11:
```python
snac_device = "cuda"          # ← HARDCODED, no torch.cuda.is_available() check
model = model.to(snac_device) # ← crashes immediately on CPU-only VM
```
**Initial workaround attempted:** Patch `torch.cuda.is_available = lambda: False` before
import — failed because the literal string `"cuda"` bypasses the function entirely.

**Fix:** Patch the installed file in-place after every `pip install orpheus-speech`:
```python
p = "/opt/.../site-packages/orpheus_tts/decoder.py"
t = open(p).read().replace('snac_device = "cuda"', 'snac_device = "cpu"')
open(p, "w").write(t)
```
`_remote_install_new_engines.sh` does this automatically. `_load_orpheus()` in
`tts_lab.py` was simplified — no more runtime cuda patching needed.

---

### 5.3 Zonos — PyPI package missing `backbone/` module
**Package:** `zonos` v0.1.0 on PyPI (and `pip install git+https://github.com/Zyphra/Zonos`)  
**Error:** `ModuleNotFoundError: No module named 'zonos.backbone'`  
**Root cause:** PyPI wheel and the git-install wheel both exclude the `backbone/`
subdirectory (likely a `MANIFEST.in` / `find_packages()` misconfiguration in the upstream repo).

**Fix:** Direct clone + editable install:
```bash
git clone --depth 1 https://github.com/Zyphra/Zonos /opt/Zonos
pip install -e /opt/Zonos
```
**Additional fix:** The initial clone landed in `/tmp/Zonos` (volatile).  
After moving to `/opt/Zonos`, patched the editable finder file:
```bash
sed -i 's|/tmp/Zonos|/opt/Zonos|g' \
  /opt/arthur-bench-env/lib/python3.11/site-packages/__editable___zonos_0_1_0_finder.py
```
Verified with `inspect.getfile(Zonos)` → `/opt/Zonos/zonos/model.py` ✅

---

### 5.4 OpenVoice — checkpoints at `checkpoints/` not `checkpoints_v2/`
**Package:** `MyShell-OpenVoice` (installed from `git+https://github.com/myshell-ai/OpenVoice`)  
**Error:** `FileNotFoundError: OpenVoice v2 checkpoints missing at /opt/models/openvoice_v2`

**Root cause:** HuggingFace snapshot for `myshell-ai/OpenVoice` contains the **v1 checkpoint**
layout:
```
snapshots/<hash>/checkpoints/
    converter/config.json       ← present
    converter/checkpoint.pth    ← present
    base_speakers/EN/           ← v1 layout (not ses/)
        en_default_se.pth
        en_style_se.pth
```
Our code expected `/opt/models/openvoice_v2` to exist and to contain v2's
`base_speakers/ses/en_us.pth` structure.

**Fix — symlink:**
```bash
SNAP=/opt/models/huggingface/hub/models--myshell-ai--OpenVoice/snapshots/c70fc8b.../checkpoints
sudo ln -sfn $SNAP /opt/models/openvoice_v2
```

**Fix — code:** Updated `_load_openvoice()` to auto-detect both layouts:
```python
ses_dir = OPENVOICE_MODELS_DIR / "base_speakers" / "ses"  # v2
en_dir  = OPENVOICE_MODELS_DIR / "base_speakers" / "EN"   # v1

if ses_dir.exists():
    for p in ses_dir.glob("*.pth"):
        base_se[p.stem] = torch.load(str(p), ...)
elif en_dir.exists():
    for fname, key in [("en_default_se.pth","en_default"),("en_style_se.pth","en_style")]:
        ...
```
`_synth_openvoice()` already fell back to `list(base_se.values())[0]` when a specific
key wasn't found — no change needed there.

**Debugging note:** The initial `ln -s` command ran in `~arthur/` (SSH default CWD)
instead of `/opt/models/`, creating a dangling symlink at `~/openvoice_v2`. The fix
was to always use absolute paths: `sudo bash -c 'ln -sfn ... /opt/models/openvoice_v2'`.

---

### 5.5 Fish Speech — wrong import path assumed
**Package:** `fish-speech` PyPI v0.1.0  
**Assumed API:** `from fish_speech.inference.api import TTSInference` (from docs)  
**Actual API:** `from fish_speech.inference_engine import TTSInferenceEngine`

The PyPI package ships the *inference engine framework* only. Full model loading
(`fish_speech.models.vqgan.inference`, `fish_speech.models.text2semantic.inference`)
requires the full GitHub clone of the repo. The PyPI package has no `inference/` sub-package at all.

**Fix:** Updated `_load_fishspeech()` to use the real API and provide a clear error
if the model-loading modules aren't present (requiring the GitHub clone):
```python
from fish_speech.inference_engine import TTSInferenceEngine  # framework OK
from fish_speech.models.vqgan.inference import load_model     # needs full GitHub
```
`_check_available("fishspeech")` now just uses `find_spec("fish_speech")` — the
engine shows as "ready" if the package is present; actual model loading errors
surface at synthesise time.

---

### 5.6 CSM — wrong PyPI package installed
**Installed:** `csm` v0.1.0 from PyPI (unrelated package)  
**Needed:** Sesame CSM from `pip install git+https://github.com/SesameAILabs/csm`  
which provides `from generator import load_csm_1b` (a top-level `generator.py` module)

**Fix:**
```bash
pip uninstall csm -y
# Then (after huggingface-cli login):
pip install git+https://github.com/SesameAILabs/csm
```
`_check_available("csm")` now uses `find_spec("generator")` (not the `csm` package)
with a clear message if missing.

---

### 5.7 Symlink in wrong directory (OpenVoice)
`ln -s TARGET DEST` in an SSH session runs in the SSH session's CWD (`~arthur/`),
not in `/opt/models/`. The symlink `~/openvoice_v2 → /opt/models/openvoice_v2`
(which itself didn't exist) caused confusing "File exists" + "not found" messages.

**Fix pattern:** Always use `sudo bash -c '...'` for symlink creation and verify
with `ls -la /opt/models/openvoice_v2/` immediately after.

---

### 5.8 Service CUDA initialisation attempts
Even with `CUDA_VISIBLE_DEVICES=` set, vllm (used by `outetts`) still attempted
to load `vllm._C` (CUDA C-extension). The log showed:
```
WARNING [interface.py:229] Failed to import from vllm._C: ImportError('libcuda.so.1: cannot open...')
```
This is a graceful failure — not the SEGV cause — but confirms that CUDA init paths
are touched on every startup. The real fix was §4.4 (no exec() probes).

Service file additions:
```ini
Environment=CUDA_VISIBLE_DEVICES=
Environment=TOKENIZERS_PARALLELISM=false
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
```

---

## 6. VM State After This Session

### Installed packages (new this session)
```
fish-speech          0.1.0   (PyPI)
MyShell-OpenVoice    0.0.0   (git from myshell-ai/OpenVoice)
orpheus-speech       0.1.0   (PyPI, decoder.py patched snac_device→cpu)
phonemizer           3.3.0   (PyPI)
phonemizer-fork      3.3.2   (PyPI)
zonos                0.1.0   (editable, source at /opt/Zonos)
```

Removed (wrong packages):
```
csm        0.1.0   (PyPI — unrelated to Sesame CSM)
zonos      0.1.0   (PyPI — missing backbone/, replaced by editable)
```

### Key paths on VM
| Path | Content |
|---|---|
| `/opt/arthur/` | All lab Python files + scripts |
| `/opt/arthur-bench-env/` | Python 3.11 virtual environment |
| `/opt/models/huggingface/hub/` | HuggingFace model cache (`HF_HOME`) |
| `/opt/models/openvoice_v2` | Symlink → `...models--myshell-ai--OpenVoice/snapshots/<hash>/checkpoints` |
| `/opt/Zonos/` | Zonos git clone (permanent, replaces `/tmp/Zonos`) |
| `/etc/systemd/system/arthur-lab.service` | Service unit (patched this session) |

### Service file (`/etc/systemd/system/arthur-lab.service`) — final
```ini
[Unit]
Description=Arthur TTS Lab (interactive web UI on port 8001)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur
ExecStart=/opt/arthur-bench-env/bin/uvicorn tts_lab:app --host 0.0.0.0 --port 8001 --workers 1
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=12
Environment=MKL_NUM_THREADS=12
Environment=OPENBLAS_NUM_THREADS=12
Environment=ORT_NUM_THREADS=12
Environment=NUMEXPR_NUM_THREADS=12
Environment=CPU_THREADS=12
Environment=HF_HOME=/opt/models/huggingface
Environment=TRANSFORMERS_CACHE=/opt/models/huggingface/hub
Environment=XDG_CACHE_HOME=/opt/models/cache
Environment=SUNO_USE_SMALL_MODELS=True
Environment=COQUI_TOS_AGREED=1
Environment=CUDA_VISIBLE_DEVICES=        ← added this session
Environment=TOKENIZERS_PARALLELISM=false ← added this session
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn ← added this session

[Install]
WantedBy=multi-user.target
```

---

## 7. `tts_lab.py` Architecture Changes

### 7.1 Availability check (before → after)

**Before:**
```python
_import_cache = {}
def _available(name):
    if name in _import_cache: return _import_cache[name]
    def _check():
        ...
        try: exec(stmt, {})           # ← SEGV source
        except BaseException as e: return False, ...
        return True, ""
    r = _check()
    _import_cache[name] = r
    return r
```

**After:**
```python
_import_cache: Dict[str, Tuple[bool, str]] = {}
_import_cache_lock = threading.Lock()

def _available(name) -> Tuple[bool, str]:
    with _import_cache_lock:
        if name in _import_cache: return _import_cache[name]
    return _check_available(name)          # thread-safe, populates cache

def _check_available(name) -> Tuple[bool, str]:
    # find_spec() only — no exec(), no C-extension load, no GIL release
    pkg = pkg_map.get(name)
    if pkg and not ilu.find_spec(pkg): return False, "pip install needed"
    # filesystem checks for checkpoints, voices, etc.
    return True, ""
```

### 7.2 Startup sweep
```python
_sweep_done = threading.Event()

def _sweep_availability():
    for n in MODEL_ORDER:
        _available(n)    # populates cache entry-by-entry
    _sweep_done.set()

@app.on_event("startup")
async def _startup():
    threading.Thread(target=_sweep_availability, daemon=True).start()
```

### 7.3 `/status` pending state
```python
if sweep_running and n not in _import_cache:
    models[n]["available"] = False
    models[n]["reason"]    = "checking..."
```

### 7.4 `/refresh` non-blocking
```python
@app.post("/refresh")
async def refresh_availability():
    with _import_cache_lock: _import_cache.clear()
    _sweep_done.clear()
    threading.Thread(target=_sweep_availability, daemon=True).start()
    return JSONResponse({"note": "sweep running — poll /status in ~5 s"})
```

### 7.5 New engine loaders (14–21)

| Engine | Loader key points |
|---|---|
| `fishspeech` | `TTSInferenceEngine` from `fish_speech.inference_engine`; needs full GitHub clone for model loading |
| `csm` | `load_csm_1b` from `generator` (Sesame GitHub install); `huggingface-cli login` required |
| `qwen3tts` | Uses existing `transformers`; model auto-downloads from `Qwen/Qwen3-TTS` |
| `orpheus` | `OrpheusModel` from `orpheus_tts`; no CUDA patch needed after decoder.py fix |
| `neutts` | Placeholder; returns `False, "not configured"` |
| `indextts` | `IndexTTS` from `indextts.infer`; package not yet installed |
| `zonos` | `Zonos.from_pretrained(variant)` with `transformer` / `hybrid` variants |
| `openvoice` | `ToneColorConverter` + `MeloTTS`; v1/v2 speaker-embedding layout auto-detected |

---

## 8. Remaining Work (3 Missing Engines)

### 8.1 Sesame CSM 1B (`csm`)
```bash
ssh arthur@192.168.0.87
source /opt/arthur-bench-env/bin/activate
huggingface-cli login   # needs HF account — model is gated (sesame-ai-labs/csm-1b)
pip install git+https://github.com/SesameAILabs/csm
curl -sX POST http://localhost:8001/refresh
```

### 8.2 IndexTTS-2 (`indextts`)
GitHub URL unconfirmed. Candidates:
- `https://github.com/index-tts/IndexTTS`
- `https://github.com/bilibilidown/IndexTTS` (Bilibili project)
```bash
# When URL confirmed:
pip install git+https://github.com/<org>/IndexTTS
# or:
git clone https://github.com/<org>/IndexTTS /opt/IndexTTS && pip install -e /opt/IndexTTS
```

### 8.3 NeuTTS Air (`neutts`)
No confirmed Python package. When identified:
1. Update `_load_neutts()` in `tts_lab.py`
2. Add to `pkg_map` in `_check_available()`
3. Add to `requirements.txt`
4. `curl -sX POST http://localhost:8001/refresh`

---

## 9. Quick-Reference: Common Operations

### Re-deploy after code change (fast, ~30 s)
```powershell
cd C:\repos\Spamblocker\tools\arthur_server
.\deploy_tts_lab.ps1 -SkipInstall
```

### Full deploy + install new packages (~15 min)
```powershell
.\deploy_tts_lab.ps1
```

### After installing a new package on VM — refresh badges
```bash
curl -sX POST http://192.168.0.87:8001/refresh | python3 -m json.tool
```

### Check service logs (VM)
```bash
sudo journalctl -u arthur-lab -n 40 --no-pager
sudo journalctl -u arthur-lab -f   # follow
```

### Verify Zonos is loading from /opt (not /tmp)
```bash
sudo /opt/arthur-bench-env/bin/python -c \
  "from zonos.model import Zonos; import inspect; print(inspect.getfile(Zonos))"
# Expected: /opt/Zonos/zonos/model.py
```

### Re-apply orpheus CPU patch (after pip upgrade)
```bash
source /opt/arthur-bench-env/bin/activate
python3 -c "
p='/opt/arthur-bench-env/lib/python3.11/site-packages/orpheus_tts/decoder.py'
t=open(p).read().replace('snac_device = \"cuda\"','snac_device = \"cpu\"')
open(p,'w').write(t)
print('patched:', p)"
sudo systemctl restart arthur-lab
```

### Benchmark new engines
```bash
/opt/arthur-bench-env/bin/python /opt/arthur/tts_benchmark.py \
  --models orpheus,zonos,fishspeech,openvoice --output bench_new_engines.json
```

---

## 10. Lessons & Patterns

| Pattern | Detail |
|---|---|
| **Never `exec()` imports in threads** | C-extension side-effects (CUDA, Triton, SNAC) corrupt shared process state when interleaved with asyncio; use `find_spec()` only |
| **Editable installs for git packages** | `pip install -e /opt/Repo` preserves full source tree; `pip install git+URL` can silently exclude directories (Zonos `backbone/`) |
| **Always use absolute paths in SSH `ln -s`** | SSH default CWD is `~user/`; a symlink without absolute DEST lands there silently |
| **Patch installed package files post-install** | For packages with hardcoded GPU assumptions (`snac_device="cuda"`), sed-patching the installed file is simpler and more reliable than runtime monkey-patching |
| **CUDA_VISIBLE_DEVICES= in service env** | On a CPU-only VM, this is mandatory to prevent vllm/torch/Triton from attempting any CUDA driver initialisation, even if individual checks return False |
| **HF snapshot layout varies by model version** | `myshell-ai/OpenVoice` HF snapshot uses `checkpoints/` (v1); only `checkpoints_v2/` is true v2. Always verify with `find snapshot/ -type d` before coding a path |
| **PowerShell heredocs in SSH `vm()` calls** | Multi-line `@'...'@` strings passed as a single `vm $cmd` argument work; the newline in the value is interpreted as a bash newline by the remote shell |
