<#
.SYNOPSIS
    Deploy the Arthur Image & Video Lab to the Ubuntu VM.

.DESCRIPTION
    8-phase idempotent deployment. Run from the Windows dev machine.
    Mirrors the structure of scripts/deploy/deploy_lab.ps1 (TTS lab).

.PARAMETER VM
    Target VM IP address. Default: 192.168.0.87

.PARAMETER User
    SSH username. Default: arthur

.PARAMETER Phase
    Run only a single phase (1-8). Omit to run all phases.

.PARAMETER SkipPhases
    Comma-separated phase numbers to skip. E.g. "4" to skip model download.

.PARAMETER HFToken
    HuggingFace token. If omitted, reads from secrets.env or $env:HF_TOKEN.

.EXAMPLE
    .\deploy_image_lab.ps1                   # Fresh VM: all phases
    .\deploy_image_lab.ps1 -Phase 5          # Re-deploy code only (~20 s)
    .\deploy_image_lab.ps1 -Phase 6          # Restart service only
    .\deploy_image_lab.ps1 -SkipPhases "4"  # Skip model download

.NOTES
    The HiDream O1-Dev engine needs the ComfyUI sidecar (arthur-comfy.service,
    port 8188) — a SEPARATE one-off bootstrap this script does not cover:
    ssh $User@$VM 'bash -s' < .\bootstrap_comfy.sh
    (own venv + ComfyUI + the fp8_scaled checkpoint, ~25 min on first run.)
#>
param(
    [string]  $VM         = "192.168.0.87",
    [string]  $User       = "arthur",
    [int]     $Phase      = 0,          # 0 = all phases
    [string]  $SkipPhases = "",
    [string]  $HFToken    = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ──────────────────────────────────────────────────────────────────────
# This script lives in scripts/deploy/ — the lab code files sit at the repo
# root, two levels up (the housekeep commit 5532d6b moved the scripts into
# subdirectories without adjusting path resolution). Fall back to the legacy
# layout (script deployed next to the code) if the files aren't at the root.
$repoRoot = Split-Path $PSScriptRoot -Parent | Split-Path -Parent
if (-not (Test-Path (Join-Path $repoRoot "image_lab.py"))) {
    $repoRoot = $PSScriptRoot
}

# ── Resolve HF token ──────────────────────────────────────────────────────────
if (-not $HFToken) {
    if ($env:HF_TOKEN) {
        $HFToken = $env:HF_TOKEN
    } elseif (Test-Path "$repoRoot\secrets.env") {
        $line = (Get-Content "$repoRoot\secrets.env") | Where-Object { $_ -match "^HF_TOKEN=" }
        if ($line) { $HFToken = ($line -split "=", 2)[1].Trim() }
    }
}
if (-not $HFToken) {
    Write-Warning "HF_TOKEN not found. FLUX.2 [dev] needs it at runtime (set in VM .env)."
    Write-Warning "You can set it later: ssh $User@$VM 'echo HF_TOKEN=hf_xxx >> /opt/arthur-img/.env'"
}

# ── SSH / SCP helpers ─────────────────────────────────────────────────────────
$SSH_KEY = "$env:USERPROFILE\.ssh\id_arthur_vm"
$SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i `"$SSH_KEY`""

function Invoke-SSH {
    param([string]$Cmd)
    $full = "ssh $SSH_OPTS $User@$VM `"$Cmd`""
    Write-Host "  » $Cmd" -ForegroundColor DarkGray
    $out = Invoke-Expression $full
    if ($LASTEXITCODE -ne 0) { throw "SSH command failed (exit $LASTEXITCODE): $Cmd" }
    return $out
}

function Invoke-SCP {
    param([string[]]$LocalFiles, [string]$RemoteDest)
    $files = $LocalFiles -join " "
    $full  = "scp $SSH_OPTS $files $User@${VM}:${RemoteDest}"
    Write-Host "  » SCP → $RemoteDest" -ForegroundColor DarkGray
    Invoke-Expression $full | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "SCP failed for: $files" }
}

# ── Phase runner ──────────────────────────────────────────────────────────────
$skip = @{}
if ($SkipPhases) { ($SkipPhases -split ",") | ForEach-Object { $skip[[int]$_.Trim()] = $true } }

function Run-Phase {
    param([int]$Num, [string]$Title, [scriptblock]$Body)
    if ($Phase -gt 0 -and $Phase -ne $Num) { return }
    if ($skip[$Num]) { Write-Host "[Phase $Num] SKIPPED — $Title" -ForegroundColor DarkYellow; return }
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Phase $Num — $Title" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    & $Body
    Write-Host "[Phase $Num] ✓ Done" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — System packages + directories
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 1 "System packages + directory layout" {
    Invoke-SSH "sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libglib2.0-0 libsm6 libxext6 libgl1 git-lfs python3.11 python3.11-venv python3-pip 2>&1 | tail -5"
    # Ubuntu jammy ships python3.11 as 3.11.0~rc1 — a PRE-RELEASE CPython that
    # segfaults triton reduction codegen (Boogu's fp8 act-quant kernels die with
    # a code_generator crash under rc1; final 3.11.x runs them fine). If the
    # distro package is an rc, upgrade to final 3.11.x from the deadsnakes PPA.
    Invoke-SSH @"
if /usr/bin/python3.11 --version | grep -qi 'rc'; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common >/dev/null 2>&1 || true
  sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11 python3.11-venv 2>&1 | tail -2
  echo 'python3.11 upgraded from rc -> final (deadsnakes)'
else
  echo 'python3.11 is already a final release'
fi
/usr/bin/python3.11 --version
"@
    Invoke-SSH "sudo mkdir -p /opt/arthur-img /opt/models/image /opt/arthur-gen/images /opt/arthur-gen/videos"
    Invoke-SSH "sudo chown -R ${User}:${User} /opt/arthur-img /opt/arthur-gen"
    Invoke-SSH "sudo git lfs install --system --skip-repo 2>/dev/null || true"
    Write-Host "  Directories and system packages ready."
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Python venv + PyTorch (CUDA 12.8)
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 2 "Python venv + PyTorch CUDA 12.8" {
    Invoke-SSH @"
sudo bash -c 'if [ ! -d /opt/arthur-img-env ]; then python3.11 -m venv /opt/arthur-img-env && chown -R ${User}:${User} /opt/arthur-img-env && echo venv created; fi'
"@
    $pip = "/opt/arthur-img-env/bin/pip"
    Invoke-SSH "$pip install --upgrade pip wheel setuptools -q"
    # PyTorch — 2.11.0+cu128 (first stable line with sm_120 for the 5060 Ti).
    # DO NOT pin 2.10.0: it lacks sm_120 and would revert a working venv.
    # Verified pairing on the VM: torchvision 0.26.0+cu128.
    Invoke-SSH "$pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128 -q"
    Invoke-SSH "/opt/arthur-img-env/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Engine Python packages
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 3 "Engine Python packages" {
    $pip = "/opt/arthur-img-env/bin/pip"

    # Core inference stack — floors reflect the verified live pairing on the
    # VM (2026-09-07→08): diffusers 0.40.0 + transformers 5.9.0 + hub 1.30.0.
    # The T2I swap first shipped on 0.38.0 (enough for ErnieImagePipeline/
    # ZImagePipeline/QwenImagePipeline + both GGUF transformer classes), but
    # ERNIE's NVFP4 route needs NunchakuLite — first released in diffusers
    # 0.40.0 (PyPI). The 0.38→0.40 bump pulls huggingface-hub ≥1.23 (1.30.0
    # live) + accelerate ≥0.31 and was regression-swept across all 6 kept
    # engines (2026-09-08, all green).
    #   diffusers >= 0.40 — NunchakuLiteQuantizer floor (ernie NVFP4 path)
    #   transformers >= 5.0 — ErnieImagePipeline raises ImportError at module
    #     import below 5.0 (hard gate, verified in diffusers main)
    Invoke-SSH "$pip install 'diffusers>=0.40.0' 'transformers>=5.0.0' accelerate safetensors sentencepiece protobuf -q"

    # Quantisation (for FLUX.2 4-bit)
    Invoke-SSH "$pip install bitsandbytes -q"

    # FastAPI server stack
    Invoke-SSH "$pip install fastapi uvicorn[standard] python-multipart -q"

    # Image/video I/O
    Invoke-SSH "$pip install Pillow imageio imageio-ffmpeg opencv-python-headless -q"

    # HuggingFace hub (for token handling)
    Invoke-SSH "$pip install huggingface_hub -q"

    # Requests (remote T5 encoder for FLUX.2)
    Invoke-SSH "$pip install requests -q"

    # GGUF loading support (required for FLUX.2 Klein 9B-KV, Z-Image and
    # Qwen-Image 2512 single-file GGUF transformers)
    Invoke-SSH "$pip install 'gguf>=0.10.0' -q"

    # Ideogram 4 (cloned from GitHub, install as editable)
    Invoke-SSH "test -d /opt/arthur-img/ideogram4 && echo 'ideogram4 already cloned' || (cd /opt/arthur-img && git clone https://github.com/ideogram-ai/ideogram4.git)"
    Invoke-SSH "$pip install -e /opt/arthur-img/ideogram4 -q"

    # Boogu-Image (cloned from GitHub, install as editable, NO deps — the venv
    # already has the required torch 2.11 / diffusers / transformers, and the
    # package pins torch<2.12 which must NOT be resolved here)
    Invoke-SSH "test -d /opt/arthur-img/Boogu-Image && echo 'Boogu-Image already cloned' || (cd /opt/arthur-img && git clone https://github.com/boogu-project/Boogu-Image.git)"
    Invoke-SSH "$pip install -e /opt/arthur-img/Boogu-Image --no-deps -q"
    # Boogu runtime deps: omegaconf (imported by the upstream pipeline) and the
    # `kernels` package (transformers 5.9's finegrained-fp8 loader requires it;
    # kernels-community/finegrained-fp8 pack v1 is resolved from HF hub at
    # first fp8 load). Pin <0.15 — 0.14.x is the verified line.
    Invoke-SSH "$pip install omegaconf 'kernels>=0.14,<0.15' -q"

    # Patch ideogram4 pipeline to add local_files_only=True for offline mode.
    # NOTE: Do NOT patch the tokenizer — Qwen3-VL tokenizer files (vocab.json,
    # merges.txt, config.json) legitimately don't exist in the nf4 repo (404).
    # AutoTokenizer handles missing optional files gracefully with network access,
    # but local_files_only=True turns that into a hard error. The tokenizer makes
    # a few cheap HEAD requests on first load, then caches results.
    # Patch ideogram4 pipelines for offline mode. Run from a FILE (not passed
    # through Invoke-SSH): Invoke-Expression re-parses the composed command
    # line, which mangled quoted heredocs twice on 2026-09-07.
    $patchScript = Join-Path $env:TEMP "ideogram4_patch.sh"
    @'
#!/bin/bash
sed -i '/trust_remote_code=True,$/s/trust_remote_code=True,/trust_remote_code=True, local_files_only=True,/' /opt/arthur-img/ideogram4/src/ideogram4/pipeline_ideogram4.py
sed -i '/trust_remote_code=True, low_cpu_mem_usage=True,/s/trust_remote_code=True, low_cpu_mem_usage=True/trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True/' /opt/arthur-img/ideogram4/src/ideogram4/pipeline_ideogram4.py
echo 'ideogram4 pipeline patched for offline mode (transformer weights only)'
'@ | Set-Content -Path $patchScript -Encoding UTF8
    Invoke-SCP $patchScript "/tmp/ideogram4_patch.sh"
    Remove-Item $patchScript -Force
    Invoke-SSH "bash /tmp/ideogram4_patch.sh"

    Write-Host "  All packages installed."
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4 — Model pre-download (optional, large)
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 4 "Model pre-download (large — skip with -SkipPhases 4)" {
    if (-not $HFToken) {
        Write-Warning "  No HF_TOKEN — skipping model download. Set token and re-run Phase 4."
        return
    }

    $py = "/opt/arthur-img-env/bin/python"

    # Write download script locally, SCP it — avoids heredoc issues in SSH
    $localScript = Join-Path $env:TEMP "imglab_download.py"
    # Fix permissions so arthur can write to the HF cache directory
    Invoke-SSH "sudo chown -R arthur:arthur /opt/models/"

    $dlScript = @"
import os, shutil
# Use HF_HOME so from_pretrained() will find cached models automatically
os.environ['HF_HOME'] = '/opt/arthur-img-models/huggingface'
from huggingface_hub import snapshot_download, hf_hub_download

# Generic HF junk excluded for every repo.
GENERIC = ['*.msgpack', '*.h5', 'flax_model*', '*.onnx']

# (repo, extra_ignore_patterns) — weight dirs the lab never loads are
# excluded from the cache to save disk:
#   * Z-Image + Qwen-Image 2512 transformers ride the GGUF route (pre-warmed
#     below); only transformer/config.json is read from the diffusers repos.
#   * Qwen-Image 2512 in-repo bf16 text_encoder (~16 GB) is never used — the
#     OzzyGT bnb-4bit mirror below replaces it (the ONLY encoder the loader
#     reads). Its VAE/tokenizer/scheduler shards stay.
#   * lite-infer ERNIE: pe (Ministral3-3B, ~7.2 GB) IS downloaded — the
#     loader drops it right after from_pretrained, but from_pretrained would
#     fetch it on first load anyway, so pre-downloading keeps loads offline.
downloads = [
    # Kept engines — SANA: the two 1.6B variants share the Gemma-2-2B encoder
    # shards, so the HF cache dedups the second download down to ~4.5 GB net.
    ('Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers', []),
    ('Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers', []),
    ('Boogu/Boogu-Image-0.1-Turbo-fp8', []),  # mllm + bf16 DiT + FLUX.1 VAE
    # --- 2026-09-07 T2I swap additions (zimage / qwenimage / ernie) ---
    ('Tongyi-MAI/Z-Image-Turbo',
     ['transformer/*.safetensors', 'transformer/*.bin']),
    ('Qwen/Qwen-Image-2512',
     ['transformer/*.safetensors', 'transformer/*.bin',
      'text_encoder/*.safetensors', 'text_encoder/*.bin']),
    ('OzzyGT/Qwen-Image-2512-bnb-4bit-text-encoder', []),
    ('lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder', []),
]

token = '$HFToken'
for repo, extra in downloads:
    print(f'Downloading {repo} ...', flush=True)
    try:
        snapshot_download(repo_id=repo, token=token,
                          ignore_patterns=GENERIC + extra)
        print(f'  OK: {repo}', flush=True)
    except Exception as e:
        print(f'  FAILED: {repo}: {e}', file=sys.stderr, flush=True)

# GGUF single-file pre-warm — drops the default quant + transformer configs
# at the exact paths the lab checks (GGUF_ROOT/<key>/...), so the first load
# of each engine runs offline. Mirrors image_lab_engines._ensure_gguf and
# _component_config_dir (GGUF_ROOT = /opt/arthur-img-models/gguf).
GGUF_ROOT = '/opt/arthur-img-models/gguf'
jobs = [
    # (repo, filename, subfolder, destination)
    # zimage default tier is Q4_K_M; qwenimage's ONLY live tier is Q4_K_S —
    # Q4_K_M (12.34 GB) OOMs this card (loads to a ~14.2 GiB process floor on a
    # 15.48 GiB card that already hosts ~1.2 GiB of TTS-container contexts;
    # verified 3× 2026-09-08, see _QWENIMAGE_GGUF). Do NOT re-add Q4_K_M here.
    ('jayn7/Z-Image-Turbo-GGUF', 'z_image_turbo-Q4_K_M.gguf', None,
     GGUF_ROOT + '/zimage/z_image_turbo-Q4_K_M.gguf'),
    ('unsloth/Qwen-Image-2512-GGUF', 'qwen-image-2512-Q4_K_S.gguf', None,
     GGUF_ROOT + '/qwenimage/qwen-image-2512-Q4_K_S.gguf'),
    # qwenimage-edit: Q4_K_S is the ONLY tier that fits (two-channel edit
    # conditioning peaks 15.40 GiB driver at 1024²/20 st/CFG 4.0 — measured
    # 2026-09-08 probe; see _QWENIMAGE_EDIT_GGUF).
    ('unsloth/Qwen-Image-Edit-2511-GGUF', 'qwen-image-edit-2511-Q4_K_S.gguf', None,
     GGUF_ROOT + '/qwenimage-edit/qwen-image-edit-2511-Q4_K_S.gguf'),
    ('Qwen/Qwen-Image-Edit-2511', 'config.json', 'transformer',
     GGUF_ROOT + '/qwenimage-edit/transformer_cfg/config.json'),
    ('Tongyi-MAI/Z-Image-Turbo', 'config.json', 'transformer',
     GGUF_ROOT + '/zimage/transformer_cfg/config.json'),
    ('Qwen/Qwen-Image-2512', 'config.json', 'transformer',
     GGUF_ROOT + '/qwenimage/transformer_cfg/config.json'),
]
for repo, fname, subfolder, dest in jobs:
    label = (subfolder + '/' if subfolder else '') + fname
    print(f'Pre-warming {repo}:{label} -> {dest} ...', flush=True)
    try:
        p = hf_hub_download(repo_id=repo, filename=fname, subfolder=subfolder,
                            token=token)
        if os.path.realpath(p) != os.path.realpath(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(p, dest)
        print(f'  OK: {dest}', flush=True)
    except Exception as e:
        print(f'  FAILED: {dest}: {e}', file=sys.stderr, flush=True)
"@
    Set-Content -Path $localScript -Value $dlScript -Encoding UTF8
    Write-Host "  » SCP download script → /tmp/imglab_download.py"
    Invoke-SCP $localScript "/tmp/imglab_download.py"
    Write-Host "  Starting downloads (this may take 30-90 min) …"
    Invoke-SSH "HF_HOME=/opt/arthur-img-models/huggingface $py /tmp/imglab_download.py"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — SCP code to VM
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 5 "SCP code files to VM" {
    $files = @(
        "$repoRoot\image_lab.py",
        "$repoRoot\image_lab_config.py",
        "$repoRoot\image_lab_engines.py",
        "$repoRoot\image_lab_dispatch.py",
        "$repoRoot\image_lab_ui.py",
        "$repoRoot\image_lab_utils.py",
        "$repoRoot\lab_infra.py",
        "$repoRoot\lab_infra_ui.py",
        "$repoRoot\scripts\download\gguf_download.py",
        "$repoRoot\scripts\utils\nvfp4_save.py",
        "$repoRoot\ideogram4_lab_engine.py",
        "$repoRoot\boogu_lab_engine.py",
        "$repoRoot\hidream_comfy_bridge.py"
    )

    foreach ($f in $files) {
        if (-not (Test-Path $f)) { throw "Missing file: $f" }
    }

    Invoke-SCP -LocalFiles $files -RemoteDest "/opt/arthur-img/"

    # Copy magic prompt template alongside the engine module for fallback
    $magicPromptDir = "$repoRoot\ideogram4\src\ideogram4\magic_prompt_system_prompts"
    if (Test-Path "$magicPromptDir\v1.txt") {
        Invoke-SCP -LocalFiles @("$magicPromptDir\v1.txt") -RemoteDest "/opt/arthur-img/"
        Write-Host "  Copied magic prompt v1.txt template to /opt/arthur-img/"
    }

    # Write .env with the runtime environment for the service.
    # If HF_TOKEN is provided, include it; otherwise still create the file so GPU-only mode is enforced.
    # HF_HUB_CACHE (added 2026-09-08): hub >= 1.20 writes $HF_HOME/hub/ and
    # hub >= 1.23 has NO legacy fallback — without this, the direct-layout
    # repos (flux2klein, ideogram4, Qwen3-8B …) read as MISSING and ~40 GB
    # silently re-downloads. Point it at the direct layout explicitly.
    # DIFFUSERS_TRUST_REMOTE_KERNELS (added 2026-09-08, user-approved): the
    # 0.40 nunchaku quantizer refuses to fetch/execute the remote
    # rootonchair/nunchaku-lite-kernels CUDA kernel repo without it (ernie
    # NVFP4 fails with a misleading "OOM" ValueError otherwise).
    $envLines = @(
        "HF_HOME=/opt/arthur-img-models/huggingface",
        "HF_HUB_CACHE=/opt/arthur-img-models/huggingface",
        "DIFFUSERS_TRUST_REMOTE_KERNELS=true",
        "IMGLAB_MODELS_ROOT=/opt/models/image",
        "IMGLAB_OUTPUT_ROOT=/opt/arthur-gen",
        "IMGLAB_PORT=8002",
        "IMGLAB_GPU_ONLY=1",
        "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1"
    )
    if ($HFToken) {
        $envLines = @("HF_TOKEN=$HFToken") + $envLines
    }
    # Add OPENROUTER_API_KEY if found in secrets.env
    $orKeyLine = Get-Content "$repoRoot\secrets.env" -ErrorAction SilentlyContinue | Where-Object { $_ -match "^OPENROUTER_API_KEY=" }
    if ($orKeyLine) {
        $orKey = ($orKeyLine -split "=", 2)[1].Trim()
        if ($orKey) { $envLines += "OPENROUTER_API_KEY=$orKey" }
    }
    # Add DEEPSEEK_API_KEY if found in secrets.env
    $dsKeyLine = Get-Content "$repoRoot\secrets.env" -ErrorAction SilentlyContinue | Where-Object { $_ -match "^DEEPSEEK_API_KEY=" }
    if ($dsKeyLine) {
        $dsKey = ($dsKeyLine -split "=", 2)[1].Trim()
        if ($dsKey) { $envLines += "DEEPSEEK_API_KEY=$dsKey" }
    }
    # Add IDEOGRAM_API_KEY if found in secrets.env (free hosted magic-prompt API)
    $igKeyLine = Get-Content "$repoRoot\secrets.env" -ErrorAction SilentlyContinue | Where-Object { $_ -match "^IDEOGRAM_API_KEY=" }
    if ($igKeyLine) {
        $igKey = ($igKeyLine -split "=", 2)[1].Trim()
        if ($igKey) { $envLines += "IDEOGRAM_API_KEY=$igKey" }
    }
    # Merge, never overwrite: keys the template has a fresh value for are
    # updated in place; any OTHER keys in the existing VM .env (manual
    # additions like EMBED_CACHE_ROOT) are preserved verbatim across
    # redeploys. To remove a key from the VM, edit /opt/arthur-img/.env by
    # hand — the merge won't delete it.
    $managed = @{}
    foreach ($line in $envLines) {
        $kv = $line -split "=", 2
        $managed[$kv[0].Trim()] = $kv[1].Trim()
    }

    $remoteEnvPath = "/opt/arthur-img/.env"
    $existingText = (Invoke-SSH "test -f $remoteEnvPath && cat $remoteEnvPath || true" | Out-String)
    $existingText = ($existingText -replace "`r`n", "`n").TrimEnd("`n")

    $outLines = [System.Collections.Generic.List[string]]::new()
    $seen     = @{}
    if ($existingText.Trim()) {
        foreach ($line in ($existingText -split "`n")) {
            $m = [regex]::Match($line, "^([A-Za-z_][A-Za-z0-9_]*)\s*=")
            if ($m.Success) {
                $key = $m.Groups[1].Value
                if ($managed.ContainsKey($key)) {
                    $outLines.Add("$key=$($managed[$key])")
                    $seen[$key] = $true
                    continue
                }
            }
            $outLines.Add($line)  # comment, blank, or manual key — keep verbatim
        }
    }
    foreach ($key in $managed.Keys) {
        if (-not $seen.ContainsKey($key)) {
            $outLines.Add("$key=$($managed[$key])")
        }
    }

    # Write locally, SCP via /tmp, then move into place — avoids shell quoting
    # issues with token values and writes the file byte-exact (UTF-8, no BOM).
    $localEnv = Join-Path $env:TEMP "arthur-img.env"
    [System.IO.File]::WriteAllText($localEnv, ($outLines -join "`n") + "`n",
                                   (New-Object System.Text.UTF8Encoding($false)))
    Invoke-SCP -LocalFiles @($localEnv) -RemoteDest "/tmp/arthur-img.env"
    Remove-Item $localEnv -Force
    Invoke-SSH "mv /tmp/arthur-img.env $remoteEnvPath && chmod 600 $remoteEnvPath"
    Write-Host "  .env merged: template keys updated, manual additions preserved."

    Invoke-SSH "ls -la /opt/arthur-img/"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 6 — Write systemd service + restart
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 6 "Write systemd service + restart" {
    $serviceUnit = @"
[Unit]
Description=Arthur Image & Video Generation Lab
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur-img
ExecStart=/opt/arthur-img-env/bin/python /opt/arthur-img/image_lab.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-/opt/arthur-img/.env

[Install]
WantedBy=multi-user.target
"@

    # Write service file locally → SCP to /tmp → sudo move into place
    $tmpSvc = [System.IO.Path]::GetTempFileName() + ".service"
    $serviceUnit | Set-Content $tmpSvc -Encoding UTF8
    Invoke-SCP -LocalFiles @($tmpSvc) -RemoteDest "/tmp/arthur-imglab.service"
    Remove-Item $tmpSvc -Force
    Invoke-SSH "sudo mv /tmp/arthur-imglab.service /etc/systemd/system/arthur-imglab.service && sudo chmod 644 /etc/systemd/system/arthur-imglab.service"
    Invoke-SSH "sudo systemctl daemon-reload"
    Invoke-SSH "sudo systemctl enable arthur-imglab.service"
    Invoke-SSH "sudo systemctl restart arthur-imglab.service"
    Start-Sleep -Seconds 3
    Invoke-SSH "sudo systemctl status arthur-imglab.service --no-pager -l | head -20"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 7 — HuggingFace CLI login (cache token on VM)
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 7 "HuggingFace CLI token cache on VM" {
    if (-not $HFToken) {
        Write-Warning "  No HF_TOKEN — skipping. Run manually: ssh $User@$VM huggingface-cli login"
        return
    }
    Invoke-SSH "/opt/arthur-img-env/bin/huggingface-cli login --token $HFToken --add-to-git-credential 2>&1 || true"
    Write-Host "  HF token cached on VM."
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 8 — Health check
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 8 "Health check" {
    Write-Host "  Waiting 5 s for service to start …"
    Start-Sleep -Seconds 5

    $url = "http://${VM}:8002/status"
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 15
        Write-Host "  /status OK — engines:"
        foreach ($e in $resp.engines) {
            $icon = if ($e.available) { "✓" } else { "✗" }
            $color = if ($e.available) { "Green" } else { "DarkYellow" }
            Write-Host ("    $icon {0,-10} {1}" -f $e.key, $e.label) -ForegroundColor $color
        }
        $v = $resp.vram
        if ($v.available) {
            Write-Host ("  VRAM: {0:F1} / {1:F1} GB free on {2}" -f $v.free_gb, $v.total_gb, $v.device_name)
        }
        Write-Host ""
        Write-Host "  Web UI → http://${VM}:8002" -ForegroundColor Cyan
        Write-Host "  API    → http://${VM}:8002/generate/{engine}" -ForegroundColor Cyan
    } catch {
        Write-Warning "  Health check failed: $_"
        Write-Warning "  Check logs: ssh $User@$VM journalctl -u arthur-imglab.service -n 50"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════" -ForegroundColor Green
Write-Host " Arthur Image Lab deployment complete" -ForegroundColor Green
Write-Host "══════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Logs:   ssh $User@$VM journalctl -u arthur-imglab.service -f"
Write-Host "  UI:     http://${VM}:8002"
Write-Host "  API:    POST http://${VM}:8002/generate/{engine}"
Write-Host "          (flux2klein · flux2klein9b · ideogram4 · sana · boogu"
Write-Host "           · zimage · qwenimage · hidream · ernie)"
Write-Host ""
Write-Host "  LICENSES — accept BEFORE running Phase 4 download:" -ForegroundColor Yellow
Write-Host "    Z-Image Apache-2.0 · Qwen-Image 2512 Apache-2.0 · HiDream MIT · ERNIE Apache-2.0"
Write-Host "    (SANA Apache-2.0 + Gemma terms; Boogu Apache-2.0 research-only — ungated)" -ForegroundColor DarkGray
Write-Host ""
