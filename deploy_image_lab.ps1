<#
.SYNOPSIS
    Deploy the Arthur Image & Video Lab to the Ubuntu VM.

.DESCRIPTION
    8-phase idempotent deployment. Run from the Windows dev machine.
    Mirrors the structure of deploy_lab.ps1 (TTS lab).

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

# ── Resolve HF token ──────────────────────────────────────────────────────────
if (-not $HFToken) {
    if ($env:HF_TOKEN) {
        $HFToken = $env:HF_TOKEN
    } elseif (Test-Path "$PSScriptRoot\secrets.env") {
        $line = (Get-Content "$PSScriptRoot\secrets.env") | Where-Object { $_ -match "^HF_TOKEN=" }
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
    # PyTorch — pin same version as TTS lab (2.10.0+cu128) to match driver
    Invoke-SSH "$pip install torch==2.10.0+cu128 torchvision==0.21.0+cu128 --index-url https://download.pytorch.org/whl/cu128 -q"
    Invoke-SSH "/opt/arthur-img-env/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Engine Python packages
# ─────────────────────────────────────────────────────────────────────────────
Run-Phase 3 "Engine Python packages" {
    $pip = "/opt/arthur-img-env/bin/pip"

    # Core inference stack
    Invoke-SSH "$pip install diffusers transformers accelerate safetensors sentencepiece protobuf -q"

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
import os, sys
# Use HF_HOME so from_pretrained() will find cached models automatically
os.environ['HF_HOME'] = '/opt/arthur-img-models/huggingface'
from huggingface_hub import snapshot_download

models = [
    'diffusers/FLUX.2-dev-bnb-4bit',
    'stabilityai/stable-diffusion-3.5-large',
    'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
    'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
]

token = '$HFToken'
for repo in models:
    print(f'Downloading {repo} ...', flush=True)
    try:
        snapshot_download(repo_id=repo, token=token, ignore_patterns=['*.msgpack','*.h5','flax_model*'])
        print(f'  OK: {repo}', flush=True)
    except Exception as e:
        print(f'  FAILED: {repo}: {e}', file=sys.stderr, flush=True)
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
        "$PSScriptRoot\image_lab.py",
        "$PSScriptRoot\image_lab_config.py",
        "$PSScriptRoot\image_lab_engines.py",
        "$PSScriptRoot\image_lab_dispatch.py",
        "$PSScriptRoot\image_lab_ui.py",
        "$PSScriptRoot\image_lab_utils.py",
        "$PSScriptRoot\gguf_download.py",
        "$PSScriptRoot\nvfp4_save.py"
    )

    foreach ($f in $files) {
        if (-not (Test-Path $f)) { throw "Missing file: $f" }
    }

    Invoke-SCP -LocalFiles $files -RemoteDest "/opt/arthur-img/"

    # Write .env with HF_TOKEN if provided (use printf to avoid heredoc issues)
    if ($HFToken) {
        $envLines = @(
            "HF_TOKEN=$HFToken",
            "HF_HOME=/opt/arthur-img-models/huggingface",
            "IMGLAB_MODELS_ROOT=/opt/models/image",
            "IMGLAB_OUTPUT_ROOT=/opt/arthur-gen",
            "IMGLAB_PORT=8002"
        ) -join '\n'
        Invoke-SSH "printf '$envLines\n' > /opt/arthur-img/.env && chmod 600 /opt/arthur-img/.env"
        Write-Host "  .env written with HF_TOKEN."
    } else {
        Write-Host "  No HF_TOKEN — .env not written. Set manually if needed."
    }

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
Write-Host "  API:    POST http://${VM}:8002/generate/flux2"
Write-Host "          POST http://${VM}:8002/generate/sd35"
Write-Host "          POST http://${VM}:8002/generate/wan"
Write-Host ""
Write-Host "  LICENSES — you must accept BEFORE running Phase 4 download:"
Write-Host "    https://huggingface.co/black-forest-labs/FLUX.2-dev" -ForegroundColor Yellow
Write-Host "    https://huggingface.co/stabilityai/stable-diffusion-3.5-large" -ForegroundColor Yellow
Write-Host ""
