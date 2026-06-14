#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Arthur TTS Lab — Zero-to-Hero deploy script.
    Handles everything from a fresh Ubuntu VM to a fully running 21-engine lab.

.DESCRIPTION
    Phases (all idempotent — safe to re-run):
      Phase 1  VM bootstrap   — apt packages, data disk, swap, Python venv
      Phase 2  PyTorch        — CPU or CUDA wheel, torchaudio, onnxruntime
      Phase 3  Engine pkgs    — all 21 engine pip installs (best-effort)
      Phase 4  Model download — Piper/Kokoro ONNX, HF snapshots
      Phase 5  Lab files      — SCP all tts_lab_*.py modules + patch scripts
      Phase 6  Site patches   — re-apply transformers/parler compat patches
      Phase 7  Service        — create/update arthur-lab.service, start it
      Phase 8  Verify         — HTTP 200, /status, quick synth test

    Flags let you skip completed phases for fast re-deploys.

.PARAMETER VM
    VM IP (default: 192.168.0.87)
.PARAMETER User
    SSH user on the VM (default: arthur)
.PARAMETER Key
    Path to SSH private key (default: ~/.ssh/id_arthur_vm)
.PARAMETER Phase
    Start from this phase number (1-8). Default: 1 (full run).
.PARAMETER SkipPhases
    Comma-separated list of phase numbers to skip. E.g. "1,2,4"
.PARAMETER GPU
    If set, installs CUDA PyTorch wheels instead of CPU-only.

.EXAMPLE
    # Fresh VM — everything from zero:
    .\deploy_lab.ps1

    # Existing VM — just redeploy code + patches + restart (phases 5-8 only):
    .\deploy_lab.ps1 -Phase 5

    # Re-apply patches and restart only:
    .\deploy_lab.ps1 -Phase 6

    # Skip model downloads (already done):
    .\deploy_lab.ps1 -SkipPhases "4"

    # GPU VM:
    .\deploy_lab.ps1 -GPU
#>
param(
    [string]$VM          = "192.168.0.87",
    [string]$User        = "arthur",
    [string]$Key         = "$env:USERPROFILE\.ssh\id_arthur_vm",
    [int]   $Phase       = 1,
    [string]$SkipPhases  = "",
    [switch]$GPU
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
$SSH_OPTS = "-i `"$Key`" -o StrictHostKeyChecking=no -o ConnectTimeout=20"
$SCP_OPTS = "-i `"$Key`" -o StrictHostKeyChecking=no -q"
$skip = @($SkipPhases -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })

function vm([string]$cmd, [switch]$nocheck) {
    $out = & ssh -i $Key -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
                "${User}@${VM}" $cmd 2>&1
    if (-not $nocheck -and $LASTEXITCODE -ne 0) { throw "SSH cmd failed:`n$cmd`n$out" }
    return $out
}

function scp_to([string]$local, [string]$remote) {
    Write-Host "    SCP  $(Split-Path $local -Leaf)" -ForegroundColor DarkGray
    & scp -i $Key -o StrictHostKeyChecking=no -q $local "${User}@${VM}:${remote}" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $local" }
}

function hdr([int]$n, [string]$txt) {
    Write-Host ""
    Write-Host ("─" * 64) -ForegroundColor DarkCyan
    Write-Host "  Phase $n — $txt" -ForegroundColor Cyan
    Write-Host ("─" * 64) -ForegroundColor DarkCyan
}

function ok  ([string]$m) { Write-Host "  ✅ $m" -ForegroundColor Green }
function warn([string]$m) { Write-Host "  ⚠️  $m" -ForegroundColor Yellow }
function info([string]$m) { Write-Host "  ℹ  $m" -ForegroundColor DarkGray }
function fail([string]$m) { Write-Host "  ❌ $m" -ForegroundColor Red; $script:failures++ }

function should_run([int]$n) {
    return ($n -ge $Phase) -and ($skip -notcontains "$n")
}

$script:failures = 0
$labDir = $PSScriptRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║      Arthur TTS Lab — Zero-to-Hero Deploy                    ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "  VM      : ${User}@${VM}" -ForegroundColor White
Write-Host "  Start   : Phase $Phase  |  Skip: $(if($SkipPhases){"$SkipPhases"}else{"none"})" -ForegroundColor White
Write-Host "  PyTorch : $(if($GPU){'CUDA'}else{'CPU-only'})" -ForegroundColor White
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
Write-Host ""

# pre-flight: SSH check
$pong = vm "echo PONG" -nocheck
if ($pong -notmatch "PONG") { Write-Error "Cannot reach ${User}@${VM} via SSH"; exit 1 }
ok "SSH reachable"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — VM Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 1) {
    hdr 1 "VM Bootstrap (apt, swap, data disk, venv)"

    # write bootstrap script to VM and run it
    $bootstrap = @'
#!/usr/bin/env bash
set -uo pipefail
ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }

# ── swap ──────────────────────────────────────────────────────────────────────
SWAP_MB=$(free -m | awk '/^Swap:/{print $2}')
if [ "${SWAP_MB:-0}" -lt 8000 ]; then
    warn "Swap ${SWAP_MB} MB — adding 8 GB..."
    if [ ! -f /swapfile ]; then
        fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile
    fi
    swapon /swapfile 2>/dev/null || true
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    ok "Swap: $(free -m | awk '/^Swap:/{print $2}') MB"
else
    ok "Swap OK (${SWAP_MB} MB)"
fi

# ── data disk → /opt/models ───────────────────────────────────────────────────
if ! mountpoint -q /opt/models; then
    DEV=$(lsblk -rno NAME,SIZE | awk '$2=="650G"||$2=="500G"||$2=="200G"{print "/dev/"$1}' | head -1)
    if [ -n "$DEV" ] && [ -b "$DEV" ]; then
        mkdir -p /opt/models
        blkid "$DEV" | grep -q ext4 || mkfs.ext4 -L models "$DEV"
        mount "$DEV" /opt/models
        UUID=$(blkid -s UUID -o value "$DEV")
        grep -q '/opt/models' /etc/fstab || \
          echo "UUID=${UUID} /opt/models ext4 defaults,nofail 0 2" >> /etc/fstab
        ok "Data disk mounted at /opt/models"
    else
        warn "No secondary disk found — using OS disk"
        mkdir -p /opt/models
    fi
else
    ok "/opt/models already mounted"
fi

mkdir -p /opt/models/{tts,huggingface,cache} /opt/arthur
chown -R arthur:arthur /opt/arthur 2>/dev/null || true

# ── apt ───────────────────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential libsndfile1 libsndfile1-dev \
    ffmpeg espeak-ng sox git wget curl > /dev/null 2>&1
ok "System packages ready"

# ── python venv ───────────────────────────────────────────────────────────────
LAB_ENV="/opt/arthur-bench-env"
if [ ! -d "$LAB_ENV" ]; then
    python3.11 -m venv "$LAB_ENV"
    ok "venv created at $LAB_ENV"
else
    ok "venv already exists"
fi
source "$LAB_ENV/bin/activate"
pip install --quiet --upgrade pip setuptools wheel
ok "pip/setuptools/wheel up to date"
'@
    $bootstrap | ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "sudo bash"
    ok "Phase 1 complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — PyTorch
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 2) {
    hdr 2 "PyTorch + torchaudio + onnxruntime"

    $torchCmd = if ($GPU) {
        "pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    } else {
        "pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    }

    $pytorch = @"
source /opt/arthur-bench-env/bin/activate
if python -c 'import torch; print(torch.__version__)' 2>/dev/null; then
    echo '  ✅ PyTorch already installed'
else
    echo '  Installing PyTorch...'
    $torchCmd
    echo '  ✅ PyTorch installed'
fi
pip install --quiet onnxruntime soundfile numpy psutil packaging huggingface_hub
echo 'PHASE2_DONE'
"@
    $r = vm $pytorch -nocheck
    if ($r -match "PHASE2_DONE") { ok "Phase 2 complete" } else { warn "Phase 2 had issues (check above)" }
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Engine Packages
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 3) {
    hdr 3 "Engine pip installs (21 engines — best-effort, ~10-20 min)"
    info "Output streamed live..."
    Write-Host ""

    # Write install script to VM, run it live
    $installScript = Get-Content (Join-Path $labDir "_remote_install_new_engines.sh") -Raw -ErrorAction SilentlyContinue
    if (-not $installScript) {
        warn "  _remote_install_new_engines.sh not found — generating minimal version"
        $installScript = @'
#!/usr/bin/env bash
source /opt/arthur-bench-env/bin/activate
PIP="pip install --quiet"
ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }

$PIP fastapi "uvicorn[standard]" pydantic httpx         && ok "web framework"
$PIP piper-tts                                           && ok "piper" || warn "piper"
$PIP kokoro-onnx                                         && ok "kokoro" || warn "kokoro"
$PIP git+https://github.com/myshell-ai/MeloTTS.git      && ok "melo" || warn "melo"
$PIP ChatTTS                                             && ok "chattts" || warn "chattts"
$PIP outetts                                             && ok "outetts" || warn "outetts"
$PIP parler-tts==0.2.3 transformers accelerate          && ok "parler" || warn "parler"
$PIP chatterbox-tts                                      && ok "chatterbox" || warn "chatterbox"
$PIP coqui-tts                                           && ok "xtts" || warn "xtts"
$PIP bark                                                && ok "bark" || warn "bark"
$PIP styletts2                                           && ok "styletts2" || warn "styletts2"
$PIP f5-tts                                              && ok "f5tts" || warn "f5tts"
$PIP fish-speech                                         && ok "fishspeech" || warn "fishspeech"
$PIP qwen-tts                                            && ok "qwen3tts" || warn "qwen3tts"
$PIP git+https://github.com/index-tts/index-tts.git     && ok "indextts" || warn "indextts"
$PIP git+https://github.com/Zyphra/Zonos.git            && ok "zonos" || warn "zonos"
$PIP git+https://github.com/myshell-ai/OpenVoice.git    && ok "openvoice" || warn "openvoice"
echo PHASE3_DONE
'@
    }
    $installScript | ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "bash"
    ok "Phase 3 complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Model Downloads
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 4) {
    hdr 4 "Model downloads (Piper ONNX, Kokoro ONNX, HF snapshots)"
    info "Large downloads — skip with -SkipPhases '4' if already done"

    $dlScript = @'
#!/usr/bin/env bash
source /opt/arthur-bench-env/bin/activate
HF_HOME=/opt/models/huggingface
export HF_HOME
ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }

# Piper voice model
mkdir -p /opt/models/tts
PIPER_VOICE="/opt/models/tts/en_US-ryan-high.onnx"
if [ ! -f "$PIPER_VOICE" ]; then
    wget -q -O "$PIPER_VOICE" \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx" \
      && wget -q -O "${PIPER_VOICE}.json" \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json" \
      && ok "Piper en_US-ryan-high" || warn "Piper download failed"
else
    ok "Piper model already present"
fi

# Kokoro ONNX
python - << 'PYEOF' || warn "Kokoro download skipped"
from kokoro_onnx import Kokoro
import os
os.environ['HF_HOME'] = '/opt/models/huggingface'
Kokoro()   # triggers download
print("  ✅ Kokoro ONNX models cached")
PYEOF

# Parler-TTS mini
python - << 'PYEOF' || warn "Parler download skipped"
import os; os.environ['HF_HOME'] = '/opt/models/huggingface'
from huggingface_hub import snapshot_download
snapshot_download("parler-tts/parler-tts-mini-v1", ignore_patterns=["*.md"])
print("  ✅ parler-tts-mini-v1 cached")
PYEOF

# IndexTTS-2
python - << 'PYEOF' || warn "IndexTTS-2 download skipped"
import os; os.environ['HF_HOME'] = '/opt/models/huggingface'
from huggingface_hub import snapshot_download
snapshot_download("IndexTeam/IndexTTS-2", ignore_patterns=["*.md","*.txt"])
print("  ✅ IndexTTS-2 cached")
PYEOF

echo PHASE4_DONE
'@
    $r = vm $dlScript -nocheck
    if ($r -match "PHASE4_DONE") { ok "Phase 4 complete" } else { warn "Phase 4 had issues (check above)" }
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Deploy Lab Files
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 5) {
    hdr 5 "Deploy lab source files"

    vm "sudo mkdir -p /opt/arthur && sudo chown ${User}:${User} /opt/arthur" -nocheck

    $coreFiles = @(
        @{ L = "tts_lab.py";                R = "/opt/arthur/tts_lab.py"                },
        @{ L = "tts_lab_shims.py";          R = "/opt/arthur/tts_lab_shims.py"          },
        @{ L = "tts_lab_config.py";         R = "/opt/arthur/tts_lab_config.py"         },
        @{ L = "tts_lab_utils.py";          R = "/opt/arthur/tts_lab_utils.py"          },
        @{ L = "tts_lab_engines.py";        R = "/opt/arthur/tts_lab_engines.py"        },
        @{ L = "tts_lab_dispatch.py";       R = "/opt/arthur/tts_lab_dispatch.py"       },
        @{ L = "tts_lab_ui.py";             R = "/opt/arthur/tts_lab_ui.py"             },
        @{ L = "patches\patch_parler_tts.py";       R = "/opt/arthur/patch_parler_tts.py"       },
        @{ L = "patches\patch_transformers_stubs.py"; R = "/opt/arthur/patch_transformers_stubs.py" },
        @{ L = "patches\fix_transformers_shims.py"; R = "/opt/arthur/fix_transformers_shims.py" }
    )

    foreach ($f in $coreFiles) {
        $full = Join-Path $labDir $f.L
        if (Test-Path $full) { scp_to $full $f.R }
        else                 { warn "  Missing locally: $($f.L)" }
    }

    # syntax check all 7 modules
    $chk = vm "source /opt/arthur-bench-env/bin/activate && python3 -c `"import ast; files=['tts_lab','tts_lab_shims','tts_lab_config','tts_lab_utils','tts_lab_engines','tts_lab_dispatch','tts_lab_ui']; [ast.parse(open('/opt/arthur/'+f+'.py').read()) for f in files]; print('SYNTAX_OK')`"" -nocheck
    if ($chk -match "SYNTAX_OK") { ok "All 7 modules syntax clean" }
    else                         { fail "Syntax error in one of the modules:`n$chk" }

    ok "Phase 5 complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Site-packages Compatibility Patches
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 6) {
    hdr 6 "Re-apply site-packages compatibility patches"
    info "Idempotent — safe to re-run after any pip upgrade"

    # Use individual vm() calls to avoid CRLF in heredoc strings
    $r1 = vm "source /opt/arthur-bench-env/bin/activate && python3 /opt/arthur/patch_transformers_stubs.py" -nocheck
    $r2 = vm "source /opt/arthur-bench-env/bin/activate && python3 /opt/arthur/fix_transformers_shims.py" -nocheck
    $r3 = vm "source /opt/arthur-bench-env/bin/activate && python3 /opt/arthur/patch_parler_tts.py" -nocheck
    $r  = "$r1`n$r2`n$r3`nPHASE6_DONE"
    $r  = vm "source /opt/arthur-bench-env/bin/activate && python3 /opt/arthur/patch_transformers_stubs.py && python3 /opt/arthur/fix_transformers_shims.py && python3 /opt/arthur/patch_parler_tts.py && echo PHASE6_DONE" -nocheck
    $r | Where-Object { $_ -notmatch "^$" } | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    if ($r -match "PHASE6_DONE") { ok "Phase 6 complete" }
    else                         { warn "Phase 6 had issues" }
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Systemd Service
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 7) {
    hdr 7 "Create/update arthur-lab.service and start it"

    # Write service file
    $svcFile = @'
[Unit]
Description=Arthur TTS Lab (21-engine edition)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/arthur
Environment=COQUI_TOS_AGREED=1
Environment=HF_HOME=/opt/models/huggingface
Environment=XDG_CACHE_HOME=/opt/models/cache
Environment=SUNO_USE_SMALL_MODELS=False
Environment=CUDA_VISIBLE_DEVICES=
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=/opt/arthur-bench-env/bin/uvicorn tts_lab:app --host 0.0.0.0 --port 8001 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
'@
    $svcFile | ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "sudo tee /etc/systemd/system/arthur-lab.service > /dev/null"

    # inject HF_TOKEN if one exists on the VM
    $injectToken = @'
HF_TOKEN=$(cat /home/arthur/.cache/huggingface/token /root/.cache/huggingface/token 2>/dev/null | head -1)
if [ -n "$HF_TOKEN" ]; then
  sudo grep -q "^Environment=HF_TOKEN" /etc/systemd/system/arthur-lab.service \
    || sudo sed -i "/^Environment=HF_HOME/a Environment=HF_TOKEN=$HF_TOKEN" \
       /etc/systemd/system/arthur-lab.service
  echo "  HF_TOKEN injected"
fi
'@
    vm $injectToken -nocheck | Out-Null

    vm "sudo systemctl daemon-reload"
    vm "sudo systemctl enable arthur-lab"
    vm "sudo systemctl restart arthur-lab"
    Start-Sleep 8

    $svc = (vm "sudo systemctl is-active arthur-lab" -nocheck).Trim()
    if ($svc -eq "active") { ok "arthur-lab.service ACTIVE" }
    else {
        warn "Service status: $svc"
        vm "sudo journalctl -u arthur-lab -n 25 --no-pager" -nocheck | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }

    ok "Phase 7 complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Verify
# ─────────────────────────────────────────────────────────────────────────────
if (should_run 8) {
    hdr 8 "Verify — HTTP, /status, quick synthesis"

    # Wait for HTTP 200
    $ready = $false
    for ($i = 1; $i -le 20; $i++) {
        $code = (vm "curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/" -nocheck).Trim()
        if ($code -eq "200") { ok "UI HTTP 200 (attempt $i)"; $ready = $true; break }
        Write-Host "  ... attempt $i — HTTP $code — waiting 3 s..." -ForegroundColor DarkGray
        Start-Sleep 3
    }
    if (-not $ready) { fail "UI did not respond HTTP 200 after 60 s" }

    # /status
    $raw = vm "curl -sf http://localhost:8001/status" -nocheck
    try {
        $data   = $raw | ConvertFrom-Json
        $models = $data.models.PSObject.Properties
        $total  = ($models | Measure-Object).Count
        $avail  = ($models | Where-Object { $_.Value.available } | Measure-Object).Count
        Write-Host ""
        Write-Host ("  {0,-14} {1,-10} {2,6}  {3}" -f "Engine","Status","RAM MB","Label") -ForegroundColor White
        Write-Host ("  " + "─" * 52)
        foreach ($m in $models | Sort-Object Name) {
            $v   = $m.Value
            $ico = if ($v.available) { "✅ ready  " } else { "🔴 missing" }
            $col = if ($v.available) { "White" } else { "DarkGray" }
            Write-Host ("  {0,-14} {1,-10} {2,6}  {3}" -f $m.Name, $ico, $v.ram_est_mb, $v.label) -ForegroundColor $col
        }
        Write-Host ""
        if ($total -eq 21) { ok "All 21 engines registered" }
        else { warn "Expected 21, got $total" }
        ok "$avail / $total packages available"
    } catch {
        warn "/status parse failed: $raw"
    }

    # Quick synthesis — piper (fastest, always CPU, no dependencies)
    $req  = '{"text":"Arthur lab is operational. All systems nominal.","params":{}}'
    $code = (vm "curl -s -o /tmp/piper_verify.wav -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '$req' http://localhost:8001/synthesize/piper" -nocheck).Trim()
    $sz   = (vm "wc -c < /tmp/piper_verify.wav" -nocheck).Trim()
    if ($code -eq "200" -and [int]$sz -gt 1000) {
        ok "Piper synthesis OK — $sz bytes of WAV"
    } else {
        fail "Piper synthesis failed — HTTP $code, $sz bytes"
    }

    ok "Phase 8 complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("═" * 64) -ForegroundColor DarkCyan
if ($script:failures -eq 0) {
    Write-Host "  🎉  DEPLOY COMPLETE — Arthur TTS Lab is live!" -ForegroundColor Green
    Write-Host "  🌐  http://${VM}:8001" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Quick re-deploy (code + patches only):" -ForegroundColor DarkGray
    Write-Host "    .\deploy_lab.ps1 -Phase 5" -ForegroundColor DarkGray
    Write-Host "  Re-apply patches + restart only:" -ForegroundColor DarkGray
    Write-Host "    .\deploy_lab.ps1 -Phase 6" -ForegroundColor DarkGray
} else {
    Write-Host "  ❌  DEPLOY FINISHED WITH $($script:failures) FAILURE(S)" -ForegroundColor Red
    Write-Host "  Check output above — re-run specific phase with -Phase N" -ForegroundColor Yellow
}
Write-Host ("═" * 64) -ForegroundColor DarkCyan
Write-Host ""

exit $script:failures
