#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Arthur TTS Lab (21-engine edition) to the Ubuntu VM.
    Copies all lab files, installs new packages 14-21, restarts the service,
    and verifies all 21 engine tabs are visible in the web UI.

.EXAMPLE
    .\deploy_tts_lab.ps1
    .\deploy_tts_lab.ps1 -SkipInstall       # just redeploy files + restart
    .\deploy_tts_lab.ps1 -VM 192.168.0.100  # different VM IP
#>
param(
    [string]$VM          = "192.168.0.87",
    [string]$User        = "arthur",
    [string]$Key         = "$env:USERPROFILE\.ssh\id_arthur_vm",
    [switch]$SkipInstall                # skip pip installs, just redeploy + restart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── helpers ───────────────────────────────────────────────────────────────────
# Pass $cmd as a single string to remote bash.
# Using & avoids Invoke-Expression so && / | are NOT parsed by PowerShell.
function vm([string]$cmd, [switch]$nocheck) {
    Write-Host "  >> $cmd" -ForegroundColor DarkGray
    $out = & ssh -i $Key -o StrictHostKeyChecking=no -o ConnectTimeout=15 `
                "${User}@${VM}" $cmd 2>&1
    if (-not $nocheck -and $LASTEXITCODE -ne 0) {
        throw "Remote command failed (exit $LASTEXITCODE): $cmd`n$out"
    }
    return $out
}

function scp_to([string]$local, [string]$remote) {
    $leaf = Split-Path $local -Leaf
    Write-Host "  SCP  $leaf" -ForegroundColor DarkGray
    & scp -i $Key -o StrictHostKeyChecking=no -q "$local" "${User}@${VM}:${remote}" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $local -> $remote" }
}

function hdr([string]$txt) {
    Write-Host ""
    Write-Host ("─" * 62) -ForegroundColor DarkCyan
    Write-Host "  $txt" -ForegroundColor Cyan
    Write-Host ("─" * 62) -ForegroundColor DarkCyan
}

# ── pre-flight ────────────────────────────────────────────────────────────────
if (-not (Test-Path $Key)) {
    Write-Error "SSH key not found: $Key"
    exit 1
}

$labDir = $PSScriptRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║       Arthur TTS Lab — Deploy (21-Engine Edition)            ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "  VM     : ${User}@${VM}  (port 8001)" -ForegroundColor White
Write-Host "  Key    : $Key" -ForegroundColor DarkGray
Write-Host "  Source : $labDir" -ForegroundColor DarkGray

# ── 1. connectivity ───────────────────────────────────────────────────────────
hdr "1 — Connectivity"
$ping = vm "echo PONG" -nocheck
if ($ping -notmatch "PONG") { Write-Error "SSH not reachable at ${User}@${VM}"; exit 1 }
Write-Host "  ✅ SSH OK" -ForegroundColor Green

# ── 2. ensure /opt/arthur exists ──────────────────────────────────────────────
hdr "2 — Remote directories"
vm "sudo mkdir -p /opt/arthur"
vm "sudo chown ${User}:${User} /opt/arthur"
Write-Host "  ✅ /opt/arthur ready" -ForegroundColor Green

# ── 3. copy lab files ─────────────────────────────────────────────────────────
hdr "3 — Copy files"

$files = @(
    # ── modular TTS lab (7 files) ─────────────────────────────────────────────
    @{ L = "tts_lab.py";           R = "/opt/arthur/tts_lab.py" },
    @{ L = "tts_lab_shims.py";     R = "/opt/arthur/tts_lab_shims.py" },
    @{ L = "tts_lab_config.py";    R = "/opt/arthur/tts_lab_config.py" },
    @{ L = "tts_lab_utils.py";     R = "/opt/arthur/tts_lab_utils.py" },
    @{ L = "tts_lab_engines.py";   R = "/opt/arthur/tts_lab_engines.py" },
    @{ L = "tts_lab_dispatch.py";  R = "/opt/arthur/tts_lab_dispatch.py" },
    @{ L = "tts_lab_ui.py";        R = "/opt/arthur/tts_lab_ui.py" },
    # ── compatibility patches ──────────────────────────────────────────────────
    @{ L = "patch_transformers_stubs.py"; R = "/opt/arthur/patch_transformers_stubs.py" },
    @{ L = "fix_transformers_shims.py";   R = "/opt/arthur/fix_transformers_shims.py" },
    @{ L = "patch_parler_tts.py";         R = "/opt/arthur/patch_parler_tts.py" },
    # ── benchmarks + support ──────────────────────────────────────────────────
    @{ L = "tts_benchmark.py";                  R = "/opt/arthur/tts_benchmark.py" },
    @{ L = "bench_all.py";                      R = "/opt/arthur/bench_all.py" },
    @{ L = "bench_warm.py";                     R = "/opt/arthur/bench_warm.py" },
    @{ L = "requirements.txt";                  R = "/opt/arthur/requirements.txt" },
    @{ L = "requirements_benchmark.txt";        R = "/opt/arthur/requirements_benchmark.txt" },
    @{ L = "setup_tts_lab.sh";                  R = "/opt/arthur/setup_tts_lab.sh" },
    @{ L = "download_models.sh";                R = "/opt/arthur/download_models.sh" },
    @{ L = "_remote_install_new_engines.sh";    R = "/tmp/_remote_install_new_engines.sh" },
    @{ L = "SESSION_2026-03-26_NEW_ENGINES.md"; R = "/opt/arthur/SESSION_2026-03-26_NEW_ENGINES.md" }
)

foreach ($f in $files) {
    $full = Join-Path $labDir $f.L
    if (Test-Path $full) {
        scp_to $full $f.R
    } else {
        Write-Host "  SKIP $($f.L) (not found locally)" -ForegroundColor DarkYellow
    }
}

vm "chmod +x /opt/arthur/setup_tts_lab.sh"
vm "chmod +x /opt/arthur/download_models.sh"
vm "chmod +x /tmp/_remote_install_new_engines.sh"
# Strip Windows CRLF line endings from shell scripts (created on Windows host)
vm "sed -i 's/\r$//' /tmp/_remote_install_new_engines.sh /opt/arthur/setup_tts_lab.sh /opt/arthur/download_models.sh"
Write-Host "  ✅ All files deployed" -ForegroundColor Green

# ── 4. syntax check ───────────────────────────────────────────────────────────
hdr "4 — Syntax check tts_lab.py"
$chk = vm "/opt/arthur-bench-env/bin/python -c `"import ast, sys; files=['tts_lab.py','tts_lab_shims.py','tts_lab_config.py','tts_lab_utils.py','tts_lab_engines.py','tts_lab_dispatch.py','tts_lab_ui.py']; [ast.parse(open('/opt/arthur/'+f).read()) for f in files]; print('SYNTAX_OK')`"" -nocheck
if ($chk -match "SYNTAX_OK") {
    Write-Host "  ✅ All 7 tts_lab_*.py modules syntax is clean" -ForegroundColor Green
} else {
    Write-Host "  ❌ SYNTAX ERROR in one of the tts_lab modules — aborting" -ForegroundColor Red
    Write-Host ($chk | Out-String)
    exit 1
}

# ── 4.5. re-apply site-packages patches (survive pip upgrades) ────────────────
hdr "4.5 — Re-apply site-packages compatibility patches"
$patches = @"
source /opt/arthur-bench-env/bin/activate
python3 /opt/arthur/patch_transformers_stubs.py
python3 /opt/arthur/fix_transformers_shims.py
python3 /opt/arthur/patch_parler_tts.py
echo PATCHES_DONE
"@
$patchResult = vm $patches -nocheck
if ($patchResult -match "PATCHES_DONE") {
    Write-Host "  ✅ Site-packages patches applied" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Patch step had warnings (non-fatal)" -ForegroundColor Yellow
    Write-Host ($patchResult | Out-String)
}

# ── 5. install new engine packages ────────────────────────────────────────────
if (-not $SkipInstall) {
    hdr "5 — Install new engines 14-21 (pip only, ~5-15 min)"
    Write-Host "  Output streamed live — Ctrl+C to abort and install manually later" -ForegroundColor DarkGray
    Write-Host ""
    # Run directly so output streams live (no -nocheck wrapping)
    & ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "bash /tmp/_remote_install_new_engines.sh"
    Write-Host ""
    Write-Host "  ✅ Install step done" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ⏭  -SkipInstall set — skipping pip installs" -ForegroundColor DarkYellow
}

# ── 6. restart arthur-lab ─────────────────────────────────────────────────────
hdr "6 — Patch service file + restart arthur-lab.service (port 8001)"
# Ensure CUDA is hidden from all ML libs (prevents SEGV on CPU-only VM)
# Also inject HF_TOKEN so gated models (orpheus etc.) can be accessed by root
$svcPatch = @'
sudo grep -q "CUDA_VISIBLE_DEVICES" /etc/systemd/system/arthur-lab.service || sudo sed -i '/^Environment=COQUI_TOS_AGREED/a Environment=CUDA_VISIBLE_DEVICES=' /etc/systemd/system/arthur-lab.service
sudo grep -q "TOKENIZERS_PARALLELISM" /etc/systemd/system/arthur-lab.service || sudo sed -i '/^Environment=CUDA_VISIBLE_DEVICES/a Environment=TOKENIZERS_PARALLELISM=false' /etc/systemd/system/arthur-lab.service
HF_TOKEN=$(cat /home/arthur/.cache/huggingface/token /root/.cache/huggingface/token 2>/dev/null | head -1)
if [ -n "$HF_TOKEN" ]; then
  sudo grep -q "^Environment=HF_TOKEN" /etc/systemd/system/arthur-lab.service \
    || sudo sed -i "/^Environment=HF_HOME/a Environment=HF_TOKEN=$HF_TOKEN" /etc/systemd/system/arthur-lab.service
fi
'@
vm $svcPatch -nocheck
vm "sudo systemctl daemon-reload"
vm "sudo systemctl restart arthur-lab"
Start-Sleep 6

$svc = vm "sudo systemctl is-active arthur-lab" -nocheck
if ($svc.Trim() -eq "active") {
    Write-Host "  ✅ arthur-lab is ACTIVE" -ForegroundColor Green
} else {
    Write-Host "  ⚠  Service status: $($svc.Trim()) — checking logs:" -ForegroundColor Yellow
    vm "sudo journalctl -u arthur-lab -n 20 --no-pager" -nocheck
    Write-Host ""
    Write-Host "  Continuing to poll — service may still be starting..." -ForegroundColor DarkGray
}

# ── 7. wait for HTTP ──────────────────────────────────────────────────────────
hdr "7 — Wait for web UI HTTP 200"
$ready = $false
for ($i = 1; $i -le 24; $i++) {
    $code = (vm "curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/" -nocheck).Trim()
    if ($code -eq "200") {
        Write-Host "  ✅ UI responding — HTTP 200 (attempt $i)" -ForegroundColor Green
        $ready = $true
        break
    }
    Write-Host "  ... attempt $i/24 — HTTP $code — retrying in 3 s..." -ForegroundColor DarkGray
    Start-Sleep 3
}

if (-not $ready) {
    Write-Host "  ❌ UI not responding after 72 s" -ForegroundColor Red
    Write-Host "  Debug: ssh ${User}@${VM} 'sudo journalctl -u arthur-lab -n 40 --no-pager'" -ForegroundColor Yellow
    exit 1
}

# ── 8. verify /status shows 21 engines ───────────────────────────────────────
hdr "8 — Verify /status endpoint"
$raw = vm "curl -sf http://localhost:8001/status" -nocheck

try {
    $data   = $raw | ConvertFrom-Json
    $models = $data.models.PSObject.Properties
    $total  = ($models | Measure-Object).Count
    $avail  = ($models | Where-Object { $_.Value.available -eq $true } | Measure-Object).Count
    $ram    = $data.system
    $pct    = if ($ram.total -gt 0) { [math]::Round($ram.used / $ram.total * 100, 1) } else { 0 }

    Write-Host ""
    Write-Host "  RAM: $($ram.used) / $($ram.total) MB  ($($ram.free) MB free  $pct%)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("  {0,-14} {1,-12} {2,5}  {3}" -f "Key", "Status", "RAM MB", "Label") -ForegroundColor White
    Write-Host ("  " + "─" * 58)

    foreach ($m in $models) {
        $v    = $m.Value
        $icon = if ($v.available) { "✅ ready" } else { "🔴 missing" }
        $col  = if ($v.available) { "Green" } else { "DarkGray" }
        Write-Host ("  {0,-14} {1,-12} {2,5}  {3}" -f $m.Name, $icon, $v.ram_est_mb, $v.label) -ForegroundColor $col
    }

    Write-Host ""
    if ($total -eq 21) {
        Write-Host "  ✅ All 21 engines registered" -ForegroundColor Green
    } else {
        Write-Host "  ⚠  Expected 21, got $total engines" -ForegroundColor Yellow
    }
    Write-Host "  $avail / $total packages installed and available" -ForegroundColor Cyan
    Write-Host "  $(21 - $avail) missing — will show red badge; Synthesise shows install hint" -ForegroundColor DarkGray

} catch {
    Write-Host "  ⚠  Could not parse /status JSON — raw output:" -ForegroundColor Yellow
    Write-Host $raw -ForegroundColor DarkGray
}

# ── done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  🎉  Arthur TTS Lab 21-engine edition is live!               ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Open  : http://${VM}:8001" -ForegroundColor White
Write-Host ""
Write-Host "  After installing a package on the VM — refresh badges:" -ForegroundColor Cyan
Write-Host "    curl -sX POST http://${VM}:8001/refresh | python3 -m json.tool" -ForegroundColor White
Write-Host ""
Write-Host "  Install remaining packages (on VM):" -ForegroundColor Cyan
Write-Host "    source /opt/arthur-bench-env/bin/activate" -ForegroundColor DarkGray
Write-Host "    pip install fish-speech orpheus-speech phonemizer" -ForegroundColor White
Write-Host "    pip install 'git+https://github.com/Zyphra/Zonos'" -ForegroundColor White
Write-Host "    pip install 'git+https://github.com/myshell-ai/OpenVoice'" -ForegroundColor White
Write-Host "    pip install 'git+https://github.com/index-tts/IndexTTS'" -ForegroundColor White
Write-Host ""
Write-Host "  Benchmark new engines (on VM):" -ForegroundColor Cyan
Write-Host "    /opt/arthur-bench-env/bin/python /opt/arthur/tts_benchmark.py --models orpheus,zonos,fishspeech,openvoice" -ForegroundColor White
Write-Host ""