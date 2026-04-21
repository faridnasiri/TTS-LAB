# tts_test.ps1 — deploy tts_lab.py, restart service, run synthesis tests
# Usage:
#   .\tts_test.ps1                      # deploy + test all engines
#   .\tts_test.ps1 -Engine xtts         # deploy + test one engine
#   .\tts_test.ps1 -NoDeploy            # skip deploy/restart, just test
#   .\tts_test.ps1 -NoDeploy -Engine bark
param(
    [string]$Engine   = "",
    [switch]$NoDeploy,
    [string]$VM       = "arthur-vm",
    [string]$SrcFile  = "$PSScriptRoot\tts_lab.py",
    [string]$DstFile  = "/opt/arthur/tts_lab.py",
    [string]$TestScript = "/opt/arthur/_tts_test.py"
)

$ErrorActionPreference = "Stop"

# ── 1. Deploy ──────────────────────────────────────────────────────────────
if (-not $NoDeploy) {
    Write-Host "`n=== Deploying tts_lab.py ===" -ForegroundColor Cyan
    scp $SrcFile "${VM}:${DstFile}"
    if ($LASTEXITCODE -ne 0) { Write-Error "scp failed"; exit 1 }

    Write-Host "=== Uploading test script ===" -ForegroundColor Cyan
    scp "$PSScriptRoot\_tts_test.py" "${VM}:${TestScript}"
    if ($LASTEXITCODE -ne 0) { Write-Error "scp test script failed"; exit 1 }

    Write-Host "=== Restarting arthur-lab ===" -ForegroundColor Cyan
    ssh $VM "sudo systemctl restart arthur-lab"
    if ($LASTEXITCODE -ne 0) { Write-Error "systemctl restart failed"; exit 1 }

    Write-Host "=== Waiting for service to be ready ===" -ForegroundColor Cyan
    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep 2
        $status = ssh $VM "curl -sf http://localhost:8001/status > /dev/null 2>&1 && echo ok || echo wait"
        if ($status -eq "ok") { $ready = $true; break }
        Write-Host "  ... waiting ($($i*2+2)s)" -ForegroundColor DarkGray
    }
    if (-not $ready) { Write-Error "Service did not become ready in 40s"; exit 1 }
    Write-Host "  Service is ready" -ForegroundColor Green
} else {
    # Still upload test script in case it's newer
    scp "$PSScriptRoot\_tts_test.py" "${VM}:${TestScript}" 2>$null
}

# ── 2. Run tests ────────────────────────────────────────────────────────────
Write-Host "`n=== Running TTS synthesis tests ===" -ForegroundColor Cyan

$engineArg = if ($Engine) { "--engine $Engine" } else { "" }
# Always unload between tests in full-suite runs to prevent VRAM exhaustion.
# Single-engine tests skip unload so the model stays hot for repeat testing.
$unloadArg = if ($Engine) { "" } else { "--unload" }
$cmd = "/opt/arthur-bench-env/bin/python3 $TestScript $engineArg $unloadArg"

ssh -t $VM $cmd
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "ALL TESTS PASSED" -ForegroundColor Green
} else {
    Write-Host "SOME TESTS FAILED (exit $exitCode)" -ForegroundColor Red
}

exit $exitCode
