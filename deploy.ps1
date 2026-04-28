###############################################################################
# deploy.ps1 — TTS-LAB repo
#
# Deploys the 21-engine TTS Lab (tts_lab*.py) to the arthur VM (port 8001).
# arthur_server.py (port 8000) is owned by C:\repos\Spamblocker — deploy it
# with: C:\repos\Spamblocker\tools\arthur_server\deploy.ps1
#
# Usage:
#   .\deploy.ps1              # copy files only
#   .\deploy.ps1 -Restart     # copy files + restart arthur-lab.service
#   .\deploy.ps1 -ServiceFile # also sync _arthur-lab.service → systemd
###############################################################################
param(
    [string]$VM          = "192.168.0.87",
    [string]$User        = "arthur",
    [string]$Key         = "$env:USERPROFILE\.ssh\id_arthur_vm",
    [switch]$Restart,
    [switch]$ServiceFile
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

if (-not (Test-Path $Key))  { Write-Error "SSH key not found: $Key"; exit 1 }
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) { Write-Error "ssh not in PATH"; exit 1 }

function Invoke-Remote([string]$cmd) {
    Write-Host "  >> $cmd" -ForegroundColor DarkGray
    ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" $cmd
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $cmd" }
}

function Send-File([string]$local, [string]$remote) {
    Write-Host "  COPY $(Split-Path $local -Leaf) → $remote" -ForegroundColor DarkGray
    scp -i $Key -o StrictHostKeyChecking=no $local "${User}@${VM}:${remote}"
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $local" }
}

# Files this repo owns on the VM (arthur_server.py is NOT in this list)
$labFiles = @(
    "tts_lab.py",
    "tts_lab_config.py",
    "tts_lab_dispatch.py",
    "tts_lab_engines.py",
    "tts_lab_shims.py",
    "tts_lab_ui.py",
    "tts_lab_utils.py",
    "bench_all.py",
    "bench_warm.py",
    "patch_parler_tts.py",
    "patch_torchaudio.py",
    "patch_torchaudio_init.py",
    "patch_transformers_stubs.py",
    "fix_transformers_shims.py"
)

Write-Host ""
Write-Host "=== Deploying TTS-LAB → $User@$VM ===" -ForegroundColor Cyan

# ── 1. Ensure target dir ──────────────────────────────────────────────────────
Invoke-Remote "sudo mkdir -p /opt/arthur && sudo chown arthur:arthur /opt/arthur"

# ── 2. Copy TTS Lab files ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Copying TTS Lab files ---" -ForegroundColor Yellow
foreach ($f in $labFiles) {
    $local = Join-Path $here $f
    if (Test-Path $local) {
        Send-File $local "/opt/arthur/$f"
    } else {
        Write-Warning "Skipping missing file: $f"
    }
}

# ── 3. Optionally deploy service unit file ────────────────────────────────────
if ($ServiceFile) {
    Write-Host ""
    Write-Host "--- Deploying arthur-lab.service unit file ---" -ForegroundColor Yellow
    Send-File "$here\_arthur-lab.service" "/tmp/arthur-lab.service"
    Invoke-Remote "sudo cp /tmp/arthur-lab.service /etc/systemd/system/arthur-lab.service && sudo systemctl daemon-reload"
    Write-Host "  service unit updated" -ForegroundColor Green
}

# ── 4. Optionally restart the service ─────────────────────────────────────────
if ($Restart) {
    Write-Host ""
    Write-Host "--- Restarting arthur-lab.service ---" -ForegroundColor Yellow
    Invoke-Remote "sudo systemctl restart arthur-lab"
    Start-Sleep -Seconds 4
}

# ── 5. Status + smoke test ────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- arthur-lab.service status ---" -ForegroundColor Yellow
Invoke-Remote "sudo systemctl status arthur-lab --no-pager -l | head -15"

Write-Host ""
Write-Host "--- Smoke test (GET /status → expect HTTP 200) ---" -ForegroundColor Yellow
Invoke-Remote "curl -s -o /dev/null -w 'HTTP %{http_code}' http://localhost:8001/status"

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "  TTS Lab files  → /opt/arthur/tts_lab*.py  (port 8001)" -ForegroundColor White
Write-Host "  arthur_server  → C:\repos\Spamblocker\tools\arthur_server\deploy.ps1" -ForegroundColor DarkGray
Write-Host ""
Write-Host "To launch the TTS Lab web UI:" -ForegroundColor Cyan
Write-Host "  ssh arthur@192.168.0.87 'sudo bash /opt/arthur/setup_tts_lab.sh'" -ForegroundColor White
Write-Host "  # Then open in browser:" -ForegroundColor DarkGray
Write-Host "  http://192.168.0.87:8001" -ForegroundColor Green
