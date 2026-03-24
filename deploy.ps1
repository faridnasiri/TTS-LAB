param(
    [string]$VM   = "192.168.0.87",   # Ubuntu VM (not the Hyper-V host at .153)
    [string]$User = "arthur",
    [string]$Key  = "$env:USERPROFILE\.ssh\id_arthur_vm"
)

$root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path $Key)) {
    Write-Error "SSH key not found: $Key"
    exit 1
}

Write-Host ""
Write-Host "=== Deploying Arthur Server to $User@$VM ===" -ForegroundColor Cyan
Write-Host "    Key: $Key" -ForegroundColor DarkGray

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error "ssh not found. Install OpenSSH (built into Windows 10+)."
    exit 1
}

function Invoke-Remote([string]$cmd) {
    Write-Host "  >> $cmd" -ForegroundColor DarkGray
    ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" $cmd
}

function Send-File([string]$local, [string]$remote) {
    Write-Host "  COPY $(Split-Path $local -Leaf) → $remote" -ForegroundColor DarkGray
    scp -i $Key -o StrictHostKeyChecking=no $local "${User}@${VM}:${remote}"
}

# ── 1. Create target dirs ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 1: Creating directories ---" -ForegroundColor Yellow
Invoke-Remote "sudo mkdir -p /opt/arthur && sudo chmod 777 /opt/arthur"

# ── 2. Copy files ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 2: Copying files ---" -ForegroundColor Yellow
Send-File "$root\arthur_server\arthur_server.py"          "/opt/arthur/arthur_server.py"
Send-File "$root\arthur_server\requirements.txt"          "/tmp/requirements.txt"
Send-File "$root\arthur_server\setup_vm.sh"               "/tmp/setup_vm.sh"
Send-File "$root\arthur_server\tts_benchmark.py"          "/opt/arthur/tts_benchmark.py"
Send-File "$root\arthur_server\requirements_benchmark.txt" "/opt/arthur/requirements_benchmark.txt"
Send-File "$root\arthur_server\run_benchmark.sh"          "/opt/arthur/run_benchmark.sh"
Send-File "$root\arthur_server\download_models.sh"        "/opt/arthur/download_models.sh"
Send-File "$root\arthur_server\tts_lab.py"                "/opt/arthur/tts_lab.py"
Send-File "$root\arthur_server\setup_tts_lab.sh"          "/opt/arthur/setup_tts_lab.sh"

# ── 3. Run setup script ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 3: Running setup_vm.sh (this takes a few minutes) ---" -ForegroundColor Yellow
Invoke-Remote "chmod +x /tmp/setup_vm.sh && sudo /tmp/setup_vm.sh"

# ── 4. Verify service ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 4: Checking service status ---" -ForegroundColor Yellow
Invoke-Remote "sudo systemctl status arthur --no-pager -l | head -20"

# ── 5. Test endpoint ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 5: Testing HTTP endpoint (should return 200) ---" -ForegroundColor Yellow
Invoke-Remote "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/incoming-call -X POST"

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. On VM: cloudflared tunnel login" -ForegroundColor White
Write-Host "  2. On VM: cloudflared tunnel create arthur" -ForegroundColor White
Write-Host "  3. On VM: cloudflared tunnel route dns arthur arthur.YOURDOMAIN.com" -ForegroundColor White
Write-Host "  4. On VM: sudo systemctl start cloudflared-arthur" -ForegroundColor White
Write-Host "  5. Telnyx webhook: https://arthur.YOURDOMAIN.com/incoming-call" -ForegroundColor White
Write-Host "  6. Update Secrets.cs: AiBridgeNumber = your Telnyx number" -ForegroundColor White
Write-Host ""
Write-Host "To run TTS benchmark on the VM:" -ForegroundColor Cyan
Write-Host "  ssh arthur@192.168.0.87 'sudo bash /opt/arthur/run_benchmark.sh'" -ForegroundColor White
Write-Host "  # Then copy WAVs to Windows to listen:" -ForegroundColor DarkGray
Write-Host "  scp arthur@192.168.0.87:/tmp/tts_bench/*.wav ." -ForegroundColor White
Write-Host ""
Write-Host "To launch the TTS Lab web UI:" -ForegroundColor Cyan
Write-Host "  ssh arthur@192.168.0.87 'sudo bash /opt/arthur/setup_tts_lab.sh'" -ForegroundColor White
Write-Host "  # Then open in browser:" -ForegroundColor DarkGray
Write-Host "  http://192.168.0.87:8001" -ForegroundColor Green
