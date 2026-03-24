param(
    [string]$VM   = "192.168.0.153",
    [string]$User = "",
    [string]$Pass = ""
)

if (-not $User) { $User = Read-Host "SSH username" }
if (-not $Pass) { $Pass = Read-Host "SSH password" -AsSecureString | ForEach-Object { [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($_)) } }

$root = Split-Path $PSScriptRoot -Parent

Write-Host ""
Write-Host "=== Deploying Arthur Server to $User@$VM ===" -ForegroundColor Cyan

# Check plink/pscp (PuTTY tools) or openssh
$usePlink = $null -ne (Get-Command plink -ErrorAction SilentlyContinue)
$useSsh   = $null -ne (Get-Command ssh   -ErrorAction SilentlyContinue)

if (-not $usePlink -and -not $useSsh) {
    Write-Error "Neither ssh nor plink found. Install OpenSSH (built into Windows 10+) or PuTTY."
    exit 1
}

function Invoke-Remote([string]$cmd) {
    if ($useSsh) {
        # openssh — password auth via sshpass (Linux) or just prompt
        Write-Host "  >> $cmd" -ForegroundColor DarkGray
        $env:SSHPASS = $Pass
        ssh -o StrictHostKeyChecking=no "${User}@${VM}" $cmd
    } else {
        Write-Host "  >> $cmd" -ForegroundColor DarkGray
        plink -ssh -pw $Pass "${User}@${VM}" $cmd
    }
}

function Send-File([string]$local, [string]$remote) {
    Write-Host "  COPY $local → $remote" -ForegroundColor DarkGray
    if ($useSsh) {
        $env:SSHPASS = $Pass
        scp -o StrictHostKeyChecking=no $local "${User}@${VM}:${remote}"
    } else {
        pscp -pw $Pass $local "${User}@${VM}:${remote}"
    }
}

# ── 1. Create target dirs ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 1: Creating directories ---" -ForegroundColor Yellow
Invoke-Remote "sudo mkdir -p /opt/arthur && sudo chmod 777 /opt/arthur"

# ── 2. Copy files ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Step 2: Copying files ---" -ForegroundColor Yellow
Send-File "$root\arthur_server\arthur_server.py" "/opt/arthur/arthur_server.py"
Send-File "$root\arthur_server\requirements.txt" "/tmp/requirements.txt"
Send-File "$root\arthur_server\setup_vm.sh"      "/tmp/setup_vm.sh"

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
