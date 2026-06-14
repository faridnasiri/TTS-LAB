# Setup-DDA.ps1
# One-time setup script: prepares the Hyper-V host + VMs for DDA GPU switching.
# Run ONCE after installing the GPU. Switch-GPUtoVM.ps1 handles daily use.
#
# Requirements:
#   - Hyper-V Server 2022 Free OR Windows Server 2022 with Hyper-V role
#   - Run as Administrator
#   - GPU physically installed (PCIe slot or riser)
#   - BIOS: Above 4G Decoding = Enabled, CSM = Disabled

#Requires -RunAsAdministrator

$VM_ARTHUR  = "arthur-ubuntu"
$VM_WINDOWS = "windows-vm"       # set to $null if no Windows VM

Write-Host ""
Write-Host "=== DDA One-Time Setup ===" -ForegroundColor Green
Write-Host ""

# ── 1. Verify Hyper-V DDA is available ───────────────────────────────────────
Write-Host "Step 1: Checking Hyper-V DDA support..." -ForegroundColor Cyan
try {
    $null = Get-Command Get-VMHostAssignableDevice -ErrorAction Stop
    Write-Host "  Hyper-V DDA cmdlets: OK" -ForegroundColor Green
} catch {
    Write-Error "Hyper-V DDA cmdlets not found. Enable Hyper-V role first."
    exit 1
}

# ── 2. Find the GPU ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2: Detecting GPU..." -ForegroundColor Cyan
$gpuDevices = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
              Where-Object { $_.FriendlyName -notlike "*Microsoft*" -and $_.FriendlyName -notlike "*Basic*" }

if (-not $gpuDevices) {
    Write-Error "No discrete GPU detected. Check PCIe connection and BIOS Above 4G Decoding setting."
    exit 1
}

Write-Host "  Found GPU(s):"
$gpuDevices | ForEach-Object { Write-Host "    $($_.FriendlyName)  [$($_.Status)]" }

$gpu     = $gpuDevices | Select-Object -First 1
$gpuPath = ($gpu | Get-PnpDeviceProperty -KeyName DEVPKEY_Device_LocationPaths).Data[0]

if (-not $gpuPath) {
    Write-Error "Could not read GPU location path. Ensure ACS is enabled in BIOS."
    exit 1
}
Write-Host "  Location path: $gpuPath" -ForegroundColor Green

# ── 3. Verify ACS (Access Control Services) ──────────────────────────────────
Write-Host ""
Write-Host "Step 3: Checking PCIe ACS support..." -ForegroundColor Cyan
$hostAssignable = Get-VMHostAssignableDevice -ErrorAction SilentlyContinue
if ($hostAssignable) {
    Write-Host "  ACS devices visible to Hyper-V: $($hostAssignable.Count)" -ForegroundColor Green
} else {
    Write-Host "  WARNING: No devices currently marked assignable." -ForegroundColor Yellow
    Write-Host "  If GPU assignment fails, enable ACS in BIOS:" -ForegroundColor Yellow
    Write-Host "    Advanced → PCIe Configuration → ACS Control = Enabled" -ForegroundColor Yellow
}

# ── 4. Configure VM MMIO settings ────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4: Configuring VM MMIO for DDA..." -ForegroundColor Cyan

foreach ($vmName in @($VM_ARTHUR, $VM_WINDOWS)) {
    if (-not $vmName) { continue }
    $vm = Get-VM -Name $vmName -ErrorAction SilentlyContinue
    if (-not $vm) {
        Write-Host "  VM '$vmName': not found, skipping" -ForegroundColor DarkGray
        continue
    }
    if ($vm.State -ne 'Off') {
        Write-Host "  VM '$vmName': must be Off to configure MMIO. Skipping — run again after stopping VM." -ForegroundColor Yellow
        continue
    }
    Set-VM -VMName $vmName -LowMemoryMappedIoSpace  1GB  -ErrorAction Stop
    Set-VM -VMName $vmName -HighMemoryMappedIoSpace 32GB -ErrorAction Stop
    Write-Host "  VM '$vmName': MMIO configured (1GB low / 32GB high)" -ForegroundColor Green
}

# ── 5. Verify BIOS checklist ─────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 5: BIOS pre-flight checklist" -ForegroundColor Cyan
Write-Host "  Verify these settings are correct in Supermicro BIOS 2.6:" -ForegroundColor White
Write-Host "    [?] Advanced → PCIe/PCI/PnP → Above 4G Decoding          = Enabled"
Write-Host "    [?] Advanced → PCIe/PCI/PnP → CSM Support                = Disabled"
Write-Host "    [?] Advanced → PCIe/PCI/PnP → ACS Control / ACS Support  = Enabled"
Write-Host "    [?] Advanced → System Agent → VT-d                       = Enabled (disable if CUDA fails)"
Write-Host ""

# ── 6. Summary ───────────────────────────────────────────────────────────────
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Daily use commands:"
Write-Host "  .\Switch-GPUtoVM.ps1 -Status           # who has the GPU?"
Write-Host "  .\Switch-GPUtoVM.ps1 -To arthur         # Ubuntu gets full CUDA"
Write-Host "  .\Switch-GPUtoVM.ps1 -To windows        # Windows VM gets GPU"
Write-Host "  .\Switch-GPUtoVM.ps1 -To none           # release GPU to host"
Write-Host ""
Write-Host "After assigning to arthur VM:"
Write-Host "  Start-VM -Name '$VM_ARTHUR'"
Write-Host "  ssh arthur@192.168.0.87 'nvidia-smi && python3 -c ""import torch;print(torch.cuda.get_device_name(0))""'"
Write-Host ""
