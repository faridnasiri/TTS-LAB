# Switch-GPUtoVM.ps1
# Moves the physical GPU between VMs via DDA (Discrete Device Assignment)
# No NVIDIA license required. Full native CUDA in whichever VM holds the GPU.
#
# Usage:
#   .\Switch-GPUtoVM.ps1 -To arthur        # assign GPU to Ubuntu/arthur VM
#   .\Switch-GPUtoVM.ps1 -To windows       # assign GPU to Windows VM
#   .\Switch-GPUtoVM.ps1 -To none          # release GPU back to host only
#   .\Switch-GPUtoVM.ps1 -Status           # show current assignment

param(
    [ValidateSet("arthur", "windows", "none")]
    [string]$To,
    [switch]$Status,
    [switch]$Force   # skip confirmation prompts
)

# ── Configuration ────────────────────────────────────────────────────────────
$VM_ARTHUR  = "arthur-ubuntu"    # name of your Ubuntu VM in Hyper-V
$VM_WINDOWS = "windows-vm"       # name of your Windows VM in Hyper-V (if any)

# Memory-mapped I/O space — set once, applies to all DDA-capable VMs
# 32GB high MMIO covers RTX 4060 Ti / RTX 5060 Ti with Above 4G Decoding on
$LOW_MMIO  = 1GB
$HIGH_MMIO = 32GB
# ─────────────────────────────────────────────────────────────────────────────

#Requires -RunAsAdministrator

function Get-GPULocationPath {
    $dev = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
           Where-Object { $_.Status -eq 'OK' -and $_.FriendlyName -notlike "*Microsoft*" } |
           Select-Object -First 1
    if (-not $dev) {
        # GPU might already be assigned to a VM (shows as Unknown on host)
        $dev = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
               Where-Object { $_.FriendlyName -notlike "*Microsoft*" } |
               Select-Object -First 1
    }
    if (-not $dev) { throw "No discrete GPU found. Is it already fully assigned to a running VM?" }
    $loc = ($dev | Get-PnpDeviceProperty -KeyName DEVPKEY_Device_LocationPaths -ErrorAction Stop).Data
    if (-not $loc) { throw "Could not read GPU location path. Try running as Administrator." }
    return $loc[0]
}

function Get-CurrentAssignment {
    foreach ($vmName in @($VM_ARTHUR, $VM_WINDOWS)) {
        try {
            $vm = Get-VM -Name $vmName -ErrorAction SilentlyContinue
            if ($vm) {
                $assigned = Get-VMAssignableDevice -VMName $vmName -ErrorAction SilentlyContinue
                if ($assigned) { return $vmName }
            }
        } catch {}
    }
    return "host (unassigned)"
}

function Ensure-VMStopped($vmName) {
    $vm = Get-VM -Name $vmName -ErrorAction Stop
    if ($vm.State -ne 'Off') {
        if ($Force) {
            Write-Host "  Stopping VM '$vmName'..." -ForegroundColor Yellow
            Stop-VM -Name $vmName -Force -ErrorAction Stop
            Start-Sleep -Seconds 3
        } else {
            $yn = Read-Host "  VM '$vmName' is running ($($vm.State)). Stop it now? [y/N]"
            if ($yn -ne 'y') { throw "Aborted — VM must be Off before GPU reassignment." }
            Stop-VM -Name $vmName -Force
            Start-Sleep -Seconds 3
        }
    }
}

function Set-VMMmioIfNeeded($vmName) {
    $vm = Get-VM -Name $vmName -ErrorAction Stop
    $currentHigh = $vm.HighMemoryMappedIoSpace
    $currentLow  = $vm.LowMemoryMappedIoSpace
    if ($currentHigh -ne $HIGH_MMIO -or $currentLow -ne $LOW_MMIO) {
        Write-Host "  Configuring MMIO for '$vmName' (needed once)..." -ForegroundColor Cyan
        Set-VM -VMName $vmName -LowMemoryMappedIoSpace  $LOW_MMIO  -ErrorAction Stop
        Set-VM -VMName $vmName -HighMemoryMappedIoSpace $HIGH_MMIO -ErrorAction Stop
    }
}

function Remove-GPUFromAllVMs($gpuPath) {
    foreach ($vmName in @($VM_ARTHUR, $VM_WINDOWS)) {
        $vm = Get-VM -Name $vmName -ErrorAction SilentlyContinue
        if (-not $vm) { continue }
        $assigned = Get-VMAssignableDevice -VMName $vmName -ErrorAction SilentlyContinue
        if ($assigned) {
            Write-Host "  Removing GPU from '$vmName'..." -ForegroundColor Yellow
            Ensure-VMStopped $vmName
            Remove-VMAssignableDevice -VMName $vmName -LocationPath $gpuPath -Confirm:$false -ErrorAction SilentlyContinue
        }
    }
}

# ── Status ───────────────────────────────────────────────────────────────────
if ($Status -or (-not $To)) {
    $current = Get-CurrentAssignment
    Write-Host ""
    Write-Host "  GPU currently assigned to: $current" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  VMs:"
    foreach ($vmName in @($VM_ARTHUR, $VM_WINDOWS)) {
        $vm = Get-VM -Name $vmName -ErrorAction SilentlyContinue
        if ($vm) {
            $state    = $vm.State
            $hasGpu   = ($null -ne (Get-VMAssignableDevice -VMName $vmName -ErrorAction SilentlyContinue))
            $gpuMark  = if ($hasGpu) { " ← GPU" } else { "" }
            Write-Host "    $vmName  [$state]$gpuMark"
        } else {
            Write-Host "    $vmName  [not found]" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
    if (-not $To) { exit 0 }
}

# ── Switch ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== DDA GPU Switch → '$To' ===" -ForegroundColor Green

try {
    $gpuPath = Get-GPULocationPath
    Write-Host "  GPU path: $gpuPath"
} catch {
    Write-Error $_
    exit 1
}

# Step 1: Remove from any current VM
Remove-GPUFromAllVMs $gpuPath

# Step 2: Return to host (dismount if still dismounted is safe)
try {
    Mount-VMHostAssignableDevice -LocationPath $gpuPath -ErrorAction SilentlyContinue
    Write-Host "  GPU returned to host." -ForegroundColor Gray
} catch {}

if ($To -eq "none") {
    Write-Host "  GPU released to host. Done." -ForegroundColor Green
    exit 0
}

# Step 3: Assign to target VM
$targetVM = if ($To -eq "arthur") { $VM_ARTHUR } else { $VM_WINDOWS }

$vm = Get-VM -Name $targetVM -ErrorAction SilentlyContinue
if (-not $vm) {
    Write-Error "VM '$targetVM' not found in Hyper-V. Check the name in the config at the top of this script."
    exit 1
}

Ensure-VMStopped $targetVM
Set-VMMmioIfNeeded $targetVM

Write-Host "  Dismounting GPU from host..." -ForegroundColor Cyan
Dismount-VMHostAssignableDevice -LocationPath $gpuPath -Force -ErrorAction Stop

Write-Host "  Assigning GPU to '$targetVM'..." -ForegroundColor Cyan
Add-VMAssignableDevice -VMName $targetVM -LocationPath $gpuPath -ErrorAction Stop

Write-Host ""
Write-Host "  Done. GPU is now assigned to: $targetVM" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:"
if ($To -eq "arthur") {
    Write-Host "    Start-VM -Name '$VM_ARTHUR'"
    Write-Host "    ssh arthur@192.168.0.87 'nvidia-smi'"
    Write-Host "    ssh arthur@192.168.0.87 'sudo systemctl restart arthur-lab'"
} else {
    Write-Host "    Start-VM -Name '$VM_WINDOWS'"
    Write-Host "    # Open VM console or RDP — GPU will appear as RTX xxxx in Device Manager"
}
Write-Host ""
