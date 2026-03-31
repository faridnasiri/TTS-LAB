#!/bin/bash
# switch_gpu_vm.sh
# Proxmox equivalent of Switch-GPUtoVM.ps1
# Moves GPU between VMs via VFIO passthrough on Proxmox VE host.
#
# Run on PROXMOX HOST as root.
# Usage:
#   ./switch_gpu_vm.sh status
#   ./switch_gpu_vm.sh arthur        # assign GPU to arthur Ubuntu VM
#   ./switch_gpu_vm.sh windows       # assign GPU to Windows VM
#   ./switch_gpu_vm.sh none          # remove GPU from all VMs

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
VM_ARTHUR_ID=100      # Proxmox VM ID for arthur-ubuntu  (check in web UI)
VM_WINDOWS_ID=101     # Proxmox VM ID for windows VM     (set to 0 if no Windows VM)
GPU_PCI="01:00.0"     # GPU PCIe address — run: lspci | grep -i nvidia
                      # Format: BB:DD.F  e.g. 01:00.0
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_NAME=$(basename "$0")

usage() {
    echo "Usage: $SCRIPT_NAME {arthur|windows|none|status}"
    echo "  arthur   → assign GPU to arthur Ubuntu VM (ID $VM_ARTHUR_ID)"
    echo "  windows  → assign GPU to Windows VM (ID $VM_WINDOWS_ID)"
    echo "  none     → remove GPU from all VMs"
    echo "  status   → show current GPU assignment"
    exit 1
}

[ $# -lt 1 ] && usage
TARGET="$1"

# ── Helpers ──────────────────────────────────────────────────────────────────
get_vm_state() {
    local vmid=$1
    qm status "$vmid" 2>/dev/null | awk '{print $2}' || echo "unknown"
}

vm_has_gpu() {
    local vmid=$1
    qm config "$vmid" 2>/dev/null | grep -q "^hostpci" && echo "yes" || echo "no"
}

stop_vm_if_running() {
    local vmid=$1
    local state
    state=$(get_vm_state "$vmid")
    if [ "$state" = "running" ]; then
        echo "  Stopping VM $vmid..."
        qm stop "$vmid"
        # Wait for VM to stop (max 30s)
        for i in $(seq 1 30); do
            sleep 1
            [ "$(get_vm_state "$vmid")" = "stopped" ] && break
        done
        echo "  VM $vmid stopped."
    fi
}

remove_gpu_from_vm() {
    local vmid=$1
    # Remove all hostpci entries from VM config
    local has
    has=$(vm_has_gpu "$vmid")
    if [ "$has" = "yes" ]; then
        stop_vm_if_running "$vmid"
        # Find and remove hostpci lines
        for key in $(qm config "$vmid" 2>/dev/null | grep "^hostpci" | cut -d: -f1); do
            echo "  Removing $key from VM $vmid..."
            qm set "$vmid" --delete "$key"
        done
    fi
}

add_gpu_to_vm() {
    local vmid=$1
    echo "  Adding GPU ($GPU_PCI) to VM $vmid..."
    # pcie=1: use PCIe passthrough (not PCI)
    # x-vga=1: pass as primary GPU (needed if VM has no virtual display)
    # rombar=1: expose ROM BAR (needed for most GPUs)
    # multifunction=on: pass all functions of the GPU (GPU + audio)
    qm set "$vmid" \
        --hostpci0 "$GPU_PCI,pcie=1,x-vga=0,rombar=1,multifunction=on"
    echo "  GPU assigned to VM $vmid."
}

# ── Status ────────────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo "=== GPU Assignment Status ==="
    echo "  GPU PCIe: $GPU_PCI"
    # Check which driver currently owns it on host
    DRIVER=$(lspci -nnk -s "$GPU_PCI" 2>/dev/null | grep "Kernel driver" | awk '{print $NF}' || echo "unknown")
    echo "  Host driver: $DRIVER"
    [ "$DRIVER" = "vfio-pci" ] && echo "  → GPU is bound to VFIO (available for passthrough)" \
                                || echo "  → GPU is claimed by host driver (passthrough blocked)"
    echo ""
    for vmid in $VM_ARTHUR_ID $VM_WINDOWS_ID; do
        [ "$vmid" -eq 0 ] 2>/dev/null && continue
        local_name=""
        [ "$vmid" -eq "$VM_ARTHUR_ID" ]  && local_name="arthur-ubuntu"
        [ "$vmid" -eq "$VM_WINDOWS_ID" ] && local_name="windows-vm"
        state=$(get_vm_state "$vmid")
        has=$(vm_has_gpu "$vmid")
        gpu_mark=""
        [ "$has" = "yes" ] && gpu_mark=" ← GPU"
        echo "  VM $vmid ($local_name)  [$state]$gpu_mark"
    done
    echo ""
}

# ── Main switch ───────────────────────────────────────────────────────────────
case "$TARGET" in

  status)
    show_status
    ;;

  none)
    echo ""
    echo "=== Removing GPU from all VMs ==="
    for vmid in $VM_ARTHUR_ID $VM_WINDOWS_ID; do
        [ "$vmid" -eq 0 ] 2>/dev/null && continue
        remove_gpu_from_vm "$vmid"
    done
    echo "  GPU released (bound to vfio-pci on host)."
    echo ""
    ;;

  arthur)
    echo ""
    echo "=== Assigning GPU to arthur-ubuntu (VM $VM_ARTHUR_ID) ==="
    # Remove from windows VM if it has it
    [ "$VM_WINDOWS_ID" -ne 0 ] 2>/dev/null && remove_gpu_from_vm "$VM_WINDOWS_ID"
    stop_vm_if_running "$VM_ARTHUR_ID"
    add_gpu_to_vm "$VM_ARTHUR_ID"
    echo ""
    echo "  Done. Start the VM and verify:"
    echo "    qm start $VM_ARTHUR_ID"
    echo "    ssh arthur@192.168.0.87 'nvidia-smi'"
    echo "    ssh arthur@192.168.0.87 'sudo systemctl restart arthur-lab'"
    echo ""
    ;;

  windows)
    if [ "$VM_WINDOWS_ID" -eq 0 ] 2>/dev/null; then
        echo "ERROR: VM_WINDOWS_ID is not configured in this script."
        exit 1
    fi
    echo ""
    echo "=== Assigning GPU to Windows VM ($VM_WINDOWS_ID) ==="
    remove_gpu_from_vm "$VM_ARTHUR_ID"
    stop_vm_if_running "$VM_WINDOWS_ID"
    add_gpu_to_vm "$VM_WINDOWS_ID"
    echo ""
    echo "  Done. Start the VM:"
    echo "    qm start $VM_WINDOWS_ID"
    echo "    # Connect via RDP or Proxmox console"
    echo ""
    ;;

  *)
    usage
    ;;
esac
