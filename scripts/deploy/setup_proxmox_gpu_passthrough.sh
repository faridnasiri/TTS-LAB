#!/bin/bash
# setup_proxmox_gpu_passthrough.sh
# One-time setup: configure GPU passthrough (VFIO) on Proxmox VE host
# for the Xeon D-1528 / X10SDV-6C-TLN4F board.
#
# Run on the PROXMOX HOST (not in a VM), as root.
# After this: use switch_gpu_vm.sh to move GPU between VMs daily.
#
# References:
#   https://pve.proxmox.com/wiki/PCI_Passthrough
#   https://pve.proxmox.com/wiki/PCI(e)_Passthrough

set -euo pipefail

GPU_VENDOR="10de"   # NVIDIA vendor ID (always 10de)
# GPU_DEVICE will be detected automatically below

echo ""
echo "=== Proxmox VE — GPU Passthrough Setup (VFIO) ==="
echo ""

# ── 1. Verify we're on Proxmox ───────────────────────────────────────────────
if ! command -v pveversion &>/dev/null; then
    echo "ERROR: This script must run on a Proxmox VE host."
    exit 1
fi
pveversion
echo ""

# ── 2. Detect GPU PCIe address and device ID ─────────────────────────────────
echo "Step 1: Detecting NVIDIA GPU..."
GPU_PCI=$(lspci | grep -i nvidia | grep -v Audio | head -1 | awk '{print $1}')
if [ -z "$GPU_PCI" ]; then
    echo "ERROR: No NVIDIA GPU found via lspci. Check PCIe connection."
    exit 1
fi
GPU_DEVICE_ID=$(lspci -n -s "$GPU_PCI" | awk '{print $3}' | cut -d: -f2)
echo "  GPU PCIe address:  $GPU_PCI"
echo "  GPU device ID:     $GPU_VENDOR:$GPU_DEVICE_ID"

# Also find GPU audio device (same card, usually .1)
GPU_AUDIO_PCI=$(lspci | grep -i "nvidia" | grep -i audio | grep "${GPU_PCI%.*}" | awk '{print $1}' | head -1)
echo "  GPU audio PCIe:    ${GPU_AUDIO_PCI:-not found}"

# ── 3. Enable IOMMU in GRUB ──────────────────────────────────────────────────
echo ""
echo "Step 2: Enabling Intel IOMMU in GRUB..."
GRUB_FILE="/etc/default/grub"
cp "$GRUB_FILE" "${GRUB_FILE}.bak.$(date +%s)"

CURRENT_CMDLINE=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE" | cut -d'"' -f2)
echo "  Current cmdline: $CURRENT_CMDLINE"

# Add IOMMU flags if not already present
NEW_CMDLINE="$CURRENT_CMDLINE"
[[ "$NEW_CMDLINE" != *"intel_iommu=on"* ]] && NEW_CMDLINE="$NEW_CMDLINE intel_iommu=on"
[[ "$NEW_CMDLINE" != *"iommu=pt"* ]]       && NEW_CMDLINE="$NEW_CMDLINE iommu=pt"

# ACS override for older boards (Broadwell-DE may have large IOMMU groups)
# Uncomment if GPU and other devices share an IOMMU group:
# [[ "$NEW_CMDLINE" != *"pcie_acs_override"* ]] && NEW_CMDLINE="$NEW_CMDLINE pcie_acs_override=downstream,multifunction"

NEW_CMDLINE=$(echo "$NEW_CMDLINE" | xargs)  # trim whitespace
sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"$NEW_CMDLINE\"|" "$GRUB_FILE"
echo "  New cmdline:     $NEW_CMDLINE"
update-grub
echo "  GRUB updated."

# ── 4. Load VFIO kernel modules ──────────────────────────────────────────────
echo ""
echo "Step 3: Configuring VFIO kernel modules..."
MODULES_FILE="/etc/modules"
for mod in vfio vfio_iommu_type1 vfio_pci vfio_virqfd; do
    if ! grep -q "^$mod$" "$MODULES_FILE" 2>/dev/null; then
        echo "$mod" >> "$MODULES_FILE"
        echo "  Added: $mod"
    else
        echo "  Already present: $mod"
    fi
done

# ── 5. Bind GPU to VFIO (blacklist NVIDIA on host) ───────────────────────────
echo ""
echo "Step 4: Binding GPU to VFIO driver (prevents host from claiming it)..."
VFIO_CONF="/etc/modprobe.d/vfio.conf"

IDS="$GPU_VENDOR:$GPU_DEVICE_ID"
if [ -n "$GPU_AUDIO_PCI" ]; then
    AUDIO_DEVICE_ID=$(lspci -n -s "$GPU_AUDIO_PCI" | awk '{print $3}' | cut -d: -f2)
    IDS="$IDS,$GPU_VENDOR:$AUDIO_DEVICE_ID"
fi

echo "options vfio-pci ids=$IDS disable_vga=1" > "$VFIO_CONF"
echo "softdep nouveau pre: vfio-pci"           >> "$VFIO_CONF"
echo "softdep nvidia pre: vfio-pci"            >> "$VFIO_CONF"
echo "softdep nvidia* pre: vfio-pci"           >> "$VFIO_CONF"
cat "$VFIO_CONF"

# Blacklist host NVIDIA drivers
BLACKLIST_FILE="/etc/modprobe.d/blacklist-nvidia.conf"
cat > "$BLACKLIST_FILE" << 'EOF'
blacklist nouveau
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidiafb
EOF
echo "  NVIDIA drivers blacklisted on host."

update-initramfs -u -k all
echo "  initramfs updated."

# ── 6. Print IOMMU groups ────────────────────────────────────────────────────
echo ""
echo "Step 5: IOMMU group for your GPU (must be its own group for clean passthrough)..."
for d in /sys/kernel/iommu_groups/*/devices/*; do
    if [[ $(readlink "$d") == *"$GPU_PCI"* ]] 2>/dev/null; then
        GROUP=$(echo "$d" | grep -oP 'iommu_groups/\K[0-9]+')
        echo "  GPU is in IOMMU group $GROUP. Devices in this group:"
        for dev in /sys/kernel/iommu_groups/$GROUP/devices/*; do
            echo "    $(basename $dev): $(lspci -s "$(basename $dev)" 2>/dev/null | cut -d' ' -f2-)"
        done
    fi
done 2>/dev/null || echo "  (IOMMU not active yet — check after reboot)"

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete. REBOOT REQUIRED. ==="
echo ""
echo "After reboot, verify with:"
echo "  dmesg | grep -i iommu          # should show IOMMU enabled"
echo "  lspci -nnk -s $GPU_PCI         # driver should be 'vfio-pci'"
echo ""
echo "Then add GPU to your Ubuntu VM in Proxmox Web UI:"
echo "  VM → Hardware → Add → PCI Device → $GPU_PCI"
echo "  Check: PCIe, All Functions, ROM-Bar, Primary GPU (if no other display)"
echo ""
echo "Or use the CLI script: switch_gpu_vm.sh"
echo ""
echo "GPU PCI address to use in VM config: $GPU_PCI"
echo "Save this: GPU_PCI=$GPU_PCI"
