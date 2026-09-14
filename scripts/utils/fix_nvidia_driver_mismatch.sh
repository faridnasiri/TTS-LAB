#!/usr/bin/env bash
# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
# fix_nvidia_driver_mismatch.sh
#
# Repairs the "Driver/library version mismatch" NVML failure on a host whose
# NVIDIA *user-space* libraries were apt-upgraded without a reboot.
#
#   $ nvidia-smi
#   Failed to initialize NVML: Driver/library version mismatch
#   NVML library version: 580.178
#
# WHY THIS HAPPENS
#   An NVIDIA driver is two halves that must match: the kernel module resident
#   in RAM, and the user-space libraries (libcuda.so / libnvidia-ml.so) on disk.
#   `apt upgrade` replaces the on-disk libraries AND rebuilds the DKMS module on
#   disk -- but the module already loaded in RAM keeps running until the machine
#   reboots. The freshly built .ko is therefore ALREADY correct for the running
#   kernel, which means a module reload fixes the mismatch with no reboot.
#
# WHY THE LAB KEEPS WORKING ANYWAY (worth knowing before panicking)
#   Container images bundle their OWN copy of the driver user-space. If that
#   copy matches the loaded module (older userspace + older module), containers
#   keep working while the host is broken. Only NEW host-side CUDA initialisations
#   fail (often as `Error 804: forward compatibility was attempted on non
#   supported HW`). A host-side process that opened its CUDA context BEFORE the
#   upgrade also keeps working -- until it restarts, at which point it will not
#   come back. That latent trap is the real reason to fix this promptly.
#
# THE SECOND, HIDDEN HALF OF THIS FAILURE (the part that breaks containers)
#   Reloading the module is NOT enough. The NVIDIA container toolkit generates
#   /var/run/cdi/nvidia.yaml, which lists ABSOLUTE host paths to the driver
#   libraries; every GPU container start bind-mounts those exact files. A driver
#   upgrade triggers nvidia-cdi-refresh.service to regenerate that spec -- but
#   its guard ran
#       nvidia-smi -L || /usr/lib/wsl/lib/nvidia-smi -L
#   and on Ubuntu the WSL path does not exist. Because nvidia-smi itself was
#   failing (this very mismatch), the shell chain ended on a missing binary and
#   the unit exited 127 "command not found" instead of failing meaningfully, so
#   the spec was never refreshed. It kept naming libraries apt had deleted, and
#   EVERY GPU container start failed with
#       failed to fulfil mount request: open .../libEGL_nvidia.so.<old version>
#   Type=oneshot also ignores Restart=on-failure, so it never retried.
#   This script therefore regenerates the CDI spec as well.
#
# WHAT THIS SCRIPT DOES
#   Stops the GPU consumers (bare-metal image lab + ComfyUI sidecar + the GPU
#   engine container), waits for the module refcounts to fall, unloads and
#   reloads nvidia, verifies the new module matches user-space, then starts
#   everything again. It will NOT proceed to unload if a consumer refuses to
#   stop, and will NOT restart the lab if the reload failed -- so a failure
#   leaves the system in a known state rather than half-broken.
#
# USAGE
#   ssh arthur@192.168.0.87 'bash -s' < fix_nvidia_driver_mismatch.sh
#   ssh arthur@192.168.0.87 'bash -s' < fix_nvidia_driver_mismatch.sh --dry-run
#
#   --dry-run   report the version mismatch and what would be done, change nothing
# ─────────────────────────────────────────────────────────────────────────────

set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# GPU consumers, in the order they are stopped and started back.
BARE_METAL_UNITS="arthur-imglab arthur-comfy"
GPU_CONTAINERS="tts-lab-engine-current"

log()  { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '  !! %s\n' "$*" >&2; }
die()  { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# ── 1. version triage ────────────────────────────────────────────────────────
log "version triage"

KREL=$(uname -r)
KO="/lib/modules/$KREL/updates/dkms/nvidia.ko"

RUN_VER=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
[ -n "$RUN_VER" ] || die "cannot read running module version from /proc/driver/nvidia/version (is nvidia loaded?)"

[ -f "$KO" ] || die "no DKMS module for the RUNNING kernel $KREL at $KO
       A standalone module reload is impossible. Reboot (the module for the
       other installed kernels is built) or rebuild DKMS first."
DISK_VER=$(modinfo -F version "$KO" 2>/dev/null)

LIB_VER=$(readlink -f /lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null)
LIB_VER=$(printf '%s' "$LIB_VER" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+$')

info "running kernel module : $RUN_VER"
info "on-disk module ($KREL) : $DISK_VER"
info "user-space libs       : ${LIB_VER:-unknown}"

if [ "$RUN_VER" = "$DISK_VER" ]; then
    info "-> module and on-disk build already agree; nothing to reload."
    if [ -n "$LIB_VER" ] && [ "$LIB_VER" != "$RUN_VER" ]; then
        warn "BUT user-space ($LIB_VER) still differs from the module ($RUN_VER)."
        warn "A reload will not help: reinstall the driver packages to match."
        exit 2
    fi
    info "-> and user-space agrees. System is healthy."
    exit 0
fi

# A reload installs DISK_VER. That only helps if user-space is already DISK_VER.
if [ -n "$LIB_VER" ] && [ "$LIB_VER" != "$DISK_VER" ]; then
    die "on-disk module ($DISK_VER) and user-space libs ($LIB_VER) disagree.
       Reloading would replace one mismatch with another. Fix the packages
       first (apt install --reinstall nvidia-driver-580-open libnvidia-compute-580
       nvidia-utils-580), then re-run this script."
fi

info "-> mismatch confirmed: reloading the module will install $DISK_VER and match user-space."

if [ "$DRY_RUN" = 1 ]; then
    log "dry run - no changes made"
    info "would stop : $BARE_METAL_UNITS ; docker stop $GPU_CONTAINERS"
    info "would run  : modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia"
    info "would run  : modprobe nvidia nvidia_uvm nvidia_drm"
    info "would start: $BARE_METAL_UNITS ; docker start $GPU_CONTAINERS"
    exit 0
fi

# ── 2. record current state so it can be restored faithfully ─────────────────
log "recording current state"
MODS_LOADED=""
for m in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
    if lsmod | grep -qE "^${m}\b"; then MODS_LOADED="$MODS_LOADED $m"; fi
done
info "modules currently loaded:$MODS_LOADED"

UNITS_RUNNING=""
for u in $BARE_METAL_UNITS; do
    if systemctl is-active --quiet "$u"; then UNITS_RUNNING="$UNITS_RUNNING $u"; fi
done
info "units currently running:$UNITS_RUNNING"

CONTAINERS_RUNNING=""
for c in $GPU_CONTAINERS; do
    if sudo -n docker inspect "$c" >/dev/null 2>&1; then
        case "$(sudo -n docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null)" in
            running) CONTAINERS_RUNNING="$CONTAINERS_RUNNING $c" ;;
        esac
    fi
done
info "gpu containers running:$CONTAINERS_RUNNING"

# Remembers whether anything was up so the restart step never "helpfully" starts
# something that was deliberately stopped before we ran.
restore() {
    log "restarting GPU consumers"
    for u in $UNITS_RUNNING; do
        sudo -n systemctl start "$u" && info "started unit $u" || warn "could not start $u"
    done
    for c in $CONTAINERS_RUNNING; do
        sudo -n docker start "$c" >/dev/null 2>&1 && info "started container $c" || warn "could not start container $c"
    done
}

# ── 3. stop consumers ────────────────────────────────────────────────────────
log "stopping GPU consumers"
for u in $UNITS_RUNNING; do
    sudo -n systemctl stop "$u" && info "stopped unit $u" || warn "could not stop $u"
done
for c in $CONTAINERS_RUNNING; do
    sudo -n docker stop "$c" >/dev/null 2>&1 && info "stopped container $c" || warn "could not stop container $c"
done

log "waiting for device handles to be released"
released=0
for i in $(seq 1 60); do
    if ! sudo -n fuser /dev/nvidia* >/dev/null 2>&1; then
        info "all /dev/nvidia* handles released after ${i}s"
        released=1
        break
    fi
    sleep 1
done

if [ "$released" != 1 ]; then
    warn "processes are STILL holding the GPU:"
    sudo -n fuser -v /dev/nvidia* 2>&1 | sed 's/^/      /'
    warn "module unload would fail - aborting before touching anything."
    warn "Find and stop the remaining holder, then re-run."
    restore
    exit 3
fi

# ── 4. unload ────────────────────────────────────────────────────────────────
log "unloading nvidia modules"
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
    lsmod | grep -qE "^${m}\b" || { info "$m already absent"; continue; }
    if sudo -n modprobe -r "$m" 2>&1; then
        info "unloaded $m"
    else
        warn "could not unload $m"
    fi
done

if lsmod | grep -qE '^nvidia\b'; then
    warn "nvidia is still resident:"
    lsmod | grep -E '^nvidia' | sed 's/^/      /'
    warn "nothing was unloaded successfully - restarting consumers, no harm done."
    restore
    exit 4
fi
info "all nvidia modules unloaded"

# ── 5. load the matching build ───────────────────────────────────────────────
log "loading $DISK_VER"
if ! sudo -n modprobe nvidia; then
    warn "modprobe nvidia FAILED. The GPU is now unavailable to the host AND to"
    warn "containers. Recover with a reboot, or retry: sudo modprobe nvidia"
    die  "reload failed - see above. Not restarting the lab."
fi
info "loaded nvidia"
sudo -n modprobe nvidia_uvm 2>&1 && info "loaded nvidia_uvm"
sudo -n modprobe nvidia_drm 2>&1 && info "loaded nvidia_drm"

# ── 6. verify before bringing anything back ──────────────────────────────────
log "verifying"
NEW_VER=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
info "running kernel module now: $NEW_VER"

if [ "$NEW_VER" != "$DISK_VER" ]; then
    die "reloaded module reports $NEW_VER but user-space on disk is $LIB_VER - mismatch remains."
fi

if nvidia-smi -L >/dev/null 2>&1; then
    info "nvidia-smi OK:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader 2>&1 | sed 's/^/      /'
else
    warn "nvidia-smi still failing:"
    nvidia-smi 2>&1 | head -5 | sed 's/^/      /'
    restore
    exit 5
fi

# ── 6b. regenerate the CDI spec -- this is what actually un-blocks containers ─
# Without this step the host is repaired but every GPU container still refuses
# to start, because the spec still names driver libraries that no longer exist.
log "refreshing CDI spec"
if command -v nvidia-ctk >/dev/null 2>&1; then
    if sudo -n nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml >/dev/null 2>&1; then
        refs=$(sudo -n grep -c "$NEW_VER" /var/run/cdi/nvidia.yaml 2>/dev/null || true)
        info "regenerated /var/run/cdi/nvidia.yaml (${refs:-0} references to $NEW_VER)"
    else
        warn "nvidia-ctk cdi generate FAILED - GPU containers will not start"
    fi
else
    warn "nvidia-ctk not found - CDI spec NOT refreshed; GPU containers will not start"
fi

restore

log "done"
info "host module and user-space now both report $DISK_VER"
