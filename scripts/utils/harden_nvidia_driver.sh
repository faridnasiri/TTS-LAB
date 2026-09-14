#!/usr/bin/env bash
# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
# harden_nvidia_driver.sh — close the holes that broke arthur-server 2026-09-11
#
# THREE independent faults had to line up. Fixing only one leaves the box just
# as breakable as before, which is why this script addresses all of them.
#
#   1. unattended-upgrades installed the NVIDIA driver WITHOUT a reboot.
#      User-space became 580.178.04 while the module in RAM stayed 580.173.02,
#      so nvidia-smi, host CuPy/PyTorch, and every NEW host CUDA process failed
#      (often as `Error 804: forward compatibility was attempted on non
#      supported HW`).
#
#   2. That same upgrade triggered nvidia-cdi-refresh.service, whose guard was
#        /usr/bin/nvidia-smi -L || /usr/sbin/nvidia-smi -L || /usr/lib/wsl/lib/nvidia-smi -L
#      On Ubuntu the WSL path does not exist. Because nvidia-smi was failing for
#      reason (1) -- exactly the condition the guard exists to notice -- the
#      shell chain ended on a missing binary and the unit died with status 127
#      (command not found) instead of a meaningful failure. The CDI spec was
#      therefore never regenerated.
#
#   3. The unit sets Restart=on-failure, but it is Type=oneshot and systemd
#      ignores Restart= for oneshot services. It never retried once, and sat
#      failed for two days without anyone being told.
#
# Net effect: /var/run/cdi/nvidia.yaml kept naming driver libraries that apt had
# deleted, so EVERY GPU container start failed -- including the ones a reboot
# was supposed to fix. Running containers masked it: they had started before the
# upgrade, so the lab looked healthy until something merely needed a restart.
#
# WHAT THIS SCRIPT CHANGES (all idempotent, all reversible)
#   A. drop-in     clean, meaningful guard for nvidia-cdi-refresh (no WSL path)
#   B. timer       retry the CDI refresh every 10 min so it self-heals
#   C. apt pin     keep the driver stack out of UNATTENDED upgrades only
#   D. health      check + timer that reports module/user-space/spec drift loudly
#
# Nothing here blocks deliberate upgrades. See the comment in the apt file for
# the supported way to apply one.
#
# Usage:
#   ssh arthur@192.168.0.87 'bash -s' < harden_nvidia_driver.sh --dry-run
#   ssh arthur@192.168.0.87 'bash -s' < harden_nvidia_driver.sh
#
# Deploy check_gpu_stack.sh to $OPS_DIR first, otherwise step D is skipped.
# ─────────────────────────────────────────────────────────────────────────────

set -u

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

OPS_DIR=/opt/tts-lab-ops
CHK=$OPS_DIR/check_gpu_stack.sh
PIN_FILE=/etc/apt/apt.conf.d/52tts-lab-nvidia-driver-pin

log()  { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '  !! %s\n' "$*" >&2; }

run() {
    if [ "$DRY" = 1 ]; then info "[dry-run] $*"; else sudo -n "$@"; fi
}

write_file() { # $1 = path, content on stdin
    if [ "$DRY" = 1 ]; then
        info "[dry-run] would write $1"
        cat >/dev/null
    else
        sudo -n tee "$1" >/dev/null || { warn "could not write $1"; return 1; }
        info "wrote $1"
    fi
}

# ── A. a guard that fails meaningfully ───────────────────────────────────────
log "A. nvidia-cdi-refresh guard"
run install -d /etc/systemd/system/nvidia-cdi-refresh.service.d
write_file /etc/systemd/system/nvidia-cdi-refresh.service.d/tts-lab-override.conf <<'EOF'
# Managed by TTS-LAB scripts/utils/harden_nvidia_driver.sh -- do not edit by hand.
#
# The vendor unit guards its refresh with
#   /usr/bin/nvidia-smi -L || /usr/sbin/nvidia-smi -L || /usr/lib/wsl/lib/nvidia-smi -L
#
# but on Ubuntu /usr/lib/wsl/lib/nvidia-smi does not exist. When nvidia-smi
# fails -- precisely the after-an-upgrade-without-a-reboot case the guard is
# meant to catch -- the chain falls through to that missing binary and the unit
# exits 127 "command not found", which looks like a broken unit rather than a
# broken driver. The CDI spec is then left stale, and a stale spec breaks every
# GPU container start.
#
# Replace the chain with an explicit precondition: no shell fallthrough, a real
# exit status, and the spec regenerated only while the driver is consistent.
[Service]
ExecStart=
ExecStartPre=/usr/bin/nvidia-smi -L
ExecStart=/usr/bin/nvidia-ctk cdi generate
EOF

# ── B. retry, because oneshot never does ─────────────────────────────────────
log "B. CDI refresh retry timer"
write_file /etc/systemd/system/nvidia-cdi-refresh.timer <<'EOF'
# Managed by TTS-LAB scripts/utils/harden_nvidia_driver.sh
#
# The vendor unit is Type=oneshot, and systemd ignores Restart= for oneshot
# services -- so one failed refresh stays failed forever. This timer re-runs it,
# letting the spec converge by itself once module and user-space agree again,
# without depending on anyone noticing the failed unit.
[Unit]
Description=Periodically refresh the NVIDIA CDI specification (TTS-LAB self-heal)

[Timer]
OnBootSec=3min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
EOF

# ── C. stop the silent driver bump ───────────────────────────────────────────
log "C. keep the driver stack out of unattended upgrades"
write_file "$PIN_FILE" <<'EOF'
// Managed by TTS-LAB scripts/utils/harden_nvidia_driver.sh
//
// An NVIDIA driver upgrade is only safe when a reboot follows it. apt replaces
// the user-space libraries immediately, but the module already loaded in RAM
// keeps running until the machine restarts. In the gap, nvidia-smi and every
// NEW host CUDA process fail -- and, because the CDI spec then goes stale, every
// GPU container start fails too. That gap is what broke this box on 2026-09-11.
//
// This blacklist affects UNATTENDED upgrades only. Deliberate upgrades still
// work normally; apply one as:
//
//     sudo apt install --only-upgrade \
//          nvidia-driver-580-open nvidia-dkms-580-open \
//          nvidia-utils-580 libnvidia-compute-580
//     sudo reboot
//
// To allow unattended driver upgrades again, delete this file.
Unattended-Upgrade::Package-Blacklist {
    "nvidia-";
    "libnvidia-";
};
EOF

# ── D. make the drift visible ────────────────────────────────────────────────
log "D. GPU stack health check"
if [ "$DRY" = 0 ] && [ ! -x "$CHK" ]; then
    warn "SKIPPED: $CHK is not installed/executable"
    warn "deploy it first, e.g.:"
    warn "  Get-Content scripts/utils/check_gpu_stack.sh | ssh arthur@192.168.0.87 'sudo tee $CHK >/dev/null && sudo chmod +x $CHK'"
else
    write_file /etc/systemd/system/tts-lab-gpu-health.service <<EOF
# Managed by TTS-LAB scripts/utils/harden_nvidia_driver.sh
#
# Reports module / user-space / CDI-spec drift. Exits non-zero on a problem, so
# `systemctl --failed` and journalctl surface it without anyone watching a log.
[Unit]
Description=TTS-LAB GPU stack health check (module / user-space / CDI drift)

[Service]
Type=oneshot
ExecStart=$CHK
EOF
    write_file /etc/systemd/system/tts-lab-gpu-health.timer <<'EOF'
# Managed by TTS-LAB scripts/utils/harden_nvidia_driver.sh
[Unit]
Description=Periodic TTS-LAB GPU stack health check

[Timer]
OnBootSec=4min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
EOF
fi

# ── apply ────────────────────────────────────────────────────────────────────
log "applying"
run systemctl daemon-reload
run systemctl reset-failed nvidia-cdi-refresh.service
# The .path unit can give up independently. On 2026-09-11 the upgrade fired it
# repeatedly, each run failed (see A), and systemd's start rate limiter tripped:
# "unit-start-limit-hit". It then stayed failed, silently disabling the
# event-driven trigger that is supposed to refresh the spec on driver/toolkit
# installs. Reset it so that trigger works again.
run systemctl reset-failed nvidia-cdi-refresh.path
run systemctl start nvidia-cdi-refresh.path
run systemctl enable --now nvidia-cdi-refresh.timer
if [ "$DRY" = 0 ] && [ -x "$CHK" ]; then
    run systemctl enable --now tts-lab-gpu-health.timer
fi

log "verifying apt config still parses"
if [ "$DRY" = 1 ]; then
    info "[dry-run] apt-config dump"
elif sudo -n apt-config dump >/dev/null 2>&1; then
    info "apt config OK"
else
    warn "APT CONFIG INVALID -- delete $PIN_FILE immediately"
fi

log "proving the refresh now succeeds"
if [ "$DRY" = 1 ]; then
    info "[dry-run] systemctl start nvidia-cdi-refresh.service"
elif run systemctl start nvidia-cdi-refresh.service; then
    info "nvidia-cdi-refresh.service: success"
else
    warn "nvidia-cdi-refresh.service: FAILED -- inspect: systemctl status nvidia-cdi-refresh"
fi

log "state"
if [ "$DRY" = 1 ]; then
    info "[dry-run] no changes made"
else
    printf '    cdi-refresh.timer : enabled=%s active=%s\n' \
        "$(systemctl is-enabled nvidia-cdi-refresh.timer 2>&1)" \
        "$(systemctl is-active  nvidia-cdi-refresh.timer 2>&1)"
    printf '    gpu-health.timer  : enabled=%s active=%s\n' \
        "$(systemctl is-enabled tts-lab-gpu-health.timer 2>&1)" \
        "$(systemctl is-active  tts-lab-gpu-health.timer 2>&1)"
    printf '    apt blacklist     : %s pattern group(s) in %s\n' \
        "$(grep -c 'Unattended-Upgrade::Package-Blacklist' "$PIN_FILE" 2>/dev/null)" "$PIN_FILE"
fi

log "done"
