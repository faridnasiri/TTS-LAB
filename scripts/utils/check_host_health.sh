#!/usr/bin/env bash
# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
# check_host_health.sh — catch the failures `systemctl --failed` cannot see
#
# Written after two silent failures on this box:
#
#   1. `arthur.service` crash-looped under `Restart=always` for ~26 days
#      (453,690 restart attempts, ~4,800 journal lines/hour). `systemctl
#      --failed` printed NOTHING the whole time, because a unit in auto-restart
#      is `activating`, not `failed`. "No failed units" is not "nothing is
#      failing".
#
#   2. Pending driver/kernel updates sat unnoticed, and the unattended driver
#      bump-without-a-reboot is what broke the whole GPU stack. So this reports
#      pending upgrades and calls out the set that needs a maintenance window.
#
# It does NOT change anything. Exit: 0 healthy, 1 problem.
#
#   Usage:  check_host_health.sh              # report
#           check_host_health.sh --quiet      # only problems and warnings
#           check_host_health.sh --install    # install systemd unit + 30-min timer
#
# --quiet never suppresses a PROBLEM or WARNING: a check that goes silent is how
# the last two failures stayed hidden.
# ─────────────────────────────────────────────────────────────────────────────

set -u

QUIET=0
case "${1:-}" in
    --quiet)   QUIET=1 ;;
    --install)
        OPS_DIR=/opt/tts-lab-ops
        SELF="$OPS_DIR/check_host_health.sh"
        run() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi; }
        run install -d "$OPS_DIR" || exit 1

        run tee /etc/systemd/system/tts-lab-host-health.service >/dev/null <<EOF
[Unit]
Description=TTS-LAB host health check (crash loops, failed units, disk, pending updates)

[Service]
Type=oneshot
ExecStart=$SELF
EOF

        run tee /etc/systemd/system/tts-lab-host-health.timer >/dev/null <<'EOF'
[Unit]
Description=Periodic TTS-LAB host health check

[Timer]
OnBootSec=6min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
EOF
        run systemctl daemon-reload
        run systemctl enable --now tts-lab-host-health.timer
        echo "installed: tts-lab-host-health.service + .timer (every 30 min)"
        exit 0
        ;;
esac

say() { printf '%s\n' "$*"; }   # problems/warnings -- ALWAYS printed
ok()  { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }  # routine OK lines
FAIL=0
WARN=0

# ── crash loops: the class `systemctl --failed` is blind to ──────────────────
loops=$(systemctl list-units --state=activating --no-legend --plain --no-pager 2>/dev/null \
        | grep "auto-restart" || true)
if [ -n "$loops" ]; then
    FAIL=1
    say "PROBLEM  unit(s) stuck in auto-restart (crash loop):"
    printf '%s\n' "$loops" | sed 's/^/         /'
    say "         cause:  systemctl status <unit>"
    say "         stop:   systemctl disable --now <unit>"
else
    ok "crash loops  : none"
fi

# ── failed units ─────────────────────────────────────────────────────────────
failed=$(systemctl --failed --no-legend --plain --no-pager 2>/dev/null | grep . || true)
if [ -n "$failed" ]; then
    FAIL=1
    say "PROBLEM  failed unit(s):"
    printf '%s\n' "$failed" | sed 's/^/         /'
else
    ok "failed units : none"
fi

# ── disk ─────────────────────────────────────────────────────────────────────
while read -r pcent target; do
    pct=${pcent%\%}
    case "$pct" in ''|*[!0-9]*) continue ;; esac
    if [ "$pct" -ge 92 ]; then
        FAIL=1; say "PROBLEM  disk $target at ${pct}% (>=92% - this box has hit full before)"
    elif [ "$pct" -ge 85 ]; then
        WARN=1; say "WARNING  disk $target at ${pct}% (>=85%)"
    else
        ok "disk $target   : ${pct}%"
    fi
done < <(df -h --output=pcent,target / 2>/dev/null | tail -n +2)

# ── pending upgrades ─────────────────────────────────────────────────────────
upg=$(apt-get -s upgrade 2>/dev/null | grep "^Inst " | awk '{print $2}' || true)
n_upg=$(printf '%s\n' "$upg" | grep -c . || true)
ok "pending pkgs : ${n_upg:-0}"
if [ "${n_upg:-0}" -gt 0 ]; then
    # This set must be applied deliberately, in a window, with a reboot. Why:
    # an NVIDIA userspace update without a reboot leaves the loaded kernel
    # module older than the libraries -> nvidia-smi dies, host CUDA dies, and
    # the container CDI spec goes stale so no GPU container can start.
    # Docker/containerd and Grafana majors are their own maintenance events.
    risky=$(printf '%s\n' "$upg" | grep -E "^(nvidia|libnvidia|cuda|libcublas|docker|containerd|grafana)" || true)
    if [ -n "$risky" ]; then
        WARN=1
        say "         DELIBERATE-UPGRADE set pending (window + reboot required):"
        printf '%s\n' "$risky" | head -12 | sed 's/^/           /'
        say "         never let an unattended job apply these."
    fi
fi

if [ "$FAIL" = 0 ]; then
    [ "$WARN" = 0 ] && ok "RESULT: healthy" || say "RESULT: healthy, with warnings"
fi
exit "$FAIL"
