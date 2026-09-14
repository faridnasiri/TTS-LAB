#!/usr/bin/env bash
# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
# check_gpu_stack.sh — detect NVIDIA module / user-space / CDI drift
#
# Catches the 2026-09-11 failure class before it takes the lab down:
#
#   1. kernel module version != user-space library version
#      -> nvidia-smi, host CuPy/PyTorch, and every NEW host CUDA process fail.
#
#   2. /var/run/cdi/nvidia.yaml references driver libraries that do not exist
#      -> EVERY GPU container start fails. This is the dangerous one because it
#         is invisible: already-running containers keep working, so the lab
#         looks healthy right up until something restarts -- a deploy, a crash,
#         or a reboot. On 2026-09-11 this state sat undetected for two days.
#
# Read-only, with ONE narrowly scoped self-heal: if the module and user-space
# agree but the CDI spec is stale, the spec is regenerated. That is idempotent,
# needs no restart, and is precisely the repair that un-blocks containers. The
# self-heal is skipped whenever the driver is itself inconsistent, because a
# freshly generated spec would then name libraries the loaded module cannot use.
# Nothing else is ever modified -- no module reloads, no package changes.
#
# Exit codes:  0 healthy   1 problem found   2 undetermined (driver not loaded)
#
# Usage:  check_gpu_stack.sh [--no-fix] [--quiet]
# ─────────────────────────────────────────────────────────────────────────────

set -u

FIX=1
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --no-fix) FIX=0 ;;
        --quiet)  QUIET=1 ;;
    esac
done

say() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }

# systemd runs this as root; interactive use needs sudo for the read of
# /var/run/cdi and for the single write. Keep one code path for both.
run() {
    if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi
}

CDI=/var/run/cdi/nvidia.yaml
FAIL=0

# ── 1. module vs user-space ──────────────────────────────────────────────────
if [ ! -r /proc/driver/nvidia/version ]; then
    say "UNDETERMINED: /proc/driver/nvidia/version unreadable (nvidia module not loaded?)"
    exit 2
fi

MOD=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
LIB=$(readlink -f /lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+$')

say "kernel module : $MOD"
say "user-space    : ${LIB:-unknown}"

if [ -n "$LIB" ] && [ "$LIB" != "$MOD" ]; then
    FAIL=1
    say "PROBLEM  module and user-space disagree."
    say "         Host CUDA is broken (nvidia-smi, CuPy, PyTorch on the host)."
    say "         Repair: sudo scripts/utils/fix_nvidia_driver_mismatch.sh"
fi

# Count CDI hostPaths that point at files which are not there.
missing_paths() {
    run grep -oE '/[^"]*\.so\.[0-9]+\.[0-9]+\.[0-9]+' "$1" 2>/dev/null \
        | sort -u \
        | while read -r p; do [ -e "$p" ] || printf '%s\n' "$p"; done \
        | grep -c . || true
}

# ── 2. CDI spec ──────────────────────────────────────────────────────────────
if [ ! -e "$CDI" ]; then
    say "CDI spec      : absent ($CDI)"
    if [ -n "$LIB" ] && [ "$FIX" = 1 ] && command -v nvidia-ctk >/dev/null 2>&1; then
        if run nvidia-ctk cdi generate --output="$CDI" >/dev/null 2>&1; then
            say "                regenerated"
        else
            say "                regeneration FAILED"
            FAIL=1
        fi
    fi
elif [ -n "$LIB" ] && [ "$LIB" = "$MOD" ]; then
    say "CDI spec refs : $(grep -oE '\.so\.[0-9]+\.[0-9]+\.[0-9]+' "$CDI" 2>/dev/null | sed 's/^\.so\.//' | sort -u | tr '\n' ' ')"

    n_missing=$(missing_paths "$CDI")
    if [ "${n_missing:-0}" -gt 0 ]; then
        FAIL=1
        say "PROBLEM  CDI spec names $n_missing driver libraries that do not exist."
        run grep -oE '/[^"]*\.so\.[0-9]+\.[0-9]+\.[0-9]+' "$CDI" 2>/dev/null \
            | sort -u | while read -r p; do [ -e "$p" ] || printf '         %s\n' "$p"; done | head -3
        say "         Every GPU container start will fail until the spec is refreshed."

        if [ "$FIX" = 1 ] && command -v nvidia-ctk >/dev/null 2>&1; then
            if run nvidia-ctk cdi generate --output="$CDI" >/dev/null 2>&1; then
                after=$(missing_paths "$CDI")
                say "         regenerated; missing paths now: ${after:-0}"
                [ "${after:-0}" -eq 0 ] && FAIL=0
            else
                say "         regeneration FAILED"
            fi
        fi
    else
        say "                all referenced libraries exist"
    fi
else
    say "CDI spec      : not validated (module and user-space already disagree)"
fi

if [ "$FAIL" = 0 ]; then
    say "RESULT: healthy"
fi
exit "$FAIL"
