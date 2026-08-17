# Incident 2026-08-17 — Lab unreachable after reboot: guest hung in initramfs (ext4 error bit)

> **Date:** 2026-08-17 | **Duration:** ~8 hours of downtime (guest hung from ~07:00 UTC to 15:16 UTC, recovered 15:20 UTC)
> **Impact:** Whole lab down — orchestrator (8009), engine containers (8101/8104), Image Lab unaffected (8002 stayed on the host's other VM? No — same VM; it was down too).
> **Related:** [KNOWN_ISSUES.md](../reference/KNOWN_ISSUES.md), [04-ADHOC-LOG.md](../containerization/04-ADHOC-LOG.md)

---

## TL;DR

The VM guest (Ubuntu 22.04, VM 104 `arthur-server2`) booted after a forced restart but **hung at the initramfs busybox shell for ~8 hours**: the root ext4 filesystem carried the error bit ("clean with errors") from an earlier unclean shutdown, so boot-time `fsck -p` refused to auto-repair and dropped to a shell. The guest sat idle there — no network, no ACPI response — which made it look like a frozen OS.

**Fix:** `e2fsck -fy /dev/sda1` from the initramfs shell, then `exit` to resume boot. Zero data loss (`/lost+found` empty).

**Secondary finding:** the legacy bare-metal `arthur-lab.service` (port 8001) had been crash-looping since **Aug 12** (venv `/opt/arthur-bench-env` and lab code at `/opt/arthur/` were removed during the container migration). It was **retired** (stopped, disabled, masked) with user approval — the orchestrator (8009) is the active lab.

---

## Timeline

| Time (UTC) | Event |
|---|---|
| Aug 12 07:35 | VM booted; `arthur-lab.service` (8001) begins crash-looping — `/opt/arthur-bench-env/bin/uvicorn` missing (venv gone; `/opt/arthur/` code dir repurposed for container deployment the same day). Unnoticed for 5 days. |
| Aug 17 ~06:08 | Previous boot ends (unclean — qmshutdown attempts at 06:05/06:06 time out; guest unresponsive). |
| Aug 17 ~07:00 | User force-stops (`qm stop`) and restarts VM. Guest kernel boots, reads 274 MB from root disk, then **drops to `(initramfs)` shell and hangs** for 8h. |
| 15:16 | Recovery: `e2fsck -fy` completed (repaired orphaned-inode list from the crash, cleared error bit), `exit` resumes boot. |
| 15:18 | VM fully up: orchestrator + engine containers healthy, 19 engines available, piper synthesis verified (RTF 1.43). |
| 15:24 | `arthur-lab.service` (8001) retired (stopped, disabled, masked). |

---

## Root Cause

### Primary: ext4 error bit → boot-time fsck refuses to auto-repair → initramfs shell

1. A previous unclean shutdown (user force-stopped the VM after ACPI `qmshutdown` timed out) left the root fs with **orphaned inodes** — files mid-write at the moment of the kill.
2. ext4 records this by setting the **error bit** in the superblock: `Filesystem state: clean with errors`. The fs is still mountable, but any subsequent boot runs a full `fsck`.
3. Ubuntu's initramfs runs `fsck -p` (preen). Preen mode **cannot** auto-answer repairs for an orphaned-inode list, so it exits non-zero → the boot scripts drop to the **busybox `(initramfs)` shell** and wait for a human.
4. The guest sat at that shell: no network (networking starts after root mount), no ACPI handling (busybox doesn't service power events), near-zero CPU. Externally this is indistinguishable from a frozen OS.

**Evidence chain (how it was diagnosed):**
- VM unreachable on all ports, ARP silent → `qmshutdown` via Proxmox API also timed out → guest not just "slow", it's hung.
- QMP (`info blockstats` via `/var/run/qemu-server/104.qmp`): root disk got 274 MB of reads then **no I/O for 8h** — classic early-boot stall.
- Serial console (`qm terminal` over the `serial0: socket` device): guest was at an **`(initramfs)` busybox prompt**.
- From the shell: `blkid` showed sda1 ext4 with the *correct* UUID (`aa0bde06-...`); `dumpe2fs -h` showed **`Filesystem state: clean with errors`**.

### Secondary: retired 8001 service (pre-existing, unrelated to the hang)

The container migration repurposed `/opt/arthur/` (now holds `docker-compose.yml`, `reference_voices/`, `ref_archive_2026-08-12/`) and deleted the `/opt/arthur-bench-env` venv. The old systemd unit still pointed at the missing venv → crash-loop every 5 s, status `203/EXEC`. Confirmed **not** caused by the fsck (journal proves it started Aug 12).

---

## Fix Applied

### Step 1 — Repair the root fs from the initramfs shell

Serial console access: Proxmox VM has `serial0: socket`; the guest kernel runs `console=tty1 console=ttyS0` (visible in `/proc/cmdline`), so the shell is interactive over serial.

```sh
# at the (initramfs) prompt:
mkdir -p /mnt && mount -r /dev/sda1 /mnt      # verify fs is readable  → MOUNT_OK
umount /mnt                                    # never fsck a mounted fs
e2fsck -fy /dev/sda1                           # full check, auto-fix
#   → repaired orphaned inode list (Inode 1869, 3033, ... FIXED)
#   → extent-tree optimizations (cosmetic)
#   → Passes 1-5 complete; ~6.2 GB read; ~21 MB written
dumpe2fs -h /dev/sda1 | grep 'Filesystem state'   # → "clean" (error bit cleared)
exit                                           # resume boot → fsck passes → systemd starts
```

Boot log confirmed: `fsck.ext4 ... cloudimg-rootfs: clean, 1729820/83865600 files`, root re-mounted r/w, `/opt/models` (sdb1) mounted, docker + all lab services started, login prompt on ttyS0.

**Data loss: none.** `/lost+found/` is empty — all repaired inodes were re-linked in place.

### Step 2 — Verify the lab

- Orchestrator `http://192.168.0.87:8009/status` → 19 engines available (piper…omnivoice), GPU RTX 5060 Ti, 15.5 GB VRAM free.
- Engine containers healthy: `tts-lab-engine-current` (8101, 23 engines probed), `tts-lab-engine-qwen` (8104).
- End-to-end synthesis: `POST /synthesize/piper` → 22050 Hz WAV, RTF 1.43 ✅.
- Image Lab 8002, Cloudflare tunnel, Grafana, nginx: all active.

### Step 3 — Retire the zombie 8001 service (user-approved)

```sh
sudo systemctl stop arthur-lab.service
sudo systemctl disable arthur-lab.service
sudo rm /etc/systemd/system/arthur-lab.service
sudo ln -s /dev/null /etc/systemd/system/arthur-lab.service   # mask
sudo systemctl daemon-reload
```

Ports now: **8009** (orchestrator, active) · **8002** (Image Lab, active) · 8001 freed.

---

## How to Recover Faster Next Time (runbook)

Symptom: VM answers nothing on any port after a reboot/restart, `qmshutdown` (ACPI) times out.

1. **Proxmox API** (host `prox` @ 192.168.0.153, port 8006):
   ```powershell
   $r = Invoke-RestMethod -Uri 'https://192.168.0.153:8006/api2/json/access/ticket' \
     -Method Post -Body @{username='root@pam';password='<root-pass>'} -SkipCertificateCheck
   # VM list / status / start / stop:
   GET  /api2/json/nodes/prox/qemu            # list VMs
   GET  /api2/json/nodes/prox/qemu/104/status/current
   POST /api2/json/nodes/prox/qemu/104/status/start
   ```
2. **QMP** — guest-level truth: `python3 qmp_query.py 104 'hm:info blockstats'` (socket `/var/run/qemu-server/104.qmp`; handshake `qmp_capabilities` first). Disk reads frozen + idle time climbing = boot stall, not a busy hang.
3. **Serial console** — `qm terminal 104` (PTY required; works with paramiko `get_pty`). Kernel must have `console=ttyS0` (this VM does).
4. **At the shell**: `blkid` → `dumpe2fs -h` → `e2fsck -fy` → `exit`. Never `qm stop` while an fsck is pending.

### Prevention notes

- **Force-stops (`qm stop`) are the entry point to this failure mode.** Prefer `qm shutdown` with a long timeout; if the guest is hung, accept the fsck dance above — it is the designed recovery path.
- The error bit is cleared by fsck, not by reboot — **rebooting without running fsck will land in the same initramfs shell**.
- `arthur-lab.service` (8001) no longer exists on this VM. Anything referencing `/opt/arthur-bench-env` or port 8001 is stale — the containerized orchestrator is the lab.
