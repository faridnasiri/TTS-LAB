# Session 2026-09-13 — NVIDIA driver mismatch + stale CDI spec

> Host: `arthur-server` (`arthur@192.168.0.87`, Proxmox VM 104) · GPU: RTX 5060 Ti 16 GB (sm_120)
> Status: **resolved** · Downtime: ~1 min (image lab + ComfyUI + engine-current restarted)
> Trigger: a third-party report that "the card is fine, the driver halves mismatch"

## TL;DR

The report was accurate but incomplete, and the missing half was the dangerous one.
There were **three** faults, not one, and they had to line up exactly:

1. **Driver upgraded without a reboot.** An NVIDIA driver is two halves that must match: the
   kernel module resident in RAM, and the user-space libraries on disk. `apt` replaced the
   on-disk libraries (`580.178.04`) but the loaded module stayed `580.173.02`.
2. **The CDI refresh failed with status 127.** `nvidia-cdi-refresh.service` uses the guard
   `nvidia-smi -L || /usr/sbin/nvidia-smi -L || /usr/lib/wsl/lib/nvidia-smi -L`. Ubuntu has no
   `/usr/lib/wsl/...`; because `nvidia-smi` was failing *for reason 1* — exactly what the guard
   exists to notice — the shell chain ended on a missing binary and exited
   **127 (command not found)**. It therefore never regenerated `/var/run/cdi/nvidia.yaml`.
3. **No retry.** The unit sets `Restart=on-failure`, but it is `Type=oneshot` and systemd
   **ignores `Restart=` for oneshot services**. It failed once and stayed failed for two days.

**Impact of the missing half:** stale CDI spec → the toolkit bind-mounts the driver libraries it
lists into every GPU container → **every GPU container start failed**:

```
failed to fulfil mount request: open /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.580.173.02
```

That includes containers restarted by a deploy, by `systemctl`, or **by a reboot**. The lab looked
healthy only because `tts-lab-engine-current` had been running since 2026-09-08 — *before* the
upgrade. It was one restart away from being down.

## Two things I got wrong before probing properly

Recorded deliberately, because both were confident and cheap to check.

| Claim I made early | Reality |
|---|---|
| "Containers are immune — the fault is host-only" | Containers only *looked* fine because they pre-dated the upgrade. **None could be restarted.** The images do not bundle their own driver user-space at all — the toolkit bind-mounts the host libraries listed in the CDI spec, so the spec governs container start. |
| "Fix = reboot, or a module reload; both work" | A module reload **alone** leaves every container broken (the spec is still stale). And a **reboot would not have fixed it** — it would have taken the entire TTS lab down with no way back short of regenerating the spec by hand. |
| "`tts-lab-gpu-probe` is a leftover container" | It is deliberate infrastructure: `tts_lab_dispatch.py::_gpu_probe_exec` creates and reuses it (`tail -f /dev/null`, host PID namespace, GPU device request) purely to run `nvidia-smi` in-container. Deleting it is futile — the next GPU probe recreates it, which is exactly what happened ~3 min later. Its permanently "unhealthy" state is real, though: it inherits the engine image's `HEALTHCHECK`. Fixed in code for that container. |

The probe that was missing is trivially cheap: **start** a container (or restart one service)
instead of `docker exec`-ing into one that is already running. `docker exec` exercises none of the
start path — not mounts, not hooks, not spec resolution. Serving is not the same as startable.

The repair also lived in **two places** — kernel module (host) and generated spec (container
plumbing). Fixing the visible one left the user-facing one broken, which is what made the first
"fixed!" report wrong.

## Diagnosis — evidence

```
/proc/driver/nvidia/version        → 580.173.02     (what is running)
/lib/modules/6.8.0-136/…/nvidia.ko → 580.178.04     (what is on disk)
libnvidia-ml.so.1                  → 580.178.04     (what apps link)
uptime                             → 27 days        (no reboot since the upgrade)
dpkg.log                           → 2026-09-11 06:48:05 nvidia 580.178.04 installed
/var/run/cdi/nvidia.yaml           → generated 2026-08-17, 89 refs to 580.173.02
nvidia-cdi-refresh.service         → failed (status 127) since 2026-09-11 06:46:14
lsmod                              → nvidia_uvm refcount 24, nvidia refcount 138
fuser /dev/nvidia*                 → host image_lab.py + ComfyUI main.py
```

Note `/var/run/reboot-required` was dated 2026-09-10 — i.e. **before** the driver upgrade. A reboot
was already pending for kernel/libc6 updates; the driver upgrade was not the original cause.

## Repair

Done with `scripts/utils/fix_nvidia_driver_mismatch.sh` (idempotent, `--dry-run` supported):

1. Stopped the GPU consumers — `arthur-imglab`, `arthur-comfy`, and `tts-lab-engine-current`
   (containers take the GPU via compose `deploy.resources.reservations.devices`).
2. Waited for the `/dev/nvidia*` handles to release (blocked by refcounts otherwise).
3. `modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia` → `modprobe nvidia nvidia_uvm nvidia_drm`.
   **No reboot needed**: the on-disk `.ko` was already `580.178.04` for the running kernel
   `6.8.0-136` *and* for the pending `6.8.0-138` (`dkms status` → installed).
4. **Regenerated the CDI spec** — `nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`.
   This is the step that actually un-blocks containers, and it is why a module reload alone is not
   a fix.
5. Restarted the consumers.

## Verification

| Check | Result |
|---|---|
| Host `nvidia-smi` | `RTX 5060 Ti, 580.178.04` ✅ |
| New container with `--gpus all` (previously failed) | `RTX 5060 Ti, 580.178.04` ✅ |
| In-container `torch.cuda.is_available()` + real 1024×1024 matmul | `True` / `True` ✅ |
| `tts-lab-engine-current` | `Up (healthy)` ✅ |
| Image lab `:8002` / orchestrator `:8009` | `200` / `200` ✅ |
| CDI spec references | 89 → all `580.178.04` ✅ |

The drift detector was also **proved on a synthetic stale spec** (pointing at the deleted
`libEGL_nvidia.so.580.173.02`) → detected 1 missing path; against the real spec → 0. A detector
that has never been shown to fire is not a detector.

## Hardening — `scripts/utils/harden_nvidia_driver.sh`

All idempotent; addresses a different one of the three faults each.

| # | Change | Closes |
|---|---|---|
| A | systemd drop-in `/etc/systemd/system/nvidia-cdi-refresh.service.d/tts-lab-override.conf` — replaces the shell chain with `ExecStartPre=nvidia-smi -L` + `ExecStart=nvidia-ctk cdi generate` | the 127 exit; failures now carry a meaningful code |
| B | `/etc/systemd/system/nvidia-cdi-refresh.timer` (3 min after boot, then every 10 min) | `Type=oneshot` never retrying — the spec now converges on its own |
| C | `/etc/apt/apt.conf.d/52tts-lab-nvidia-driver-pin` — `Unattended-Upgrade::Package-Blacklist { "nvidia-"; "libnvidia-"; }` | fault 1 at its source: a silent driver bump. Deliberate upgrades are unaffected (procedure is in the file) |
| D | `/opt/tts-lab-ops/check_gpu_stack.sh` + `tts-lab-gpu-health.service/.timer` (every 15 min) | invisibility — drift is reported, and `systemctl --failed` shows it |

The health check is read-only except for one narrowly scoped self-heal: it regenerates the CDI spec
**only when module and user-space already agree** (a freshly generated spec would otherwise name
libraries the loaded module cannot use). It never reloads modules or changes packages.

**`tts-lab-gpu-probe` deserves a note, because it was this session's first false trail.** It is not a
leftover: `tts_lab_dispatch.py::_gpu_probe_exec` deliberately creates and reuses it (`tail -f
/dev/null`, host PID namespace, GPU device request) so that `nvidia-smi` can be run inside a
container. Deleting it is pointless — the next GPU probe recreates it, which is exactly what
happened ~3 minutes after the first removal.

What *was* wrong is its health state. It inherits the engine image's `HEALTHCHECK`
(`curl :8105/health`), which can never pass in a container running only `tail`, so `docker ps`
permanently showed an unhealthy container on a perfectly healthy GPU host — a signal that reads
exactly like a broken GPU stack, and it did send this session's first diagnosis down the wrong
path. Fixed in `tts_lab_dispatch.py` by creating the probe with `"Healthcheck": {"Test":
["NONE"]}`; that takes effect when the dispatch code is next rebuilt/deployed (the running
orchestrator container still has the old behaviour).

## Operator runbook

```bash
# Is the driver stack self-consistent?  (exit 1 = problem)
ssh arthur@192.168.0.87 'sudo /opt/tts-lab-ops/check_gpu_stack.sh'

# Inspect before/after without changing anything
Get-Content scripts/utils/fix_nvidia_driver_mismatch.sh | ssh arthur@192.168.0.87 'bash -s -- --dry-run'

# Repair a mismatch (module reload + CDI regen; ~1 min downtime)
Get-Content scripts/utils/fix_nvidia_driver_mismatch.sh | ssh arthur@192.168.0.87 'bash -s'

# Apply/re-apply the hardening
Get-Content scripts/utils/harden_nvidia_driver.sh | ssh arthur@192.168.0.87 'bash -s'
```

Deliberate driver upgrade (the supported path — automatic ones are now blocked):

```bash
sudo apt install --only-upgrade nvidia-driver-580-open nvidia-dkms-580-open \
     nvidia-utils-580 libnvidia-compute-580
sudo reboot
```

## Facts worth remembering

- The card was never at fault. The whole incident was a software version skew plus a broken
  self-repair path.
- The on-disk module being correct for the **running** kernel is what makes a no-reboot repair
  possible — check `modinfo -F version /lib/modules/$(uname -r)/updates/dkms/nvidia.ko` first.
- A driver fault has **two** blast radii with **two** different remedies: host CUDA (module ↔
  user-space) and container start (CDI spec). Always verify both.
- Verify by *starting* something, not by inspecting something already running.
