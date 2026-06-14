# Session 2026-04-22 — GPU Passthrough Fix + flash-attn Investigation

## Summary
Started with the goal of installing `flash-attn` to speed up Qwen3-TTS inference.
Ended up diagnosing and fixing a Proxmox VM misconfiguration (balloon RAM + GPU passthrough OVMF hang),
confirming flash-attn is optional, and leaving the system fully healthy.

---

## Timeline

### Phase 1 — HuggingFace Token + Qwen3-TTS Enablement
- Token provided and saved via `huggingface-cli login` + `/etc/environment`
- Discovered `Qwen/Qwen3-TTS` does not exist — real models are:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — voice clone only, needs ref audio
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` ← **default chosen**, 9 built-in speakers
- Wrong package (`AutoProcessor/AutoModel`) replaced with `pip install -U qwen-tts`
- `transformers` upgraded from `4.52.1` → `5.6.0.dev0` (git HEAD, required by qwen-tts)
- Fixed kwargs: `speaker_name=` → `speaker=`
- Test result: **PASS  dur=4560ms  rtf=4.75×  sr=24000Hz**

### Phase 2 — flash-attn Installation Attempt
**Goal:** reduce RTF from 4.75× to ~2–3× by enabling Flash Attention 2.

#### What flash-attn does
- Rewrites the GPU attention kernel using tiling + fused ops
- Never writes the full N×N attention matrix to HBM
- Same mathematical output, 2–4× faster, 30–50% less VRAM for attention layers

#### Problem: VM only had 2.6 GB RAM
- `free -h` showed 2.6 GB despite Proxmox host having 80 GB
- Root cause: **Proxmox memory balloon** set to `balloon: 4096` (capped at 4 GB, deflated to 2.6 GB)
- nvcc OOM-killed during compile with multiple GPU targets

#### Fix: Disable balloon
```bash
# On Proxmox
qm set 104 --balloon 0 --memory 32768
```
- Set to 32 GB (enough for compilation, fast UEFI init)

### Phase 3 — GPU Passthrough OVMF Hang
After RAM fix, VM stopped booting when GPU passthrough was enabled.

#### Root cause
- Old EFI NVRAM disk had stale GPU device state from config changes
- OVMF hung during POST trying to enumerate RTX 5060 Ti (Blackwell GB206) BAR regions
- Tried: `rombar=0` (GPU fell off bus), `x-vga=1` (still hung), `vga=none` (no screen)

#### Fix: Wipe + recreate EFI disk
```bash
# On Proxmox
qm stop 104 --skiplock
# Old approach (zeros existing disk):
dd if=/dev/zero of=/dev/pve-nvme/vm-104-disk-2 bs=1M count=4
# Better approach (recreate fresh):
qm set 104 --delete efidisk0
qm set 104 --efidisk0 'nvme-lvm:4,efitype=4m,pre-enrolled-keys=0'
qm start 104
```

#### Final working GPU config
```
hostpci0: 0000:05:00,pcie=1   # no rombar/x-vga flags needed with clean EFI
memory: 32768
balloon: 0
cpu: host
vga: std
```

### Phase 4 — flash-attn Compile Attempts (6 hours, abandoned)

#### Pre-built wheels — no match
- Highest available: `torch2.8+cu12` — incompatible with `torch2.11` (ABI mismatch, segfault)
- No wheel exists for `torch2.11+cu128` as of 2026-04-22

#### Source compile blockers fixed one by one
| Error | Fix |
|-------|-----|
| `nvcc exit 255` (OOM) | Fixed balloon → 32 GB RAM |
| `gcc not found` | `update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110` |
| `x86_64-linux-gnu-gcc not found` | `ln -sf /usr/bin/gcc-11 /usr/bin/x86_64-linux-gnu-gcc` |
| `as: execvp not found` | Symlink in wrong dir; moved to `/usr/bin` |
| `version 0.0.0` | `export TORCH_DONT_CHECK_COMPILER_ABI=1` |
| Compile running 6 hours | MAX_JOBS=4 + sm_120 only still too slow — killed |

#### Why it's slow
- flash-attn 2.x compiles ~150–200 `.cu` files
- Each file with sm_120 generates large intermediate PTX + SASS
- Even with `MAX_JOBS=4` and single arch target, takes 3–8 hours on a 12-core VM

#### Conclusion
**flash-attn is optional for Qwen3-TTS.** The import is wrapped in `try/except ImportError` in `qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py`. The engine works perfectly without it.

### Phase 5 — Final State
```
  ▶ qwen3tts   PASS  dur=3920ms  rtf=4.46×  sr=24000Hz  load=9.72s  synth=17498ms
Results: 1 passed  0 failed  0 skipped
```
- VM: 31 GB RAM, RTX 5060 Ti working, `arthur-lab` active
- Qwen3-TTS: fully operational, 9 built-in speakers
- flash-attn: not installed — optional, no functional impact

---

## Proxmox SSH Key Setup (permanent)
```powershell
# Generate passwordless key for Proxmox root (one-time)
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_proxmox" -N ""

# Install on Proxmox (requires password once)
$pub = Get-Content "$env:USERPROFILE\.ssh\id_proxmox.pub" -Raw
ssh root@192.168.0.153 "echo '$($pub.Trim())' >> ~/.ssh/authorized_keys"
```

`~/.ssh/config` entry:
```
Host proxmox
    HostName 192.168.0.153
    User root
    IdentityFile ~/.ssh/id_proxmox
    PasswordAuthentication no
    StrictHostKeyChecking no
    ConnectTimeout 10
```

---

## VM Reference
| Item | Value |
|------|-------|
| Proxmox host | `192.168.0.153` |
| VM ID | `104` |
| VM name | `arthur-server2` |
| VM IP | `192.168.0.87` |
| SSH alias | `arthur-vm` |
| RAM | 32 GB (max 80 GB, balloon disabled) |
| GPU | RTX 5060 Ti (Blackwell GB206, sm_120, PCI 0000:05:00) |
| OS | Ubuntu 22.04 |
| Python env | `/opt/arthur-bench-env` |
| Service | `arthur-lab` on port 8001 |
| Code | `/opt/arthur/tts_lab.py` |
| Test harness | `/opt/arthur/_tts_test.py` |

---

## If flash-attn is needed in future
Options in order of preference:
1. **Wait** — a `torch2.11+cu128` pre-built wheel will appear on the flash-attn releases page eventually
2. **Compile overnight** — `MAX_JOBS=1`, `TORCH_CUDA_ARCH_LIST=12.0`, run in a `screen` session
3. **Use sageattention** — installs instantly but segfaulted on this GPU (Blackwell too new as of 2026-04-22)
4. **Use `torch.compile`** — built into PyTorch 2.x, no install needed, provides similar speedups for some models

```bash
# Check periodically for a compatible pre-built wheel:
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest \
  | python3 -m json.tool | grep browser_download_url | grep cp311 | grep cu12
```

---

## Commits This Session
| Hash | Message |
|------|---------|
| `4a2c3b1` | chore: set HF_TOKEN in environment |
| `bb669d6` | fix(tts_lab): update Qwen3-TTS to correct released model ID |
| `87c69da` | fix(tts_lab): rewrite Qwen3-TTS to use qwen-tts package |
| `d6cfbe0` | fix(tts_lab): switch qwen3tts default to CustomVoice model |
| `8b5daf9` | fix(tts_lab): fix generate_custom_voice kwarg speaker_name -> speaker |
| `ee3bdcd` | test(tts_lab): enable qwen3tts in test harness |
| `839c34b` | docs: add session notes for Qwen3-TTS enablement |
| `61453ac` | docs: add deployGpuPackage reusable prompt |

---

## ⚠️ Action Required
**Rotate the HF token shared in this session:**
https://huggingface.co/settings/tokens
Token prefix: `hf_YlZz...` — revoke and create a new one.
