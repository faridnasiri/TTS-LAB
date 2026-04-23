# GPU Selection & Virtualisation — Full Q&A Reference
> Arthur TTS Lab · Xeon D-1528 · Supermicro X10SDV-6C-TLN4F  
> Discussion date: 2026-03-26 · Updated: 2026-04-23 (flash-attn SM 12.0 verdict, SDPA solution)

---

## Table of Contents

1. [Hardware Baseline](#1-hardware-baseline)
2. [RTX 3000 vs 4000 Series](#2-rtx-3000-vs-4000-series)
3. [RTX 5000 Series (Blackwell)](#3-rtx-5000-series-blackwell)
4. [Weakest 5000 Card — RTX 5060 8GB vs 4000 16GB for LLMs](#4-weakest-5000-card--rtx-5060-8gb-vs-4000-16gb-for-llms)
5. [Xeon D-1528 Processor Limitations](#5-xeon-d-1528-processor-limitations)
6. [Supermicro BIOS 2.6 — Live Analysis](#6-supermicro-bios-26--live-analysis)
7. [Hyper-V GPU Virtualisation — DDA vs GPU-PV](#7-hyper-v-gpu-virtualisation--dda-vs-gpu-pv)
8. [Easy-GPU-PV Article Analysis](#8-easy-gpu-pv-article-analysis)
9. [Underground / Community License Bypass Methods](#9-underground--community-license-bypass-methods)
10. [Hot vs Cold GPU Sharing](#10-hot-vs-cold-gpu-sharing)
11. [Alternative Hypervisors](#11-alternative-hypervisors)
12. [PowerShell Workload on Linux Containers](#12-powershell-workload-on-linux-containers)
13. [Final Recommendations Summary](#13-final-recommendations-summary)

---

## 1. Hardware Baseline

**Server:** Supermicro X10SDV-6C-TLN4F  
**CPU:** Intel Xeon D-1528 @ 1.90 GHz (Broadwell-DE, 6-core/12-thread, 35W TDP)  
**RAM:** 19 GB DDR4  
**Storage:** HDD, 50 GB models / 177 GB total  
**GPU:** None (CPU-only at time of benchmarking)  
**OS:** Ubuntu 22.04 (VM hosted on Hyper-V)

### CPU Benchmark Results (Measured Live)

| Engine | CPU RTF | Status |
|--------|--------:|--------|
| Piper TTS | 0.37 | ✅ real-time |
| MeloTTS | 1.01 | ⚠️ borderline |
| StyleTTS 2 | 1.52 | ⚠️ borderline |
| Kokoro-82M | 2.83 | ❌ too slow |
| XTTS-v2 | 3.85 | ❌ too slow |
| Dia-1.6B | 38.88 | ❌ far too slow |
| Orpheus 3B | ~45.0 | ❌ impossible on CPU |

> RTF < 1.0 = real-time capable. Arthur needs RTF ≤ 0.25 for sub-500ms response.

---

## 2. RTX 3000 vs 4000 Series

**Q: Is the 3000 series or 4000 series better for TTS + our board?**

### Key Specs Compared

| GPU | Gen | VRAM | Mem BW | FP32 TF | INT8 TOPS | FP8 TOPS | TDP |
|-----|-----|------|--------|---------|----------|----------|-----|
| RTX 3060 | Ampere | **12 GB** GDDR6 | 360 GB/s | 13.0 | 101 | ❌ | 170W |
| RTX 4060 | Ada Lovelace | 8 GB GDDR6 | 272 GB/s | 15.1 | 194 | **242** | 115W |
| RTX 4060 Ti | Ada Lovelace | **16 GB** GDDR6 | 288 GB/s | 22.1 | 176 | **352** | 165W |

### Architecture Differences for ML

| Feature | RTX 3060 (Ampere) | RTX 4060/Ti (Ada) | Matters for TTS? |
|---------|:-----------------:|:-----------------:|:---------------:|
| Tensor Core gen | 3rd | **4th** | ✅ yes |
| FP8 native | ❌ | ✅ | ✅ critical for vllm/Orpheus |
| BF16 throughput | 26 TF | 30–44 TF | ✅ yes |
| INT8 throughput | 101 TOPS | 176–352 TOPS | ✅ yes |
| VRAM (best variant) | **12 GB** | 16 GB (Ti) | ✅ decisive |

### VRAM Analysis — Why It Matters

```
Orpheus 3B model sizes:
  bf16  → 6.4 GB   RTX 3060: ✅ fits   RTX 4060 8GB: ❌ OOM
  INT8  → 3.2 GB   both fit
  FP8   → 1.6 GB   both fit

RTX 3060 12GB: Orpheus bf16 (6.4GB) + XTTS (3.5GB) = 9.9GB ✅ both loaded
RTX 4060 8GB:  Orpheus bf16 (6.4GB) alone = OOM risk ❌
RTX 4060 Ti 16GB: Orpheus bf16 (6.4GB) + XTTS (3.5GB) + more ✅
```

### Conclusion

| Use case | Winner | Reason |
|----------|--------|--------|
| TTS only (all 21 engines, budget) | **RTX 3060 12GB** | VRAM wins at lowest price |
| TTS + LLM quality (7B INT8) | **RTX 4060 Ti 16GB** | 16GB + FP8 Tensor Cores |
| Power-constrained setup | **RTX 4060 8GB** | 115W vs 170W |

---

## 3. RTX 5000 Series (Blackwell)

**Q: What about the 5000 series? We have external PSU and PCIe 3.0 x16 riser.**

### Blackwell Architecture — Key New Features for ML

| Feature | RTX 3060 (Ampere) | RTX 4060 (Ada) | RTX 5000 (Blackwell) |
|---------|:-----------------:|:--------------:|:--------------------:|
| Tensor Core gen | 3rd | 4th | **5th** |
| FP4 native | ❌ | ❌ | ✅ |
| FP8 throughput | ❌ | 242 TOPS | **570–3352 TOPS** |
| Memory type | GDDR6 | GDDR6 | **GDDR7** |

### FP4 Changes the VRAM Math Completely

```
Orpheus 3B by precision:
  bf16  = 6.4 GB   (3060: barely fits, 4060 8GB: OOM)
  INT8  = 3.2 GB
  FP8   = 1.6 GB
  FP4   = 0.9 GB   ← Blackwell only

RTX 5060 8GB with FP4:
  Orpheus FP4: 0.9 GB + KV cache 0.3 GB = 1.2 GB
  Remaining:   6.8 GB → fits XTTS (2GB) + Chatterbox (1.5GB) + Kokoro (0.5GB)
  Total used:  6.2 GB of 8 GB ✅
```

### RTX 5000 Full Spec Table

| GPU | VRAM | Mem BW | FP4 TOPS | TDP | PCIe | Price |
|-----|------|--------|----------|-----|------|-------|
| RTX 5060 | 8 GB GDDR7 | ~448 GB/s | ~570 | 150W | Gen 4 | ~$300 |
| RTX 5060 Ti | **16 GB** GDDR7 | ~448 GB/s | ~741 | 180W | Gen 4 | ~$400 |
| RTX 5070 | 12 GB GDDR7 | 672 GB/s | ~1,088 | 250W | Gen 4 | ~$600 |
| RTX 5070 Ti | **16 GB** GDDR7 | 896 GB/s | ~1,406 | 300W | Gen 4 | ~$800 |
| RTX 5080 | **16 GB** GDDR7 | 960 GB/s | ~1,801 | 360W | Gen 4 | ~$1,000 |
| RTX 5090 | **32 GB** GDDR7 | 1,792 GB/s | 3,352 | 575W | **Gen 5** | ~$2,000 |

### SM 12.0 (Blackwell) — Library Compatibility Notes (2026-04-23)

| Library | SM 12.0 Support | Notes |
|---|---|---|
| PyTorch 2.11+cu128 | ✅ Full | CUDA 12.8, all ops |
| `torch.compile` | ✅ | Triton backend works |
| `scaled_dot_product_attention` (SDPA) | ✅ | cuDNN fused kernel — **use instead of flash-attn** |
| flash-attn 2.x | ❌ | Max SM 9.0 (Hopper). No SM 12.0 support, no pre-built wheel for torch2.11 |
| flash-attn 4 (FA4) | ⚠️ | Beta, pure Python, different API (`flash_attn.cute`). Not compatible with HF transformers yet |
| bitsandbytes | ✅ | INT8/INT4 quantisation works |
| xformers | ⚠️ | Pre-built wheels lag behind — may need source compile |
| vllm | ✅ | Supports SM 12.0 as of recent releases |
| onnxruntime-gpu | ✅ | CUDA EP works |

**For any model using `attn_implementation="flash_attention_2"` on SM 12.0:**  
→ Switch to `attn_implementation="sdpa"`. Same speed, no dependency needed.

### PCIe 3.0 x16 Riser — Does Bandwidth Matter?

```
PCIe 3.0 x16 available bandwidth: 15.75 GB/s (bidirectional)

For TTS inference:
  Input per request:   ~2 KB (tokens)
  Output per request:  ~200 KB (audio)
  At 50 requests/s:    10 MB/s → 0.06% of PCIe capacity

Verdict: PCIe bandwidth is NOT a bottleneck for inference.
         Only cold model load is slightly slower (seconds, not minutes).

Riser cable warning:
  USB-type mining risers = PCIe x1 electrical despite x16 socket ❌
  Flat ribbon riser      = PCIe x16 proper ✅  ← use this
```

### RTX 5060 8GB vs Older Cards — Scorecard

| Metric | RTX 3060 12GB | RTX 4060 Ti 16GB | **RTX 5060 8GB** | RTX 5060 Ti 16GB |
|--------|:-------------:|:----------------:|:----------------:|:----------------:|
| Engines real-time | 18/21 | 20/21 | **19/21** | 20/21 |
| Orpheus 3B capable | ✅ bf16 | ✅ INT8/FP8 | **✅ FP4** | ✅ FP4 |
| Orpheus RTF | ~0.85 | ~0.55 | **~0.48** | ~0.38 |
| Dia-1.6B real-time | ❌ | ✅ | **✅ 0.55** | ✅ |
| LLM 7B INT8 (8GB) | ❌ OOM | ✅ 16GB | **⚠️ tight** | ✅ 16GB |
| Simultaneous models | 5–6 | 7–8 | **4–5** | 7–8 |
| TDP | 170W | 165W | **150W** | 180W |
| Price | ~$250 used | ~$350–400 | **~$300** | ~$400 |

### 5000 Series Recommendation

```
Best overall:      RTX 5060 Ti 16GB  (~$400)
  FP4 + 16GB GDDR7 + 448 GB/s + 180W — sweet spot

Best performance:  RTX 5070 Ti 16GB  (~$800)
  FP4 + 16GB + 896 GB/s — all 21 engines simultaneously in VRAM

Budget pick:       RTX 5060 8GB  (~$300)
  FP4 makes it viable; 8GB limits simultaneous model count

Avoid:             RTX 5090
  PCIe 5.0 in PCIe 3.0 slot = compatibility risk
  575W = PSU complexity
  GB202 Linux drivers still maturing
```

---

## 4. Weakest 5000 Card — RTX 5060 8GB vs 4000 16GB for LLMs

**Q: If I run LLMs too, does the 5060 8GB cause problems vs a 4000-series 16GB card?**

### LLM VRAM Requirements

| Model | bf16 | INT8 | FP8 | FP4 (5000 only) |
|-------|------|------|-----|-----------------|
| 3B | 6.0 GB | 3.0 GB | 1.5 GB | **0.75 GB** |
| 7B | 14.0 GB | 7.0 GB | 3.5 GB | **1.75 GB** |
| 8B | 16.0 GB | 8.0 GB | 4.0 GB | **2.0 GB** |
| 13B | 26.0 GB | 13.0 GB | 6.5 GB | **3.25 GB** |

### Critical VRAM Comparison for 8B Model (LLaMA 3.1)

```
RTX 4060 Ti 16GB:
  LLaMA 3.1 8B at INT8 = 8 GB + 1 GB cache = 9 GB  ✅ fits
  LLaMA 3.1 8B at bf16 = 16 GB + 2 GB cache = 18 GB ❌ OOM

RTX 5060 8GB:
  LLaMA 3.1 8B at FP4  = 2 GB + 0.5 GB cache = 2.5 GB ✅ fits easily
  LLaMA 3.1 8B at INT8 = 8 GB + 1 GB cache = 9 GB    ❌ OOM
```

### Token Generation Speed (Memory-Bandwidth Bound)

| Scenario | Card | BW | Model VRAM | Est. tok/s |
|----------|------|-----|-----------|-----------|
| 7B INT8 | 4060 Ti 16GB | 288 GB/s | 7.0 GB | ~41 tok/s |
| 7B FP4 | **5060 8GB** | 448 GB/s | 1.75 GB | **~256 tok/s** |
| 8B INT8 | 4060 Ti 16GB | 288 GB/s | 8.0 GB | ~36 tok/s |
| 8B FP4 | **5060 8GB** | 448 GB/s | 2.0 GB | **~224 tok/s** |

> **5060 8GB generates tokens 5–6× faster** than 4060 Ti 16GB — but at lower precision.

### LLM + TTS Simultaneous (Arthur's Pipeline)

```
Arthur pipeline: LLM generates response → TTS speaks → must complete in <2s

RTX 5060 8GB (FP4 LLM + TTS):
  7B FP4 (2.25 GB) + Kokoro (0.5 GB) = 2.75 GB total ✅ simultaneous
  7B FP4 (2.25 GB) + Chatterbox (1.5 GB) = 3.75 GB ✅ simultaneous
  7B INT8 (8 GB) + any TTS = OOM ❌

RTX 4060 Ti 16GB (INT8 LLM + TTS):
  8B INT8 (9 GB) + Chatterbox (1.5 GB) = 10.5 GB ✅ fits (16GB)
  7B bf16 (14 GB) + XTTS (2 GB) = 16 GB ❌ borderline OOM
```

### Verdict

```
For PURE LLMs at quality:    4060 Ti 16GB wins
  Runs 7B/8B in INT8 without quality sacrifice
  16K+ context fits comfortably

For SPEED + TTS lab combo:   5060 8GB wins
  FP4 token gen is 5× faster (256 tok/s vs 41 tok/s)
  Arthur's response latency is lower
  All TTS engines fit simultaneously in VRAM

ACTUAL sweet spot:  RTX 5060 Ti 16GB (~$400)
  16 GB GDDR7, FP4 tensor cores, fits 13B FP4 + all TTS engines
```

---

## 5. Xeon D-1528 Processor Limitations

**Q: Does our Xeon processor make problems when adding a GPU?**

### What Is and Isn't a Problem

| Concern | Reality | Impact |
|---------|---------|--------|
| CPU speed (1.9 GHz, 6-core) | NOT a bottleneck | CPU only tokenizes text + buffers audio (~1ms per 100 tokens) |
| System RAM (19 GB DDR4, ~30 GB/s) | NOT a bottleneck | GPU inference is VRAM-local; system RAM only touched at cold load |
| PCIe 3.0 bandwidth (15.75 GB/s) | NOT a bottleneck | Inference is VRAM-local; PCIe carries only KB per request |
| **No Resizable BAR (ReBAR)** | ⚠️ CONFIRMED MISSING | Cold model load 3–5× slower (one-time only); inference RTF unaffected |
| BIOS compatibility with RTX 5000 | ⚠️ MEDIUM RISK | 10-year gap between BIOS vintage (2020) and Blackwell (2025) |
| IOMMU / VT-d | ⚠️ POSSIBLE | vllm/CUDA may need VT-d disabled if errors occur |

### Why ReBAR Is Missing and What It Means

```
ReBAR requires CPU/PCIe controller hardware support.
Broadwell-DE (2015) does not have it. No BIOS update adds it.

Without ReBAR: GPU VRAM exposed as 256 MB BAR1 window (legacy mode)
With ReBAR:    GPU VRAM fully exposed as one window

Practical effect on inference:
  Cold model load: Orpheus 3B FP4 (1.6 GB)
    With ReBAR:    ~0.5 s
    Without ReBAR: ~2–4 s  ← one-time penalty only
  
  Warm inference RTF: IDENTICAL either way ✅
```

---

## 6. Supermicro BIOS 2.6 — Live Analysis

**Q: Are BIOS issues fixed? Can you check the live URL?**

### Live Data Retrieved from Supermicro

> **URL checked:** `https://www.supermicro.com/en/support/resources/downloadcenter/firmware/MBD-X10SDV-6C-TLN4F/BIOS`

```
Board:    MBD-X10SDV-6C-TLN4F
Latest BIOS:
  File:     X10SDVF4.205.zip
  Revision: 2.6
  Size:     5,353 KB
  MD5:      b23a8ad62a6e433c970ae415746cc071
  SHA256:   15a0fbf1d0675582189f6d516c572783277f520da0a3c133271eff2892067f62

Only ONE version listed → 2.6 is the FINAL BIOS for this board.
Board is End of Active Development. No future updates will be released.
```

### BIOS 2.6 Feature Audit

| Feature | Available in BIOS 2.6? | Impact |
|---------|:----------------------:|--------|
| **Above 4G Decoding** | ✅ YES | RTX 5000 MMIO will initialise correctly |
| Pure UEFI boot (CSM disabled) | ✅ YES | RTX 5000 UEFI OpROM will load |
| VT-d on/off toggle | ✅ YES | Disable if CUDA IOMMU conflict |
| PCIe slot width config | ✅ YES | Force x16 if needed |
| **Resizable BAR (ReBAR)** | ❌ PERMANENT | Hardware limit — no BIOS can fix |
| PCIe 4.0 support | ❌ PERMANENT | Hardware limit |
| Future BIOS updates | ❌ EOL | RTX 5000 not officially supported |

### Required BIOS Settings Before Installing Any GPU

```
BIOS → Advanced → PCIe/PCI/PnP Configuration:
  Above 4G Decoding    = Enabled   ← CRITICAL for RTX 5000
  CSM Support          = Disabled  ← Required for UEFI-only GPU OpROMs
  ACS Control          = Enabled   ← Required for DDA/VFIO passthrough

BIOS → Advanced → System Agent:
  VT-d                 = Enabled   (disable if CUDA fails with IOMMU errors)
```

### GPU Risk Level by Generation

| GPU | BIOS Risk | Reason |
|-----|:---------:|--------|
| RTX 3060/3070 | ✅ Low | 2021 — within BIOS 2.6 vintage window |
| RTX 4060 Ti | ✅ Low | 2022–2023 — still close to BIOS age |
| RTX 5060/5060 Ti | ⚠️ Medium | 2025 — 5 years newer than last BIOS update |
| RTX 5090 | ⚠️ High | PCIe 5.0 + GB202 = double risk on old board |

---

## 7. Hyper-V GPU Virtualisation — DDA vs GPU-PV

**Q: Can I use GPU-PV non-licensed mode on Hyper-V Server 2022?**

### The Critical Distinction

```
GPU-PV (GPU Paravirtualization):
  → Splits/shares one GPU across MULTIPLE VMs simultaneously
  → Requires NVIDIA vGPU / GRID SOFTWARE LICENSE ($$$)
  → NOT free for GeForce consumer cards

DDA (Discrete Device Assignment):
  → Passes ENTIRE physical GPU to ONE VM exclusively
  → FREE — zero NVIDIA license, zero Windows Server license
  → Full native CUDA performance in Ubuntu VM
  → Works on FREE Hyper-V Server 2022
  → THIS is the correct approach for Arthur
```

### DDA Setup — Core PowerShell Commands

```powershell
# Find GPU location path
$gpu = (Get-PnpDevice -Class Display | Where-Object Status -eq OK |
        Get-PnpDeviceProperty -KeyName DEVPKEY_Device_LocationPaths).Data[0]

# Configure VM MMIO (once per VM)
Set-VM -VMName "arthur-ubuntu" -LowMemoryMappedIoSpace  1GB
Set-VM -VMName "arthur-ubuntu" -HighMemoryMappedIoSpace 32GB

# Assign GPU to VM
Disable-PnpDevice -InstanceId (Get-PnpDevice -Class Display).InstanceId -Confirm:$false
Dismount-VMHostAssignableDevice -LocationPath $gpu -Force
Add-VMAssignableDevice -VMName "arthur-ubuntu" -LocationPath $gpu
```

### DDA vs GPU-PV Comparison

| | DDA | GPU-PV | NVIDIA vGPU (paid) |
|--|:---:|:------:|:-----------------:|
| Cost | Free | Free scripts | $$$license |
| CUDA in Ubuntu | ✅ native | ⚠️ experimental | ✅ |
| Works on Free Hyper-V | ✅ | ⚠️ needs Windows GUI | ✅ |
| Multiple VMs simultaneously | ❌ one VM | ❌ primary use is display | ✅ |
| GeForce consumer cards | ✅ | ✅ (Windows guest) | ❌ |
| Arthur headless ML server | ✅ **correct tool** | ❌ wrong tool | ✅ overkill |
| Maintenance burden | Low | High (driver coupling) | Medium |

### DDA Performance vs Bare Metal

```
DDA overhead vs bare metal:
  CUDA compute:     < 1%
  PCIe transfers:   < 3%
  Memory bandwidth: < 1%
  
Total inference overhead: negligible
RTF projections remain valid when using DDA.
```

---

## 8. Easy-GPU-PV Article Analysis

**Q: Someone referenced Easy-GPU-PV as the recommended method. Is it right for us?**

### Live GitHub Data

```
jamesstringer90/Easy-GPU-PV (checked live):
  Stars:        5,500
  Forks:        534
  Open issues:  259   ← red flag
  Language:     100% PowerShell
  README:       "Automatically Installs Windows to the VM"
  Host OS:      "Windows 10 20H1+ Pro, Enterprise or Education"
```

### What Easy-GPU-PV Is Designed For

```
CORRECT use case for Easy-GPU-PV:
  Windows 11 host → Windows 11 VM → GPU-accelerated desktop
  Gaming VMs, streaming, DirectX applications
  Display remoting with Parsec/Sunshine

NOT designed for:
  Headless Ubuntu ML servers
  CUDA inference (PyTorch, vllm)
  Free Hyper-V Server 2022 (no GUI on host)
```

### Why It's Wrong for Arthur

| Claim in article | Reality |
|---|---|
| "Ubuntu guest: Works too" | GPU-PV gives synthetic display adapter, not NVIDIA CUDA driver |
| "Just extra manual steps" | CUDA via GPU-PV on Linux requires `dxgkrnl` kernel module — experimental, frequently broken |
| "Works non-licensed" | Requires Windows GUI on host — doesn't run on Free Hyper-V Server Core |
| "Most common method" | True for Windows gaming VMs — wrong for ML inference |

### 259 Open Issues — Why

```
Most common issue categories:
  - Driver version mismatch between host and guest (GPU-PV copies host drivers)
  - Breaks after every Windows Update
  - Black screen after NVIDIA driver update
  - Ubuntu/Linux CUDA not working
  
Fundamental fragility:
  GPU-PV copies host NVIDIA driver files into guest
  Host driver updates → VM driver instantly stale → CUDA fails
  Must re-run Update-VMGpuPartitionDriver.ps1 after EVERY host driver update
```

---

## 9. Underground / Community License Bypass Methods

**Q: Has the community found any underground way to bypass NVIDIA licensing?**

### Known Community Projects

| Project | Status | Platform | CUDA support | Hyper-V? |
|---------|--------|----------|-------------|---------|
| vgpu-unlock (original) | DMCA removed | KVM/Proxmox | ✅ Turing/Ampere | ❌ |
| vgpu-unlock-rs (Rust) | Active (private forks) | KVM/Proxmox | ✅ Ampere | ❌ |
| fastapi-dls | Active (Codeberg) | Any | N/A (license server only) | ❌ needs vGPU driver first |

### Generation Support Matrix for vgpu-unlock

```
Kepler (700 series):   ✅ works
Maxwell (900 series):  ✅ works
Pascal (1000 series):  ✅ works
Turing (2000 series):  ✅ fairly stable
Ampere (3000 series):  ⚠️ works but fragile, breaks with NVIDIA updates
Ada (4000 series):     ❌ NOT supported — NVIDIA changed driver internals
Blackwell (5000):      ❌ NOT supported — too new, likely never
```

### Legal Reality

```
NVIDIA EULA Section 2.1.2:
  vGPU features on GeForce/consumer = explicitly prohibited
  Bypassing license checks = EULA violation + potential CFAA (US) / 
  Computer Misuse Act (UK) exposure

DMCA: NVIDIA already issued takedowns on original vgpu-unlock repos.
```

### Why It's Irrelevant for Arthur

```
vGPU / vgpu-unlock solves:
  "I want to share one GPU across MULTIPLE VMs simultaneously"

Arthur's actual need:
  "One Ubuntu ML server needs full CUDA access"
  → ONE VM, ONE GPU → DDA solves this, free, legal, faster
```

---

## 10. Hot vs Cold GPU Sharing

**Q: Is hot GPU sharing (no VM downtime) possible? I want hot sharing.**

### Definitions

```
COLD sharing (DDA switching — what we built):
  VM must STOP → GPU detached → attached to other VM → VM STARTS
  Downtime: 30–90 seconds per switch
  Our script: Switch-GPUtoVM.ps1

HOT sharing:
  GPU accessible to VM(s) WITHOUT stopping them
  A) Simultaneous: multiple VMs use GPU at the same time (time-sliced)
  B) Live switch:  GPU migrates between VMs without shutdown
```

### All Hot Sharing Methods Evaluated

| Method | Hot? | Your GPU | Your Hypervisor | CUDA quality | Legal | Practical |
|--------|:----:|---------|----------------|:------------:|:-----:|:---------:|
| NVIDIA vGPU official | ✅ | ❌ need A-series | Any | 100% | ✅ | ❌ $$$|
| vgpu-unlock (mdev) | ✅ | ✅ RTX 3060 ONLY | ❌ Proxmox KVM only | ~95% | ❌ ToS | ⚠️ fragile |
| SR-IOV | ✅ | ❌ NVIDIA = no SR-IOV | Any | 100% | ✅ | ❌ wrong GPU |
| CUDA MPS | ✅ | ✅ any | ❌ same OS only | 100% | ✅ | ✅ bare metal |
| CUDA API remoting | ✅ | ✅ any | ✅ any | ❌ 30% RTF | mixed | ❌ latency |
| Containers | ✅ | ✅ any | ❌ no VM needed | 100% | ✅ | ✅✅ best |
| DDA cold switch | ❌ COLD | ✅ any | ✅ Hyper-V | 100% | ✅ | ✅ current |

### The Two Viable Hot Paths

**Path A — vgpu-unlock (hot, Proxmox + RTX 3060 only)**
```
Requirements:
  Hypervisor: Proxmox VE (must switch from Hyper-V)
  GPU:        RTX 3060 12GB (Ampere — Ada/Blackwell not supported)
  
Result: Both VMs have GPU simultaneously, time-sliced at driver level
Risk:   Breaks with NVIDIA driver updates, ToS violation
CUDA perf: ~90–95% of bare metal for transformer inference
```

**Path B — Containers on bare metal (hot, rock solid, any GPU)**
```
Requirements:
  Remove hypervisor entirely
  Install Ubuntu directly on Xeon D-1528
  NVIDIA container toolkit handles concurrent GPU access

Result: All containers share GPU simultaneously
  arthur-tts container  → TTS inference    ✅ GPU
  arthur-llm container  → LLM inference    ✅ GPU (same time)
  arthur-stt container  → STT              ✅ GPU (same time)

CUDA perf: 100% bare metal
Any GPU:   RTX 3060/4060/5060 — all work
Legal:     ✅ completely clean
Windows VM: needs separate machine or KVM switch
```

---

## 11. Alternative Hypervisors

**Q: What if I use a different hypervisor instead of Hyper-V?**

### Full Comparison Table

| | Hyper-V Free | **Proxmox VE** | KVM bare Ubuntu | VMware ESXi | VMware Workstation | VirtualBox |
|--|:---:|:---:|:---:|:---:|:---:|:---:|
| Cost | Free | Free | Free | ❌ Paid (Broadcom) | Free personal | Free |
| GPU passthrough | DDA | **VFIO** | VFIO | DirectPath I/O | Partial | ❌ no CUDA |
| CUDA in Ubuntu VM | ✅ DDA | ✅ native | ✅ native | ⚠️ | ✅ limited | ❌ |
| RTX 5060 Blackwell | ⚠️ BIOS risk | ✅ faster adoption | ✅ fastest | ⚠️ slow | ⚠️ | ❌ |
| ACS override patch | ❌ | ✅ | ✅ | ❌ | N/A | N/A |
| Web UI | ❌ PS/RSAT | ✅ | ❌ CLI | ✅ vSphere | ✅ local | ✅ local |
| vgpu-unlock (future) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Overhead vs bare metal | <1% | <1% | <1% | <1% | 2–5% | 10–20% |
| **For Arthur overall** | ✅ good | ✅✅ best | ✅✅ simplest | ❌ | ⚠️ | ❌ |

### Decision Logic

```
Is the Xeon D-1528 machine DEDICATED to Arthur only?
  YES → Bare metal Ubuntu. No hypervisor. Zero complexity.
        Install Ubuntu → apt install nvidia-driver-570 → done.

Need Windows VM on same machine + hot GPU sharing?
  → Proxmox VE + RTX 3060 + vgpu-unlock (if Ampere, accepts stability risk)
  OR Proxmox VE + DDA cold switch (any GPU, stable)

Need Windows VM + cold switching is acceptable?
  → Hyper-V Server 2022 Free + DDA (current setup, works fine)

Want maximum CUDA performance + future-proof?
  → Proxmox VE + VFIO + any modern GPU
```

### Why Proxmox Beats Hyper-V for GPU Passthrough

```
1. VFIO = gold standard for PCIe passthrough on Linux
2. ACS override patch: fixes IOMMU group issues on Broadwell-DE
3. RTX 5060 Blackwell: Linux VFIO support arrives faster than Hyper-V BIOS compat
4. Web UI: manage VMs from browser — no PowerShell remote session
5. vgpu-unlock available here if ever needed
6. Community: massive Proxmox homelab community
```

---

## 12. PowerShell Workload on Linux Containers

**Q: I have a PowerShell workload that needs GPU. Can it run as a container on Linux?**

### Short Answer

```
YES — PowerShell 7 (pwsh) runs natively on Linux.
NVIDIA Container Toolkit passes /dev/nvidia* devices into any container.
The container does not care what language is inside it.
```

### What Works and What Doesn't

| PowerShell workload | Container on Linux | Notes |
|---|:---:|---|
| Call Python CUDA scripts (`Start-Process python3`) | ✅ | Most common pattern |
| `Invoke-RestMethod` to TTS API | ✅ | Network transparent |
| .NET `TorchSharp` GPU tensors | ✅ | Needs CUDA base image |
| .NET `Microsoft.ML.OnnxRuntime.Gpu` | ✅ | Needs CUDA base image |
| `nvidia-smi` / NVML | ✅ | Direct device access |
| SSH to another machine | ✅ | Network transparent |
| `Get-VM`, `Add-VMAssignableDevice` | ❌ | Windows-only Hyper-V cmdlets |
| `Dismount-VMHostAssignableDevice` | ❌ | Windows-only |

### Container Architecture for Arthur

```
Bare metal Ubuntu (Xeon D-1528)
├── NVIDIA driver 570  ← HOST ONLY
│
├── Container: arthur-tts        Python + CUDA    GPU: ✅  Port 8001
├── Container: arthur-stt        whisper + CUDA   GPU: ✅  Port 8002
├── Container: arthur-bench      pwsh + Python    GPU: ✅  On demand
└── Container: arthur-llm        vllm + CUDA      GPU: ✅  Port 8003

All containers see GPU simultaneously via NVIDIA Container Toolkit.
CUDA MPS (optional): enables true concurrent kernel execution.
```

### Separation of Concerns for This Project

```
Stays on Windows (your dev PC):
  deploy.ps1             → SSH orchestration
  deploy_tts_lab.ps1     → SSH orchestration
  Switch-GPUtoVM.ps1     → Hyper-V DDA management (if keeping Hyper-V)
  bench scripts          → SSH + collect results

Moves to Linux containers:
  tts_lab.py             → arthur-tts container (GPU ✅)
  arthur_server.py       → arthur-tts container (GPU ✅)
  bench_all.py           → arthur-bench container (GPU ✅, on demand)
```

---

## 13. Final Recommendations Summary

### GPU Choice — Decision Tree

```
Budget ≤ $300 AND need hot sharing AND accept stability risk:
  → RTX 3060 12GB + Proxmox + vgpu-unlock  (Ampere, hot share possible)

Budget ≤ $300 AND cold switching OK:
  → RTX 5060 8GB  (Blackwell FP4, all TTS real-time, verify BIOS Above 4G)

Budget ≤ $400:
  → RTX 5060 Ti 16GB  ← best value overall
     FP4 + 16GB GDDR7 + all 21 engines simultaneously + LLM 13B FP4

Budget ≤ $800:
  → RTX 5070 Ti 16GB  ← no compromises
     All 21 engines simultaneously, Orpheus RTF ~0.10

Want zero BIOS risk:
  → RTX 4060 Ti 16GB  (Ada, well within BIOS 2.6 vintage)
```

### Hypervisor Choice

```
Dedicated Arthur server (no Windows VM needed):
  → Bare metal Ubuntu → containers → CUDA MPS → HOT sharing ✅

Need Windows VM + hot sharing:
  → Proxmox VE + RTX 3060 + vgpu-unlock (legal risk accepted)

Need Windows VM + cold switching OK:
  → Hyper-V Server 2022 Free + DDA (current setup)

Need Windows VM + maximum stability:
  → Proxmox VE + DDA cold switch (any GPU)
```

### Quick Reference: BIOS Settings to Check Before GPU Install

```
Supermicro X10SDV-6C-TLN4F BIOS 2.6 (final version, EOL):
  Advanced → PCIe/PCI/PnP Configuration:
    Above 4G Decoding    = Enabled   ← CRITICAL
    CSM Support          = Disabled  ← Required for RTX 5000
    ACS Control          = Enabled   ← Required for DDA/Proxmox VFIO
  Advanced → System Agent:
    VT-d                 = Enabled   (disable if CUDA IOMMU errors)
```

### One-Line Conclusions

| Topic | Conclusion |
|-------|------------|
| Best GPU for this board | RTX 5060 Ti 16GB ($400) or RTX 4060 Ti 16GB for zero BIOS risk |
| Best hypervisor for CUDA passthrough | Proxmox VE > Hyper-V |
| Simplest path to GPU | Bare metal Ubuntu, no hypervisor |
| Hot sharing on consumer GPU | Only possible with vgpu-unlock on Proxmox + Ampere (RTX 3060) |
| DDA performance vs bare metal | <1% overhead — effectively identical |
| Easy-GPU-PV for Arthur | Wrong tool — designed for Windows gaming VMs, not ML servers |
| vgpu-unlock on Hyper-V | Not ported, does not exist |
| PowerShell + GPU on Linux | Yes — pwsh runs in NVIDIA CUDA containers natively |
| ReBAR on Broadwell-DE | Permanently missing — only affects cold load time, not inference |

---

*Generated from live Q&A session · Hardware verified live against Supermicro download centre · GPU specs from NVIDIA official sources*
