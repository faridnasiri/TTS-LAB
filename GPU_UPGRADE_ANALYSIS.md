# GPU Upgrade Analysis — RTX A1000 8GB vs RTX A2000 6GB
> For Arthur Server VM (Xeon D-1528, Hyper-V, Ubuntu 22.04)
> All CPU RTF figures are live-measured on the VM. GPU RTF figures are **estimates**,
> derived by scaling from known community benchmarks on comparable GPUs.
> See Section 9 for methodology.
> **Specs sourced from: https://technical.city/en/video/RTX-A2000-vs-RTX-A1000**

---

## ⚠️ Key Finding (Corrected)

**The RTX A2000 6GB is the faster card for AI inference — by ~25%.**
My earlier analysis had the bandwidth wrong. Actual figures:
- A2000: **288 GB/s** memory bandwidth (was incorrect at 192)
- A1000: **192 GB/s** memory bandwidth (was incorrect at 224)
- Both are **Ampere architecture** (A1000 is GA107, not Ada Lovelace as previously assumed)
- A2000 has **44% more Tensor Cores** (104 vs 72) and **19% more FP32 TFLOPS** (7.99 vs 6.74)

The tradeoff is now: **A2000 = faster** vs **A1000 = more VRAM + lower power + single-slot**.

---

## 1. GPU Specifications (Corrected — sourced from technical.city)

| Spec | **RTX A2000 6 GB** | **RTX A1000 8 GB** | CPU baseline |
|---|---|---|---|
| **Architecture** | Ampere (GA106) | Ampere (GA107) | Broadwell-DE |
| **Generation** | Same — both 8 nm Ampere | Same — both 8 nm Ampere | 2015 |
| **Release date** | 10 Aug 2021 | 16 Apr 2024 | — |
| **CUDA cores** | — | — | — |
| **Tensor Cores** | **104** (3rd gen) | 72 (3rd gen) | — |
| **RT Cores** | **26** | 18 | — |
| **TMUs** | **104** | 72 | — |
| **ROPs** | **48** | 32 | — |
| **FP32 throughput** | **7.99 TFLOPS** | 6.74 TFLOPS | ~0.2 TFLOPS |
| **FP16 Tensor (est)** | **~16 TFLOPS** | ~10.7 TFLOPS | — |
| **VRAM** | 6 GB GDDR6 | **8 GB GDDR6** | 32 GB DDR4 |
| **Memory bus** | **192-bit** | 128-bit | 128-bit (2ch) |
| **Memory bandwidth** | **288 GB/s** ← faster | 192 GB/s | ~25 GB/s |
| **Memory clock** | 1500 MHz | 1500 MHz | — |
| **L1 Cache** | **3.3 MB** | 2.3 MB | — |
| **L2 Cache** | **3 MB** | 2 MB | — |
| **TDP** | 70 W | **50 W** ← lower | — |
| **Form factor** | Low-profile **dual slot** | Low-profile **single slot** | — |
| **PCIe** | Gen 4 | Gen 4 | — |
| **Transistors** | **12 000M** | 8 700M | — |
| **Manufacturing** | 8 nm | 8 nm | — |
| **Hyper-V DDA** | ✅ | ✅ | — |
| **Launch price** | **$449** | no data | — |
| **Aggregate score** | **32.12** (+24.6%) | 25.78 | — |
| **GeekBench 5 OpenCL** | **73 415** (+37.3%) | 53 482 | — |
| **GeekBench 5 Vulkan** | **69 653** (+38.6%) | 50 266 | — |
| **Passmark** | **13 427** (+24.5%) | 10 786 | — |

### The bandwidth story — why A2000 wins inference
Memory bandwidth is the dominant bottleneck for neural network inference (loading
weights each forward pass). The wider 192-bit bus on the A2000 gives it a 50%
bandwidth advantage over the A1000's 128-bit bus, despite both using the same
1500 MHz GDDR6 memory.

```
A2000: 192-bit × 2 × 1500 MHz × 2 (DDR) = 288 GB/s
A1000: 128-bit × 2 × 1500 MHz × 2 (DDR) = 192 GB/s
Ratio: 288 / 192 = 1.50 — A2000 is 50% more bandwidth
```

---

## 2. Master RTF Comparison — All 10 Models (Corrected)

Scale factors from RTX 3060 (360 GB/s, Ampere, community benchmarks available):
- A2000: 360 / 288 = **×1.25** slower than RTX 3060
- A1000: 360 / 192 = **×1.875** slower than RTX 3060

Both Ampere → no architecture correction factor needed.

| Engine | CPU RTF | **A2000 6 GB** | **A1000 8 GB** | Speedup A2000 | Speedup A1000 | Real-time A2000? | Real-time A1000? |
|---|---|---|---|---|---|---|---|
| **Piper** | 0.08× | ~0.006× | ~0.009× | ~13× | ~9× | ✅✅ | ✅✅ |
| **MeloTTS** | 1.08× | ~0.038× | ~0.056× | ~28× | ~19× | ✅✅ | ✅✅ |
| **StyleTTS 2** | 1.67× | ~0.040× | ~0.060× | ~42× | ~28× | ✅✅ | ✅✅ |
| **Kokoro-82M** | 3.07× | ~0.063× | ~0.094× | ~49× | ~33× | ✅✅ | ✅✅ |
| **XTTS-v2** | 4.74× | ~0.19× | ~0.28× | ~25× | ~17× | ✅ | ✅ |
| **F5-TTS** | ~5× | ~0.13× | ~0.19× | ~38× | ~26× | ✅ | ✅ |
| **Chatterbox** | 11.7× | ~0.25× | ~0.38× | ~47× | ~31× | ✅ | ✅ |
| **Bark** | 20.3× | ~0.63× | ~0.94× | ~32× | ~22× | ✅ | **⚠️ marginal** |
| **Parler-TTS** | 23.4× | ~0.31× | ~0.47× | ~75× | ~50× | ✅ | ✅ |
| **Dia-1.6B** | ~55× | **~1.9×** | **~2.9×** | ~29× | ~19× | ❌ | ❌ |

> **RTF < 1.0 = real-time.** All models except Dia become real-time on both cards.
> Bark on A1000 is ~0.94× — technically real-time but almost no margin; stutter risk on long utterances.
> Bark on A2000 is ~0.63× — comfortable.

---

## 3. VRAM Fit Analysis

Models use FP16 on GPU — half the CPU FP32 weight size.

| Engine | FP32 RAM (CPU) | FP16 VRAM (weights) | + Activations | Total VRAM | A2000 6 GB (5.5 usable) | A1000 8 GB (7.5 usable) |
|---|---|---|---|---|---|---|
| **Piper** | 200 MB | ~100 MB | ~50 MB | ~150 MB | ✅ | ✅ |
| **Kokoro** | 500 MB | ~250 MB | ~100 MB | ~350 MB | ✅ | ✅ |
| **MeloTTS** | 1 200 MB | ~600 MB | ~200 MB | ~800 MB | ✅ | ✅ |
| **StyleTTS 2** | 1 500 MB | ~750 MB | ~250 MB | ~1 000 MB | ✅ | ✅ |
| **F5-TTS** | 2 000 MB | ~1 000 MB | ~300 MB | ~1 300 MB | ✅ | ✅ |
| **Bark** | 1 500 MB | ~750 MB | ~400 MB | ~1 150 MB | ✅ | ✅ |
| **Chatterbox** | 1 800 MB | ~900 MB | ~300 MB | ~1 200 MB | ✅ | ✅ |
| **Parler-TTS** | 1 500 MB | ~750 MB | ~350 MB | ~1 100 MB | ✅ | ✅ |
| **XTTS-v2** | 3 200 MB | ~1 600 MB | ~400 MB | ~2 000 MB | ✅ | ✅ |
| **Dia-1.6B** | 3 000+ MB | ~3 200 MB | ~600 MB | ~3 800 MB | ✅ ⚠️ tight | ✅ comfortable |
| **All 9 non-Dia** | — | ~6 200 MB | ~2 000 MB | ~8 200 MB | ❌ overflow | ⚠️ tight |
| **8 small models** | — | ~4 500 MB | ~1 500 MB | ~6 000 MB | ❌ overflow | ✅ |
| **4 lightest models** | — | ~1 700 MB | ~600 MB | ~2 300 MB | ✅ | ✅ |

### VRAM strategy summary

**A2000 6 GB (5.5 GB usable after driver/OS overhead):**
- Persistent: Piper + Kokoro + MeloTTS + StyleTTS2 → ~2.3 GB ✅
- + Chatterbox or Bark or Parler or F5 → ~3.4–3.5 GB ✅
- + XTTS-v2 → ~4.3 GB ✅
- + Dia-1.6B → ~6.1 GB ❌ over budget (must evict everything else first)
- **Load Dia alone on demand** — evict all others before loading

**A1000 8 GB (7.5 GB usable):**
- Persistent: Piper + Kokoro + MeloTTS + StyleTTS2 → ~2.3 GB ✅
- All 9 non-Dia models → ~8.2 GB ❌ slightly over
- 8 smallest models simultaneously → ~6.0 GB ✅
- Dia alone → ~3.8 GB ✅
- Dia + persistent 4 lightweight → ~6.1 GB ✅
- **Dia can coexist with the 4 lightweight models** — A1000 exclusive advantage

---

## 4. Dia-1.6B — Why It's Still Not Real-Time

```
DAC codec @ 44 100 Hz with 512 samples/frame = 86 audio frames per second
Dia model (FP16): 1.6B params × 2 bytes = 3.2 GB

Effective GPU bandwidth (50% practical efficiency):
  A2000: 288 GB/s × 0.50 = 144 GB/s
  A1000: 192 GB/s × 0.50 = 96 GB/s

Forward passes achievable per second:
  A2000: 144 / 3.2 = 45 passes/sec  →  RTF = 86 / 45 = 1.9×  (too slow)
  A1000: 192 / 3.2 = 30 passes/sec  →  RTF = 86 / 30 = 2.9×  (too slow)

Minimum bandwidth needed for RTF = 1.0:
  86 frames × 3.2 GB × (1/0.50) = 550 GB/s theoretical needed

Cards that can achieve RTF < 1.0 for Dia:
  RTX 3090:  936 GB/s  → ~0.6× RTF  ✅
  RTX 4080:  736 GB/s  → ~0.8× RTF  ✅
  RTX 4090: 1 008 GB/s → ~0.5× RTF  ✅
  RTX A6000:  768 GB/s → ~0.7× RTF  ✅
```

Dia is only for offline dialogue generation on these cards — not live phone calls.

---

## 5. Bark on A1000 — Marginal Case

```
Bark runs 3 serial autoregressive stages:
  text encoder → coarse codec AR → fine codec AR → EnCodec decoder

Combined model size (FP16 small models): ~750 MB
Effective bandwidth: 192 × 0.50 = 96 GB/s
Passes per second: 96 / 0.75 = 128 passes/sec

EnCodec audio token rate (24 kHz / 320): ~75 tokens/sec × 3 stages = ~225 effective
RTF ≈ 225 / 128 = ~1.75× per stage... but stages run serially

Community benchmarks suggest Bark on RTX 3060 = ~0.5× RTF.
A1000 estimate: 0.5 × (360/192) = ~0.94×  ← within 6% of real-time.

In practice this means:
  - Short phrases (<5s): fine, GPU completes before buffer runs out
  - Long phrases (>8s): risk of audio stutter or slight delay
  - With Bark's natural pauses in text, actual heard delay may be acceptable
```

For live phone calls with Bark on A1000: **generate in short sentences ≤ 5s each** to stay safe.

---

## 6. Side-by-Side Decision Matrix

| Criterion | **RTX A2000 6 GB** | **RTX A1000 8 GB** | Winner |
|---|---|---|---|
| **Street price (2025)** | ~$280–380 | ~$350–480 | A2000 |
| **Aggregate benchmark** | 32.12 **(+24.6%)** | 25.78 | **A2000** |
| **Memory bandwidth** | **288 GB/s** | 192 GB/s | **A2000** |
| **FP32 throughput** | **7.99 TFLOPS** | 6.74 TFLOPS | **A2000** |
| **Tensor Cores** | **104** | 72 | **A2000** |
| **VRAM** | 6 GB | **8 GB** | A1000 |
| **Power draw** | 70 W | **50 W** | A1000 |
| **Form factor** | Dual slot | **Single slot** | A1000 |
| **Bark RTF** | ~0.63× ✅ safe | ~0.94× ⚠️ marginal | **A2000** |
| **Dia-1.6B RTF** | ~1.9× ❌ | ~2.9× ❌ | **A2000** (less bad) |
| **All other RTF** | ✅ all real-time | ✅ all real-time | A2000 faster |
| **Dia + small models resident** | ❌ won't fit | ✅ fits | A1000 |
| **All 9 models resident** | ❌ won't fit | ❌ barely over | Tie |
| **Architecture** | Ampere GA106 | Ampere GA107 | Same generation |

### Recommendations by use case

**Buy A2000 6GB if:**
- Performance is the priority (25% faster inference across all models)
- You have dual-slot PCIe space available in the Hyper-V host
- 70W TDP is acceptable
- You won't need Dia and multiple other models resident simultaneously

**Buy A1000 8GB if:**
- Physical space is critical (single-slot server chassis)
- Power budget is tight (50W vs 70W)
- You want Dia-1.6B loaded alongside other models without GPU eviction
- You're willing to accept ~25% slower inference everywhere and marginal Bark performance

**Bottom line:** For the Arthur scam-baiter use case, **A2000 is the better TTS card** —
faster inference on all models, Bark becomes genuinely safe for live calls,
and 6 GB is enough VRAM since Dia is never used in live calls anyway.

---

## 7. Bandwidth-to-CPU Speedup Summary

| GPU | Bandwidth | Vs CPU (25 GB/s) | Practical FP16 speedup over CPU |
|---|---|---|---|
| Xeon D-1528 (CPU, FP32) | ~25 GB/s | 1× | baseline |
| **RTX A1000 8 GB** | 192 GB/s | 7.7× raw | ~15× (FP16 + parallelism) |
| **RTX A2000 6 GB** | 288 GB/s | 11.5× raw | ~22× (FP16 + parallelism) |
| RTX 3060 12 GB | 360 GB/s | 14.4× raw | ~28× |
| RTX 3080 | 760 GB/s | 30.4× raw | ~60× |
| RTX 4090 | 1 008 GB/s | 40.3× raw | ~80× |

FP16 on GPU loads weights at half the bytes vs CPU FP32, effectively doubling
the practical bandwidth advantage. Parallelism factor adds another 1.5× for
feedforward models (Piper, Kokoro, StyleTTS2).

---

## 8. Hyper-V DDA (GPU Passthrough) Setup

Both cards support Hyper-V Discrete Device Assignment (DDA).

```powershell
# On Hyper-V HOST (Windows) — run as Administrator

# 1. Find the GPU PCI location path
$gpu = Get-PnpDevice | Where-Object { $_.FriendlyName -like "*NVIDIA*RTX*" }
$pci = (Get-PnpDeviceProperty -InstanceId $gpu.InstanceId `
    -KeyName DEVPKEY_Device_LocationInfo).Data

# 2. Dismount from host (makes it assignable to VM)
Dismount-VMHostAssignableDevice -LocationPath $pci -Force

# 3. Assign to arthur VM
Add-VMAssignableDevice -VMName "arthur" -LocationPath $pci

# 4. Set MMIO space (required for GPU passthrough)
Set-VM -Name "arthur" -LowMemoryMappedIoSpace 3GB -HighMemoryMappedIoSpace 32GB

# To revert
Remove-VMAssignableDevice -VMName "arthur" -LocationPath $pci
Mount-VMHostAssignableDevice -LocationPath $pci
```

```bash
# On VM (Ubuntu 22.04) — after DDA assignment

# Install NVIDIA driver
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-545
sudo reboot

# Verify
nvidia-smi
# Expect: RTX A2000 or RTX A1000 visible with VRAM shown

# Install PyTorch CUDA build (replace current CPU-only build)
source /opt/arthur-bench-env/bin/activate
pip uninstall torch torchaudio -y
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install OnnxRuntime GPU (for Piper + Kokoro)
pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.20.0

# Verify GPU visible to PyTorch
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True  NVIDIA RTX A2000
```

**Code changes needed in `tts_lab.py`:**
```python
# Add near top (after import torch)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Piper + Kokoro OnnxRuntime — add CUDA provider
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# pass providers= to Kokoro() constructor

# All PyTorch models — add .to(DEVICE) after loading
model = model.to(DEVICE)

# Chatterbox
ChatterboxTTS.from_pretrained(device=DEVICE)

# Bark
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # keep on GPU

# XTTS
# TTS() auto-detects GPU if CUDA is available — no code change needed
```

---

## 9. Estimation Methodology

```
Reference GPU: NVIDIA RTX 3060 12GB (Ampere, 360 GB/s, ~12 TFLOPS FP32)
Source: Community benchmarks for XTTS, Bark, Parler, Chatterbox on RTX 3060.

RTX 3060 baseline RTF (community benchmarks):
  XTTS-v2:    0.15×  Bark:      0.50×  Parler:    0.25×
  Chatterbox: 0.20×  F5-TTS:    0.10×  Piper:     0.005×
  Kokoro:     0.05×  MeloTTS:   0.03×  StyleTTS2: 0.03×
  Dia-1.6B:   1.50×  (calculated from bandwidth model)

Scale factors — both A2000 and A1000 are Ampere (same architecture as RTX 3060):
  RTX A2000:  RTF = RTX3060_RTF × (360/288) = × 1.25
  RTX A1000:  RTF = RTX3060_RTF × (360/192) = × 1.875

GPU specs sourced from: https://technical.city/en/video/RTX-A2000-vs-RTX-A1000
  A2000: 288 GB/s, 7.99 TFLOPS FP32, 104 Tensor Cores, GA106, 8 nm
  A1000: 192 GB/s, 6.74 TFLOPS FP32,  72 Tensor Cores, GA107, 8 nm

Uncertainty: ±25%. Actual numbers: run bench_warm.py after GPU install.
```

---

## 10. After GPU Install — Verify & Benchmark

```bash
# GPU health check
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//1024//1024, 'MB')"

# Quick per-model RTF check (replace 'chatterbox' with each model name)
curl -X POST http://localhost:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"Oh my goodness, just a moment dear, let me find my glasses.","params":{"exaggeration":0.6}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('RTF='+str(round(d['rtf'],3)))"

# Full warm benchmark (all models)
/opt/arthur-bench-env/bin/python3 /opt/arthur/bench_warm.py

# VRAM usage after loading all models
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```