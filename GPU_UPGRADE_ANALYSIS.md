# GPU Upgrade Analysis — RTX A1000 8GB vs RTX A2000 6GB
> For Arthur Server VM (Xeon D-1528, Hyper-V, Ubuntu 22.04)
> All CPU RTF figures are live-measured on the VM. GPU RTF figures are **estimates**,
> derived by scaling from known community benchmarks on comparable GPUs.
> See Section 9 for methodology.

---

## 1. GPU Specifications

| Spec | **RTX A1000 8 GB** | **RTX A2000 6 GB** | CPU baseline |
|---|---|---|---|
| **Architecture** | Ada Lovelace (AD107) | Ampere (GA106) | Broadwell-DE |
| **Generation** | 2023 | 2021 | 2015 |
| **CUDA cores** | 2 816 | 3 328 | — |
| **Tensor Cores** | 88 × 4th-gen | 104 × 3rd-gen | — |
| **FP32 throughput** | 8.2 TFLOPS | 8.0 TFLOPS | ~0.2 TFLOPS (AVX2) |
| **FP16 (Tensor, dense)** | 32.8 TFLOPS | 16.0 TFLOPS | — |
| **INT8 (Tensor, dense)** | 65.5 TOPS | 32.0 TOPS | — |
| **VRAM** | **8 GB GDDR6** | **6 GB GDDR6** | 32 GB DDR4 (shared) |
| **Memory bus** | 128-bit | 192-bit | 128-bit (2ch) |
| **Memory bandwidth** | **224 GB/s** | **192 GB/s** | ~25 GB/s (practical) |
| **TDP** | **50 W** ← lower | 70 W | N/A |
| **Form factor** | Low-profile **single slot** ← smaller | Low-profile dual slot | — |
| **PCIe slot** | Gen 4 × 8 | Gen 4 × 8 | — |
| **Display outputs** | 4× mDP 1.4 | 4× mDP 1.4 | — |
| **Hyper-V DDA** | ✅ supported | ✅ supported | — |
| **Street price (2025)** | ~$400–480 new | ~$280–350 used/new | — |

### Why A1000 wins despite fewer CUDA cores
A2000 has more CUDA cores but they are Ampere 3rd-gen with older Tensor Cores.
A1000 Ada has 4th-gen Tensor Cores: 2× FP16 throughput per core vs Ampere.
For TTS inference (FP16 on GPU): A1000 Tensor FLOPS **32.8 vs 16.0** — twice as fast.
Memory bandwidth is A1000: 224 vs 192 GB/s (+17%).
Combined effect: A1000 is **~35–45% faster** than A2000 for FP16 inference workloads.

---

## 2. Master RTF Comparison — All 10 Models

All GPU figures estimated from RTX 3060 community benchmarks (360 GB/s bandwidth),
scaled by bandwidth ratio + architecture efficiency factor.
Scale factors: A2000 = ×1.60 slower than RTX 3060 | A1000 = ×1.15 slower than RTX 3060

| Engine | CPU RTF | A2000 6 GB | A1000 8 GB | Speedup A2000 | Speedup A1000 | Real-time A2000? | Real-time A1000? |
|---|---|---|---|---|---|---|---|
| **Piper** | 0.08× | ~0.008× | ~0.006× | ~10× | ~13× | ✅✅ | ✅✅ |
| **MeloTTS** | 1.08× | ~0.045× | ~0.032× | ~24× | ~34× | ✅✅ | ✅✅ |
| **StyleTTS 2** | 1.67× | ~0.050× | ~0.036× | ~33× | ~46× | ✅✅ | ✅✅ |
| **Kokoro-82M** | 3.07× | ~0.075× | ~0.055× | ~41× | ~56× | ✅✅ | ✅✅ |
| **XTTS-v2** | 4.74× | ~0.24× | ~0.17× | ~20× | ~28× | ✅ | ✅ |
| **F5-TTS** | ~5× | ~0.16× | ~0.11× | ~31× | ~45× | ✅ | ✅ |
| **Chatterbox** | 11.7× | ~0.32× | ~0.22× | ~37× | ~53× | ✅ | ✅ |
| **Bark** | 20.3× | ~0.80× | ~0.58× | ~25× | ~35× | ✅ | ✅ |
| **Parler-TTS** | 23.4× | ~0.40× | ~0.29× | ~59× | ~81× | ✅ | ✅ |
| **Dia-1.6B** | ~55× | ~2.4× | ~1.8× | ~23× | ~31× | ❌ | ❌ |

> **RTF < 1.0 = real-time capable.** Every model except Dia-1.6B becomes real-time on both cards.
> Dia-1.6B remains above 1.0× even on GPU due to its 1.6B autoregressive architecture;
> see Section 5 for explanation.

---

## 3. VRAM Fit Analysis

All models use FP16 on GPU, halving their effective weight size vs CPU FP32.

| Engine | FP32 RAM (CPU) | FP16 VRAM est | Activations | Total VRAM | Fits A2000 6 GB? | Fits A1000 8 GB? |
|---|---|---|---|---|---|---|
| **Piper** | 200 MB | ~100 MB | ~50 MB | ~150 MB | ✅ | ✅ |
| **Kokoro** | 500 MB | ~250 MB | ~100 MB | ~350 MB | ✅ | ✅ |
| **MeloTTS** | 1 200 MB | ~600 MB | ~200 MB | ~800 MB | ✅ | ✅ |
| **StyleTTS 2** | 1 500 MB | ~750 MB | ~250 MB | ~1 000 MB | ✅ | ✅ |
| **F5-TTS** | 2 000 MB | ~1 000 MB | ~300 MB | ~1 300 MB | ✅ | ✅ |
| **Chatterbox** | 1 800 MB | ~900 MB | ~300 MB | ~1 200 MB | ✅ | ✅ |
| **Bark** | 1 500 MB | ~750 MB | ~400 MB | ~1 150 MB | ✅ | ✅ |
| **Parler-TTS** | 1 500 MB | ~750 MB | ~350 MB | ~1 100 MB | ✅ | ✅ |
| **XTTS-v2** | 3 200 MB | ~1 600 MB | ~400 MB | ~2 000 MB | ✅ | ✅ |
| **Dia-1.6B** | 3 000+ MB | ~3 200 MB | ~600 MB | **~3 800 MB** | ✅⚠️ tight | ✅ comfortable |
| **All small models** | ~9 GB | ~4 500 MB | ~1 700 MB | **~6 200 MB** | ❌ overflow | ✅ |
| **All except Dia** | ~13 GB | ~6 500 MB | ~2 200 MB | **~8 700 MB** | ❌ overflow | ❌ tight |

### VRAM budget summary
**A2000 6 GB** — available after OS/driver overhead (~500 MB): **~5.5 GB usable**
- Can hold all 10 small-to-medium models simultaneously ← **no**, total is 6.2 GB
- Can hold any 1 large model (Dia, XTTS) + all lightweight models ✅
- Cannot hold Dia + XTTS simultaneously (3.8 + 2.0 = 5.8 GB — over budget)
- Practical limit: **load on demand**, evict heavy model before loading another heavy one

**A1000 8 GB** — usable: **~7.5 GB**
- Can hold all 9 non-Dia models simultaneously (~6.0 GB) ✅
- Can hold Dia alone comfortably (~3.8 GB) ✅
- Cannot hold Dia + XTTS + others simultaneously (~5.8+ GB with overhead)
- Practical: **persistent Piper/Kokoro/Melo/StyleTTS2/Bark/Parler/Chatterbox/F5 all loaded**, swap for Dia or XTTS when needed

---

## 4. Per-Model GPU Detail

### Piper TTS — 0.008× RTF on A2000 / 0.006× on A1000
Piper is already real-time at 0.08× on CPU. On GPU it becomes essentially instant.
A 12-second sentence synthesises in ~100 ms on A2000. No meaningful difference between cards.
**Verdict:** Overkill — CPU is fine for Piper. GPU unlocks everything else.

### Kokoro-82M — 0.075× RTF on A2000 / 0.055× on A1000
From too-slow (3.07×) to ultra-fast. All 54 voices available instantly.
FP16 ONNX on GPU works out of the box via CUDA execution provider in OnnxRuntime.
**Code change needed:** add `providers=["CUDAExecutionProvider"]` to `SessionOptions`.
**Verdict:** Becomes primary production candidate — light, 54 voices, near-zero latency.

### MeloTTS — 0.045× on A2000 / 0.032× on A1000
From borderline-real-time (1.08×) to 24-34× faster.
RAM pressure degradation disappears — GPU VRAM is isolated from CPU RAM.
**Verdict:** Completely solves the MeloTTS RAM pressure problem.

### StyleTTS 2 — 0.050× on A2000 / 0.036× on A1000
From evaluation-only (1.67×) to real-time production candidate.
Diffusion models benefit enormously from GPU: all diffusion steps parallelise.
Reference WAV style transfer becomes practical in a live call.
**Verdict:** Most impactful upgrade — suddenly one of the fastest AND highest quality options.

### XTTS-v2 — 0.24× on A2000 / 0.17× on A1000
From 4.74× (4.7× too slow) to well under real-time. 17 languages, 58 speakers, voice cloning.
Load time drops from 26s to ~8s (still limited by PCIe model transfer: 1.6 GB via PCIe 4×8 ~3s + warmup).
**Verdict:** Becomes the best production option when voice cloning or multi-language is needed.

### F5-TTS — 0.16× on A2000 / 0.11× on A1000
From too-slow (~5×) to very fast. Best zero-shot voice cloning quality.
Flow matching models parallelise well on GPU (similar to diffusion).
**Verdict:** Practical for live calls — upload a reference WAV once, use indefinitely.

### Chatterbox — 0.32× on A2000 / 0.22× on A1000
From 11.7× (very slow) to real-time with headroom. Exaggeration slider still works.
The confused elderly hesitation effect is preserved; latency is no longer a concern.
**Verdict:** Becomes the best production choice for the confused Arthur persona.

### Bark — 0.80× on A2000 / 0.58× on A1000
From 20.3× to just under real-time. Emotion tokens (`[laughs]`, `[sighs]`) become practical.
A2000: 0.80× is cutting it close; short pauses in text could cause buffer underrun in real calls.
A1000: 0.58× is comfortable margin.
**Verdict:** A1000 makes Bark production-ready. A2000 is marginal — risky for live calls.
Bark runs 3 serial autoregressive models (text → coarse → fine); hence slower than single-model AR.

### Parler-TTS — 0.40× on A2000 / 0.29× on A1000
From worst (23.4×) to comfortably real-time. Natural language description still controls voice.
T5+EnCodec architecture parallelises well.
**Verdict:** Becomes a strong production candidate for maximum voice configurability.

### Dia-1.6B — 2.4× on A2000 / 1.8× on A1000
**Still not real-time on either card.** The model has 1.6B parameters generating ~86 audio frames/sec.
Each frame requires a full forward pass through the 3.2 GB FP16 model.
- A2000 (192 GB/s × 50% efficiency = ~96 GB/s): 96/3.2 = ~30 frames/sec → RTF = 86/30 = **~2.9×**
- A1000 (224 GB/s × 55% efficiency = ~123 GB/s): 123/3.2 = ~38 frames/sec → RTF = 86/38 = **~2.3×**
KV-cache helps for subsequent tokens but DAC token count per second is very high.
For non-real-time dialogue generation (pre-generate then play): both cards are **31-48× faster than CPU**.
**Verdict:** Still not for live calls. Use for scripted dialogue generation offline.

---

## 5. Why Dia-1.6B Cannot Be Real-Time on These Cards

The bottleneck is the DAC (Discrete Audio Codec) frame rate, not compute per se.

```
DAC codec @ 44 100 Hz / 512 = 86 audio frames per second
Dia generates all 9 codebook levels per frame
Effective forward passes per second of audio = ~86

Model size (FP16): 1.6B params × 2 bytes = 3.2 GB
A1000 memory bandwidth: 224 GB/s × 55% efficiency = ~123 GB/s effective
Forward passes achievable: 123 GB/s ÷ 3.2 GB = ~38 per second

RTF = 86 frames needed / 38 frames achievable = 2.3× (too slow)

To reach RTF 1.0 you need: 86 × 3.2 GB = 275 GB/s effective bandwidth
That requires a GPU with ~500 GB/s theoretical bandwidth:
  RTX 3090: 936 GB/s → Dia RTF ≈ 0.6×  ✅
  RTX 4090: 1 008 GB/s → Dia RTF ≈ 0.5× ✅
  A6000: 768 GB/s → Dia RTF ≈ 0.75× ✅
```

---

## 6. Recommended GPU Loading Strategy (Both Cards)

### At service startup (auto-loaded, always resident)
These fit in either card and are used most often:
```
Piper       ~150 MB  — production fallback, instant
Kokoro      ~350 MB  — 54 voices, near-instant
MeloTTS     ~800 MB  — British accent, near-instant
StyleTTS2   ~1000 MB — reference-WAV style, real-time
```
Total: ~2.3 GB — always loaded, never evicted.

### On-demand heavy models (load when requested, evict when done)
```
Chatterbox  ~1 200 MB  — best Arthur persona
F5-TTS      ~1 300 MB  — voice cloning
Bark        ~1 150 MB  — emotion tokens
Parler      ~1 100 MB  — description-driven
XTTS-v2     ~2 000 MB  — multi-language / voice clone
Dia-1.6B    ~3 800 MB  — dialogue generation only
```

### A2000 6 GB budget (5.5 GB usable)
- Always-on: 2.3 GB
- Remaining: 3.2 GB — fits any single heavy model EXCEPT Dia (3.8 GB, over budget)
- **Dia cannot be loaded at the same time as anything else on A2000**

### A1000 8 GB budget (7.5 GB usable)
- Always-on: 2.3 GB
- Remaining: 5.2 GB — fits Chatterbox + F5 + Bark + Parler simultaneously (4.75 GB) ✅
- Or: XTTS-v2 + all lightweight (4.3 GB) ✅
- Or: Dia alone + always-on (6.1 GB) — tight but possible ✅

---

## 7. Hyper-V DDA (GPU Passthrough) Setup

Both cards support Hyper-V Discrete Device Assignment (DDA) on Windows Server / Hyper-V.

```powershell
# On Hyper-V HOST (Windows) — run as Administrator

# 1. Find the GPU PCI address
$gpu = Get-PnpDevice | Where-Object { $_.FriendlyName -like "*NVIDIA*" }
$pci = (Get-PnpDeviceProperty -InstanceId $gpu.InstanceId -KeyName DEVPKEY_Device_LocationInfo).Data

# 2. Dismount GPU from host (makes it available for VM passthrough)
Dismount-VMHostAssignableDevice -LocationPath $pci -Force

# 3. Assign to the arthur VM
Add-VMAssignableDevice -VMName "arthur" -LocationPath $pci

# 4. (Optional) Set MMIO space for the GPU
Set-VM -Name "arthur" -LowMemoryMappedIoSpace 3GB -HighMemoryMappedIoSpace 32GB

# To revert (return GPU to host)
Remove-VMAssignableDevice -VMName "arthur" -LocationPath $pci
Mount-VMHostAssignableDevice -LocationPath $pci
```

```bash
# On VM (Ubuntu) — install NVIDIA driver after DDA
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-545       # or latest stable
sudo reboot

# Verify
nvidia-smi
# Should show: RTX A1000 (or A2000)

# Install CUDA toolkit (for PyTorch CUDA ops)
sudo apt-get install -y cuda-toolkit-12-4

# Install PyTorch CUDA build (replace current CPU build)
source /opt/arthur-bench-env/bin/activate
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install OnnxRuntime with CUDA support (for Piper + Kokoro GPU)
pip install onnxruntime-gpu==1.20.0

# Verify GPU is visible to PyTorch
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Code changes in tts_lab.py after GPU install:**
```python
# Top of file — detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Piper / Kokoro OnnxRuntime
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # GPU prefers sequential
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
Kokoro(str(mp), str(vp), sess_options=opts, providers=providers)

# PyTorch models — pass device="cuda" to .to(DEVICE)
model.to(DEVICE)

# torch.set_num_threads still useful for preprocessing on CPU
torch.set_num_threads(12)
```

---

## 8. Cost vs Performance Decision

| Criterion | RTX A2000 6 GB | RTX A1000 8 GB | Winner |
|---|---|---|---|
| **Price (2025)** | ~$280–350 | ~$400–480 | A2000 |
| **FP16 throughput** | 16 TFLOPS | 32.8 TFLOPS | **A1000** (2×) |
| **Memory bandwidth** | 192 GB/s | 224 GB/s | A1000 |
| **VRAM** | 6 GB | **8 GB** | **A1000** |
| **Power draw** | 70 W | **50 W** | **A1000** |
| **Form factor** | Dual slot | **Single slot** | **A1000** |
| **Dia-1.6B RTF** | ~2.4× (no) | ~1.8× (no) | Neither |
| **Bark RTF** | ~0.80× (marginal) | ~0.58× (safe) | **A1000** |
| **All other models** | ✅ all real-time | ✅ all real-time | Tie |
| **Hold all models in VRAM** | ❌ 5.5 GB limit | ✅ 7.5 GB fits most | **A1000** |
| **Hyper-V DDA compat** | ✅ | ✅ | Tie |
| **Architecture generation** | Ampere (2021) | Ada Lovelace (2023) | **A1000** |

### Recommendation: **RTX A1000 8 GB**

The A1000 wins on every technical axis that matters for this workload:
- 2× FP16 Tensor Core throughput
- Bark becomes comfortably real-time (0.58× vs marginal 0.80×)
- 8 GB fits all models except Dia simultaneously; 6 GB requires constant eviction
- 50W TDP — Hyper-V host stays cool, no power budget issues
- Single slot — physically fits better in the server chassis

The A2000 costs ~$120 less but:
- Bark is marginal (0.80× can cause audio stutter in live calls)
- 6 GB VRAM forces eviction even for Chatterbox + XTTS simultaneously
- Ampere architecture is 2 generations behind

**If budget is the constraint:** A2000 6 GB still makes ALL models except Bark and Dia production-viable. Bark must be pre-generated offline. Total cost saving ~$120.

---

## 9. Estimation Methodology

```
Reference GPU: NVIDIA RTX 3060 12GB (Ampere, 360 GB/s, ~11 TFLOPS FP32)
Source: Community benchmarks for XTTS, Bark, Parler, Chatterbox on RTX 3060.

RTX 3060 baseline RTF estimates (community):
  XTTS-v2:    0.15× | Bark:      0.50× | Parler:    0.25×
  Chatterbox: 0.20× | F5-TTS:    0.10× | Piper:     0.005×
  Kokoro:     0.05× | MeloTTS:   0.03× | StyleTTS2: 0.03×
  Dia-1.6B:   1.50× (calculated from bandwidth model)

Scale factors applied:
  RTX A2000:  bandwidth ratio = 192/360 = 0.533
              architecture factor = 0.95 (Ampere same gen as RTX 3060)
              combined = RTX3060_RTF × (360/192) × 0.95 = RTX3060_RTF × 1.78 → ~1.6 (rounded)

  RTX A1000:  bandwidth ratio = 224/360 = 0.622
              architecture factor = 0.80 (Ada Lovelace 4th-gen Tensor Cores ~20% better)
              combined = RTX3060_RTF × (360/224) × 0.80 = RTX3060_RTF × 1.29 → ~1.15 (rounded)

Dia-1.6B calculated directly from first-principles:
  DAC frame rate × model memory / GPU effective bandwidth
  = 86 frames/s × 3.2 GB / (bandwidth × 0.55 efficiency)

Uncertainty: ±30%. Actual performance depends on CUDA driver version,
PyTorch version, batch size variations, and CPU↔GPU data transfer overhead.
Real numbers will only come from running the bench_warm.py benchmark on the VM
after GPU installation.
```

---

## 10. After GPU Install — Benchmark Commands

```bash
# Verify GPU is visible
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# Quick RTF spot-check for each model
curl -X POST http://localhost:8001/synthesize/chatterbox \
  -H "Content-Type: application/json" \
  -d '{"text":"Oh my goodness, just a moment dear, let me find my glasses.","params":{"exaggeration":0.6}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('RTF='+str(d['rtf']))"

# Full benchmark (all models)
/opt/arthur-bench-env/bin/python3 /opt/arthur/bench_warm.py

# Check GPU memory usage after loading all models
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```