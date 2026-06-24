---
name: 'run-benchmark'
description: 'Run TTS benchmarks — RTF measurement, batch runs, report generation'
---

# Run TTS Benchmarks

## Prerequisites
- TTS Lab must be running (bare-metal service or containers)
- Benchmark scripts are in `scripts/benchmark/`

## Basic Benchmark (on VM)
```bash
source /opt/arthur-bench-env/bin/activate
cd /opt/arthur
python scripts/benchmark/tts_benchmark.py
```

## Batch Benchmark (calls running server via HTTP)
```bash
python scripts/benchmark/bench_all.py
```

## Warm-Cache Benchmark (excludes model load time)
```bash
python scripts/benchmark/bench_warm.py
```

## Full Benchmark Run
```bash
bash scripts/benchmark/_run_full_bench.sh
```

## Quick GPU Check
```bash
python scripts/benchmark/_gpu_bench_quick.py
```

## Generate Report
```bash
python scripts/benchmark/bench_report_gen.py
```

## Image Lab Benchmark
```bash
python scripts/benchmark/bench_image_lab.py
```

## View Historical Results
Results are stored in `docs/benchmarks/`:
- `BENCHMARK_RESULTS_2026-04-20_RTX5060Ti.md` — First GPU results with RTX 5060 Ti
- `BENCHMARK_RESULTS_2026-04-23.md` — Post-fix results
- `BENCHMARK_RESULTS_2026-03-26.md` — CPU-only baseline (pre-GPU)

## Performance Targets
- **RTF < 1.0** — Real-time capable (engine can keep up with live speech)
- **RTF ≤ 0.25** — Sub-500ms response (target latency for Arthur)
- Key performers: Piper (0.36 CPU), StyleTTS2 (0.35 GPU), Melo (0.30 GPU)
