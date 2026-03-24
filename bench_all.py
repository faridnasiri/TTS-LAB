#!/usr/bin/env python3
"""Quick RTF benchmark — runs all 6 models via the live tts_lab API."""
import json, base64, urllib.request, time, psutil, shutil

TEXT = ("Oh my goodness, just a moment dear, let me find my reading glasses. "
        "Now you said I owe money to the IRS? Can you give me that case number "
        "again, nice and slow?")

MODELS = [
    ("piper",      {"voice": "en_US-ryan-high"}),
    ("kokoro",     {"voice": "bm_lewis", "speed": "0.85"}),
    ("melo",       {"speaker": "EN-US", "speed": "0.85"}),
    ("parler",     {"description": "An elderly man with a slow warm slightly confused voice speaks gently."}),
    ("chatterbox", {"exaggeration": "0.65", "cfg_weight": "0.5"}),
    ("xtts",       {"speaker": "Torcull Diarmuid", "language": "en"}),
]

print(f"\n{'Model':<14} {'RTF':>6}  {'Verdict':<14} {'Synth':>8}  {'Audio':>8}  {'Load':>7}")
print("-" * 72)

for name, params in MODELS:
    body = json.dumps({"text": TEXT, "params": params}).encode()
    req  = urllib.request.Request(
        f"http://localhost:8001/synthesize/{name}",
        data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        if "audio_b64" in d:
            wav  = base64.b64decode(d["audio_b64"])
            open(f"/tmp/final_{name}.wav", "wb").write(wav)
            rtf  = d["rtf"]
            flag = "REAL-TIME ✅" if rtf < 1.0 else ("borderline ⚠" if rtf < 1.5 else "too slow  ❌")
            print(f"{name:<14} {rtf:>6.3f}  {flag:<14} "
                  f"{d['synth_time_ms']:>6}ms  {d['audio_dur_ms']:>6}ms  {d['load_time_s']:>5.1f}s")
        else:
            print(f"{name:<14}  FAIL: {d.get('error','?')[:70]}")
    except Exception as e:
        print(f"{name:<14}  ERROR: {str(e)[:70]}")

print()
vm = psutil.virtual_memory()
print(f"RAM  : {vm.used//1024//1024:,} / {vm.total//1024//1024:,} MB  ({vm.available//1024//1024:,} MB free)")
disk = shutil.disk_usage("/opt/models")
print(f"Disk : {disk.used//1024**3:.1f} / {disk.total//1024**3:.1f} GB used on /opt/models")
