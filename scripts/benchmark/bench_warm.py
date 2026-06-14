#!/usr/bin/env python3
"""Isolated warm-RTF benchmark: 2 calls per model (cold then warm), with unload between heavy models."""
import json, base64, urllib.request, time, shutil

SHORT = "Oh my goodness, just a moment dear."
LONG  = ("Oh my goodness, just a moment dear, let me find my reading glasses. "
         "Now you said I owe money to the IRS? Can you give me that case number "
         "again, nice and slow?")

MODELS = [
    ("piper",      LONG, {"voice": "en_US-ryan-high"},       False),
    ("kokoro",     LONG, {"voice": "bm_lewis", "speed": "0.85"}, False),
    ("melo",       LONG, {"speaker": "EN-US", "speed": "0.85"},  False),
    ("chattts",    LONG, {"prompt": "[speed_5]", "temperature": "0.3", "seed": "0"}, True),
    ("outetts",    LONG, {"model_path": "OuteAI/OuteTTS-0.3-500M", "temperature": "0.4"}, True),
    ("parler",     LONG, {"description": "An elderly man with a slow warm slightly confused voice."}, True),
    ("chatterbox", LONG, {"exaggeration": "0.65", "cfg_weight": "0.5"}, True),
    ("xtts",       LONG, {"speaker": "Torcull Diarmuid", "language": "en"}, True),
    # -- New engines (warm-test subset) --
    ("fishspeech", LONG, {"speed": "1.0"},                        True),
    ("orpheus",    LONG, {"voice": "tara"},                        True),
    ("zonos",      LONG, {"variant": "transformer", "speaking_rate": "13.0"}, True),
    ("openvoice",  LONG, {"speaker": "EN-US", "speed": "0.85"},   True),
]

BASE = "http://localhost:8001"

def post(path, body, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def unload(name):
    try: post(f"/unload/{name}", {}, timeout=10)
    except: pass

print(f"\n{'Model':<14} {'Cold RTF':>9}  {'Warm RTF':>9}  {'Verdict':<14}  {'Load':>7}  {'Audio':>8}")
print("-" * 80)

for name, text, params, heavy in MODELS:
    # unload any heavy model that might be resident
    if heavy:
        for m in ["parler","chatterbox","xtts"]:
            unload(m)
        time.sleep(1)

    cold_rtf = warm_rtf = load_s = audio_ms = None

    # ── Call 1: cold (model loads) ──
    try:
        d = post(f"/synthesize/{name}", {"text": text, "params": params})
        if "audio_b64" in d:
            cold_rtf = d["rtf"];  load_s = d["load_time_s"]
            audio_ms = d["audio_dur_ms"]
    except Exception as e:
        print(f"{name:<14}  cold FAIL: {e}"); continue

    # ── Call 2: warm (model already in memory) ──
    try:
        d = post(f"/synthesize/{name}", {"text": text, "params": params})
        if "audio_b64" in d:
            wav = base64.b64decode(d["audio_b64"])
            open(f"/tmp/warm_{name}.wav", "wb").write(wav)
            warm_rtf  = d["rtf"]
            audio_ms  = d["audio_dur_ms"]
    except Exception as e:
        print(f"{name:<14}  warm FAIL: {e}"); continue

    flag = "REAL-TIME ✅" if warm_rtf < 1.0 else ("borderline ⚠" if warm_rtf < 1.5 else "too slow  ❌")
    print(f"{name:<14} {cold_rtf:>9.3f}  {warm_rtf:>9.3f}  {flag:<14}  {load_s:>5.1f}s  {audio_ms:>6}ms")

print()
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM : {vm.used//1024//1024:,} / {vm.total//1024//1024:,} MB  ({vm.available//1024//1024:,} MB free)")
except:
    pass
disk = shutil.disk_usage("/opt/models")
print(f"Disk: {disk.used//1024**3:.1f} / {disk.total//1024**3:.1f} GB  on /opt/models")
