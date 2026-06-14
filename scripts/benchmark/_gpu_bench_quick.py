#!/usr/bin/env python3
"""
GPU Benchmark — RTX 5060 Ti edition.
Hits the live tts_lab API on port 8001.
Unloads each model after use to prevent VRAM exhaustion / SEGV.
Run:  /opt/arthur-bench-env/bin/python3 /opt/arthur/_gpu_bench_quick.py
"""
import json, subprocess, time, urllib.request, urllib.error

PHRASE = (
    "Oh my goodness, just a moment dear, I need to find my reading glasses. "
    "Now, you said I owe money to the IRS? "
    "Can you give me that case number again, nice and slow? "
    "My son always tells me to write these things down."
)

# (engine, params, restart_after)  — restart_after=True for VRAM-heavy models
ENGINES = [
    ("piper",       {"voice": "en_US-ryan-high"},                                         False),
    ("kokoro",      {"voice": "bm_lewis", "speed": "0.85"},                               False),
    ("melo",        {"speaker": "EN-US", "speed": "0.85"},                                False),
    ("chattts",     {"prompt": "[speed_5]", "temperature": "0.3"},                        True),
    ("bark",        {"voice_preset": "v2/en_speaker_6"},                                  True),
    ("styletts2",   {"alpha": "0.3", "beta": "0.7", "diffusion_steps": "5"},              True),
    ("f5tts",       {},                                                                    True),
    ("dia",         {"cfg_scale": "3.0", "temperature": "1.2"},                           True),
    ("xtts",        {"speaker": "Torcull Diarmuid", "language": "en"},                    True),
    ("cosyvoice",   {"speaker": "English Female"},                                        True),
    ("parler",      {"description": "An elderly man with a slow warm slightly confused voice speaks gently."}, True),
    ("chatterbox",  {"exaggeration": "0.65", "cfg_weight": "0.5"},                        True),
    ("orpheus",     {"voice": "tara"},                                                    True),
    ("outetts",     {"model_path": "OuteAI/OuteTTS-0.3-500M", "temperature": "0.4"},      True),
    ("zonos",       {"variant": "transformer", "speaking_rate": "13.0"},                  True),
    ("openvoice",   {"speaker": "EN-US", "speed": "0.85"},                                True),
    ("fishspeech",  {},                                                                    True),
    ("csm",         {},                                                                    True),
    ("indextts",    {},                                                                    True),
    ("qwen3tts",    {},                                                                    True),
]

BASE = "http://localhost:8001"

def api(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}

def restart_server():
    print("    restart arthur-lab...", flush=True)
    subprocess.run(["sudo", "systemctl", "restart", "arthur-lab"], capture_output=True)
    for _ in range(20):
        time.sleep(3)
        try:
            urllib.request.urlopen(BASE + "/", timeout=3)
            print("    server ready", flush=True)
            return
        except Exception:
            pass
    print("    server still not ready", flush=True)

results = []
print(f"\n{'engine':<14} {'rtf':>8} {'synth_ms':>9} {'audio_ms':>9} {'load_s':>7} {'hz':>7}  status")
print("-" * 72)

for name, params, do_restart in ENGINES:
    print(f"{name:<14}", end=" ", flush=True)
    d = api("POST", f"/synthesize/{name}", {"text": PHRASE, "params": params}, timeout=600)

    api("DELETE", f"/models/{name}", timeout=10)

    err = d.get("error", "")
    if err:
        short = err[:80].replace("\n", " ")
        print(f"{'ERROR':>8}  {short}", flush=True)
        results.append({"engine": name, "status": "error", "error": err})
    else:
        rtf  = d.get("rtf", 0)
        s    = d.get("synth_time_ms", 0)
        a    = d.get("audio_dur_ms", 0)
        lo   = d.get("load_time_s", 0)
        hz   = d.get("sample_rate", 0)
        flag = "real-time" if rtf < 1 else ("borderline" if rtf < 2 else "too slow")
        print(f"{rtf:>8.4f} {s:>9} {a:>9} {lo:>7} {hz:>7}  {flag}", flush=True)
        results.append({"engine": name, "status": "pass", "rtf": rtf,
                        "synth_ms": s, "audio_ms": a, "load_s": lo, "hz": hz})

    if do_restart:
        restart_server()

with open("/tmp/gpu_bench_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved -> /tmp/gpu_bench_results.json")
