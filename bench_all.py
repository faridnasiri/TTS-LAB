#!/usr/bin/env python3
"""Quick RTF benchmark — runs all 6 models via the live tts_lab API."""
import json, base64, urllib.request, time, psutil, shutil

TEXT = ("Oh my goodness, just a moment dear, let me find my reading glasses. "
        "Now you said I owe money to the IRS? Can you give me that case number "
        "again, nice and slow?")

MODELS = [
    ("piper",      {"voice": "en_US-ryan-high"},                                        False),
    ("kokoro",     {"voice": "bm_lewis", "speed": "0.85"},                              False),
    ("melo",       {"speaker": "EN-US", "speed": "0.85"},                               False),
    ("chattts",    {"prompt": "[speed_5]", "temperature": "0.3", "seed": "0"},          True),
    ("outetts",    {"model_path": "OuteAI/OuteTTS-0.3-500M", "temperature": "0.4"},     True),
    ("bark",       {"voice_preset": "v2/en_speaker_6"},                                 True),
    ("styletts2",  {"alpha": "0.3", "beta": "0.7", "diffusion_steps": "5"},             True),
    ("f5tts",      {"speed": "1.0", "nfe_step": "32"},                                  True),
    ("dia",        {"cfg_scale": "3.0", "temperature": "1.2"},                          True),
    ("xtts",       {"speaker": "Torcull Diarmuid", "language": "en"},                   True),
    ("cosyvoice",  {"speaker": "English Female"},                                        True),
    ("parler",     {"description": "An elderly man with a slow warm slightly confused voice speaks gently."}, True),
    ("chatterbox", {"exaggeration": "0.65", "cfg_weight": "0.5"},                       True),
    # -- New engines 14-21 --
    ("fishspeech", {"speed": "1.0"},                                                    True),
    ("csm",        {"speaker_id": "0"},                                                 True),
    ("qwen3tts",   {},                                                                  True),
    ("orpheus",    {"voice": "tara"},                                                   True),
    ("neutts",     {},                                                                  True),
    ("indextts",   {},                                                                  True),
    ("zonos",      {"variant": "transformer", "speaking_rate": "13.0"},                 True),
    ("openvoice",  {"speaker": "EN-US", "speed": "0.85"},                               True),
]

print(f"\n{'Model':<14} {'RTF':>6}  {'Verdict':<14} {'Synth':>8}  {'Audio':>8}  {'Load':>7}")
print("-" * 72)

for name, params, heavy in MODELS:
    body = json.dumps({"text": TEXT, "params": params}).encode()
    req  = urllib.request.Request(
        f"http://localhost:8001/synthesize/{name}",
        data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=(600 if heavy else 300)) as r:
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
