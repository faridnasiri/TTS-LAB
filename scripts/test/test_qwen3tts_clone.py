import requests, json, base64, time

ref_id = "el_ryan_kurk_pleasant_and_smooth"

# Test 1: x-vector only
print("=== Test 1: x-vector only (no ref_text) ===")
t0 = time.time()
r = requests.post("http://engine-qwen:8104/synthesize", json={
    "engine": "qwen3tts", "text": "Hello, this is the x vector only voice cloning test.",
    "params": {"language": "english", "audio_prompt_id": ref_id}
}, timeout=300)
t1 = time.time() - t0
d = r.json()
print(f"Status: {r.status_code}  time: {t1:.0f}s")
if r.status_code == 200:
    print(f"dur_ms={d.get('audio_dur_ms')}  synth_ms={d.get('synth_time_ms')}  RTF={d.get('rtf')}  sr={d.get('sample_rate')}")
else:
    print(f"ERROR: {r.text[:400]}")

# Test 2: ICL mode (with transcript)
print()
print("=== Test 2: ICL mode (with ref_text) ===")
t0 = time.time()
r = requests.post("http://engine-qwen:8104/synthesize", json={
    "engine": "qwen3tts", "text": "This is the ICL voice cloning test with reference transcript for best quality.",
    "params": {"language": "english", "audio_prompt_id": ref_id, "ref_text": "This is a reference audio clip for testing voice cloning."}
}, timeout=300)
t1 = time.time() - t0
d = r.json()
print(f"Status: {r.status_code}  time: {t1:.0f}s")
if r.status_code == 200:
    print(f"dur_ms={d.get('audio_dur_ms')}  synth_ms={d.get('synth_time_ms')}  RTF={d.get('rtf')}  sr={d.get('sample_rate')}")
else:
    print(f"ERROR: {r.text[:400]}")

# Test 3: No reference at all (default voice)
print()
print("=== Test 3: No reference (default voice) ===")
t0 = time.time()
r = requests.post("http://engine-qwen:8104/synthesize", json={
    "engine": "qwen3tts", "text": "This is a default voice test with no reference audio.",
    "params": {"language": "english"}
}, timeout=300)
t1 = time.time() - t0
d = r.json()
print(f"Status: {r.status_code}  time: {t1:.0f}s")
if r.status_code == 200:
    print(f"dur_ms={d.get('audio_dur_ms')}  synth_ms={d.get('synth_time_ms')}  RTF={d.get('rtf')}  sr={d.get('sample_rate')}")
else:
    print(f"ERROR: {r.text[:400]}")

print()
print("All tests done!")
