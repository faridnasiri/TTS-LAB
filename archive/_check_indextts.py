import json, urllib.request, urllib.error

url = "http://localhost:8001/synthesize/indextts"
data = json.dumps({"text": "Hello world.", "params": {"ref_audio_id": ""}}).encode()
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
        print("OK" if d.get("audio_b64") else "ERROR: " + d.get("error","?")[:200])
except urllib.error.HTTPError as e:
    d = json.loads(e.read().decode(errors="replace"))
    print("HTTP", e.code, d.get("error","?")[:200])
    t = d.get("trace","")
    if t:
        lines = [l for l in t.split("\n") if l.strip()]
        print("\n".join(lines[-6:]))
except Exception as e:
    print("FAIL:", e)
