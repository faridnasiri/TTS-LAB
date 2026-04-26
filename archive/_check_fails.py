#!/usr/bin/env python3
import json, urllib.request, urllib.error

API = "http://localhost:8001"
ENGINES = ["parler", "fishspeech", "csm", "orpheus", "openvoice"]
TEXT = "Hello there."

for name in ENGINES:
    url = f"{API}/synthesize/{name}"
    data = json.dumps({"text": TEXT, "params": {}}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    print(f"\n=== {name} ===")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
            err = d.get("error", "")
            if err:
                print("ERROR:", err[:300])
                trace = d.get("trace", "")
                if trace:
                    # Last 3 lines of traceback are most informative
                    lines = [l for l in trace.split("\n") if l.strip()]
                    print("TRACE:", "\n       ".join(lines[-3:]))
            else:
                print("OK  audio_b64 len:", len(d.get("audio_b64", "")))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            d = json.loads(body)
            print("HTTP", e.code, "ERROR:", d.get("error", body[:200]))
            trace = d.get("trace", "")
            if trace:
                lines = [l for l in trace.split("\n") if l.strip()]
                print("TRACE:", "\n       ".join(lines[-4:]))
        except Exception:
            print("HTTP", e.code, body[:200])
    except Exception as e:
        print("FAIL:", e)
