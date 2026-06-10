#!/opt/arthur-img-env/bin/python
import urllib.request, urllib.parse, json, time, sys

# Test 1: 256x256 with TURBO preset (minimum VRAM needed)
payload = {
    "prompt": '{"high_level_description":"a serene mountain lake","style_description":{"aesthetics":"cinematic","medium":"photograph"},"compositional_deconstruction":{"background":"snow-capped mountains reflecting on a lake","elements":[{"type":"obj","desc":"pine tree"},{"type":"obj","desc":"wooden dock"}]}}',
    "width": "256", "height": "256", "preset": "V4_TURBO_12",
    "num_inference_steps": "0", "guidance_scale": "7.0",
    "mu": "0.0", "std": "1.75", "seed": "42", "quant": "nf4",
    "use_magic_prompt": "false",
    "magic_prompt_aspect_ratio": "1:1"
}

print("TEST: 256x256 TURBO nf4", flush=True)
t0 = time.time()
data = urllib.parse.urlencode(payload, doseq=True).encode()
req = urllib.request.Request("http://localhost:8002/generate/ideogram4", data=data, method="POST")
try:
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    print(f"OK HTTP {resp.status} in {time.time()-t0:.0f}s", flush=True)
    print(f"Results: {len(result.get('results',[]))}", flush=True)
    for r in result.get("results",[]):
        print(f"  URL: {r.get('url','N/A')}", flush=True)
except Exception as exc:
    print(f"FAILED: {exc}", flush=True)
