#!/opt/arthur-img-env/bin/python
import urllib.request, urllib.parse, json, time

payload = {
    "prompt": '{"high_level_description":"a serene mountain lake at sunset","style_description":{"aesthetics":"cinematic","lighting":"golden hour","photo":"35mm f/2.8","medium":"photograph"},"compositional_deconstruction":{"background":"snow-capped mountains reflecting on a calm lake","elements":[{"type":"obj","desc":"a tall pine tree silhouetted against the sunset"},{"type":"obj","desc":"a wooden dock extending into the lake"}]}}',
    "width": "512", "height": "512", "preset": "V4_TURBO_12",
    "num_inference_steps": "0", "guidance_scale": "7.0",
    "mu": "0.0", "std": "1.75", "seed": "42", "quant": "nf4",
    "use_magic_prompt": "false", "magic_prompt_input": "",
    "magic_prompt_aspect_ratio": "1:1"
}

print("Generating nf4 TURBO...", flush=True)
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
    print(f"FAILED after {time.time()-t0:.0f}s: {exc}", flush=True)