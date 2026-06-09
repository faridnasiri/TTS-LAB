#!/opt/arthur-img-env/bin/python
"""Ideogram 4 benchmark — test multiple resolutions with nf4 quant."""
import urllib.request, urllib.parse, json, time, sys

QUANT = "nf4"
PRESET = "V4_TURBO_12"
SEED = 42
RESOLUTIONS = [256, 384, 512, 768]

results = []

for res in RESOLUTIONS:
    print(f"\n{'='*50}", flush=True)
    print(f"BENCH: {res}x{res} | quant={QUANT} | preset={PRESET}", flush=True)
    
    payload = {
        "prompt": '{"high_level_description":"a serene mountain lake at sunset with pine trees","style_description":{"aesthetics":"cinematic photorealistic","lighting":"golden hour warm","photo":"35mm f/2.8","medium":"photograph"},"compositional_deconstruction":{"background":"snow-capped mountains reflecting on calm lake water","elements":[{"type":"obj","desc":"pine tree silhouette"},{"type":"obj","desc":"wooden dock"},{"type":"obj","desc":"small rowboat"}]}}',
        "width": str(res), "height": str(res),
        "preset": PRESET, "num_inference_steps": "0",
        "guidance_scale": "1.0", "mu": "0.0", "std": "1.75",
        "seed": str(SEED), "quant": QUANT,
        "use_magic_prompt": "false", "magic_prompt_input": "",
        "magic_prompt_aspect_ratio": "1:1"
    }

    t0 = time.time()
    data = urllib.parse.urlencode(payload, doseq=True).encode()
    req = urllib.request.Request("http://localhost:8002/generate/ideogram4", data=data, method="POST")
    
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        result = json.loads(resp.read())
        elapsed = time.time() - t0
        
        img_info = result.get("results", [{}])[0]
        base64_size = len(img_info.get("base64", ""))
        
        print(f"  OK  HTTP {resp.status}  Time: {elapsed:.1f}s  Base64: {base64_size}B", flush=True)
        print(f"  URL: {img_info.get('url', 'N/A')}", flush=True)
        
        results.append({
            "resolution": f"{res}x{res}",
            "quant": QUANT,
            "preset": PRESET,
            "time_s": round(elapsed, 1),
            "base64_bytes": base64_size,
            "url": img_info.get("url", ""),
            "status": "OK"
        })
        
        # Quick cooldown between generations
        time.sleep(3)
        
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.0f}s: {exc}", flush=True)
        results.append({
            "resolution": f"{res}x{res}",
            "quant": QUANT,
            "time_s": round(elapsed, 1),
            "error": str(exc)[:200],
            "status": "FAILED"
        })

print(f"\n{'='*50}", flush=True)
print("BENCHMARK COMPLETE", flush=True)
print(json.dumps(results, indent=2), flush=True)
