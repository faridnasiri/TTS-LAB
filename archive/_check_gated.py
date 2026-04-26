import json, urllib.request
for repo in ["sesame/csm-1b", "canopylabs/orpheus-3b-0.1-ft"]:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=8) as r:
            d = json.loads(r.read())
            print(f"{repo}: gated={d.get('gated')}, private={d.get('private')}")
    except Exception as e:
        print(f"{repo}: ERROR {e}")
# Also check if raw file is accessible without token
for url in [
    "https://huggingface.co/sesame/csm-1b/resolve/main/config.json",
    "https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/resolve/main/config.json",
]:
    try:
        import urllib.error
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=8) as r:
            print(f"  {url.split('/')[4]+'/'+url.split('/')[5]}: HTTP {r.status} (public)")
    except urllib.error.HTTPError as e:
        print(f"  {url.split('/')[4]+'/'+url.split('/')[5]}: HTTP {e.code} (gated)")
    except Exception as e:
        print(f"  ERROR: {e}")
