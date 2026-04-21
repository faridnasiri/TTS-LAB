#!/usr/bin/env python3
"""
TTS Lab — automated synthesis test.
Tests every available engine, verifies audio was produced, reports RTF.

Usage:
    python3 _tts_test.py [--engine ENGINE] [--text "..."] [--url URL] [--timeout N]

Exit code:
    0 = all tested engines passed
    1 = one or more failures
"""
import argparse, base64, json, sys, time, urllib.request, urllib.error, wave, io

API   = "http://localhost:8001"
TEXT  = "Oh my goodness, just a moment dear. You said I owe money?"
# None = skip (needs ref WAV, gated model, or known incompatible).
# Reason string shown in skip output.
ENGINE_PARAMS = {
    "piper":      {"voice": "en_US-ryan-high"},
    "kokoro":     {"voice": "bm_lewis", "speed": "0.9"},
    "melo":       {"speaker": "EN-US", "speed": "0.9"},
    "chattts":    {"prompt": "[speed_5]", "seed": "2024"},
    "outetts":    {},
    "bark":       {"voice_preset": "v2/en_speaker_6"},
    "styletts2":  {},
    "f5tts":      None,          # needs ref WAV — skip
    "dia":        {"max_tokens": "256"},
    "xtts":       {"speaker": "Torcull Diarmuid", "language": "en"},
    "cosyvoice":  None,          # needs model download — skip
    "parler":     None,          # incompatible: needs transformers<=4.46.1, env has 4.57
    "chatterbox": {},
    "fishspeech": {},
    "csm":        None,          # gated: sesame/csm-1b requires HF login + access approval
    "qwen3tts":   None,          # gated: Qwen/Qwen3-TTS requires Alibaba access request
    "orpheus":    None,          # gated: canopylabs/orpheus-3b-0.1-ft requires HF login
    "neutts":     None,          # placeholder — package not yet identified
    "indextts":   None,          # needs ref WAV — skip
    "zonos":      {"speaking_rate": "18", "max_new_tokens": "300", "cfg_scale": "1.5"},
    "openvoice":  {"speaker": "EN-US", "speed": "0.9"},
}
SLOW_ENGINES  = {"bark", "dia", "zonos"}        # allow longer timeout
HEAVY_TIMEOUT = 240   # seconds for slow engines
LIGHT_TIMEOUT = 90    # seconds for fast engines

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Capture the error body from 500 responses
        body = e.read().decode(errors="replace")
        try:
            return json.loads(body)
        except Exception:
            return {"error": f"HTTP {e.code}: {body[:200]}"}


def _check_status(api):
    with urllib.request.urlopen(f"{api}/status", timeout=10) as r:
        return json.loads(r.read())


def _unload(api, name):
    """Unload a model to free VRAM between tests."""
    try:
        req = urllib.request.Request(f"{api}/models/{name}", method="DELETE")
        with urllib.request.urlopen(req, timeout=10): pass
    except Exception:
        pass


def _wav_duration_ms(b64):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        return int(wf.getnframes() / wf.getframerate() * 1000)


def test_engine(name, params, text, api, timeout):
    url = f"{api}/synthesize/{name}"
    t0  = time.time()
    try:
        d = _post(url, {"text": text, "params": params}, timeout)
    except urllib.error.URLError as e:
        return False, f"HTTP error: {e}"
    except Exception as e:
        return False, f"Request failed: {e}"
    elapsed = time.time() - t0

    if d.get("error"):
        trace = d.get("trace", "")
        # First line of error only for display; full trace on verbose
        first = d["error"].split("\n")[0][:120]
        return False, first + (f"\n        trace: {trace.split(chr(10))[0][:100]}" if trace else "")

    b64 = d.get("audio_b64", "")
    if not b64:
        return False, "No audio_b64 in response"

    dur_ms = _wav_duration_ms(b64)
    if dur_ms < 50:
        return False, f"Audio too short ({dur_ms} ms) — likely silence or WAV header only"

    rtf  = float(d.get("rtf", 0))
    sr   = d.get("sample_rate", "?")
    load = d.get("load_time_s", "?")
    synth= d.get("synth_time_ms", "?")
    return True, f"dur={dur_ms}ms  rtf={rtf:.2f}×  sr={sr}Hz  load={load}s  synth={synth}ms  wall={elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",  help="Test only this engine")
    ap.add_argument("--text",    default=TEXT)
    ap.add_argument("--url",     default=API)
    ap.add_argument("--timeout", type=int, default=0, help="Override timeout (seconds)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--unload",  action="store_true",
                    help="Unload each model after testing it (frees VRAM, avoids OOM in full suite)")
    args = ap.parse_args()

    # Verify server is up
    try:
        status = _check_status(args.url)
    except Exception as e:
        print(f"{RED}✗ Server not reachable at {args.url}: {e}{RESET}")
        sys.exit(1)

    models     = status.get("models", {})
    order      = list(models.keys())
    test_names = [args.engine] if args.engine else order

    passed = failed = skipped = 0
    results = []

    print(f"\n{BOLD}TTS Lab — synthesis test   {args.url}{RESET}")
    print(f"Text: \"{args.text[:80]}\"")
    print("─" * 72)

    for name in test_names:
        info    = models.get(name, {})
        avail   = info.get("available", False)
        params  = ENGINE_PARAMS.get(name)

        if params is None:
            print(f"  {YELLOW}↷ {name:<14}{RESET} skipped (needs ref WAV / not configured)")
            skipped += 1
            results.append((name, "skip", "needs ref WAV / not configured"))
            continue

        if not avail:
            reason = info.get("reason", "not available")[:60]
            print(f"  {YELLOW}↷ {name:<14}{RESET} not available: {reason}")
            skipped += 1
            results.append((name, "skip", reason))
            continue

        timeout = args.timeout or (HEAVY_TIMEOUT if name in SLOW_ENGINES else LIGHT_TIMEOUT)
        print(f"  {CYAN}▶ {name:<14}{RESET}", end=" ", flush=True)
        ok, detail = test_engine(name, params, args.text, args.url, timeout)

        if ok:
            print(f"{GREEN}PASS{RESET}  {detail}")
            passed += 1
            results.append((name, "pass", detail))
        else:
            print(f"{RED}FAIL{RESET}  {detail}")
            failed += 1
            results.append((name, "fail", detail))

        if args.unload:
            _unload(args.url, name)

    print("─" * 72)
    color = GREEN if failed == 0 else RED
    print(f"{color}{BOLD}Results: {passed} passed  {failed} failed  {skipped} skipped{RESET}\n")

    if failed:
        print(f"{RED}Failed engines:{RESET}")
        for name, status_str, detail in results:
            if status_str == "fail":
                print(f"  ✗ {name}: {detail}")
        print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
