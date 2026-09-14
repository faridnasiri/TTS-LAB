#!/usr/bin/env python3
"""E2E test: Orchestrator -> engine-current -> Omnivoice.

Tests the exact path reported by the external service:
  - Short text (~60 chars) — baseline
  - Long text (~190 chars) — the original failing case
  - Rapid successive calls (VRAM accumulation stress)
"""
import json
import time
import urllib.request
import sys

ORCHESTRATOR = "http://localhost:8001"


def synth(text, label):
    data = json.dumps({
        "engine": "omnivoice",
        "text": text,
        "params": {"language": "en"}
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(
        f"{ORCHESTRATOR}/synthesize/omnivoice",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        t1 = time.time()
        st = result.get("synth_time_ms", 0)
        ad = result.get("audio_dur_ms", 0)
        rtf = result.get("rtf", 0)
        print(
            f"  {label}: HTTP 200 | "
            f"synth={st}ms | "
            f"dur={ad}ms | "
            f"RTF={rtf} | "
            f"total={(t1 - t0):.1f}s"
        )
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        print(f"  {label}: HTTP {e.code} — {body}")
        return False
    except Exception as e:
        print(f"  {label}: ERROR — {e}")
        return False


def main():
    print("=== E2E: Orchestrator (8001) -> engine-current (8101) -> Omnivoice ===")
    print()

    passed = 0
    failed = 0

    # Test A: Short text
    print("Test A: Short text (~60 chars)")
    if synth("Hello world, this is a short test of the Omnivoice engine.", "Short"):
        passed += 1
    else:
        failed += 1

    # Test B: Long text (~190 chars — the original failing case)
    print("Test B: Long text (~190 chars)")
    if synth(
        "This is a substantially longer test passage designed to thoroughly "
        "evaluate the Omnivoice text to speech engine through the orchestrator "
        "dispatch layer. We need enough text to generate considerable audio output.",
        "Long",
    ):
        passed += 1
    else:
        failed += 1

    # Test C: Rapid successive calls (VRAM stress)
    print("Test C: Rapid successive calls (VRAM accumulation)")
    for i in range(1, 6):
        if synth(
            f"Call number {i}. This is a moderately long test sentence to verify "
            f"the orchestrator path handles VRAM correctly across multiple "
            f"successive synthesis requests without accumulating leaked memory.",
            f"Call {i}",
        ):
            passed += 1
        else:
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("FINAL: ALL PASS")
    else:
        print(f"FINAL: {failed} FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    main()
