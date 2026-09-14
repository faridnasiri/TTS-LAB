#!/usr/bin/env bash
# VM test script — checks service health, engine count, tests existing + new engines
set -e
BASE="http://localhost:8001"
PASS=0; FAIL=0

green() { printf '\033[32m  %-20s PASS\033[0m\n' "$1"; PASS=$((PASS+1)); }
red()   { printf '\033[31m  %-20s FAIL  %s\033[0m\n' "$1" "$2"; FAIL=$((FAIL+1)); }
yell()  { printf '\033[33m  %-20s SKIP  %s\033[0m\n' "$1" "$2"; }

# Wait for service to be ready (up to 60s)
echo "Waiting for service..."
for i in $(seq 1 30); do
    if curl -s "$BASE/status" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        echo "Service ready after ${i}s"
        break
    fi
    sleep 2
done

# Count engines
ENGINE_COUNT=$(curl -s "$BASE/status" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))" 2>/dev/null || echo 0)
echo "Engines registered: $ENGINE_COUNT"
if [ "$ENGINE_COUNT" -ge 28 ]; then
    green "engine-count ($ENGINE_COUNT)"
else
    red "engine-count" "expected 28, got $ENGINE_COUNT"
fi

# Check new engines appear in status
for eng in chatterboxturbo vibevoice higgs omnivoice s2pro; do
    if curl -s "$BASE/status" | python3 -c "import sys,json; d=json.load(sys.stdin); keys=[m['key'] for m in d['models']]; assert '$eng' in keys" 2>/dev/null; then
        green "status:$eng"
    else
        red "status:$eng" "not found in /status"
    fi
done

# Synthesize test — fast existing engines (verify no regression)
echo ""
echo "=== Existing engine regression tests ==="
for eng in piper kokoro melo matcha; do
    RESULT=$(curl -s -w "\n%{http_code}" -X POST "$BASE/synthesize/$eng" \
        -H 'Content-Type: application/json' \
        -d '{"text":"Hello, this is a regression test.","params":{}}' 2>/dev/null)
    HTTP=$(echo "$RESULT" | tail -1)
    if [ "$HTTP" = "200" ]; then
        green "synth:$eng"
    else
        ERR=$(echo "$RESULT" | head -1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','?')[:80])" 2>/dev/null || echo "HTTP $HTTP")
        if echo "$ERR" | grep -qi "not available\|not configured\|gated\|not installed\|huggingface-cli"; then
            yell "synth:$eng" "$ERR"
        else
            red "synth:$eng" "$ERR"
        fi
    fi
done

# Synthesize test — new engines (expect not-available or success)
echo ""
echo "=== New engine tests ==="
for eng in chatterboxturbo vibevoice higgs omnivoice s2pro; do
    RESULT=$(curl -s -w "\n%{http_code}" -X POST "$BASE/synthesize/$eng" \
        -H 'Content-Type: application/json' \
        -d '{"text":"Hello, testing new engine.","params":{}}' 2>/dev/null)
    HTTP=$(echo "$RESULT" | tail -1)
    BODY=$(echo "$RESULT" | sed '$d')
    if [ "$HTTP" = "200" ]; then
        green "synth:$eng"
    elif [ "$HTTP" = "400" ] || [ "$HTTP" = "500" ] || [ "$HTTP" = "408" ]; then
        ERR=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','?')[:100])" 2>/dev/null || echo "HTTP $HTTP")
        if echo "$ERR" | grep -qi "not available\|not configured\|not installed\|gated\|huggingface-cli login\|SGLang\|requires a CUDA\|cuda gpu"; then
            yell "synth:$eng" "$ERR"
        else
            red "synth:$eng" "$ERR"
        fi
    else
        red "synth:$eng" "HTTP $HTTP"
    fi
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
