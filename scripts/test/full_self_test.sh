#!/usr/bin/env bash
# Run on VM: bash /tmp/full_test.sh > /tmp/full_test_results.txt 2>&1
set -o pipefail
BASE="http://localhost:8001"
OUT="/tmp/full_test_results.txt"

# Wait for service
for i in $(seq 1 30); do
    if curl -s "$BASE/status" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        echo "Service ready after ${i}s"
        break
    fi
    sleep 2
done

echo "=== Starting engine tests at $(date) ==="

test_one() {
    local eng=$1 text=$2 params=$3
    local resp=$(curl -s --max-time 180 -X POST "$BASE/synthesize/$eng" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"$text\",\"params\":$params}" 2>&1)
    local err=$(echo "$resp" | python3 -c "
import sys,json
d=json.load(sys.stdin)
e=d.get('error','')
print(e[:150] if e else 'OK')
" 2>/dev/null)
    echo "$eng|$err"
}

# Fast engines
test_one piper     "Hello." "{}"
test_one kokoro    "Hello." "{}"
test_one melo      "Hello." "{}"
test_one matcha    "Hello." "{}"

# Medium
test_one chattts   "Hello." "{}"
test_one outetts   "Hello." "{}"
test_one styletts2 "Hello." "{}"
test_one fishspeech "Hello." "{}"
test_one openvoice "Hello." "{}"
test_one zonos     "Hello." "{}"
test_one orpheus   "Hello." "{}"

# Heavy
test_one bark      "Hello." "{}"
test_one f5tts     "Hello." "{}"
test_one dia       "Hello." "{}"
test_one xtts      "Hello." "{}"
test_one chatterbox "Hello." '{"model":"default"}'
test_one qwen3tts  "Hello." "{}"
test_one cosyvoice "Hello." "{}"
test_one parler    "Hello." "{}"
test_one indextts  "Hello." "{}"
test_one manatts   "Hello." "{}"
test_one csm       "Hello." "{}"
test_one neutts    "Hello." "{}"

# New engines
test_one chatterboxturbo "Hello." "{}"
test_one omnivoice       "Hello." '{"language":"en"}'
test_one vibevoice       "Hello." "{}"
test_one higgs           "Hello." "{}"
test_one s2pro           "Hello." "{}"
test_one editx           "Hello." "{}"

echo "=== All done at $(date) ==="
