#!/usr/bin/env bash
# test_all_engines.sh — comprehensive synthesis test on all 28 engines
# Usage: bash /tmp/test_all_engines.sh
set -o pipefail

BASE="http://localhost:8001"
TEXT="Hello, this is an automated engine test."
PASS=0; FAIL=0; SKIP=0

green() { printf '\033[32m  PASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
red()   { printf '\033[31m  FAIL\033[0m  %-20s %s\n' "$1" "$2"; FAIL=$((FAIL+1)); }
yell()  { printf '\033[33m  SKIP\033[0m  %-20s %s\n' "$1" "$2"; SKIP=$((SKIP+1)); }

test_engine() {
    local eng=$1 text=$2 params=$3 timeout=${4:-120}
    local resp=$(curl -s --max-time "$timeout" -X POST "$BASE/synthesize/$eng" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"$text\",\"params\":$params}" 2>&1)
    local err=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','')[:120])" 2>/dev/null)

    if [ -z "$err" ]; then
        green "$eng"
    elif echo "$err" | grep -qi "not available\|not configured\|gated\|huggingface-cli\|CUDA GPU\|SGLang\|requires a cuda\|cuda gpu"; then
        yell "$eng" "$err"
    else
        red "$eng" "$err"
    fi
}

echo "============================================================"
echo "  TTS Lab — All-Engine Synthesis Test (28 engines)"
echo "  $(date)"
echo "============================================================"
echo ""

# === Fast engines (CPU-friendly, should always work) ===
echo "--- Fast engines ---"
test_engine piper      "Hello piper test." "{}" 30
test_engine kokoro     "Hello kokoro test." "{}" 30
test_engine melo       "Hello melo test." "{}" 30
test_engine matcha     "Hello matcha test." "{}" 30

echo ""
echo "--- Medium-weight engines ---"
test_engine chattts    "Hello chattts test." "{}" 60
test_engine outetts    "Hello outetts test." "{}" 90
test_engine styletts2  "Hello styletts2 test." "{}" 90
test_engine fishspeech "Hello fishspeech test." "{}" 90
test_engine orpheus    "Hello orpheus test." "{}" 30
test_engine openvoice  "Hello openvoice test." "{}" 60
test_engine zonos      "Hello zonos test." "{}" 60

echo ""
echo "--- Heavy engines (GPU recommended) ---"
test_engine bark       "Hello bark test." "{}" 120
test_engine f5tts      "Hello f5tts test." "{}" 30
test_engine dia        "Hello dia test." "{}" 120
test_engine xtts       "Hello xtts test." "{}" 90
test_engine chatterbox "Hello cb test." '{"model":"default"}' 120
test_engine qwen3tts   "Hello qwen3tts test." "{}" 120
test_engine cosyvoice  "Hello cosyvoice test." "{}" 30
test_engine parler     "Hello parler test." "{}" 90
test_engine indextts   "Hello indextts test." "{}" 30
test_engine manatts    "Hello manatts test." "{}" 30
test_engine neutts     "Hello neutts test." "{}" 10
test_engine csm        "Hello csm test." "{}" 30

echo ""
echo "--- New engines ---"
test_engine chatterboxturbo "Hello cbt test." "{}" 120
test_engine omnivoice       "Hello omni test." '{"language":"en"}' 180
test_engine vibevoice       "Hello vibe test." "{}" 10
test_engine higgs           "Hello higgs test." "{}" 10
test_engine s2pro           "Hello s2pro test." "{}" 10
test_engine editx           "Hello editx test." "{}" 600

echo ""
echo "============================================================"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"
exit $FAIL
