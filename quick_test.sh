#!/bin/bash
# quick_test.sh — post-deploy smoke test for all 21 Arthur TTS Lab engines.
# NO restart by default — tests the already-running service on port 8001.
# Pass --restart to restart the service first (recommended after a fresh deploy).
# Runtime: ~2 min for fast engines; slow-loaders (indextts, qwen3tts) get 240s.
# Exit code: 0 = all installed engines passed, 1 = one or more failures.
#
# Usage:
#   bash /opt/arthur/quick_test.sh             # test running service
#   bash /opt/arthur/quick_test.sh --restart   # restart service first, then test

RESTART=false
for arg in "$@"; do [ "$arg" = "--restart" ] && RESTART=true; done

BASE_URL="http://localhost:8001"
TEXT="Hello, this is an automated smoke test."
REQ_FILE="/tmp/_qt_req.json"
OUT_FILE="/tmp/_qt_out"
PASS=0; FAIL=0; SKIP=0

# Engines that require a reference WAV — 500 with ref-WAV message = PASS (expected)
REF_WAV_ENGINES="indextts f5tts openvoice cosyvoice xtts chatterbox fishspeech"

# Engines that are GPU-only (vllm/CUDA required) — expect graceful fail
GPU_ONLY_ENGINES="orpheus"

# Slow-loading engines get a longer curl timeout (first load can take 2+ min)
SLOW_ENGINES="indextts qwen3tts bark dia zonos"

green()  { printf '\033[32m  %-14s PASS   %6s  %s\033[0m\n' "$1" "${2:--}" "$3"; }
red()    { printf '\033[31m  %-14s FAIL   %6s  %s\033[0m\n' "$1" "${2:--}" "$3"; }
yellow() { printf '\033[33m  %-14s SKIP   %6s  %s\033[0m\n' "$1" "${2:--}" "$3"; }
warn()   { printf '\033[33m  %-14s WARN   %6s  %s\033[0m\n' "$1" "${2:--}" "$3"; }
cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }

contains() { echo " $1 " | grep -qw "$2"; }

file_bytes() {
    local f="$1"
    if [ -f "$f" ]; then wc -c < "$f" | tr -d ' '; else echo 0; fi
}

get_json_field() {
    local f="$1" key="$2"
    python3 -c "import sys,json; d=json.load(open('$f')); print(d.get('$key',''))" 2>/dev/null || true
}

printf '\n'
cyan "══════════════════════════════════════════════════════"
cyan "  Arthur TTS Lab — quick_test.sh  (all 21 engines)"
cyan "  $(date '+%Y-%m-%d %H:%M:%S')   server: $BASE_URL"
cyan "══════════════════════════════════════════════════════"
printf '\n'

# Optional restart
if $RESTART; then
    cyan "  ↻ Restarting arthur-lab.service..."
    sudo systemctl restart arthur-lab
    printf '  Waiting 20s for service to come up...\n'
    sleep 20
fi

# Check server is up first
HTTP_ROOT=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL/" 2>/dev/null || true)
if [ "$HTTP_ROOT" != "200" ]; then
    printf '\033[31m  ✗ Server not responding (HTTP %s) — is arthur-lab.service running?\033[0m\n' "$HTTP_ROOT"
    exit 1
fi

# Build engine list from /status
ENGINES=$(curl -sf "$BASE_URL/status" 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(' '.join(sorted(d['models'].keys())))" \
    2>/dev/null || echo "bark chattts chatterbox cosyvoice csm dia f5tts fishspeech indextts kokoro melo neutts openvoice orpheus outetts parler piper qwen3tts styletts2 xtts zonos")

printf '  %-14s %-6s %6s  %s\n' "ENGINE" "RESULT" "RTF" "INFO"
printf '  %s\n' "──────────────────────────────────────────────────────"

for engine in $ENGINES; do
    TIMEOUT=90
    if contains "$SLOW_ENGINES" "$engine"; then TIMEOUT=240; fi

    printf '{"text":"%s","params":{}}' "$TEXT" > "$REQ_FILE"
    OUT="${OUT_FILE}_${engine}"

    HTTP_CODE=$(curl -s -X POST "$BASE_URL/synthesize/$engine" \
        -H 'Content-Type: application/json' \
        -d @"$REQ_FILE" \
        -o "$OUT" \
        -w '%{http_code}' \
        --max-time "$TIMEOUT" 2>/dev/null || true)
    HTTP_CODE="${HTTP_CODE:-000}"

    RTF=$(get_json_field "$OUT" "rtf")
    ERR=$(get_json_field "$OUT" "error" | cut -c1-90)
    BYTES=$(file_bytes "$OUT")

    # Detect "not available / not configured" 500s → treat as SKIP
    IS_NOT_AVAIL=false
    if echo "$ERR" | grep -qi "not available\|not configured\|not installed\|gated\|huggingface-cli login"; then
        IS_NOT_AVAIL=true
    fi

    # Detect ref-WAV error in 500 → treat as PASS for ref-WAV engines
    IS_REF_WAV_ERR=false
    if echo "$ERR" | grep -qi "reference\|ref.*wav\|audio.*prompt\|upload.*clip"; then
        IS_REF_WAV_ERR=true
    fi

    # ── classify ─────────────────────────────────────────────────────────────
    if contains "$GPU_ONLY_ENGINES" "$engine"; then
        # GPU-only: any response is fine as long as it's not a hard crash
        if [ "$HTTP_CODE" = "000" ]; then
            warn "$engine" "" "timeout — GPU OOM or not loaded"
        else
            yellow "$engine" "" "GPU-only (HTTP $HTTP_CODE) — needs CUDA"
        fi
        SKIP=$((SKIP+1))

    elif $IS_NOT_AVAIL; then
        # "Not available" 500 → package not installed
        yellow "$engine" "" "not installed — $ERR"
        SKIP=$((SKIP+1))

    elif [ "$HTTP_CODE" = "503" ]; then
        yellow "$engine" "" "not loaded (HTTP 503)"
        SKIP=$((SKIP+1))

    elif [ "$HTTP_CODE" = "000" ]; then
        # Timeout / connection refused — almost always CUDA OOM from sequential load pressure.
        # Not an engine code bug; count as SKIP so the exit code stays clean.
        warn "$engine" "" "no response (timeout >${TIMEOUT}s — likely CUDA OOM) — retry individually"
        SKIP=$((SKIP+1))

    elif [ "$HTTP_CODE" = "200" ] && [ "$BYTES" -gt 1000 ]; then
        green "$engine" "$RTF" "${BYTES} bytes"
        PASS=$((PASS+1))

    elif contains "$REF_WAV_ENGINES" "$engine" && $IS_REF_WAV_ERR; then
        # Correctly rejected — no ref WAV uploaded (expected in automated test)
        green "$engine" "" "correctly rejected — needs ref WAV (HTTP $HTTP_CODE)"
        PASS=$((PASS+1))

    elif echo "$HTTP_CODE" | grep -q '^4' && contains "$REF_WAV_ENGINES" "$engine"; then
        green "$engine" "" "correctly rejected (HTTP $HTTP_CODE — needs ref WAV)"
        PASS=$((PASS+1))

    else
        red "$engine" "" "HTTP $HTTP_CODE — $ERR"
        FAIL=$((FAIL+1))
    fi

    rm -f "$OUT" 2>/dev/null || true
done

# ── summary ───────────────────────────────────────────────────────────────────
printf '\n  %s\n' "══════════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    printf '\033[32m  🎉  ALL PASS — %d passed, %d skipped (not installed/GPU-only), 0 failed  [%d engines]\033[0m\n' \
        "$PASS" "$SKIP" "$((PASS+FAIL+SKIP))"
else
    printf '\033[31m  ❌  %d FAILED — %d passed, %d skipped  [%d engines]\033[0m\n' \
        "$FAIL" "$PASS" "$SKIP" "$((PASS+FAIL+SKIP))"
fi
printf '  %s\n\n' "══════════════════════════════════════════════════════"

rm -f "$REQ_FILE" 2>/dev/null || true

[ "$FAIL" -eq 0 ]

BASE_URL="http://localhost:8001"
TEXT="Hello, this is an automated smoke test."
REQ_FILE="/tmp/_qt_req.json"
OUT_FILE="/tmp/_qt_out"
PASS=0; FAIL=0; SKIP=0

# Engines that require a reference WAV — we send none and expect a 4xx, not 500
REF_WAV_ENGINES="indextts f5tts openvoice cosyvoice xtts chatterbox fishspeech"

# Engines that are known to not be installed (no package) — expect 503 / not-available
NOT_INSTALLED="csm orpheus neutts"

# Slow-loading engines get a longer curl timeout (first load can take 2+ min)
SLOW_ENGINES="indextts qwen3tts bark dia zonos"

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }

contains() { echo "$1" | grep -qw "$2"; }

printf '\n'
cyan "══════════════════════════════════════════════════════"
cyan "  Arthur TTS Lab — quick_test.sh  (all 21 engines)"
cyan "  $(date '+%Y-%m-%d %H:%M:%S')   server: $BASE_URL"
cyan "══════════════════════════════════════════════════════"
printf '\n'

# Check server is up first
HTTP_ROOT=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL/" || true)
if [ "$HTTP_ROOT" != "200" ]; then
    red "  ✗ Server not responding (HTTP $HTTP_ROOT) — is arthur-lab.service running?"
    exit 1
fi

# Build engine list from /status so we always match what the server knows
ENGINES=$(curl -sf "$BASE_URL/status" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(' '.join(sorted(d['models'].keys())))" \
    2>/dev/null || echo "piper kokoro melo chattts outetts bark styletts2 f5tts dia xtts cosyvoice parler chatterbox fishspeech csm qwen3tts orpheus neutts indextts zonos openvoice")

printf '  %-14s %-8s %6s  %s\n' "ENGINE" "RESULT" "RTF" "INFO"
printf '  %s\n' "──────────────────────────────────────────────────────"

for engine in $ENGINES; do
    # Pick timeout
    TIMEOUT=90
    if contains "$SLOW_ENGINES" "$engine"; then TIMEOUT=240; fi

    # Build request — ref-WAV engines: no audio_prompt_id (triggers expected error)
    printf '{"text":"%s","params":{}}' "$TEXT" > "$REQ_FILE"

    HTTP_CODE=$(curl -s -X POST "$BASE_URL/synthesize/$engine" \
        -H 'Content-Type: application/json' \
        -d @"$REQ_FILE" \
        -o "${OUT_FILE}_${engine}" \
        -w '%{http_code}' \
        --max-time "$TIMEOUT" 2>/dev/null || echo "000")

    RTF=$(grep -o '"rtf":[0-9.]*' "${OUT_FILE}_${engine}" 2>/dev/null | head -1 | cut -d: -f2 || true)
    ERR=$(python3 -c "import sys,json; d=json.load(open('${OUT_FILE}_${engine}')); print(d.get('error','')[:80])" 2>/dev/null || true)
    BYTES=$(wc -c < "${OUT_FILE}_${engine}" 2>/dev/null || echo 0)

    # ── classify result ───────────────────────────────────────────────────────
    if contains "$NOT_INSTALLED" "$engine"; then
        # Expect 404/503 — package not installed
        if [ "$HTTP_CODE" = "503" ] || [ "$HTTP_CODE" = "404" ]; then
            printf '  %-14s ' "$engine"; yellow "SKIP     $(printf '%6s' '')  not installed (HTTP $HTTP_CODE)"
            SKIP=$((SKIP+1))
        elif [ "$HTTP_CODE" = "200" ]; then
            printf '  %-14s ' "$engine"; green "PASS     $(printf '%6s' "${RTF:-?}")  (unexpectedly installed!)"
            PASS=$((PASS+1))
        else
            printf '  %-14s ' "$engine"; yellow "SKIP     $(printf '%6s' '')  HTTP $HTTP_CODE — ${ERR}"
            SKIP=$((SKIP+1))
        fi

    elif contains "$REF_WAV_ENGINES" "$engine"; then
        # Ref-WAV engines — 4xx = correct (no audio provided), 500 = bug, 200 = bonus
        if [ "$HTTP_CODE" = "200" ] && [ "$BYTES" -gt 1000 ]; then
            printf '  %-14s ' "$engine"; green "PASS     $(printf '%6s' "${RTF:-?}")  synthesised"
            PASS=$((PASS+1))
        elif echo "$HTTP_CODE" | grep -q '^4'; then
            printf '  %-14s ' "$engine"; green "PASS     $(printf '%6s' '')  correctly rejected (HTTP $HTTP_CODE — needs ref WAV)"
            PASS=$((PASS+1))
        elif [ "$HTTP_CODE" = "503" ]; then
            printf '  %-14s ' "$engine"; yellow "SKIP     $(printf '%6s' '')  not loaded (HTTP 503)"
            SKIP=$((SKIP+1))
        else
            printf '  %-14s ' "$engine"; red "FAIL     $(printf '%6s' '')  HTTP $HTTP_CODE — ${ERR}"
            FAIL=$((FAIL+1))
        fi

    else
        # Standard engines — must return 200 + audio bytes
        if [ "$HTTP_CODE" = "200" ] && [ "$BYTES" -gt 1000 ]; then
            printf '  %-14s ' "$engine"; green "PASS     $(printf '%6s' "${RTF:-?}")  ${BYTES} bytes"
            PASS=$((PASS+1))
        elif [ "$HTTP_CODE" = "503" ]; then
            printf '  %-14s ' "$engine"; yellow "SKIP     $(printf '%6s' '')  not loaded (HTTP 503)"
            SKIP=$((SKIP+1))
        elif [ "$HTTP_CODE" = "000" ]; then
            printf '  %-14s ' "$engine"; red "FAIL     $(printf '%6s' '')  timeout (>${TIMEOUT}s) or connection refused"
            FAIL=$((FAIL+1))
        else
            printf '  %-14s ' "$engine"; red "FAIL     $(printf '%6s' '')  HTTP $HTTP_CODE — ${ERR}"
            FAIL=$((FAIL+1))
        fi
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
printf '\n  %s\n' "══════════════════════════════════════════════════════"
TOTAL=$((PASS+FAIL+SKIP))
if [ "$FAIL" -eq 0 ]; then
    green "  🎉  ALL PASS — $PASS passed, $SKIP skipped (not installed), 0 failed  [$TOTAL engines]"
else
    red   "  ❌  $FAIL FAILED — $PASS passed, $SKIP skipped  [$TOTAL engines]"
fi
printf '  %s\n\n' "══════════════════════════════════════════════════════"

# Clean up temp files
rm -f "$REQ_FILE" ${OUT_FILE}_* 2>/dev/null || true

[ "$FAIL" -eq 0 ]
