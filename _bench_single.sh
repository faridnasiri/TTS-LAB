#!/usr/bin/env bash
# Run one model via the live API, print RTF + timing, exit 0 always.
# Usage: _bench_single.sh <model> <params_json> <timeout_s>
MODEL=${1:-piper}
PARAMS=${2:-'{}'}
TIMEOUT=${3:-300}
TEXT='Oh my goodness, just a moment dear, I need to find my reading glasses. Now, you said I owe money to the IRS? Can you give me that case number again, nice and slow? My son always tells me to write these things down.'

BODY=$(python3 -c "import json,sys; print(json.dumps({'text':sys.argv[1],'params':json.loads(sys.argv[2])}))" "$TEXT" "$PARAMS" 2>/dev/null)

T0=$(date +%s%3N)
RESP=$(curl -sf -m "$TIMEOUT" -X POST http://localhost:8001/synthesize/"$MODEL" \
  -H "Content-Type: application/json" -d "$BODY" 2>&1)
RC=$?
T1=$(date +%s%3N)

if [ $RC -ne 0 ]; then
  echo "$MODEL|FAIL|timeout_or_error|0|0|0|0"
  exit 0
fi

RTF=$(echo "$RESP"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('rtf',0))" 2>/dev/null || echo 0)
SYNTH=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('synth_time_ms',0))" 2>/dev/null || echo 0)
AUDIO=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('audio_dur_ms',0))" 2>/dev/null || echo 0)
LOAD=$(echo "$RESP"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('load_time_s',0))" 2>/dev/null || echo 0)
HZ=$(echo "$RESP"      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sample_rate',0))" 2>/dev/null || echo 0)
ERR=$(echo "$RESP"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "")

if [ -n "$ERR" ] && [ "$ERR" != "None" ] && [ "$ERR" != "null" ]; then
  echo "$MODEL|FAIL|$ERR|0|0|0|0"
else
  echo "$MODEL|PASS|$RTF|$SYNTH|$AUDIO|$LOAD|$HZ"
fi
