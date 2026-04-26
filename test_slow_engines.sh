#!/bin/bash
for e in indextts qwen3tts openvoice; do
  curl -s -X POST "http://localhost:8001/synthesize/$e" \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello world.","params":{}}' \
    -o "/tmp/r_${e}.json" -m 300
  rtf=$(grep -o '"rtf":[0-9.]*' "/tmp/r_${e}.json" | head -1)
  err=$(grep -o '"error":"[^"]*"' "/tmp/r_${e}.json" | head -1 | cut -c1-70)
  echo "$e: ${rtf:-NO_AUDIO}  $err"
done
