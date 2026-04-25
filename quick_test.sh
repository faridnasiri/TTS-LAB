#!/bin/bash
# Quick synthesis test - NO restart, just sends requests to already-running server
echo '{"text":"Hello world.","params":{}}' > /tmp/simple_req.json
for engine in piper kokoro melo xtts cosyvoice fishspeech parler indextts qwen3tts openvoice; do
  curl -s -X POST http://localhost:8001/synthesize/$engine \
    -H 'Content-Type: application/json' -d @/tmp/simple_req.json \
    -o /tmp/fresh_${engine}.json --max-time 90
  rtf=$(grep -o '"rtf":[0-9.]*' /tmp/fresh_${engine}.json | head -1)
  err=$(grep -o '"error":"[^"]*"' /tmp/fresh_${engine}.json | head -1 | cut -c1-80)
  echo "$engine: ${rtf:-NO_AUDIO}  ${err}"
done
