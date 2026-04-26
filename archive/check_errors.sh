#!/bin/bash
echo '{"text":"Hello world.","params":{}}' > /tmp/simple_req.json
for engine in openvoice parler indextts qwen3tts; do
  curl -s -X POST http://localhost:8001/synthesize/$engine \
    -H 'Content-Type: application/json' -d @/tmp/simple_req.json \
    -o /tmp/result_${engine}.json --max-time 60
  echo "=== $engine ==="
  cat /tmp/result_${engine}.json | tr ',' '\n' | grep -v audio_b64 | grep -E '"error"|"rtf"|"trace"' | head -4
done
