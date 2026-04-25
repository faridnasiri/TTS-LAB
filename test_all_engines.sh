#!/bin/bash
source /opt/arthur-bench-env/bin/activate
sudo systemctl restart arthur-lab
echo "Waiting for server..."
sleep 10
echo '{"text":"Hello world.","params":{}}' > /tmp/req.json
for engine in piper kokoro melo chattts outetts bark styletts2 f5tts xtts cosyvoice parler chatterbox fishspeech qwen3tts indextts zonos openvoice; do
  result=$(curl -s -X POST http://localhost:8001/synthesize/$engine \
    -H 'Content-Type: application/json' -d @/tmp/req.json --max-time 90)
  rtf=$(echo "$result" | grep -o '"rtf":[0-9.]*' | head -1)
  err=$(echo "$result" | grep -o '"error":"[^"]*"' | head -1 | cut -c1-70)
  echo "$engine: ${rtf:-NO_AUDIO}  ${err}"
done
