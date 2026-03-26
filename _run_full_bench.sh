#!/usr/bin/env bash
# Full sequential benchmark — one model at a time, server restarts between RAM-heavy ones.
# Results written to /tmp/bench_full.tsv
set -euo pipefail

PY=/opt/arthur-bench-env/bin/python
SCRIPT=/opt/arthur/_bench_single.sh
OUT=/tmp/bench_full.tsv
RESTART_AFTER="outetts bark styletts2 f5tts dia xtts cosyvoice parler chatterbox"

echo "model|status|rtf|synth_ms|audio_ms|load_s|hz" > "$OUT"

restart_server() {
  echo "  [restart]" >&2
  systemctl restart arthur-lab
  for i in $(seq 1 20); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/ 2>/dev/null || true)
    [ "$CODE" = "200" ] && return 0
    sleep 3
  done
  echo "  [WARN] server not ready after restart" >&2
}

run_model() {
  local NAME=$1 PARAMS=$2 TIMEOUT=$3
  echo -n "  $NAME ... " >&2
  RES=$(bash "$SCRIPT" "$NAME" "$PARAMS" "$TIMEOUT" 2>/dev/null)
  echo "$RES" >> "$OUT"
  echo "$RES" >&2
}

# Light engines (fast, no restart needed)
run_model piper      '{"voice":"en_US-ryan-high"}'                                            120
run_model kokoro     '{"voice":"bm_lewis","speed":"0.85"}'                                    120
run_model melo       '{"speaker":"EN-US","speed":"0.85"}'                                     180

# Medium
run_model chattts    '{"prompt":"[speed_5]","temperature":"0.3","seed":"0"}'                  300
restart_server

run_model outetts    '{"model_path":"OuteAI/OuteTTS-0.3-500M","temperature":"0.4"}'          600
restart_server

run_model bark       '{"voice_preset":"v2/en_speaker_6"}'                                    600
restart_server

run_model styletts2  '{"alpha":"0.3","beta":"0.7","diffusion_steps":"5"}'                    300
restart_server

run_model f5tts      '{"speed":"1.0","nfe_step":"32"}'                                       600
restart_server

run_model dia        '{"cfg_scale":"3.0","temperature":"1.2"}'                               600
restart_server

run_model xtts       '{"speaker":"Torcull Diarmuid","language":"en"}'                        600
restart_server

run_model cosyvoice  '{"speaker":"English Female"}'                                          600
restart_server

run_model parler     '{"description":"An elderly man with a slow warm slightly confused voice speaks gently."}' 600
restart_server

run_model chatterbox '{"exaggeration":"0.65","cfg_weight":"0.5"}'                            600
restart_server

# New engines 14-21
run_model fishspeech '{"speed":"1.0"}'                                                       600
restart_server

run_model qwen3tts   '{}'                                                                    600
restart_server

run_model orpheus    '{"voice":"tara"}'                                                      600
restart_server

run_model zonos      '{"variant":"transformer","speaking_rate":"13.0"}'                     600
restart_server

run_model openvoice  '{"speaker":"EN-US","speed":"0.85"}'                                    600

echo ""
echo "Done → $OUT"
cat "$OUT"
