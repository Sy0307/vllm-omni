#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
TASK_TYPE="${TASK_TYPE:-CustomVoice}"
VOICE="${VOICE:-Vivian}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEXT="${1:-你好，世界。}"
OUT="${OUT:-/tmp/qwen3tts_test.wav}"
REQ="$(mktemp /tmp/qwen3tts_req.XXXXXX.json)"

cat > "$REQ" <<JSON
{"input":"${TEXT}","task_type":"${TASK_TYPE}","voice":"${VOICE}","response_format":"wav","max_new_tokens":${MAX_NEW_TOKENS}}
JSON

if ! curl -sS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] server not ready on 127.0.0.1:${PORT}" >&2
  echo "Start server first:" >&2
  echo "  VLLM_OMNI_PYTHON=/home/sy03/bpftime_sy03/vllm-omni/.venv/bin/python ./examples/online_serving/qwen3_tts/run_server.sh CustomVoice" >&2
  exit 1
fi

HTTP_CODE="$(curl -sS -o "$OUT" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d @"$REQ" \
  "http://127.0.0.1:${PORT}/v1/audio/speech")"

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "[ERROR] HTTP=${HTTP_CODE}" >&2
  exit 1
fi

ls -lah "$OUT"
file "$OUT"

python - <<'PY'
import wave, numpy as np
p = '/tmp/qwen3tts_test.wav'
with wave.open(p, 'rb') as wf:
    sr = wf.getframerate()
    n = wf.getnframes()
    x = np.frombuffer(wf.readframes(n), dtype='<i2').astype(np.float32) / 32768.0
rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
peak = float(np.max(np.abs(x))) if x.size else 0.0
print(f"sr={sr} frames={n} dur_s={n/sr:.3f} rms={rms:.4f} peak={peak:.4f}")
PY

echo "Request file: $REQ"
echo "Output wav : $OUT"
