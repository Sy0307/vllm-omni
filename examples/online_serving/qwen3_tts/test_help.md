# Qwen3-TTS Test Help

## 1) Start server

```bash
cd /home/sy03/bpftime_sy03/vllm-omni-tts_disaggregation
VLLM_OMNI_PYTHON=/home/sy03/bpftime_sy03/vllm-omni/.venv/bin/python \
  ./examples/online_serving/qwen3_tts/run_server.sh CustomVoice
```

## 2) Manual request smoke test

```bash
cat > /tmp/qwen3tts_customvoice_req.json <<'JSON'
{"input":"你好，世界。","task_type":"CustomVoice","voice":"Vivian","response_format":"wav","max_new_tokens":256}
JSON

curl -sS -o /tmp/qwen3tts_customvoice.wav \
  -H "Content-Type: application/json" \
  -d @/tmp/qwen3tts_customvoice_req.json \
  http://127.0.0.1:8000/v1/audio/speech

ls -lah /tmp/qwen3tts_customvoice.wav
file /tmp/qwen3tts_customvoice.wav
```

## 3) Script-equivalent test flow (from previous test script)

```bash
PORT=8000
TASK_TYPE=CustomVoice
VOICE=Vivian
MAX_NEW_TOKENS=256
TEXT="你好，世界。"
OUT=/tmp/qwen3tts_test.wav
REQ="$(mktemp /tmp/qwen3tts_req.XXXXXX.json)"

cat > "$REQ" <<JSON
{"input":"${TEXT}","task_type":"${TASK_TYPE}","voice":"${VOICE}","response_format":"wav","max_new_tokens":${MAX_NEW_TOKENS}}
JSON

curl -sS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

HTTP_CODE="$(curl -sS -o "$OUT" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d @"$REQ" \
  "http://127.0.0.1:${PORT}/v1/audio/speech")"

echo "HTTP=${HTTP_CODE}"
ls -lah "$OUT"
file "$OUT"
```

## 4) Quick waveform stats

```bash
python - <<'PY'
import wave
import numpy as np
p = '/tmp/qwen3tts_test.wav'
with wave.open(p, 'rb') as wf:
    sr = wf.getframerate()
    n = wf.getnframes()
    x = np.frombuffer(wf.readframes(n), dtype='<i2').astype(np.float32) / 32768.0
rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
peak = float(np.max(np.abs(x))) if x.size else 0.0
print(f"sr={sr} frames={n} dur_s={n/sr:.3f} rms={rms:.4f} peak={peak:.4f}")
PY
```
