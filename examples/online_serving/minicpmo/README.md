# MiniCPM-o 4.5: Online serving

OpenAI-compatible `/v1/chat/completions` serving for **MiniCPM-o 4.5**, plus a
Gradio UI and curl / Python clients.

Inputs: text / image / audio / video. Outputs: text and optional **24 kHz** speech.

## Installation

Please refer to [README.md](../../../README.md). Install the talker extra:

```bash
pip install 'vllm-omni[minicpmo]'
```

## Launch the server

Pick a deploy config that matches your GPU layout:

| config | GPUs | TP | Notes |
|---|---|---|---|
| `minicpmo_4_5.yaml` | 1 | 1 | Thinker and talker+t2w co-located on GPU0. |
| `minicpmo_4_5_2gpu.yaml` | 2 | 1 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | 2 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, talker+t2w on GPU4. |
| `minicpmo_4_5_3gpu_stage1_replicas.yaml` | 3 | 1 | Thinker on GPU0, two talker+Token2wav replicas on GPU1/2 for concurrent text+audio serving. |
| `minicpmo_4_5_4gpu_stage1_replicas.yaml` | 4 | 1 | Thinker on GPU0, three talker+Token2wav replicas on GPU1/2/3. |
| `minicpmo_4_5_8x4090_stage1_replicas.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, four talker+Token2wav replicas on GPU4-7. |

Then:

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

For production or internal networks where Hugging Face downloads are slow, pass
a local ModelScope-downloaded checkpoint path instead of `openbmb/MiniCPM-o-4_5`.

### TTS throughput notes

MiniCPM-o 4.5's remote-code `MiniCPMTTS.generate()` currently runs as a
single-request whole-waveform path, so the deploy configs keep Stage1
`max_num_seqs: 1`. Use the `*_stage1_replicas.yaml` configs to scale concurrent
text+audio throughput horizontally.

```bash
vllm-omni serve /path/to/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_4gpu_stage1_replicas.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Talker/token2wav runtime behavior uses checked-in defaults rather than
MiniCPM-specific environment variables. If these knobs need to be exposed, add
them through a first-class stage/model config so deployments have one clear
configuration surface.

Request-level reference audio is cached by content hash before it is passed to
Token2wav. This keeps repeated requests with the same voice prompt from
thrashing Token2wav's prompt cache while still resetting the cache when the
reference audio changes.

### Experimental native duplex

MiniCPM-o 4.5 supports the experimental `/v1/duplex` and
`/v1/realtime?duplex=1` WebSocket entry points. The native path streams audio
through a resumable scheduler data-plane request and forwards Stage0 output
through the existing Stage1/TTS pipeline.

Start the server with the duplex-specific deploy config. The regular
`minicpmo_4_5.yaml` deploy does not opt into duplex sessions and keeps the
non-streaming Stage1 token budget.

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Clients enable the model-native path explicitly:

```json
{"extra_body": {"minicpmo45_native_duplex": true}}
```

The normal vLLM model runner still owns attention metadata, sampling, and
request KV. The current append path is not a scheduler-native append primitive
or persistent KV lease. The browser continuously uploads PCM while unmuted,
including during assistant playback; it does not run VAD or generate
`input_audio_buffer.commit`. MiniCPM owns listen/speak progression at model-unit
boundaries. This is not a deterministic VAD-triggered barge-in guarantee, and
the checked-in duplex profile is validated only for its server-owned limit of
two concurrent sessions, not production-scale fairness or arbitrary capacity.
See
[`docs/design/minicpmo45_duplex_runtime_architecture.md`](../../../docs/design/minicpmo45_duplex_runtime_architecture.md)
for the active runtime path, lifecycle invariants, capability boundary, and
validation scope.

Fresh response-scoped H20 measurements put engine Stage0 TTFT at approximately
31-310 ms, TPOT at 15-21 ms, ITL p95 at 16-33 ms, explicit input-commit to first
audio at 387-1033 ms, and maximum client-observed audio chunk gaps at 388-1316
ms. These small warm-state-sensitive samples are not SLOs, and the slowest
first-audio result does not meet a 200 ms target. The demo reports
`response_created_to_first_audio_ms` and `commit_to_first_audio_ms` separately;
do not call the former end-to-end TTFP or infer TPOT from transcript deltas.

### Stage-based CLI (optional)

Stage 0 (thinker + API) and stage 1 (talker) can run in separate processes:

```bash
# Stage 0
CUDA_VISIBLE_DEVICES=0 vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code --port 8099 --stage-id 0 \
    --omni-master-address 127.0.0.1 --omni-master-port 26000

# Stage 1 (headless)
CUDA_VISIBLE_DEVICES=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code --stage-id 1 --headless \
    --omni-master-address 127.0.0.1 --omni-master-port 26000
```

### Per-stage overrides

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8099 \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.65}}'
```

## Send multimodal requests

```bash
cd examples/online_serving/minicpmo
```

### curl

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio '["text"]'   # text-only
```

Text + speech smoke test (TTS needs top-level `chat_template_kwargs`):

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello, then introduce vLLM in one sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": true}
    }'
```

### Python client

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --port 8099 \
    --host localhost

# Text-only (faster; no <|tts_bos|>)
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type text \
    --modalities text \
    --prompt "Briefly introduce yourself."
```

Shared helpers also work if you pass MiniCPM defaults yourself:

```bash
python ../openai_chat_completion_client_for_multimodal_generation.py \
    --model openbmb/MiniCPM-o-4_5 \
    --query-type text \
    --port 8099
```

(Note: the shared client does **not** set `use_tts_template`; prefer the
MiniCPM-specific client above for speech.)

### Gradio demo

```bash
bash run_gradio_demo.sh

# Or:
python gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862`. Uncheck **"Generate speech output (TTS)"** for
text-only responses.

## Modality control

| Modalities | Output |
|---|---|
| `["text"]` | Text only (no TTS bos) |
| `["text", "audio"]` / unset | Text + 24 kHz speech |

Speech requires `chat_template_kwargs.use_tts_template=true` so the chat
template appends `<|tts_bos|>`. For **curl**, put that field at the request
root; nested `extra_body` is ignored. The OpenAI Python SDK may use
`extra_body` because it merges those fields into the root.

## 3. Run the Realtime duplex demo

After the server is running, the user-facing demo streams one WAV, accepts the
model-owned speak or listen decision, and writes the received events and audio:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo
```

The demo relies on `auto_response`; it never sends `response.create` or a
serving-side barge-in signal. Reusable WebSocket, PCM, and event collection
helpers live in `vllm_omni.experimental.fullduplex.client`.

Strict lifecycle and scenario validation is kept with the E2E suite:

```bash
python tests/e2e/online_serving/minicpmo_realtime_duplex_scenarios.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/turn-1.wav \
    --turn-input-wav /path/to/turn-2.wav \
    --turn-input-wav /path/to/turn-3.wav \
    --turns 3 --require-distinct-inputs --require-audio

python tests/e2e/online_serving/run_minicpmo_realtime_duplex_multi_session.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/fallback_16k_mono_pcm16.wav \
    --session-input-wav /path/to/session-a.wav \
    --session-input-wav /path/to/session-b.wav \
    --session-expected-token expected-a \
    --session-expected-token expected-b \
    --sessions 2
```

The strict runner checks event ordering, transcript delta/done integrity,
stale output, overlap, playback/history acknowledgement, response cardinality,
session identity and semantic isolation, disconnect/resume, and expiry behavior.

## 4. Open the experimental browser client

The canonical browser UI lives with the experimental runtime. It serves the
page and proxies the same-origin Realtime WebSocket to the backend:

```bash
python -m vllm_omni.experimental.fullduplex.web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav
```

Open `http://<host>:7862/`. When using a reverse proxy, open the proxy URL that
maps to port `7862`. The browser derives its WebSocket endpoint relative to
that URL, preserving any proxy path prefix.

If the page proxy serves HTTP but does not forward WebSocket upgrades, point
the browser directly at a separately exposed Realtime endpoint:

```bash
python -m vllm_omni.experimental.fullduplex.web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --public-realtime-url wss://public.example/v1/realtime
```

Client behavior and options:

- **Prompt presets**: the system prompt defaults to the official
  MiniCPM-o-Demo presets — `Streaming Omni Conversation.` (omni preset, for
  camera + voice) with the audio-call personas (中文通话 / English Call)
  selectable, or fully custom text.
- **Reference voice**: `--ref-audio` points at a wav whose voice the TTS
  clones (the official demo defaults to its signature voice at
  `assets/ref_audio/ref_minicpm_signature.wav`). Without it the model's
  built-in timbre is used.
- **Camera**: the **Camera** button streams ~1 fps JPEG frames riding the
  audio appends (`video_frames` on `input_audio_buffer.append`, the official
  omni contract) so the model sees while it listens.
- **Continuous input**: the browser does not run VAD, send
  `input_audio_buffer.commit`, or request `response.create`. It streams PCM
  while unmuted and lets MiniCPM own listen/speak/turn progression.

## Notes

- Stage 1 is capped at `max_num_seqs: 1` in the deploy YAML (talker shares
  request-0 audio metadata).
- Output audio is base64 WAV in `message.audio.data` (24 kHz mono).
- Offline counterpart:
  [`examples/offline_inference/minicpmo/`](../../offline_inference/minicpmo/)
- Recipe:
  [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)
- **TTS trigger**: the demo sets
  `extra_body.chat_template_kwargs.use_tts_template=True`, which appends
  `<|tts_bos|>` to the assistant prefix.
- Uncheck **"Generate speech output (TTS)"** to get text-only responses
  (faster).
- The audio output is the raw WAV returned by the stage-1 talker +
  Token2Wav; sample rate is 24 kHz.
- Video input is forwarded as a base64 `video_url` entry; the server needs
  decord/torchvision to decode it.
