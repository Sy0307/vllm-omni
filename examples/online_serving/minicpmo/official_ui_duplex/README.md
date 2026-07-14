# Official MiniCPM-o Demo UI on the vLLM-Omni Realtime Duplex Endpoint

Run the **official OpenBMB/MiniCPM-o-Demo audio-duplex voice-chat page**
directly against vLLM-Omni's experimental `/v1/realtime?duplex=1` endpoint —
no official gateway, no worker process, no protocol bridge.

```text
official audio-duplex page (browser)
  -> serve.py            static host + gateway-API stubs + WS byte-proxy
  -> /v1/realtime?duplex=1   vLLM-Omni full-duplex runtime (this PR)
```

The trick: the official frontend isolates all transport in one ES module
(`static/duplex/lib/duplex-session.js`, a hook-based session class). The
bundled `realtime-duplex-session.js` is a drop-in replacement with the same
hook surface that speaks the realtime duplex protocol from the browser:

- mic chunks (f32le 16 kHz base64) -> `input_audio_buffer.append` (pcm16)
- `response.audio.delta` (pcm16 24 kHz) -> official `AudioPlayer` (f32le)
- `response.audio_transcript.delta` -> AI chat bubbles (transcript deltas
  only — `response.speak` text repeats them and would duplicate the text)
- `conversation.item.input_audio_transcription.completed` -> "You:" bubbles
- ref audio / presets / system prompt: official assets, unchanged

`serve.py` builds a **runtime overlay** of the official static tree (the
audio-duplex page only — other demo pages are not served), swaps the one
import line, and serves it. The official demo checkout is never modified,
and no third-party frontend code is vendored into this repository.

## 1. Serve MiniCPM-o 4.5 with the duplex deploy config

The checkpoint must be a duplex-capable MiniCPM-o 4.5 build: the tokenizer
must define the duplex special tokens (`<unit>`, `<|listen|>`, `<|speak|>`,
`<|tts_bos|>`, ...). If serving aborts with "native duplex requires
tokenizer-defined special tokens", your snapshot predates duplex support —
update the model files.

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Python deps for `serve.py` beyond vllm-omni's own: `soundfile` (and the
MiniCPM-o serving deps `librosa`, `stepaudio2-minicpmo`, `minicpmo`).

## 2. Get the official demo frontend

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
# tested against the June 2026 tree; pin if the upstream layout drifts:
git -C MiniCPM-o-Demo checkout c2c8acbf5cb44c3b9c248dde8109da584d0b2d3c
```

(Later MiniCPM-o-Demo revisions replaced `duplex-session.js` with a
different worker protocol; `serve.py` fails fast with a pin hint if the
checkout is incompatible.)

## 3. Serve the UI

```bash
python examples/online_serving/minicpmo/official_ui_duplex/serve.py \
    --port 8006 \
    --demo-root /path/to/MiniCPM-o-Demo \
    --ws-backend ws://127.0.0.1:8099
```

Open `http://localhost:8006` — on a remote box, tunnel first
(`ssh -N -L 8006:127.0.0.1:8006 <host>`; the mic requires a secure origin,
which `localhost` satisfies). Pick a preset (中文通话 / English call) or edit
the system prompt, press Start, and talk. Barge-in, waveform, voice presets,
reference-audio cloning, and session recording all work as in the official
deployment.

## Notes and limitations

- **Pause** is client-side only (stops sending mic audio); the realtime
  protocol has no server-side pause.
- **KV-cache metrics** are not shown — the realtime protocol does not expose
  `kv_cache_length`, so the auto-stop-at-KV-limit feature is inactive.
- `force_listen` is forwarded on appends for parity with the official
  worker protocol; server-side handling depends on the runtime.
- The session sends `session.update` with
  `extra_body: {auto_response: true, minicpmo45_native_duplex: true}` and,
  when a reference voice is selected, `extra_body.ref_audio` as a WAV data
  URI. Turn-taking is server-VAD driven.
