# Experimental Full-Duplex Runtime

This package contains two experimental integrations:

- the existing JoyVL framework and example integration;
- the MiniCPM-o 4.5 native audio path used by `/v1/duplex` and
  `/v1/realtime?duplex=1`.

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).
For the MiniCPM active runtime path, lifecycle invariants, capability boundary,
and validation scope, see
[`docs/design/minicpmo45_duplex_runtime_architecture.md`](../../../docs/design/minicpmo45_duplex_runtime_architecture.md).

## Package boundaries

```text
core/       existing JoyVL framework and experimental compatibility exports
engine/     AsyncOmni/orchestrator scheduler data-plane adapter
openai/     WebSocket transport, Realtime projection, and audio codecs
minicpmo45/ MiniCPM input framing, policy, compatibility, and Stage0 state
joyvl/      JoyVL model-specific integration
web/        MiniCPM browser demo and WebSocket proxy
```

MiniCPM does not run through the removed experimental `core.DuplexRuntime`
facade. Its active path uses the `openai` session controller, the stable engine
duplex contract, the standard scheduler/model runners, and an injected
MiniCPM-specific runtime extension from `minicpmo45/runtime.py`.

## Browser demo

With the MiniCPM backend listening on port `8099`, run:

```bash
python -m vllm_omni.experimental.fullduplex.web \
  --port 7862 \
  --ws-backend ws://127.0.0.1:8099
```

Open `http://<host>:7862/`. The browser uses a same-origin WebSocket path, so a
reverse-proxy path prefix is retained automatically.

## Capability boundary

The MiniCPM checkpoint supports sequential clean turns in one session,
model-owned `listen`/`speak` decisions on the normal auto-response path, and
streamed audio responses over the implemented Realtime event subset.

It does not advertise scheduler-native append, a persistent KV lease,
automatic/VAD barge-in, production multi-session concurrency, or bounded
minute-scale KV. These remain separate design and validation work.

The Realtime adapter does not implement `server_vad` or `semantic_vad`.
Clients must set `turn_detection` to `null` and either commit input explicitly
or use the model-owned native duplex policy. Unsupported turn-detection objects
are rejected instead of being accepted and echoed without their advertised VAD
semantics.

## Realtime response contract

The experimental Realtime projection uses one text channel and one audio
channel per response:

| Event | Cardinality and ordering |
| --- | --- |
| `response.created` | Exactly once, before response content events. |
| `response.speak` | At most once; decision-only, with no transcript text. |
| `response.audio.delta` | Zero or more, before `response.audio.done`. |
| `response.audio_transcript.delta` | Zero or more; concatenation is the response transcript. |
| `response.audio.done` | Exactly once for a response that emitted audio. |
| `response.audio_transcript.done` | At most once; equals the concatenated transcript deltas. |
| `response.done` | Exactly once and terminal; no later audio is valid. |

`response.output_audio.delta` is the internal duplex dialect. Realtime clients
receive `response.audio.delta`; clients must not consume both dialects as
independent audio streams.
