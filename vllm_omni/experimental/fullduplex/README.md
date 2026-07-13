# Experimental Full-Duplex Runtime

This package contains two experimental integrations:

- the existing JoyVL framework and example integration;
- the MiniCPM-o 4.5 native audio path used by `/v1/duplex` and
  `/v1/realtime?duplex=1`.

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).
For the MiniCPM production path, lifecycle invariants, capability boundary, and
review commands, see
[`docs/design/minicpmo45_full_duplex_runtime_review.md`](../../../docs/design/minicpmo45_full_duplex_runtime_review.md).

## Package boundaries

```text
core/       shared DuplexFence plus the existing JoyVL framework
engine/     AsyncOmni/orchestrator scheduler data-plane adapter
openai/     WebSocket transport, Realtime projection, and audio codecs
minicpmo45/ MiniCPM input framing, policy, compatibility, and Stage0 state
joyvl/      JoyVL model-specific integration
web/        MiniCPM browser demo and WebSocket proxy
```

MiniCPM does not run through `core.DuplexRuntime`. Its active path uses the
`openai` session controller, the `engine` compatibility adapter, the standard
orchestrator/model runners, and the model-specific `minicpmo45` helpers.

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
