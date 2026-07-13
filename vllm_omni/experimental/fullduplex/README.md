# Experimental Full-Duplex Runtime

This package contains the experimental session-oriented runtime used by the
MiniCPM-o 4.5 native duplex path and the JoyVL example integration.

The current architecture, lifecycle invariants, MiniCPM Stage0/Stage1 data
flow, deploy configuration, validation evidence, and reviewer reproduction
steps are documented in:

[`docs/design/minicpmo45_full_duplex_runtime_review.md`](../../../docs/design/minicpmo45_full_duplex_runtime_review.md)

The existing JoyVL integration recipe remains available at
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

## Package boundaries

```text
core/       JoyVL session runtime plus the shared immutable DuplexFence
engine/     current vLLM-Omni scheduler/orchestrator adapter
openai/     Realtime protocol projection and WebSocket transport
minicpmo45/ MiniCPM-o model policy and Stage0/Stage1 runtime adapters
joyvl/      JoyVL model-specific implementation
```

The original `core` runtime remains the model-agnostic base used by JoyVL;
MiniCPM-o does not run through a second generic reducer. Model token IDs, input
framing, and stage state belong in the model package. Scheduler request details
belong in `engine`. OpenAI event names and audio codecs belong in `openai`.

## Browser demo

With the MiniCPM-o backend listening on port `8099`, start the canonical
experimental browser client with:

```bash
python -m vllm_omni.experimental.fullduplex.web \
  --port 7862 \
  --ws-backend ws://127.0.0.1:8099
```

Open `http://<host>:7862/`. In Cloud Studio use the complete proxy URL:
`https://<aop-host>/aoplab/<workspace>/studio/proxy/7862/`. The page uses a
same-origin WebSocket path, so the proxy prefix is retained automatically.

The page exposes the currently supported model-policy session only. It does
not present automatic/VAD barge-in or multi-session controls.

## Scope

The verified MiniCPM-o checkpoint supports model-owned listen/speak on the
normal auto-response path and clean multi-turn native audio streaming. Its
capability contract intentionally disables explicit and automatic/VAD
barge-in and does not advertise multi-session support. Scheduler-native append,
bounded long-session KV, and production multi-session concurrency remain
follow-up work.
