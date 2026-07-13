# MiniCPM-o 4.5 Native Duplex Runtime Review

## Scope

PR #3907 adds an experimental MiniCPM-o 4.5 path for continuous audio input,
model-owned `listen`/`speak` decisions, streamed speech output, and sequential
turns within one WebSocket session. It uses the existing vLLM-Omni stage
pipeline and scheduler data plane; it does not add a second model runtime.

The implementation is intentionally under
`vllm_omni.experimental.fullduplex`. Its current capability boundary is:

| Capability | Current contract |
| --- | --- |
| OpenAI-style Realtime audio session | Supported for the implemented event subset |
| Sequential clean turns in one session | Supported |
| Model-native `listen`/`speak` policy | Supported on the normal auto-response path |
| Resumable scheduler request | Supported by the compatibility data plane |
| Scheduler-native append / persistent KV lease | Not supported |
| Automatic or VAD-driven barge-in | Not supported or advertised |
| Production multi-session concurrency | Not supported or advertised |
| Bounded long-session KV | Not implemented |

`supports_scheduler_native_append`, `supports_barge_in`,
`supports_multi_session`, and `supports_multi_session_same_replica` are all
`false` for MiniCPM-o 4.5. A session registry being able to store multiple
logical sessions is not a concurrency, fairness, isolation, or capacity
guarantee.

## Production Architecture

### Ownership boundaries

```text
vllm_omni/experimental/fullduplex/
  core/       shared immutable DuplexFence and the existing JoyVL framework
  engine/     AsyncOmni/orchestrator compatibility adapter
  openai/     WebSocket actor, Realtime projection, audio codecs, session control
  minicpmo45/ MiniCPM input framing, policy, compatibility, and Stage0 state
  joyvl/      existing JoyVL integration
  web/        browser demo and same-origin WebSocket proxy
```

The MiniCPM production path does not use `core.DuplexRuntime`. The `core`
runtime remains part of the existing JoyVL framework. MiniCPM uses the
`openai` session controller, the `engine` data-plane adapter, the standard
orchestrator and model runners, and its model-specific `minicpmo45` helpers.

The main ownership rules are:

- OpenAI event names, response lifecycle, playback acknowledgement, and audio
  format conversion belong to `openai`.
- Model token IDs, one-second audio framing, reference-audio preparation, and
  Stage0 state belong to `minicpmo45`.
- Request append, output collection, cancellation, and stage binding belong to
  the engine/orchestrator boundary.
- The normal model runner owns attention metadata, sampling, and request KV.
- Stage1/TTS owns turn-local consumed cursors, Token2Wav state, and vocoder
  streaming state.

### Request path

The active path is:

```text
Realtime or /v1/duplex WebSocket
  -> DuplexSessionActor
  -> MiniCPMO45PcmAppendBuffer
  -> AsyncOmni append_duplex_input
  -> orchestrator scheduler data plane
  -> standard Stage0 model runner
  -> Stage0-to-Stage1 handoff
  -> Stage1 / Token2Wav
  -> AsyncOmni output collection
  -> Realtime response projection
```

Audio appends update a stable resumable stage request for the active session
and epoch. This is a compatibility mechanism over current scheduler requests,
not a scheduler-native append primitive and not a core KV lease.

The native duplex deployment uses the standard deploy-config pipeline:

```text
vllm_omni/deploy/minicpmo_4_5_duplex.yaml
```

It sets `session_mode: duplex`, uses one active stream window, and currently
selects synchronous scheduling for both stages. Other deploy configurations
keep the default `session_mode: turn` behavior.

## Identity and Lifetimes

`DuplexFence(session_id, epoch, turn_id, response_seq)` is the cross-layer
identity used by serving, engine messages, orchestrator state, stage metadata,
output cursors, and teardown operations.

The fields have distinct lifetimes:

| Field | Advances when | Purpose |
| --- | --- | --- |
| `session_id` | a WebSocket session opens | owns the conversation lifetime |
| `epoch` | an interruption invalidates prior work | rejects stale output |
| `turn_id` | a new user turn is committed | keys turn-local model output |
| `response_seq` | a new assistant response starts | keys protocol lifecycle |

A clean `response.done` closes one assistant response. It must not destroy the
WebSocket session or Stage0 conversational context required by the next turn.
An epoch change invalidates output from the previous interruption generation.

## Turn and Response Lifecycle

For `auto_response=true`, the client appends audio and commits the input turn;
it does not also send `response.create`. MiniCPM then samples its native
`listen` or `speak` decision. A commit is therefore not a promise that the
model will speak.

The normal spoken path is:

```text
input_audio_buffer.append ...
input_audio_buffer.commit
model speak
response.created
response.output_item.added
response.content_part.added
response.audio.delta / response.audio_transcript.delta ...
response.audio.done
response.content_part.done
response.output_item.done
response.done
```

The following invariants are required:

1. One fenced response emits `response.created`, `response.speak`, and each
   terminal event at most once.
2. TTS `segment_end` may flush a speech segment but cannot close the assistant
   turn.
3. Model `turn_eos` is the authoritative clean turn boundary.
4. Turn end resets all Stage1 turn-local cursors and vocoder state together.
5. Clean turn end preserves Stage0 conversation state.
6. Empty or terminal-only turns cannot replay text or audio from a prior turn.
7. Audio cannot arrive after the response terminal event.
8. Missing or mismatched fence metadata fails or stale-drops; it must not
   silently disable lifecycle protection.

MiniCPM's normal auto-response path leaves `listen`/`speak` selection to the
model. Some compatibility paths can apply a `force_listen` safety override;
that is not evidence for model-native or automatic barge-in.

## Interruption Boundary

The runtime has epoch fencing, stale-output rejection, response cancellation,
and playback-aware history bookkeeping. Those mechanisms are necessary
infrastructure, but MiniCPM-o 4.5 currently has no validated automatic or VAD
interruption source. Explicit MiniCPM barge-in requests are not advertised as
a supported capability.

Any future interruption implementation must use the existing fence transition:

1. capture the old fence;
2. preserve only playback-committed assistant history;
3. advance the epoch;
4. cancel and release old-fence work;
5. reject late old-fence output;
6. resume input under the new fence.

This PR does not claim that sequence as a completed MiniCPM barge-in E2E.

## Reviewer Map

Start with these files:

- `vllm_omni/experimental/fullduplex/openai/serving.py`: session and engine
  orchestration.
- `vllm_omni/experimental/fullduplex/openai/realtime_session.py`: Realtime
  event projection and response state.
- `vllm_omni/experimental/fullduplex/openai/protocol.py`: session schema and
  advertised capabilities.
- `vllm_omni/experimental/fullduplex/engine/omni.py`: scheduler data-plane
  compatibility adapter.
- `vllm_omni/engine/orchestrator.py`: stage request and replica binding.
- `vllm_omni/experimental/fullduplex/minicpmo45/`: MiniCPM-specific policy,
  input framing, and Stage0 helpers.
- `vllm_omni/model_executor/models/minicpmo_4_5/`: model and Stage1/TTS
  behavior.
- `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`:
  Stage0-to-Stage1 handoff and turn metadata.
- `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`: deployment overlay.

Review the implementation against the lifecycle invariants above, especially
the separation between segment end, model turn end, response end, and session
end.

## Reproduction

### Start the server

```bash
python -m vllm_omni.entrypoints.cli.main serve \
  openbmb/MiniCPM-o-4_5 \
  --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099
```

### Model-policy E2E

Use distinct 16 kHz mono PCM WAV inputs and real-time pacing. This mode accepts
either `listen` or `speak` as the model outcome while checking session, fence,
and lifecycle integrity.

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav /path/to/turn1.wav \
  --turn-input-wav /path/to/turn2.wav \
  --turn-input-wav /path/to/turn3.wav \
  --turns 3 \
  --turn-duration-ms 0 \
  --turn-duration-ms 0 \
  --turn-duration-ms 0 \
  --chunk-ms 200 \
  --omit-transcript-hints \
  --realtime-input \
  --require-distinct-inputs \
  --validation-mode model-policy \
  --output-dir /tmp/minicpmo45_duplex_model_policy
```

Check that every turn produces an observable model outcome, created responses
have symmetric terminal lifecycles, no server `error` event occurs, and no
stale output, forced-speak fallback, timeout, or cross-turn tail appears.

### Response-required E2E

This mode must use a pinned fixture known to select `speak`; arbitrary audio is
not guaranteed to produce a response.

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav /path/to/pinned-turn1.wav \
  --turn-input-wav /path/to/pinned-turn2.wav \
  --turn-input-wav /path/to/pinned-turn3.wav \
  --turns 3 \
  --turn-duration-ms 0 \
  --turn-duration-ms 0 \
  --turn-duration-ms 0 \
  --chunk-ms 200 \
  --omit-transcript-hints \
  --realtime-input \
  --require-distinct-inputs \
  --require-audio \
  --validation-mode response-required \
  --output-dir /tmp/minicpmo45_duplex_response_required
```

Check each response independently: transcript delta equals transcript done,
audio precedes `response.done`, no audio follows it, no prior-turn suffix leaks
into the next response, and generated WAV content agrees with the transcript.
Use listening or an independent ASR pass when audio content is an acceptance
criterion.

### Focused tests

Run GPU/runtime tests in the supported CUDA environment:

```bash
pytest -q \
  tests/fullduplex \
  tests/engine/test_duplex_runtime.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/test_async_omni_duplex.py \
  tests/entrypoints/test_duplex_fence_propagation.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/examples/test_minicpmo_realtime_web.py
```

Deploy-config composition is covered separately by the relevant
`TestStageConfig` and `TestDeployConfigLoading` cases in
`tests/test_config_factory.py`.

## Not Declared by This Checkpoint

The following require separate design and validation before capabilities can
be enabled:

- an upstream scheduler-native append/session-KV API;
- automatic or VAD-driven MiniCPM barge-in;
- interruption-time Stage0 rollback and playback-history reconstruction E2E;
- bounded or windowed KV for minute-scale conversations;
- multi-session admission, fairness, isolation, and capacity guarantees;
- production-scale ASR, MOS, and speaker-similarity quality gates.

These limitations do not mean sequential turns in one session are disabled.
They define the boundary between the current experimental checkpoint and a
production full-duplex runtime.
