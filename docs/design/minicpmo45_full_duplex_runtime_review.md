# MiniCPM-o 4.5 Native Full-Duplex Runtime Review Guide

## 1. Purpose and Current Status

This document describes the full-duplex runtime work in PR #3907 and the
follow-up migration from the legacy `--stage-configs-path` deployment to the
standard deploy-config pipeline.

The implemented and remotely verified checkpoint is:

- MiniCPM-o 4.5 owns the `listen`/`speak` decision.
- An audio-only client can commit speech without sending `response.create`.
- One WebSocket session can complete at least three clean, distinct input turns.
- Stage0 conversation context survives clean turn boundaries.
- Stage1 TTS and Token2Wav state is reset at the model turn boundary.
- Text, audio, response, and scheduler lifetimes are separated.
- Duplex identity is carried as one `DuplexFence` across serving, engine,
  orchestrator, scheduler output, and stage handoff paths.
- The native duplex deployment uses `--deploy-config` and does not depend on the
  legacy MiniCPM streaming stage-config file.

This checkpoint does not claim:

- automatic or VAD-driven barge-in;
- scheduler-native KV append or a first-class persistent KV lease;
- bounded KV for minute-scale conversations;
- production multi-session capacity or concurrency guarantees.

Epoch fencing and stale-output rejection exist in the active path, but
MiniCPM-o 4.5 does not currently provide a validated automatic interruption
source. A clean multi-turn native duplex checkpoint must not be presented as a
validated barge-in implementation.

## 2. Base and Compatibility

The review branch is rebased on the current local `origin/main` used for this
checkpoint:

```text
origin/main: d3c47efc4f818931f29e5f1567c94c454561631c
runtime checkpoint before this cleanup:
             b14a01d6d3bc7299a822e3eefc5b97d1da37f77a
```

`origin/main` is an ancestor of the branch. The implementation was validated in
the remote CUDA environment with vLLM 0.25.0. The full-duplex
package remains under `vllm_omni.experimental` because the scheduler append and
production lifetime contracts are not upstream-stable APIs yet.

## 3. Design Principles

### 3.1 Separate logical lifetimes

The previous implementation mixed five different lifetimes. The revised design
defines them explicitly:

| Lifetime | Identity / owner | Ends when |
| --- | --- | --- |
| WebSocket session | `session_id` | client closes or runtime fails |
| interruption generation | `epoch` | interruption/context rebuild |
| user turn | `turn_id` | the next committed user input allocates a new turn |
| OpenAI response | `response_seq` / `response_id` | model `turn_eos` or failure |
| engine stage resource | fence-derived request ID | epoch/session close or stage failure |

An OpenAI response is not the scheduler request. Clean turns reuse session-level
engine resources while each committed user turn gets a new external response.
This is what allows `response.done` to close one assistant reply without
destroying the session or the Stage0 conversational context needed by the next
turn.

### 3.2 One cross-layer identity

`vllm_omni.experimental.fullduplex.core.identity.DuplexFence` is the canonical
cross-layer identity:

```python
DuplexFence(
    session_id: str,
    epoch: int,
    turn_id: int,
    response_seq: int,
)
```

The fence is attached to engine messages, orchestrator request state, stage
bindings, multimodal metadata, output events, cursors, and teardown operations.
Missing fence metadata is an error on fenced duplex paths. A valid old fence is
stale output and is dropped; an impossible newer or mismatched fence fails
loudly.

Clean turns advance `turn_id` and `response_seq` while preserving `epoch`.
Interruption advances `epoch` atomically and invalidates all prior output.

### 3.3 Keep normal turn sampling model-owned

MiniCPM-o samples native `listen` and `speak` tokens. On the validated
`auto_response` path, serving does not make the model speak by rewriting
probabilities, forcing a token, or converting a `listen` decision into a
response. The Realtime session controller performs transport and lifecycle
operations; MiniCPM Stage0/Stage1 adapters expose model outputs to that active
path.

The non-auto-response short-commit and overlap compatibility paths may set
`force_listen`. This does not rewrite the sampled token; it prevents the
MiniCPM continuation guard from converting a sampled `listen` transition back
to TTS BOS. It is a serving safety override, not model-native turn policy, and
is not used as evidence for automatic barge-in support.

The implementation therefore removed the successful-run dependency on:

- `listen_prob_scale=0.0` as a forced-speak workaround;
- serving-side `force_speak` turn policy;
- client `response.create` in `auto_response` mode;
- punctuation or TTS segment completion as the assistant turn boundary.

### 3.4 Isolate the current scheduler workaround

Current vLLM does not expose the scheduler-native append/session-KV primitive
needed by the target architecture. The existing resumable request and scheduler
data-plane behavior is isolated in `experimental.fullduplex.engine.omni` and the
fenced AsyncOmni/orchestrator methods. Placeholder-token accounting remains at
that engine boundary rather than in OpenAI protocol or MiniCPM turn policy.

A future native append implementation can replace these compatibility helpers
without changing the Realtime response lifecycle or model adapters. The PR does
not carry an unconnected second engine-port/reducer implementation.

## 4. Package Boundaries

```text
vllm_omni/experimental/fullduplex/
  core/
    identity.py       immutable cross-layer `DuplexFence`
    adapter.py        original JoyVL adapter contract
    runtime.py        original JoyVL session runtime
    session.py        original JoyVL session state
  engine/
    omni.py           current orchestrator/scheduler data-plane adapter
    intermediate.py   typed Stage0 -> Stage1 payload helpers
  openai/
    protocol.py       duplex protocol schema and session registry
    realtime_session.py Realtime input/output schema projection
    websocket.py      reader/writer queues and task ownership
    serving.py        session controller and engine/protocol orchestration
    audio.py          audio format conversion
  minicpmo45/
    policy.py         token names, framing, and scheduler accounting rules
    input.py          PCM chunk buffering and commit accounting
    compat.py         checkpoint config compatibility fixes
    adapter.py        serving-side session preparation
    stage0.py         scheduler-owned Stage0 input builder
  joyvl/
    ...               existing integration using the original core runtime
```

The active MiniCPM path owns Realtime lifecycle in `openai/protocol.py` and
`openai/serving.py`, while `DuplexFence` is the cross-layer identity. Model token
IDs, MiniCPM audio-unit sizing, reference-audio setup, Stage0 state, and Stage1
handoff rules stay inside `minicpmo45`. The original JoyVL runtime remains
separate and unchanged.

`openai/serving.py` remains the largest integration module because it projects
the existing `/v1/duplex` and `/v1/realtime?duplex=1` protocols onto the engine
adapter. It no longer contains the model Stage0/Stage1 implementation, audio
codecs, WebSocket actor, or Realtime schema state. Further reduction of this
controller is possible, but is independent of scheduler-native append work.

MiniCPM does not install a second loaded-model provider lifecycle. Stage0 builds
audio embeddings during the normal model `preprocess()` call, and the existing
runner owns attention metadata, sampling, and request KV. Stage0 output then
continues through the normal orchestrator pipeline into the model's Stage1/TTS
implementation. Open, signal, and close remain session/control-plane operations;
they do not invoke parallel worker RPC wrappers.

## 5. State Machine and Normal Turn Flow

The active Realtime session controller follows these logical turn phases:

```text
IDLE
  -> INPUT_STREAMING
  -> TURN_COMMITTED
  -> AWAITING_MODEL
  -> RESPONDING
  -> IDLE
```

The normal native audio flow is:

1. `input_audio_buffer.append` is decoded and normalized to 16 kHz
   `pcm_f32le`.
2. `MiniCPMO45PcmAppendBuffer` emits only complete model audio units. It tracks
   per-turn speech independently from residual bytes; the serving loop tracks
   input and speech observed since the last commit.
3. Incremental units are appended with `final=False` to the stable fenced
   engine session.
4. `input_audio_buffer.commit` advances the logical turn and reserves one
   response when `auto_response` is enabled and the turn had speech.
5. Auto-response does not depend on `flush()` returning residual PCM. A commit
   exactly on a model-unit boundary is still a valid committed speech turn.
6. The model continues its native loop and samples `listen` or `speak`.
7. The first `speak` transition creates exactly one external response.
8. Stage0 text/hidden-state handoffs drive Stage1, which streams PCM/audio
   deltas.
9. TTS segment end closes the segment only.
10. Model `turn_eos` closes audio/content/response exactly once and resets
    turn-local Stage1 state.
11. The WebSocket session and Stage0 conversational state remain available for
    the next input commit.

In `auto_response=true`, the client sends audio append and commit events only.
Sending a manual `response.create` at the same time would enable two response
drivers and is intentionally rejected instead of being timed around an active
response.

## 6. Stage0 and Stage1 Data Flow

### 6.1 Stage0 ownership

Stage0 owns the model-native continuous state:

- streaming audio encoder state;
- thinker/talker model state used for native listen/speak decisions;
- conversational KV and model context across clean turns;
- current turn-ended latch;
- accumulated Stage0 TTS conditioning for the active turn.

Stage0 state is session-scoped unless explicitly documented as turn-local.
Clean `turn_eos` must not erase conversational context. An epoch change may
rebuild Stage0 from playback-committed history.

### 6.2 Stage0 to Stage1 handoff

`stage_input_processors/minicpmo_4_5_omni.py` converts the Stage0 multimodal
output into the existing Stage1 input shape. The handoff carries the complete
accumulated TTS condition for the active turn because Stage1 tracks a consumed
cursor. It does not treat a new handoff as a new conversation turn.

Handoff metadata includes flat scalar keys such as:

- `meta.duplex_epoch`
- `meta.duplex_turn_id`
- `meta.segment_end`
- `meta.turn_end`
- `meta.tts_is_last_chunk`

Flat metadata avoids the previous nested/flat merge mismatch and passes through
the output processor's explicit metadata handling.

### 6.3 Stage1 ownership

Stage1 owns turn-local speech generation state:

- consumed-token cursor into the cumulative handoff;
- talker LM turn state;
- Token2Wav token buffer and vocoder stream state;
- audio offset and text/audio alignment metadata.

These are reset on model turn end or a fenced interruption. They are not reset
at punctuation, a TTS segment boundary, or every engine output batch.

### 6.4 Token2Wav continuity fix

The prior path could finalize or clear Token2Wav at ordinary punctuation
segments. That broke multi-segment speech and could replay an earlier turn's
tail when the next turn was short or empty.

The revised contract distinguishes:

- `segment_end`: one TTS segment is complete; retain turn state;
- `turn_end`: the model sampled its turn EOS; finalize and clear turn state.

The consumed cursor and cumulative handoff are reset together at the turn
boundary. Stage0 KV is deliberately preserved. This prevents both directions
of the bug: prior-turn leakage and cross-turn amnesia.

## 7. Response and Protocol Lifecycle

The Realtime projection enforces one lifecycle per fenced response:

```text
response.created
response.output_item.added
response.content_part.added
response.audio.delta / transcript delta ...
response.audio.done
response.content_part.done
response.output_item.done
response.done
```

Client audio output uses one event family:
`response.audio.delta/done` and
`response.audio_transcript.delta/done`. The serving controller's internal
`response.output_audio.*` events are projected at the Realtime boundary and are
not also exposed to the client. Removed legacy/output event query and session
switches are ignored. PCM, WAV, and G.711 conversion is implemented only in
`openai/audio.py`.

Per-response state and terminal sets in the active protocol/session controller
make terminal events idempotent. `response.created` is keyed by the fenced
response rather than by a raw string grep count, which avoids both actual
duplicate creation and false diagnosis from nested event payload text.

Text and audio use per-fence cursors. A new turn cannot reuse the previous
turn's cumulative text cursor, audio offset, or text/audio marks. Late output
after `response.done` is rejected or stale-dropped depending on its fence.

Playback acknowledgement tracks four monotonic positions: generated, sent,
played, and committed. Only playback-committed assistant history is retained
by serving for conversation continuation after interruption. This contract is
present even though automatic barge-in is out of scope for this checkpoint.

## 8. Interruption Contract and Current Barge-In Scope

The fenced serving/engine contract defines one interruption transition:

1. capture the old fence;
2. commit only playback-visible assistant history in serving;
3. increment `epoch` for subsequent input;
4. signal and release runtime work with the captured old fence;
5. stale-drop late old-fence output and reset old Stage1 state;
6. accept subsequent input and output only under the new fence.

The current scheduler data plane does not implement Stage0 KV rollback or
history reconstruction. Conversation truncation remains serving-owned; runtime
signals intentionally carry only event, fence, and timeout.

There is no independently advertised barge-in capability in this checkpoint. A
future VAD, model signal, or explicit client control must enter through this
same fenced transition rather than create a second response lifecycle.

MiniCPM-o 4.5 currently exposes model-owned `listen`/`speak`, but the validated
official loop does not provide a reliable, explicit automatic barge-in event.
For that reason, its capability payload reports `supports_barge_in=false` and
`target_barge_in_latency_ms=null`. MiniCPM sessions default to `listen_only`;
explicit `barge_in`, `turn.signal(barge_in)`, and input-side barge-in hints are
rejected or deferred as listen input. Generic response cancellation and the
fencing mechanics remain unit-tested infrastructure, not an E2E barge-in
capability claim.

## 9. Engine and Scheduler Integration

The current engine adapter exposes typed fenced operations:

- open duplex session;
- append duplex input;
- signal turn lifecycle;
- stream fenced output;
- cancel a fence;
- close session resources.

The orchestrator binds one replica per stage for a session/epoch and derives
stage request IDs from the fence. It rejects missing fence state and keeps
Stage0 -> Stage1 output identity intact.

This is still a compatibility implementation over current scheduler requests.
It does not claim a core KV lease or scheduler-native append. The capability
surface reports `supports_scheduler_native_append=false` while retaining
`supports_core_resumable_request=true`, the `scheduler_data_plane` adapter
pattern, and scheduler data-plane stage handoff. Callers therefore cannot infer
a stronger runtime guarantee from a successful demo.

The session registry can track more than one logical session, but this
checkpoint has no validated admission, fairness, isolation, or capacity
contract for concurrent MiniCPM sessions. Its capability payload therefore
reports `supports_multi_session=false` and
`supports_multi_session_same_replica=false`. These fields can be enabled only
after concurrent E2E coverage proves the corresponding runtime guarantees.

The MiniCPM duplex deployment selects the synchronous AR scheduler for both
stages. This is deliberate, not a general recommendation. The current
resumable-request bridge requires serialized stage admission with
`active_stream_window: 1`; the async scheduler path previously admitted
overlapping lifecycle work that this compatibility layer could not fence
reliably. This does not imply one response per commit. MiniCPM owns turn policy,
and one committed input may produce `listen`, an empty terminal model turn, or
a spoken response. Async scheduling can be re-enabled only after a separate
scheduler-level identity and ordering contract is validated end to end.

## 10. Deploy-Config Migration

### 10.1 Why migrate

The legacy launch required:

```text
--stage-configs-path \
  vllm_omni/model_executor/stage_configs/minicpmo45_2gpu_streaming.yaml
```

That bypassed the current pipeline/deploy composition and left duplex session
mode outside the standard deployment schema.

The replacement is:

```text
--deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml
```

The deploy overlay reuses the registered `minicpmo_4_5` pipeline and only
overrides duplex-specific runtime behavior:

```yaml
base_config: minicpmo_4_5.yaml
pipeline: minicpmo_4_5
session_mode: duplex
active_stream_window: 1

stages:
  - stage_id: 0
    async_scheduling: false
  - stage_id: 1
    async_scheduling: false
    default_sampling_params:
      max_tokens: 4096
      extra_args:
        stop_token_names: ["<|im_end|>"]
```

`DeployConfig.session_mode` is propagated into every merged `StageConfig` and
its OmegaConf representation. Non-duplex deploys default to `session_mode:
turn`, preserving existing behavior.

`--trust-remote-code` remains an explicit launch option. MiniCPM-o 4.5 requires
it, but a model-specific deploy file must not silently weaken the global CLI
trust boundary.

### 10.2 Migration files

- Added `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`.
- Added new-schema Stage1 replica overlays for 3, 4, and 8 GPU layouts.
- Added `session_mode` parsing/propagation in
  `vllm_omni/config/stage_config.py`.
- Removed the duplicate legacy `minicpmo45_*.yaml` `stage_args` configs. The
  standard 2, 3, and 8 GPU layouts use the existing `minicpmo_4_5*.yaml`
  deploy configs; replica and duplex variants are thin overlays on those
  configs.
- Updated `examples/online_serving/minicpmo/README.md`.
- Added real deploy composition contract tests in
  `tests/test_config_factory.py`.

## 11. Review Map by Subsystem

### Shared identity and existing JoyVL runtime

- `vllm_omni/experimental/fullduplex/core/identity.py`
- `vllm_omni/experimental/fullduplex/core/runtime.py`
- `vllm_omni/experimental/fullduplex/core/adapter.py`
- `vllm_omni/experimental/fullduplex/core/session.py`
- `vllm_omni/experimental/fullduplex/joyvl/adapter.py`

### OpenAI Realtime and WebSocket projection

- `vllm_omni/experimental/fullduplex/openai/serving.py`
- `vllm_omni/experimental/fullduplex/openai/realtime_session.py`
- `vllm_omni/experimental/fullduplex/openai/websocket.py`
- `vllm_omni/experimental/fullduplex/openai/protocol.py`
- `vllm_omni/experimental/fullduplex/openai/audio.py`
- `vllm_omni/entrypoints/openai/api_server.py`
- `vllm_omni/entrypoints/openai/serving_chat.py`

### Engine, orchestrator, scheduler, and worker path

- `vllm_omni/experimental/fullduplex/engine/omni.py`
- `vllm_omni/experimental/fullduplex/engine/intermediate.py`
- `vllm_omni/experimental/fullduplex/engine/worker.py`
- `vllm_omni/engine/messages.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/engine/orchestrator.py`
- `vllm_omni/engine/output_processor.py`
- `vllm_omni/engine/stage_pool.py`
- `vllm_omni/core/sched/omni_ar_scheduler.py`
- `vllm_omni/worker/gpu_ar_model_runner.py`
- `vllm_omni/worker/gpu_model_runner.py`
- `vllm_omni/worker/mixins.py`

### MiniCPM-o model and bridge path

- `vllm_omni/experimental/fullduplex/minicpmo45/`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`
- `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`
- `vllm_omni/inputs/data.py`
- `vllm_omni/inputs/preprocess.py`
- `vllm_omni/utils/mm_outputs.py`
- `vllm_omni/data_entry_keys.py`

### Demo and verification

- `examples/online_serving/minicpmo/realtime_duplex_demo.py`
- `vllm_omni/experimental/fullduplex/web/`
- `tests/fullduplex/`
- `tests/entrypoints/openai/test_duplex_protocol.py`
- `tests/entrypoints/openai_api/test_duplex_handler.py`
- `tests/entrypoints/test_async_omni_duplex.py`
- `tests/entrypoints/test_duplex_fence_propagation.py`
- `tests/engine/test_duplex_runtime.py`
- `tests/worker/test_native_duplex_hooks.py`
- `tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py`

## 12. Main Bugs Fixed

The runtime work closes these observed failure classes:

1. Auto-response and manual `response.create` both driving one turn.
2. Commit auto-response depending on residual bytes returned by `flush()`.
3. Model forced to speak by serving-side probability/token policy.
4. Stage output audio dropped by a base pooling early-return path.
5. Flat and nested metadata representations losing turn identity.
6. TTS segment end incorrectly treated as assistant turn end.
7. Stage1 consumed cursor, cumulative handoff, or Token2Wav state leaking into
   the next turn.
8. Stage0 state being cleared too aggressively and losing conversation memory.
9. Empty or short turns replaying a prior turn's text/audio tail.
10. Response lifecycle closing the external response while also destroying the
    persistent session request.
11. Missing fence metadata silently disabling stale-output protection.
12. Model-specific code and WebSocket/protocol code accumulating in one generic
    serving module.
13. Legacy deploy configuration bypassing the standard pipeline composition.
14. Audio committed while an earlier response was playing being stranded after
    that response completed. Auto-response overlap is now latched from the
    first chunk, buffered for the whole input turn, and promoted as one final
    new-turn append after the active response reaches its terminal event.
15. Stage1 transcript metadata dropping text buffered by Token2Wav before the
    first waveform was emitted. Segment text is now accumulated with the
    vocoder turn state and drained only with the waveform that contains it.

## 13. Verification Evidence

All pytest, E2E, ASR, and audio-quality work for this checkpoint is run on the
remote H20 environment, not on the local macOS Python environment.

### 13.1 Runtime regression before deploy migration

```text
265 passed
log: /tmp/remote_gpu_logs/6aa1bc29.log
```

Prior clean multi-turn E2E and ASR evidence:

```text
/tmp/remote_gpu_logs/f9acd944.log
/tmp/remote_gpu_logs/6ee3df18.log
```

### 13.2 Deploy migration tests

Focused deploy/session contract:

```text
2 passed
/tmp/remote_gpu_logs/3a138f97.log
```

Config impact classes:

```text
36 passed
/tmp/remote_gpu_logs/23974d95.log
```

Final MiniCPM native duplex regression suite after the deploy migration:

```text
265 passed
/tmp/remote_gpu_logs/cc7aaf15.log
```

Scheduler selection RED/GREEN:

```text
RED, async scheduler selected:
  /tmp/remote_gpu_logs/11e25a35.log
GREEN, synchronous scheduler selected:
  /tmp/remote_gpu_logs/f116b812.log
```

The full config-factory file was not accepted as evidence because it stalled in
an unrelated remote model auto-detection test before reaching this change:

```text
/tmp/remote_gpu_logs/1a327e3a.log
```

### 13.3 Deploy-config E2E

Server startup:

```text
/tmp/remote_gpu_logs/1db6b37e.log
```

The original three-turn response-producing fixture run was:

```text
PASS
/tmp/remote_gpu_logs/032bb67b.log
artifacts: /tmp/minicpmo_e2e_pr3907_deploy_config_sync_20260711
```

Observed protocol counts:

```text
response.created: 3
response.done: 3
response.audio.done: 3
response.audio.delta: 15
cancel/listen/stale terminal residue: 0
```

Observed response transcripts:

```text
1. 你好呀，有什么我可以帮到你的吗？
2. 哎，不是说好不聊八卦的吗？
3. 哈，那我们就不聊八卦了嘛。
```

Whisper large-v3 ASR was run against the three per-response WAV artifacts on the
remote H20 host:

```text
/tmp/remote_gpu_logs/01e659b0.log

response_01.wav: 你好呀,有什么我可以帮到你的吗?
response_02.wav: 诶不,是说好不聊八卦的吗?
response_03.wav: 那我们就不聊八卦了嘛
```

All three files contain intelligible Chinese speech and agree semantically with
the protocol transcripts. This is an audio-content sanity check, not a
large-corpus MOS or speaker-similarity claim.

This is fixture-specific evidence. MiniCPM-o owns the listen/speak decision, so
an arbitrary WAV is not guaranteed to produce one spoken response per commit.
The current gate separates that model policy from response-required audio
validation:

- `model-policy`: every streamed input turn must receive a model `listen` or
  `speak` decision; an all-listen run is valid if session, commit, fence, and
  lifecycle invariants hold.
- `response-required`: every requested turn must select a completed response
  containing audio and transcript. Use only a pinned fixture known to produce
  speech; this does not claim that arbitrary speech forces the model to speak.

The final H20 validation at this checkpoint used the same code with profile
logging disabled. The distinct-input model-policy run completed three commits
as three valid `listen` outcomes, with no timeout, stale output, cancellation,
or truncation:

```text
PASS
/tmp/remote_gpu_logs/52b0707e.log
turn outcomes: listen, listen, listen
model listen events: 4
input commits/transcriptions: 3/3
```

The earlier three-turn response-required run exposed an audio-only empty model
turn that the gate had skipped. Whisper large-v3 decoded its 1.28-second audio
as unrelated speech (`料汤`) while the protocol transcript was empty. The root
cause was Stage1 starting TTS from a terminal-only `[speak, turn_eos]` handoff.
The runtime now treats `turn_eos` as a flush only when a spoken Stage1 turn is
already open; an empty terminal still reaches the active session controller so
model-turn identity advances, but it does not create an OpenAI response or
synthesize audio.

The earlier response-required contract was intentionally single-turn. The
latest checkpoint adds a pinned three-turn semantic chain with transcript hints
disabled and 200 ms real-time input pacing. It proves that the model understands
the input audio, preserves clean-turn context, and completes three independent
Realtime response lifecycles without serving-side force-speak:

```text
PASS
/tmp/remote_gpu_logs/20c3f90d.log
artifacts: /tmp/minicpmo_e2e_required_chain_textfixed
turn outcomes: speak, speak, speak
response.created/done/audio.done: 3/3/3
response.audio.delta: 13
input transcription hints/events: disabled/0
real-time input pacing: 200 ms chunks
all_audio_responses_have_transcript: true
transcripts:
  1. 好的，我记住了，暗号是鲸鱼。
  2. 你好，有什么可以帮您的吗？
  3. 没问题，刚才的暗号是鲸鱼。
```

Whisper large-v3 also decoded all three fresh WAVs as continuous Chinese
speech. It matched the middle response exactly and retained the code-word
semantics in the first and third responses despite homophone substitutions:

```text
/tmp/remote_gpu_logs/1610897a.log
```

This is an intelligibility sanity check, not a corpus-level MOS claim.

The third answer depends on the first input, while the unrelated short second
turn has its own response. This is evidence for sequential multi-turn audio
conversation on the pinned fixtures, not a promise that arbitrary audio makes
MiniCPM select `speak` after every commit.

The listen-only overlap scenario starts turn 2 while turn 1 audio is still
active. It does not interrupt or truncate the old response. Instead, the
runtime buffers the overlapping user turn and promotes it after the old
response completes:

```text
short overlap, 540 ms in 200/200/140 ms chunks:
  PASS /tmp/remote_gpu_logs/84bd74af.log
  artifacts: /tmp/minicpmo_e2e_overlap_textfixed
  overlap decisions: 3, all action=listen and defer_runtime_append=true
  response.created/done/audio.done: 2/2/2
  cancel/truncate/stale: 0/0/0

long overlap, 2383 ms in 12 chunks:
  PASS /tmp/remote_gpu_logs/6da167a1.log
  artifacts: /tmp/minicpmo_e2e_listen_only_overlap_long_fixed
  overlap decisions: 12, cumulative speech_ms=2383
  response.created/done/audio.done: 2/2/2
  cancel/truncate/stale: 0/0/0
```

Here `listen` means that the model/session keeps accepting later audio; it is
not a terminal response owed to each commit. The overlap scenario validates
continuous listen-only buffering, not automatic or VAD barge-in.

Whisper large-v3 was also run on the latest generated WAV files:

```text
/tmp/remote_gpu_logs/7f683913.log

three-turn response-required:
  protocol: 你好，有什么可以帮您的吗？
  Whisper:  你好,有什么可以帮您的吗?

short-overlap second response:
  protocol: 你好，有什么可以帮你的吗？
  Whisper:  你好,有什么可以帮你的吗?
```

The other two semantic-chain outputs were normal, non-empty 24 kHz speech;
Whisper rendered the uncommon word `鲸鱼` as homophones. The short-overlap long
response matched Whisper nearly verbatim. This establishes intelligible human
speech and transcript/audio semantic alignment for the fixtures, not a MOS or
corpus-level quality certification.

The final affected regression suites passed together on the same remote H20
checkout. They cover the async engine and runtime port, capability contract,
MiniCPM stage input and terminal-only behavior, demo gate, and realtime duplex
handler:

```text
225 passed
/tmp/remote_gpu_logs/122d06f0.log
```

After correcting the multi-session capability contract, the documented focused
runtime, fence, MiniCPM adapter, stage-input, and deploy-config suites passed:

```text
274 passed
/tmp/remote_gpu_logs/d8a2d700.log
```

After adding no-hint real-time validation, overlap deferral, and vocoder text
alignment coverage, the expanded focused suite passed together:

```text
423 passed
/tmp/remote_gpu_logs/227cc4af.log
```

The no-profile server used for these runs was recorded separately:

```text
/tmp/remote_gpu_logs/2a84346e.log
```

The earlier arbitrary-input timeout was therefore not accepted as either a
clean E2E pass or proof that the model had simply chosen to listen. It exposed
both a contract problem and runtime loss of a real Stage0 listen decision. The
runtime now snapshots raw segment token metadata before output processing,
routes direct listen output without Stage1, preserves its outer duplex metadata
through async collection, and keeps a resumable request alive after a
pre-response listen. The `model-policy` gate verifies this observable listen
path. `response-required` is now also verified for one pinned three-turn
semantic chain, but remains fixture-specific.

The overlap-follow-up checkpoint also covered a stricter event ordering: the
second response was created before the first response's playback acknowledgement
arrived. Playback state is now scoped by `response_id`, so a late acknowledgement
cannot advance the active response's cursor or commit the wrong assistant item.
The H20 run produced three natural speak turns and three committed history items:

```text
response 1 done: generated/sent=11480/11480, played/committed=0/0
response 1 ack:  generated/sent=11480/11480, played/committed=11480/11480
response 2 done: generated/sent=2200/2200, played/committed=0/0
response 2 ack:  generated/sent=2200/2200, played/committed=2200/2200
response 3 done: generated/sent=3960/3960, played/committed=0/0
response 3 ack:  generated/sent=3960/3960, played/committed=3960/3960
history_committed=true for all three responses
```

The third answer was `没问题，刚才的暗号是鲸鱼。`, showing that the second
assistant item was committed before the follow-up turn consumed history. The
vLLM 0.25 rebase checkpoint committed all three response-scoped playback
cursors and produced the same semantic follow-up. Its artifacts and logs are:

```text
/tmp/minicpmo_e2e_v025_rebase
/tmp/remote_gpu_logs/7f679267.log
/tmp/remote_gpu_logs/78a35335.log
```

The expanded affected-test matrix, including response-scoped playback cursor,
partial-history truncation, runtime, orchestrator, output processor, MiniCPM
adapter, Stage0/Stage1 hooks, Realtime handler, and demo gates, passed on H20:

```text
550 passed
/tmp/remote_gpu_logs/6875d31f.log
```

The matrix uses vLLM 0.25.0 and includes the latest-main stage-init and output
processor tests that cover both rebase conflict resolutions. The final cleanup
removes tests that existed only to verify deleted reducer, provider, runner-hook,
and worker-RPC implementations. The post-cleanup H20 matrix below is the
authoritative regression evidence for the remaining Stage0, Stage1/TTS,
protocol, and E2E path.

The post-cleanup tree passed three disjoint H20 regression groups:

```text
duplex runtime, Realtime serving, fence, web, and MiniCPM integration:
  337 passed
  /tmp/remote_gpu_logs/80721cab.log

stage initialization, output processor, intermediate data, and deploy config:
  86 passed
  /tmp/remote_gpu_logs/04af649a.log

scheduler streaming, MiniCPM model/TTS, and GPU AR runner:
  66 passed
  /tmp/remote_gpu_logs/73955fb9.log
```

The final cleanup also removed a stale signal payload contract. Runtime signals
now carry only event, fence, and timeout; conversation and playback mutation
remain serving-owned. The fake engine uses the same strict signature as the
real engine, and the demo treats every server `error` event as a hard failure.
Cancellation now signals the captured pre-increment fence, so the orchestrator
can release old Stage0/Stage1 bindings even if the session has already advanced
to the next epoch. A focused H20 test covers both immediate and late delivery.
The fresh pinned three-turn H20 run completed without any error event:

```text
PASS
/tmp/remote_gpu_logs/b615de0e.log
server: /tmp/remote_gpu_logs/42710bc9.log
artifacts: /tmp/pr3907_cleanup_response_required_cancel_fence
turn outcomes: speak, speak, speak
response.created/audio.done/done: 3/3/3
response.audio.delta: 13
playback history committed: 3/3
cancel/truncate/stale/error: 0/0/0/0
transcripts:
  1. 好的，我记住了，暗号是鲸鱼。
  2. 你好，有什么可以帮您的吗？
  3. 没问题，刚才的暗号是鲸鱼。
```

## 14. Reviewer Reproduction

### 14.1 Start the server

```bash
python3 -m vllm_omni.entrypoints.cli.main serve \
  openbmb/MiniCPM-o-4_5 \
  --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099
```

Wait for the server to report readiness before starting the client.

### 14.2 Run arbitrary distinct inputs under model policy

```bash
python3 examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav /path/to/turn1.wav \
  --turn-input-wav /path/to/turn2.wav \
  --turn-input-wav /path/to/turn3.wav \
  --turns 3 \
  --turn-duration-ms 3000 \
  --turn-duration-ms 3000 \
  --turn-duration-ms 3000 \
  --require-distinct-inputs \
  --omit-transcript-hints \
  --realtime-input \
  --validation-mode model-policy \
  --output-dir /tmp/minicpmo45_duplex_model_policy
```

The client must not send `response.create` or a force-barge-in event. A turn may
produce `response.listen` instead of `response.created`. Verify:

- every input is committed and every turn reports `listen` or `speak`;
- any created response has a symmetric terminal lifecycle;
- no missing-fence, stale output, cancellation, truncation, timeout, or forced
  speak fallback occurs;
- the result reports `error_count: 0`; any server `error` event fails the run.

### 14.3 Run the pinned response-required fixture

The H20 inputs used for the latest no-hint checkpoint are:

| File | SHA256 | Source duration | Validation use |
| --- | --- | ---: | --- |
| `minicpmo_chain_1_16k.wav` | `0f01b7647dedec0fb40400af7b2a45eb7b3effb51ff568c06fdfdf6b279f70fc` | 2383 ms | remember `鲸鱼` |
| `minicpmo_nihao.wav` | `910d953895f28d6ce4ef515e8e9ec5a75bbf95ed7851ef3e45267ba7f97055e5` | 540 ms | unrelated short middle turn |
| `minicpmo_chain_2_16k.wav` | `df5f3072dead62bc6f9dfbc178aa6f4b1424a6c3129af3bfdf722ec01f78d014` | 3219 ms | ask for the remembered code word |
| `minicpmo_chain_3_16k.wav` | `cdb48d3756118a00ad948961204c7696444715bbcd356cad84077938e86d9b4a` | 2778 ms | optional follow-up fixture |
| `minicpmo_pr3907_jiayan_16k.wav` | `2e5fd4eb3ee434ce107ee3a0591fa624a33f7683c7462f45fe651c443c9af941` | 5469 ms | overlap turn 1, first 1400 ms |

Verify the hashes before running:

```bash
sha256sum \
  minicpmo_chain_1_16k.wav \
  minicpmo_nihao.wav \
  minicpmo_chain_2_16k.wav

python3 examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav minicpmo_chain_1_16k.wav \
  --turn-input-wav minicpmo_nihao.wav \
  --turn-input-wav minicpmo_chain_2_16k.wav \
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

Review the per-response WAV files and verify:

- three selected response IDs with audio and transcript;
- all model-created responses have symmetric
  `response.created`, `response.audio.done`, and `response.done` lifecycles;
- every response containing audio has a non-empty transcript;
- transcript delta concatenation equals transcript done;
- no previous-turn suffix in the next response;
- audio exists before done and no audio arrives after done;
- no missing-fence, stale-guard-inert, timeout, forced-speak, or forced-listen
  fallback in the server log;
- `error_count` is zero. The demo must exit nonzero for any server `error`
  event, even when all response lifecycle counts otherwise look valid.

`--turn-transcript` labels the local fixture and expected result only when
`--omit-transcript-hints` is set; it is not sent to the server. Validate
generated speech separately with ASR or listening when audio content is part of
the acceptance criteria.

### 14.4 Run listen-only overlap without interruption

```bash
python3 examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav minicpmo_pr3907_jiayan_16k.wav \
  --turn-input-wav minicpmo_nihao.wav \
  --turns 2 \
  --first-turn-ms 1400 \
  --chunk-ms 200 \
  --omit-transcript-hints \
  --realtime-input \
  --require-distinct-inputs \
  --require-audio \
  --validation-mode model-policy \
  --scenario listen-only-overlap \
  --output-dir /tmp/minicpmo45_duplex_overlap
```

The second `input_audio_buffer.speech_started` must occur before the first
`response.done`. Every overlap decision must be `listen` with deferred runtime
append; the first response must not be cancelled or truncated; both responses
must still close normally.

### 14.5 Focused unit tests

```bash
pytest -q \
  tests/fullduplex \
  tests/engine/test_duplex_runtime.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/test_duplex_fence_propagation.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/examples/test_minicpmo_realtime_web.py \
  tests/test_config_factory.py::TestStageConfig \
  tests/test_config_factory.py::TestDeployConfigLoading

```

## 15. Review Priorities

Reviewers should focus on these invariants rather than only the demo output:

1. A clean commit advances turn/response once but preserves epoch and Stage0
   context.
2. `auto_response` has one response driver and never requires
   `response.create`.
3. Every data-plane output and terminal event carries the same complete fence.
4. Segment end cannot close a response or clear turn-local state.
5. Turn end clears all Stage1 turn-local accumulators together.
6. Clean turn reset does not clear Stage0 conversational KV.
7. An empty/short turn cannot replay prior text or audio.
8. A stage failure or session close releases all fenced request and replica
   bindings.
9. The deploy overlay preserves default non-duplex behavior.
10. Capability reporting does not claim scheduler-native append, core KV lease,
    automatic barge-in, or production concurrency.

## 16. Follow-Up Work

The next architectural tiers are intentionally separate from this checkpoint:

- add an upstream scheduler-native append/session-KV primitive and replace the
  compatibility helpers in the AsyncOmni/orchestrator engine boundary;
- select and validate an automatic interruption source for MiniCPM-o;
- run the existing epoch interruption contract through full E2E audio fencing;
- implement bounded/windowed conversational KV for minute-scale sessions;
- add multi-session admission, fairness, failure, and capacity tests;
- add a larger audio-quality corpus with ASR and MOS comparison against the
  official Hugging Face loop.

These are not hidden blockers for reviewing the clean multi-turn runtime, but
they remain blockers for a production-level general full-duplex claim.
