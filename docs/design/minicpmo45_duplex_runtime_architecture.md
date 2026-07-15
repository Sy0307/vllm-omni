# MiniCPM-o 4.5 Full-Duplex Runtime Architecture

## Purpose

This document records the runtime architecture implemented by PR #3907, the
cleanup applied after published commit `e011d936`, and the remaining work that
must not be advertised as complete. Implemented contracts and future
architecture work are called out separately below.

The checkpoint preserves the active runtime path already exercised on H20. It does
not introduce another reducer, controller, worker provider, or shadow runtime.

## Review Snapshot

- PR: `vllm-project/vllm-omni#3907`
- Published head before this refactor: `e011d936`
- Published base snapshot: `62589203`
- Published diff: 93 files, approximately `+29.2k/-0.6k`
- Local runtime and pytest validation: intentionally not used
- Required validation environment: an NVIDIA H20 CUDA host

An earlier synchronized refactor snapshot passed 421 affected tests on H20,
with 18 warnings. Its shutdown, mailbox, response-listen, cancellable-append,
and MiniCPM compatibility paths also passed a 127-test affected subset. Those
results are historical evidence, not validation of the current dirty tree. The
current tree adds an irreversible cancel fence, scheduler and model-runner
decoupling, and stage-transition cleanup. Its affected H20 matrix passes 353
tests with 18 warnings; the exact E2E evidence is recorded below.

## Scope

The checkpoint keeps these verified contracts:

- MiniCPM Stage0 conversation KV continuity;
- Stage1 TTS and Token2Wav continuity;
- model-owned listen/speak decisions on the normal auto-response path;
- segment EOS and turn EOS as different boundaries;
- transcript/audio cursors scoped to a response and turn;
- playback acknowledgement and history commit;
- scheduler data-plane append over a resumable request;
- stale epoch/turn/response fencing;
- `/v1/duplex` and OpenAI Realtime projection;
- existing JoyVL behavior.

The checkpoint does not claim:

- scheduler-native KV append;
- automatic or VAD barge-in;
- multi-session admission, fairness, or isolation;
- bounded long-session KV;
- production capacity or fault recovery;
- video input or audio/video synchronization.

## Active Runtime Path

The actual path is:

```text
WebSocket
  -> OmniDuplexSessionHandler
  -> DuplexSession + MiniCPMO45ServingSessionState
  -> AsyncOmni open/append/signal/close
  -> Orchestrator duplex session and stage bindings
  -> StagePool
  -> resumable scheduler request
  -> MiniCPM Stage0
  -> Stage1 TTS / Token2Wav
  -> output processor
  -> Realtime protocol projection
  -> WebSocket writer
```

The removed typed reducer/facade and worker-provider runtimes were not in this
path. This refactor continues to modify only the path above.

## State Ownership

The architecture allows several state objects only when their facts are
orthogonal.

| Owner | Authoritative facts |
| --- | --- |
| `DuplexSession` | session state, epoch, turn, active/last response, pending input, overlap duration, playback cursor, history |
| `MiniCPMO45ServingSessionState` | MiniCPM PCM buffer, deferred overlap payload, data-plane task, continuation effect counter |
| `DuplexWebSocketActor` | inbound mailbox, outbound queue, writer, transport close state, effect task handles |
| `NativeRealtimeSessionProtocol` | wire-only response/item projection and input-buffer projection |
| engine duplex runtime | fence snapshot, stage bindings, scheduler request ownership |

Important distinctions:

- Realtime response IDs are projection caches, not domain decision sources.
- Engine fences duplicate identity across a process boundary by design; they
  are immutable validation snapshots.
- Stage bindings are engine resources, not serving response state.
- `continuation_response_id` scopes an effect counter to one response; it is
  not the authoritative active response ID.
- runtime open/close acknowledgements are handler-local effect bookkeeping,
  not Session lifecycle.

## Implemented Concurrency Contracts

### Single inbound mailbox

The Actor now uses one:

```python
asyncio.Queue[event]
```

This replaces competing control/input/event `Queue.get()` tasks and the
deferred-event list while preserving WebSocket arrival order. A later close,
cancel, or clear must not overtake earlier append, playback ACK, or
session-update events. Slow runtime appends remain background tasks. Cancelling
their asyncio task does not stop a synchronous callable already running in the
default executor, so cancellation correctness is provided by the engine fence,
not by `Task.cancel()`.

Properties covered by focused tests:

- every enqueued event is delivered exactly once;
- terminal control preserves wire order while cancelling background work;
- clear and cancel never scan or delete later mailbox events;
- one writer owns all WebSocket sends.

### Correlated RPC result router

`AsyncOmniEngine` now has one consumer for `rpc_output_queue`. Results route by:

```text
("duplex", control_id)
("collective", rpc_id)
```

The router replaces the global RPC lock. It supports out-of-order replies,
timeout unregister, late-result rejection, fatal-error broadcast, and close
unblocking.

Additional lifecycle rules:

- uncorrelated non-fatal errors are not broadcast to unrelated waiters;
- a latched fatal rejects new calls before they enter the request queue;
- `EngineDeadError` is broadcast directly to the RPC queue even though the
  orchestrator consumes that exception during teardown;
- shutdown closes the router before joining the orchestrator;
- failure to enqueue shutdown closes the request queue and still attempts a
  bounded orchestrator join;
- `StageRuntime` cleanup runs only after the orchestrator has stopped, avoiding
  concurrent client teardown.

The router removes wait-side head-of-line blocking. The orchestrator request
handler remains ordered, but executor submission order is not itself a session
ordering guarantee.

### Irreversible cancellation fence

Every cancel control carries the cancelled fence and the next session fence.
The `AsyncOmni` facade preserves both values across the serving,
entrypoint, engine-message, and orchestrator boundaries. The orchestrator then
atomically:

1. advances the runtime state to the next fence;
2. releases and aborts bindings owned by the cancelled fence;
3. permanently rejects an append that arrives later with the old fence.

Missing `next_fence` is rejected without releasing bindings; the engine never
advances independently from serving. A late cancel cannot move a session
backward when the runtime is already on a newer fence. Idle playback clear does
not send an engine cancel because no runtime resource is being cancelled.

`model_native_duplex` also validates its engine-client contract before opening
the session. The open, append, signal, and close methods must exist and declare
the required `fence` arguments; signal must additionally accept `next_fence`.
An older or custom client that cannot preserve this boundary fails with
`runtime_contract_invalid` instead of silently running an unfenced native
session. The generic serving-session adapter retains its compatibility behavior.

`DuplexFence` also carries a server-side session incarnation. Reusing a public
session ID after close increments that incarnation, and resource request IDs
include it after the first incarnation. A synchronous append from a closed
session therefore cannot be accepted by a newly opened session with the same
public ID. Stale close validates the full fence before removing the runtime
registry entry or releasing bindings.

## Implemented Serving Ownership

### Session-owned response and overlap state

The Actor no longer stores:

- a Session reference;
- lifecycle mirror state;
- active or last response ID;
- overlap duration;
- response-in-progress or playback business predicates;
- runtime open/close acknowledgements.

`DuplexSession.begin_response()` owns both active and last response identity.
The Session owns overlap accumulation/reset and playback state.

One `native_response_in_progress()` predicate now covers:

- active response ID;
- active engine request ID;
- uncommitted ACK-only playback;
- active generic response task;
- response-bound append tasks;
- MiniCPM data-plane task;
- assistant-generating turn state.

The write-only `MiniCPMO45ServingSessionState.response_emitted` field and a dead
data-plane task mirror were removed.

### Explicit event effects and pure writer

Business paths call `emit_event()`. It serializes domain effects and outbound
queue insertion with one session-local lock. Deferred append starts outside the
lock so recursive event emission cannot deadlock.

The writer only:

1. dequeues an event;
2. applies late stale filtering to streaming model output;
3. projects the event to the selected wire protocol;
4. sends JSON.

The writer does not promote overlap, clear input, advance Session state, call
the engine, or commit history.

### Single terminal acceptance point

Terminal events are accepted or rejected before domain effects run:

```text
response.done
response.listen
audio.cancelled
input.cancelled
session.closed
```

For a terminal carrying an epoch, that epoch must match the current Session.
Stale cancellation therefore cannot reset overlap belonging to a later turn.
Once a terminal is accepted and its effect is applied, the writer cannot revoke
it because close or epoch state changed while it waited in the output queue.

Streaming audio/text deltas still receive late stale filtering at send time,
which prevents queued old audio from leaking after a cancel or epoch bump.

## Stable Type Boundary

`DuplexFence` now lives in:

```text
vllm_omni.engine.duplex_types
```

Stable engine, orchestrator, messages, and entrypoint modules import that type.
The experimental identity module re-exports the same class for compatibility.
This removes the `DuplexFence` stable-to-experimental dependency without
creating a second identity type. Other stable modules still import the
experimental duplex manager and helpers; moving those boundaries is listed
below rather than claimed as complete.

## Generic-Path Cleanup

Model-specific `MINICPMO45_PROFILE_LOGS` probes were removed from:

- the generic AR scheduler;
- orchestrator;
- AsyncOmni;
- the generic GPU AR runner.

The cleanup removes the temporary serving and generic-runtime probes. The
generic AR runner now builds `DuplexSamplingRow` values and invokes the optional
model `prepare_duplex_sampling()` hook exactly once before the normal sampler.
MiniCPM owns force-listen logits, turn-ended state, and its row-local sampling
policy behind that hook; the runner no longer reads `_minicpmo45_*` attributes
or retries sampling after matching `TypeError` text.

The scheduler no longer parses serving `session_config`, `extra_body`, or
`duplex_stage_sampling_params`. Stage-specific overrides are materialized by
the orchestrator before scheduler admission. Segment-boundary detection still
uses typed `SamplingParams`, resumable-request state, and one narrow
`model_intermediate_buffer["duplex"]["data_plane"]` marker. Replacing that marker
with a typed resumable-segment policy remains follow-up work; this checkpoint
does not claim that boundary is complete.

Local `turn.signal` transitions such as `user_started` and
`assistant_started` no longer perform an engine RPC. Only cancellation and
`session.update` cross the engine boundary because those operations change
runtime resource identity or runtime configuration.

During a stage transition, `StreamingInputState.source_token_decoder` exposes a
short-lived, model-neutral decoder sourced from the upstream output processor.
The MiniCPM Stage1 input processor consumes that capability for token-faithful
delta transcripts. The generic orchestrator no longer installs or names a
MiniCPM-specific bridge-state key.

The request-ID prefix fallback in `OmniBase` remains. The data-plane request ID
is currently returned in the control acknowledgement, while output can race
that acknowledgement. Until request state is registered before stage submit,
removing the fallback can silently drop the first output batch.

## Why Scheduler Changes Remain

Serving cannot preserve model KV after a segment ends. The scheduler must keep
one resumable request in a waiting state and accept a later update:

```text
RUNNING
  -> segment stop
WAITING_FOR_STREAMING_REQ  (KV retained)
  -> next append
RUNNING
  -> session close
FINISHED                   (KV released)
```

Scheduler responsibilities are limited to resumable request state, runtime
context update, stop/boundary handling, and final release. Scheduler must not
own response IDs, playback, overlap, Realtime events, or model policy.

## Remaining Architecture Work

The following work is deliberately not hidden inside this checkpoint.

### Orchestrator extension

Move session registry, fence validation, stage binding, and resumable update
coordination behind a stable extension. The main orchestrator should retain
message dispatch and stage-output hooks.

### Explicit request registration

Return or reserve the data-plane request identity before output can be emitted,
register the AsyncOmni client state first, then remove the `duplex-` prefix
fallback.

### Session internal ledgers

`DuplexSession` is the serving aggregate root, but it still contains input,
response, playback, and conversation-history fields in one dataclass. The
current refactor removes competing owners and most mirrored state; it does not
claim that those internal ledgers are fully encapsulated. A later maintenance
change may compose them into private `InputBufferState`, `ResponseState`,
`PlaybackLedger`, and `ConversationHistory` values while keeping
`DuplexSession` as the only transition owner. It must not introduce another
reducer or a second session state machine.

### Non-cancel signal semantics

`turn.signal` remains a public compatibility surface and drives local Session
and Realtime transitions. Serving forwards only cancellation and
`session.update` to the engine. The lower-level engine API can still accept an
arbitrary non-cancel signal and return a fence-validation acknowledgement, but
that compatibility entry point has no stage or worker effect and must not be
described as model control.

### Resource capabilities

Lease TTL, orphan reaping, admission, fairness, capacity, and same-replica
multi-session isolation belong to a later production-capability change. The
current capability response must remain `false` for those features.

## Validation Plan

All pytest and runtime evidence for this branch must run on the remote H20.

### Current H20 evidence

The current dirty tree was synchronized to an isolated H20 worktree running
vLLM 0.25.0.

- Full affected matrix: 353 passed, 18 warnings.
- Full duplex handler suite after fenced native-contract validation: 131 passed,
  18 warnings.
- Executor cancel, late-append rejection, runtime fence, and Stage0 request
  cleanup focus: 4 passed, 16 warnings.
- Executor cancel and irreversible-fence focused tests: passed.
- Three semantically distinct, no-transcript-hint turns sent in 200 ms
  real-time chunks: `speak / speak / listen`, two complete audio responses,
  two playback history commits, no stale/cancel/truncate/error event, and no
  duplicate `response.speak`.
- Fixed 1400 ms response-required input: one complete response, eight
  audio/transcript deltas, matching transcript delta/final content, and one
  playback history commit.
- Request cleanup initially reproduced a Stage0-fatal `Set changed size during
  iteration`; the regression test failed before the fix and passed after the
  cleanup switched to a snapshot of completed force-listen segments.
- The API server remained healthy after both E2E sessions. The final server
  error scan was empty, and shutdown released the two test GPUs.
- The response-required WAV was 24 kHz mono PCM with 168,960 samples and RMS
  0.0946. External Whisper ASR matched the emitted transcript semantically.

### Historical H20 evidence

The last complete validation snapshot was the local refactor based on published commit
`e011d936`, synchronized file-for-file to an isolated H20 worktree. A checksum
dry run reported no differences before validation.

- Post-review engine and MiniCPM affected subset: 127 passed, 16 warnings.
- Full affected matrix: 421 passed, 18 warnings.
- Three-turn distinct-input model-policy E2E, without transcript hints and with
  200 ms input pacing: `speak / speak / listen`, two complete responses, two
  playback history commits, and no stale, cancel, timeout, or error event.
- Fixed response-required E2E: one complete response with 12 audio/transcript
  deltas, one `response.speak`, one playback history commit, and matching
  transcript delta/final content.
- Generated response audio: 24 kHz mono PCM, 11.12 seconds, non-empty waveform;
  Whisper large-v3 was semantically aligned with the final transcript with
  minor homophone substitutions.
- Final server-log scan: zero traceback, error, stale/late-audio, or runtime
  failure matches. The validation service exited cleanly and released its two
  GPUs.

### Focused suites

- RPC router and AsyncOmni shutdown/error routing;
- orchestrator fatal and duplex control paths;
- WebSocket Actor mailbox and writer;
- duplex protocol and Session ownership;
- Realtime handler, terminal ordering, overlap, playback ACK;
- scheduler resumable boundary;
- MiniCPM runner and Stage0/Stage1 lifecycle.

### Real E2E

Run the fixed, no-transcript-hint, 200 ms paced fixtures:

- three-turn distinct input;
- model-policy distinct input, allowing natural listen or speak decisions;
- a fixed response-required fixture that naturally selects speak;
- late and old-response playback ACK handling in the affected test matrix;
- close/disconnect with no post-terminal audio;
- transcript delta/final consistency;
- generated WAV validity and external ASR sanity.

Distinct-input fixtures must also be semantically distinct. Different files or
waveform hashes containing the same greeting can legitimately produce the same
reply and are not sufficient evidence for cross-turn transcript independence.

For each run record the exact SHA/worktree, command, server log, client log,
WAV path, and cleanup result.

## Acceptance Criteria

This checkpoint is ready to publish only when:

- one mailbox passes exactly-once and wire-order tests;
- cancel advances the engine fence even when the old append continues in an
  executor, and the next-epoch append remains accepted;
- request completion removes MiniCPM helper and force-listen segment state
  without mutating a set during iteration;
- RPC fatal/close wakes every waiter and no new work enters after terminal;
- Session is the only serving response/overlap authority;
- stale terminal events cannot mutate a newer epoch;
- an accepted terminal cannot be dropped by the writer;
- no duplicate `response.speak` or transcript delta appears;
- Stage0 KV continuity and Stage1 turn reset pass;
- the real H20 multi-turn audio E2E passes;
- ordinary non-duplex paths in the affected matrix still pass;
- no local path, proxy token, temporary profiling probe, or test-only switch is present.

Passing this checkpoint supports the statement:

> Single-session, no-barge MiniCPM-o 4.5 native duplex is reviewable on the
> validated H20 configuration.

It does not support claims for automatic barge-in, multi-session production
concurrency, bounded long-session KV, scheduler-native append, or video input.
