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
- Validated local commit: `945cc70a54ba9d6d7ccd194aefca1b1bfa20ea64`
- Validated tree: the uncommitted refactor synchronized file-for-file to the
  isolated H20 worktree on 2026-07-16

The current tree has received fresh H20 validation. The affected matrix passed
415 tests, and the same public session ID completed two independent three-turn
audio E2E runs with a close/reopen boundary against vLLM 0.25.0. The validated
tree includes the stable runtime extension, typed resumable policy, extracted
control plane and clients, request preregistration, typed direct-output
decision, server-owned single-session admission, separate public/runtime
configuration channels, private Session ledgers, ordered `session.update`, the
MiniCPM runner fast path, and the final Realtime input-lifecycle fixes. Exact
evidence and scope limits are recorded below.

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
  -> DuplexRequestClient
  -> AsyncOmni thin open/append/signal/close proxies
  -> DuplexControlClient over the engine-owned correlated RPC transport
  -> DuplexControlPlane + DuplexSessionRuntimeManager
  -> stable DuplexRuntimeExtension + engine session/stage bindings
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
| `DuplexSessionRuntimeManager` | fence snapshot, append reservation, completed-operation cache, resource reservations, stage bindings |
| `DuplexControlPlane` | control-message validation, transactional open/append/update/close, extension invocation, typed segment/output decisions |

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

### Extracted control ownership

The stable engine mechanism is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `engine.duplex_runtime` | immutable contracts, extension validation, typed decisions, request-ID codec |
| `engine.duplex_session` | session resources, fences, append reservations, idempotency cache, binding cleanup |
| `engine.duplex_control_plane` | control algorithms and the narrow `DuplexStagePort` used to submit and clean up stage requests |
| `engine.duplex_control_client` | typed control-message construction and calls through the engine-owned correlated RPC transport |
| `entrypoints.duplex_request_client` | deterministic request identity, client-state preregistration, acknowledgement validation, data-plane collection, rollback |

`Orchestrator`, `AsyncOmniEngine`, and `AsyncOmni` retain explicit wiring and
compatibility proxies. They no longer own the corresponding duplex algorithms.
The ControlClient does not own waiters or consume the result queue;
`RpcResultRouter` remains the single correlation owner for both duplex and
collective RPC. The ControlPlane receives a narrow stage port rather than the
Orchestrator object.

Control-plane enablement is independent from extension presence.
`PipelineConfig.duplex_control_enabled` determines whether an Orchestrator
constructs a ControlPlane. This preserves extension-free
`TURN_COMMIT_ONLY` deployments while allowing ordinary deployments to use a
single `None` fast-path check. The MiniCPM pipeline explicitly enables control
and separately selects its runtime extension.

### Transactional open and append

Opening a native session registers its engine state and reserves the Stage0
request resource as one externally atomic operation. If capability, sampling,
extension, or request-state initialization fails, the ControlPlane removes the
session, reserved request resources, and any preregistered request state before
returning the error. Cleanup distinguishes a reserved resource from a submitted
request, so it does not decrement the running counter for unrelated work.
Closing a session that never appended also removes the preregistered resource.

Append uses prepare/submit/commit ordering. `prepare_append()` computes the
next sequence values without mutating the session. Prompt planning and stage
submission use that reservation, and only a successful submit commits
`input_seq`, `input_turn_seq`, and the accepted fence. Serving uses a matching
PCM reservation and consumes bytes only after the append acknowledgement;
failure or a closed predecessor rolls the bytes back. An `operation_id` caches
completed append results so a retry cannot submit the same physical append
twice.

### Ordered append and configuration effects

Native append effects form a per-session tail. Each append awaits its
predecessor before entering the engine, and `session.update` awaits that same
tail before changing either the serving aggregate or engine configuration.
Consequently, an append received before an update uses the old immutable
sampling/policy snapshot, while the first accepted append after the update uses
the new generation.

The engine increments `config_generation` when it accepts the replacement
configuration. At the next append boundary, the orchestrator rebuilds sampling
parameters and `ResumableSegmentPolicy` from stage defaults through the model
extension. It never mutates the snapshot of a segment already in progress.

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

### Response-scoped options

`response.create` is represented by an immutable `ResponseCreateOptions` value.
The Session reserves it for one response, applies it to a response-local copy
when that response begins, and restores the base session configuration on
completion, listen, cancellation, or failure. Instructions, voice, modalities,
temperature, token limit, format, tools, and conversation metadata therefore do
not mutate the permanent session configuration or leak into a later response.

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

The stable engine contract now lives in:

```text
vllm_omni.engine.duplex_types          # identity fence
vllm_omni.engine.duplex_runtime        # immutable contracts and extension protocol
vllm_omni.engine.duplex_session        # session and resource ownership
vllm_omni.engine.duplex_control_plane  # engine-side control algorithms
vllm_omni.engine.duplex_control_client # correlated control client
vllm_omni.engine.resumable             # scheduler segment policy
```

Stable engine, orchestrator, messages, and entrypoint modules import that type.
The experimental identity module re-exports the same class for compatibility.
This removes the `DuplexFence` stable-to-experimental dependency without
creating a second identity type. Stable engine, scheduler, worker, and request
modules no longer import `experimental.fullduplex`.

`PipelineConfig.duplex_control_enabled` enables the generic mechanism, while
`PipelineConfig.duplex_runtime_extension` separately selects the model adapter.
The MiniCPM pipeline opts into both and installs
`MiniCPMO45DuplexRuntimeExtension`; ordinary pipelines construct neither a
ControlPlane nor a ControlClient. The stable protocol exposes
sampling-parameter configuration, append planning, a typed
`ResumableSegmentPolicy`, and a typed stage-output decision. MiniCPM owns
stage-specific overrides, PCM-to-token budgeting, force-listen payload policy,
special-token interpretation, and `listen` response metadata. The generic
Orchestrator does not parse MiniCPM session keys, `listen_token_id`, or construct
MiniCPM metadata, and it does not import the adapter.

Extension loading is fail-fast. Startup validates every required callable,
the number and type of returned stage sampling parameters, and the typed
segment policy. Invalid custom extensions fail before a session can open.

Direct listen/speak outcomes use `DuplexOutputDecision` on
`OmniRequestOutput`. That typed envelope is authoritative over inner completion
metadata for native projection, so a processed output cannot hide a listen
decision merely because it retains unrelated multimodal metadata.

### Public and runtime configuration channels

Control messages carry two independent snapshots:

- `session_config` contains the public Realtime/session values used by serving
  and model prompt policy;
- `runtime_config` contains server-derived MiniCPM sampling, scheduler, context
  reserve, prefix-budget, and resolved reference-audio values.

The MiniCPM serving adapter derives `runtime_config` from deploy/model defaults
and validated public options. Clients cannot provide private runtime keys in
`session.create` or `session.update`; response-scoped `extra_body` drops those
keys rather than turning them into engine policy. The ControlPlane constructs
sampling parameters and `ResumableSegmentPolicy` only from `runtime_config`,
while append planning receives both snapshots explicitly. Raw Realtime
`extra_body` is not used as the engine's sampling-policy channel.

`session.update` prepares both candidates, validates the runtime candidate in
the engine, and replaces the serving snapshots only after the engine ACK. A
rejected candidate therefore leaves both the public configuration and runtime
sampling generation unchanged.

## Generic-Path Cleanup

Model-specific `MINICPMO45_PROFILE_LOGS` probes were removed from:

- the generic AR scheduler;
- orchestrator;
- AsyncOmni;
- the generic GPU AR runner.

The cleanup removes the temporary serving and generic-runtime probes. The
generic AR runner builds `DuplexSamplingRow` values only when the model exposes
the optional `prepare_duplex_sampling()` hook, then invokes that hook exactly
once before the normal sampler. Ordinary models do not scan request rows.
MiniCPM owns force-listen logits, turn-ended state, and its row-local sampling
policy behind that hook; the runner no longer reads `_minicpmo45_*` attributes
or retries sampling after matching `TypeError` text.

The scheduler no longer parses serving `session_config`, `extra_body`, or
`duplex_stage_sampling_params`. Stage-specific overrides are materialized by
the model runtime extension before scheduler admission. The orchestrator puts a
typed `ResumableSegmentPolicy` on each resumable stage request, and streaming
updates preserve that policy. The scheduler consumes only the policy and
resumable-request state; it does not inspect a duplex dictionary, Realtime
configuration, or MiniCPM sampling fields. A resumable request without a policy
does not gain MiniCPM segment-stop behavior.

The generic runner's unused `_duplex_force_listen_applied_segments` cleanup and
the obsolete `streaming_accumulated_keys` accumulation hook were removed. The
latter had no model producer after its upstream Qwen removal; streaming input
still performs the ordinary replacement merge needed by active models.

Local `turn.signal` transitions such as `user_started` and
`assistant_started` no longer perform an engine RPC. Only cancellation and
`session.update` cross the engine boundary because those operations change
runtime resource identity or runtime configuration.

During a stage transition, `StreamingInputState.source_token_decoder` exposes a
short-lived, model-neutral decoder sourced from the upstream output processor.
The MiniCPM Stage1 input processor consumes that capability for token-faithful
delta transcripts. The generic orchestrator no longer installs or names a
MiniCPM-specific bridge-state key.

The stable API server inspects the deployment `session_mode` before importing or
constructing the experimental duplex handler. Ordinary deployments do not load
the full-duplex serving package or create its registries and background state.
For an enabled deployment, the client still selects the Realtime duplex route
with `?duplex=1` (or an equivalent explicit true value). Model-name matching is
not used for routing or native-runtime activation. MiniCPM clients explicitly
set `extra_body.minicpmo45_native_duplex=true`; repository demos do so in their
session payloads.

`DuplexRequestClient` derives the data-plane request identity from the accepted
fence and preregisters a resumable `ClientRequestState` before the append crosses
the engine boundary. The acknowledgement must return the same identity or the
operation fails. Unregistered outputs, including names that merely use the old
`duplex-` prefix, are dropped. Timeout, cancel, and close remove the
preregistered state.

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

## Serving and Compatibility Boundaries

### Session internal ledgers

`DuplexSession` remains the serving aggregate root and now composes
`InputBufferState`, `ResponseState`, `PlaybackLedger`, and
`ConversationHistory`. The ledgers are private. Read-only properties return
tuples, snapshots, or mapping proxies, and serving modules mutate them through
Session transition methods for request, response-turn, lifecycle, playback,
capability replacement, and history. The Realtime session-update parser still
normalizes individual fields before atomically retaining or rolling back the
aggregate configuration. Further work may narrow the read-only compatibility
surface, but it must not introduce another reducer or a second session state
machine.

### Realtime input ownership

The Realtime translator validates the wire input buffer before producing a
commit carrying `realtime_item_id`. Native auto-response may already have
streamed those PCM samples into the runtime, so the runner accepts that
validated commit even when no runtime-side chunk remains and still commits the
conversation item. A truly empty wire buffer continues to return
`input_audio_buffer_empty`.

During auto-response overlap, `preserve_realtime_input` distinguishes "do not
append this silent chunk to the native buffer" from "clear the open Realtime
item". Silent overlap no longer discards earlier user PCM. This is an input
ownership correction, not a VAD policy or an automatic barge-in claim.

The first chunk of one overlapping input item also reserves its target model
turn. A later Realtime commit uses that reserved identity even if response EOS
has already advanced the Session turn; clear, cancel, close, and successful
promotion release the reservation.

### Physical input and model-turn identity

A Realtime input item, its `input_audio_buffer.commit`, and a MiniCPM model turn
are intentionally different identities. Native Stage0 evaluates streamed audio
in approximately one-second model units. A sampled `<|turn_eos|>` closes the
current model turn and may allow the next streamed unit to start another turn
before the client commits its physical input item. Conversely, a committed
input may produce only model-listen decisions and no spoken response.

Therefore neither `model_turn_id` nor response cardinality is derived from the
number of physical inputs. `model_turn_id` advances at model EOS. The
`response-required` fixture may require exactly one response per requested
turn, but `model-policy` validation must accept additional model-owned responses
and cannot assert cross-turn transcript independence through physical-input
attribution.
Before closing, the model-policy demo drains every created response through
`response.done`, acknowledges every completed audio playback, and requires a
bounded quiet interval with no newly created response. This validates lifecycle
completion without suppressing or overriding the model's listen/speak policy.

### Non-cancel signal semantics

`turn.signal` remains a public compatibility surface for local Session and
Realtime transitions. Serving forwards only cancellation and `session.update`
to the engine. The engine rejects arbitrary non-cancel signal names instead of
returning a misleading supported acknowledgement. `session.update` remains a
real engine operation because it replaces the configuration consumed by later
append plans.

## Remaining Architecture Work

The following work is deliberately not hidden inside this checkpoint.

### Serving composition

State ownership is narrower, but serving behavior is still composed through
`DuplexSessionRunnerMixin`, `NativeRuntimeBridgeMixin`, and
`ChatFallbackProjectorMixin`. The generic handler, runner, bridge, and protocol
modules remain a large, mutually dependent behavior surface. A later refactor
should replace host-method coupling with explicit `ServingRuntimeAdapter`,
protocol, and effect-runner dependencies while preserving one actor mailbox and
one `DuplexSession` transition owner. Splitting those components must not create
a second reducer or concurrent state machine.

The four private Session ledgers are data partitions, not independent owners.
`DuplexSession` intentionally remains the aggregate root. Further encapsulation
may narrow compatibility properties, but serving code must continue to perform
state changes through Session transitions.

### Plugin descriptor and typed payloads

The pipeline selects the MiniCPM engine extension, and the Realtime request
selects the MiniCPM serving adapter, but there is not yet one versioned plugin
descriptor that binds those identities in an open handshake. Adding a second
native model should first introduce that descriptor and reject serving/engine
plugin mismatches explicitly.

The stable boundary has typed fences, plans, policies, decisions, and control
messages, but `session_config`, `runtime_config`, and append payloads still use
generic mappings or objects at the extension boundary. Model-neutral typed
session and input-chunk contracts remain follow-up work, especially before
adding video.

### Resource capabilities

The MiniCPM pipeline currently sets the server-owned
`max_native_duplex_sessions` limit to one. The API handler receives that deploy
value from the engine and uses it as the admission gate; client session fields
cannot raise or disable the limit. This is a deterministic single-session
checkpoint, not a production admission controller.

Session leases, lease TTL, orphan reaping, fairness, KV-aware capacity,
backpressure, and same-replica multi-session isolation belong to a later
production-capability change. The current capability response must remain
`false` for those features.

## Validation Evidence

All pytest and runtime evidence for this branch must run on the remote H20.

### Current synchronized tree

The current dirty tree was synchronized file-for-file to an isolated H20
worktree. The base commit on both sides was
`945cc70a54ba9d6d7ccd194aefca1b1bfa20ea64`; SHA-256 comparison covered all 62
modified or untracked files before final validation, with zero mismatches.

- Full affected matrix: 460 passed, 19 warnings in 16.47 seconds. The combined
  run is task `08435ce3` (`/tmp/remote_gpu_logs/08435ce3.log`) and includes the
  extracted ControlPlane/ControlClient/RequestClient, correlated RPC routing,
  transactional open/append/update, admission, lazy-load, scheduler, runner,
  fence, mailbox, protocol, serving, and Realtime helper/web regressions.
- Model-policy E2E: three distinct WAV inputs were sent without transcript
  hints in 200 ms real-time chunks. It completed three input commits, two
  model-owned spoken responses, 34 audio/transcript deltas, and two playback
  history commits. It had zero error/cancel/truncate/stale events and exactly
  one `response.speak` per response. Physical inputs, model turns, and response
  cardinality were deliberately not treated as a 1:1 mapping. The client task
  is `e6757f74` (`/tmp/remote_gpu_logs/e6757f74.log`).
- Pinned response-required E2E: the first 1400 ms of the SHA-pinned fixture
  produced one complete spoken response, 14 audio/transcript deltas, and one
  playback history commit, with symmetric created/audio-done/done lifecycle
  and no duplicate `response.speak`. The client task is `2615ca06`
  (`/tmp/remote_gpu_logs/2615ca06.log`).
- The pinned output is 24 kHz mono 16-bit PCM, 13.68 seconds, with RMS amplitude
  0.096499 and peak amplitude 0.944733. Whisper-small independently produced a
  non-empty Chinese transcription semantically consistent with the protocol
  transcript; the ASR task is `4bb31d78`
  (`/tmp/remote_gpu_logs/4bb31d78.log`).
- The validation server task `bf2f8b78` exited cleanly, port 8107 was released,
  and the two GPUs used by that service returned to idle. Other workloads on
  the host were not modified.

### Regression lineage

Earlier synchronized checkpoints passed broader affected matrices before the
extension, policy, and ledger consolidation. Those runs were useful while
developing the refactor but are superseded by the current file-identical H20
matrix and E2E evidence above; they are intentionally omitted to avoid mixing
different trees in the acceptance argument.

### Focused suites

- RPC router and AsyncOmni shutdown/error routing;
- orchestrator fatal and duplex control paths;
- WebSocket Actor mailbox and writer;
- duplex protocol and Session ownership;
- Realtime handler, terminal ordering, overlap, playback ACK;
- scheduler resumable boundary;
- MiniCPM runner and Stage0/Stage1 lifecycle.

### E2E contract

Run two deliberately separate fixture contracts:

- model-policy uses three distinct inputs, no transcript hints, and 200 ms
  real-time pacing. It allows natural listen or speak decisions and assumes no
  1:1 mapping between physical inputs, model turns, and responses;
- response-required uses `minicpmo_pr3907_jiayan_16k.wav` pinned to SHA-256
  `2e5fd4eb3ee434ce107ee3a0591fa624a33f7683c7462f45fe651c443c9af941`,
  sends its first 1400 ms, and requires one complete spoken response.
  It is an output-lifecycle fixture, not evidence that arbitrary no-hint,
  real-time input must select speak;
- late and old-response playback ACK handling in the affected test matrix;
- final model-policy response drain before `session.close`;
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
- request completion removes MiniCPM helper and model-owned force-listen state;
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
