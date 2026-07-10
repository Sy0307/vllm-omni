# Full-Duplex Runtime Consolidation Design

## Status

Proposed for PR #3907.

This design consolidates the MiniCPM-o 4.5 duplex implementation into
`vllm_omni.experimental.fullduplex`. It targets stable native listen/speak,
automatic response creation, and clean multi-turn audio conversations. Automatic
VAD/model barge-in, scheduler-native append, and bounded KV are explicit follow-up
capabilities.

## Problem

The current implementation has three overlapping sources of session state:

1. OpenAI protocol state in `openai.protocol.DuplexSession`.
2. Engine state in `engine.DuplexSessionRuntimeState`.
3. WebSocket task state in `openai.serving.DuplexSessionActor`.

All three carry parts of epoch, turn, response, request, playback, and teardown
state. The serving handler also owns model policy, engine RPC, data-plane decoding,
OpenAI event translation, and task cancellation. This makes identity and reset
semantics dependent on which path handled an event.

The result is a recurring class of failures:

- clean turns and interrupted turns advance different identity fields;
- Stage0, Stage1, serving cursors, and orchestrator bindings reset on different
  signals;
- model metadata is interpreted differently at each boundary;
- protocol state and engine state can both create or finish a response;
- MiniCPM-specific policy leaks into generic serving and orchestrator code;
- base engine output paths can silently discard duplex metadata or audio.

## Goals

- Establish one source of truth for session, turn, response, and epoch identity.
- Define legal lifecycle transitions in one reducer.
- Separate transport, protocol, runtime, engine, and model responsibilities.
- Preserve MiniCPM-o 4.5 model-owned listen/speak and turn-end behavior.
- Support audio-only clients that never send `response.create`.
- Support at least three distinct-input clean turns without state leakage.
- Keep the current scheduler data-plane implementation behind an interface that
  can later be replaced by scheduler-native append.
- Rebase PR #3907 onto the current `origin/main` and resolve runner conflicts
  through the new adapter boundary.

## Non-Goals

- Claiming automatic VAD or model-driven barge-in support.
- Implementing a block-table append primitive in vLLM.
- Implementing production windowed KV or session migration.
- Claiming production multi-session capacity.
- Preserving internal APIs under `experimental.duplex`.

The runtime will include an interruption fence and cancellation contract. No
automatic source will invoke it for MiniCPM-o 4.5 in this checkpoint.

## Package Layout

```text
vllm_omni/experimental/fullduplex/
  core/
    identity.py       immutable session/turn/response fence
    state.py          domain state and transition reducer
    events.py         typed commands, model events, and effects
    runtime.py        event loop, task ownership, stale filtering
    ports.py          model, engine, protocol, and event-sink interfaces
    playback.py       generated/sent/played/committed cursor
  openai/
    realtime.py       OpenAI Realtime <-> domain event translation
    websocket.py      socket reader/writer and queue ownership
    history.py        conversation item and playback-commit projection
  minicpmo45/
    adapter.py        capabilities and model-event mapping
    input.py          PCM buffering, speech accounting, commit framing
    stage0.py         Stage0 native session and KV/model state
    stage1.py         TTS/vocoder handoff and per-turn state
    policy.py         token and audio framing contract
    worker.py         runner/provider integration
  engine/
    omni.py           current orchestrator/data-plane port implementation
```

The existing JoyVL demonstration will use the same core contracts. Its serving
implementation may remain model-specific, but a second generic runtime must not
remain beside this one.

## Identity

`DuplexFence` is the only identity value transported across layers:

```python
@dataclass(frozen=True, slots=True)
class DuplexFence:
    session_id: str
    epoch: int
    turn_id: int
    response_seq: int
```

Rules:

- `session_id` is allocated once by the runtime.
- `epoch` changes only for interruption or context rebuild.
- `turn_id` advances exactly once when user input is committed.
- `response_seq` advances exactly once when a response is reserved.
- request IDs are derived transport identifiers, never independent domain
  identity.
- every engine append, Stage0 output, Stage1 handoff, cursor key, terminal event,
  and teardown signal carries a complete fence.
- missing identity on a duplex output is an error, not a fail-open condition.

Clean multi-turn retains the same epoch so Stage0 context can continue. An explicit
interruption atomically advances epoch and invalidates every old-fence output.

## Domain State

The core owns one authoritative `DuplexSessionState` value. The reducer returns a
new state plus effects and never mutates its input. Other layers own resources,
not copies of domain state.

```text
Session: OPEN -> CLOSING -> CLOSED

Turn:
  IDLE
    -> INPUT_STREAMING
    -> TURN_COMMITTED
    -> AWAITING_MODEL
    -> RESPONDING
    -> IDLE
```

The state contains:

- current `DuplexFence`;
- pending input accounting;
- active response lifecycle;
- playback cursor;
- committed conversation projection;
- terminal reason;
- capability set.

The WebSocket actor owns queues and tasks only. The engine port owns stage request
handles only. Stage adapters own model tensors and caches only. None of them may
advance epoch, turn, or response independently.

## Events and Effects

Inputs to the reducer are typed domain events:

- `InputStarted`
- `InputChunk`
- `InputCommitted`
- `ModelListening`
- `ModelSpeaking`
- `ModelTextDelta`
- `ModelAudioDelta`
- `ModelSegmentEnded`
- `ModelTurnEnded`
- `PlaybackAcknowledged`
- `InterruptRequested`
- `SessionCloseRequested`
- `EngineFailed`

The reducer is pure and returns effects:

- `AppendToEngine`
- `ReserveResponse`
- `EmitProtocolEvent`
- `CancelFence`
- `ResetStage1`
- `RebuildStage0Context`
- `CloseSessionResources`

The runtime executes effects and feeds resulting events back to the reducer. This
keeps I/O and state transitions separate and makes every transition testable.

## Clean Multi-Turn Flow

1. The OpenAI adapter translates PCM append events to `InputChunk`.
2. The core marks `INPUT_STREAMING` and emits `AppendToEngine(final=False)`.
3. Commit advances `turn_id`, reserves one `response_seq`, and emits
   `AppendToEngine(final=True)`.
4. The MiniCPM adapter provides a terminal framing unit even when the PCM buffer
   was already drained by incremental appends. Response creation must not depend
   on residual bytes in the input buffer.
5. `ModelListening` keeps the response reserved without fabricating speech.
6. The first `ModelSpeaking` starts one external response lifecycle.
7. Text and audio deltas carry the same fence and are stale-checked before
   protocol translation.
8. `ModelSegmentEnded` closes only a TTS segment. It does not finish the
   response.
9. `ModelTurnEnded`, derived from the model's `turn_eos`, is the sole normal
   assistant-turn terminal signal. It emits audio/content/response done exactly
   once and resets Stage1 turn-local state.
10. Stage0 conversational KV remains across clean turns. The next commit advances
    `turn_id` but not `epoch`.

## Interruption Contract

`InterruptRequested` is implemented even though no automatic MiniCPM source is
enabled in this checkpoint.

It performs one atomic transition:

1. capture the old fence;
2. advance epoch;
3. cancel old-fence runtime tasks and engine requests;
4. drop all late old-fence outputs;
5. reset Stage1 turn-local state using the old fence;
6. rebuild Stage0 from playback-committed history before accepting new output.

VAD, client control, or a future reliable model-listen transition can later emit
the same event. No source gets a separate barge-in state machine.

## MiniCPM-o Adapter Boundary

MiniCPM-specific behavior is limited to `minicpmo45`:

- special token IDs and audio-unit sizing;
- final framing for commit;
- model `listen`, `speak`, segment EOS, and turn EOS mapping;
- Stage0 streaming audio encoding and persistent context;
- Stage0-to-Stage1 TTS handoff;
- Stage1 vocoder and cursor reset.

Generic serving must not:

- rewrite listen to speak;
- inject role prefixes into a native session;
- inspect MiniCPM token IDs;
- calculate MiniCPM placeholder budgets;
- decide speech from RMS to control model turn-taking.

RMS may be retained only as observable input accounting or an explicit client
policy. It cannot silently suppress an auto-response commit.

## Engine Port

`OmniDuplexEnginePort` adapts the current engine:

```python
class DuplexEnginePort(Protocol):
    async def open(self, session: SessionOpened) -> EngineSessionHandle: ...
    async def append(self, command: AppendToEngine) -> None: ...
    def outputs(self, handle: EngineSessionHandle) -> AsyncIterator[ModelEvent]: ...
    async def cancel(self, fence: DuplexFence) -> None: ...
    async def close(self, session_id: str) -> None: ...
```

The current port may still convert an append into scheduler token budgets and
resumable requests. That limitation stays inside `engine/omni.py`. Core,
OpenAI, and MiniCPM lifecycle code cannot depend on placeholder lengths.

When vLLM exposes scheduler-native append, a new port implementation replaces this
adapter without changing domain state or protocol behavior.

## OpenAI Realtime Boundary

OpenAI translation is projection code, not runtime policy:

- parse and validate inbound protocol events;
- translate them to typed domain events;
- project domain output events into OpenAI Realtime events;
- enforce exactly-once `response.created`, content-part, audio-done, and
  response-done ordering;
- maintain conversation item IDs as a protocol projection of core history.

Audio format conversion is a separate helper. WebSocket queueing and backpressure
are separate from protocol translation.

## Reset Ownership

All accumulators must declare one lifetime:

- session lifetime: Stage0 conversational context and committed history;
- epoch lifetime: engine bindings and stale-output generation;
- turn lifetime: transcript cursor, Stage1 consumed cursor, token-to-wave buffer,
  TTS audio offsets, audio-text marks;
- segment lifetime: segment-local TTS input only;
- transport lifetime: WebSocket queues and writer task.

Turn-local reset is triggered only by the reducer's `ModelTurnEnded` transition.
Segment EOS cannot trigger turn reset. Interruption runs the same reset effects
while also advancing epoch.

## Error Handling

- Unknown session, illegal transition, missing fence, and fence mismatch fail
  loudly with a structured runtime error.
- Stale output with a valid old fence is counted and dropped, not treated as an
  exception.
- Duplicate terminal events are idempotently ignored and counted.
- Adapter failures close the active response and session resources deterministically.
- Broad `except: log-and-continue` paths are not allowed for identity or metadata.

## Migration

1. Rebase the branch onto the latest `origin/main`.
2. Add characterization tests for current clean multi-turn behavior.
3. Add core identity, events, reducer, and transition tests.
4. Add the engine port around existing orchestrator/data-plane hooks.
5. Migrate MiniCPM Stage0/Stage1 code into focused adapter modules.
6. Migrate OpenAI protocol and WebSocket code onto core events.
7. Migrate the JoyVL core demonstration to the consolidated contracts.
8. Delete `experimental/duplex` and compatibility-only duplicate state.
9. Run unit, protocol, scheduler, and remote E2E verification.

Each migration step must keep tests green and land as a reviewable commit.

## Verification

Unit tests:

- full legal and illegal transition matrix;
- turn and response allocation occurs exactly once;
- clean turn advances turn but preserves epoch and Stage0 state;
- turn EOS clears every turn-local accumulator;
- segment EOS does not finish a response;
- old-fence output is dropped after interruption;
- missing fence fails loudly;
- empty/turn-EOS-only output cannot replay prior text or audio;
- OpenAI lifecycle events are emitted exactly once and in order;
- commit auto-response is independent of residual PCM buffer contents.

Remote E2E on vLLM 0.24:

- audio-only client, no `response.create`, no force-barge-in;
- three distinct prompts, including long-turn followed by short-turn;
- three unique responses with one created and one done each;
- transcript delta concatenation equals transcript done;
- each turn transcript matches that turn's model token delta;
- no prior-turn suffix in the next turn;
- audio delta exists for each speaking response;
- no audio after response done;
- model `turn_eos` precedes normal response completion;
- no missing-identity, stale-guard-inert, timeout, forced-speak, or forced-listen
  fallback.

## Completion Boundary

This checkpoint may claim:

- model-owned MiniCPM listen/speak;
- automatic native response after committed speech;
- clean multi-turn no-barge full-duplex streaming;
- consistent text/audio response lifecycle;
- a unified interruption contract ready for a future source.

It may not claim:

- automatic VAD/model barge-in;
- scheduler-native append;
- bounded minute-scale KV;
- production concurrency.
