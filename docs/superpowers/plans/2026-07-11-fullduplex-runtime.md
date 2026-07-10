# Full-Duplex Runtime Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #3907 onto current main and replace its duplicated duplex state machines with one typed runtime under `experimental/fullduplex`, while preserving MiniCPM-o 4.5 clean multi-turn audio E2E behavior.

**Architecture:** A pure reducer owns `DuplexFence` and all session/turn/response transitions. OpenAI, engine, and MiniCPM modules communicate through typed events and ports; WebSocket actors and engine handles own resources but never advance domain identity. The existing scheduler placeholder/resumable-request implementation remains isolated in the Omni engine port.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, typing Protocol, pytest, vLLM 0.24, vLLM-Omni orchestrator and OpenAI Realtime serving.

---

## File Map

Create:

- `vllm_omni/experimental/fullduplex/core/identity.py`: immutable stream fence.
- `vllm_omni/experimental/fullduplex/core/events.py`: typed domain events and effects.
- `vllm_omni/experimental/fullduplex/core/state.py`: pure transition reducer.
- `vllm_omni/experimental/fullduplex/core/playback.py`: playback cursor.
- `vllm_omni/experimental/fullduplex/core/ports.py`: engine and sink protocols.
- `vllm_omni/experimental/fullduplex/engine/omni.py`: current engine adapter.
- `vllm_omni/experimental/fullduplex/openai/audio.py`: Realtime audio codecs.
- `vllm_omni/experimental/fullduplex/openai/history.py`: conversation projection.
- `vllm_omni/experimental/fullduplex/openai/realtime.py`: protocol translation.
- `vllm_omni/experimental/fullduplex/openai/data_plane.py`: output normalization.
- `vllm_omni/experimental/fullduplex/openai/websocket.py`: queue/task actor.
- `vllm_omni/experimental/fullduplex/openai/serving.py`: thin session controller.
- `vllm_omni/experimental/fullduplex/minicpmo45/input.py`: PCM append/commit state.
- `vllm_omni/experimental/fullduplex/minicpmo45/adapter.py`: model event mapping.
- `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py`: Stage0 runtime.
- `vllm_omni/experimental/fullduplex/minicpmo45/stage1.py`: Stage1 runtime.
- `vllm_omni/experimental/fullduplex/minicpmo45/policy.py`: token/framing contract.
- `vllm_omni/experimental/fullduplex/minicpmo45/worker.py`: provider integration.
- focused tests under `tests/fullduplex/`.

Modify:

- existing `experimental/fullduplex/core` exports and JoyVL adapter/tests.
- orchestrator, async engine, worker mixins, runner hooks, OpenAI API registration.
- MiniCPM model and stage input processor imports.
- realtime demo gates and MiniCPM documentation.

Delete after migration:

- `vllm_omni/experimental/duplex/`.
- tests that only assert duplicate legacy state rather than public behavior.

## Task 1: Rebase and Establish the Baseline

**Files:**
- Modify through conflict resolution: `vllm_omni/worker/gpu_ar_model_runner.py`
- Modify through conflict resolution: `vllm_omni/worker/gpu_model_runner.py`

- [ ] **Step 1: Record current references**

Run:

```bash
rtk git fetch origin main
rtk git rev-parse HEAD origin/main
rtk git rev-list --left-right --count HEAD...origin/main
```

Expected before rebase: the PR branch is ahead by its WIP commits and behind the current main.

- [ ] **Step 2: Rebase onto current main**

Run:

```bash
rtk git rebase origin/main
```

Resolve runner conflicts by retaining current-main runner APIs and only the smallest duplex provider hooks. Do not retain vLLM 0.24 compatibility copies when current main already supplies the equivalent behavior.

- [ ] **Step 3: Verify ancestry and imports**

Run:

```bash
rtk git merge-base --is-ancestor origin/main HEAD
rtk python3 -m compileall -q vllm_omni
```

Expected: both commands exit 0.

- [ ] **Step 4: Run existing focused tests**

Run:

```bash
rtk pytest -q tests/fullduplex tests/engine/test_duplex_runtime.py tests/entrypoints/openai/test_duplex_protocol.py
```

Expected: capture the exact baseline pass/fail list. Failures caused by import movement become RED inputs to later tasks; unrelated failures are fixed before continuing.

## Task 2: Add the Single Fence and Pure Reducer

**Files:**
- Create: `vllm_omni/experimental/fullduplex/core/identity.py`
- Create: `vllm_omni/experimental/fullduplex/core/events.py`
- Create: `vllm_omni/experimental/fullduplex/core/playback.py`
- Create: `vllm_omni/experimental/fullduplex/core/state.py`
- Test: `tests/fullduplex/test_identity.py`
- Test: `tests/fullduplex/test_state.py`

- [ ] **Step 1: Write RED fence tests**

Tests assert:

```python
def test_fence_advances_turn_without_advancing_epoch():
    state = DuplexSessionState.open("s")
    new_state, effects = reduce_event(state, InputCommitted())
    assert new_state.fence == DuplexFence("s", epoch=0, turn_id=1, response_seq=1)
    assert state.fence == DuplexFence("s")
    assert [type(effect) for effect in effects] == [ReserveResponse, AppendToEngine]


def test_interruption_invalidates_old_fence_atomically():
    state = responding_state("s", epoch=0, turn_id=2, response_seq=2)
    old = state.fence
    new_state, effects = reduce_event(state, InterruptRequested(reason="test"))
    assert new_state.fence.epoch == 1
    assert new_state.active_response is None
    assert CancelFence(old) in effects
```

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk pytest -q tests/fullduplex/test_identity.py tests/fullduplex/test_state.py
```

Expected: import or symbol failures because the new core is absent.

- [ ] **Step 3: Implement immutable identity and events**

Implement:

```python
@dataclass(frozen=True, slots=True)
class DuplexFence:
    session_id: str
    epoch: int = 0
    turn_id: int = 0
    response_seq: int = 0

    def next_turn(self) -> "DuplexFence":
        return replace(self, turn_id=self.turn_id + 1, response_seq=self.response_seq + 1)

    def next_epoch(self) -> "DuplexFence":
        return replace(self, epoch=self.epoch + 1)
```

Add frozen dataclasses for every input event and effect listed in the design. Effects that target engine or Stage1 carry the complete fence.

- [ ] **Step 4: Implement reducer transition table**

The reducer returns a new `DuplexSessionState` plus effects, rejects illegal
transitions with `DuplexTransitionError`, and never mutates the input state.
`ModelSegmentEnded` does not finish the response; `ModelTurnEnded` does.

- [ ] **Step 5: Verify GREEN and edge cases**

Run:

```bash
rtk pytest -q tests/fullduplex/test_identity.py tests/fullduplex/test_state.py
```

Expected: all pass, including duplicate terminal idempotency, missing fence error, empty turn, clean-turn epoch preservation, and stale output classification.

- [ ] **Step 6: Commit**

```bash
rtk git add vllm_omni/experimental/fullduplex/core tests/fullduplex
rtk git commit -s -m "refactor(fullduplex): add single-fence session reducer"
```

## Task 3: Replace the Demonstration Runtime with the Core Event Loop

**Files:**
- Modify: `vllm_omni/experimental/fullduplex/core/runtime.py`
- Create: `vllm_omni/experimental/fullduplex/core/ports.py`
- Modify: `vllm_omni/experimental/fullduplex/core/adapter.py`
- Modify: `vllm_omni/experimental/fullduplex/core/__init__.py`
- Modify: `vllm_omni/experimental/fullduplex/joyvl/adapter.py`
- Test: `tests/fullduplex/test_runtime.py`

- [ ] **Step 1: Write RED runtime tests**

Use an in-memory engine port and assert:

- input loop remains responsive while output streams;
- every engine command has the reducer fence;
- old-fence model output is dropped after interruption;
- exactly one terminal event is emitted;
- engine errors deterministically close active response resources.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk pytest -q tests/fullduplex/test_runtime.py
```

Expected: failures against the old task-based demonstration runtime.

- [ ] **Step 3: Implement port-driven runtime**

`DuplexRuntime` owns one reducer state and resource tasks. It executes effects through `DuplexEnginePort` and `DuplexEventSink`. It does not parse dictionaries or know OpenAI event names.

- [ ] **Step 4: Migrate JoyVL demonstration**

Adapt JoyVL to emit typed model events. Keep its HTTP serving behavior unchanged.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
rtk pytest -q tests/fullduplex
```

Expected: all full-duplex core and JoyVL tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add vllm_omni/experimental/fullduplex tests/fullduplex
rtk git commit -s -m "refactor(fullduplex): run sessions through typed engine ports"
```

## Task 4: Isolate the Current Omni Engine Port

**Files:**
- Create: `vllm_omni/experimental/fullduplex/engine/__init__.py`
- Create: `vllm_omni/experimental/fullduplex/engine/omni.py`
- Modify: `vllm_omni/engine/orchestrator.py`
- Modify: `vllm_omni/engine/async_omni_engine.py`
- Modify: `vllm_omni/engine/messages.py`
- Test: `tests/fullduplex/test_omni_engine_port.py`
- Test: `tests/engine/test_orchestrator.py`

- [ ] **Step 1: Write RED engine-port contract tests**

Assert open, append, output, cancel, and close preserve the same `DuplexFence`. Assert placeholder budgets are computed only inside the Omni port and missing output identity raises a structured error.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk pytest -q tests/fullduplex/test_omni_engine_port.py
```

Expected: missing port implementation.

- [ ] **Step 3: Move generic engine state**

Move `DuplexSessionRuntimeManager`, stage bindings, data-plane request extraction, and current append planning from `experimental.duplex.engine` into the Omni port. Replace independent `epoch/turn_id/turn_seq` construction with the incoming fence.

- [ ] **Step 4: Thin orchestrator hooks**

Orchestrator handlers accept typed messages containing a fence, bind stage resource handles, and delegate MiniCPM budgeting to the model adapter through the port. They do not own domain lifecycle.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
rtk pytest -q tests/fullduplex/test_omni_engine_port.py tests/engine/test_orchestrator.py tests/engine/test_async_omni_engine_input.py tests/engine/test_async_omni_engine_outputs.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
rtk git add vllm_omni/experimental/fullduplex/engine vllm_omni/engine tests
rtk git commit -s -m "refactor(fullduplex): isolate omni engine data plane"
```

## Task 5: Split and Migrate the MiniCPM-o Adapter

**Files:**
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/__init__.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/input.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/adapter.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/policy.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/stage1.py`
- Create: `vllm_omni/experimental/fullduplex/minicpmo45/worker.py`
- Test: `tests/fullduplex/minicpmo45/test_input.py`
- Test: `tests/fullduplex/minicpmo45/test_adapter.py`
- Test: `tests/fullduplex/minicpmo45/test_stage_state.py`

- [ ] **Step 1: Write RED commit-framing tests**

Assert commit with speech starts final framing even when incremental append drained the PCM buffer. Assert speech accounting is cumulative per turn and reset only by turn end/interruption.

- [ ] **Step 2: Write RED model-event tests**

Assert:

- listen maps to `ModelListening`;
- first speak maps to one `ModelSpeaking`;
- TTS segment EOS maps to `ModelSegmentEnded`;
- only model `turn_eos` maps to `ModelTurnEnded`;
- every output carries the input fence;
- turn end clears Stage1 state but preserves Stage0 conversational context.

- [ ] **Step 3: Verify RED**

Run:

```bash
rtk pytest -q tests/fullduplex/minicpmo45
```

Expected: missing adapter modules.

- [ ] **Step 4: Move policy and input framing**

Move token IDs, 16 kHz unit sizing, append budgets, and terminal silence framing into `policy.py` and `input.py`. No generic module imports MiniCPM token names.

- [ ] **Step 5: Split Stage0 and Stage1**

Move Stage0 persistent model/KV state into `stage0.py`. Move TTS handoff, token-to-wave state, audio offsets, and teardown into `stage1.py`. Both index state by the complete fence and reject mismatches.

- [ ] **Step 6: Implement adapter mapping**

The adapter translates engine outputs to typed model events and provides engine append plans. Characterize the official HF listen/speak behavior before changing the current in-turn listen handling.

- [ ] **Step 7: Verify GREEN and model tests**

Run:

```bash
rtk pytest -q tests/fullduplex/minicpmo45 tests/model_executor/models/minicpmo_4_5 tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py tests/worker/test_native_duplex_hooks.py
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
rtk git add vllm_omni/experimental/fullduplex/minicpmo45 vllm_omni/model_executor tests
rtk git commit -s -m "refactor(minicpmo): split native duplex model adapter"
```

## Task 6: Split OpenAI Realtime Transport and Projection

**Files:**
- Create modules under `vllm_omni/experimental/fullduplex/openai/`
- Modify: `vllm_omni/entrypoints/openai/api_server.py`
- Modify: `vllm_omni/entrypoints/openai/serving_chat.py`
- Test: `tests/fullduplex/openai/test_realtime.py`
- Test: `tests/fullduplex/openai/test_data_plane.py`
- Test: `tests/fullduplex/openai/test_websocket_actor.py`
- Test: `tests/entrypoints/openai_api/test_duplex_handler.py`

- [ ] **Step 1: Write RED lifecycle tests**

For each fence, assert one `response.created`, ordered content parts, no late audio after terminal, one audio done, and one response done. Assert a second turn cannot reuse first-turn text/audio cursors.

- [ ] **Step 2: Write RED protocol purity tests**

Assert OpenAI translation consumes/produces typed domain events and cannot mutate `DuplexSessionState` identity. Assert audio codec helpers are independent pure functions.

- [ ] **Step 3: Verify RED**

Run:

```bash
rtk pytest -q tests/fullduplex/openai
```

Expected: missing split modules.

- [ ] **Step 4: Extract audio and history projection**

Move PCM/G711 conversion to `audio.py`. Move conversation item IDs, truncation, and playback-committed text to `history.py`.

- [ ] **Step 5: Extract data-plane normalization**

Move mm metadata decoding, cumulative audio slicing, token/text cursor handling, and terminal metadata interpretation to `data_plane.py`. Key every cursor by `DuplexFence`; missing identity raises.

- [ ] **Step 6: Implement thin protocol and actor**

`realtime.py` maps OpenAI events to/from domain events. `websocket.py` owns queues, writer task, cancellation, and backpressure. `serving.py` only wires runtime, adapters, and ports.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
rtk pytest -q tests/fullduplex/openai tests/entrypoints/openai/test_duplex_protocol.py tests/entrypoints/openai_api/test_duplex_handler.py tests/entrypoints/test_async_omni_duplex.py
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
rtk git add vllm_omni/experimental/fullduplex/openai vllm_omni/entrypoints tests
rtk git commit -s -m "refactor(fullduplex): split realtime protocol and transport"
```

## Task 7: Remove the Duplicate Runtime and Harden Gates

**Files:**
- Delete: `vllm_omni/experimental/duplex/`
- Modify: all imports returned by `rg experimental.duplex`
- Modify: `examples/online_serving/minicpmo/realtime_duplex_demo.py`
- Modify: `examples/online_serving/minicpmo/README.md`
- Test: `tests/examples/test_minicpmo_realtime_web.py`

- [ ] **Step 1: Add RED demo-gate tests**

Assert the demo rejects:

- duplicate response lifecycle events;
- transcript delta/done mismatch;
- prior-turn suffix leakage;
- empty turn replay;
- audio after response done;
- repeated response IDs.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk pytest -q tests/examples/test_minicpmo_realtime_web.py
```

Expected: the new leakage/lifecycle cases fail against the old gate.

- [ ] **Step 3: Update imports and delete legacy package**

Run after edits:

```bash
rtk rg -n "experimental\.duplex" vllm_omni tests examples
```

Expected: no matches.

- [ ] **Step 4: Strengthen the demo gate**

Use distinct input fixtures and per-response records. Compare transcript delta concatenation to transcript done and, when backend token records are enabled, compare transcript to decoded per-turn token deltas.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
rtk pytest -q tests/examples/test_minicpmo_realtime_web.py tests/fullduplex
rtk python3 -m compileall -q vllm_omni examples/online_serving/minicpmo
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
rtk git add -A
rtk git commit -s -m "refactor(fullduplex): remove duplicate duplex runtime"
```

## Task 8: Full Local Verification

- [ ] **Step 1: Run focused unit matrix**

```bash
rtk pytest -q \
  tests/fullduplex \
  tests/engine/test_orchestrator.py \
  tests/engine/test_output_processor.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/test_async_omni_duplex.py \
  tests/model_executor/models/minicpmo_4_5 \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/worker/test_native_duplex_hooks.py \
  tests/examples/test_minicpmo_realtime_web.py
```

Expected: zero failures.

- [ ] **Step 2: Run formatting and static checks**

```bash
rtk pre-commit run --files $(rtk git diff --name-only origin/main...HEAD)
rtk python3 -m compileall -q vllm_omni
```

Expected: zero failures and no generated-file drift.

- [ ] **Step 3: Audit forbidden patterns**

```bash
rtk rg -n "experimental\.duplex|force_speak|identity absent|stale guard is inert" vllm_omni tests examples
rtk rg -n "except Exception" vllm_omni/experimental/fullduplex
```

Expected: no legacy import or force-speak path. Every broad exception has deterministic error/cleanup behavior.

## Task 9: Remote vLLM 0.24 E2E

- [ ] **Step 1: Sync exact head to the canonical GPU worktree**

Record:

- local SHA;
- remote worktree path and SHA;
- backend command and PID/task;
- backend and demo log paths.

- [ ] **Step 2: Restart backend from the synced worktree**

Verify the backend log reports vLLM 0.24 and the same git SHA.

- [ ] **Step 3: Run distinct-input three-turn E2E**

The client sends only audio append/commit events. It sends neither `response.create` nor force-barge-in. Use a long first/second turn followed by a short turn.

- [ ] **Step 4: Evaluate hard gates**

Require:

- three unique response IDs;
- exactly one created/audio-done/response-done per response;
- audio for every speaking response;
- no late audio;
- transcript delta equals transcript done;
- transcript equals decoded per-turn model tokens;
- no prior-turn suffix;
- model turn EOS precedes response done;
- no timeout, cap, forced listen/speak, missing identity, or stale-guard-inert warning.

- [ ] **Step 5: Preserve artifacts**

Store the event stream, backend log, demo log, transcript/token comparison, and generated audio paths. Report any failed gate with its exact log path before changing code.

## Task 10: Update PR #3907

- [ ] **Step 1: Verify final branch state**

```bash
rtk git status --short
rtk git merge-base --is-ancestor origin/main HEAD
rtk git log --show-signature -1
rtk git log --format='%h %s%n%(trailers:key=Signed-off-by)' origin/main..HEAD
```

Expected: clean tree, current main ancestor, and DCO sign-off on every new commit.

- [ ] **Step 2: Push the PR head**

```bash
rtk git push --force-with-lease sy0307 HEAD:sy03/minicpmo45-duplex-runtime
```

Expected: PR #3907 head updates to the verified SHA. Do not push directly to `origin/main`.

- [ ] **Step 3: Update PR description/comment**

Include:

- architecture summary and package map;
- exact base/head SHAs;
- local test commands/results;
- remote worktree and vLLM version;
- E2E artifact/log paths;
- explicit non-goals: automatic barge-in, scheduler-native append, bounded KV, production concurrency.

- [ ] **Step 4: Check required statuses**

```bash
rtk gh pr checks 3907 --repo vllm-project/vllm-omni
```

Expected: checks are queued or running against the new SHA, with no merge conflict status.
