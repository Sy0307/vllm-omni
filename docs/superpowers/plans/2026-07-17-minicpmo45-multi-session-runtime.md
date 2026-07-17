# MiniCPM-o 4.5 Native Duplex Multi-Session Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support an unbounded number of logical MiniCPM-o 4.5 native duplex sessions on one Stage0/Stage1 replica pair with scheduler-managed compute admission, transport-neutral leases, secure resume, bounded replay, and per-session model-state isolation.

**Architecture:** Extend the existing `DuplexSessionRuntimeManager` and `DuplexControlPlane` as the runtime lifecycle authority. Keep authentication, transport attachment, protocol ledgers, and output journals in focused serving components, while moving every mutable Stage0 and Stage1 streaming cache into typed session/response state. No fixed serving admission limit is retained.

**Tech Stack:** Python 3.10+, asyncio, msgspec, dataclasses, vLLM v1 scheduler, OpenAI Realtime WebSocket protocol, NumPy, PyTorch, pytest, Ruff, remote NVIDIA H20 validation.

---

## File map

### New focused modules

- `vllm_omni/engine/duplex_lease.py`: immutable lease configuration plus monotonic lease state transitions.
- `vllm_omni/experimental/fullduplex/openai/session_attachment.py`: resume credentials, attachment generations, event journal, and atomic serving registry.
- `tests/engine/test_duplex_lease.py`: pure lease and runtime-manager lifecycle tests.
- `tests/entrypoints/openai/test_duplex_session_attachment.py`: token, takeover, journal, and backpressure tests.
- `examples/online_serving/minicpmo/realtime_duplex_multi_session_e2e.py`: engine/public two-session and reconnect E2E driver.

### Existing files with bounded responsibilities

- `vllm_omni/config/stage_config.py`: typed deploy configuration and YAML parsing.
- `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py`: MiniCPM capability defaults; remove fixed session admission.
- `vllm_omni/engine/messages.py`: typed touch/resume/lifecycle messages.
- `vllm_omni/engine/duplex_session.py`: runtime state and manager transitions.
- `vllm_omni/engine/duplex_control_plane.py`: control handlers and cleanup transaction.
- `vllm_omni/engine/orchestrator.py`: periodic reaper tick and lifecycle sink wiring only.
- `vllm_omni/engine/duplex_control_client.py`: correlated touch/resume client methods.
- `vllm_omni/engine/async_omni_engine.py`: expose typed control client; remove capacity propagation.
- `vllm_omni/entrypoints/duplex_request_client.py`: transport-neutral public engine-client facade.
- `vllm_omni/experimental/fullduplex/openai/protocol.py`: capabilities and protocol session projection.
- `vllm_omni/experimental/fullduplex/openai/realtime_input.py`: resume, heartbeat, and event-ack parsing.
- `vllm_omni/experimental/fullduplex/openai/realtime_output.py`: event sequence projection without reducer replay.
- `vllm_omni/experimental/fullduplex/openai/session_runner.py`: attach/resume/detach flow; no unconditional disconnect close.
- `vllm_omni/experimental/fullduplex/openai/serving.py`: compose attachment registry and config; no session-count semaphore.
- `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py`: session-local streaming processor view.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`: response-owned vocoder caches.
- `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`: lease, replay, and input-buffer defaults.

The pre-existing dirty video files are out of scope and must remain unstaged:

- `vllm_omni/entrypoints/openai/video_stream_base.py`
- `tests/entrypoints/openai_api/test_serving_video_stream.py`
- `examples/online_serving/minicpmo/mmf_video_input_experiment.py`
- `tests/examples/test_minicpmo_mmf_video_input_experiment.py`

## Remote validation convention

All pytest and runtime commands run on H20 from:

```bash
cd /home/admin/workspace/aop_lab/model_runner_v2/vllm-omni-worktrees/pr3907-ready-rebase-0717
```

The local Mac runs only Ruff, formatting, compile, diff, git, and file sync.

### Task 1: Typed duplex deploy configuration and capability schema

**Files:**
- Modify: `vllm_omni/config/stage_config.py`
- Modify: `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/protocol.py`
- Modify: `vllm_omni/engine/async_omni_engine.py`
- Modify: `vllm_omni/entrypoints/async_omni.py`
- Modify: `vllm_omni/entrypoints/openai/api_server.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/serving.py`
- Modify: `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`
- Test: `tests/config/test_deploy_config.py`
- Test: `tests/model_executor/models/minicpmo_4_5/test_pipeline.py`
- Test: `tests/entrypoints/openai/test_duplex_protocol.py`

- [ ] **Step 1: Write RED config and capability tests**

Add assertions that the nested YAML resolves to a typed immutable config with
the approved defaults and rejects zero/negative values:

```python
assert deploy.duplex_session.idle_ttl_s == 300.0
assert deploy.duplex_session.disconnect_grace_s == 30.0
assert deploy.duplex_session.reaper_interval_s == 5.0
assert deploy.duplex_session.resume_replay_max_bytes_per_session == 8 * 1024 * 1024
assert deploy.duplex_session.max_pending_input_bytes_per_session == 16 * 1024 * 1024
assert deploy.duplex_session.max_pending_turns_per_session == 4
```

Assert MiniCPM capabilities report multi-session/session-resume support while
both KV-lease fields remain false. Assert `PipelineConfig` and the serving
handler no longer expose `max_native_duplex_sessions`.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q \
  tests/config/test_deploy_config.py \
  tests/model_executor/models/minicpmo_4_5/test_pipeline.py \
  tests/entrypoints/openai/test_duplex_protocol.py
```

Expected: failures for missing `duplex_session`, missing capability fields, and
the old fixed session limit.

- [ ] **Step 3: Implement typed config and capability fields**

Add `DuplexSessionRuntimeConfig` to `stage_config.py`:

```python
@dataclass(frozen=True)
class DuplexSessionRuntimeConfig:
    idle_ttl_s: float | None = 300.0
    disconnect_grace_s: float = 30.0
    reaper_interval_s: float = 5.0
    resume_replay_ttl_s: float = 60.0
    resume_replay_max_bytes_per_session: int = 8 * 1024 * 1024
    max_pending_input_bytes_per_session: int = 16 * 1024 * 1024
    max_pending_turns_per_session: int = 4

    def __post_init__(self) -> None:
        positive = {
            "disconnect_grace_s": self.disconnect_grace_s,
            "reaper_interval_s": self.reaper_interval_s,
            "resume_replay_ttl_s": self.resume_replay_ttl_s,
            "resume_replay_max_bytes_per_session": self.resume_replay_max_bytes_per_session,
            "max_pending_input_bytes_per_session": self.max_pending_input_bytes_per_session,
            "max_pending_turns_per_session": self.max_pending_turns_per_session,
        }
        if self.idle_ttl_s is not None and self.idle_ttl_s <= 0:
            raise ValueError("duplex_session.idle_ttl_s must be positive or null")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"duplex_session.{name} must be positive")
```

Parse `duplex_session` in `load_deploy_config`, propagate the typed value to
the engine client and serving handler, delete `max_native_duplex_sessions`, and
add capability fields `supports_session_lease`, `supports_session_resume`, and
`session_admission_mode`.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 1 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add vllm_omni/config/stage_config.py \
  vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py \
  vllm_omni/experimental/fullduplex/openai/protocol.py \
  vllm_omni/engine/async_omni_engine.py \
  vllm_omni/entrypoints/async_omni.py \
  vllm_omni/entrypoints/openai/api_server.py \
  vllm_omni/experimental/fullduplex/openai/serving.py \
  vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  tests/config/test_deploy_config.py \
  tests/model_executor/models/minicpmo_4_5/test_pipeline.py \
  tests/entrypoints/openai/test_duplex_protocol.py
git commit -s -m "feat: configure duplex multi-session runtime"
```

### Task 2: Monotonic engine lease state

**Files:**
- Create: `vllm_omni/engine/duplex_lease.py`
- Modify: `vllm_omni/engine/duplex_session.py`
- Create: `tests/engine/test_duplex_lease.py`
- Modify: `tests/engine/test_duplex_runtime.py`

- [ ] **Step 1: Write RED lease transition tests**

Use an injected clock and cover open, touch, detach, grace-end response cancel,
resume, token-independent lease generation, idle expiry, active-operation
protection, and close/reaper races. The core test shape is:

```python
clock = FakeMonotonicClock(100.0)
manager = DuplexSessionRuntimeManager(clock=clock)
session = manager.open_session(fence, lease_config=lease_config)
session.detach(fence)
clock.advance(29.0)
assert manager.collect_expired() == []
clock.advance(272.0)
assert [item.session_id for item in manager.collect_expired()] == [fence.session_id]
```

Add a test proving session A touch/close never changes session B deadlines or
resources.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q tests/engine/test_duplex_lease.py tests/engine/test_duplex_runtime.py
```

Expected: import or constructor failures for the missing lease types.

- [ ] **Step 3: Implement lease state and manager APIs**

Create frozen `DuplexLeaseConfig` and mutable `DuplexLeaseState` with explicit
methods:

```python
class DuplexLeaseActivity(str, Enum):
    APPEND = "append"
    SIGNAL = "signal"
    PLAYBACK_ACK = "playback_ack"
    HEARTBEAT = "heartbeat"
    ATTACH = "attach"
    DETACH = "detach"
    RESUME = "resume"
    MODEL_OUTPUT = "model_output"

@dataclass
class DuplexLeaseState:
    generation: int
    last_activity: float
    detached_at: float | None = None
    active_operations: set[str] = field(default_factory=set)
    terminal_reason: str | None = None
```

Manager transitions validate `DuplexFence`, use `time.monotonic` through an
injected callable, and return immutable expiry records before removal. Close
and expiry use one terminal compare-and-transition method.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add vllm_omni/engine/duplex_lease.py \
  vllm_omni/engine/duplex_session.py \
  tests/engine/test_duplex_lease.py \
  tests/engine/test_duplex_runtime.py
git commit -s -m "feat: add duplex session leases"
```

### Task 3: Typed touch/resume control and periodic reaper

**Files:**
- Modify: `vllm_omni/engine/messages.py`
- Modify: `vllm_omni/engine/duplex_control_plane.py`
- Modify: `vllm_omni/engine/duplex_control_client.py`
- Modify: `vllm_omni/engine/orchestrator.py`
- Modify: `vllm_omni/entrypoints/duplex_request_client.py`
- Modify: `tests/engine/test_duplex_control_plane.py`
- Modify: `tests/engine/test_duplex_control_client.py`
- Modify: `tests/engine/test_orchestrator.py`
- Modify: `tests/entrypoints/test_async_omni_duplex.py`

- [ ] **Step 1: Write RED control/reaper tests**

Add `TouchDuplexSessionMessage`, `ResumeDuplexSessionMessage`, and
`DuplexSessionLifecycleMessage` round-trip tests. Verify the control plane:

- touches append/signal automatically;
- accepts heartbeat/playback touch without scheduler submission;
- resumes only the expected lease generation;
- reaps A through the standard cleanup port while leaving B bound;
- emits an unsolicited lifecycle message through the lifecycle sink;
- does not put lifecycle events into an RPC waiter.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q \
  tests/engine/test_duplex_control_plane.py \
  tests/engine/test_duplex_control_client.py \
  tests/engine/test_orchestrator.py \
  tests/entrypoints/test_async_omni_duplex.py
```

Expected: missing message and client-method failures.

- [ ] **Step 3: Implement control messages and reaper tick**

Add client methods with the existing correlation-ID router:

```python
async def touch_session(
    self,
    session_id: str,
    *,
    fence: DuplexFence,
    activity: DuplexLeaseActivity,
) -> DuplexControlResultMessage:
    return await self._call_control(
        TouchDuplexSessionMessage(
            control_id=random_uuid(),
            session_id=session_id,
            fence=fence,
            activity=activity.value,
        )
    )
```

Add `DuplexControlPlane.reap_expired(now)` and call it from the orchestrator
event loop no more often than `reaper_interval_s`. Expiry invokes the same
submitted/reserved cleanup split as explicit close. Wire a dedicated lifecycle
output path that does not require a pending control ID.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 3 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add vllm_omni/engine/messages.py \
  vllm_omni/engine/duplex_control_plane.py \
  vllm_omni/engine/duplex_control_client.py \
  vllm_omni/engine/orchestrator.py \
  vllm_omni/entrypoints/duplex_request_client.py \
  tests/engine/test_duplex_control_plane.py \
  tests/engine/test_duplex_control_client.py \
  tests/engine/test_orchestrator.py \
  tests/entrypoints/test_async_omni_duplex.py
git commit -s -m "feat: route duplex lease control"
```

### Task 4: Serving credential, attachment, and event journal primitives

**Files:**
- Create: `vllm_omni/experimental/fullduplex/openai/session_attachment.py`
- Create: `tests/entrypoints/openai/test_duplex_session_attachment.py`

- [ ] **Step 1: Write RED primitive tests**

Cover 256-bit token creation, digest-only storage, constant-time validation,
rotation, stale-token rejection, atomic attachment replacement, monotonic event
sequence, ACK removal, replay, TTL pruning, byte overflow, and independent A/B
journals. Assert `repr` never contains plaintext tokens.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q tests/entrypoints/openai/test_duplex_session_attachment.py
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement focused serving components**

Implement these public types:

```python
@dataclass(frozen=True)
class ResumeToken:
    plaintext: str

@dataclass
class DuplexTransportAttachment:
    generation: int
    send: Callable[[dict[str, object]], Awaitable[None]]
    close: Callable[[str], Awaitable[None]]

@dataclass(frozen=True)
class JournalEntry:
    sequence: int
    created_monotonic: float
    encoded_bytes: int
    payload: Mapping[str, object]
```

`DuplexResumeCredential` stores `sha256(token)` only and uses
`hmac.compare_digest`. `DuplexEventJournal.record` serializes once to calculate
the exact byte cost. `DuplexSessionAttachmentRegistry.resume` verifies and
rotates under one asyncio lock, installs the new attachment generation, and
returns retained events after the acknowledged sequence.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 4 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add vllm_omni/experimental/fullduplex/openai/session_attachment.py \
  tests/entrypoints/openai/test_duplex_session_attachment.py
git commit -s -m "feat: add duplex resume attachments"
```

### Task 5: Realtime resume, heartbeat, ACK, and disconnect lifecycle

**Files:**
- Modify: `vllm_omni/experimental/fullduplex/openai/protocol.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/realtime_input.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/realtime_output.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/session_runner.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/serving.py`
- Modify: `tests/entrypoints/openai_api/test_duplex_handler.py`
- Modify: `tests/entrypoints/openai/test_duplex_protocol.py`

- [ ] **Step 1: Write RED protocol lifecycle tests**

Test `session.created` token emission, `session.resume`, token rotation,
`session.heartbeat_ack`, `session.event_ack`, `session.replaced`, replay order,
invalid token, resume conflict, resume after expiry, and journal-gap
`session.resync_required`. Add a disconnect test proving the runner detaches
without calling runtime close, and a grace-expiry test proving only the active
response is cancelled before idle TTL closes the session.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/openai/test_duplex_protocol.py
```

Expected: old disconnect-close assertions fail and new events are unsupported.

- [ ] **Step 3: Integrate attachment registry with the session runner**

Create one attachment record at session open. Every outbound event passes
through `journal.record` before transport send and receives
`server_event_seq`. Resume authenticates before engine `resume_session`, then
atomically swaps attachments and replays payloads without invoking session
domain transitions. Replace the current `finally` close sequence with:

```python
if explicit_close:
    await close_runtime_and_projection(session, reason=close_reason)
else:
    await attachment_registry.detach(session.session_id, attachment_generation)
    await duplex_client.touch_session(
        session.session_id,
        fence=DuplexFence(
            session_id=session.session_id,
            incarnation=session.incarnation,
            epoch=session.epoch,
            turn_id=session.turn_id,
            response_seq=session.response_seq,
        ),
        activity=DuplexLeaseActivity.DETACH,
    )
```

Schedule grace-end response cancellation as an attachment-registry task and
cancel it after successful resume.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 5 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add vllm_omni/experimental/fullduplex/openai/protocol.py \
  vllm_omni/experimental/fullduplex/openai/realtime_input.py \
  vllm_omni/experimental/fullduplex/openai/realtime_output.py \
  vllm_omni/experimental/fullduplex/openai/session_runner.py \
  vllm_omni/experimental/fullduplex/openai/serving.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/openai/test_duplex_protocol.py
git commit -s -m "feat: resume duplex realtime sessions"
```

### Task 6: Per-session input backpressure

**Files:**
- Modify: `vllm_omni/experimental/fullduplex/openai/protocol.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/realtime_input.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/session_runner.py`
- Modify: `tests/entrypoints/openai_api/test_duplex_handler.py`

- [ ] **Step 1: Write RED A/B backpressure tests**

Fill A to the configured byte and pending-turn bounds, assert the next event is
rejected before mutation with `input_backpressure`, and assert B can still
append, commit, and receive output. Add a test proving accepted PCM length is
unchanged.

- [ ] **Step 2: Run RED test on H20**

```bash
python -m pytest -q tests/entrypoints/openai_api/test_duplex_handler.py -k backpressure
```

Expected: A accepts data beyond the new bound.

- [ ] **Step 3: Implement reducer-owned reservation**

Add `DuplexSession.reserve_input_bytes(size)` and
`DuplexSession.release_committed_turn(size)` so limit checks and input mutation
are one transaction. The runner reads server-owned limits only; session.update
and `extra_body` cannot override them.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 6 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add vllm_omni/experimental/fullduplex/openai/protocol.py \
  vllm_omni/experimental/fullduplex/openai/realtime_input.py \
  vllm_omni/experimental/fullduplex/openai/session_runner.py \
  tests/entrypoints/openai_api/test_duplex_handler.py
git commit -s -m "feat: bound duplex session input"
```

### Task 7: Stage0 streaming processor isolation

**Files:**
- Modify: `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py`
- Modify: `tests/worker/test_native_duplex_hooks.py`
- Modify: `tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py`

- [ ] **Step 1: Write RED alternating-session processor tests**

Use a fake processor whose streaming mel state increments per processed chunk.
Open A and B, alternate A1/B1/A2/B2, and assert both produce local counters
`[1, 2]`. Assert opening or closing B never invokes A's reset. Add stale
incarnation/fence rejection.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q \
  tests/worker/test_native_duplex_hooks.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  -k "duplex and (session or streaming or stale)"
```

Expected: the shared processor counter/reset contaminates the other session.

- [ ] **Step 3: Implement the session-local processor view**

Extend `_MiniCPMO45Stage0SessionState` with `streaming_processor`. On first
audio append:

```python
session_processor = copy.copy(self.processor)
shared_mel = getattr(self.processor, "_streaming_mel_processor", None)
session_processor._streaming_mel_processor = copy.deepcopy(shared_mel)
self._configure_streaming_processor(session_processor)
state.streaming_processor = session_processor
```

Change `_configure_streaming_processor`, `_process_streaming_audio`, and
consumed-sample calculations to accept the session processor explicitly.
Session cleanup drops only that reference and never resets the shared
processor.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 7 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add vllm_omni/experimental/fullduplex/minicpmo45/stage0.py \
  tests/worker/test_native_duplex_hooks.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py
git commit -s -m "fix: isolate MiniCPM duplex audio processors"
```

### Task 8: Stage1 talker and Token2Wav cache isolation

**Files:**
- Modify: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`
- Modify: `tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py`

- [ ] **Step 1: Write RED interleaved vocoder tests**

Create A and B response identities with a fake tokenizer that records installed
cache markers. Alternate A1/B1/A2/B2 windows. Assert observed markers are
`A0, B0, A1, B1`, both outputs retain their own continuity, closing A does not
clear B, and the shared tokenizer fields are empty after each call.

- [ ] **Step 2: Run RED test on H20**

```bash
python -m pytest -q \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  -k "token2wav and (interleave or session or response)"
```

Expected: shared `stream_cache`/`hift_cache_dict` crossover or global reset.

- [ ] **Step 3: Move vocoder cache into `_TalkerTurnState`**

Add `stream_cache`, `hift_cache_dict`, and `vocoder_initialized` slots. Replace
global begin/reset with `_run_vocoder_window(state, token_list, last_chunk)`:

```python
with self._token2wav_state_lock:
    self.audio_tokenizer.stream_cache = _torch_clone_recursive(state.stream_cache)
    self.audio_tokenizer.hift_cache_dict = _torch_clone_recursive(state.hift_cache_dict)
    try:
        waveform = self.audio_tokenizer.stream(
            token_list,
            state.prompt_wav_path,
            last_chunk=last_chunk,
            return_waveform=True,
        )
        state.stream_cache = _torch_clone_recursive(self.audio_tokenizer.stream_cache)
        state.hift_cache_dict = _torch_clone_recursive(self.audio_tokenizer.hift_cache_dict)
    finally:
        self.audio_tokenizer.stream_cache = None
        self.audio_tokenizer.hift_cache_dict = {}
```

Use the full typed duplex identity for the state key and keep the lock around
one model call only.

- [ ] **Step 4: Run GREEN tests on H20**

Run the Task 8 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py
git commit -s -m "fix: isolate MiniCPM duplex vocoder state"
```

### Task 9: Scheduler-managed multi-session and preemption correctness

**Files:**
- Modify: `tests/core/sched/test_omni_ar_scheduler_streaming.py`
- Modify: `tests/core/sched/test_omni_scheduling_coordinator.py`
- Modify: `tests/engine/test_orchestrator.py`
- Modify: `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`
- Modify only if RED requires it: `vllm_omni/core/sched/omni_ar_scheduler.py`
- Modify only if RED requires it: `vllm_omni/core/sched/omni_scheduling_coordinator.py`

- [ ] **Step 1: Write RED scheduler tests with two parked resumable requests**

Run A to `WAITING_FOR_CHUNK`, admit B, append A again, and verify both request
identities, append cursors, and segment policies survive waiting/preemption.
Add a constrained-KV test that either recomputes A correctly or returns an
explicit context-invalidating error; it must never resume with empty context.

- [ ] **Step 2: Run RED tests on H20**

```bash
python -m pytest -q \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/engine/test_orchestrator.py \
  -k "resumable or waiting_for_chunk or duplex"
```

Expected: any real parked-request or preemption defect fails before scheduler
changes are made. If the existing scheduler passes, preserve it unchanged.

- [ ] **Step 3: Implement only the RED-proven scheduler correction**

Keep the typed `ResumableSegmentPolicy` boundary. Do not parse MiniCPM tokens or
Realtime config in scheduler code. Ensure parked requests do not count as an
application session cap and that restore preserves request-owned append state.
Set `max_num_seqs: 2` in the duplex deploy overlay for the correctness E2E;
larger stress loads remain scheduler queued.

- [ ] **Step 4: Run GREEN scheduler tests on H20**

Run the Task 9 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/engine/test_orchestrator.py \
  vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  vllm_omni/core/sched/omni_ar_scheduler.py \
  vllm_omni/core/sched/omni_scheduling_coordinator.py
git commit -s -m "fix: schedule resumable duplex sessions fairly"
```

Before staging, omit either scheduler source file if it remained unchanged.

### Task 10: Multi-session engine and WebSocket E2E driver

**Files:**
- Create: `examples/online_serving/minicpmo/realtime_duplex_multi_session_e2e.py`
- Modify: `tests/examples/test_minicpmo_realtime_web.py`
- Modify: `docs/design/minicpmo45_duplex_runtime_architecture.md`

- [ ] **Step 1: Write driver contract tests**

Add pure tests for two-client event attribution, token redaction, replay
sequence, expected model-policy cardinality, response-required cardinality,
per-session WAV collection, playback ACK, lease expiry, and pressure summary.

- [ ] **Step 2: Run RED driver tests on H20**

```bash
python -m pytest -q tests/examples/test_minicpmo_realtime_web.py -k multi_session
```

Expected: missing driver helpers or assertions.

- [ ] **Step 3: Implement the E2E driver**

The driver exposes:

```text
--sessions N
--realtime-input
--chunk-ms 200
--disconnect-session-index 0
--resume-after-ms 1000
--expire-session-index
--response-required
--output-dir PATH
```

It creates independent clients, alternates chunk sends with `asyncio.gather`,
tracks every event by session/response/item ID and `server_event_seq`, ACKs
playback by response, verifies rotated tokens are redacted, saves one WAV per
spoken response, and prints a machine-readable JSON summary.

- [ ] **Step 4: Run GREEN driver tests on H20**

Run the Task 10 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 10**

```bash
git add examples/online_serving/minicpmo/realtime_duplex_multi_session_e2e.py \
  tests/examples/test_minicpmo_realtime_web.py \
  docs/design/minicpmo45_duplex_runtime_architecture.md
git commit -s -m "test: cover MiniCPM duplex multi-session E2E"
```

### Task 11: Final remote matrix, pressure, ASR, and cleanup

**Files:**
- Modify only for defects proven by this task: files from Tasks 1-10
- Update: `docs/design/minicpmo45_duplex_runtime_architecture.md`

- [ ] **Step 1: Sync exact changed files and verify SHA-256 parity**

Generate a NUL-safe changed-file list excluding the four pre-existing video
paths, sync to the remote worktree, and compare SHA-256 for every runtime and
test file before executing pytest.

- [ ] **Step 2: Run the remote affected matrix**

```bash
python -m pytest -q \
  tests/engine/test_duplex_lease.py \
  tests/engine/test_duplex_runtime.py \
  tests/engine/test_duplex_control_plane.py \
  tests/engine/test_duplex_control_client.py \
  tests/engine/test_orchestrator.py \
  tests/entrypoints/test_async_omni_duplex.py \
  tests/entrypoints/openai/test_duplex_session_attachment.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/worker/test_native_duplex_hooks.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/model_executor/models/minicpmo_4_5/test_pipeline.py \
  tests/config/test_deploy_config.py \
  tests/examples/test_minicpmo_realtime_web.py
```

Expected: all selected tests pass; warnings are recorded separately.

- [ ] **Step 3: Run engine and public two-session E2E**

Launch the exact final tree with `minicpmo_4_5_duplex.yaml`, then run:

```bash
python examples/online_serving/minicpmo/realtime_duplex_multi_session_e2e.py \
  --base-url http://127.0.0.1:8113 \
  --model openbmb/MiniCPM-o-4_5 \
  --sessions 2 \
  --realtime-input \
  --chunk-ms 200 \
  --disconnect-session-index 0 \
  --resume-after-ms 1000 \
  --response-required \
  --output-dir /tmp/minicpmo_pr3907_multi_session_e2e
```

Expected: both sessions complete, the disconnected session replays without a
gap, the token rotates, no output crosses identities, and all response audio is
ACKed exactly once.

- [ ] **Step 4: Run TTL and pressure matrix**

Use a test overlay with short TTL for expiry. Then run the driver with
`--sessions 2`, `4`, `8`, `16`, and `32`, without response-required exact
cardinality for model-policy pressure runs. Record every passed and failed
scale; treat OOM, context loss, stale output, or cross-session contamination as
a defect to fix, not as a successful test.

- [ ] **Step 5: Validate audio and ASR**

For every response-required WAV, assert 24 kHz, mono, 16-bit, non-zero RMS and
peak. Run Whisper on both sessions and compare non-empty ASR with the protocol
transcript and the session-specific secret/context.

- [ ] **Step 6: Run static checks locally**

```bash
python3.10 -m compileall -q vllm_omni examples/online_serving/minicpmo
rtk proxy git diff --name-only -z --diff-filter=ACMR 5b43b8fa...HEAD -- '*.py' \
  | xargs -0 rtk ruff check
rtk proxy git diff --name-only -z --diff-filter=ACMR 5b43b8fa...HEAD -- '*.py' \
  | xargs -0 rtk ruff format --check
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 7: Stop services and verify GPU cleanup**

Stop only the service and E2E tasks launched by this plan. Verify port 8113 has
no listener and the selected H20 GPUs return to their pre-test process state.

- [ ] **Step 8: Update final evidence and commit fixes**

Record exact remote log paths, test counts, passed pressure scales, WAV
metadata, ASR summaries, and all explicit non-goals in the architecture doc.
Commit only after the final tree has rerun the affected matrix and E2E.
