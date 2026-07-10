# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib
from collections.abc import AsyncIterator

import pytest

from vllm_omni.experimental.fullduplex.core import runtime as runtime_module
from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    DomainEvent,
    EmitProtocolEvent,
    EngineAppendAccepted,
    InputChunk,
    InputCommitted,
    InterruptRequested,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
    ProtocolEventKind,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
    SessionCloseRequested,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.joyvl.adapter import JoyVLDuplexAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Sink:
    def __init__(self) -> None:
        self.events: list[EmitProtocolEvent] = []
        self._seen: dict[ProtocolEventKind, asyncio.Event] = {}

    async def emit(self, event: EmitProtocolEvent) -> None:
        self.events.append(event)
        self._seen.setdefault(event.kind, asyncio.Event()).set()

    async def wait_for(self, kind: ProtocolEventKind) -> None:
        await asyncio.wait_for(
            self._seen.setdefault(kind, asyncio.Event()).wait(),
            timeout=1,
        )


class _BlockedSink(_Sink):
    def __init__(self) -> None:
        super().__init__()
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event: EmitProtocolEvent) -> None:
        if event.kind is ProtocolEventKind.RESPONSE_STARTED:
            self.blocked.set()
            await self.release.wait()
        await super().emit(event)


class _FailingSink(_Sink):
    async def emit(self, event: EmitProtocolEvent) -> None:
        raise RuntimeError("sink exploded")


class _Engine:
    def __init__(self) -> None:
        self.commands: list[object] = []
        self._events: asyncio.Queue[DomainEvent | None] = asyncio.Queue()

    async def reserve(self, command: ReserveResponse) -> None:
        self.commands.append(command)

    async def append(self, command: AppendToEngine) -> None:
        self.commands.append(command)
        if command.final:
            await self._events.put(EngineAppendAccepted(fence=command.fence))

    async def cancel(self, fence: DuplexFence) -> None:
        self.commands.append(("cancel", fence))

    async def reset(self, command: ResetStage1) -> None:
        self.commands.append(command)

    async def rebuild(self, command: RebuildStage0Context) -> None:
        self.commands.append(command)

    async def close(self, fence: DuplexFence) -> None:
        self.commands.append(("close", fence))
        await self._events.put(None)

    async def emit(self, event: DomainEvent) -> None:
        await self._events.put(event)

    async def events(self) -> AsyncIterator[DomainEvent]:
        while (event := await self._events.get()) is not None:
            yield event


async def _single_turn(runtime: DuplexRuntime, engine: _Engine, sink: _Sink):
    yield InputCommitted()
    while runtime.state.turn_phase.value == "turn_committed":
        await asyncio.sleep(0)
    fence = runtime.state.fence
    await engine.emit(ModelSpeaking(fence=fence))
    await engine.emit(ModelTextDelta(text="hello", fence=fence))
    await engine.emit(ModelTurnEnded(fence=fence))
    await sink.wait_for(ProtocolEventKind.RESPONSE_COMPLETED)
    yield SessionCloseRequested()


@pytest.mark.asyncio
async def test_runtime_executes_every_engine_effect_with_the_reducer_fence():
    engine = _Engine()
    sink = _Sink()
    runtime = DuplexRuntime("s", engine, sink)

    await runtime.run(_single_turn(runtime, engine, sink))

    turn_fence = DuplexFence("s", turn_id=1, response_seq=1)
    assert engine.commands == [
        ReserveResponse(turn_fence),
        AppendToEngine(turn_fence, final=True),
        ResetStage1(turn_fence),
        ("close", turn_fence),
    ]
    assert [event.kind for event in sink.events] == [
        ProtocolEventKind.RESPONSE_STARTED,
        ProtocolEventKind.TEXT_DELTA,
        ProtocolEventKind.RESPONSE_COMPLETED,
    ]


class _BlockedOutputEngine(_Engine):
    def __init__(self) -> None:
        super().__init__()
        self.release_old_output = asyncio.Event()

    async def events(self) -> AsyncIterator[DomainEvent]:
        accepted = await self._events.get()
        assert isinstance(accepted, EngineAppendAccepted)
        yield accepted
        yield ModelSpeaking(fence=accepted.fence)
        await self.release_old_output.wait()
        yield ModelTextDelta(text="stale", fence=accepted.fence)
        yield ModelTurnEnded(fence=accepted.fence)
        while (event := await self._events.get()) is not None:
            yield event


@pytest.mark.asyncio
async def test_input_remains_responsive_and_stale_output_is_dropped():
    engine = _BlockedOutputEngine()
    sink = _Sink()
    runtime = DuplexRuntime("responsive", engine, sink)

    async def inputs():
        yield InputCommitted()
        await sink.wait_for(ProtocolEventKind.RESPONSE_STARTED)
        old_fence = runtime.state.fence
        yield InterruptRequested(reason="barge-in", fence=old_fence)
        while ("cancel", old_fence) not in engine.commands:
            await asyncio.sleep(0)
        engine.release_old_output.set()
        await asyncio.sleep(0)
        yield SessionCloseRequested()

    await asyncio.wait_for(runtime.run(inputs()), timeout=1)

    assert ProtocolEventKind.TEXT_DELTA not in [event.kind for event in sink.events]
    assert runtime.state.stale_event_count == 2
    old_fence = DuplexFence("responsive", turn_id=1, response_seq=1)
    assert ("cancel", old_fence) in engine.commands
    assert ResetStage1(old_fence) in engine.commands
    assert (
        RebuildStage0Context(
            old_fence.next_epoch(),
            committed_history=(),
            committed_playback_position=0,
        )
        in engine.commands
    )
    assert ("close", old_fence.next_epoch()) in engine.commands


@pytest.mark.asyncio
async def test_blocked_sink_does_not_block_interrupt_or_close_state_transitions():
    engine = _Engine()
    sink = _BlockedSink()
    runtime = DuplexRuntime(
        "blocked-sink",
        engine,
        sink,
        shutdown_timeout=0.02,
    )
    owned_tasks: list[asyncio.Task] = []
    create_resource_task = runtime._create_resource_task

    def capture_resource_task(coroutine):
        task = create_resource_task(coroutine)
        owned_tasks.append(task)
        return task

    runtime._create_resource_task = capture_resource_task

    async def inputs():
        yield InputCommitted()
        while runtime.state.turn_phase.value == "turn_committed":
            await asyncio.sleep(0)
        fence = runtime.state.fence
        await engine.emit(ModelSpeaking(fence=fence))
        await sink.blocked.wait()
        yield InterruptRequested(reason="blocked sink", fence=fence)
        yield SessionCloseRequested()

    run_task = asyncio.create_task(runtime.run(inputs()))

    async def wait_until_closed():
        while runtime.state.session_phase.value != "closed":
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_closed(), timeout=0.2)
    assert runtime.state.fence.epoch == 1
    assert not run_task.done()
    await asyncio.wait_for(run_task, timeout=0.5)

    closed_fence = DuplexFence(
        "blocked-sink",
        epoch=1,
        turn_id=1,
        response_seq=1,
    )
    assert ("close", closed_fence) in engine.commands
    assert owned_tasks
    assert all(task.done() for task in owned_tasks)
    assert not runtime._resource_tasks


@pytest.mark.asyncio
async def test_external_runtime_cancellation_still_propagates():
    engine = _Engine()
    sink = _Sink()
    runtime = DuplexRuntime("external-cancel", engine, sink)
    input_started = asyncio.Event()
    input_closed = asyncio.Event()

    async def inputs():
        try:
            input_started.set()
            yield InputChunk(data="waiting", modality="text")
            await asyncio.Event().wait()
        finally:
            input_closed.set()

    run_task = asyncio.create_task(runtime.run(inputs()))
    await input_started.wait()
    await asyncio.sleep(0)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert input_closed.is_set()
    assert not runtime._resource_tasks


@pytest.mark.asyncio
async def test_cancellation_resistant_owned_task_surfaces_shutdown_error():
    engine = _FailingAppendEngine()
    sink = _Sink()
    runtime = DuplexRuntime(
        "resistant-input",
        engine,
        sink,
        shutdown_timeout=0.01,
    )
    release_input = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def inputs():
        yield InputCommitted()
        while not release_input.is_set():
            try:
                await release_input.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()

    with pytest.raises(runtime_module.DuplexShutdownError) as exc_info:
        await asyncio.wait_for(runtime.run(inputs()), timeout=0.5)

    assert cancellation_seen.is_set()
    assert "duplex-input" in str(exc_info.value)
    assert runtime._resource_tasks
    assert any(not task.done() for task in runtime._resource_tasks)

    release_input.set()
    await asyncio.gather(*runtime._resource_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert not runtime._resource_tasks


class _FailingAppendEngine(_Engine):
    async def append(self, command: AppendToEngine) -> None:
        self.commands.append(command)
        raise RuntimeError("append exploded")


@pytest.mark.asyncio
async def test_append_failure_closes_once_and_attempts_all_cleanup():
    engine = _FailingAppendEngine()
    sink = _Sink()
    runtime = DuplexRuntime("append-failure", engine, sink)

    async def inputs():
        yield InputCommitted()
        while runtime.state.session_phase.value != "closed":
            await asyncio.sleep(0)

    await asyncio.wait_for(runtime.run(inputs()), timeout=1)

    fence = DuplexFence("append-failure", turn_id=1, response_seq=1)
    assert runtime.state.terminal_reason == "append exploded"
    assert [event.kind for event in sink.events].count(ProtocolEventKind.ENGINE_FAILED) == 1
    assert ("cancel", fence) in engine.commands
    assert ResetStage1(fence) in engine.commands
    assert ("close", fence) in engine.commands


@pytest.mark.asyncio
async def test_internal_failure_cancels_blocked_input_and_wakes_run():
    engine = _FailingAppendEngine()
    sink = _Sink()
    runtime = DuplexRuntime(
        "blocked-input",
        engine,
        sink,
        shutdown_timeout=0.02,
    )
    input_closed = asyncio.Event()
    block_input = asyncio.Event()
    owned_tasks: list[asyncio.Task] = []
    create_resource_task = runtime._create_resource_task

    def capture_resource_task(coroutine):
        task = create_resource_task(coroutine)
        owned_tasks.append(task)
        return task

    runtime._create_resource_task = capture_resource_task

    async def inputs():
        try:
            yield InputCommitted()
            await block_input.wait()
        finally:
            input_closed.set()

    await asyncio.wait_for(runtime.run(inputs()), timeout=0.5)

    fence = DuplexFence("blocked-input", turn_id=1, response_seq=1)
    assert input_closed.is_set()
    assert runtime.state.terminal_reason == "append exploded"
    assert ("close", fence) in engine.commands
    assert owned_tasks
    assert all(task.done() for task in owned_tasks)
    assert not runtime._resource_tasks


@pytest.mark.asyncio
async def test_sink_failure_closes_once_and_does_not_skip_engine_cleanup():
    engine = _Engine()
    sink = _FailingSink()
    runtime = DuplexRuntime("sink-failure", engine, sink)

    async def inputs():
        yield InputCommitted()
        while runtime.state.turn_phase.value == "turn_committed":
            await asyncio.sleep(0)
        await engine.emit(ModelSpeaking(fence=runtime.state.fence))
        while runtime.state.session_phase.value != "closed":
            await asyncio.sleep(0)

    await asyncio.wait_for(runtime.run(inputs()), timeout=1)

    fence = DuplexFence("sink-failure", turn_id=1, response_seq=1)
    assert runtime.state.terminal_reason == "sink exploded"
    assert ("cancel", fence) in engine.commands
    assert ResetStage1(fence) in engine.commands
    assert ("close", fence) in engine.commands
    assert not runtime._resource_tasks


@pytest.mark.asyncio
async def test_runtime_emits_response_terminal_exactly_once():
    engine = _Engine()
    sink = _Sink()
    runtime = DuplexRuntime("terminal", engine, sink)
    duplicate_consumed = asyncio.Event()

    async def inputs():
        yield InputCommitted()
        while runtime.state.turn_phase.value == "turn_committed":
            await asyncio.sleep(0)
        fence = runtime.state.fence
        await engine.emit(ModelSpeaking(fence=fence))
        await engine.emit(ModelTurnEnded(fence=fence))
        await engine.emit(ModelTurnEnded(fence=fence))
        await sink.wait_for(ProtocolEventKind.RESPONSE_COMPLETED)
        while runtime.state.duplicate_terminal_count == 0:
            await asyncio.sleep(0)
        duplicate_consumed.set()
        yield SessionCloseRequested()

    await runtime.run(inputs())

    assert duplicate_consumed.is_set()
    assert [event.kind for event in sink.events].count(ProtocolEventKind.RESPONSE_COMPLETED) == 1
    assert runtime.state.duplicate_terminal_count == 1


class _FailingEngine(_Engine):
    async def events(self) -> AsyncIterator[DomainEvent]:
        accepted = await self._events.get()
        assert isinstance(accepted, EngineAppendAccepted)
        yield accepted
        yield ModelSpeaking(fence=accepted.fence)
        raise RuntimeError("engine exploded")
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_engine_failure_closes_response_and_session_resources():
    engine = _FailingEngine()
    sink = _Sink()
    runtime = DuplexRuntime("failure", engine, sink)

    async def inputs():
        yield InputCommitted()
        await sink.wait_for(ProtocolEventKind.ENGINE_FAILED)

    await asyncio.wait_for(runtime.run(inputs()), timeout=1)

    fence = DuplexFence("failure", turn_id=1, response_seq=1)
    assert runtime.state.terminal_reason == "engine exploded"
    assert [event.kind for event in sink.events][-1] is ProtocolEventKind.ENGINE_FAILED
    assert ("cancel", fence) in engine.commands
    assert ResetStage1(fence) in engine.commands
    assert ("close", fence) in engine.commands


@pytest.mark.asyncio
async def test_joyvl_adapter_emits_typed_model_events():
    replies = iter(["</response> a fire is breaking out"])

    async def fake_generate(messages):
        return next(replies)

    engine = JoyVLDuplexAdapter(fake_generate, num_frames=4)
    sink = _Sink()
    runtime = DuplexRuntime("joyvl", engine, sink)

    async def inputs():
        yield InputChunk("alert me if a fire breaks out", modality="text")
        yield InputChunk("data:image/jpeg;base64,AAA", modality="video")
        yield InputChunk("data:image/jpeg;base64,BBB", modality="video")
        yield InputCommitted()
        await sink.wait_for(ProtocolEventKind.RESPONSE_COMPLETED)
        yield SessionCloseRequested()

    await runtime.run(inputs())

    deltas = [event.payload.text for event in sink.events if event.kind is ProtocolEventKind.TEXT_DELTA]
    assert deltas == ["a fire is breaking out"]
    assert all(not isinstance(event, dict) for event in sink.events)


def test_public_exports_do_not_expose_legacy_session_state_machine():
    import vllm_omni.experimental.fullduplex as fullduplex

    assert not hasattr(fullduplex, "DuplexSession")
    assert not hasattr(fullduplex, "DuplexSessionConfig")
    assert not hasattr(fullduplex, "DuplexState")


def test_legacy_adapter_api_is_not_importable_or_exported():
    import vllm_omni.experimental.fullduplex as fullduplex

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("vllm_omni.experimental.fullduplex.core.adapter")
    assert not hasattr(fullduplex, "DuplexAdapter")
    assert not hasattr(fullduplex, "DuplexCapability")
    assert not hasattr(fullduplex, "OutputChunk")
