"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import asyncio
import inspect
import queue
import threading
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import DuplexControlResultMessage, ErrorMessage, OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.engine.omni import DuplexOutputFenceError, OmniDuplexEnginePort
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, mocker: MockerFixture, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = mocker.MagicMock(
        is_alive=mocker.MagicMock(return_value=thread_alive),
    )
    engine._ensure_output_router()
    return engine


def test_try_get_output_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_queue = mocker.MagicMock()
    # First call succeeds; second call finds the queue empty.
    mock_queue.sync_q.get.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

    # Collect the one buffered result.
    assert engine.try_get_output().request_id == "r1"

    # Orchestrator thread crashes between polls.
    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Same scenario as above but for the async variant."""
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        )
    )

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=True)

    assert (await engine.try_get_output_async()).request_id == "r1"

    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()


def test_fatal_error_message_surfaces_through_try_get_output(mocker: MockerFixture):
    """When the orchestrator thread crashes, it enqueues a fatal error message.

    ``try_get_output`` must return this message so the caller
    (``OmniBase._handle_output_message``) can detect the fatal flag.
    """
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = engine.try_get_output()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
    assert "crashed" in msg.error


def test_fenced_duplex_methods_require_explicit_fence() -> None:
    for method_name in (
        "open_duplex_session_fenced_async",
        "append_duplex_input_fenced_async",
        "signal_duplex_turn_fenced_async",
        "close_duplex_session_fenced_async",
    ):
        signature = inspect.signature(getattr(AsyncOmniEngine, method_name))
        assert signature.parameters["fence"].default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_fenced_append_registers_only_transport_routing_metadata() -> None:
    engine = object.__new__(AsyncOmniEngine)
    captured = []

    async def control(msg, *, timeout):
        captured.append((msg, timeout))
        return {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-request",
                    }
                }
            ]
        }

    engine._duplex_control_async = control
    fence = DuplexFence("sid", epoch=2, turn_id=3, response_seq=4)

    await engine.append_duplex_input_fenced_async(
        fence,
        mode="append_audio_chunk",
        payload={},
    )

    assert captured[0][0].fence is fence
    assert engine._typed_duplex_sessions == {"sid"}
    assert engine._duplex_request_sessions == {"duplex-request": "sid"}


@pytest.mark.asyncio
async def test_fatal_error_message_surfaces_through_try_get_output_async(mocker: MockerFixture):
    """Async variant of the fatal error message test."""
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    raw_queue = queue.Queue()
    raw_queue.put_nowait(fatal_msg)

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True


@pytest.mark.asyncio
async def test_sync_generic_reader_does_not_steal_missing_fence_duplex_output(mocker: MockerFixture):
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="duplex-request",
            fence=None,
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="duplex-request"),
            finished=False,
        )
    )
    raw_queue.put_nowait(
        OutputMessage(
            request_id="generic-request",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="generic-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)
    engine._typed_duplex_sessions = {"sid"}
    engine._duplex_request_sessions = {"duplex-request": "sid"}

    generic = engine.try_get_output(timeout=0.01)
    duplex = await engine.get_duplex_output_async("sid")

    assert generic.request_id == "generic-request"
    assert duplex.request_id == "duplex-request"
    assert duplex.fence is None


@pytest.mark.asyncio
async def test_missing_fence_duplex_output_reaches_port_error_path(mocker: MockerFixture):
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="duplex-request",
            fence=None,
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="duplex-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)
    engine._typed_duplex_sessions = {"sid"}
    engine._duplex_request_sessions = {"duplex-request": "sid"}
    port = OmniDuplexEnginePort(engine)
    port._session_id = "sid"
    port._request_ids = {"duplex-request"}

    with pytest.raises(DuplexOutputFenceError, match="missing a DuplexFence"):
        await asyncio.wait_for(anext(port.events()), timeout=0.1)


@pytest.mark.asyncio
async def test_duplex_reader_does_not_steal_generic_output_from_async_reader(mocker: MockerFixture):
    fence = DuplexFence("sid", turn_id=1, response_seq=1)
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="generic-request",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="generic-request"),
            finished=False,
        )
    )
    raw_queue.put_nowait(
        OutputMessage(
            request_id="duplex-request",
            fence=fence,
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="duplex-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)
    engine._typed_duplex_sessions = {"sid"}
    engine._duplex_request_sessions = {"duplex-request": "sid"}

    duplex = await engine.get_duplex_output_async("sid")
    generic = await engine.try_get_output_async()

    assert duplex.request_id == "duplex-request"
    assert duplex.fence == fence
    assert generic.request_id == "generic-request"


def test_legacy_fenced_output_remains_on_generic_compatibility_path(mocker: MockerFixture):
    fence = DuplexFence("legacy-session")
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="legacy-duplex-request",
            fence=fence,
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="legacy-duplex-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)

    output = engine.try_get_output(timeout=0.01)

    assert output.request_id == "legacy-duplex-request"
    assert engine._duplex_output_queues == {}


def test_open_duplex_session_waits_for_control_ack(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    rpc_q.put_nowait(
        DuplexControlResultMessage(
            control_id="ctrl-1",
            fence=DuplexFence("sid"),
            operation="open",
            session_id="sid",
            ok=False,
            stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False}}],
            unsupported_count=1,
            error_count=0,
        )
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._rpc_lock = threading.Lock()
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-1"))

    result = engine.open_duplex_session("sid", timeout=1)

    msg = request_q.get_nowait()
    assert msg.type == "open_duplex_session"
    assert msg.control_id == "ctrl-1"
    assert msg.timeout == 1
    assert result["unsupported_count"] == 1
    assert result["stage_results"][0]["result"]["supported"] is False


def _duplex_streaming_req_state(*, segment_finished: bool = True):
    req_state = OrchestratorRequestState(request_id="req", final_stage_id=1)
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = segment_finished
    req_state.streaming.bridge_states["duplex"] = {"session_id": "sid"}
    return req_state


def test_duplex_model_listen_segment_uses_raw_streaming_new_token_ids():
    req_state = _duplex_streaming_req_state()

    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace()],
        new_token_ids=[151705],
    )

    assert Orchestrator._is_duplex_model_listen_segment(0, output, req_state)


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_duplex_model_listen_segment_does_not_use_output_level_history(attr):
    req_state = _duplex_streaming_req_state()

    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace()],
        **{attr: [42, 151705]},
    )

    assert not Orchestrator._is_duplex_model_listen_segment(0, output, req_state)


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_duplex_model_listen_segment_uses_completion_token_ids(attr):
    req_state = _duplex_streaming_req_state()

    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(**{attr: [42, 151705]})],
    )

    assert Orchestrator._is_duplex_model_listen_segment(0, output, req_state)


def test_duplex_model_listen_segment_uses_completion_stop_reason():
    req_state = _duplex_streaming_req_state()

    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(stop_reason=151705)],
    )

    assert Orchestrator._is_duplex_model_listen_segment(0, output, req_state)


def test_open_duplex_session_raises_on_stage_control_error(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    rpc_q.put_nowait(
        DuplexControlResultMessage(
            control_id="ctrl-error",
            fence=DuplexFence("sid"),
            operation="open",
            session_id="sid",
            ok=False,
            stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False, "error": "boom"}}],
            unsupported_count=1,
            error_count=1,
        )
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._rpc_lock = threading.Lock()
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-error"))

    with pytest.raises(RuntimeError, match="duplex open failed"):
        engine.open_duplex_session("sid", timeout=1)
