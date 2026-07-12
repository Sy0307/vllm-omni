# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

if importlib.util.find_spec("torch") is None:
    _ROOT = Path(__file__).resolve().parents[2]
    for _name, _relative_path in (
        ("vllm_omni", "vllm_omni"),
        ("vllm_omni.experimental", "vllm_omni/experimental"),
        ("vllm_omni.experimental.fullduplex", "vllm_omni/experimental/fullduplex"),
    ):
        _module = ModuleType(_name)
        _module.__path__ = [str(_ROOT / _relative_path)]
        sys.modules.setdefault(_name, _module)

from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    EngineAppendAccepted,
    InputChunk,
    ModelSpeaking,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.engine.omni import (
    DuplexInputMode,
    DuplexOutputFenceError,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    OmniDuplexEnginePort,
    build_duplex_data_plane_prompt,
    duplex_resource_request_id,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.outputs: list[object] = []

    async def open_duplex_session_fenced_async(self, fence: DuplexFence, **kwargs):
        self.calls.append(("open", {"fence": fence, **kwargs}))
        return {"stage_results": []}

    async def append_duplex_input_fenced_async(self, fence: DuplexFence, **kwargs):
        self.calls.append(("append", {"fence": fence, **kwargs}))
        return {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": duplex_resource_request_id(fence, "stage0"),
                        "response_stage_id": 1,
                    }
                }
            ]
        }

    async def signal_duplex_turn_fenced_async(self, fence: DuplexFence, **kwargs):
        self.calls.append(("signal", {"fence": fence, **kwargs}))
        return {"stage_results": []}

    async def close_duplex_session_fenced_async(self, fence: DuplexFence, **kwargs):
        self.calls.append(("close", {"fence": fence, **kwargs}))
        return {"stage_results": []}

    async def open_duplex_session_async(self, *args, **kwargs):
        raise AssertionError("typed port called legacy open wrapper")

    async def append_duplex_input_async(self, *args, **kwargs):
        raise AssertionError("typed port called legacy append wrapper")

    async def signal_duplex_turn_async(self, *args, **kwargs):
        raise AssertionError("typed port called legacy signal wrapper")

    async def close_duplex_session_async(self, *args, **kwargs):
        raise AssertionError("typed port called legacy close wrapper")

    async def try_get_output_async(self):
        raise AssertionError("duplex port must not consume the shared output queue")

    async def get_duplex_output_async(self, session_id: str):
        assert session_id
        return self.outputs.pop(0)


async def _first(iterator: AsyncIterator[object]) -> object:
    return await anext(iterator)


@pytest.mark.asyncio
async def test_port_preserves_complete_fence_for_open_append_and_output() -> None:
    engine = _Engine()
    port = OmniDuplexEnginePort(engine)
    fence = DuplexFence("sid", epoch=2, turn_id=3, response_seq=4)

    await port.reserve(ReserveResponse(fence))
    await port.append(AppendToEngine(fence, InputChunk(b"\x00\x00\x00\x00"), final=True))
    engine.outputs.append(
        SimpleNamespace(
            request_id=duplex_resource_request_id(fence, "stage0"),
            fence=fence,
            engine_outputs=EngineAppendAccepted(fence=fence),
        )
    )

    assert await _first(port.events()) == EngineAppendAccepted(fence=fence)
    assert engine.calls[0][0] == "open"
    assert engine.calls[0][1] == {
        "fence": fence,
        "session_mode": "duplex",
        "capabilities": port._capabilities.as_dict(),
        "session_config": {},
        "timeout": 10.0,
    }
    assert engine.calls[1][0] == "append"
    assert engine.calls[1][1]["fence"] is fence
    assert engine.calls[1][1]["expected_epoch"] == fence.epoch


@pytest.mark.asyncio
async def test_port_preserves_complete_fence_for_resource_controls() -> None:
    engine = _Engine()
    port = OmniDuplexEnginePort(engine)
    fence = DuplexFence("sid", epoch=5, turn_id=8, response_seq=13)
    next_fence = DuplexFence("sid", epoch=6, turn_id=8, response_seq=13)
    rebuild = RebuildStage0Context(next_fence, (), 21)

    await port.cancel(fence)
    await port.reset(ResetStage1(fence))
    await port.rebuild(rebuild)
    await port.close(next_fence)

    assert [name for name, _ in engine.calls] == ["signal", "signal", "signal", "close"]
    assert [kwargs["fence"] for _, kwargs in engine.calls] == [fence, fence, next_fence, next_fence]
    assert [engine.calls[index][1]["event"] for index in range(3)] == [
        "response.cancel",
        "reset_stage1",
        "rebuild_stage0",
    ]


@pytest.mark.asyncio
async def test_port_rejects_duplex_output_without_complete_fence() -> None:
    engine = _Engine()
    port = OmniDuplexEnginePort(engine)
    fence = DuplexFence("sid", turn_id=1, response_seq=1)
    await port.append(AppendToEngine(fence, final=True))
    engine.outputs.append(
        SimpleNamespace(
            request_id=duplex_resource_request_id(fence, "stage0"),
            fence=None,
            engine_outputs=ModelSpeaking(fence=None),
        )
    )

    with pytest.raises(DuplexOutputFenceError, match="missing a DuplexFence"):
        await _first(port.events())


@pytest.mark.asyncio
async def test_request_id_routes_outputs_without_becoming_identity() -> None:
    engine = _Engine()
    port = OmniDuplexEnginePort(engine)
    first = DuplexFence("sid", turn_id=1, response_seq=1)
    second = DuplexFence("sid", turn_id=2, response_seq=2)
    await port.append(AppendToEngine(first, final=True))
    await port.append(AppendToEngine(second, final=True))
    engine.outputs.append(
        SimpleNamespace(
            request_id=duplex_resource_request_id(first, "stage0"),
            fence=first,
            engine_outputs=EngineAppendAccepted(fence=first),
        )
    )

    assert await _first(port.events()) == EngineAppendAccepted(fence=first)


@pytest.mark.asyncio
async def test_typed_port_has_no_legacy_identity_fallback() -> None:
    class _LegacyOnlyEngine:
        async def open_duplex_session_async(self, session_id: str, **kwargs):
            raise AssertionError("legacy identity synthesis must be unreachable")

    port = OmniDuplexEnginePort(_LegacyOnlyEngine())

    with pytest.raises(AttributeError, match="open_duplex_session_fenced_async"):
        await port.reserve(ReserveResponse(DuplexFence("sid")))


def test_resource_state_rejects_fence_regression_and_requires_explicit_fence() -> None:
    current = DuplexFence("sid", epoch=2, turn_id=3, response_seq=4)
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        current,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )

    for stale in (
        DuplexFence("sid", epoch=1, turn_id=99, response_seq=99),
        DuplexFence("sid", epoch=2, turn_id=2, response_seq=4),
        DuplexFence("sid", epoch=2, turn_id=3, response_seq=3),
    ):
        with pytest.raises(RuntimeError, match="fence mismatch"):
            session.accept_fence(stale)
        assert session.fence == current

    with pytest.raises(TypeError, match="fence"):
        session.bind_stage_request(0, "request")
    with pytest.raises(TypeError, match="fence"):
        session.append_input({}, mode=DuplexInputMode.APPEND_AUDIO_CHUNK)
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.open_session("legacy-session")
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.close_session("legacy-session")
    assert not hasattr(manager, "open_session_legacy")
    assert not hasattr(manager, "close_session_legacy")


def test_resource_request_id_is_derived_from_fence_and_role() -> None:
    fence = DuplexFence("sid-with-dashes", epoch=7, turn_id=11, response_seq=13)

    assert duplex_resource_request_id(fence, "stage0") == "duplex-sid-with-dashes-e7-stage0"
    assert duplex_resource_request_id(fence, "stage1") == "duplex-sid-with-dashes-e7-stage1"


def test_placeholder_budget_is_planned_inside_omni_engine_boundary() -> None:
    fence = DuplexFence("sid", turn_id=1, response_seq=1)
    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        seq=2,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={
            "audio": "AAAAAA==",
            "format": "pcm_f32le",
            "duplex_num_input_tokens": 999,
            "num_input_tokens": 999,
        },
        final=False,
    )

    assert len(prompt["prompt_token_ids"]) == 16
    assert prompt["model_intermediate_buffer"]["duplex"]["fence"] == fence
    assert prompt["model_intermediate_buffer"]["duplex"]["scheduler_token_budget"] == 16


def test_engine_port_import_is_canonical() -> None:
    from vllm_omni.experimental.fullduplex.engine.omni import OmniDuplexEnginePort as ImportedPort

    assert ImportedPort is OmniDuplexEnginePort
