# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Gander history through the engine session and ordinary stage port."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness
from vllm_omni.engine.duplex.commands import SignalTurn
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.context_history import DuplexContextHistory
from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import GanderContextPolicy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def initialized():
    h = await open_harness()
    h.session.replace_runtime_config(
        {**h.session.runtime_config, "gander_enabled": True, "gander_context_version": 0, "duplex_context_version": 0}
    )
    h.runner.ctx.history = DuplexContextHistory(
        h.runner.ctx,
        GanderContextPolicy(),
        out=h.runner.out,
        model=h.runner.model,
        max_tokens=40960,
        wait_for_append_tail=h.runner._wait_for_append_tail,
        close_from_runtime=h.runner._close_from_runtime,
    )
    await h.run(append_audio())
    complete(h, seq=1)
    await h.settle()
    return h


def complete(h, *, seq):
    rid = h.stage0_request_id()
    output = SimpleNamespace(
        request_id=rid,
        finished=True,
        outputs=[SimpleNamespace(token_ids=[7], stop_reason=7)],
        multimodal_output={"special_token_ids": {"listen_token_id": 7, "gander_append_seq": seq}},
    )
    h.deliver(output, stage_id=0, segment_finished=True, segment_token_ids=(7,))


def edit(h, edits, event_id="edit"):
    return SignalTurn(
        event="input.context.replace",
        signal_payload={
            "kind": "history_edit",
            "event_id": event_id,
            "epoch": h.session.epoch,
            "base_version": h.session.runtime_config["duplex_context_version"],
            "edits": edits,
        },
    )


@pytest.mark.asyncio
async def test_invalid_edit_preserves_request_and_accepts_next_audio():
    h = await initialized()
    try:
        ids = h.session.resource_request_ids()
        before = h.runner.ctx.history.snapshot()
        await h.run(edit(h, [{"op": "delete", "unit_id": "missing"}]))
        assert h.runner.ctx.history.snapshot() == before
        assert h.session.resource_request_ids() == ids
        assert not h.port.cleanups
        await h.run(append_audio())
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
        await h.runner.ctx.history.wait_applied()
        assert len(h.runner.ctx.history.prompts) == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_replacement_waits_for_model_completion_and_suppresses_replay():
    h = await initialized()
    task = None
    try:
        key = h.runner.ctx.history.snapshot()["units"][0]["unit_id"]
        task = asyncio.create_task(h.runner._on_command(edit(h, [{"op": "pin", "unit_id": key}])))
        for _ in range(100):
            if len(h.port.submissions) == 2:
                break
            await asyncio.sleep(0.005)
        assert len(h.port.submissions) == 2
        assert not task.done(), "submission is not a reconstruction completion receipt"
        assert h.session.epoch == 1
        assert not h.port.submissions[-1].already_submitted
        assert h.port.cleanups and h.port.cleanups[-1][1]
        before = list(h.runner.ctx.history.prompts[0]["model_intermediate_buffer"]["duplex"]["gander_output_ids"])
        complete(h, seq=1)
        await asyncio.wait_for(task, 2)
        await h.settle()
        assert h.runner.ctx.history.snapshot()["units"][0]["pinned"]
        assert h.runner.ctx.history.prompts[0]["model_intermediate_buffer"]["duplex"]["gander_output_ids"] == before
        assert any(e.to_realtime().get("type") == "duplex.input.context.replaced" for e in h.events)
        await h.run(append_audio())
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


def test_context_wire_command_uses_engine_command_parser():
    command = translate_realtime_command({"type": "input.context.replace", "context": {"event_id": "x"}})
    assert isinstance(command, SignalTurn)
    assert command.event == "input.context.replace"
    assert command.signal_payload["event_id"] == "x"


@pytest.mark.asyncio
async def test_automatic_rollover_submits_next_audio_in_new_epoch():
    h = await initialized()
    task = None
    try:
        h.session.replace_runtime_config(
            {**h.session.runtime_config, "gander_history": {"max_units": 2, "retain_units": 1}}
        )
        await h.run(append_audio())
        complete(h, seq=2)
        task = asyncio.create_task(
            h.runner.model.append_runtime_input(
                {"audio": "", "sample_rate": 16000},
                final=False,
                expected_epoch=h.session.epoch,
            )
        )
        for _ in range(100):
            if len(h.port.submissions) >= 3:
                break
            await asyncio.sleep(0.005)
        assert h.session.epoch == 1
        complete(h, seq=1)
        ok, _ = await asyncio.wait_for(task, 2)
        assert ok
        assert h.port.submissions[-1].context.fence.epoch == 1
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
        await h.runner.ctx.history.wait_applied()
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_function_result_acknowledged_only_after_model_application(monkeypatch):
    h = await initialized()
    task = None
    try:

        def prepare(item, runtime, *, epoch):
            return {**runtime, "duplex_context_version": 1}, {"gander_control": True, "token_ids": [42]}

        monkeypatch.setattr(h.runner.ctx.history.policy, "prepare_input", prepare)
        item = {"type": "function_call_output", "id": "result", "call_id": "call", "output": "ok"}
        task = asyncio.create_task(h.runner.control._on_conversation_item_create({"payload": {"item": item}}))
        for _ in range(100):
            if len(h.port.submissions) == 2:
                break
            await asyncio.sleep(0.005)
        assert not task.done()
        assert not any(e.to_realtime().get("item", {}).get("id") == "result" for e in h.events)
        complete(h, seq=2)
        await asyncio.wait_for(task, 2)
        await h.settle()
        assert any(
            e.to_realtime().get("type") == "conversation.item.created"
            and e.to_realtime().get("item", {}).get("id") == "result"
            for e in h.events
        )
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)
