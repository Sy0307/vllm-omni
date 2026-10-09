# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o closes a cancelled response's turn inside its running Stage-0 request."""

import base64

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as module
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import MiniCPMO45ServingSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PAYLOAD = {
    "audio": base64.b64encode(bytes(16000 * 4)).decode(),
    "format": "pcm_f32le",
    "sample_rate_hz": 16000,
}


async def _plan(plugin, state, *, seq):
    config = plugin.prepare_prompt_config({"instructions": "Be brief."}, state=state, payload=dict(_PAYLOAD))
    plan = await plugin.prepare_append_plan(
        request_id="request",
        fence=DuplexFence("sid"),
        session_config=config,
        runtime_config={"duplex_first_append_context_tokens": 5},
        seq=seq,
        turn_seq=seq,
        payload=dict(_PAYLOAD),
        final=False,
        sampling_params=None,
    )
    return plan.prompt


@pytest.mark.asyncio
async def test_close_model_turn_marks_only_the_next_append():
    plugin = module.MiniCPMO45DuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()
    assert isinstance(state, MiniCPMO45ServingSessionState)
    plain = await _plan(plugin, state, seq=2)
    assert plugin.close_model_turn(state) is True

    closing = await _plan(plugin, state, seq=2)
    after = await _plan(plugin, state, seq=3)

    duplex = closing["model_intermediate_buffer"]["duplex"]
    assert duplex["payload"]["close_turn"] is True
    assert duplex["payload"]["force_listen"] is True
    # The marker never reaches the worker's session config, and the turn closes
    # in place of the unit terminator, so the scheduler reserve is unchanged.
    assert "_minicpmo45_close_turn" not in duplex["session_config"]
    assert len(closing["prompt_token_ids"]) == len(plain["prompt_token_ids"])
    assert "close_turn" not in after["model_intermediate_buffer"]["duplex"]["payload"]
    assert "force_listen" not in after["model_intermediate_buffer"]["duplex"]["payload"]


@pytest.mark.asyncio
async def test_first_append_of_a_request_has_no_turn_to_close():
    plugin = module.MiniCPMO45DuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()
    plugin.close_model_turn(state)
    first = await _plan(plugin, state, seq=1)
    assert "close_turn" not in first["model_intermediate_buffer"]["duplex"]["payload"]
    assert state.close_turn_pending is False
