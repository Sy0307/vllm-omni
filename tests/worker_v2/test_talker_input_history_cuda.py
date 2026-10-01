# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact conditioned-input history across buffer and request-slot reuse."""

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def _batch(ids, slots, counts, offsets):
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    positions = np.concatenate([np.arange(offset, offset + count) for offset, count in zip(offsets, counts)])
    batch = InputBatch.__new__(InputBatch)
    batch.req_ids = ids
    batch.idx_mapping_np = np.asarray(slots)
    batch.idx_mapping = torch.tensor(slots, device="cuda", dtype=torch.int32)
    batch.query_start_loc_np = starts
    batch.query_start_loc = torch.tensor(starts, device="cuda")
    batch.num_computed_tokens_np = np.asarray(offsets)
    batch.positions = torch.tensor(positions, device="cuda")
    return batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_history_survives_reordering_preemption_and_slot_overwrite(mocker, dtype):
    owner = mocker.Mock()
    owner.scheduler_config.max_num_seqs = 3
    owner.vllm_config.model_config.max_model_len = 32
    owner._stream_pos = {"b": 7}
    owner._mtp_generators = {}
    owner._eager_ready = {}
    owner._eager_embeds = torch.arange(21, device="cuda").reshape(3, 7).to(dtype)
    owner.intermediate_buffer.buffers = [{} for _ in range(3)]
    owner.model.stream_decoder.save_slot.return_value = [torch.arange(5, device="cuda")]
    state = EagerMTPState(owner)
    first = torch.arange(35, device="cuda").reshape(5, 7).to(dtype)
    second = torch.arange(35, 70, device="cuda").reshape(5, 7).to(dtype)
    expected = {"a": torch.cat([first[:3], second[1:3]]), "b": torch.cat([first[3:4], second[:1]])}
    # Extra rows in the input buffer belong to graph padding and are not history.
    state.record_inputs(_batch(["a", "b"], [0, 1], [3, 1], [0, 0]), first)
    state.record_inputs(_batch(["b", "a"], [1, 0], [1, 2], [1, 3]), second)
    first.fill_(-99)
    second.fill_(-99)
    for rid, wanted in expected.items():
        assert state._talker_inputs[rid].length == len(wanted)
        assert torch.equal(state._talker_inputs[rid].embeds[: len(wanted)], wanted)

    state.suspend_audio("b", 1)
    snapshot = state._talker_inputs["b"].embeds.clone()
    other = torch.full((2, 7), 88, device="cuda", dtype=dtype)
    state.record_inputs(_batch(["c"], [1], [2], [0]), other)
    assert torch.equal(state._talker_inputs["b"].embeds, snapshot)
    state.resume_audio("b", 2)
    state.suspend_audio("b", 2)  # Preempt again before the new slot has been used.
    state.resume_audio("b", 2)
    replay = torch.empty_like(expected["b"])
    assert state.replay_inputs("b", 2, 0, torch.zeros(len(replay), device="cuda"), replay)
    assert torch.equal(replay, expected["b"])
    state.record_inputs(_batch(["b"], [2], [len(replay)], [0]), replay)
    assert torch.equal(state._talker_inputs["b"].embeds[: len(replay)], expected["b"])
    assert torch.equal(state._talker_inputs["c"].embeds[:2], other)
    state.finish_audio({"b", "c"})
    assert not state._suspended_audio and not state._restore_audio
    assert "b" not in state._history_slots and "c" not in state._history_slots
    state.record_inputs(_batch(["b"], [1], [1], [0]), other[:1])
    assert state._talker_inputs["b"].length == 1
    assert torch.equal(state._talker_inputs["b"].embeds[:1], other[:1])
    assert torch.equal(state._talker_inputs["a"].embeds[: len(expected["a"])], expected["a"])


@pytest.mark.parametrize(
    "slots,offsets", [([0, 0], [0, 0]), ([-1, 1], [0, 0]), ([0, 2], [0, 0]), ([0, 1], [-1, 0]), ([0, 1], [0, 8])]
)
def test_history_rejects_invalid_metadata_before_device_write(mocker, slots, offsets):
    owner = mocker.Mock()
    owner.scheduler_config.max_num_seqs = 2
    owner.vllm_config.model_config.max_model_len = 8
    state = EagerMTPState(owner)
    with pytest.raises(ValueError, match="history"):
        state.record_inputs(_batch(["a", "b"], slots, [1, 1], offsets), torch.ones(2, 7, device="cuda"))
    assert state._history_storage is None


@pytest.mark.parametrize("invalid", ["short_embeds", "short_positions", "hidden_width"])
def test_history_rejects_incompatible_input_tensors(mocker, invalid):
    owner = mocker.Mock()
    owner.scheduler_config.max_num_seqs = 2
    owner.vllm_config.model_config.max_model_len = 8
    state = EagerMTPState(owner)
    batch = _batch(["a"], [0], [2], [0])
    embeds = torch.ones(2, 7, device="cuda")
    state.record_inputs(batch, embeds)
    expected = state._talker_inputs["a"].embeds[:2].clone()
    if invalid == "short_embeds":
        embeds = embeds[:1]
    elif invalid == "short_positions":
        batch.positions = batch.positions[:1]
    else:
        embeds = torch.ones(2, 8, device="cuda")
    with pytest.raises(ValueError, match="history"):
        state.record_inputs(batch, embeds)
    assert torch.equal(state._talker_inputs["a"].embeds[:2], expected)
