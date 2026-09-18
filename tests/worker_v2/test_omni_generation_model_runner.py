# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniGenerationModelRunner.sample_tokens (V2).

Covers the core multimodal_outputs construction paths via _build_pooler_output:
  - OmniOutput with batched tensor multimodal_outputs → per-request slicing
  - OmniOutput with list multimodal_outputs → direct mapping (including None)
  - OmniOutput with dict scalar values → broadcast to all requests
  - None model output → returns None
  - Non-dict multimodal_outputs → [{}] * num_reqs
  - sampled_token_ids always emits empty lists per request (no token sampling)
  - req_states.num_computed_tokens updated to prompt_len after sample_tokens
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_control_only_step_keeps_lifecycle_and_skips_input_construction():
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    runner = object.__new__(OmniGenerationModelRunner)
    order = []
    for name in (
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "_apply_block_table_staged_writes_if_available",
    ):
        setattr(runner, name, lambda *args, name=name: order.append(name))
    for name in ("_handle_async_chunk_updates", "add_requests", "update_requests", "_sync_native_data_plane_payloads"):
        setattr(runner, name, MagicMock(side_effect=AssertionError("control step built model inputs")))
    output = object()
    runner.kv_connector = SimpleNamespace(no_forward=lambda _: output)
    runner._merge_ec_connector_no_forward = lambda _scheduler, value: value
    runner._attach_native_data_plane_signals = lambda value: value
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0, scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[])
    )
    assert runner.execute_model(scheduler_output) is output
    assert order == [
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "_apply_block_table_staged_writes_if_available",
    ]


def test_execute_model_propagates_make_omni_output_failure(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    runner = object.__new__(generation_runner.OmniGenerationModelRunner)
    runner._prepare_native_data_plane = MagicMock()
    runner.finish_requests = MagicMock()
    runner.free_states = MagicMock()
    runner._handle_async_chunk_updates = MagicMock()
    runner.add_requests = MagicMock()
    runner.update_requests = MagicMock()
    runner._sync_native_data_plane_payloads = MagicMock()
    runner._apply_block_table_staged_writes_if_available = MagicMock()
    runner._dispatch_batch_descriptor = MagicMock(
        return_value=(
            SimpleNamespace(
                num_tokens=1,
                num_active_loras=0,
                cg_mode=None,
            ),
            None,
        )
    )
    input_batch = SimpleNamespace(
        positions=torch.zeros(1, dtype=torch.long),
        num_tokens=1,
        num_tokens_after_padding=1,
        is_padding=False,
    )
    runner.gather_batch_req_state = MagicMock(return_value=(SimpleNamespace(num_tokens=1), 1))
    runner.prepare_inputs = MagicMock(return_value=input_batch)
    runner.gather_batch_req_state = MagicMock(return_value=(SimpleNamespace(num_tokens=1), None))
    runner._prepare_mm_inputs = MagicMock(return_value=(torch.zeros(1, dtype=torch.long), None, None))
    runner._add_legacy_forward_inputs = MagicMock()
    runner.model_state = SimpleNamespace(
        prepare_inputs=lambda *_args: {},
        intermediate_buffer=SimpleNamespace(gather=lambda _batch: [{}]),
    )
    runner.req_states = object()
    runner.lora_config = None
    runner.vllm_config = object()
    runner._dummy_hidden = torch.zeros(1)
    runner.kv_connector = SimpleNamespace(
        pre_forward=lambda _output: None,
        post_forward=lambda _finished: None,
    )
    runner.model = MagicMock(return_value=torch.zeros(1))
    runner.model.requires_native_model_intermediate_buffer = True
    runner.model.make_omni_output.side_effect = RuntimeError("broken Code2Wav output")
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
    )
    monkeypatch.setattr(generation_runner, "set_forward_context", lambda *_args, **_kwargs: nullcontext())

    with pytest.raises(RuntimeError, match="broken Code2Wav output"):
        generation_runner.OmniGenerationModelRunner.execute_model(runner, scheduler_output)


def test_released_chunk_uses_typed_scheduler_output_with_inherited_add_requests():
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.core.sched.output import OmniCachedRequestData
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.model = SimpleNamespace(logits_processor=None)
    runner._remove_request = MagicMock()
    req_id_to_index = MagicMock()
    req_id_to_index.get.return_value = None
    req_id_to_index.__getitem__.return_value = 0
    runner.req_states = SimpleNamespace(
        add_request=MagicMock(),
        req_id_to_index=req_id_to_index,
        apply_staged_writes=MagicMock(),
    )
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.model_state = SimpleNamespace(
        add_request=MagicMock(),
        apply_staged_writes=MagicMock(),
    )
    runner.block_tables = SimpleNamespace(append_block_ids=MagicMock())
    runner.lora_state = SimpleNamespace(add_request=MagicMock())
    runner.is_last_pp_rank = False
    runner.adaptive_verification = None
    runner.sampler = None
    cached = OmniCachedRequestData(
        req_ids=["req"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={"req": [1]},
        new_block_ids=[()],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        prompt_token_ids={"req": [1]},
        additional_information={"req": None},
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    runner._handle_async_chunk_updates(scheduler_output)

    runner.req_states.add_request.assert_called_once()
    added_request = runner.model_state.add_request.call_args.args[1]
    assert added_request.req_id == "req"
    assert added_request.prompt_token_ids == [1]
    assert added_request.prefill_token_ids == [1]


class _FakeInputBatch:
    """Minimal input batch for sample_tokens."""

    def __init__(self, num_reqs: int = 1, req_ids: list[str] | None = None):
        self.num_reqs = num_reqs
        self.req_ids = req_ids or [f"req-{i}" for i in range(num_reqs)]
        self.idx_mapping_np = np.arange(num_reqs, dtype=np.int32)


class _FakeStagedField:
    """Minimal mock for req_states fields that support staged writes."""

    def __init__(self, data: np.ndarray):
        self.np = data
        self._staged: list[tuple[int, int]] = []

    def stage_write_elem(self, idx: int, value: int) -> None:
        self._staged.append((idx, value))

    def apply_write(self) -> None:
        for idx, value in self._staged:
            self.np[idx] = value
        self._staged.clear()


class _FakeNpField:
    """Minimal mock for req_states fields with .np attribute."""

    def __init__(self, data: np.ndarray):
        self.np = data


def _make_omni_output(multimodal_outputs: dict | None = None) -> OmniOutput:
    """Create an OmniOutput with given multimodal_outputs."""
    return OmniOutput(
        text_hidden_states=torch.zeros(1),
        multimodal_outputs=multimodal_outputs,
    )


def _make_runner(
    model_output,
    num_reqs: int = 1,
    prompt_len: int = 10,
):
    """Build a minimal OmniGenerationModelRunner for sample_tokens testing."""
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.device = torch.device("cpu")

    mc = MagicMock()
    del mc.eos_token_id
    mc.hf_text_config = None
    runner.model_config = mc

    runner.postprocess = lambda *a, **kw: None

    input_batch = _FakeInputBatch(num_reqs)
    runner._gen_model_output = model_output
    runner._gen_input_batch = input_batch
    runner.execute_model_state = SimpleNamespace(finished_req_ids={"finished"}, ec_connector_output=None)
    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(return_value=None))
    runner.check_ep_fault = False

    req_states = MagicMock()
    req_states.prompt_len = _FakeNpField(
        np.full(num_reqs, prompt_len, dtype=np.int32),
    )
    req_states.num_computed_tokens = _FakeStagedField(
        np.zeros(num_reqs, dtype=np.int32),
    )
    runner.req_states = req_states

    return runner


class TestSampleTokensNoneOutput(unittest.TestCase):
    def test_none_model_output(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(None, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result is None


class TestReqStatesUpdate(unittest.TestCase):
    """Verify that sample_tokens marks all tokens as computed."""

    def test_num_computed_tokens_set_to_prompt_len(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        prompt_len = 15
        output = _make_omni_output({"model_outputs": torch.randn(2, 4)})
        runner = _make_runner(output, num_reqs=2, prompt_len=prompt_len)

        OmniGenerationModelRunner.sample_tokens(runner)

        for i in range(2):
            assert runner.req_states.num_computed_tokens.np[i] == prompt_len

    def test_connector_runs_after_computed_token_state_is_applied(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import (
            OmniGenerationModelRunner,
        )

        prompt_len = 15
        output = _make_omni_output({"model_outputs": torch.randn(2, 4)})
        runner = _make_runner(output, num_reqs=2, prompt_len=prompt_len)

        def post_forward(_finished_req_ids):
            assert runner.req_states.num_computed_tokens.np.tolist() == [
                prompt_len,
                prompt_len,
            ]

        runner.kv_connector.post_forward.side_effect = post_forward

        OmniGenerationModelRunner.sample_tokens(runner)

        runner.kv_connector.post_forward.assert_called_once_with({"finished"})


def test_sample_tokens_uses_async_output_for_cuda(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(generation_runner, "_contains_cuda_tensor", lambda _: True)
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "OmniGenerationAsyncOutput", _FakeAsyncOutput)

    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, _FakeAsyncOutput)
    assert captured["multimodal_outputs"] is output.multimodal_outputs
    assert captured["num_reqs"] == 1
    assert captured["main_stream"] is runner.main_stream
    assert captured["copy_stream"] is runner.output_copy_stream
    assert captured["finalize_output"] is runner._finalize_native_data_plane_output
    assert captured["model_runner_output"].sampled_token_ids == [[]]
    runner._release_generation_slots.assert_called_once()


def test_sample_tokens_snapshots_request_ids_before_async_finalize(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(generation_runner, "_contains_cuda_tensor", lambda _: True)
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "OmniGenerationAsyncOutput", _FakeAsyncOutput)

    input_batch = runner._gen_input_batch
    generation_runner.OmniGenerationModelRunner.sample_tokens(runner)
    output_req_ids = captured["model_runner_output"].req_ids
    assert output_req_ids == ["req-0"]

    # The scheduler can reuse and mutate the input-batch list while the async
    # output is still waiting for its D2H event.
    input_batch.req_ids[0] = "reused-request"
    assert output_req_ids == ["req-0"]


def test_sample_tokens_keeps_sync_output_for_cpu(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, OmniModelRunnerOutput)
    assert result.multimodal_outputs[0]["model_outputs"].device.type == "cpu"


def test_generation_pooler_materializes_nested_multimodal_payloads():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
        _materialize_generation_value,
    )

    output = _make_omni_output(
        {
            "codes": {"audio": torch.tensor([[1, 2], [3, 4]])},
        }
    )
    runner = _make_runner(output, num_reqs=2)

    result = OmniGenerationModelRunner.sample_tokens(runner)

    assert torch.equal(result.multimodal_outputs[0]["codes.audio"], torch.tensor([1, 2]))
    assert torch.equal(result.multimodal_outputs[1]["codes.audio"], torch.tensor([3, 4]))
    assert _materialize_generation_value({"metadata": [1, 2]}, 0, 2)["metadata"] == [1, 2]


def test_sample_tokens_reserves_native_output_before_sync_finalize(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner._reserve_native_data_plane_outputs = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock(side_effect=lambda value: value)
    generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    runner._reserve_native_data_plane_outputs.assert_called_once_with(["req-0"])


def test_async_chunk_slot_recycle_clears_model_state():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"req": 0},
        prompt_len=SimpleNamespace(np=np.zeros(1, dtype=np.int32)),
        prefill_len=SimpleNamespace(np=np.zeros(1, dtype=np.int32)),
        total_len=MagicMock(),
        all_token_ids=MagicMock(),
        num_computed_tokens=MagicMock(),
        num_computed_prefill_tokens=np.zeros(1, dtype=np.int32),
        apply_staged_writes=MagicMock(),
    )
    runner.model_state = SimpleNamespace(
        remove_request=MagicMock(),
        intermediate_buffer=SimpleNamespace(remove_request=MagicMock()),
    )
    cached = SimpleNamespace(
        req_ids=["req"],
        prompt_token_ids={"req": [7]},
        new_block_ids=[()],
        additional_information={},
    )

    with patch(
        "vllm_omni.worker_v2.omni_generation_model_runner.OmniCachedRequestData",
        type(cached),
    ):
        runner._handle_async_chunk_updates(SimpleNamespace(scheduled_cached_reqs=cached))

    runner.model_state.remove_request.assert_called_once_with(0)
    runner.model_state.intermediate_buffer.remove_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()


def test_per_request_waveform_list_does_not_slice_sample_axis():
    from vllm_omni.worker_v2.omni_generation_model_runner import _materialize_generation_value

    waves = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
    for index in range(2):
        payload = _materialize_generation_value({"audio": waves}, index, 2)
        assert torch.equal(payload["audio"], waves[index])


def test_nested_generation_preserves_per_request_sample_rates():
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    outputs = {"codes": {"audio": [torch.arange(2), torch.arange(4)]}, "sr": [24000, 16000]}
    result = OmniGenerationModelRunner._build_pooler_output_from_cpu(outputs, 2)
    assert result[0]["sr"] == 24000
    assert result[1]["sr"] == 16000
    assert torch.equal(result[0]["codes.audio"], torch.arange(2))
    assert torch.equal(result[1]["codes.audio"], torch.arange(4))


def test_cpu_generation_output_owns_waveform_after_model_buffer_reuse():
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    waveform = torch.arange(4, dtype=torch.float32)
    output = OmniOutput(text_hidden_states=torch.empty(0), multimodal_outputs={"codes": {"audio": [waveform]}})
    result = OmniGenerationModelRunner._build_pooler_output(output, 1)
    waveform.fill_(-1)
    assert torch.equal(result[0]["codes.audio"], torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize(
    "payload,num_reqs,key,shapes",
    [
        ({"model_outputs": torch.ones(1, 4, 8)}, 1, "model_outputs", [(4, 8)]),
        ({"model_outputs": torch.ones(3, 2, 5)}, 3, "model_outputs", [(2, 5)] * 3),
        ({"model_outputs": [torch.ones(3, 2)]}, 1, "model_outputs", [(3, 2)]),
        ({"model_outputs": [None]}, 1, None, []),
        ({"audio": torch.ones(2, 16), "sr": 24000}, 2, "audio", [(16,)] * 2),
        ({"chunks": [torch.ones(10), torch.ones(20)]}, 2, "chunks", [(10,), (20,)]),
        (None, 1, None, []),
        (None, 2, None, []),
    ],
)
def test_generation_output_partition(payload, num_reqs, key, shapes):
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    runner = _make_runner(_make_omni_output(payload), num_reqs=num_reqs)
    result = OmniGenerationModelRunner.sample_tokens(runner)
    assert isinstance(result, OmniModelRunnerOutput)
    assert result.pooler_output is None
    assert result.sampled_token_ids == [[] for _ in range(num_reqs)]
    assert len(result.multimodal_outputs) == num_reqs
    assert runner.req_states.num_computed_tokens.np.tolist() == [10] * num_reqs
    if key is None:
        assert result.multimodal_outputs == [{} for _ in range(num_reqs)]
    else:
        assert [tuple(item[key].shape) for item in result.multimodal_outputs] == shapes
        for item in result.multimodal_outputs:
            assert torch.equal(item[key], torch.ones_like(item[key]))
        if payload is not None and "sr" in payload:
            assert all(item["sr"].item() == 24000 for item in result.multimodal_outputs)


@pytest.mark.parametrize("has_writer", [False, True])
def test_block_table_staged_writes_require_writer(has_writer):
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    runner = object.__new__(OmniGenerationModelRunner)
    runner.block_tables = MagicMock(fused_writer=object() if has_writer else None)
    runner._apply_block_table_staged_writes_if_available()
    assert runner.block_tables.apply_staged_writes.call_count == int(has_writer)
