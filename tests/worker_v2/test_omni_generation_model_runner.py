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

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = []


def test_execute_model_does_not_reference_removed_perf_hook():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    source = inspect.getsource(OmniGenerationModelRunner.execute_model)
    assert "_record_execution_batch" not in source


def test_execute_model_uses_shared_vllm_025_mm_input_contract():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    source = inspect.getsource(OmniGenerationModelRunner.execute_model)
    assert "self._prepare_mm_inputs(" in source
    assert "self.model_state.get_mm_embeddings(" not in source
    assert '"input_ids": input_ids' in source


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
    runner._gen_kv_connector_output = None
    runner.execute_model_state = None

    req_states = MagicMock()
    req_states.prompt_len = _FakeNpField(
        np.full(num_reqs, prompt_len, dtype=np.int32),
    )
    req_states.num_computed_tokens = _FakeStagedField(
        np.zeros(num_reqs, dtype=np.int32),
    )
    runner.req_states = req_states

    return runner


class TestSampleTokensTensorOutput(unittest.TestCase):
    def test_single_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(1, 4, 8)})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert isinstance(result, OmniModelRunnerOutput)
        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 1
        assert result.multimodal_outputs[0]["model_outputs"].shape == (4, 8)

    def test_multi_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(3, 2, 5)})
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 3
        for i in range(3):
            assert result.multimodal_outputs[i]["model_outputs"].shape == (2, 5)


class TestSampleTokensListOutput(unittest.TestCase):
    def test_list_of_tensors(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [torch.randn(3, 2)]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 1
        assert result.multimodal_outputs[0]["model_outputs"].shape == (3, 2)

    def test_list_with_none(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [None]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert result.multimodal_outputs == [{}]


class TestSampleTokensDictOutput(unittest.TestCase):
    def test_dict_with_batched_tensor(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"audio": torch.randn(2, 16000), "sr": 24000})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 2
        assert result.multimodal_outputs[0]["audio"].shape == (16000,)
        assert result.multimodal_outputs[1]["audio"].shape == (16000,)
        assert torch.is_tensor(result.multimodal_outputs[0]["sr"])
        assert result.multimodal_outputs[0]["sr"].item() == 24000

    def test_dict_with_list_values(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"chunks": [torch.randn(10), torch.randn(20)]})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 2
        assert result.multimodal_outputs[0]["chunks"].shape == (10,)
        assert result.multimodal_outputs[1]["chunks"].shape == (20,)


class TestSampleTokensNoneOutput(unittest.TestCase):
    def test_none_model_output(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(None, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result is None


class TestNonDictMultimodalOutputs(unittest.TestCase):
    """When multimodal_outputs is None or non-dict, per-request output is empty."""

    def test_none_multimodal_outputs(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output(multimodal_outputs=None)
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert result.multimodal_outputs == [{}, {}]


class TestSampledTokenIds(unittest.TestCase):
    def test_empty_sampled_token_ids_per_request(self):
        """Generation models emit empty sampled_token_ids (no token sampling)."""
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(3, 2)})
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.sampled_token_ids) == 3
        for ids in result.sampled_token_ids:
            assert ids == []


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


def test_sample_tokens_uses_async_output_for_cuda_async_scheduler(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "_uses_async_output", lambda _runner: True)
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
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "_uses_async_output", lambda _runner: True)
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
    monkeypatch.setattr(generation_runner, "_uses_async_output", lambda _runner: True)

    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, OmniModelRunnerOutput)
    assert result.multimodal_outputs[0]["model_outputs"].device.type == "cpu"


class TestMultimodalOutputsPassthrough(unittest.TestCase):
    """multimodal_outputs is a per-request list (tensor-only) on OmniModelRunnerOutput.

    OmniGenerationScheduler indexes it as mm_outputs[req_index], so it must be a
    list (not the raw dict). Each entry mirrors the per-request pooler payload.
    """

    def test_multimodal_outputs_on_result(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        mm = {"audio": [torch.randn(10)]}
        output = _make_omni_output(mm)
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert isinstance(result.multimodal_outputs, list)
        assert len(result.multimodal_outputs) == 1
        assert "audio" in result.multimodal_outputs[0]
        assert torch.is_tensor(result.multimodal_outputs[0]["audio"])

    def test_none_multimodal_outputs_becomes_empty_dict(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output(multimodal_outputs=None)
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        # No multimodal data -> one empty dict per request.
        assert result.multimodal_outputs == [{}]


class TestBlockTableWrites(unittest.TestCase):
    def test_skips_no_kv_block_table_without_fused_writer(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = object.__new__(OmniGenerationModelRunner)
        block_tables = MagicMock()
        block_tables.fused_writer = None
        runner.block_tables = block_tables

        runner._apply_block_table_staged_writes_if_available()

        block_tables.apply_staged_writes.assert_not_called()

    def test_applies_block_table_writes_when_writer_exists(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = object.__new__(OmniGenerationModelRunner)
        block_tables = MagicMock()
        block_tables.fused_writer = object()
        runner.block_tables = block_tables

        runner._apply_block_table_staged_writes_if_available()

        block_tables.apply_staged_writes.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
