from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


def _make_runner(multimodal_outputs):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert "audio" in output.pooler_output[0]
    assert "unused" not in output.pooler_output[0]
    assert output.pooler_output[0]["audio"].shape == (1, 4)


class _DummyFishDACModel:
    have_multimodal_outputs = True

    def __init__(self):
        self.runtime_infos = None

    def forward(self, input_ids, positions, runtime_additional_information):
        assert input_ids is None
        assert positions is None
        self.runtime_infos = runtime_additional_information
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [
                    torch.tensor([1.0, 2.0]),
                    torch.tensor([3.0, 4.0]),
                ],
                "sr": [44100, 44100],
            },
            intermediate_tensors=None,
        )


def test_fish_dac_runner_fastpath_executes_runtime_infos(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_DAC_RUNNER_FASTPATH", "1")
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = _DummyFishDACModel()
    runner.model_config = SimpleNamespace(
        model_stage="dac_decoder",
        async_chunk=True,
    )
    runner.execute_model_state = None

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="new-req",
                additional_information={"code_predictor_codes": "new-codes"},
            ),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["cached-req"],
            additional_information={
                "cached-req": {"code_predictor_codes": "cached-codes"},
            },
        ),
    )

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert output.req_ids == ["new-req", "cached-req"]
    assert output.req_id_to_index == {"new-req": 0, "cached-req": 1}
    assert runner.model.runtime_infos == [
        {"code_predictor_codes": "new-codes"},
        {"code_predictor_codes": "cached-codes"},
    ]
    assert output.pooler_output[0]["model_outputs"].tolist() == [1.0, 2.0]
    assert output.pooler_output[1]["model_outputs"].tolist() == [3.0, 4.0]
    assert output.pooler_output[0]["sr"] == 44100
