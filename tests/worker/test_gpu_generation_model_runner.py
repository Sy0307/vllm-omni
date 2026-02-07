import torch
from types import SimpleNamespace

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


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
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
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


def _make_stage_runner(model_stage=None, model_arch=None, cls_name="DummyModel", uses_mrope=True):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.uses_mrope = uses_mrope
    runner.model_config = SimpleNamespace(
        model_stage=model_stage,
        model_arch=model_arch,
        hf_config=SimpleNamespace(architectures=None),
    )
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model_stage=model_stage,
            model_arch=model_arch,
        )
    )
    model_cls = type(cls_name, (), {})
    runner.model = model_cls()
    return runner


def test_disable_mrope_for_code2wav_by_model_stage():
    runner = _make_stage_runner(model_stage="code2wav", model_arch=None, uses_mrope=True)
    runner._maybe_disable_mrope_for_code2wav()
    assert runner.uses_mrope is False


def test_disable_mrope_for_code2wav_by_model_arch():
    runner = _make_stage_runner(model_stage=None, model_arch="Qwen3TTSCode2Wav", uses_mrope=True)
    runner._maybe_disable_mrope_for_code2wav()
    assert runner.uses_mrope is False


def test_disable_mrope_for_code2wav_by_model_class_name():
    runner = _make_stage_runner(model_stage=None, model_arch=None, cls_name="Qwen3TTSCode2Wav", uses_mrope=True)
    runner._maybe_disable_mrope_for_code2wav()
    assert runner.uses_mrope is False
