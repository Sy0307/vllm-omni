# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniGPUModelRunner v2 dispatch and lifecycle overrides.

Retained contracts: V1/V2 forward-input dispatch, empty-vs-new admission,
GPU-resident cached side state, native data plane terminal/abort split,
finalize/reserve ownership order, vLLM 0.29 capture contract (tuple unwrap +
FULL-graph exclusion + MTP follow-up), descriptor dispatch, and the
init_omni_model_state factory boundary.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner():
    """Create an OmniGPUModelRunner without calling __init__."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MagicMock()
    runner.req_states = SimpleNamespace(req_id_to_index={"r1": 0, "r2": 1})
    runner.execute_model_state = None
    return runner


@pytest.mark.parametrize("native_data_plane", [False, True])
def test_legacy_forward_inputs_dispatch(native_data_plane):
    runner = _make_runner()
    runner._omni_data_plane = object() if native_data_plane else None
    runner.sampler = object()
    sampling_metadata = object()
    logits_indices = torch.tensor([0])
    input_batch = SimpleNamespace(sampling_metadata=sampling_metadata, logits_indices=logits_indices)
    model_inputs = {"input_ids": torch.tensor([1])}

    runner._add_legacy_forward_inputs(model_inputs, input_batch)

    if native_data_plane:
        assert set(model_inputs) == {"input_ids"}
    else:
        assert model_inputs["sampling_metadata"] is sampling_metadata
        assert model_inputs["logits_index"] is logits_indices
        assert model_inputs["sampler"] is runner.sampler


def test_add_requests_empty_admission_and_stop_id_sanitization():
    runner = _make_runner()
    runner.sampler = Sampler.__new__(Sampler)
    with patch.object(GPUModelRunner, "add_requests", return_value=None) as parent:
        runner.add_requests(SchedulerOutput.make_empty())
        parent.assert_not_called()

        # New requests always go upstream; narrow logits heads sanitize stop ids.
        sampling_params = SamplingParams(min_tokens=2, stop_token_ids=[2150])
        sampling_params.update_from_generation_config({}, 151645)
        runner.model = SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=3072))
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [SimpleNamespace(sampling_params=sampling_params)]
        runner.add_requests(output)
        parent.assert_called_once_with(output)
    assert sampling_params.all_stop_token_ids == {2150}
    assert sampling_params.eos_token_id == 151645


def test_update_requests_preserves_cached_gpu_resident_side_state():
    runner = _make_runner()
    gpu_keys = {("hidden_states", "last"), ("hidden_states", "trailing_text")}
    runner.model = SimpleNamespace(gpu_resident_buffer_keys=gpu_keys)
    update_calls = []
    runner.model_state = SimpleNamespace(intermediate_buffer=SimpleNamespace(update=lambda *a: update_calls.append(a)))
    hidden = torch.randn(4)
    sched_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(additional_information={"r1": {"hidden_states": {"last": hidden}}})
    )

    with patch.object(GPUModelRunner, "update_requests", return_value=None):
        runner.update_requests(sched_output)

    assert update_calls == [(0, {"hidden_states": {"last": hidden}}, gpu_keys)]


def test_prepare_native_data_plane_terminal_abort_split_and_warmup_skip():
    runner = _make_runner()
    plane = SimpleNamespace(
        register_request=MagicMock(),
        register_receivers=MagicMock(),
        request_terminal=MagicMock(),
        abort_requests=MagicMock(),
    )
    runner._omni_data_plane = plane
    new_req = SimpleNamespace(req_id="r1")
    warmup_req = SimpleNamespace(req_id="_warmup_0_")
    handle = SimpleNamespace(request_id="r2", external_req_id="ext-r2")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_req, warmup_req],
        pending_input_registrations=[handle],
        data_plane_terminal_req_ids={"r0"},
        finished_req_ids={"r0", "aborted"},
    )

    runner._prepare_native_data_plane(scheduler_output)

    plane.register_request.assert_called_once_with(new_req)  # warmup skipped
    plane.register_receivers.assert_called_once_with([handle])
    plane.request_terminal.assert_called_once_with({"r0"})
    plane.abort_requests.assert_called_once_with({"aborted"})


def test_finalize_reserves_and_routes_native_data_plane_output():
    runner = _make_runner()
    connector_output = SimpleNamespace(chunk_ready_req_ids={"ready"})
    plane = SimpleNamespace(
        enqueue_outputs=MagicMock(),
        reserve_outputs=MagicMock(),
        get_omni_connector_output=MagicMock(return_value=connector_output),
    )
    runner._omni_data_plane = plane
    output = SimpleNamespace(
        req_ids=["r1"],
        inter_stage_outputs=[{"codes.audio": "gpu-tensor"}],
        sampled_token_ids=[[21]],
        omni_connector_output=None,
    )
    result = runner._finalize_native_data_plane_output(output)
    assert result is output
    plane.enqueue_outputs.assert_called_once_with(
        req_ids=["r1"], inter_stage_outputs=[{"codes.audio": "gpu-tensor"}], sampled_token_ids=[[21]]
    )
    assert output.inter_stage_outputs is None
    assert output.omni_connector_output is connector_output
    runner._reserve_native_data_plane_outputs(["r1", "r2"])
    plane.reserve_outputs.assert_called_once_with(["r1", "r2"])


@pytest.mark.parametrize("output_form", ["tuple", "omni"])
def test_capture_model_unwraps_exclude_full_and_capture_mtp(output_form):
    runner = object.__new__(OmniGPUModelRunner)
    hidden = torch.ones(1, 2)

    def original_forward():
        if output_form == "tuple":
            return hidden, {"layers": {}}
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    runner.model = SimpleNamespace(forward=original_forward)
    runner._model_returns_tuple = True
    runner._exclude_full_graph = True
    runner.use_aux_hidden_state_outputs = False
    piecewise = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE)
    full = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.PIECEWISE: [piecewise], CUDAGraphMode.FULL: [full]},
        _candidates={(1, 0): [piecewise, full]},
    )
    runner.model_state = SimpleNamespace(capture_talker_mtp_graphs=MagicMock())
    runner._dispatch_mtp_batch_descriptor = MagicMock(return_value="desc")

    def assert_unwrapped(_self):
        assert torch.equal(runner.model.forward(), hidden)  # forward unwrapped during capture
        return 3

    with patch.object(GPUModelRunner, "capture_model", assert_unwrapped):
        assert runner.capture_model() == 3

    assert runner.model.forward is original_forward  # restored after capture
    assert runner.cudagraph_manager._capture_descs == {CUDAGraphMode.PIECEWISE: [piecewise]}
    runner.model_state.capture_talker_mtp_graphs.assert_called_once_with(runner._dispatch_mtp_batch_descriptor)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_descriptor_dispatch_and_mtp_bucket(dp_size):
    runner = object.__new__(OmniGPUModelRunner)
    batch_desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    runner.cudagraph_manager = SimpleNamespace(dispatch=MagicMock(return_value=batch_desc))
    runner.dp_size = dp_size
    runner.dp_rank = 0
    with patch("vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding", return_value=("synced", "tokens")) as sync:
        result = runner._dispatch_batch_descriptor(
            num_reqs=2, num_toks=8, uniform_tok_count=4, num_active_loras=3, use_eager=False, max_query_len=4
        )
    runner.cudagraph_manager.dispatch.assert_called_once_with(2, 8, 4, num_active_loras=3, max_query_len=4)
    if dp_size == 1:
        sync.assert_not_called()
        assert result == (batch_desc, None)
    else:
        sync.assert_called_once()
        assert result == ("synced", "tokens")

    # MTP dispatch uses the largest captured bucket, or falls back to eager.
    runner.scheduler_config = SimpleNamespace(max_num_seqs=6)
    runner.model_state = SimpleNamespace(_get_talker_mtp_capture_sizes=MagicMock(return_value=[4, 2, 1]))
    runner.cudagraph_manager.dispatch.reset_mock()
    assert runner._dispatch_mtp_batch_descriptor(3) is batch_desc
    runner.cudagraph_manager.dispatch.assert_called_once_with(4, 4, 1, 0)
    result = runner._dispatch_mtp_batch_descriptor(6)
    assert result.cg_mode == CUDAGraphMode.NONE and result.num_tokens == 6


@pytest.mark.parametrize(
    "parallel_config,match",
    [
        (dict(pipeline_parallel_size=2, prefill_context_parallel_size=1), "pipeline parallel"),
        (dict(pipeline_parallel_size=1, prefill_context_parallel_size=2), "prefill context parallelism"),
    ],
)
def test_mrv2_rejects_parallel_modes_at_startup(parallel_config, match):
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(**parallel_config))
    with pytest.raises(NotImplementedError, match=match):
        runner._validate_parallel_support()


@pytest.mark.parametrize(
    "architectures,expect_omni",
    [(["LlamaForCausalLM"], False), (["Qwen3TTSTalkerForConditionalGeneration"], True)],
)
def test_init_model_state_factory_dispatches_omni_only(monkeypatch, architectures, expect_omni):
    upstream = MagicMock(return_value=object())
    monkeypatch.setattr("vllm_omni.worker_v2.model_states._upstream_init_model_state", upstream)
    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(architectures=architectures))

    state = init_omni_model_state(cfg, SimpleNamespace(), None, torch.device("cpu"))

    if expect_omni:
        assert isinstance(state, OmniModelState)
        upstream.assert_not_called()
    else:
        upstream.assert_called_once()
        assert state is upstream.return_value


def test_finish_requests_notifies_model_and_cleans_only_known_slots(monkeypatch):
    runner = _make_runner()
    calls = []
    runner.model = SimpleNamespace(on_requests_finished=lambda ids: calls.append(set(ids)))
    runner.model_state = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "finish_requests", lambda *args: None)
    runner.finish_requests(SimpleNamespace(finished_req_ids={"released"}, preempted_req_ids={"r1"}))
    assert calls == [{"released"}]
    assert sorted(c.args[0] for c in runner.model_state.remove_request.call_args_list) == [0]
