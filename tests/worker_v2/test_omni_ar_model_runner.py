# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniARModelRunner v2: async output staging, snapshot ownership, payload slicing."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.outputs import RoutedExpertsTensors
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner, OmniAsyncOutput, _async_copy_mm
from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        pass


class _FakeEvent:
    def record(self, _stream) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _async_output(req_ids=("req-0",), **overrides) -> OmniAsyncOutput:
    kwargs = dict(
        model_runner_output=omni_ar_model_runner.OmniModelRunnerOutput(
            req_ids=list(req_ids),
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[123]]), logprobs_tensors=None, num_nans=None, sampling_mask_tensors=None
        ),
        num_sampled_tokens=torch.ones(len(req_ids), dtype=torch.long),
        main_stream=_FakeStream(),
        copy_stream=_FakeStream(),
        copy_event=_FakeEvent(),
    )
    kwargs.update(overrides)
    return OmniAsyncOutput(**kwargs)


def test_async_output_blocking_event_and_routing_masks(monkeypatch) -> None:
    event_kwargs = []
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kw: event_kwargs.append(kw) or _FakeEvent())

    output = _async_output(
        req_ids=["decode", "prefill"],
        sampler_output=SamplerOutput(
            sampled_token_ids=torch.tensor([[2], [0]]),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.tensor([1, 0]),
            sampling_mask_tensors=SamplingMaskTensors(
                torch.tensor([[5], [0]], dtype=torch.uint8), torch.tensor([2, 0]), 4
            ),
        ),
        num_sampled_tokens=torch.tensor([1, 0]),
        copy_event=None,  # force the constructor to create the default event
        routed_experts=RoutedExpertsTensors(torch.tensor([[[2, 3]], [[4, 5]]]), torch.tensor([7, 9])),
    ).get_output()

    assert event_kwargs == [{"blocking": True}]  # blocking event by default
    assert output.sampled_token_ids == [[2], []] and output.sampling_masks.cu_num_generated_tokens == [0, 1, 1]
    np.testing.assert_array_equal(output.routed_experts.routing_data, [[[2, 3]], [[4, 5]]])
    np.testing.assert_array_equal(output.routed_experts.slot_mapping, [7, 9])
    np.testing.assert_array_equal(output.sampling_masks.token_ids, [0, 2])


@pytest.mark.parametrize("needs_history", [False, True])
def test_last_pp_rank_sampling_context_and_connector_orchestration(monkeypatch, needs_history) -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.execute_model_state = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["req"],
            idx_mapping=torch.tensor([0]),
            seq_lens=torch.tensor([3]),
            query_start_loc=torch.tensor([0, 1]),
            num_reqs=1,
        ),
        hidden_states=torch.zeros(1, 2),
        finished_req_ids={"finished"},
        ec_connector_output=None,
        routed_experts=None,
    )
    runner._kv_extracted_req_ids = runner._last_aux_output = None
    runner._last_multimodal_outputs = runner._last_multimodal_snapshot_slot = None
    runner.is_last_pp_rank, runner.pp_handler, runner.check_ep_fault = True, None, False
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    runner.model_state = SimpleNamespace(postprocess_model_output=MagicMock(return_value=(torch.zeros(1, 2), None)))
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1]])),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([0])),
        prompt_len=SimpleNamespace(np=np.array([1]), gpu=torch.tensor([1])),
    )
    runner.main_stream = runner.output_copy_stream = MagicMock()
    runner.eplb = runner._finalize_native_data_plane_output = runner._reserve_native_data_plane_outputs = MagicMock()
    runner.sample = MagicMock(
        return_value=(SimpleNamespace(sampled_token_ids=torch.tensor([[2]])), torch.tensor([1]), torch.tensor([0]))
    )
    sampling_active = False

    @contextmanager
    def sampling_context(*, req_ids, num_output_tokens):
        nonlocal sampling_active
        assert needs_history and req_ids == ["req"]
        sampling_active = True
        yield
        sampling_active = False

    runner.sample.side_effect = (
        lambda *_: runner.sample.return_value if sampling_active is needs_history else pytest.fail("outside ctx")
    )
    runner.model = SimpleNamespace(
        compute_logits=MagicMock(),
        logitsprocs_need_output_token_ids=needs_history,
        mrv2_sampling_context=sampling_context,
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=MagicMock(side_effect=lambda *_: pytest.fail() if sampling_active else {})
    )
    runner.postprocess_sampled, connector_output = MagicMock(), object()

    def post_forward(finished_req_ids):
        runner.postprocess_sampled.assert_called_once()  # postprocess precedes kv_connector.post_forward
        assert finished_req_ids == {"finished"}
        return connector_output

    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(side_effect=post_forward))
    monkeypatch.setattr(
        omni_ar_model_runner, "OmniAsyncOutput", MagicMock(return_value=SimpleNamespace(copy_event=None))
    )

    assert runner.sample_tokens(None) is omni_ar_model_runner.OmniAsyncOutput.return_value
    built = omni_ar_model_runner.OmniAsyncOutput.call_args.kwargs["model_runner_output"]
    assert built.kv_connector_output is connector_output


def test_kv_transfer_request_id_resolver_reads_intermediate_buffer() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.req_states = SimpleNamespace(req_id_to_index={"req-0": 2, "req-1": 3})
    runner.model_state = SimpleNamespace(
        intermediate_buffer=SimpleNamespace(
            buffers={2: {"global_request_id": b"g-0"}, 3: {"global_request_id": ["g-3"]}}
        )
    )
    assert runner._resolve_global_request_id("req-0") == "g-0"  # bytes decoded
    assert runner._resolve_global_request_id("req-1") == "g-3"  # list resolves to first entry
    assert runner._resolve_global_request_id("unknown") == "unknown"  # fallback to the local id


def test_async_mm_snapshot_owns_output_until_copy_finishes() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner._async_mm_snapshot_slots, runner._async_mm_snapshot_events = [{}], [None]
    runner._async_mm_snapshot_pending, runner._async_mm_snapshot_cursor = [False], 0
    runner._last_multimodal_snapshot_slot = None
    waited = []
    runner.main_stream = SimpleNamespace(wait_event=waited.append)
    source = torch.tensor([[7, 8]], dtype=torch.long)

    snapshot = runner._retain_multimodal_outputs({"codes": {"audio": source}})
    source.fill_(99)

    # Snapshot owns the data (graph replay cannot overwrite it)...
    assert snapshot["codes"]["audio"].tolist() == [[7, 8]]
    assert snapshot["codes"]["audio"].data_ptr() != source.data_ptr()
    assert runner._last_multimodal_snapshot_slot == 0
    # ...and slot reuse waits for the previous D2H copy event.
    runner._release_multimodal_snapshot(0, copy_event := object())
    runner._retain_multimodal_outputs({"codes": {"audio": torch.zeros(1, 2)}})
    assert waited == [copy_event]


def test_snapshot_slots_bucket_by_shape_and_stay_bounded(monkeypatch) -> None:
    slot = {}
    omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(1, 2), slot)
    omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(4, 2), slot)
    assert len(slot) == 2  # separate shapes do not share a buffer
    monkeypatch.setattr(omni_ar_model_runner, "_ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT", 1)
    bounded = {}
    kept = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(2, 2), bounded)
    overflow = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(5, 2), bounded)
    assert len(bounded) == 1 and overflow.data_ptr() != kept.data_ptr()  # overflow clones without evicting


def test_packed_snapshot_dtype_grouping_nesting_and_publish_isolation() -> None:
    source = torch.arange(12, dtype=torch.int64).view(3, 4).t()
    payload = {"noncontiguous": source, "nested": [torch.tensor(7), (torch.tensor([1.5]),)], "meta": "ok"}
    slot = {}
    snapshot = pack_output_snapshot(payload, slot, max_buckets=4)
    source.fill_(99)
    copies = []

    def copy(tensor):
        copies.append(tensor.numel())
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)
    assert len(copies) == 2  # grouped by dtype (int64, float32), not tensor count
    assert host["noncontiguous"].tolist() == torch.arange(12).view(3, 4).t().tolist()
    assert host["nested"][0].item() == 7 and isinstance(host["nested"][1], tuple) and host["meta"] == "ok"
    pack_output_snapshot(payload, slot, max_buckets=4)
    assert host["noncontiguous"][0, 0].item() == 0  # repacking must not corrupt published host data


@pytest.mark.parametrize("need_pooler", [False, True])
@pytest.mark.parametrize("async_chunk", [False, True])
def test_guard_graph_replay_for_pooler_copy(need_pooler, async_chunk) -> None:
    main_stream = MagicMock()
    omni_ar_model_runner._guard_graph_replay_for_pooler_copy(
        main_stream, object(), need_pooler=need_pooler, async_chunk=async_chunk
    )
    # Non-async pooler copies must gate the next graph replay on the copy event.
    assert main_stream.wait_event.call_count == (1 if need_pooler and not async_chunk else 0)


@pytest.mark.parametrize(
    "mm,aux,expected",
    [
        ({"latent": torch.randn(3, 4)}, None, "omni"),  # multimodal → OmniOutput
        (None, {"layers": torch.randn(3, 2)}, "tuple"),  # aux only → (hidden, aux)
        ({}, None, "raw"),  # empty multimodal dict is falsy → bare hidden
    ],
)
def test_reconstruct_raw_model_output_forms(mm, aux, expected) -> None:
    hidden = torch.randn(3, 4)
    raw = OmniARModelRunner._reconstruct_raw_model_output(hidden_states=hidden, multimodal_outputs=mm, aux=aux)
    if expected == "omni":
        assert isinstance(raw, OmniOutput) and raw.text_hidden_states is hidden
    elif expected == "tuple":
        assert raw == (hidden, aux)
    else:
        assert raw is hidden


def test_build_pooler_output_nested_slices_owned_storage_and_qwen3_round_trip() -> None:
    mm_cpu = {
        "feat": torch.randn(6, 2),  # sliced along the token axis
        "items": [torch.randn(2), torch.randn(3)],  # per-request list elements pass through
        "nested": {"a": torch.randn(6, 2)},  # recursive dicts slice too
    }
    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        torch.randn(6, 4),
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3]),
        num_reqs=2,
    )
    assert pooler[0]["feat"].shape == (3, 2) and pooler[1]["nested"]["a"].shape == (3, 2)
    assert isinstance(pooler[0]["items"], torch.Tensor)
    sliced = pooler[0]["hidden"]
    # The slice owns its storage; later writes to the source cannot corrupt it.
    assert sliced.is_contiguous() and sliced.untyped_storage().nbytes() == sliced.numel() * sliced.element_size()

    # Qwen3 nested payload: flatten to dotted keys → slice → unflatten back.
    mm = {
        "hidden_states": {"layers": {0: torch.randn(4, 4), 24: torch.randn(4, 4)}},
        "embed": {"tts_bos": [torch.randn(1, 1, 4)], "tts_eos": [torch.randn(1, 1, 4)]},
        "codes": {"audio": torch.randn(4, 16)},
    }
    mm_cpu = _async_copy_mm(mm, total_tokens=4)
    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        torch.randn(4, 4),
        mm_cpu,
        query_start_loc_np=np.array([0, 2]),
        num_scheduled_tokens=np.array([2, 2]),
        num_reqs=2,
    )
    assert "hidden_states" not in pooler[0] and pooler[0]["codes.audio"].shape == (2, 16)
    payload = unflatten_payload(pooler[0])
    assert payload["hidden_states"]["layers"][0].shape == (2, 4) and payload["hidden_states"]["layers"][24].shape == (
        2,
        4,
    )
    assert payload["embed"]["tts_bos"].shape == (1, 1, 4)


def test_build_async_chunk_outputs_slices_padded_axis_and_splits_channels() -> None:
    # Graph-padded batch: padded_total_tokens > total_tokens; slice by real tokens.
    padded_codes = torch.arange(16, dtype=torch.long).reshape(8, 2)
    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        {"codes": {"audio": padded_codes}},
        np.array([0, 1, 2]),
        np.array([1, 1], dtype=np.int32),
        num_reqs=2,
        total_tokens=2,
        padded_total_tokens=8,
    )
    assert client is None
    assert torch.equal(inter_stage[0]["codes.audio"], padded_codes[0:1])
    assert torch.equal(inter_stage[1]["codes.audio"], padded_codes[1:2])

    codes = torch.arange(16, dtype=torch.long).reshape(4, 4)
    audio = torch.randn(4, 8)
    req_codes = [torch.arange(16 * i, 16 * (i + 1), dtype=torch.long).reshape(1, 16) for i in range(2)]
    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        {"codes": {"audio": codes}, "audio": audio, "req": {"codes": req_codes}},
        np.array([0, 2, 4]),
        np.array([2, 2], dtype=np.int32),
        num_reqs=2,
        total_tokens=4,
    )
    assert "hidden" not in inter_stage[0]  # no hidden materialized on this path
    assert torch.equal(inter_stage[1]["codes.audio"], codes[2:])  # inter-stage channel
    assert torch.equal(client[1]["audio"], audio[2:])  # client-visible channel
    assert torch.equal(inter_stage[0]["req.codes"], req_codes[0])  # per-request lists pass through


def test_async_chunk_output_stages_mm_on_copy_stream_before_get_output(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    calls = []

    def copy_mm(mm_outputs, total_tokens, **ctx):
        calls.append((total_tokens, ctx))
        return {"codes": {"audio": mm_outputs["codes"]["audio"].clone()}}

    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm", copy_mm)
    source_codes = torch.tensor([[7, 8]], dtype=torch.long)
    output = _async_output(
        multimodal_outputs={"codes": {"audio": source_codes}},
        input_batch=SimpleNamespace(
            query_start_loc_np=np.array([0, 1], dtype=np.int32),
            num_scheduled_tokens=np.array([1]),
            num_reqs=1,
            num_tokens_after_padding=1,
        ),
        copy_stream=(copy_stream := _FakeStream()),
        async_chunk=True,
    )

    # The mm payload was staged once on the copy stream during construction,
    # reusing one resolved pin-memory context for all D2H helpers.
    [(total_tokens, ctx)] = calls
    assert total_tokens == 1 and ctx["copy_stream"] is copy_stream and ctx["pin_memory"] is not None
    assert output._mm_snapshot["codes"]["audio"].device.type == "cpu"
    source_codes.fill_(99)  # a later graph replay cannot leak into the snapshot
    finalized = output.get_output()
    assert torch.equal(finalized.inter_stage_outputs[0]["codes.audio"], torch.tensor([[7, 8]], dtype=torch.long))


def test_async_copy_mm_nested_payload_scalars_and_leaf_failure(monkeypatch) -> None:
    payload = {"codes": {"audio": torch.randn(2, 3)}, "items": [torch.randn(3), "text"], "meta": "s"}
    result = _async_copy_mm(payload, total_tokens=2)
    assert result["codes"]["audio"].device == torch.device("cpu")
    assert isinstance(result["items"][0], torch.Tensor) and result["items"][1] == "text"
    assert _async_copy_mm({}, 10) == {}
    assert not omni_ar_model_runner._has_cuda_tensor({"codes": {"audio": torch.ones(1)}})

    good, bad = torch.randn(2), torch.randn(2)
    original = omni_ar_model_runner._async_copy_mm_value

    def fail_leaf(value, **kw):
        if value is bad:
            raise RuntimeError("D2H failure")
        return original(value, **kw)

    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm_value", fail_leaf)
    with pytest.raises(RuntimeError, match="D2H failure"):
        _async_copy_mm({"good": good, "bad": bad}, 10)
