# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 duplex admission, row ownership and partial-prefill policy."""

from pathlib import Path

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45DuplexSampler
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
    _MiniCPMO45PendingSamples,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_profile_preserves_session_capacity_and_transport(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    path = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/minicpmo_4_5_duplex_mrv2_h200.yaml"
    deploy = _apply_platform_overrides(load_deploy_config(path), platform="cuda")
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, deploy)
    assert deploy.session_mode == "duplex" and deploy.duplex_session.max_sessions == 16
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True] * 3
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [16] * 3
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [False, True, True]
    # Only the Thinker overlaps host scheduling with its running decode step.
    assert [s.yaml_engine_args["async_scheduling"] for s in stages] == [True, False, False]
    assert not any(s.yaml_engine_args["supports_native_mrv2_data_plane"] for s in stages)
    # The checkpoint has 36 layers, 8 KV heads and head_dim 128 in BF16.
    # All sixteen 8k basic windows plus one full prefill batch must fit.
    kv_bytes_per_token = 36 * 8 * 128 * 2 * 2
    thinker_args = stages[0].yaml_engine_args
    assert thinker_args["kv_cache_memory_bytes"] // kv_bytes_per_token >= (
        thinker_args["max_num_seqs"] * 8000 + thinker_args["max_num_batched_tokens"]
    )
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == 4 * 1024**3
    from vllm_omni.config.omni_config import VllmOmniConfig

    structured = VllmOmniConfig.from_pipeline_config(MINICPMO_4_5_PIPELINE, deploy_config_path=str(path))
    assert all(s.model_config.use_v2_model_runner for s in structured.stage_configs)
    assert not any(s.model_config.supports_native_mrv2_data_plane for s in structured.stage_configs)
    # Stage connector extras are attached after the structured model config.
    # Follow the actual transfer resolver, including base YAML inheritance.
    from vllm_omni.config.stage_config import resolve_deploy_yaml
    from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    transfer = load_omni_transfer_config(config_dict=resolve_deploy_yaml(path))
    codec_extra = get_stage_connector_spec(transfer, 2, async_chunk=True)["extra"]
    assert codec_extra["cfm_fused_body"] and codec_extra["cfm_slot_pool"]
    assert codec_extra["code2wav_bfloat16_attention_cache"] is False


def test_sampler_skips_partial_prefill_and_resets_history_on_new_segment(mocker):
    params = SamplingParams(temperature=0.0, top_k=100, top_p=0.8, seed=42, max_tokens=20)
    info = {
        "req_id": "r",
        "sampling_params": params,
        "duplex": {"data_plane": True, "session_id": "s", "seq": 0, "payload": {}},
    }
    model = mocker.Mock(
        spec=MiniCPMO45OmniForConditionalGeneration,
        _mrv2_duplex_infos=[info],
        prepare_duplex_sampling=mocker.Mock(),
        _minicpmo45_native_duplex_token_ids=lambda: {},
        _minicpmo45_chunk_terminator_token_ids=lambda _: {7},
        _sample_minicpmo45_native_duplex_rows=mocker.Mock(return_value=[7]),
        _sample_minicpmo45_native_duplex_rows_deferred=mocker.Mock(return_value=None),
        _record_minicpmo45_duplex_terminator=mocker.Mock(),
    )
    base = mocker.Mock(spec=Sampler, return_value="sampled")
    sampler = MiniCPMO45DuplexSampler(base, model)
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=1,
        idx_mapping_np=np.array([5]),
        num_computed_prefill_tokens_np=np.array([0]),
        num_scheduled_tokens=np.array([2]),
        prefill_len_np=np.array([4]),
    )
    assert sampler(torch.zeros(1, 10), batch) == "sampled"
    model.prepare_duplex_sampling.assert_not_called()
    batch.num_computed_prefill_tokens_np[0] = 2
    sampler(torch.zeros(1, 10), batch)
    model._sample_minicpmo45_native_duplex_rows.assert_called_once()
    # A lookahead after the terminator does not mutate the session policy.
    sampler(torch.zeros(1, 10), batch)
    model._sample_minicpmo45_native_duplex_rows.assert_called_once()
    info["duplex"]["seq"] = 1
    sampler(torch.zeros(1, 10), batch)
    assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    assert model.prepare_duplex_sampling.call_args.args[2][0].row_idx == 0
    assert sampler._requests["r"][0] == 1
    generator = sampler._requests["r"][2]
    assert generator is sampler._generators["r"]
    # Preemption and slot reassignment preserve accepted history and RNG.
    torch.rand(1, generator=generator)
    rng_state = generator.get_state()
    history = sampler._requests["r"][1]
    batch.idx_mapping_np[0] = 2
    sampler.add_request(2, params)
    sampler(torch.zeros(1, 10), batch)
    assert sampler._requests["r"][2] is generator
    assert sampler._requests["r"][1] is history
    assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    assert torch.equal(generator.get_state(), rng_state)
    sampler.on_requests_finished({"r"})
    assert not sampler._requests and not sampler._generators


def test_warmup_logprobs_delegate_but_duplex_diagnostics_fail_explicitly(mocker):
    params = SamplingParams(logprobs=1)
    model = mocker.Mock(spec=MiniCPMO45OmniForConditionalGeneration, _mrv2_duplex_infos=[])
    base = mocker.Mock(spec=Sampler)
    sampler = MiniCPMO45DuplexSampler(base, model)
    sampler.add_request(0, params)
    base.add_request.assert_called_once_with(0, params)
    batch = mocker.Mock(spec=InputBatch, num_reqs=1)
    sampler(torch.zeros(1, 10), batch)
    base.assert_called_once()
    model._mrv2_duplex_infos = [{"sampling_params": params, "duplex": {"data_plane": True}}]
    with pytest.raises(ValueError, match="does not support output logprobs"):
        sampler(torch.zeros(1, 10), batch)


@pytest.mark.parametrize("diagnostics", [{"logprobs": 1}, {"logprob_token_ids": [7]}])
def test_duplex_plugin_rejects_output_logprobs_before_worker_submission(diagnostics):
    from vllm_omni.engine.duplex.plugin import validate_duplex_plugin_sampling
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

    plugin = MiniCPMO45DuplexPlugin(lambda *args: None)
    with pytest.raises(ValueError, match="does not support output logprobs"):
        validate_duplex_plugin_sampling(plugin, sampling_defaults=(SamplingParams(**diagnostics),))
    with pytest.raises(ValueError, match="does not support output logprobs"):
        plugin.configure_sampling_params(
            runtime_config={"duplex_stage_sampling_params": {"0": diagnostics}},
            defaults=(SamplingParams(),),
        )


@pytest.mark.parametrize("shared_session", [False, True])
def test_mixed_batch_preserves_sampling_params_and_shared_session_order(mocker, shared_session):
    params = [SamplingParams(temperature=t, top_k=k, top_p=p, seed=42) for t, k, p in ((0.0, 100, 0.8), (0.7, 17, 0.9))]
    infos = [
        {
            "req_id": f"r-{i}",
            "sampling_params": p,
            "duplex": {"data_plane": True, "seq": 0, "session_id": "s" if shared_session else f"s-{i}"},
        }
        for i, p in enumerate(params)
    ]
    policies = []

    def sample(_logits, metadata, *, row_idxs, token_ids, row_params):
        policies.append((list(row_idxs), metadata.all_greedy, row_params))
        return [7] * len(row_idxs)

    model = mocker.Mock(
        spec=MiniCPMO45OmniForConditionalGeneration,
        _mrv2_duplex_infos=infos,
        _minicpmo45_native_duplex_token_ids=lambda: {},
        _minicpmo45_chunk_terminator_token_ids=lambda _: {7},
        _sample_minicpmo45_native_duplex_rows=mocker.Mock(side_effect=sample),
        _sample_minicpmo45_native_duplex_rows_deferred=mocker.Mock(return_value=None),
    )
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=2,
        idx_mapping_np=np.array([5, 2]),
        num_computed_prefill_tokens_np=np.array([0, 0]),
        num_scheduled_tokens=np.array([2, 2]),
        prefill_len_np=np.array([2, 2]),
    )
    sampler = MiniCPMO45DuplexSampler(mocker.Mock(spec=Sampler), model)
    sampler(torch.zeros(2, 10), batch)
    # vLLM normalizes greedy parameters before the model sees them.
    expected_params = [(0.0, 0, 1.0), (0.7, 17, 0.9)]
    if shared_session:
        assert policies == [([0], False, expected_params), ([1], False, expected_params)]
        assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    else:
        assert policies == [([0, 1], False, expected_params)]
        model._sample_minicpmo45_native_duplex_rows.assert_called_once()
        call = model._sample_minicpmo45_native_duplex_rows.call_args
        assert call.kwargs["row_idxs"] == [0, 1]
        torch.testing.assert_close(call.args[1].temperature, torch.tensor([0.0, 0.7]))
    prepared = model.prepare_duplex_sampling.call_args.args[2]
    assert [(r.temperature, r.top_k, r.top_p) for r in prepared] == expected_params
    assert sampler._requests["r-0"][0] == 0
    assert sampler._requests["r-1"][0] == 0


@pytest.mark.parametrize("seeded", [True, False])
@pytest.mark.parametrize("deferred", [True, False])
def test_mrv2_policy_matches_shared_v1_tokens_session_history_and_rng(mocker, seeded, deferred):
    token_ids = {
        "chunk_eos_token_id": 2,
        "chunk_tts_eos_token_id": 3,
        "turn_eos_token_id": 4,
        "tts_bos_token_id": 5,
        "listen_token_id": 6,
        "unit_token_id": 7,
    }

    class CharTokenizer:
        bad_token_ids = [8, 9]
        all_special_ids = list(token_ids.values())
        clean_up_tokenization_spaces = False

        def __len__(self):
            return 32

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(97 + i % 26) for i in ids)

        def batch_decode(self, batch, skip_special_tokens=False):
            return [self.decode(ids, skip_special_tokens) for ids in batch]

    tokenizer = CharTokenizer()

    def make_model():
        model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.model_stage = "llm"
        model._minicpmo45_native_duplex_token_ids_cache = token_ids
        model.max_new_speak_tokens_per_chunk = 20
        model.max_speak_chars_per_chunk = 6
        model._minicpmo45_tokenizer = lambda: tokenizer
        model._minicpmo45_duplex_row_sessions = {i: str(i) for i in range(8)}
        model._minicpmo45_duplex_row_payloads = {i: {} for i in range(8)}
        model._minicpmo45_duplex_row_max_tokens = {0: 1, 1: 5, 2: 20}
        states = {
            str(i): _MiniCPMO45Stage0SessionState(str(i), generated_tokens=[10, 11], current_turn_ended=i % 2 == 0)
            for i in range(8)
        }
        model._minicpmo45_duplex_data_plane_helper = mocker.Mock(spec=MiniCPMO45Stage0DuplexRuntime, sessions=states)
        return model, states

    def metadata():
        return mocker.Mock(
            spec=SamplingMetadata,
            output_token_ids=[[] for _ in range(8)],
            generators={i: torch.Generator().manual_seed(41 + i) for i in range(8)} if seeded else {},
            temperature=torch.tensor([0, 0.7, 0.9, 0, 0.3, 0.8, 0, 1.0]),
            top_k=torch.tensor([4, 8, 0, 0, 4, 12, 5, 3]),
            top_p=torch.tensor([1.0, 0.8, 1.0, 0.4, 0.75, 0.9, 0.85, 0.5]),
            all_greedy=False,
        )

    scalar, scalar_states = make_model()
    batched, batch_states = make_model()
    left, right = metadata(), metadata()
    # Compare the registered V2 policy against the shared V1 implementation.
    # Main's candidate/boundary draws replace the previous multinomial stream.
    if not deferred:
        mocker.patch.object(scalar, "_sample_minicpmo45_native_duplex_rows_deferred", return_value=None)
        mocker.patch.object(batched, "_sample_minicpmo45_native_duplex_rows_deferred", return_value=None)
    params = [
        SamplingParams(
            temperature=float(right.temperature[i]),
            top_k=int(right.top_k[i]) or -1,
            top_p=float(right.top_p[i]),
            seed=41 + i if seeded else None,
            max_tokens={0: 1, 1: 5}.get(i, 20),
        )
        for i in range(8)
    ]
    infos = [
        dict(
            req_id=f"r-{i}",
            sampling_params=params[i],
            duplex=dict(data_plane=True, seq=0, session_id=str(i), payload=dict(is_speech=True)),
        )
        for i in range(8)
    ]
    batched._mrv2_duplex_infos = infos
    scalar._minicpmo45_duplex_row_payloads = {i: dict(is_speech=True) for i in range(8)}
    base = mocker.Mock(spec=Sampler, side_effect=lambda logits, _batch: logits.argmax(-1).tolist())
    sampler = MiniCPMO45DuplexSampler(base, batched)
    sampler._requests = {f"r-{i}": (0, right.output_token_ids[i], right.generators.get(i)) for i in range(8)}
    sampler._generators = {f"r-{i}": g for i, g in right.generators.items()}
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=8,
        idx_mapping_np=np.arange(8),
        num_computed_prefill_tokens_np=np.ones(8),
        num_scheduled_tokens=np.ones(8),
        prefill_len_np=np.full(8, 2),
    )
    with torch.random.fork_rng(devices=[]):
        for step in range(32):
            torch.manual_seed(1000 + step)
            logits = torch.randn(8, 32)
            if step % 5 == 0:
                logits[2, 2] = 12.0  # Boundary draw skips the text draw.
            if step % 7 == 0:
                logits[3, :] = float("-inf")
                logits[3, 6] = 0.0  # Forced listen and ongoing-turn conversion.
            for histories in (left.output_token_ids, right.output_token_ids):
                for row in range(8):
                    if histories[row] and histories[row][-1] in token_ids.values():
                        histories[row] = []
                if step % 11 == 0:
                    histories[4] = list(range(10, 31))  # Token limit returns without a draw.
            # Advance the condition when a native boundary retired its history.
            for row in range(8):
                info = infos[row]
                if sampler._requests[f"r-{row}"][1] is not right.output_token_ids[row]:
                    info["duplex"]["seq"] += 1
                    sampler._requests[f"r-{row}"] = (
                        info["duplex"]["seq"],
                        right.output_token_ids[row],
                        right.generators.get(row),
                    )
            torch.manual_seed(5000 + step)
            expected = (
                scalar._sample_minicpmo45_native_duplex_stage0(logits.clone(), left, duplex_rows=list(range(8)))
                .sampled_token_ids.reshape(-1)
                .tolist()
            )
            scalar._commit_minicpmo45_duplex_pending_samples()
            # Boundary-hit rewinds are part of the accepted RNG state.
            expected_global_rng = torch.random.get_rng_state()
            torch.manual_seed(5000 + step)
            actual = sampler(logits.clone(), batch)
            assert (sampler._pending_history is not None) is (deferred and seeded)
            sampler._finish_deferred_history()
            assert actual == expected
            assert torch.equal(torch.random.get_rng_state(), expected_global_rng)
            for row in range(8):
                left.output_token_ids[row].append(expected[row])
                assert right.output_token_ids[row] == left.output_token_ids[row]
                assert scalar_states[str(row)].generated_tokens == batch_states[str(row)].generated_tokens
                assert (
                    scalar_states[str(row)].pending_terminator_token == batch_states[str(row)].pending_terminator_token
                )
                assert scalar_states[str(row)].current_turn_ended == batch_states[str(row)].current_turn_ended
                if seeded:
                    assert torch.equal(left.generators[row].get_state(), right.generators[row].get_state())


def test_deferred_policy_history_fences_finished_or_replaced_requests(mocker):
    model = mocker.Mock(spec=MiniCPMO45OmniForConditionalGeneration)
    sampler = MiniCPMO45DuplexSampler(mocker.Mock(spec=Sampler), model)
    old_history: list[int] = []
    live_history: list[int] = []
    successor = [9]
    pending = _MiniCPMO45PendingSamples(
        host=torch.tensor([[1, 1, 0], [2, 2, 0], [3, 3, 0]]),
        event=None,
        row_idxs=[0, 1, 2],
        stage2=[True] * 3,
        rewinds={},
        token_ids={},
        row_sessions=None,
        row_payloads=None,
    )
    sampler._requests = {"old": (1, successor, None), "live": (0, live_history, None), "finished": (0, [], None)}
    sampler._pending_history = (
        pending,
        [("old", old_history, 0), ("live", live_history, 1), ("finished", sampler._requests["finished"][1], 2)],
    )
    sampler.on_requests_finished({"finished"})
    sampler._finish_deferred_history()
    assert old_history == [] and successor == [9] and live_history == [2]
    assert "finished" not in sampler._requests and sampler._pending_history is None
    model._commit_minicpmo45_duplex_pending_samples.assert_called_once()
    sampler._finish_deferred_history()
    assert live_history == [2]
    model._commit_minicpmo45_duplex_pending_samples.assert_called_once()


def test_reanchor_uses_mrv2_slot_and_post_compaction_blocks(mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import window_kv

    rotate = mocker.patch.object(window_kv, "rotate_cached_keys")
    mocker.patch.object(window_kv.MiniCPMO45DuplexWorkerHelper, "get_rope_inv_freq", return_value=torch.ones(2))
    command = {"moved_from": 32, "delta": 16, "sink_blocks": 1, "old_computed_tokens": 48}
    buffers = [{}, {}, {"duplex": {"stage0_reanchor": command}}]
    intermediate = OmniIntermediateBuffer(3)
    intermediate.buffers = buffers
    runner = mocker.Mock(
        spec=OmniGPUModelRunner,
        req_states=mocker.Mock(
            spec=RequestState, req_id_to_index={"r": 2}, num_computed_tokens_np=np.array([0, 0, 32])
        ),
        model_state=mocker.Mock(spec=OmniModelState, intermediate_buffer=intermediate),
        block_tables=mocker.Mock(
            num_blocks=mocker.Mock(np=np.array([[0, 0, 2]])),
            block_tables=[mocker.Mock(gpu=torch.tensor([[0, 0], [0, 0], [4, 9]]))],
        ),
        model=object(),
        device="cpu",
        cache_config=mocker.Mock(block_size=16),
        kv_caches=[torch.zeros(1)],
    )
    schedule = mocker.Mock(spec=SchedulerOutput, num_scheduled_tokens={"r": 1}, scheduled_new_reqs=[])
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    torch.testing.assert_close(rotate.call_args.kwargs["block_ids"], torch.tensor([4, 9]))
    assert runner.req_states.num_computed_tokens_np[2] == 32
    assert "stage0_reanchor" not in buffers[2]["duplex"]
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    rotate.assert_called_once()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for staged block-table writes")
@torch.inference_mode()
def test_reanchor_reads_real_mrv2_staged_block_tables_after_overwrite(mocker):
    from vllm.v1.worker.gpu.block_table import BlockTables

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import window_kv

    device = torch.device("cuda")
    blocks = BlockTables([16, 16], 3, 32, [4, 4], device, [16, 16])
    # Slot 2 belongs to this request even when it is the only scheduled row.
    blocks.append_block_ids(2, ([3, 5, 9], [1, 6, 8]), overwrite=True)
    blocks.apply_staged_writes()
    blocks.append_block_ids(2, ([3, 9], [1, 8]), overwrite=True)
    blocks.apply_staged_writes()
    original = torch.arange(10 * 2 * 16 * 8, device=device, dtype=torch.float32).reshape(10, 2, 16, 8) / 1000
    kv_caches = [original.clone(), original.clone()]
    command = {"moved_from": 32, "delta": 16, "sink_blocks": 1, "old_computed_tokens": 48}
    intermediate = OmniIntermediateBuffer(3)
    intermediate.buffers[2] = {"duplex": {"stage0_reanchor": command}}
    runner = mocker.Mock(
        spec=OmniGPUModelRunner,
        req_states=mocker.Mock(
            spec=RequestState, req_id_to_index={"r": 2}, num_computed_tokens_np=np.array([0, 0, 32])
        ),
        block_tables=blocks,
        model_state=mocker.Mock(spec=OmniModelState, intermediate_buffer=intermediate),
        device=device,
        cache_config=mocker.Mock(block_size=16),
        kv_caches=kv_caches,
        kv_cache_group_ids=[0, 1],
        model=object(),
        _duplex_inv_freq=torch.tensor([1.0, 0.01], device=device),
    )
    schedule = mocker.Mock(spec=SchedulerOutput, num_scheduled_tokens={"r": 1}, scheduled_new_reqs=[])
    for group, expected_ids in enumerate(([3, 9], [1, 8])):
        row = window_kv.MiniCPMO45DuplexWorkerHelper.resolve_group_block_ids(runner, "r", 0, group)
        torch.testing.assert_close(row, torch.tensor(expected_ids, device=device, dtype=torch.int32))
        assert row.data_ptr() == blocks.block_tables[group].gpu[2].data_ptr()
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    for cache, retained in zip(kv_caches, (9, 8), strict=True):
        expected = original.clone()
        expected[retained, :, :, :4] = window_kv.rotate_keys(
            original[retained, :, :, :4].transpose(0, 1), 16, runner._duplex_inv_freq
        ).transpose(0, 1)
        torch.testing.assert_close(cache, expected)
    assert runner.req_states.num_computed_tokens_np[2] == 32
    assert "stage0_reanchor" not in intermediate.buffers[2]["duplex"]
    snapshots = [cache.clone() for cache in kv_caches]
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    for actual, expected in zip(kv_caches, snapshots, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device,cuda_platform", [("cpu", True), ("cpu", False), ("cuda", False)])
def test_sampler_warmups_skip_without_cuda(mocker, monkeypatch, device, cuda_platform):
    from types import SimpleNamespace

    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu.sample import gumbel

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: cuda_platform)
    gumbel_sample = mocker.patch.object(gumbel, "gumbel_sample")
    top_k_top_p = mocker.patch.object(topk_topp_sampler, "apply_top_k_top_p")
    base = SimpleNamespace(req_states=SimpleNamespace(device=torch.device(device), max_num_reqs=4, vocab_size=8))
    model = mocker.Mock(spec=MiniCPMO45OmniForConditionalGeneration)
    MiniCPMO45DuplexSampler(base, model).warmup()
    codec = object.__new__(MiniCPMO45SeededCodecSampler)
    codec.base_sampler = base
    codec.warmup()
    gumbel_sample.assert_not_called()
    top_k_top_p.assert_not_called()


@pytest.mark.parametrize(
    "max_rows,num_sm,expected",
    [
        (1, 132, [1]),
        (3, 132, [1, 2, 3]),
        # H200 (132 SMs): split counts 32/16/8 for 1-4/5-8/9-16 rows; 16 is also % 16.
        (16, 132, [1, 2, 3, 5, 9, 16]),
        # Split counts 4/2 for 17-33/34-64 rows, then only the monolithic kernel.
        (80, 132, [1, 2, 3, 5, 9, 16, 17, 32, 34, 48, 65, 80]),
    ],
)
def test_top_k_top_p_warmup_covers_each_triton_specialization(monkeypatch, max_rows, num_sm, expected):
    from vllm.utils import platform_utils

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import mrv2

    monkeypatch.setattr(platform_utils, "num_compute_units", lambda device_index=None: num_sm)
    assert mrv2._top_k_top_p_warmup_batches(max_rows, torch.device("cuda", 0)) == expected
