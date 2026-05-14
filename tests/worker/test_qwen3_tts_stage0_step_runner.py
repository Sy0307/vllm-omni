from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.qwen3_tts_stage0_step_runner import (
    Qwen3TTSSlotTable,
    Qwen3TTSStage0StepRunner,
    flatten_codec_frames_for_code2wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_tts_slot_table_allocates_and_reuses_slots():
    table = Qwen3TTSSlotTable(
        max_slots=2,
        hidden_size=4,
        num_quantizers=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    s0 = table.allocate("r1")
    s1 = table.allocate("r2")
    assert s0 != s1
    assert table.allocate("r1") == s0

    table.free("r1")
    s2 = table.allocate("r3")
    assert s2 == s0
    assert "r1" not in table.req_to_slot
    assert table.slots[s2].req_id == "r3"


def test_qwen3_tts_slot_table_exhaustion_is_explicit():
    table = Qwen3TTSSlotTable(
        max_slots=1,
        hidden_size=4,
        num_quantizers=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table.allocate("r1")

    with pytest.raises(RuntimeError, match="slot table exhausted"):
        table.allocate("r2")


def _runner_config(async_chunk=True, model_stage="qwen3_tts", has_talker_mtp=True):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                async_chunk=async_chunk,
                model_stage=model_stage,
            )
        ),
        has_talker_mtp=has_talker_mtp,
    )


def test_stage0_step_runner_supports_decode_only_qwen3_tts_async_chunk():
    step_runner = Qwen3TTSStage0StepRunner(max_slots=4, hidden_size=8, num_quantizers=16)

    assert step_runner.supports_step(
        runner=_runner_config(),
        request_ids=["r1", "r2"],
        num_scheduled_tokens=[1, 1],
        is_prefill_by_req={"r1": False, "r2": False},
    )


def test_stage0_step_runner_rejects_prefill_or_wrong_stage():
    step_runner = Qwen3TTSStage0StepRunner(max_slots=4, hidden_size=8, num_quantizers=16)

    assert not step_runner.supports_step(
        runner=_runner_config(model_stage="code2wav"),
        request_ids=["r1"],
        num_scheduled_tokens=[1],
        is_prefill_by_req={"r1": False},
    )
    assert not step_runner.supports_step(
        runner=_runner_config(),
        request_ids=["r1"],
        num_scheduled_tokens=[1],
        is_prefill_by_req={"r1": True},
    )
    assert not step_runner.supports_step(
        runner=_runner_config(async_chunk=False),
        request_ids=["r1"],
        num_scheduled_tokens=[1],
        is_prefill_by_req={"r1": False},
    )


def test_stage0_step_runner_commits_next_embeds_and_codes():
    class FakeTalkerMTP:
        def __call__(self, input_ids, req_embeds, last_hidden, text_step, **kwargs):
            codes = torch.arange(input_ids.shape[0] * 16, dtype=torch.long).reshape(input_ids.shape[0], 16)
            return req_embeds + 10, codes

    runner = SimpleNamespace(
        talker_mtp=FakeTalkerMTP(),
        input_batch=SimpleNamespace(req_ids=["r1", "r2"]),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1], dtype=torch.int32)),
        model_intermediate_buffer={},
        requests={
            "r1": SimpleNamespace(additional_information_cpu=None),
            "r2": SimpleNamespace(additional_information_cpu=None),
        },
        model=SimpleNamespace(talker_mtp_output_key=("codes", "audio"), gpu_resident_buffer_keys=set()),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={})),
    )
    inputs_embeds = torch.zeros((2, 4), dtype=torch.float32)

    step_runner = Qwen3TTSStage0StepRunner(max_slots=2, hidden_size=4, num_quantizers=16)
    prepared = step_runner.prepare_step(
        request_ids=["r1", "r2"],
        runner=runner,
        input_ids=torch.tensor([101, 102], dtype=torch.long),
        req_embeds=torch.ones((2, 4), dtype=torch.float32),
        last_talker_hidden=torch.ones((2, 4), dtype=torch.float32) * 2,
        text_step=torch.ones((2, 4), dtype=torch.float32) * 3,
    )
    step_runner.run_step(prepared=prepared, runner=runner)
    step_runner.commit_step(prepared=prepared, runner=runner, inputs_embeds=inputs_embeds)

    assert torch.equal(inputs_embeds, torch.ones((2, 4), dtype=torch.float32) * 11)
    assert torch.equal(runner.model_intermediate_buffer["r1"]["codes"]["audio"], torch.arange(16).reshape(1, 16))
    assert torch.equal(runner.model_intermediate_buffer["r2"]["codes"]["audio"], torch.arange(16, 32).reshape(1, 16))
    assert runner.requests["r1"].additional_information_cpu is runner.model_intermediate_buffer["r1"]


def test_stage0_step_runner_records_fast_path_and_fallback_counts():
    step_runner = Qwen3TTSStage0StepRunner(max_slots=2, hidden_size=4, num_quantizers=16)

    step_runner.record_fast_path(batch_size=2)
    step_runner.record_fallback("prefill")

    assert step_runner.stats.fast_path_steps == 1
    assert step_runner.stats.fast_path_requests == 2
    assert step_runner.stats.fallback_reasons["prefill"] == 1


def test_qwen3_tts_slot_codec_frames_match_legacy_flattening():
    frames_fq = torch.arange(4 * 16, dtype=torch.long).reshape(4, 16)
    legacy_flat = frames_fq.transpose(0, 1).contiguous().reshape(-1)

    assert torch.equal(flatten_codec_frames_for_code2wav(frames_fq), legacy_flat)
