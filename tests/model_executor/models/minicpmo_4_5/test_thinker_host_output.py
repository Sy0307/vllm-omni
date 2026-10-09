# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 Thinker output host path: same llm2tts inputs with less per-step work."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.pipeline  # noqa: F401  (accumulation registrations)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_processor import OmniRequestState
from vllm_omni.worker_v2 import omni_ar_model_runner as runner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model():
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model.model = torch.nn.Module()
    return model


def _batch(n):
    return SimpleNamespace(input_ids=torch.arange(n), positions=torch.arange(n))


def _info(req_id, prompt):
    return {
        "req_id": req_id,
        "duplex": {"duplex_prompt_token_ids": prompt, "special_token_ids": {"listen_token_id": 7}},
    }


def test_prompt_snapshot_is_emitted_once_per_append_and_meta_tensors_are_shared():
    model = _model()
    first, other = [1, 2, 3], [9]
    infos = [_info("a", first), _info("b", other)]
    out = model.make_omni_output_mrv2(
        torch.zeros(2, 4), input_batch=_batch(2), req_states=None, model_intermediate_buffer=infos
    ).multimodal_outputs
    assert out["duplex_prompt_token_ids"] == [first, other]
    assert out["duplex_prompt_token_ids"][0] is not first  # never the buffer's own list
    listen = out["meta"]["listen_token_id"]
    assert listen[0] is listen[1] and listen[0].tolist() == [7] and listen[0].device.type == "cpu"
    # Decode steps of the same append re-send nothing.
    out = model.make_omni_output_mrv2(
        torch.zeros(2, 4), input_batch=_batch(2), req_states=None, model_intermediate_buffer=infos
    ).multimodal_outputs
    assert "duplex_prompt_token_ids" not in out
    # The next append stores a new snapshot: only that row emits.
    infos[0]["duplex"]["duplex_prompt_token_ids"] = [1, 2, 3, 4]
    out = model.make_omni_output_mrv2(
        torch.zeros(2, 4), input_batch=_batch(2), req_states=None, model_intermediate_buffer=infos
    ).multimodal_outputs
    assert out["duplex_prompt_token_ids"] == [[1, 2, 3, 4], None]
    model.on_requests_finished({"a"})
    assert "a" not in model._mrv2_emitted_duplex_prompts


def test_front_end_keeps_the_latest_prompt_snapshot():
    def accumulate(steps):
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type, state.mm_accumulated = None, MultimodalPayload()
        for prompt in steps:
            payload = {"latent": torch.ones(1, 2)}
            if prompt is not None:
                payload["duplex_prompt_token_ids"] = torch.tensor(prompt)
            state.add_multimodal_tensor(payload, "latent")
            state._consolidate_multimodal_tensors()
        return state.mm_accumulated

    every_step = accumulate([[5, 6], [5, 6], [5, 6, 7, 8], [5, 6, 7, 8]])
    once = accumulate([[5, 6], None, [5, 6, 7, 8], None])
    assert every_step["duplex_prompt_token_ids"].tolist() == once["duplex_prompt_token_ids"].tolist() == [5, 6, 7, 8]
    assert torch.equal(every_step["latent"], once["latent"])


def test_token_id_lists_copy_without_per_item_recursion():
    ids = list(range(50))
    copied = runner._async_copy_mm_value(ids)
    assert copied == ids and copied is not ids
    sliced = runner._slice_pooler_value([ids, [1]], req_index=0, start=0, end=1, total_tokens=2)
    assert sliced == ids and sliced is not ids
    mixed = [1, torch.tensor([2])]
    assert runner._async_copy_mm_value(mixed)[1] is not mixed[1]


def test_aliased_latent_reuses_the_hidden_host_copy(monkeypatch):
    class _No:
        def __getattr__(self, _):
            return lambda *a, **k: None

    monkeypatch.setattr(torch.cuda, "set_stream", lambda *_: None)
    hidden = torch.randn(3, 4)
    mm = {"latent": hidden, "latent_input_ids": torch.arange(3).reshape(-1, 1), "meta": {"x": [torch.tensor([1])] * 2}}
    batch = SimpleNamespace(
        query_start_loc_np=np.array([0, 2, 3], dtype=np.int32),
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        num_reqs=2,
        num_tokens_after_padding=3,
    )
    from vllm_omni.outputs import OmniModelRunnerOutput

    output = runner.OmniAsyncOutput(
        model_runner_output=OmniModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={},
            sampled_token_ids=None,
            prompt_logprobs_dict={},
            prompt_token_id_logprobs_dict={},
            kv_connector_output=None,
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.zeros(2, 1, dtype=torch.long),
            num_rejected=torch.zeros(2),
            sampling_mask_tensors=None,
            logprobs_tensors=None,
            num_nans=None,
        ),
        num_sampled_tokens=torch.ones(2, dtype=torch.int32),
        main_stream=_No(),
        copy_stream=_No(),
        copy_event=_No(),
        text_hidden=hidden,
        multimodal_outputs=mm,
        input_batch=batch,
    )
    assert output._mm_cpu["latent"] is output._hidden_cpu
    assert list(output._mm_cpu) == list(mm)
    rows = output.get_output().multimodal_outputs
    assert [list(row) for row in rows] == [["hidden", "latent", "latent_input_ids", "meta.x"]] * 2
    assert torch.equal(rows[0]["latent"], hidden[:2]) and torch.equal(rows[1]["hidden"], hidden[2:])
