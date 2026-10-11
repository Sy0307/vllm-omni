# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA: the Talker's fused decode output and graph-replayed seeded codec draw are bit-identical to eager."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_VOCAB = 6562
_EOS = _VOCAB - 1


def _talker():
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker._codec_eos_id = _EOS
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    return talker


def _setup(graph: bool, capacity: int, requests: list[tuple[str, int, list[int], dict]]):
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    reqs = RequestState(capacity, 512, 64, 0, _VOCAB, device)
    talker = _talker()
    base = Sampler(talker.vllm_config, capacity, _VOCAB, device, reqs)
    sampler, _ = talker.mrv2_custom_sampler(base)
    if not graph:
        talker._mrv2_seeded_codec_sampler.decode_graphs = None
    talker.capture_auxiliary_graphs()
    slots = [_admit(talker, sampler, reqs, *request) for request in requests]
    return talker, sampler, reqs, slots


def _admit(talker, sampler, reqs, req_id, prompt_len, token_ids, params, condition_seq=0):
    """Runner admission of one (re)scheduled Talker condition, with its staged writes."""
    from vllm import SamplingParams

    reqs.remove_request(req_id)
    reqs.add_request(req_id, prompt_len, token_ids, len(token_ids), 256)
    talker._request_condition_states[req_id] = {"condition_seq": condition_seq}
    slot = reqs.req_id_to_index[req_id]
    sampler.add_request(slot, SamplingParams(**params))
    reqs.apply_staged_writes()
    sampler.apply_staged_writes()
    return slot


def _batch(reqs, slots, step):
    device = torch.device("cuda")
    rows = len(slots)
    prefill = np.array([int(reqs.prefill_len.np[s]) for s in slots])
    idx = torch.tensor(slots, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(prefill + step + 1, dtype=torch.int32, device=device)
    return SimpleNamespace(
        num_reqs=rows,
        idx_mapping_np=np.array(slots),
        idx_mapping=idx,
        expanded_idx_mapping=idx,
        expanded_local_pos=torch.zeros(rows, dtype=torch.int32, device=device),
        positions=seq_lens.long() - 1,
        logits_indices=torch.arange(rows, device=device),
        input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
        cu_num_logits_np=np.arange(rows + 1),
        cu_num_logits=torch.arange(rows + 1, dtype=torch.int32, device=device),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_computed_prefill_tokens_np=prefill,
        num_scheduled_tokens=np.ones(rows, dtype=np.int32),
        prefill_len_np=prefill,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 2, 3, 5])
@pytest.mark.parametrize("min_tokens", [0, 50])
@torch.inference_mode()
def test_seeded_codec_graph_matches_eager_draws_and_generators(mocker, rows, min_tokens):
    capacity = 8
    rng = random.Random(rows)
    params: dict[str, object] = dict(temperature=0.8, top_k=25, top_p=0.85, repetition_penalty=1.05)
    params.update(min_tokens=min_tokens, stop_token_ids=[_EOS], max_tokens=256, detokenize=False)
    requests = []
    for i in range(rows):
        prompt = [rng.randrange(_VOCAB - 1) for _ in range(8 + i)]
        history = [rng.choice([3, 5, 7]) for _ in range(rng.randrange(0, 20))]
        requests.append((f"r{i}", len(prompt), prompt + history, dict(params, seed=40 + i)))
    eager = _setup(False, capacity, requests)
    graph = _setup(True, capacity, requests)
    graphs = graph[0]._mrv2_seeded_codec_sampler.decode_graphs
    assert graphs is not None and sorted(graphs.graphs) == list(range(1, capacity + 1))
    spy = mocker.spy(graphs, "try_sample")
    gen = torch.Generator(device="cuda").manual_seed(rows)
    onset = 0
    for step in range(84):
        if step == 30:
            update = ("r0", 30, list(range(30)) + [5, 5, 5], dict(params, seed=40, temperature=0.7, top_k=20))
            for setup in (eager, graph):
                talker, sampler, reqs, slots = setup
                slots[0] = _admit(talker, sampler, reqs, *update, condition_seq=1)
            onset = step
        logits = torch.randn((rows, _VOCAB), generator=gen, device="cuda") * 3
        logits[:, _EOS] += 6.0
        forced = torch.rand(rows, generator=gen, device="cuda") < 0.05
        mask = torch.full((rows,), step - onset < 6, dtype=torch.bool, device="cuda") & ~forced
        outputs = []
        for talker, sampler, reqs, slots in (eager, graph):
            talker._mrv2_output_infos = [{"native_duplex": True} for _ in range(rows)]
            talker._mrv2_forced_eos = forced.clone()
            talker._mrv2_mask_eos = mask.clone()
            out = sampler(logits.clone(), _batch(reqs, slots, step - onset))
            outputs.append(out)
            tokens = out.sampled_token_ids.view(-1).tolist()
            for slot, token in zip(slots, tokens):
                total = int(reqs.total_len.gpu[slot])
                reqs.all_token_ids.stage_write(slot, total, [token])
                reqs.total_len.stage_write_elem(slot, total + 1)
            reqs.apply_staged_writes()
        torch.testing.assert_close(outputs[1].sampled_token_ids, outputs[0].sampled_token_ids, rtol=0, atol=0)
        for i in range(rows):
            got = graph[0]._mrv2_seeded_codec_sampler._generators[f"r{i}"]
            want = eager[0]._mrv2_seeded_codec_sampler._generators[f"r{i}"]
            torch.testing.assert_close(got.get_state(), want.get_state(), rtol=0, atol=0)
    assert spy.call_count > 0 and all(value is not None for value in spy.spy_return_list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "unsupported",
    [
        dict(min_p=0.1),
        dict(frequency_penalty=0.3),
        dict(presence_penalty=0.3),
        dict(logprobs=1),
        dict(temperature=0.0),
        dict(top_k=-1, top_p=1.0),
    ],
)
@torch.inference_mode()
def test_seeded_codec_graph_declines_unsupported_rows(unsupported):
    params = dict(temperature=0.8, top_k=25, top_p=0.85, repetition_penalty=1.05, max_tokens=256)
    requests = [
        ("a", 4, [1, 2, 3, 4], dict(params, seed=1)),
        ("b", 4, [1, 2, 3, 4], dict(params, seed=2, **unsupported)),
    ]
    talker, sampler, reqs, slots = _setup(True, 4, requests)
    graphs = talker._mrv2_seeded_codec_sampler.decode_graphs
    talker._mrv2_output_infos = [{"native_duplex": True}, {"native_duplex": True}]
    logits = torch.randn((2, _VOCAB), device="cuda")
    batch = _batch(reqs, slots, 0)
    false = torch.zeros(2, dtype=torch.bool, device="cuda")
    assert graphs.try_sample(logits, batch, false, false) is None
    one = _batch(reqs, slots[:1], 0)
    talker._mrv2_output_infos = [{"native_duplex": True}]
    assert graphs.try_sample(logits[:1].contiguous(), one, false[:1], false[:1]) is not None
