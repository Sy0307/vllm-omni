# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 penalties must count generated speech tokens only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


class _Uva:
    def copy_to_uva(self) -> None:
        pass


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_generated_only_penalty_writes_skip_prompt_and_keep_resumed_outputs():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _generated_only_penalty_writes

    vocab, rows, width = 6761, 3, 16
    tokens = torch.zeros(rows, width, dtype=torch.int32, device="cuda")
    # Row 0: fresh request whose prompt holds text ids beyond the speech head.
    tokens[0, :4] = torch.tensor([151645, 12, 7, 99999])
    # Row 1: resumed request, prompt [5, 6] then generated [7, 7, 9].
    tokens[1, :5] = torch.tensor([5, 6, 7, 7, 9])
    prompt_len = torch.tensor([4, 2, 0], dtype=torch.int32, device="cuda")
    prefill_len = torch.tensor([4, 5, 0], dtype=torch.int32)
    state = SimpleNamespace(
        _new_penalties_reqs=[0, 1],
        device=torch.device("cuda"),
        prompt_bin_mask=torch.full((rows, (vocab + 31) // 32), -1, dtype=torch.int32, device="cuda"),
        output_bin_counts=torch.full((rows, vocab), 3, dtype=torch.int32, device="cuda"),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=tokens),
            prompt_len=SimpleNamespace(gpu=prompt_len),
            prefill_len=SimpleNamespace(gpu=prefill_len.cuda(), np=np.asarray(prefill_len)),
        ),
        repetition_penalty=_Uva(),
        frequency_penalty=_Uva(),
        presence_penalty=_Uva(),
    )
    _generated_only_penalty_writes(state)
    torch.accelerator.synchronize()
    assert state._new_penalties_reqs == []
    # No prompt token is penalized for the new requests.
    assert int(state.prompt_bin_mask[:2].abs().sum()) == 0
    counts = state.output_bin_counts
    assert int(counts[0].sum()) == 0
    assert counts[1, 7].item() == 2 and counts[1, 9].item() == 1 and int(counts[1].sum()) == 3
    # Rows that were not added keep their state.
    assert int(counts[2, 0]) == 3 and int(state.prompt_bin_mask[2, 0]) == -1


class _States:
    def __init__(self, temperature, top_k, top_p, seeds):
        self.temperature = SimpleNamespace(gpu=temperature)
        self.seeds = SimpleNamespace(gpu=seeds)
        self._top_k, self._top_p = top_k, top_p

    def get_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np):
        rows = expanded_idx_mapping.long()
        return self._top_k[rows], self._top_p[rows]


def _ras_sampler(tokens, prompt_len, total_len, temperature, top_k, top_p, seeds):
    return SimpleNamespace(
        logit_bias_state=SimpleNamespace(apply_logit_bias=lambda *args: None),
        sampling_states=_States(temperature, top_k, top_p, seeds),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=tokens),
            prompt_len=SimpleNamespace(gpu=prompt_len),
            total_len=SimpleNamespace(gpu=total_len),
        ),
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ras_mrv2_sample_rejects_recent_repeats_on_device():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _ras_mrv2_sample

    dev, vocab, rows = "cuda", 64, 4
    logits = torch.full((rows, vocab), -4.0, device=dev)
    logits[:, 7] = 8.0  # token 7 dominates every row
    tokens = torch.zeros(rows, 32, dtype=torch.int32, device=dev)
    # Row 0: token 7 generated recently -> rejected. Row 1: 7 only in the prompt.
    # Row 2: no output yet. Row 3: greedy, never rejected.
    tokens[0, :6] = torch.tensor([1, 2, 3, 7, 4, 5])
    tokens[1, :3] = torch.tensor([7, 7, 9])
    tokens[3, :4] = torch.tensor([1, 7, 7, 7])
    prompt_len = torch.tensor([3, 2, 5, 1], dtype=torch.int32, device=dev)
    total_len = torch.tensor([6, 3, 5, 4], dtype=torch.int32, device=dev)
    temperature = torch.tensor([1.0, 1.0, 1.0, 0.0], device=dev)
    top_k = torch.full((rows,), 1, dtype=torch.int32, device=dev)
    top_p = torch.full((rows,), 0.8, device=dev)
    seeds = torch.arange(rows, dtype=torch.int64, device=dev)
    sampler = _ras_sampler(tokens, prompt_len, total_len, temperature, top_k, top_p, seeds)
    idx = torch.arange(rows, dtype=torch.int32, device=dev)
    sampled, _ = _ras_mrv2_sample(
        sampler,
        logits,
        idx,
        idx,
        np.arange(rows),
        torch.full((rows,), 11, dtype=torch.int64, device=dev),
        None,
        None,
        default_top_p=0.8,
        default_top_k=25,
        win_size=10,
        tau_r=0.1,
        eps=1e-5,
    )
    sampled = sampled.tolist()
    assert sampled[0] != 7 and 0 <= sampled[0] < vocab
    assert sampled[1:] == [7, 7, 7]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ras_mrv2_sample_draws_from_top_p_capped_by_top_k():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _ras_mrv2_sample

    dev, vocab, rows = "cuda", 32, 2048
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1] + [0.0] * (vocab - 4), device=dev) + 1e-9
    logits = probs.log().expand(rows, vocab).contiguous()
    tokens = torch.zeros(rows, 4, dtype=torch.int32, device=dev)
    no_history = torch.zeros(rows, dtype=torch.int32, device=dev)
    # Row halves: top-p 0.6 keeps {0, 1}; top-p 0.95 with top-k 3 keeps {0, 1, 2}.
    top_p = torch.where(torch.arange(rows, device=dev) < rows // 2, 0.6, 0.95).float()
    top_k = torch.full((rows,), 3, dtype=torch.int32, device=dev)
    sampler = _ras_sampler(
        tokens,
        no_history,
        no_history,
        torch.ones(rows, device=dev),
        top_k,
        top_p,
        torch.arange(rows, dtype=torch.int64, device=dev),
    )
    idx = torch.arange(rows, dtype=torch.int32, device=dev)
    sampled, _ = _ras_mrv2_sample(
        sampler,
        logits,
        idx,
        idx,
        np.arange(rows),
        torch.zeros(rows, dtype=torch.int64, device=dev),
        None,
        None,
        default_top_p=0.8,
        default_top_k=25,
        win_size=10,
        tau_r=0.1,
        eps=1e-5,
    )
    low, high = sampled[: rows // 2], sampled[rows // 2 :]
    assert set(low.tolist()) == {0, 1} and set(high.tolist()) == {0, 1, 2}
    # Kept tokens are renormalized: 0.4 / 0.7 of the top-p 0.6 rows pick token 0.
    assert abs((low == 0).float().mean().item() - 4 / 7) < 0.06
