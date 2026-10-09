# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-side Talker trims: admission copies, condition constants and graph gating."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _CodecWindowPenaltiesState,
    _MiniCPMTTSProjector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("base_applies", [False, True])
def test_codec_penalty_admission_hands_upstream_only_penalty_fields(mocker, base_applies):
    state = object.__new__(_CodecWindowPenaltiesState)
    # The stock state's return value decides; its own array is not consulted.
    state.base = mocker.Mock(use_penalty=np.zeros(4, dtype=bool))
    state.base.add_request.return_value = base_applies
    state.repetition_penalty = SimpleNamespace(np=np.ones(4, dtype=np.float32))
    state.use_window = np.zeros(4, dtype=bool)
    state.use_penalty = np.zeros(4, dtype=bool)
    state.prefix_history = torch.zeros((4, 16), dtype=torch.long)
    extra = {"payload": list(range(1000))}
    params = SamplingParams(repetition_penalty=1.05, frequency_penalty=0.25, seed=7, extra_args=extra)
    assert state.add_request(1, params)
    (slot, handed), _ = state.base.add_request.call_args
    assert slot == 1
    # Only the three fields PenaltiesState reads; no copy of request payloads.
    assert vars(handed) == {"repetition_penalty": 1.0, "frequency_penalty": 0.25, "presence_penalty": 0.0}
    assert params.repetition_penalty == 1.05
    assert state.use_window[1] and state.prefix_history[1].tolist() == [-1] * 16
    # Without the window, admission reports exactly what the stock state applies.
    assert state.add_request(2, SamplingParams(repetition_penalty=1.0)) is base_applies
    assert not state.use_window[2]


def _condition_talker():
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    torch.manual_seed(0)
    talker.emb_text = torch.nn.Embedding(12, 6)
    talker.projector_semantic = _MiniCPMTTSProjector(5, 6)
    talker._normalize = True
    talker._text_eos_id = 9
    talker._tts_bos_id = 4
    return talker


@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_condition_constants_match_embedding_lookup(native):
    talker = _condition_talker()
    ids = torch.tensor([1, 2, 3])
    hidden = torch.randn(3, 5)
    boundary = talker.emb_text(torch.tensor([9, 4]))
    assert torch.equal(talker._boundary_embeddings(), boundary)
    condition = talker.emb_text(ids) + torch.nn.functional.normalize(talker.projector_semantic(hidden), p=2, dim=-1)
    tail = talker.emb_text(torch.tensor([4])) if native else boundary
    expected = torch.cat([condition, tail])
    actual = talker._build_condition_embeddings(ids, hidden, native_duplex=native)
    assert torch.equal(actual, expected)
    empty = talker._build_condition_embeddings(torch.empty(0, dtype=torch.long), torch.empty(0, 5))
    assert torch.equal(empty, boundary)
    # The returned tensors own their storage: callers may write them.
    assert empty.data_ptr() != talker.emb_text.weight.data_ptr()


def _graphs(monkeypatch, mocker, slots: int = 4, vocab: int = 32):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import mrv2

    monkeypatch.setattr(mrv2, "LegacySampler", mocker.Mock())

    base = mocker.Mock()
    base.sampling_states = SimpleNamespace(
        vocab_size=vocab,
        temperature=SimpleNamespace(np=np.full(slots, 0.8, dtype=np.float32)),
        min_p=SimpleNamespace(np=np.zeros(slots, dtype=np.float32)),
        top_k=SimpleNamespace(np=np.full(slots, 25, dtype=np.int32)),
        top_p=SimpleNamespace(np=np.full(slots, 0.85, dtype=np.float32)),
        num_logprobs=np.full(slots, -1, dtype=np.int32),
    )
    base.needs_logits_processing = np.ones(slots, dtype=bool)
    base.logits_processors = [
        SimpleNamespace(
            use_logit_bias=np.zeros(slots, dtype=bool),
            num_stop_token_ids=SimpleNamespace(np=np.zeros(slots, dtype=np.int32)),
            restore_when_all_masked=SimpleNamespace(np=np.zeros(slots, dtype=np.int32)),
        ),
        SimpleNamespace(base=SimpleNamespace(use_penalty=np.zeros(slots, dtype=bool))),
        SimpleNamespace(num_bad_words=SimpleNamespace(np=np.zeros(slots, dtype=np.int32))),
    ]
    base.thinking_budget_state = SimpleNamespace(enabled=False)
    base.num_speculative_tokens = 1
    base.req_states = SimpleNamespace(max_num_reqs=slots, index_to_req_id={s: f"r{s}" for s in range(slots)})
    model = mocker.Mock(spec=MiniCPMO45OmniTTSForConditionalGeneration)
    model._mrv2_output_infos = [{"native_duplex": True}]
    sampler = mrv2.MiniCPMO45SeededCodecSampler(base, model)
    assert sampler.decode_graphs is None  # opt-in only
    graphs = mrv2.SeededCodecDecodeGraphs(sampler)
    sampler.decode_graphs = graphs
    return sampler, graphs, base


@pytest.mark.parametrize(
    "change",
    [
        None,
        ("seed", None),
        ("temperature", 0.0),
        ("min_p", 0.1),
        ("top_k", 32),
        ("top_p", 1.0),
        ("logprobs", 2),
        ("logit_bias", True),  # replayed (min_tokens / allowed ids / bias): still accepted
        ("stop_restore", 1),  # structured-output restore variant is not replayed
        ("upstream_penalty", True),
        ("bad_words", 1),
    ],
)
def test_decode_graph_admission_signature(monkeypatch, mocker, change):
    sampler, graphs, base = _graphs(monkeypatch, mocker)
    states = base.sampling_states
    seed = 3
    if change is not None:
        name, value = change
        if name == "seed":
            seed = value
        elif name == "temperature":
            states.temperature.np[1] = value
        elif name == "min_p":
            states.min_p.np[1] = value
        elif name == "top_k":
            states.top_k.np[1] = value
        elif name == "top_p":
            states.top_p.np[1] = value
        elif name == "logprobs":
            states.num_logprobs[1] = value
        elif name == "logit_bias":
            base.logits_processors[0].use_logit_bias[1] = value
            base.logits_processors[0].num_stop_token_ids.np[1] = 1
        elif name == "stop_restore":
            base.logits_processors[0].restore_when_all_masked.np[1] = value
        elif name == "upstream_penalty":
            base.logits_processors[1].base.use_penalty[1] = value
        elif name == "bad_words":
            base.logits_processors[2].num_bad_words.np[1] = value
    temperature = 0.0 if change == ("temperature", 0.0) else 0.8
    sampler.add_request(1, SamplingParams(seed=seed, temperature=temperature))
    base.add_request.assert_called_once()
    assert bool(graphs.slot_ok[1]) is (change is None or change[0] == "logit_bias")


def test_decode_graph_declines_steps_it_did_not_capture(monkeypatch, mocker):
    sampler, graphs, base = _graphs(monkeypatch, mocker)
    sampler.add_request(1, SamplingParams(seed=3, temperature=0.8))
    batch = SimpleNamespace(
        idx_mapping_np=np.array([1]),
        expanded_idx_mapping=torch.zeros(1, dtype=torch.int32),
        num_computed_prefill_tokens_np=np.array([4]),
        num_scheduled_tokens=np.array([1]),
        prefill_len_np=np.array([4]),
    )
    logits = torch.zeros(1, 32)
    flags = torch.zeros(1, dtype=torch.bool)
    # Nothing captured for this size.
    assert graphs.try_sample(logits, batch, flags, flags) is None
    graphs.graphs[1] = mocker.Mock()
    # Warmup steps carry no EOS controls; other dtypes/vocab sizes stay eager.
    assert graphs.try_sample(logits, batch, None, flags) is None
    assert graphs.try_sample(logits, batch, flags, None) is None
    assert graphs.try_sample(logits.double(), batch, flags, flags) is None
    assert graphs.try_sample(torch.zeros(1, 31), batch, flags, flags) is None
    # A partial prefill row is not accepted this step.
    partial = SimpleNamespace(**{**vars(batch), "num_computed_prefill_tokens_np": np.array([2])})
    assert graphs.try_sample(logits, partial, flags, flags) is None
    # An unadmitted slot.
    other = SimpleNamespace(**{**vars(batch), "idx_mapping_np": np.array([2])})
    assert graphs.try_sample(logits, other, flags, flags) is None
    graphs.graphs[1].graph.replay.assert_not_called()
    assert not sampler._generators


def test_fused_decode_output_kernel_matches_eager_under_triton_interpreter():
    """The fused decode-output kernel against the eager Torch path (CPU, interpreted)."""
    pytest.importorskip("triton")
    script = Path(__file__).with_name("_talker_decode_output_interp.py")
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, str(script), "60"], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "OK:" in result.stdout
