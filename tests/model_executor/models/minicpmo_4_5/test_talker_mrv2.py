# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 path of MiniCPM-o 4.5's Talker: device-side codec output,
EOS control and the 16-frame codec penalty must match the V1 host path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _OFFLINE_CODEC_MAX_NEW_TOKENS,
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EOS = 6561


def _talker(max_reqs: int = 8, max_position_embeddings: int = 4096):
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker._codec_eos_id = _EOS
    talker._tts_config = SimpleNamespace(max_position_embeddings=max_position_embeddings)
    talker._mrv2_empty_speech = torch.zeros(max_reqs, dtype=torch.bool)
    talker._mrv2_forced_eos = None
    talker._mrv2_decode_rows_logged = False
    return talker


def _batch(rows: list[dict], *, pad_to: int | None = None):
    """rows: one dict per request with slot, prompt_len, span (token ids) and prefill flag."""
    ids = [token for row in rows for token in row["span"]]
    num_tokens = len(ids)
    padded = pad_to or num_tokens
    starts = np.cumsum([0] + [len(row["span"]) for row in rows])
    seq_lens = [row["computed"] + len(row["span"]) for row in rows]
    return SimpleNamespace(
        num_reqs=len(rows),
        has_prefill=any(row["prefill"] for row in rows),
        is_prefilling_np=np.array([row["prefill"] for row in rows]),
        idx_mapping_np=np.array([row["slot"] for row in rows]),
        idx_mapping=torch.tensor([row["slot"] for row in rows]),
        input_ids=torch.tensor(ids + [0] * (padded - num_tokens), dtype=torch.int32),
        logits_indices=torch.tensor(starts[1:] - 1),
        seq_lens=torch.tensor(seq_lens + [0] * 2, dtype=torch.int32),
        query_start_loc_np=starts,
        num_scheduled_tokens=np.diff(starts),
    ), padded


def _req_states(prompt_lens: dict[int, int], max_reqs: int = 8):
    prompt_len = torch.zeros(max_reqs, dtype=torch.int32)
    for slot, value in prompt_lens.items():
        prompt_len[slot] = value
    return SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt_len))


def test_mrv2_prefill_keeps_interleaved_session_conditions_and_codec_history_separate(mocker):
    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.zeros(2, 4))
    # One session advances while another starts. The V2 runner supplies only
    # req_id; falling back to a shared default key makes seq 3 follow seq 0.
    for request_id, seq in [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("a", 3)]:
        _, _, updates = talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            req_id=request_id,
            _omni_is_prefill=True,
            _omni_prompt_len=2,
            native_duplex=True,
            ids={"tts": torch.tensor([7])},
            hidden_states={"tts": torch.zeros(1, 4)},
            meta={"streaming_condition_seq": seq, "turn_start": seq == 0},
        )
        assert talker._request_audio_states[request_id] is updates["audio_state"]
        updates["audio_state"]["recent_codes"] = [11 if request_id == "a" else 22]
    assert talker._request_condition_states["a"]["condition_seq"] == 3
    assert talker._request_condition_states["b"]["condition_seq"] == 0
    assert talker._request_audio_states["a"]["recent_codes"] == [11]
    assert talker._request_audio_states["b"]["recent_codes"] == [22]


@pytest.mark.parametrize("session_mode", ["turn", "duplex"])
@pytest.mark.parametrize("warmup_fails", [False, True])
def test_worker_readiness_reaches_wrapped_talker_and_propagates_failure(mocker, session_mode, warmup_fails):
    from transformers import LlamaConfig, PretrainedConfig
    from vllm.config import VllmConfig

    from vllm_omni.config.model import OmniModelConfig
    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni as omni
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler
    from vllm_omni.worker.base import OmniGPUWorkerBase
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._codec_eos_id = _EOS
    talker._num_audio_tokens = _EOS + 1
    talker._tts_config = LlamaConfig(max_position_embeddings=4096)
    talker._mrv2_empty_speech = torch.zeros(8, dtype=torch.bool)
    talker.make_empty_intermediate_tensors = lambda: None
    sampler = mocker.Mock(spec=MiniCPMO45SeededCodecSampler)
    if warmup_fails:
        sampler.warmup.side_effect = RuntimeError("codec priming failed")
    talker._mrv2_seeded_codec_sampler = sampler
    config = VllmConfig()
    config.model_config = mocker.Mock(
        spec=OmniModelConfig,
        hf_config=PretrainedConfig(),
        multimodal_config=None,
        model_stage="tts",
        use_v2_model_runner=True,
        session_mode=session_mode,
        async_chunk=True,
    )
    mocker.patch.object(omni, "init_vllm_registered_model", return_value=talker)
    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat.patch_minicpmo_remote_config")
    model = omni.MiniCPMO45OmniForConditionalGeneration(vllm_config=config)
    # The runner skips its async snapshot copy only for the Talker stage.
    assert model.mm_outputs_fresh_per_step is True
    worker = mocker.Mock(
        spec=OmniGPUWorkerBase,
        model_runner=mocker.Mock(spec=OmniARModelRunner, model=model),
    )
    if warmup_fails:
        with pytest.raises(RuntimeError, match="codec priming failed"):
            OmniGPUWorkerBase._capture_auxiliary_graphs(worker)
    else:
        OmniGPUWorkerBase._capture_auxiliary_graphs(worker)
    sampler.warmup.assert_called_once_with()


def test_codec_penalty_runs_through_upstream_processor_registration(monkeypatch, mocker):
    """Exercise vLLM admission and processor dispatch, not the helper alone."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
    from vllm.v1.worker.gpu.sample.sampler import Sampler as DeviceSampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _apply_batched_repetition_penalty,
        _install_mrv2_talker_sampler,
    )

    def cpu_buffer(size, dtype):
        tensor = torch.zeros(size, dtype=dtype)
        return mocker.Mock(spec=UvaBackedTensor, np=tensor.numpy(), gpu=tensor, copy_to_uva=lambda: None)

    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.UvaBackedTensor", cpu_buffer)
    monkeypatch.setattr("vllm.v1.worker.gpu.sample.penalties.UvaBackedTensor", cpu_buffer)
    states = mocker.Mock(
        spec=RequestState,
        device=torch.device("cpu"),
        max_num_reqs=3,
        vocab_size=12,
        all_token_ids=mocker.Mock(gpu=torch.tensor([[0, 0, 3, 3, 7], [0] * 5, [0, 0, 8, 8, 8]])),
        prompt_len=mocker.Mock(gpu=torch.tensor([2, 2, 2])),
        total_len=mocker.Mock(gpu=torch.tensor([5, 2, 5])),
    )
    sampler = object.__new__(DeviceSampler)
    sampler.req_states = states
    original = PenaltiesState(None, states)
    sampler.penalties_state = original
    before, after = mocker.Mock(), mocker.Mock()
    before.apply.side_effect = after.apply.side_effect = lambda logits, ctx: logits
    before.add_request.return_value = after.add_request.return_value = False
    sampler.logits_processors = [before, original, after]
    sampler.sampling_states = mocker.Mock()
    sampler.sampling_states.add_request.return_value = False
    sampler.sampling_states.apply_top_k_top_p.side_effect = lambda logits, *args: logits
    sampler.thinking_budget_state = mocker.Mock()
    sampler.thinking_budget_state.add_request.return_value = False
    sampler.logprob_token_ids_state = mocker.Mock()
    sampler.trace_replay_state = None
    sampler.needs_logits_processing = np.zeros(3, dtype=bool)
    adapter = _install_mrv2_talker_sampler(sampler, _talker())
    codec = sampler.penalties_state
    assert sampler.logits_processors == [before, codec, after]
    params = SamplingParams(repetition_penalty=1.05)
    adapter.add_request(2, params)
    adapter.add_request(0, params)
    adapter.apply_staged_writes()
    assert sampler.needs_logits_processing.tolist() == [True, False, True]
    assert params.repetition_penalty == 1.05
    assert original.repetition_penalty.np[[0, 2]].tolist() == [1.0, 1.0]
    logits = torch.linspace(-2, 2, 24).reshape(2, 12)
    expected = _apply_batched_repetition_penalty(
        logits, [torch.tensor([8, 8, 8]), torch.tensor([3, 3, 7])], penalty=1.05, window_size=16
    )
    mapping = torch.tensor([2, 0])
    actual = sampler.apply_sampling_params(
        logits,
        mapping,
        mapping,
        np.array([2, 0]),
        torch.tensor([4, 4]),
        torch.tensor([8, 7]),
        torch.zeros(2, dtype=torch.long),
        np.array([5, 5]),
    )
    torch.testing.assert_close(actual, expected)
    assert before.apply.call_count == after.apply.call_count == 1
    codec.prefix_history[2].fill_(7)
    adapter.add_request(2, SamplingParams(repetition_penalty=1.0))
    assert not sampler.needs_logits_processing[2]
    assert codec.prefix_history[2].tolist() == [-1] * 16


@pytest.mark.parametrize("fallback", ["expanded", "speculative", "ordinary", "unseeded"])
def test_seeded_codec_unsupported_batches_keep_original_sampler(monkeypatch, mocker, fallback):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import mrv2

    monkeypatch.setattr(mrv2, "LegacySampler", mocker.Mock())
    original = mocker.Mock(spec=object.__new__(Sampler))
    original.num_speculative_tokens = 2 if fallback == "speculative" else 1
    original.req_states = mocker.Mock(spec=RequestState, index_to_req_id={2: "live"})
    model = mocker.Mock(
        spec=MiniCPMO45OmniTTSForConditionalGeneration,
        _mrv2_output_infos=[{"native_duplex": fallback != "ordinary"}],
    )
    core = mrv2.MiniCPMO45SeededCodecSampler(original, model)
    core.add_request(2, SamplingParams(seed=None if fallback == "unseeded" else 42))
    batch = mocker.Mock(spec=InputBatch, idx_mapping_np=np.array([2]))
    logits = torch.ones(2 if fallback == "expanded" else 1, 10)
    output = core(logits, batch)
    assert output is original.return_value
    original.assert_called_once_with(logits, batch)
    assert not core._generators


def test_terminal_cleanup_releases_only_finished_codec_rng(monkeypatch, mocker):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import mrv2

    monkeypatch.setattr(mrv2, "LegacySampler", mocker.Mock())
    original = mocker.Mock(spec=object.__new__(Sampler))
    talker = _talker()
    torch.nn.Module.__init__(talker)
    core = mrv2.MiniCPMO45SeededCodecSampler(original, talker)
    live = torch.Generator().manual_seed(17)
    core._generators = {"ended": torch.Generator().manual_seed(42), "live": live}
    # Admission can reuse or move a slot without resetting a live request's RNG.
    for slot, seed in [(2, 42), (3, 17), (2, 99), (4, 17)]:
        core.add_request(slot, SamplingParams(seed=seed))
    assert len(core._params_by_slot) == 3
    assert core._generators["live"] is live
    talker._mrv2_seeded_codec_sampler = core
    talker._request_audio_states = {"ended": {}, "live": {"recent_codes": [2]}}
    talker._request_condition_states = {"ended": {}, "live": {}}
    talker._decode_codec_ids = {"ended": (), "live": ()}
    talker._deferred_cleanup_ids = set()
    talker.on_requests_finished({"ended"})
    talker._flush_deferred_cleanup()
    assert core._generators == {"live": live}
    assert set(talker._request_audio_states) == {"live"}
    assert set(talker._request_condition_states) == {"live"}
    assert set(talker._decode_codec_ids) == {"live"}
    assert not talker._deferred_cleanup_ids


def test_mrv2_output_emits_decode_input_codes_with_validity_and_forced_eos() -> None:
    talker = _talker()
    rows = [
        # Final prefill chunk: no code yet (V1 emits an empty delta).
        dict(slot=2, prompt_len=5, computed=0, span=[0, 0, 0, 0, 0], prefill=True),
        # Decode with a codec id: emitted this step.
        dict(slot=0, prompt_len=4, computed=6, span=[17], prefill=False),
        # Decode whose input is codec EOS: request ended, force EOS again.
        dict(slot=1, prompt_len=4, computed=8, span=[_EOS], prefill=False),
    ]
    batch, padded = _batch(rows, pad_to=8)
    buffers = [{"audio_state": {"finished": False}}, {}, {}]
    hidden = torch.zeros((padded, 4))
    out = talker.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=_req_states({cast(int, row["slot"]): cast(int, row["prompt_len"]) for row in rows}),
        model_intermediate_buffer=buffers,
    )
    assert isinstance(out, OmniOutput)
    codes = out.multimodal_outputs["codes"]["audio"]
    valid = out.multimodal_outputs["meta"]["codec_frame_valid"]
    assert codes.shape == (padded, 1) and valid.shape == (padded,)
    assert codes[:, 0].tolist() == batch.input_ids.tolist()
    assert valid.tolist() == [False] * 5 + [True, False] + [False]
    assert talker.take_mrv2_forced_eos(batch, None, 3).tolist() == [False, False, True]
    # Consumed once; a warmup sampler call sees nothing.
    assert talker.take_mrv2_forced_eos(batch, None, 3) is None


def test_mrv2_output_empty_condition_and_length_cap() -> None:
    talker = _talker()
    prompt_lens = {3: 6, 4: 4096 - 100}
    # Empty Thinker condition: preprocess marks the request finished at prefill.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=0, span=[0] * 6, prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros((6, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=[{"audio_state": {"finished": True}}],
    )
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]
    # Its later decode rows stay forced and emit nothing, whatever the input id.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=6, span=[42], prefill=False)])
    out = talker.make_omni_output_mrv2(
        torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
    )
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [False]
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]

    # Offline cap: min(2048, context - prompt) samples, the last one EOS.
    limit = min(_OFFLINE_CODEC_MAX_NEW_TOKENS, 100) - 1
    for step, forced in ((limit - 1, False), (limit, True)):
        batch, _ = _batch(
            [dict(slot=4, prompt_len=prompt_lens[4], computed=prompt_lens[4] + step - 1, span=[5], prefill=False)]
        )
        out = talker.make_omni_output_mrv2(
            torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
        )
        assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [True]
        assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [forced]


@pytest.mark.parametrize("forced", [None, [False, True]])
def test_sampler_adapter_keeps_upstream_counts_and_only_forces_codec_eos(mocker, forced):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45TalkerSampler

    output = SimpleNamespace(sampled_token_ids=torch.tensor([[2], [3]]), num_sampled=torch.tensor([1, 0]))
    base = mocker.Mock(return_value=output)
    base.req_states = object()
    mask = None if forced is None else torch.tensor(forced)
    talker = SimpleNamespace(_codec_eos_id=7, take_mrv2_forced_eos=mocker.Mock(return_value=mask))
    sampler = MiniCPMO45TalkerSampler(base, talker)
    logits, batch = torch.zeros(2, 8), object()
    assert sampler(logits, batch) is output
    assert output.sampled_token_ids.tolist() == [[2], [3 if forced is None else 7]]
    assert output.num_sampled.tolist() == [1, 0]
    base.assert_called_once_with(logits, batch)
    talker.take_mrv2_forced_eos.assert_called_once_with(batch, base.req_states, 2)


@pytest.mark.cuda
def test_mrv2_sampler_applies_codec_window_penalty_instead_of_stock_penalty():
    """The real MRv2 sampler pipeline scores the V1 16-frame codec penalty.

    The stock penalty would tax every code in the prompt (the scheduler's
    placeholder id 0) and in the whole output once; the Talker's penalty taxes
    ``penalty ** count`` over the last 16 sampled codes only.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _CODEC_PENALTY_WINDOW,
        _apply_batched_repetition_penalty,
        _install_mrv2_talker_sampler,
    )

    device, vocab, penalty = torch.device("cuda"), 32, 1.05
    reqs = RequestState(4, 64, 64, 0, vocab, device)
    sampler = Sampler(VllmConfig(), 4, vocab, device, reqs)
    talker = _talker(max_reqs=4)
    wrapped = _install_mrv2_talker_sampler(sampler, talker)

    prompt = [0, 0, 0]
    # Code 5 only before the window, code 7 three times and code 9 once inside it.
    output = [5, 5, 5, 5] + [7, 1, 2, 7, 3, 4, 7, 6, 8, 10, 11, 12, 13, 14, 15, 9]
    assert len(output) - 4 == _CODEC_PENALTY_WINDOW
    reqs.add_request("talker", len(prompt), prompt + output, len(prompt) + len(output), 64)
    slot = reqs.req_id_to_index["talker"]
    wrapped.add_request(slot, SamplingParams(repetition_penalty=penalty, temperature=1.0))
    reqs.apply_staged_writes()
    wrapped.apply_staged_writes()

    logits = torch.linspace(-3.0, 3.0, vocab, device=device).reshape(1, vocab)
    idx = torch.tensor([slot], dtype=torch.int32, device=device)
    processed = wrapped.apply_sampling_params(
        logits.clone(),
        idx,
        idx,
        np.array([slot]),
        torch.tensor([len(prompt) + len(output)], device=device),
        torch.tensor([output[-1]], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        np.array([len(prompt) + len(output) + 1]),
        skip_top_k_top_p=True,
    )
    expected = _apply_batched_repetition_penalty(
        logits.cpu(), [torch.tensor(output)], penalty=penalty, window_size=_CODEC_PENALTY_WINDOW
    )
    torch.testing.assert_close(processed.cpu(), expected)
    # Neither the placeholder prompt id nor codes outside the window are taxed.
    assert processed[0, 0].item() == pytest.approx(logits[0, 0].item())
    assert processed[0, 5].item() == pytest.approx(logits[0, 5].item())
    assert processed[0, 7].item() == pytest.approx(logits[0, 7].item() * penalty**3)
    # The runner still finds the output bin counts on ``penalties_state``.
    assert sampler.penalties_state.output_bin_counts is not None


@pytest.mark.parametrize(
    "turn_end,step,masked,forced",
    [
        (False, 0, True, False),
        (False, 24, True, False),
        (False, 25, False, True),
        (True, 24, False, False),
        (True, 25, True, False),
        (True, 29, True, False),
        (True, 30, False, False),
        (True, 99, False, True),
    ],
)
def test_duplex_device_cadence_and_drain(mocker, turn_end, step, masked, forced):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _CodecWindowPenaltiesState

    talker = _talker()
    talker._mrv2_penalties = mocker.Mock(
        spec=_CodecWindowPenaltiesState, prefix_history=torch.full((8, 16), -1, dtype=torch.long)
    )
    info = {
        "req_id": "r",
        "native_duplex": True,
        "duplex": {"epoch": 2, "turn_id": 3},
        "audio_state": {
            "finished": False,
            "max_tokens": 100 if turn_end else 26,
            "min_tokens": 0 if turn_end else 26,
            "turn_end_drain": turn_end,
            "recent_codes": [9, 8],
        },
    }
    batch, _ = _batch([dict(slot=3, prompt_len=4, computed=0, span=[0] * 4, prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros((4, 4)), input_batch=batch, req_states=_req_states({3: 4}), model_intermediate_buffer=[info]
    )
    assert talker._mrv2_penalties.prefix_history[3, -2:].tolist() == [9, 8]
    if step:
        batch, _ = _batch([dict(slot=3, prompt_len=4, computed=3 + step, span=[17], prefill=False)])
        infos = [info]
        out = talker.make_omni_output_mrv2(
            torch.zeros((1, 4)), input_batch=batch, req_states=_req_states({3: 4}), model_intermediate_buffer=infos
        )
        # Per-request duplex metadata is restored on the host by the finalizer.
        talker._request_audio_states = {}
        partition = talker.mrv2_codec_history_finalizer(batch, infos)(out.multimodal_outputs, [1])
        assert partition.inter_stage[0]["meta.duplex_epoch"].item() == 2
    assert talker._mrv2_mask_eos.tolist() == [masked]
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [forced]


@pytest.mark.parametrize("new_count", [0, 1, 15, 16, 20])
def test_device_penalty_keeps_cross_condition_history(new_count):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _apply_batched_repetition_penalty,
        _apply_codec_window_penalty_gpu,
    )

    seed, new_codes = [3, 3, 7, 8], [7] * new_count
    # Prompt zeros must not become codec history.
    ids = torch.tensor([[0] * 6 + new_codes + [0]])
    seed_tensor = torch.tensor([[-1] * 12 + seed])
    logits = torch.linspace(-2, 2, 12).reshape(1, 12)
    expected = _apply_batched_repetition_penalty(logits, [torch.tensor(seed + new_codes)], penalty=1.05, window_size=16)
    _apply_codec_window_penalty_gpu(
        logits,
        torch.tensor([0]),
        ids,
        torch.tensor([6 + new_count]),
        torch.tensor([6]),
        torch.tensor([1.05]),
        window_size=16,
        prefix_history=seed_tensor,
    )
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("nested", [False, True])
def test_delayed_history_copy_cannot_update_replaced_condition(nested):
    talker = _talker()
    previous = {"recent_codes": [7]}
    current = {"recent_codes": [9]}
    talker._request_audio_states = {"r": previous}
    batch, _ = _batch([dict(slot=0, prompt_len=4, computed=4, span=[8], prefill=False)])
    finalize = talker.mrv2_codec_history_finalizer(batch, [{"request_id": "r", "req_id": "r", "audio_state": previous}])
    talker._request_audio_states["r"] = current
    output = {"codes.audio": torch.tensor([[8]]), "meta.codec_frame_valid": torch.tensor([True])}
    if nested:
        output = {
            "codes": {"audio": output["codes.audio"]},
            "meta": {"codec_frame_valid": output["meta.codec_frame_valid"]},
        }
    assert finalize(output, [1]) is output
    assert current == {"recent_codes": [9]}
    assert previous == {"recent_codes": [7]}


def test_history_finalizer_reads_nested_snapshot_and_filters_partial_prefill():
    talker = _talker()
    state = {"recent_codes": [7]}
    partial = {"recent_codes": [9]}
    talker._request_audio_states = {"r": state, "partial": partial}
    batch, _ = _batch(
        [
            dict(slot=0, prompt_len=4, computed=4, span=[8], prefill=False),
            dict(slot=1, prompt_len=4, computed=0, span=[0, 0], prefill=True),
        ]
    )
    finalize = talker.mrv2_codec_history_finalizer(
        batch, [{"req_id": "r", "audio_state": state}, {"req_id": "partial", "audio_state": partial}]
    )
    payload = {
        "codes": {"audio": torch.tensor([[8], [0], [0]])},
        "meta": {"codec_frame_valid": torch.tensor([True, False, False])},
    }
    assert finalize(payload, [1, 0]) is payload
    assert state["recent_codes"] == [7, 8]
    assert partial["recent_codes"] == [9]


@pytest.mark.parametrize(
    "prefill,code,count,committed",
    [(True, _EOS, 1, True), (False, 8, 1, True), (False, _EOS, 1, False), (False, 8, 0, False)],
)
def test_async_codec_rng_commits_real_draws_and_excludes_eos_lookahead(mocker, prefill, code, count, committed):
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler

    # This CPU test exercises checkpoints; the CUDA oracle exercises real draws.
    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2.LegacySampler")
    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker.vllm_config.scheduler_config.async_scheduling = True
    state: dict[str, object] = {}
    talker._request_audio_states = {"r": state}
    talker._request_condition_states = {"r": {"condition_seq": 0}}
    core = MiniCPMO45SeededCodecSampler(mocker.Mock(spec=Sampler), talker)
    generator = mocker.Mock(spec=torch.Generator, get_offset=mocker.Mock(return_value=104))
    core._generators["r"] = generator
    core._prepare_condition_rng("r", generator)
    talker._mrv2_seeded_codec_sampler = core
    batch, _ = _batch([dict(slot=0, prompt_len=4, computed=4, span=[code], prefill=prefill)])
    finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "r", "native_duplex": True}])
    # Capturing and committing do not restore the generator on each step.
    assert not generator.set_offset.called
    generator.get_offset.return_value = 108  # An extra async draw was launched.
    payload = {"codes.audio": torch.tensor([[code]]), "meta.codec_frame_valid": torch.tensor([code != _EOS])}
    finalize(payload, [count])
    assert ("r" in core._committed_offsets) is committed
    assert not generator.set_offset.called
    talker._request_condition_states["r"] = {"condition_seq": 1}
    core._prepare_condition_rng("r", generator)
    if committed:
        generator.set_offset.assert_called_once_with(104)
    else:
        generator.set_offset.assert_not_called()
    core.on_requests_finished({"r"})
    assert not core._condition_seqs and not core._committed_offsets and not core._generators


@pytest.mark.parametrize("replace", ["condition", "generator", "finish"])
def test_async_codec_rng_rejects_stale_cpu_callbacks(mocker, replace):
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler

    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2.LegacySampler")
    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker.vllm_config.scheduler_config.async_scheduling = True
    talker._request_audio_states = {"r": {}}
    talker._request_condition_states = {"r": {"condition_seq": 0}}
    core = MiniCPMO45SeededCodecSampler(mocker.Mock(spec=Sampler), talker)
    generator = mocker.Mock(spec=torch.Generator, get_offset=mocker.Mock(return_value=104))
    core._generators["r"] = generator
    core._prepare_condition_rng("r", generator)
    talker._mrv2_seeded_codec_sampler = core
    batch, _ = _batch([dict(slot=0, prompt_len=4, computed=4, span=[8], prefill=False)])
    finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "r", "native_duplex": True}])
    if replace == "condition":
        talker._request_audio_states["r"] = {}
    elif replace == "generator":
        core._generators["r"] = mocker.Mock(spec=torch.Generator)
    else:
        core.on_requests_finished({"r"})
    finalize({"codes.audio": torch.tensor([[8]]), "meta.codec_frame_valid": torch.tensor([True])}, [1])
    assert not core._committed_offsets


def test_history_survives_real_intermediate_buffer_merge_and_next_condition(mocker):
    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.zeros(2, 4))
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(
        0,
        OmniNewRequestData(
            req_id="r",
            prompt_token_ids=[0, 0],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
            model_intermediate_buffer={
                "native_duplex": True,
                "ids": {"tts": torch.tensor([7])},
                "hidden_states": {"tts": torch.zeros(1, 4)},
                "meta": {"streaming_condition_seq": 0, "turn_start": True},
            },
        ),
    )

    def prefill():
        _, _, updates = talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            **buffer.buffers[0],
            _omni_is_prefill=True,
            _omni_prompt_len=2,
        )
        buffer.update(0, updates)
        return updates["audio_state"]

    previous = prefill()
    # The real buffer copies/merges dictionary fields; its audio_state is
    # not the model-owned object even immediately after preprocess.
    assert buffer.buffers[0]["audio_state"] is not previous
    batch, _ = _batch([dict(slot=0, prompt_len=2, computed=2, span=[8], prefill=False)])
    finalize = talker.mrv2_codec_history_finalizer(batch, buffer.gather(batch))
    for code in [8, 9, 8]:
        payload = {"codes.audio": torch.tensor([[code]]), "meta.codec_frame_valid": torch.tensor([True])}
        assert finalize(payload, [1]) is payload
    assert previous["recent_codes"] == [8, 9, 8]

    buffer.update(0, {"meta": {"streaming_condition_seq": 1, "turn_start": False}})
    current = prefill()
    assert current is not previous
    assert current["recent_codes"] == [8, 9, 8]
    # A delayed copy still belongs to the old condition despite both buffer
    # dictionaries being copies; it must not pollute the successor's prefix.
    finalize(payload, [1])
    assert previous["recent_codes"] == current["recent_codes"] == [8, 9, 8]


def _native_info(slot: int, *, turn_end: bool = False, native: bool = True, empty: bool = False) -> dict:
    info: dict[str, Any] = {
        "req_id": f"r{slot}",
        "duplex": {"epoch": 10 + slot, "turn_id": 20 + slot},
        "meta": {"native_duplex_segment_text": "你好" * (slot + 1), "turn_eos_token_id": 99},
        "ids": {"tts": torch.tensor([5, 99] if turn_end else [5])},
        "audio_state": {
            "finished": empty,
            "max_tokens": 100 if turn_end else 26,
            "min_tokens": 0 if turn_end else 26,
            "turn_end_drain": turn_end,
            "recent_codes": [slot],
        },
    }
    if native:
        info["native_duplex"] = True
    return info


def _reference_mrv2_controls(talker, batch, req_states, num_tokens: int, native: bool):
    """The original op-by-op formulation of forced EOS, EOS mask and frame validity."""
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _DUPLEX_CODEC_FRAMES_PER_CHUNK,
        _DUPLEX_TURN_END_BOUNDARY_MASK_STEPS,
    )

    n = batch.num_reqs
    token_ids = batch.input_ids[:num_tokens]
    last_rows = batch.logits_indices[:n].long()
    slot_ids = batch.idx_mapping[:n].long()
    prompt_len = req_states.prompt_len.gpu.index_select(0, slot_ids).long()
    step = batch.seq_lens[:n].long() - prompt_len
    last_ids = token_ids.index_select(0, last_rows)
    empty = talker._mrv2_empty_speech.index_select(0, slot_ids)
    decode = step > 0
    eos_input = decode & (last_ids == _EOS)
    valid = decode & ~eos_input & ~empty
    remaining = int(talker._tts_config.max_position_embeddings) - prompt_len
    limit = torch.clamp(remaining, min=1, max=_OFFLINE_CODEC_MAX_NEW_TOKENS) - 1
    mask = None
    if native:
        controls = talker._mrv2_codec_controls.index_select(0, slot_ids)
        limit = torch.where(controls[:, 0] >= 0, controls[:, 0], limit)
        boundary = (step >= _DUPLEX_CODEC_FRAMES_PER_CHUNK) & (
            step.remainder(_DUPLEX_CODEC_FRAMES_PER_CHUNK) < _DUPLEX_TURN_END_BOUNDARY_MASK_STEPS
        )
        mask = (step < controls[:, 1]) | ((controls[:, 2] == 1) & boundary)
    forced = empty | eos_input | (step >= limit)
    if native:
        mask &= ~forced
    frame_valid = torch.zeros(num_tokens, dtype=torch.bool)
    frame_valid.index_copy_(0, last_rows, valid)
    return forced, mask, frame_valid


def _prefill_native_slots(talker, prompt_lens: dict[int, int], infos: list[dict]):
    talker._mrv2_penalties = SimpleNamespace(prefix_history=torch.full((8, 16), -1, dtype=torch.long))
    rows = [dict(slot=slot, prompt_len=p, computed=0, span=[0] * p, prefill=True) for slot, p in prompt_lens.items()]
    batch, _ = _batch(rows)
    talker.make_omni_output_mrv2(
        torch.zeros((batch.input_ids.shape[0], 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=infos,
    )


@pytest.mark.parametrize("pad", [None, 8])
@pytest.mark.parametrize("step", [1, 2, 24, 25, 26, 29, 30, 31, 50, 54, 55, 95, 96, 99, 100])
def test_mrv2_decode_controls_match_reference(step, pad):
    """Plain-decode fast path (no logits gather, int32 step math) equals the original ops."""
    talker = _talker()
    # Slot 4 is a non-native row of a native batch: offline length cap near the context end.
    prompt_lens = {0: 4, 1: 5, 2: 4, 3: 6, 4: 4096 - 97, 5: 4}
    infos = [
        _native_info(0),
        _native_info(1, turn_end=True),
        _native_info(2),
        _native_info(3, turn_end=True),
        _native_info(4, native=False),
        _native_info(5, empty=True),
    ]
    _prefill_native_slots(talker, prompt_lens, infos)
    ids = [17, 18, _EOS, 19, 20, 21]
    rows = [
        dict(slot=slot, prompt_len=p, computed=p + step - 1, span=[ids[slot]], prefill=False)
        for slot, p in prompt_lens.items()
    ]
    batch, num_tokens = _batch(rows, pad_to=pad)
    req_states = _req_states(prompt_lens)
    out = talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)), input_batch=batch, req_states=req_states, model_intermediate_buffer=infos
    )
    forced, mask, frame_valid = _reference_mrv2_controls(talker, batch, req_states, num_tokens, native=True)
    assert talker._mrv2_forced_eos.tolist() == forced.tolist()
    assert talker._mrv2_mask_eos.tolist() == mask.tolist()
    meta = out.multimodal_outputs["meta"]
    assert meta["codec_frame_valid"].dtype == torch.bool
    assert meta["codec_frame_valid"].tolist() == frame_valid.tolist()
    assert meta["finished"].tolist() == forced.tolist()


def test_mrv2_mixed_prefill_decode_controls_match_reference():
    talker = _talker()
    prompt_lens = {0: 4, 1: 5, 2: 4}
    infos = [_native_info(0), _native_info(1, turn_end=True), _native_info(2)]
    _prefill_native_slots(talker, prompt_lens, infos)
    # Slot 1 starts a new condition while the others decode.
    prompt_lens[1] = 7
    infos = [_native_info(0), _native_info(1), _native_info(2)]
    rows = [
        dict(slot=0, prompt_len=4, computed=4 + 30, span=[_EOS], prefill=False),
        dict(slot=1, prompt_len=7, computed=0, span=[0] * 7, prefill=True),
        dict(slot=2, prompt_len=4, computed=4 + 3, span=[33], prefill=False),
    ]
    batch, num_tokens = _batch(rows, pad_to=16)
    req_states = _req_states(prompt_lens)
    out = talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)), input_batch=batch, req_states=req_states, model_intermediate_buffer=infos
    )
    forced, mask, frame_valid = _reference_mrv2_controls(talker, batch, req_states, num_tokens, native=True)
    assert talker._mrv2_forced_eos.tolist() == forced.tolist()
    assert talker._mrv2_mask_eos.tolist() == mask.tolist()
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == frame_valid.tolist()


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_mrv2_payload_is_owned_per_step(input_dtype):
    """The runner skips the async snapshot: no leaf may alias a reused buffer."""
    talker = _talker()
    assert MiniCPMO45OmniTTSForConditionalGeneration.mm_outputs_fresh_per_step is True
    prompt_lens = {0: 4, 1: 4}
    infos = [_native_info(0), _native_info(1)]
    _prefill_native_slots(talker, prompt_lens, infos)
    rows = [dict(slot=s, prompt_len=4, computed=5, span=[40 + s], prefill=False) for s in prompt_lens]
    batch, num_tokens = _batch(rows)
    batch.input_ids = batch.input_ids.to(input_dtype)
    out = talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=infos,
    )
    codes = out.multimodal_outputs["codes"]["audio"]
    meta = out.multimodal_outputs["meta"]
    assert codes.dtype == torch.long
    assert codes.untyped_storage().data_ptr() != batch.input_ids.untyped_storage().data_ptr()
    assert set(meta) == {"codec_frame_valid", "finished"}
    batch.input_ids.fill_(0)
    talker._mrv2_empty_speech.fill_(True)
    assert codes[:, 0].tolist() == [40, 41]
    assert meta["codec_frame_valid"].tolist() == [True, True]


def _list_layout(talker, out, batch):
    """The pre-finalizer per-request list layout the runner used to partition."""
    meta = out.multimodal_outputs["meta"]
    layout = {
        "codes": {"audio": out.multimodal_outputs["codes"]["audio"]},
        "meta": {"codec_frame_valid": meta["codec_frame_valid"], "finished": list(meta["finished"].unbind())},
    }
    for key in ("native_duplex", "duplex_epoch", "duplex_turn_id", "llm_output_text_utf8", "turn_end"):
        layout["meta"][key] = [talker._mrv2_metadata_by_slot[int(slot)][key][0] for slot in batch.idx_mapping_np]
    return layout


def _assert_same_partition(actual, expected, cached: list[torch.Tensor]):
    assert len(actual) == len(expected)
    cached_ptrs = {tensor.untyped_storage().data_ptr() for tensor in cached}
    for got, want in zip(actual, expected, strict=True):
        assert list(got) == list(want)
        for key, value in want.items():
            assert got[key].dtype == value.dtype and got[key].shape == value.shape, key
            assert torch.equal(got[key], value), key
            # Codes/validity are token-axis views; every scalar/text row is owned.
            assert (got[key]._base is None) == (value._base is None), key
            if got[key].dim() == 0 or key == "meta.llm_output_text_utf8":
                assert got[key].untyped_storage().data_ptr() not in cached_ptrs, key


@pytest.mark.parametrize("pad", [None, 16])
def test_native_finalizer_partition_matches_runner_partition(pad):
    from vllm_omni.model_executor.output_snapshot import RequestOutputSnapshot
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    talker = _talker()
    prompt_lens = {0: 4, 1: 5, 2: 4}
    infos = [_native_info(0), _native_info(1, turn_end=True), _native_info(2)]
    _prefill_native_slots(talker, prompt_lens, infos)
    prompt_lens[2] = 6
    infos = [_native_info(0), _native_info(1, turn_end=True), _native_info(2, turn_end=True)]
    rows = [
        dict(slot=1, prompt_len=5, computed=5 + 7, span=[71], prefill=False),
        dict(slot=0, prompt_len=4, computed=4 + 30, span=[_EOS], prefill=False),
        dict(slot=2, prompt_len=6, computed=0, span=[0] * 6, prefill=True),
    ]
    batch, num_tokens = _batch(rows, pad_to=pad)
    batch.num_tokens_after_padding = num_tokens
    out = talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=infos,
    )
    expected_inter, expected_client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        _list_layout(talker, out, batch),
        batch.query_start_loc_np,
        batch.num_scheduled_tokens,
        batch.num_reqs,
        int(batch.query_start_loc_np[batch.num_reqs]),
        num_tokens,
    )
    talker._request_audio_states = {}
    partition = talker.mrv2_codec_history_finalizer(batch, infos)(out.multimodal_outputs, [1, 1, 1])
    assert isinstance(partition, RequestOutputSnapshot)
    cached = [t[0] for entry in talker._mrv2_metadata_by_slot.values() for t in entry.values()]
    _assert_same_partition(partition.inter_stage, expected_inter, cached)
    _assert_same_partition(partition.client, expected_client, cached)
    for inter, client in zip(partition.inter_stage, partition.client, strict=True):
        assert all(client[key] is inter[key] for key in client)


def test_native_finalizer_restores_list_layout_for_unrecognized_payload():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _mrv2_duplex_output_partition

    talker = _talker()
    prompt_lens = {0: 4, 1: 4}
    infos = [_native_info(0), _native_info(1)]
    _prefill_native_slots(talker, prompt_lens, infos)
    rows = [dict(slot=s, prompt_len=4, computed=5, span=[40 + s], prefill=False) for s in prompt_lens]
    batch, num_tokens = _batch(rows)
    out = talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=infos,
    )
    expected = _list_layout(talker, out, batch)
    stashed = talker._mrv2_output_meta[1]
    # An unexpected token axis keeps the runner's generic partition.
    restored = _mrv2_duplex_output_partition(out.multimodal_outputs, stashed, [(0, 1), (1, 1)], set())
    assert restored is out.multimodal_outputs
    assert list(restored["meta"]) == list(expected["meta"])
    for key, values in expected["meta"].items():
        got = restored["meta"][key]
        if isinstance(values, list):
            assert [torch.equal(a, b) for a, b in zip(got, values, strict=True)] == [True] * len(values)


def test_finalizer_rejects_metadata_from_another_batch():
    talker = _talker()
    prompt_lens = {0: 4}
    infos = [_native_info(0)]
    _prefill_native_slots(talker, prompt_lens, infos)
    batch, num_tokens = _batch([dict(slot=0, prompt_len=4, computed=5, span=[40], prefill=False)])
    talker.make_omni_output_mrv2(
        torch.zeros((num_tokens, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=infos,
    )
    talker._request_audio_states = {}
    with pytest.raises(RuntimeError, match="different batch"):
        talker.mrv2_codec_history_finalizer(batch, list(infos))
    # Consumed: a later step without native output is unaffected.
    assert talker._mrv2_output_meta is None


@pytest.mark.parametrize("valid_rows", [[True], [False], [True, False, True], [False, False]])
def test_history_finalizer_batched_read_matches_per_row_masks(valid_rows):
    talker = _talker()
    states = {"a": {"recent_codes": list(range(14))}, "b": {"recent_codes": [3]}}
    talker._request_audio_states = dict(states)
    n = len(valid_rows)
    rows = [dict(slot=0, prompt_len=4, computed=6, span=list(range(50, 50 + n)), prefill=False)]
    rows.append(dict(slot=1, prompt_len=4, computed=6, span=[90], prefill=False))
    batch, _ = _batch(rows)
    finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "a"}, {"req_id": "b"}])
    payload = {
        "codes": {"audio": torch.tensor([[50 + i] for i in range(n)] + [[90]])},
        "meta": {"codec_frame_valid": torch.tensor([*valid_rows, True])},
    }
    assert finalize(payload, [1, 1]) is payload
    kept = [50 + i for i, keep in enumerate(valid_rows) if keep]
    assert states["a"]["recent_codes"] == (list(range(14)) + kept)[-16:]
    assert states["b"]["recent_codes"] == [3, 90]
