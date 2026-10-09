# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA codec-window penalties against independently assembled V1 histories."""

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _apply_batched_repetition_penalty,
    _apply_codec_window_penalty_gpu,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("asynchronous", [False, True])
@torch.inference_mode()
def test_codec_rng_matches_v1_across_conditions_after_async_eos_lookahead(mocker, asynchronous):
    import numpy as np
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.sampler import Sampler as LegacySampler
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    device = torch.device("cuda")
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker.vllm_config.scheduler_config.async_scheduling = asynchronous
    talker._codec_eos_id = 63
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    reqs = mocker.Mock(spec=RequestState, index_to_req_id={0: "r", 2: "r"})
    base = mocker.Mock(spec=Sampler, req_states=reqs)
    core = MiniCPMO45SeededCodecSampler(base, talker)
    talker._mrv2_seeded_codec_sampler = core
    reference = torch.Generator(device=device).manual_seed(42)
    legacy = LegacySampler()
    core._rows = core._accepted = (0,)
    tensor = torch.zeros(1, device=device, dtype=torch.int32)
    for seq in range(3):
        # A returning request can move slots without changing its random stream.
        slot = 0 if seq == 0 else 2
        core.add_request(slot, SamplingParams(seed=42, temperature=1.0))
        talker._request_condition_states["r"] = {"condition_seq": seq}
        talker._request_audio_states["r"] = {}
        previous = 0
        for step in range(3 + (3 if asynchronous else 0)):
            lookahead = step >= 3
            logits = torch.linspace(-2, 2, 64, device=device).reshape(1, 64)
            if step >= 2:
                logits.fill_(float("-inf"))
                logits[:, 63] = 0.0
            base.apply_sampling_params.return_value = logits
            actual, _ = core.sample(
                logits, tensor, tensor, np.array([slot]), tensor, tensor, tensor, np.array([4]), False
            )
            if not lookahead:
                expected, _ = legacy.topk_topp_sampler(logits, {0: reference}, None, None)
                torch.testing.assert_close(actual, expected.long(), rtol=0, atol=0)
                assert torch.equal(core._generators["r"].get_state(), reference.get_state())
            batch = mocker.Mock(
                spec=InputBatch,
                query_start_loc_np=np.array([0, 1]),
                num_scheduled_tokens=np.array([1]),
                is_prefilling_np=np.array([step == 0]),
            )
            finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "r", "native_duplex": True}])
            payload = {
                "codes.audio": torch.tensor([[previous]]),
                "meta.codec_frame_valid": torch.tensor([step != 0 and not lookahead]),
            }
            finalize(payload, [1])
            previous = int(actual[0])
        if asynchronous:
            assert core._generators["r"].get_offset() == reference.get_offset() + 12
        else:
            assert torch.equal(core._generators["r"].get_state(), reference.get_state())
    core.on_requests_finished({"r"})
    assert not core._committed_offsets and not core._condition_seqs and not core._generators


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("capacity", [1, 2, 3, 16])
@torch.inference_mode()
def test_seeded_codec_readiness_warmup_preserves_live_and_default_rng(capacity):
    from triton import knobs
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )
    from vllm_omni.utils.seeded_exponential import fill_exponential_rows

    device = torch.device("cuda")
    reqs = RequestState(capacity, 128, 32, 0, 6562, device)
    base = Sampler(VllmConfig(), capacity, 6562, device, reqs)
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.mrv2_custom_sampler(base)
    sampler = talker._mrv2_seeded_codec_sampler
    generator = torch.Generator(device=device).manual_seed(42)
    sampler._generators["live"] = generator
    live_state = generator.get_state().clone()
    default_state = torch.cuda.get_rng_state().clone()

    original = knobs.runtime.jit_post_compile_hook
    live = False
    late_compiles = []

    def compiled(**kwargs):
        if live and getattr(kwargs.get("fn"), "name", "") == "_generator_exponential_kernel":
            late_compiles.append(kwargs)
        if original is not None:
            return original(**kwargs)

    knobs.runtime.jit_post_compile_hook = compiled
    try:
        talker.capture_auxiliary_graphs()
        talker.capture_auxiliary_graphs()
        live = True
        for rows in range(2, capacity + 1):
            noise = torch.empty((rows, 6562), dtype=torch.float32, device=device)
            generators = [torch.Generator(device=device).manual_seed(row + 17) for row in range(rows)]
            fill_exponential_rows(noise, generators, list(range(rows)))
            reference = torch.empty_like(noise)
            for row in range(rows):
                other = torch.Generator(device=device).manual_seed(row + 17)
                reference[row].exponential_(generator=other)
                torch.testing.assert_close(generators[row].get_state(), other.get_state(), rtol=0, atol=0)
            torch.testing.assert_close(noise, reference, rtol=0, atol=0)
        assert not late_compiles
    finally:
        knobs.runtime.jit_post_compile_hook = original
    torch.testing.assert_close(generator.get_state(), live_state, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), default_state, rtol=0, atol=0)
    assert sampler._generators == {"live": generator}
    assert not sampler._params_by_slot
    assert not sampler._rows and not sampler._accepted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 16, 65])
@pytest.mark.parametrize("with_prefix", [False, True])
def test_fused_codec_penalty_matches_request_histories(dtype, batch_size, with_prefix):
    generator = torch.Generator().manual_seed(812)
    vocab_size, window = 6562, 16
    slots = torch.randperm(128, generator=generator)[:batch_size]
    all_ids = torch.randint(0, vocab_size, (128, 4096), generator=generator, dtype=torch.int32)
    ids = all_ids[:, ::2]
    prompt = torch.randint(0, 100, (128,), generator=generator, dtype=torch.int32)
    counts = torch.arange(128, dtype=torch.int32) % 33
    total = prompt + counts
    prefix = torch.randint(-2, vocab_size + 2, (128, window), generator=generator) if with_prefix else None
    penalties = torch.linspace(0.8, 1.2, 128)
    penalties[slots[::3]] = 1
    # Assemble each request's actual codec history without the device helper.
    histories = []
    for slot in slots.tolist():
        seed = prefix[slot] if prefix is not None else torch.empty(0, dtype=torch.long)
        history = torch.cat([seed, ids[slot, prompt[slot] : total[slot]]])[-window:]
        histories.append(history[(history >= 0) & (history < vocab_size)].cuda())
    values = torch.randn((batch_size, vocab_size * 2), generator=generator, dtype=dtype).cuda()[:, ::2]
    expected = _apply_batched_repetition_penalty(values, histories, penalty=penalties[slots].cuda(), window_size=window)
    _apply_codec_window_penalty_gpu(
        values,
        slots.int().cuda(),
        all_ids.cuda()[:, ::2],
        total.cuda(),
        prompt.cuda(),
        penalties.cuda(),
        window_size=window,
        prefix_history=prefix.cuda() if prefix is not None else None,
    )
    torch.testing.assert_close(values, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_codec_penalty_graph_replay_reads_updated_history():
    vocab_size, window = 12, 16
    slots = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    ids = torch.zeros((3, 32), dtype=torch.int32, device="cuda")
    prompt = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    total = prompt.clone()
    prefix = torch.full((3, window), -1, device="cuda", dtype=torch.long)
    penalties = torch.tensor([1.05, 1.0, 0.95], device="cuda")
    original = torch.linspace(-2, 2, vocab_size, device="cuda").expand(2, -1).clone()
    values = original.clone()

    def apply():
        _apply_codec_window_penalty_gpu(
            values, slots, ids, total, prompt, penalties, window_size=window, prefix_history=prefix
        )

    apply()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply()
    ids[2, 4:7] = torch.tensor([3, 3, 7], device="cuda")
    prefix[0, -2:] = torch.tensor([8, 8], device="cuda")
    total[2] = 7
    values.copy_(original)
    expected = _apply_batched_repetition_penalty(
        original,
        [torch.tensor([3, 3, 7], device="cuda"), torch.tensor([8, 8], device="cuda")],
        penalty=penalties[slots.long()],
        window_size=window,
    )
    graph.replay()
    torch.testing.assert_close(values, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_seeded_codec_draw_preserves_mixed_rows_partial_prefill_and_slot_migration(monkeypatch, mocker):
    import numpy as np
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler as LegacySampler
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    reqs = RequestState(8, 128, 32, 0, 64, device)
    base = Sampler(VllmConfig(), 8, 64, device, reqs)
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    talker = mocker.Mock(spec=MiniCPMO45OmniTTSForConditionalGeneration, _codec_eos_id=63)
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler

    core = MiniCPMO45SeededCodecSampler(base, talker)
    legacy = LegacySampler()
    params = {
        "a": SamplingParams(seed=42, temperature=0.8, top_k=25, top_p=0.85, logprobs=2),
        "b": SamplingParams(seed=17, temperature=1.0, logprobs=2),
        "u": SamplingParams(temperature=1.0, logprobs=2),
        "d": SamplingParams(seed=31, temperature=1.0, logprobs=2),
    }
    for name, p in params.items():
        reqs.add_request(name, 2, [0, 0], 0, 100)
        core.add_request(reqs.req_id_to_index[name], p)
    reqs.apply_staged_writes()
    core.apply_staged_writes()
    generators = {
        name: torch.Generator(device=device).manual_seed(p.seed) for name, p in params.items() if p.seed is not None
    }
    empty = torch.empty(0, device=device)
    order = ["a", "u", "b", "d"]
    calls = []
    apply = base.apply_sampling_params

    def counted(*args, **kwargs):
        calls.append(True)
        return apply(*args, **kwargs)

    monkeypatch.setattr(base, "apply_sampling_params", counted)
    for step in range(16):
        if step in [4, 9]:
            order.reverse()
        if step == 8:
            old = reqs.remove_request("a")
            params["c"] = SamplingParams(seed=99, temperature=1.0, logprobs=2)
            reqs.add_request("c", 2, [0, 0], 0, 100)
            core.add_request(reqs.req_id_to_index["c"], params["c"])
            assert reqs.req_id_to_index["c"] == old
            reqs.add_request("a", 2, [0, 0], 0, 100)
            core.add_request(reqs.req_id_to_index["a"], params["a"])
            assert reqs.req_id_to_index["a"] != old
            reqs.apply_staged_writes()
            core.apply_staged_writes()
        slots = np.array([reqs.req_id_to_index[name] for name in order], dtype=np.int32)
        mapping = torch.tensor(slots, device=device)
        logits = torch.sin(torch.arange(256, device=device).reshape(4, 64) + step)
        partial = {order.index("a")} if step in [3, 10] else set()
        gpu_lens = [1 if row in partial else 2 + step for row in range(4)]
        batch = mocker.Mock(
            spec=InputBatch,
            num_reqs=4,
            expanded_idx_mapping=mapping,
            idx_mapping=mapping,
            idx_mapping_np=slots,
            cu_num_logits_np=np.arange(5, dtype=np.int32),
            cu_num_logits=torch.arange(5, device=device, dtype=torch.int32),
            expanded_local_pos=torch.zeros(4, device=device, dtype=torch.int32),
            positions=torch.full((4,), 1 + step, device=device, dtype=torch.int64),
            logits_indices=torch.arange(4, device=device),
            input_ids=torch.zeros(4, device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.full((4,), 2 + step, dtype=torch.int32),
            seq_lens=torch.tensor(gpu_lens, device=device, dtype=torch.int32),
            num_computed_prefill_tokens_np=np.array([0 if row in partial else 1 for row in range(4)]),
            num_scheduled_tokens=np.ones(4, dtype=np.int32),
            prefill_len_np=np.full(4, 2, dtype=np.int32),
        )
        talker._mrv2_output_infos = [dict(native_duplex=name != "d") for name in order]
        processed = base.apply_sampling_params(
            logits.clone(),
            mapping,
            mapping,
            slots,
            batch.positions,
            batch.input_ids,
            batch.expanded_local_pos,
            batch.seq_lens_cpu_upper_bound.numpy(),
        )
        standard = base(logits.clone(), batch)
        accepted = [row for row, name in enumerate(order) if name in ["a", "b"] and row not in partial]
        metadata = SamplingMetadata(
            output_token_ids=[[] for _ in accepted],
            generators={i: generators[order[row]] for i, row in enumerate(accepted)},
            temperature=torch.ones(len(accepted), device=device),
            top_k=None,
            top_p=None,
            all_greedy=False,
            all_random=True,
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=empty,
            presence_penalties=empty,
            repetition_penalties=empty,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        expected = legacy(processed[accepted].clone(), metadata).sampled_token_ids.to(dtype=torch.int64)
        before = {name: g.get_state().clone() for name, g in core._generators.items()}
        before_calls = len(calls)
        actual = core(logits.clone(), batch)
        assert len(calls) == before_calls + 1
        torch.testing.assert_close(actual.sampled_token_ids[accepted], expected, rtol=0, atol=0)
        other = [row for row, name in enumerate(order) if name in ["u", "d"]]
        torch.testing.assert_close(actual.sampled_token_ids[other], standard.sampled_token_ids[other], rtol=0, atol=0)
        assert actual.num_sampled.tolist() == [0 if row in partial else 1 for row in range(4)]
        assert actual.num_rejected.tolist() == [0] * 4
        assert actual.logprobs_tensors is not None
        if partial:
            assert torch.equal(before["a"], core._generators["a"].get_state())
        # In the raw-logprobs mode, selected values come from the original
        # distribution and remain aligned with the replacement sampled IDs.
        scores = torch.log_softmax(logits.float(), dim=-1).gather(1, actual.sampled_token_ids)
        torch.testing.assert_close(actual.logprobs_tensors.logprobs[:, :1], scores)
    # Greedy native rows should not draw or advance RNG when all selected rows are greedy.
    for name in ["a", "b"]:
        params[name] = SamplingParams(seed=params[name].seed, temperature=0, logprobs=2)
        core.add_request(reqs.req_id_to_index[name], params[name])
    core.apply_staged_writes()
    before = {name: g.get_state().clone() for name, g in core._generators.items()}
    greedy = core(logits.clone(), batch)
    seeded = [row for row, name in enumerate(order) if name in ["a", "b"]]
    torch.testing.assert_close(greedy.sampled_token_ids[seeded, 0], logits[seeded].argmax(-1), rtol=0, atol=0)
    assert all(torch.equal(state, core._generators[name].get_state()) for name, state in before.items())
    core.on_requests_finished(list(params))
    assert not core._generators


def _record_late_compiles(knobs, names):
    original = knobs.runtime.jit_post_compile_hook
    late: list[str] = []
    state = {"live": False, "late": late}

    def compiled(**kwargs):
        name = getattr(kwargs.get("fn"), "name", "")
        if state["live"] and name in names:
            late.append(name)
        if original is not None:
            return original(**kwargs)

    knobs.runtime.jit_post_compile_hook = compiled
    return original, state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("capacity", [1, 2, 3, 16])
@torch.inference_mode()
def test_talker_readiness_warmup_compiles_live_top_k_top_p(capacity):
    import numpy as np
    from triton import knobs
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.ops import topk_topp_triton
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    device = torch.device("cuda")
    reqs = RequestState(capacity, 128, 32, 0, 6562, device)
    base = Sampler(VllmConfig(), capacity, 6562, device, reqs)
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.mrv2_custom_sampler(base)
    caches = ("_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE", "_TRITON_SPLIT_CACHE")
    before = {name: dict(getattr(topk_topp_triton, name)) for name in caches}
    default_state = torch.cuda.get_rng_state().clone()
    names = {"_topk_topp_kernel", "_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel"}
    original, state = _record_late_compiles(knobs, names)
    try:
        talker.capture_auxiliary_graphs()
        # No persistent allocation and no default-generator draw.
        assert {name: dict(getattr(topk_topp_triton, name)) for name in caches} == before
        torch.testing.assert_close(torch.cuda.get_rng_state(), default_state, rtol=0, atol=0)
        state["live"] = True
        # The deployed Talker sampling parameters, admitted like live requests.
        for slot in range(capacity):
            base.add_request(slot, SamplingParams(temperature=0.8, top_p=0.85, top_k=25, seed=42))
        base.apply_staged_writes()
        for rows in range(1, capacity + 1):
            idx_np = np.arange(rows, dtype=np.intp)
            idx = torch.arange(rows, dtype=torch.int32, device=device)
            logits = torch.randn(rows, 6562, device=device)
            base.sampling_states.apply_top_k_top_p(logits, idx, idx_np)
        torch.accelerator.synchronize()
        assert not state["late"]
    finally:
        knobs.runtime.jit_post_compile_hook = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_thinker_readiness_warmup_compiles_live_gumbel_draw(use_fp64, model_dtype):
    from types import SimpleNamespace

    import numpy as np
    from triton import knobs
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45DuplexSampler

    device, vocab, capacity = torch.device("cuda"), 151_748, 4
    reqs = RequestState(capacity, 128, 32, 0, vocab, device)
    base = Sampler(VllmConfig(), capacity, vocab, device, reqs, use_fp64_gumbel=use_fp64)
    model = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=model_dtype)))
    sampler = MiniCPMO45DuplexSampler(base, model)
    default_state = torch.cuda.get_rng_state().clone()
    original, state = _record_late_compiles(knobs, {"_gumbel_sample_kernel"})
    try:
        sampler.warmup()
        torch.testing.assert_close(torch.cuda.get_rng_state(), default_state, rtol=0, atol=0)
        state["live"] = True
        for slot in range(capacity):
            base.add_request(slot, SamplingParams(temperature=0.0, seed=42))
        base.apply_staged_writes()
        for rows in range(1, capacity + 1):
            idx_np = np.arange(rows, dtype=np.intp)
            idx = torch.arange(rows, dtype=torch.int32, device=device)
            pos = torch.arange(rows, dtype=torch.int64, device=device)
            # Raw head logits, and the float32 copy made when a processor is active.
            for dtype in (model_dtype, torch.float32):
                logits = torch.randn(rows, vocab, device=device).to(dtype)
                base._sample_random(logits, idx, idx_np, pos, None, None, False)
        torch.accelerator.synchronize()
        assert not state["late"]
    finally:
        knobs.runtime.jit_post_compile_hook = original
