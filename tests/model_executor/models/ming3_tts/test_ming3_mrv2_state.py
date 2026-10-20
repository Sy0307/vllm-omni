# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.ming3_tts.model_state import Ming3ModelState, _LatentRowsSnapshot
from vllm_omni.model_executor.output_snapshot import RequestOutputSnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

CONTINUE, STOP = 7, 9
LATENT_DIM, HIDDEN = 4, 6


class _FakeFlow:
    """Flow head stand-in: latent and next embedding come from the condition row, stop probability from a table."""

    latent_patch_key = "ming3_latent_patch"
    has_patch_key = "ming3_has_patch"
    fm_dtype = torch.float32

    def __init__(self, stop_probs: dict[float, float]):
        self.ming_config = SimpleNamespace(latent_dim=LATENT_DIM)
        self.stop_probs = stop_probs
        self.calls: list[tuple[float, float, float, int]] = []

    def _sample_flow_rows(self, cond, history, generators, *, cfg_scale, sigma, temperature, check_finite=True):
        assert not check_finite
        self.calls.append((cfg_scale, sigma, temperature, cond.shape[0]))
        latent = cond[:, None, :LATENT_DIM].clone()
        stop = torch.tensor([self.stop_probs.get(float(row[0]), 0.1) for row in cond])
        return latent, cond.clone(), stop, None


def _state(slots: int, stop_probs: dict[float, float], *, max_steps=5, min_steps=2) -> Ming3ModelState:
    state = object.__new__(Ming3ModelState)
    state.llm = _FakeFlow(stop_probs)
    state.history = torch.zeros((slots, 1, LATENT_DIM))
    state.next_embed = torch.zeros((slots, HIDDEN))
    state.active = torch.ones(slots, dtype=torch.bool)
    state.min_steps = torch.full((slots,), min_steps, dtype=torch.long)
    state.max_steps = torch.full((slots,), max_steps, dtype=torch.long)
    state._sde = [(1.0, 0.25, 1.5)] * slots
    state._seeds = [None] * slots
    state._flow_generators = [None] * slots
    state._continue_token, state._stop_token = CONTINUE, STOP
    state._threshold = 0.5
    state._patch_size = 1
    state._partitioned = True
    state._step_vae = None
    state.draining = torch.zeros(slots, dtype=torch.bool)
    return state


def _batch(slots, *, computed, scheduled, prompt_len, seq_lens):
    n = len(slots)
    return SimpleNamespace(
        num_reqs=n,
        req_ids=[f"req-{slot}" for slot in slots],
        idx_mapping_np=np.asarray(slots),
        idx_mapping=torch.tensor(slots, dtype=torch.int32),
        num_computed_tokens_np=np.asarray(computed),
        num_scheduled_tokens=np.asarray(scheduled),
        prefill_len_np=np.asarray(prompt_len),
        logits_indices=torch.arange(n),
        seq_lens=torch.tensor(seq_lens),
    )


def _hidden(values: list[float]) -> torch.Tensor:
    # Row i's first feature identifies it; the rest is arbitrary but finite.
    return torch.tensor([[value] + [0.5] * (HIDDEN - 1) for value in values])


def _run(state, batch, hidden, prompt_len_by_slot):
    req_states = SimpleNamespace(prompt_len=SimpleNamespace(gpu=torch.tensor(prompt_len_by_slot)))
    _, snapshot = state.postprocess_model_output(hidden, batch, req_states)
    return state._tokens, snapshot


def test_stop_head_and_max_steps_decide_tokens_on_device():
    # Slot 0 emits its 3rd latent with a firing stop head; slot 1 its 5th (= max) with a quiet one;
    # slot 2 its 1st (below the minimum) with a firing stop head.
    state = _state(3, {1.0: 0.9, 2.0: 0.1, 3.0: 0.9})
    prompt = [10, 10, 10]
    batch = _batch([0, 1, 2], computed=[11, 13, 9], scheduled=[1, 1, 1], prompt_len=prompt, seq_lens=[12, 14, 10])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, 3.0]), prompt)

    assert tokens.tolist() == [STOP, STOP, CONTINUE]
    assert state.active.tolist() == [False, False, True]
    # Every live row emits the latent that triggered its decision.
    assert snapshot["ming3_has_patch"].tolist() == [True, True, True]
    torch.testing.assert_close(state.history[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(state.next_embed[:, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_step_after_stop_emits_nothing_and_keeps_state():
    state = _state(2, {1.0: 0.1, 2.0: 0.1})
    state.active[1] = False  # stopped last step; async scheduling ran it once more
    state.history[1] = 5.0
    prompt = [10, 10]
    batch = _batch([0, 1], computed=[11, 12], scheduled=[1, 1], prompt_len=prompt, seq_lens=[12, 13])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0]), prompt)

    assert tokens.tolist() == [CONTINUE, STOP]
    assert snapshot["ming3_has_patch"].tolist() == [True, False]
    assert state.history[1].eq(5.0).all()
    assert state.next_embed[1].eq(0.0).all()


def test_partial_prefill_and_non_finite_rows_are_not_emitted():
    state = _state(3, {1.0: 0.1, 2.0: 0.1})
    prompt = [10, 40, 10]
    # Row 1 is a non-final prefill chunk; row 2's flow output is not finite.
    batch = _batch([0, 1, 2], computed=[11, 0, 12], scheduled=[1, 16, 1], prompt_len=prompt, seq_lens=[12, 16, 13])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, float("nan")]), prompt)

    assert snapshot["ming3_has_patch"].tolist() == [True, False, False]
    assert tokens.tolist() == [CONTINUE, STOP, STOP]
    assert state.active.tolist() == [True, True, False]
    assert state.llm.calls == [(1.0, 0.25, 1.5, 2)]


def test_rows_with_different_sampling_controls_are_grouped():
    state = _state(3, {1.0: 0.1, 2.0: 0.1, 3.0: 0.1})
    state._sde = [(1.0, 0.25, 1.5), (2.0, 0.25, 1.5), (1.0, 0.25, 1.5)]
    prompt = [10, 10, 10]
    batch = _batch([0, 1, 2], computed=[11, 11, 11], scheduled=[1, 1, 1], prompt_len=prompt, seq_lens=[12, 12, 12])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, 3.0]), prompt)

    assert sorted(state.llm.calls) == [(1.0, 0.25, 1.5, 2), (2.0, 0.25, 1.5, 1)]
    assert tokens.tolist() == [CONTINUE] * 3
    torch.testing.assert_close(snapshot["ming3_latent_patch"][:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_partitioned_snapshot_rows_follow_request_order():
    latent = torch.arange(3 * LATENT_DIM, dtype=torch.float32).view(3, 1, LATENT_DIM)
    emitted = torch.tensor([True, False, True])
    snapshot = _LatentRowsSnapshot(latent, emitted, latent_key="lat", flag_key="flag", partitioned=True)
    host = snapshot.copy_to_cpu(lambda tensor: tensor.clone())

    assert isinstance(host, RequestOutputSnapshot)
    assert [row["flag"].tolist() for row in host.inter_stage] == [[True], [False], [True]]
    torch.testing.assert_close(host.inter_stage[2]["lat"], latent[2:3])


def test_sampler_features_fall_back_to_logits():
    state = _state(1, {})
    state._tokens = torch.tensor([CONTINUE])
    assert state.sample_determined_tokens(SimpleNamespace(num_reqs=1), sampler=object()) is None


def test_seeded_flow_noise_uses_a_fresh_generator_per_request():
    # As on the V1 runner, flow-head noise starts from the request seed
    # regardless of the draws its prompt-latent posterior made at prefill.
    state = _state(2, {})
    state._seeds = [43, None]
    first = state._generator(0, torch.device("cpu"))
    assert state._generator(0, torch.device("cpu")) is first
    assert first.initial_seed() == 43
    assert torch.equal(torch.randn(3, generator=first), torch.randn(3, generator=torch.Generator().manual_seed(43)))
    assert state._generator(1, torch.device("cpu")) is None


class _FakeStepVAE:
    """Records the single-stage decode calls; every decoded row yields 3 samples."""

    scale = 4

    def __init__(self):
        self.primed, self.steps = [], []

    def prime(self, slots, latents):
        self.primed.append(slots.tolist())

    def step(self, slots, positions, first, latents, last, valid):
        self.steps.append((slots.tolist(), positions.tolist(), first.tolist(), last.tolist(), valid.tolist()))
        rows = slots.shape[0]
        audio = torch.arange(rows * 5, dtype=torch.float32).view(rows, 5)
        start = torch.zeros(rows, dtype=torch.long)
        return audio, start, torch.where(valid, 3, 0)


def test_single_stage_rows_drain_one_step_before_stopping():
    import numpy as np

    from vllm_omni.model_executor.models.ming3_tts.model_state import _StepAudioOutput

    state = _state(2, {1.0: 0.9, 2.0: 0.1})
    state._step_vae = vae = _FakeStepVAE()
    state._audio_chunks = {}
    state._chunk_latents = (4, 25)
    state._samples_per_latent = 8
    state._vae_positions = np.zeros(2, dtype=np.int64)
    state._sample_rate = torch.tensor(44100)
    prompt = [10, 10]
    # Slot 0 decodes its 3rd latent with a firing stop head; slot 1 finishes its prompt (first latent).
    batch = _batch([0, 1], computed=[11, 8], scheduled=[1, 2], prompt_len=prompt, seq_lens=[12, 10])
    tokens, _ = _run(state, batch, _hidden([1.0, 2.0]), prompt)

    # Halting continues for one flush step; the first latent is held, not decoded.
    assert tokens.tolist() == [CONTINUE, CONTINUE]
    assert state.draining.tolist() == [True, False]
    assert vae.primed == [[1]]
    assert vae.steps[-1] == ([0], [0], [True], [False], [True])
    output = state.prepare_streaming_audio_output(batch, None, {})
    assert isinstance(output, _StepAudioOutput)
    assert [None if item is None else item["model_outputs"].tolist() for item in output.get_output()] == [
        [0.0, 1.0, 2.0],
        None,
    ]

    # Next step: slot 0 flushes and stops; slot 1 decodes its first output.
    batch = _batch([0, 1], computed=[12, 10], scheduled=[1, 1], prompt_len=prompt, seq_lens=[13, 11])
    tokens, _ = _run(state, batch, _hidden([1.0, 2.0]), prompt)
    assert tokens.tolist() == [STOP, CONTINUE]
    assert state.draining.tolist() == [False, False]
    assert vae.steps[-1] == ([0, 1], [4, 0], [False, True], [True, False], [True, True])
    # Both rows release their audio: slot 0 its final chunk, slot 1 its first.
    output = state.prepare_streaming_audio_output(batch, None, {}).get_output()
    assert [item["model_outputs"].tolist() for item in output] == [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]


def test_single_stage_audio_is_released_in_growing_chunks():
    from vllm_omni.model_executor.models.ming3_tts.model_state import _AudioChunks

    chunks = _AudioChunks(samples_per_latent=2, chunk_latents=(2, 3))
    pushes = [(torch.ones(1), False)] + [(torch.ones(2), False)] * 6 + [(torch.ones(1), True)]
    released = [None if (out := chunks.push(pcm, ended)) is None else out.numel() for pcm, ended in pushes]
    # First audio at once, then 2 and 3 latents (4 and 6 samples), then the remainder at the end.
    assert released == [1, None, 4, None, None, 6, None, 3]
