# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "prompts,positions,widths,expected",
    [
        ([[1, 2, 5], [5]], [2, 7], [1, 1], 2),
        ([[1, 2, 5], [5]], [1, 7], [1, 1], 0),
        ([[1, 2, 7], [5]], [2, 7], [1, 1], 0),
        ([None, [5]], [2, 7], [1, 1], 0),
        ([[5], [1, 2, 5]], [7, 2], [1, 1], 2),
        ([[1, 2, 5], [5]], [0, 12], [3, 1], 2),
        ([[1, 2, 5], [5]], [0, 12], [2, 1], 0),
    ],
)
def test_prompt_position_eligibility(monkeypatch, prompts, positions, widths, expected):
    r = object.__new__(OmniGPUModelRunner)
    seen = {}
    r.model = SimpleNamespace(
        config=SimpleNamespace(audio_async_prompt_mode=True, audio_mixed_direct_sampling=True),
        _audio_continuation_id=5,
        supports_omni_decode_step_metadata=True,
        update_decode_step_metadata=lambda **kw: seen.update(kw),
    )
    r.use_async_scheduling = True
    r.input_batch = SimpleNamespace(req_ids=["a", "b"], num_computed_tokens_cpu_tensor=torch.tensor(positions))
    r.requests = {rid: SimpleNamespace(prompt_token_ids=p) for rid, p in zip(["a", "b"], prompts)}
    ids = torch.full((sum(widths),), 99)
    r.input_ids = SimpleNamespace(gpu=ids, cpu=ids)
    r.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, widths[0], sum(widths)]))
    r.positions = torch.tensor(positions)
    r._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *a, **kw: torch.ones(2))
    r._model_forward(input_ids=ids)
    assert seen["audio_prompt_mode_rows"] == expected
    assert seen["cpu_input_tail_ids"] is None  # Never trust stale sampled CPU IDs.


@pytest.mark.parametrize("enabled", [False, True])
def test_model_metadata_reset(enabled):
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
        HiggsAudioV3TalkerForConditionalGeneration as C,
    )

    t = C.__new__(C)
    torch.nn.Module.__init__(t)
    t.config = SimpleNamespace(audio_async_prompt_mode=enabled, audio_mixed_direct_sampling=False)
    t._set_last_step_query_start_loc = lambda x: None
    t._sync_decode_state_with_batch = lambda x: None
    t.update_decode_step_metadata(audio_prompt_mode_rows=2)
    assert t._step_audio_mode_rows == (2 if enabled else 0)
    t.update_decode_step_metadata()
    assert t._step_audio_mode_rows == 0


@pytest.mark.parametrize("bad_state", [False, True])
def test_actual_state_guard_and_eos(bad_state):
    import importlib.util
    from pathlib import Path

    import vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker as mod

    root = Path(mod.__file__).parents[4]
    spec = importlib.util.spec_from_file_location(
        "helper_fixture", root / "tests/model_executor/models/higgs_audio_v3/test_higgs_audio_v3.py"
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    t = helpers.TestSamplerMethods()._make_batched_sampler_talker(2)
    t._restore_terminal_audio_rows = (
        mod.HiggsAudioV3TalkerForConditionalGeneration._restore_terminal_audio_rows.__get__(t)
    )
    t.config = SimpleNamespace(
        audio_mixed_direct_sampling=True, audio_async_prompt_mode=True, audio_full_sample_graph=False
    )
    t._resolve_token_ids = lambda: None
    t._audio_continuation_id = 99999
    t._eos_token_id = 151671
    t._last_logits_hidden = torch.zeros(2, 16)
    t._last_step_input_ids = torch.tensor([99999, 12345 if bad_state else 151671])
    t._last_step_query_start_loc = None
    t._decode_has_codes = torch.tensor([True, False])
    t._decode_generation_done = torch.tensor([False, False])
    t._decode_delay_count = torch.zeros(2, dtype=torch.long)
    t._decode_eoc_countdown = torch.full((2,), -1, dtype=torch.long)
    t._fast_audio_direct_rows = 0
    t._step_audio_tail_rows = 0
    t._step_audio_mode_rows = 2
    t._fast_audio_sampler_gpu_fallback_reason = lambda **kw: None
    t._audio_codebook_logits_from_rows = lambda hidden, rows, all_rows=False: torch.zeros(2, 8, 1026)
    t._apply_delay_pattern_masking_batched = lambda *a, **kw: None
    t._sample_audio_codes = lambda logits, *a, **kw: torch.zeros(logits.shape[0], dtype=torch.long)
    seen_codes = {}
    t._update_delay_state_batched = lambda *a, **kw: seen_codes.update(kw)

    def call():
        return mod.HiggsAudioV3TalkerForConditionalGeneration.sample(
            t, torch.zeros(2, 200000), SimpleNamespace(no_penalties=True)
        )

    if bad_state:
        with pytest.raises(RuntimeError, match="tail mismatch"):
            call()
    else:
        assert call().sampled_token_ids.tolist() == [[99999], [151671]]
        assert seen_codes["code_row_mask"].tolist() == [True, False]
