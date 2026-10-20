# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming3_tts import ming3_tts
from vllm_omni.model_executor.models.ming3_tts.ming3_tts import Ming3TTSForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

SPK, PATCH, START, END = 90, 91, 92, 93
DIM = 4
# Text, the speaker marker and its slot (an audio patch token), then five prompt-audio patches.
PROMPT = [10, 11, SPK, PATCH, 12, START, PATCH, PATCH, PATCH, PATCH, PATCH, END, 13, 14]


def _wrapper(monkeypatch, draws: list[int]) -> Ming3TTSForConditionalGeneration:
    wrapper = object.__new__(Ming3TTSForConditionalGeneration)
    wrapper.ming_config = SimpleNamespace(
        speaker_placeholder_token_id=SPK,
        audio_dummy_token_id=PATCH,
        audio_start_token_id=START,
        audio_end_token_id=END,
        history_patch_size=1,
        latent_dim=DIM,
    )
    wrapper._prompt_projection_graphs = None
    wrapper._prefetched_prompt_outputs = {}
    wrapper._prompt_request_id = None
    wrapper._prompt_cache_key = None
    wrapper._deferred_prompt_projections = None
    wrapper._chunked_prompt_patches = {}
    wrapper._prefetched_speakers = {}
    wrapper.__dict__["model"] = SimpleNamespace(
        fm_dtype=torch.float32,
        linear_proj_audio=lambda patches: patches * 2,
        project_speaker_embedding=lambda speaker: speaker * 10,
        set_initial_history=lambda request_id, history: None,
    )
    patches = torch.arange(5 * DIM, dtype=torch.float32).view(5, 1, DIM) + 1

    def resolve(_wrapper, info, **_):
        draws.append(1)
        return {"frames": patches.view(5, DIM), "patches": patches}

    monkeypatch.setattr(ming3_tts, "_resolve_prompt_latents", resolve)
    monkeypatch.setattr(ming3_tts, "resolve_speaker_embeddings", lambda _wrapper, info: [torch.full((DIM,), 0.5)])
    return wrapper


def _prefill(wrapper, bounds: list[int]) -> torch.Tensor:
    full = torch.tensor(PROMPT)
    chunks = []
    for start, end in zip(bounds, bounds[1:]):
        info = {
            "request_id": "req",
            "_omni_num_computed_tokens": start,
            "_omni_prompt_ids_cpu": full[start:end],
            "_omni_full_prompt_ids_cpu": full,
        }
        embeds = torch.zeros(end - start, DIM)
        _, embeds, _ = wrapper._prefill_preprocess(full[start:end], embeds, info)
        chunks.append(embeds)
    return torch.cat(chunks)


@pytest.mark.parametrize("bounds", [[0, 4, 14], [0, 3, 8, 14], [0, 7, 9, 12, 14], [0, 1, 2, 3, 4, 5, 6, 14]])
def test_chunked_prompt_prefill_matches_whole_prompt(monkeypatch, bounds):
    whole_draws: list[int] = []
    chunk_draws: list[int] = []
    expected = _prefill(_wrapper(monkeypatch, whole_draws), [0, len(PROMPT)])
    wrapper = _wrapper(monkeypatch, chunk_draws)
    actual = _prefill(wrapper, bounds)

    torch.testing.assert_close(actual, expected)
    assert expected[3].tolist() == [5.0] * DIM and expected[6].tolist() == [2.0, 4.0, 6.0, 8.0]
    # Latents are drawn once, with the first chunk, and released with the last.
    assert len(chunk_draws) == 1 and wrapper._chunked_prompt_patches == {}
