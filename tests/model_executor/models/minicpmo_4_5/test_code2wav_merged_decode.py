# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merged Code2Wav CFM decode: requests at different chunk positions in one call.

Each merged row must reproduce its own bucketed ``_decode_cfm`` call: the same
valid frames (up to float rounding of the larger batch) and the same cache it
carries into its next chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _FakeToken2Wav, _forward, _info, _model
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav, MergedRows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MEL = 4
_STEPS = 2


class _Decoder(nn.Module):
    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.randn(1, _MEL, 400, generator=torch.Generator().manual_seed(3)))


def _eager_graph(estimator):
    """Stand-in for the CUDA-graph wrapper: same call, eager, so bucketing is on."""

    def replay(estimator_input, time_emb, cnn_cache, att_cache, cnn_out, att_out, attn_mask=None):
        result = estimator.blocks_forward_chunk(
            estimator_input, time_emb, attn_mask, cnn_cache, att_cache, cnn_out, att_out
        )
        return result, cnn_out, att_out

    return SimpleNamespace(enabled=True, replay=replay)


def _backend():
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    estimator = decoder_dit.DiT(
        in_channels=4 * _MEL, out_channels=_MEL, depth=2, num_heads=2, head_dim=8, hidden_size=16
    )
    with torch.no_grad():
        for parameter in estimator.parameters():
            parameter.normal_(0.0, 0.3)
    token2wav = _FakeToken2Wav()
    token2wav.n_timesteps = _STEPS
    token2wav.flow.decoder = _Decoder(estimator.eval())
    backend = BatchedToken2Wav(token2wav)
    backend._cfm_graph_wrapper = _eager_graph(estimator)
    backend._cfm_graph_bucket_frames = 16
    return backend, estimator


def _row_inputs(frames: int, cache_len: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, _MEL, frames, generator=generator)
    speakers = torch.randn(1, _MEL, generator=generator)
    cnn = torch.randn(_STEPS, 2, 2, 2 * 16, 2, generator=generator)
    att = torch.randn(_STEPS, 2, 2, 2, cache_len, 16, generator=generator)
    return mu, speakers, cnn, att


def test_merged_cfm_rows_match_their_own_bucketed_calls():
    backend, _ = _backend()
    layout = [(50, 30, False), (50, 46, False), (20, 46, True)]
    inputs = [_row_inputs(frames, cache_len, seed) for seed, (frames, cache_len, _) in enumerate(layout)]

    separate = []
    with torch.inference_mode():
        for (frames, _, _), (mu, speakers, cnn, att) in zip(layout, inputs, strict=True):
            separate.append(
                backend._decode_cfm(
                    mu, speakers, torch.zeros_like(mu), cnn_cache=cnn, att_cache=att.reshape(_STEPS, 2, 2, 2, -1, 16)
                )
            )

        batch = len(layout)
        longest_frames = max(frames for frames, _, _ in layout)
        longest_cache = max(cache_len for _, cache_len, _ in layout)
        mu = torch.zeros(batch, _MEL, longest_frames)
        cnn = torch.cat([row[2][:, :, 0:1] for row in inputs] + [row[2][:, :, 1:2] for row in inputs], dim=2)
        att = torch.zeros(_STEPS, 2, 2 * batch, 2, longest_cache, 16)
        for index, ((frames, cache_len, _), row) in enumerate(zip(layout, inputs, strict=True)):
            mu[index, :, :frames] = row[0][0]
            att[:, :, index, :, :cache_len] = row[3][:, :, 0]
            att[:, :, batch + index, :, :cache_len] = row[3][:, :, 1]
        speakers = torch.cat([row[1] for row in inputs], dim=0)
        rows = MergedRows(frames=tuple(f for f, _, _ in layout), offsets=tuple(c for _, c, _ in layout))
        merged_x, merged_cnn, merged_att = backend._decode_cfm(
            mu, speakers, torch.zeros_like(mu), cnn_cache=cnn, att_cache=att, rows=rows
        )

    width = merged_att.shape[4] - longest_cache
    assert width == 64
    for index, ((frames, cache_len, last), (x, row_cnn, row_att)) in enumerate(zip(layout, separate, strict=True)):
        torch.testing.assert_close(merged_x[index, :, :frames], x[0], rtol=1e-4, atol=1e-5)
        if last:
            continue
        # The cache this row carries forward has its own layout and values.
        assert row_att.shape[4] == width + cache_len
        merged_row_att = torch.stack(
            (merged_att[:, :, index, :, : width + cache_len], merged_att[:, :, batch + index, :, : width + cache_len]),
            dim=2,
        )
        torch.testing.assert_close(merged_row_att, row_att, rtol=1e-4, atol=1e-5)
        merged_row_cnn = torch.stack((merged_cnn[:, :, index], merged_cnn[:, :, batch + index]), dim=2)
        torch.testing.assert_close(merged_row_cnn, row_cnn, rtol=1e-4, atol=1e-5)


def test_can_merge_requires_matching_padded_width_for_continuing_rows():
    backend, _ = _backend()
    backend._pre_lookahead_len = lambda: 3  # type: ignore[method-assign]
    backend._upsample_stride = lambda: 2  # type: ignore[method-assign]
    backend._max_encode_token_frames = lambda states: 1024  # type: ignore[method-assign]

    def state(cache_len):
        return SimpleNamespace(
            flow_cache={
                "estimator_att_cache": torch.zeros(_STEPS, 2, 2, 2, cache_len, 16),
                "estimator_cnn_cache": torch.zeros(_STEPS, 2, 2, 32, 2),
            }
        )

    states = [state(30), state(46), state(46)]
    # 28 tokens -> 50 frames -> padded to 64; a final 16-token chunk -> 32 frames.
    assert backend.can_merge([28, 28, 16], [False, False, True], states)
    # A continuing row that pads to a narrower width than the batch cannot merge.
    assert not backend.can_merge([28, 20, 16], [False, False, True], states)
    # A longer final chunk widens the batch past a continuing row's own width.
    assert not backend.can_merge([28, 28, 36], [False, False, True], states)
    # Bucketing off: no common grid to merge onto.
    backend._cfm_graph_bucket_frames = 0
    assert not backend.can_merge([28, 28, 16], [False, False, True], states)


def test_code2wav_keeps_duplex_streams_on_bucketed_decode(monkeypatch):
    model, _ = _model()
    monkeypatch.setattr(model.backend, "can_merge", lambda *args: pytest.fail("duplex rows must not merge"))
    infos = [_info("a", 0, [10, 11, 12]), _info("b", 0, [13, 14])]
    for info in infos:
        info["meta"]["duplex_epoch"] = 0
        info["meta"]["duplex_turn_id"] = 0
    output = _forward(model, infos)
    assert all(audio.numel() > 0 for audio in output.multimodal_outputs["model_outputs"])
