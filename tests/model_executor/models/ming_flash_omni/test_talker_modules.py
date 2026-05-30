# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest

from transformers import Qwen2Config, Qwen2Model

from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    CFM,
    Aggregator,
    Attention,
    DiT,
    pack_qwen2_attention_qkv_projections,
)

torch = pytest.importorskip("torch")
pytest.importorskip("x_transformers")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_LATENT_DIM = 8
_PATCH_SIZE = 4
_HIS_PATCH_SIZE = 8
_LLM_HIDDEN = 16
_DIT_HIDDEN = 32
_AGG_HIDDEN = 32
_NUM_HEADS = 4
_DEPTH = 2
_STEPS = 5


def test_attention_packed_qkv_matches_separate_projections() -> None:
    attn = Attention(dim=_DIT_HIDDEN, heads=_NUM_HEADS).eval()
    x = torch.randn(2, 5, _DIT_HIDDEN)

    with torch.no_grad():
        separate = attn(x)
        attn.pack_qkv()
        packed = attn(x)

    assert attn.to_qkv is not None
    assert torch.allclose(separate, packed, rtol=1e-5, atol=1e-6)


def test_qwen2_attention_packed_qkv_matches_separate_projections() -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2Model(config).eval()
    inputs_embeds = torch.randn(2, 5, 32)

    with torch.no_grad():
        separate = model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
        packed_count = pack_qwen2_attention_qkv_projections(model)
        packed = model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state

    assert packed_count == 1
    assert torch.allclose(separate, packed, rtol=1e-5, atol=1e-6)


def _make_dit() -> DiT:
    return DiT(
        in_channels=_LATENT_DIM,
        hidden_size=_DIT_HIDDEN,
        depth=_DEPTH,
        num_heads=_NUM_HEADS,
        mlp_ratio=2.0,
        llm_cond_dim=_LLM_HIDDEN,
    )


def _make_aggregator() -> Aggregator:
    return Aggregator(
        in_channels=_LATENT_DIM,
        hidden_size=_AGG_HIDDEN,
        depth=_DEPTH,
        num_heads=_NUM_HEADS,
        mlp_ratio=2.0,
        llm_input_dim=_LLM_HIDDEN,
    )


class TestDiTDummyForward:
    """DiT with dummy weights runs forward + CFG-doubled forward."""

    def test_forward_shape(self) -> None:
        dit = _make_dit().eval()
        bsz = 2
        x = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.zeros(bsz)
        c = torch.randn(bsz, 1, _LLM_HIDDEN)
        latent_history = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            out = dit(x, t, c, latent_history)

        # Output preserves the concatenated (history + time/cond prefix + x)
        # token axis: history + 1 (time+cond) + patch.
        assert out.shape == (bsz, _HIS_PATCH_SIZE + 1 + _PATCH_SIZE, _LATENT_DIM)

    def test_forward_with_cfg_trims_to_patch(self) -> None:
        dit = _make_dit().eval()
        bsz = 1
        x = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.zeros(())
        c = torch.randn(bsz, 1, _LLM_HIDDEN)
        latent_history = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            out = dit.forward_with_cfg(x, t, c, latent_history)

        # CFG doubles the batch and trims the output to the patch window.
        assert out.shape == (2 * bsz, _PATCH_SIZE, _LATENT_DIM)


class TestAggregatorDummyForward:
    """Aggregator with dummy weights maps latent patch -> LLM hidden."""

    def test_forward_shape(self) -> None:
        agg = _make_aggregator().eval()
        bsz = 3
        gen_lat = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            out = agg(gen_lat)

        assert out.shape == (bsz, 1, _LLM_HIDDEN)

    def test_forward_is_finite(self) -> None:
        agg = _make_aggregator().eval()
        gen_lat = torch.randn(1, _PATCH_SIZE, _LATENT_DIM)
        with torch.no_grad():
            out = agg(gen_lat)
        assert torch.isfinite(out).all()


class TestCFMSampleDummy:
    """CFM.sample drives DiT.forward_with_cfg through the integration loop."""

    def test_sample_shape_and_finite(self) -> None:
        cfm = CFM(_make_dit(), steps=_STEPS, sway_sampling_coef=-1.0).eval()
        bsz = 1
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        # Grid used by the talker; must span [0, 1] inclusive.
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([2.0, 0.0, 0.0])  # cfg=2.0, sigma=0, temp=0
        sde_rnd = torch.zeros(_STEPS, bsz, _PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            out = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)

        assert out.shape == y0.shape
        assert torch.isfinite(out).all()

    def test_temperature_zero_allows_skipping_sde_noise(self) -> None:
        cfm = CFM(_make_dit(), steps=_STEPS, sway_sampling_coef=-1.0).eval()
        bsz = 1
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([2.0, 0.25, 0.0])
        sde_rnd = torch.randn(_STEPS, bsz, _PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            with_noise_input = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
            without_noise_input = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, None)

        assert torch.equal(with_noise_input, without_noise_input)

    def test_prepared_timesteps_match_inline_sway(self) -> None:
        cfm = CFM(_make_dit(), steps=_STEPS, sway_sampling_coef=-1.0).eval()
        bsz = 1
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([2.0, 0.25, 0.0])

        with torch.no_grad():
            inline = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, None)
            prepared = cfm.sample(
                llm_cond,
                lat_cond,
                y0,
                cfm.prepare_timesteps(t),
                sde_args,
                None,
                timesteps_are_swayed=True,
            )

        assert torch.equal(inline, prepared)

    def test_prepared_cfg_paths_match_original_cfg(self) -> None:
        cfm = CFM(_make_dit(), steps=_STEPS, sway_sampling_coef=-1.0).eval()
        bsz = 2
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([2.0, 0.25, 0.0])

        with torch.no_grad():
            original = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, None)
            cfm.use_prepared_cfg = True
            prepared = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, None)
            cfm.use_prepared_cfg = False
            cfm.use_preembedded_cfg = True
            preembedded = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, None)

        assert torch.equal(original, prepared)
        assert torch.equal(original, preembedded)

    def test_sample_zero_cfg_reduces_to_unguided(self) -> None:
        """With cfg=0 the guidance term drops, but output shape is still valid."""
        cfm = CFM(_make_dit(), steps=_STEPS, sway_sampling_coef=None).eval()
        bsz = 2
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.zeros(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([0.0, 0.0, 0.0])
        sde_rnd = torch.zeros(_STEPS, bsz, _PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            out = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)

        assert out.shape == (bsz, _PATCH_SIZE, _LATENT_DIM)
        assert torch.isfinite(out).all()


class TestTalkerPipelineDummyWiring:
    """End-to-end wiring of DiT -> CFM.sample -> Aggregator with dummy weights."""

    def test_cfm_then_aggregator(self) -> None:
        dit = _make_dit().eval()
        cfm = CFM(dit, steps=_STEPS, sway_sampling_coef=-1.0).eval()
        agg = _make_aggregator().eval()

        bsz = 1
        llm_cond = torch.randn(bsz, 1, _LLM_HIDDEN)
        lat_cond = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM)
        y0 = torch.randn(bsz, _PATCH_SIZE, _LATENT_DIM)
        t = torch.linspace(0.0, 1.0, _STEPS + 1)
        sde_args = torch.tensor([2.0, 0.0, 0.0])
        sde_rnd = torch.zeros(_STEPS, bsz, _PATCH_SIZE, _LATENT_DIM)

        with torch.no_grad():
            gen_lat = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
            agg_out = agg(gen_lat)

        assert gen_lat.shape == (bsz, _PATCH_SIZE, _LATENT_DIM)
        assert agg_out.shape == (bsz, 1, _LLM_HIDDEN)
        assert torch.isfinite(agg_out).all()
