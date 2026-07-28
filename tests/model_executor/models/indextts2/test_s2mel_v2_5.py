# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vllm_omni.model_executor.models.indextts2 import indextts2_s2mel_decoder
from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import (
    BASECFM,
)


class _FakeEnhancedCodec:
    downsample_scale = 2

    def __init__(self):
        self.decode_calls = 0

    def decode(self, codes):
        self.decode_calls += 1
        return torch.ones(codes.shape[0], codes.shape[1] * 2, 4)


class _FakeS2Mel:
    def forward_gpt(self, latent):
        return torch.full((*latent.shape[:-1], 4), 2.0)


class _FakeBigVGAN:
    def __init__(self):
        self.events = []

    def remove_weight_norm(self):
        self.events.append("remove_weight_norm")

    def to(self, *, device, dtype):
        self.events.append(("to", device, dtype))
        return self

    def eval(self):
        self.events.append("eval")
        return self

    def parameters(self):
        return []


class _NoiseEchoCFM(BASECFM):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.in_channels = 2

    def solve_euler(
        self,
        x,
        x_lens,
        prompt,
        mu,
        style,
        f0,
        t_span,
        inference_cfg_rate=0.5,
    ):
        return x


def _make_decoder(*, use_gpt_latent: bool):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.use_gpt_latent = use_gpt_latent
    decoder.semantic_codec_type = "enhanced"
    decoder.s2mel = _FakeS2Mel()
    return decoder


def test_v25_code_only_semantic_embedding_uses_enhanced_decode():
    decoder = _make_decoder(use_gpt_latent=False)
    codec = _FakeEnhancedCodec()

    result = decoder._build_semantic_embedding(
        codec=codec,
        mel_codes=torch.tensor([[1, 2, 3]]),
        latent=None,
        dtype=torch.float32,
    )

    assert codec.decode_calls == 1
    assert result.shape == (1, 6, 4)
    assert torch.equal(result, torch.ones(1, 6, 4))


def test_v25_latent_mode_upsamples_projected_latent_before_addition():
    decoder = _make_decoder(use_gpt_latent=True)
    codec = _FakeEnhancedCodec()

    result = decoder._build_semantic_embedding(
        codec=codec,
        mel_codes=torch.tensor([[1, 2, 3]]),
        latent=torch.ones(1, 3, 8),
        dtype=torch.float32,
    )

    assert result.shape == (1, 6, 4)
    assert torch.equal(result, torch.full((1, 6, 4), 3.0))


def test_s2mel_payload_latent_policy_must_match_stage_config():
    decoder = _make_decoder(use_gpt_latent=False)

    decoder._validate_payload_policy([{"use_gpt_latent": False}])
    with pytest.raises(ValueError, match="use_gpt_latent"):
        decoder._validate_payload_policy([{"use_gpt_latent": True}])


def test_cfm_inference_uses_explicit_initial_noise():
    cfm = _NoiseEchoCFM()
    initial_noise = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

    result = cfm.inference(
        mu=torch.zeros(1, 6, 4),
        x_lens=torch.tensor([6]),
        prompt=torch.zeros(1, 80, 0),
        style=torch.zeros(1, 192),
        f0=None,
        n_timesteps=1,
        temperature=0.5,
        initial_noise=initial_noise,
    )

    assert torch.equal(result, initial_noise * 0.5)


def test_seeded_cfm_noise_is_request_local_and_batch_order_independent():
    kwargs = {
        "channels": 2,
        "length": 6,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }

    noise_40_42 = IndexTTS2S2MelDecoder._sample_cfm_noise(
        seeds=[40, 42],
        **kwargs,
    )
    torch.randn(257)
    noise_42_40 = IndexTTS2S2MelDecoder._sample_cfm_noise(
        seeds=[42, 40],
        **kwargs,
    )

    assert noise_40_42 is not None
    assert noise_42_40 is not None
    assert torch.equal(noise_40_42[0], noise_42_40[1])
    assert torch.equal(noise_40_42[1], noise_42_40[0])
    assert not torch.equal(noise_40_42[0], noise_40_42[1])


def test_unseeded_cfm_noise_preserves_upstream_global_rng_behavior():
    assert (
        IndexTTS2S2MelDecoder._sample_cfm_noise(
            seeds=[None, None],
            channels=2,
            length=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )


def test_bigvgan_folds_weight_norm_before_low_precision_cast(monkeypatch):
    from vllm_omni.model_executor.models.indextts2.s2mel.modules import bigvgan

    model = _FakeBigVGAN()
    monkeypatch.setattr(
        bigvgan.BigVGAN,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_patch_bigvgan_compat",
        lambda _cls: None,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_repo_prefetch_lock",
        lambda _name: None,
    )
    indextts2_s2mel_decoder._bigvgan_models.clear()

    loaded = indextts2_s2mel_decoder._load_bigvgan(
        "fake-bigvgan",
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert loaded is model
    assert model.events == [
        "remove_weight_norm",
        ("to", torch.device("cpu"), torch.bfloat16),
        "eval",
    ]


def test_v25_full_mask_fast_path_is_guarded_independently_of_cuda_graph():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.s2mel_dit_cuda_graph = False
    layers = [SimpleNamespace(attention=SimpleNamespace()) for _ in range(13)]
    estimator = SimpleNamespace(
        transformer=SimpleNamespace(layers=layers),
    )

    decoder._set_dit_full_mask_fast_path(estimator, enabled=True)
    decoder._enable_dit_cuda_graph(estimator)

    assert all(layer.attention._assume_full_mask is True for layer in layers)
    assert not hasattr(estimator, "_cuda_graph_runner")

    decoder._set_dit_full_mask_fast_path(estimator, enabled=False)

    assert all(layer.attention._assume_full_mask is False for layer in layers)


def test_v25_ref_mel_lengths_come_from_unpadded_shapes():
    ref_mels = [
        torch.zeros(80, 5),
        torch.ones(80, 3),
    ]

    assert IndexTTS2S2MelDecoder._ref_mel_lengths(ref_mels) == [5, 3]
    assert IndexTTS2S2MelDecoder._ref_mel_lengths([None, ref_mels[1]]) == []


def test_v25_resolved_vocoder_source_is_cached(monkeypatch):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.model_path = "/models/indextts25"
    decoder.config = SimpleNamespace(vocoder={"name": "fake-bigvgan"})
    decoder._resolved_vocoder_source = None
    calls = []

    def fake_resolve(model_path, vocoder_name):
        calls.append((model_path, vocoder_name))
        return "/models/indextts25/hf_cache/bigvgan"

    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_resolve_bigvgan_source",
        fake_resolve,
    )

    first = decoder._get_resolved_vocoder_source()
    second = decoder._get_resolved_vocoder_source()

    assert first == "/models/indextts25/hf_cache/bigvgan"
    assert second == first
    assert calls == [("/models/indextts25", "fake-bigvgan")]


@pytest.mark.parametrize(
    "recipe_name",
    ["indextts2_5.yaml", "indextts2_5_latent.yaml"],
)
def test_v25_recipe_enables_only_exact_shape_dit_graph(recipe_name):
    repo_root = Path(__file__).parents[4]
    recipe_path = repo_root / "vllm_omni" / "deploy" / recipe_name
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 1)
    overrides = stage1["hf_overrides"]

    assert overrides["s2mel_dit_cuda_graph"] is True
    assert overrides["s2mel_dit_cuda_graph_max_graphs"] == 4
    assert overrides.get("s2mel_vocoder_cuda_graph", False) is False
