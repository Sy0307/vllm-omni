# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for pi0.5 VLA support.

These tests intentionally avoid model weights. They pin the contracts that
distinguish pi0.5 from pi0: robot state is normalized/discretized into the
language prompt, not passed as a separate suffix tensor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.config.stage_config import _PIPELINE_REGISTRY
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import get_extra_body_params
from vllm_omni.diffusion.models.pi05.config import Pi05Config
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    Pi05ImageProcessor,
    build_model_inputs,
    build_pi05_prompt,
    discretize_state,
    normalize_state,
)
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_LEROBOT_PI05_CFG = {
    "type": "pi05",
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "chunk_size": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "image_resolution": [224, 224],
    "tokenizer_max_length": 200,
    "dtype": "float32",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "training_only_key": "must be ignored",
}

_EXPECTED_CAMERA_ORDER = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]


class _RecordingTokenizer:
    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, text, *, padding, max_length, truncation, add_special_tokens, return_tensors):
        del padding, truncation, add_special_tokens, return_tensors
        self.texts.append(text)
        ids = list(range(10, 10 + min(len(text), max_length)))
        attn = [1] * len(ids)
        if len(ids) < max_length:
            ids.extend([0] * (max_length - len(ids)))
            attn.extend([0] * (max_length - len(attn)))
        return {"input_ids": ids, "attention_mask": attn}


def test_config_defaults_match_pi05_reference_contract():
    config = Pi05Config.from_model_config(None)

    assert config.chunk_size == 50
    assert config.max_action_dim == 32
    assert config.max_state_dim == 32
    assert config.num_inference_steps == 10
    assert config.image_resolution == (224, 224)
    assert config.tokenizer_max_length == 200
    assert config.max_cameras == 3
    assert config.state_num_bins == 256


def test_config_filters_training_keys_and_derives_camera_order():
    config = Pi05Config.from_model_config(_LEROBOT_PI05_CFG)

    assert not hasattr(config, "training_only_key")
    assert config.image_resolution == (224, 224)
    assert config.image_feature_keys == _EXPECTED_CAMERA_ORDER


def test_config_accepts_tokenizer_override():
    config = Pi05Config.from_model_config({**_LEROBOT_PI05_CFG, "tokenizer": "/tmp/paligemma"})

    assert config.tokenizer == "/tmp/paligemma"


def test_config_rejects_non_square_image_resolution():
    with pytest.raises(ValueError, match="square"):
        Pi05Config(image_resolution=(224, 256))


def test_normalize_state_supports_quantile_stats_and_clips():
    state = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    stats = {"q01": [-1.0] * 5, "q99": [1.0] * 5}

    out = normalize_state(state, max_state_dim=5, state_norm_stats=stats)

    assert np.allclose(out, [-1.0, -1.0, 0.0, 1.0, 1.0])


def test_normalize_state_rejects_unknown_stats_mode():
    with pytest.raises(ValueError, match="Unsupported pi0.5 state_norm_stats mode"):
        normalize_state(
            [0.0, 0.5],
            max_state_dim=2,
            state_norm_stats={"mode": "mystery"},
        )


def test_discretize_state_maps_clipped_range_to_256_bins():
    state = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    bins = discretize_state(state, num_bins=256)

    assert bins.tolist() == [0, 128, 255]


def test_build_pi05_prompt_serializes_state_bins_not_raw_floats():
    prompt = build_pi05_prompt(
        task="pick_up\nred_block",
        state=[-1.0, 0.0, 1.0],
        max_state_dim=3,
        state_norm_stats=None,
        state_num_bins=256,
    )

    assert prompt == "Task: pick up red block, State: 0 128 255;\nAction: "


def test_build_model_inputs_pads_cameras_and_encodes_state_into_prompt():
    config = Pi05Config.from_model_config(
        {
            **_LEROBOT_PI05_CFG,
            "image_resolution": [8, 8],
            "max_state_dim": 3,
            "tokenizer_max_length": 24,
        }
    )
    tokenizer = _RecordingTokenizer()
    robot_obs = {
        "prompt": "open drawer",
        "state": [-1.0, 0.0, 1.0],
        "images": {
            "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    }

    images, image_masks, lang_tokens, lang_masks = build_model_inputs(
        robot_obs,
        config,
        tokenizer,
        torch.device("cpu"),
    )

    assert len(images) == 3
    assert [mask.item() for mask in image_masks] == [True, False, False]
    assert images[0].shape == (1, 3, 8, 8)
    assert torch.equal(images[1], torch.full((1, 3, 8, 8), -1.0))
    assert lang_tokens.shape == (1, 24)
    assert lang_masks.shape == (1, 24)
    assert tokenizer.texts == ["Task: open drawer, State: 0 128 255;\nAction: "]


def test_build_model_inputs_compacts_left_padding_for_realtime_prefix_metadata():
    class LeftPaddingTokenizer:
        def __call__(self, text, *, padding, max_length, truncation, add_special_tokens, return_tensors):
            del text, padding, truncation, add_special_tokens, return_tensors
            return {
                "input_ids": [0, 0, 11, 12, 13] + [0] * (max_length - 5),
                "attention_mask": [0, 0, 1, 1, 1] + [0] * (max_length - 5),
            }

    config = Pi05Config.from_model_config(
        {
            **_LEROBOT_PI05_CFG,
            "image_resolution": [8, 8],
            "max_state_dim": 3,
            "tokenizer_max_length": 8,
        }
    )
    robot_obs = {
        "prompt": "open drawer",
        "state": [-1.0, 0.0, 1.0],
        "images": {
            "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    }

    _, _, lang_tokens, lang_masks, metadata = build_model_inputs(
        robot_obs,
        config,
        LeftPaddingTokenizer(),
        torch.device("cpu"),
        max_cameras=1,
        return_metadata=True,
    )

    assert lang_tokens[0, :3].tolist() == [11, 12, 13]
    assert lang_masks[0].tolist() == [True, True, True, False, False, False, False, False]
    assert metadata["prefix_valid_len"] == 256 + 3
    assert metadata["prefix_masks_contiguous"] is True


def test_build_model_inputs_ignores_top_level_metadata_when_images_are_flat():
    config = Pi05Config.from_model_config(
        {
            **_LEROBOT_PI05_CFG,
            "image_resolution": [8, 8],
            "max_state_dim": 3,
            "tokenizer_max_length": 24,
        }
    )
    tokenizer = _RecordingTokenizer()
    robot_obs = {
        "prompt": "open drawer",
        "state": [-1.0, 0.0, 1.0],
        "timestamp": 1234567890,
        "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    images, image_masks, _, _ = build_model_inputs(robot_obs, config, tokenizer, torch.device("cpu"))

    assert [mask.item() for mask in image_masks] == [True, False, False]
    assert images[0].shape == (1, 3, 8, 8)


def test_image_processor_rejects_empty_images_with_clear_error():
    processor = Pi05ImageProcessor(image_size=8)

    with pytest.raises(ValueError, match="non-empty image array"):
        processor.preprocess_single(np.zeros((0, 0, 3), dtype=np.uint8))


def test_pi05_pipeline_registry_entries_are_loadable():
    assert _PIPELINE_REGISTRY["pi05"].model_arch == "Pi05Pipeline"
    assert DiffusionModelRegistry._try_load_model_cls("Pi05Pipeline") is Pi05Pipeline

    od_config = OmniDiffusionConfig(model="dummy", model_class_name="Pi05Pipeline")
    post_process = get_diffusion_post_process_func(od_config)

    marker = object()
    assert post_process(marker) is marker


def test_pi05_extra_body_params_include_serving_backend_knobs():
    params = get_extra_body_params("Pi05Pipeline")

    assert {
        "robot_obs",
        "pi05_execution_backend",
        "execution_backend",
        "pi05_max_cameras",
        "pi05_realtime_max_cameras",
        "use_realtime_triton_decoder",
        "use_realtime_triton_prefix_encoder",
        "use_realtime_image_embed_cache",
    }.issubset(params)


def test_pi05_realtime_image_embed_cache_key_tracks_image_content():
    pipeline = object.__new__(Pi05Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = Pi05Config.from_model_config(
        {
            **_LEROBOT_PI05_CFG,
            "image_resolution": [8, 8],
            "image_feature_keys": ["observation.images.base_0_rgb"],
        }
    )
    pipeline._device = torch.device("cpu")
    pipeline.model = torch.nn.Module()
    pipeline.model.action_in_proj = torch.nn.Linear(1, 1)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    same_image = image.copy()
    changed_image = image.copy()
    changed_image[0, 0, 0] = 1

    first_key = pipeline._make_realtime_image_embed_cache_keys(
        robot_obs={"images": {"observation.images.base_0_rgb": image}},
        input_max_cameras=1,
    )
    same_key = pipeline._make_realtime_image_embed_cache_keys(
        robot_obs={"images": {"observation.images.base_0_rgb": same_image}},
        input_max_cameras=1,
    )
    changed_key = pipeline._make_realtime_image_embed_cache_keys(
        robot_obs={"images": {"observation.images.base_0_rgb": changed_image}},
        input_max_cameras=1,
    )

    assert same_key == first_key
    assert changed_key != first_key


def test_pi05_enrich_config_detects_lerobot_type(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: (
            None
            if path == "model_index.json"
            else {
                "type": "pi05",
                "input_features": _LEROBOT_PI05_CFG["input_features"],
            }
        ),
    )

    od_config = OmniDiffusionConfig(model="lerobot/pi05_base")
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "Pi05Pipeline"


def test_pi05_pipeline_forward_uses_prompt_encoded_state_without_state_tensor(monkeypatch):
    class FakeTokenizer(_RecordingTokenizer):
        pass

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.clear_image_cache_calls = 0
            self.action_in_proj = torch.nn.Linear(1, 1)

        def has_realtime_prefix_kv_cache(self, cache_key):
            del cache_key
            return False

        def has_realtime_prefix_emb_cache(self, cache_key):
            del cache_key
            return False

        def clear_realtime_prefix_caches(self):
            return None

        def clear_realtime_image_embed_cache(self):
            self.clear_image_cache_calls += 1

        def sample_actions(
            self,
            *,
            images,
            image_masks,
            tokens,
            masks,
            noise=None,
            num_steps=None,
            timing=None,
            direct_suffix=False,
            static_denoise=False,
            cuda_graph_denoise=False,
            cuda_graph_image_embed=False,
            cuda_graph_prefix=False,
            cuda_graph_prefix_denoise=False,
            torch_compile_image_embed=False,
            torch_compile_image_embed_fullgraph=False,
            torch_compile_prefix=False,
            torch_compile_prefix_fullgraph=False,
            torch_compile_suffix=False,
            torch_compile_suffix_fullgraph=False,
            torch_compile_denoise_loop=False,
            use_packed_prefix_qkv=False,
            use_packed_prefix_mlp=False,
            use_packed_qkv=False,
            use_packed_mlp=False,
            use_triton_qkv_rope=False,
            use_no_cat_suffix_attn=False,
            use_triton_final_head=False,
            use_triton_final_head_euler=False,
            use_realtime_triton_decoder=False,
            use_realtime_triton_prefix_encoder=False,
            use_realtime_triton_prefix_emb_cache=False,
            use_realtime_triton_prefix_kv_cache=False,
            use_realtime_image_embed_cache=False,
            realtime_prefix_cache_key=None,
            realtime_image_embed_cache_keys=None,
            prefix_valid_len=None,
            prefix_masks_contiguous=False,
            profile_nvtx=False,
            precompute_adarms=False,
            precompute_rope=False,
        ):
            del prefix_valid_len, prefix_masks_contiguous
            self.calls.append(
                {
                    "num_images": len(images),
                    "image_masks": [mask.item() for mask in image_masks],
                    "tokens_shape": tuple(tokens.shape),
                    "masks_shape": tuple(masks.shape),
                    "noise_shape": tuple(noise.shape) if noise is not None else None,
                    "num_steps": num_steps,
                    "has_timing": timing is not None,
                    "direct_suffix": direct_suffix,
                    "static_denoise": static_denoise,
                    "cuda_graph_denoise": cuda_graph_denoise,
                    "cuda_graph_image_embed": cuda_graph_image_embed,
                    "cuda_graph_prefix": cuda_graph_prefix,
                    "cuda_graph_prefix_denoise": cuda_graph_prefix_denoise,
                    "torch_compile_image_embed": torch_compile_image_embed,
                    "torch_compile_image_embed_fullgraph": torch_compile_image_embed_fullgraph,
                    "torch_compile_prefix": torch_compile_prefix,
                    "torch_compile_prefix_fullgraph": torch_compile_prefix_fullgraph,
                    "torch_compile_suffix": torch_compile_suffix,
                    "torch_compile_suffix_fullgraph": torch_compile_suffix_fullgraph,
                    "torch_compile_denoise_loop": torch_compile_denoise_loop,
                    "use_packed_prefix_qkv": use_packed_prefix_qkv,
                    "use_packed_prefix_mlp": use_packed_prefix_mlp,
                    "use_packed_qkv": use_packed_qkv,
                    "use_packed_mlp": use_packed_mlp,
                    "use_triton_qkv_rope": use_triton_qkv_rope,
                    "use_no_cat_suffix_attn": use_no_cat_suffix_attn,
                    "use_triton_final_head": use_triton_final_head,
                    "use_triton_final_head_euler": use_triton_final_head_euler,
                    "use_realtime_triton_decoder": use_realtime_triton_decoder,
                    "use_realtime_triton_prefix_encoder": use_realtime_triton_prefix_encoder,
                    "use_realtime_triton_prefix_emb_cache": use_realtime_triton_prefix_emb_cache,
                    "use_realtime_triton_prefix_kv_cache": use_realtime_triton_prefix_kv_cache,
                    "use_realtime_image_embed_cache": use_realtime_image_embed_cache,
                    "has_realtime_prefix_cache_key": realtime_prefix_cache_key is not None,
                    "has_realtime_image_embed_cache_keys": realtime_image_embed_cache_keys is not None,
                    "profile_nvtx": profile_nvtx,
                    "precompute_adarms": precompute_adarms,
                    "precompute_rope": precompute_rope,
                }
            )
            return torch.zeros(1, 2, 4, dtype=torch.float32)

        def _unnormalize_actions(self, actions):
            return actions + 1.0

    fake_model = FakeModel()
    monkeypatch.setattr(Pi05Pipeline, "_resolve_model_dir", staticmethod(lambda _model: None))
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: FakeTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: fake_model)

    od_config = OmniDiffusionConfig(
        model=None,
        model_class_name="Pi05Pipeline",
        dtype=torch.float32,
        model_config={
            **_LEROBOT_PI05_CFG,
            "image_resolution": [8, 8],
            "chunk_size": 2,
            "max_action_dim": 4,
            "max_state_dim": 3,
            "tokenizer_max_length": 24,
        },
    )
    pipeline = Pi05Pipeline(od_config=od_config)
    req = OmniDiffusionRequest(
        prompts=["ignored"],
        request_id="pi05-unit",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "prompt": "open drawer",
                    "state": [-1.0, 0.0, 1.0],
                    "images": {
                        "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                },
                "noise": np.zeros((1, 2, 4), dtype=np.float32),
                "num_inference_steps": 3,
                "return_timing": True,
            }
        ),
    )

    out = pipeline.forward(req)

    actions = out.output["actions"]
    assert actions.shape == (2, 4)
    assert np.allclose(actions, 1.0)
    timing = out.custom_output["policy_timing"]
    assert timing["preprocess_ms"] >= 0.0
    assert timing["sample_actions_ms"] >= 0.0
    assert timing["postprocess_ms"] >= 0.0
    assert timing["total_ms"] >= timing["sample_actions_ms"]
    assert fake_model.calls == [
        {
            "num_images": 3,
            "image_masks": [True, False, False],
            "tokens_shape": (1, 24),
            "masks_shape": (1, 24),
            "noise_shape": (1, 2, 4),
            "num_steps": 3,
            "has_timing": True,
            "direct_suffix": False,
            "static_denoise": False,
            "cuda_graph_denoise": False,
            "cuda_graph_image_embed": False,
            "cuda_graph_prefix": False,
            "cuda_graph_prefix_denoise": False,
            "torch_compile_image_embed": False,
            "torch_compile_image_embed_fullgraph": False,
            "torch_compile_prefix": False,
            "torch_compile_prefix_fullgraph": False,
            "torch_compile_suffix": False,
            "torch_compile_suffix_fullgraph": False,
            "torch_compile_denoise_loop": False,
            "use_packed_prefix_qkv": False,
            "use_packed_prefix_mlp": False,
            "use_packed_qkv": False,
            "use_packed_mlp": False,
            "use_triton_qkv_rope": False,
            "use_no_cat_suffix_attn": False,
            "use_triton_final_head": False,
            "use_triton_final_head_euler": False,
            "use_realtime_triton_decoder": False,
            "use_realtime_triton_prefix_encoder": False,
            "use_realtime_triton_prefix_emb_cache": False,
            "use_realtime_triton_prefix_kv_cache": False,
            "use_realtime_image_embed_cache": False,
            "has_realtime_prefix_cache_key": False,
            "has_realtime_image_embed_cache_keys": False,
            "profile_nvtx": False,
            "precompute_adarms": False,
            "precompute_rope": False,
        }
    ]
    fake_model.calls.clear()
    req.sampling_params.extra_args.update(
        {
            "direct_suffix": True,
            "static_denoise": True,
            "cuda_graph_denoise": True,
            "cuda_graph_image_embed": True,
            "cuda_graph_prefix": True,
            "cuda_graph_prefix_denoise": True,
            "torch_compile_image_embed": True,
            "torch_compile_image_embed_fullgraph": True,
            "torch_compile_prefix": True,
            "torch_compile_prefix_fullgraph": True,
            "torch_compile_suffix": True,
            "torch_compile_suffix_fullgraph": True,
            "torch_compile_denoise_loop": True,
            "use_packed_prefix_qkv": True,
            "use_packed_prefix_mlp": True,
            "use_packed_qkv": True,
            "use_packed_mlp": True,
            "use_triton_qkv_rope": True,
            "use_no_cat_suffix_attn": True,
            "use_triton_final_head": True,
            "use_triton_final_head_euler": True,
            "use_realtime_triton_decoder": True,
            "use_realtime_triton_prefix_encoder": True,
            "use_realtime_triton_prefix_emb_cache": True,
            "use_realtime_triton_prefix_kv_cache": True,
            "profile_nvtx": True,
            "precompute_adarms": True,
            "precompute_rope": True,
        }
    )

    pipeline.forward(req)

    fast_call = fake_model.calls[-1]
    assert fast_call["direct_suffix"] is True
    assert fast_call["static_denoise"] is True
    assert fast_call["cuda_graph_denoise"] is True
    assert fast_call["cuda_graph_image_embed"] is True
    assert fast_call["cuda_graph_prefix"] is True
    assert fast_call["cuda_graph_prefix_denoise"] is True
    assert fast_call["torch_compile_image_embed"] is True
    assert fast_call["torch_compile_image_embed_fullgraph"] is True
    assert fast_call["torch_compile_prefix"] is True
    assert fast_call["torch_compile_prefix_fullgraph"] is True
    assert fast_call["torch_compile_suffix"] is True
    assert fast_call["torch_compile_suffix_fullgraph"] is True
    assert fast_call["torch_compile_denoise_loop"] is True
    assert fast_call["use_packed_prefix_qkv"] is True
    assert fast_call["use_packed_prefix_mlp"] is True
    assert fast_call["use_packed_qkv"] is True
    assert fast_call["use_packed_mlp"] is True
    assert fast_call["use_triton_qkv_rope"] is True
    assert fast_call["use_no_cat_suffix_attn"] is True
    assert fast_call["use_triton_final_head"] is True
    assert fast_call["use_triton_final_head_euler"] is True
    assert fast_call["use_realtime_triton_decoder"] is True
    assert fast_call["use_realtime_triton_prefix_encoder"] is True
    assert fast_call["use_realtime_triton_prefix_emb_cache"] is True
    assert fast_call["use_realtime_triton_prefix_kv_cache"] is True
    assert fast_call["has_realtime_prefix_cache_key"] is True
    assert fast_call["profile_nvtx"] is True
    assert fast_call["precompute_adarms"] is True
    assert fast_call["precompute_rope"] is True

    fake_model.calls.clear()
    req.sampling_params.extra_args.update(
        {
            "robot_obs": {
                "prompt": "open drawer",
                "state": [-1.0, 0.0, 1.0],
                "images": {
                    "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    "observation.images.left_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    "observation.images.right_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                },
            },
            "pi05_realtime_max_cameras": 1,
            "return_timing": True,
        }
    )

    out = pipeline.forward(req)

    realtime_call = fake_model.calls[-1]
    realtime_timing = out.custom_output["policy_timing"]
    assert realtime_call["num_images"] == 1
    assert realtime_call["image_masks"] == [True]
    assert realtime_call["use_realtime_triton_decoder"] is True
    assert realtime_timing["pi05_input_max_cameras"] == 1
    assert realtime_timing["pi05_input_image_count"] == 1
    assert realtime_timing["pi05_input_image_masks"] == [True]

    def backend_req(backend: str, **extra_args) -> OmniDiffusionRequest:
        return OmniDiffusionRequest(
            prompts=["ignored"],
            request_id=f"pi05-backend-{backend}",
            sampling_params=OmniDiffusionSamplingParams(
                extra_args={
                    "robot_obs": {
                        "prompt": "open drawer",
                        "state": [-1.0, 0.0, 1.0],
                        "images": {
                            "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                        },
                    },
                    "noise": np.zeros((1, 2, 4), dtype=np.float32),
                    "return_timing": True,
                    "pi05_execution_backend": backend,
                    **extra_args,
                }
            ),
        )

    fake_model.calls.clear()
    pipeline.forward(backend_req("realtime_triton"))
    realtime_backend_call = fake_model.calls[-1]
    assert realtime_backend_call["use_realtime_triton_decoder"] is True
    assert realtime_backend_call["use_realtime_triton_prefix_encoder"] is False

    fake_model.calls.clear()
    pipeline.forward(backend_req("realtime_triton_prefix"))
    realtime_prefix_backend_call = fake_model.calls[-1]
    assert realtime_prefix_backend_call["use_realtime_triton_decoder"] is True
    assert realtime_prefix_backend_call["use_realtime_triton_prefix_encoder"] is True

    fake_model.calls.clear()
    pipeline.forward(backend_req("realtime_triton_prefix_image_cache"))
    realtime_prefix_image_cache_call = fake_model.calls[-1]
    assert realtime_prefix_image_cache_call["use_realtime_triton_decoder"] is True
    assert realtime_prefix_image_cache_call["use_realtime_triton_prefix_encoder"] is True
    assert realtime_prefix_image_cache_call["use_realtime_image_embed_cache"] is True
    assert realtime_prefix_image_cache_call["has_realtime_image_embed_cache_keys"] is True

    fake_model.calls.clear()
    assert fake_model.clear_image_cache_calls == 0
    pipeline.forward(backend_req("realtime_triton_prefix_image_cache", reset=True))
    assert fake_model.clear_image_cache_calls == 1

    fake_model.calls.clear()
    pipeline.forward(backend_req("realtime_triton_prefix_emb_cache"))
    realtime_prefix_emb_cache_call = fake_model.calls[-1]
    assert realtime_prefix_emb_cache_call["use_realtime_triton_decoder"] is True
    assert realtime_prefix_emb_cache_call["use_realtime_triton_prefix_encoder"] is True
    assert realtime_prefix_emb_cache_call["use_realtime_triton_prefix_emb_cache"] is True
    assert realtime_prefix_emb_cache_call["has_realtime_prefix_cache_key"] is True

    fake_model.calls.clear()
    pipeline.forward(backend_req("realtime_triton_prefix_cache"))
    realtime_prefix_cache_call = fake_model.calls[-1]
    assert realtime_prefix_cache_call["use_realtime_triton_decoder"] is True
    assert realtime_prefix_cache_call["use_realtime_triton_prefix_encoder"] is True
    assert realtime_prefix_cache_call["use_realtime_triton_prefix_kv_cache"] is True
    assert realtime_prefix_cache_call["has_realtime_prefix_cache_key"] is True


def test_pi05_prefix_emb_cache_hit_bypasses_preprocess_and_sample_actions(monkeypatch):
    class ExplodingTokenizer:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("prefix embedding cache hit should bypass tokenization")

    class CacheHitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(1, 1)
            self.cache_calls = []

        def has_realtime_prefix_emb_cache(self, cache_key):
            self.cache_calls.append(("has_emb", cache_key))
            return True

        def has_realtime_prefix_kv_cache(self, cache_key):
            self.cache_calls.append(("has_kv", cache_key))
            return False

        def clear_realtime_prefix_caches(self):
            self.cache_calls.append(("clear", None))

        def sample_actions_from_realtime_prefix_emb_cache(
            self,
            *,
            realtime_prefix_cache_key,
            noise=None,
            num_steps=None,
            timing=None,
            profile_nvtx=False,
            precompute_rope=False,
        ):
            self.cache_calls.append(
                (
                    "hit",
                    {
                        "has_key": realtime_prefix_cache_key is not None,
                        "noise_shape": tuple(noise.shape) if noise is not None else None,
                        "num_steps": num_steps,
                        "has_timing": timing is not None,
                        "profile_nvtx": profile_nvtx,
                        "precompute_rope": precompute_rope,
                    },
                )
            )
            if timing is not None:
                timing["realtime_prefix_emb_cache_hit"] = True
            return torch.zeros(1, 2, 4, dtype=torch.float32)

        def sample_actions(self, **kwargs):
            del kwargs
            raise AssertionError("prefix embedding cache hit should bypass sample_actions")

        def _unnormalize_actions(self, actions):
            return actions + 1.0

    fake_model = CacheHitModel()
    monkeypatch.setattr(Pi05Pipeline, "_resolve_model_dir", staticmethod(lambda _model: None))
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: ExplodingTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: fake_model)

    pipeline = Pi05Pipeline(
        od_config=OmniDiffusionConfig(
            model=None,
            model_class_name="Pi05Pipeline",
            dtype=torch.float32,
            model_config={
                **_LEROBOT_PI05_CFG,
                "image_resolution": [8, 8],
                "chunk_size": 2,
                "max_action_dim": 4,
                "max_state_dim": 3,
                "tokenizer_max_length": 24,
            },
        )
    )
    req = OmniDiffusionRequest(
        prompts=["ignored"],
        request_id="pi05-prefix-emb-cache-hit",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "prompt": "open drawer",
                    "state": [-1.0, 0.0, 1.0],
                    "images": {
                        "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                },
                "noise": np.zeros((1, 2, 4), dtype=np.float32),
                "num_inference_steps": 3,
                "return_timing": True,
                "profile_nvtx": True,
                "precompute_rope": True,
                "pi05_execution_backend": "realtime_triton_prefix_emb_cache",
            }
        ),
    )

    out = pipeline.forward(req)

    assert out.error is None
    assert np.allclose(out.output["actions"], 1.0)
    timing = out.custom_output["policy_timing"]
    assert timing["realtime_prefix_emb_cache_hit_before_preprocess"] is True
    assert timing["preprocess_ms"] == 0.0
    assert fake_model.cache_calls[0][0] == "has_emb"
    assert fake_model.cache_calls[1] == (
        "hit",
        {
            "has_key": True,
            "noise_shape": (1, 2, 4),
            "num_steps": 3,
            "has_timing": True,
            "profile_nvtx": True,
            "precompute_rope": True,
        },
    )


def test_pi05_prefix_kv_cache_hit_bypasses_preprocess_and_sample_actions(monkeypatch):
    class ExplodingTokenizer:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("prefix KV cache hit should bypass tokenization")

    class CacheHitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(1, 1)
            self.cache_calls = []

        def has_realtime_prefix_kv_cache(self, cache_key):
            self.cache_calls.append(("has_kv", cache_key))
            return True

        def has_realtime_prefix_emb_cache(self, cache_key):
            self.cache_calls.append(("has_emb", cache_key))
            return False

        def clear_realtime_prefix_caches(self):
            self.cache_calls.append(("clear", None))

        def sample_actions_from_realtime_prefix_kv_cache(
            self,
            *,
            realtime_prefix_cache_key,
            noise=None,
            num_steps=None,
            timing=None,
            profile_nvtx=False,
            precompute_rope=False,
        ):
            self.cache_calls.append(
                (
                    "hit",
                    {
                        "has_key": realtime_prefix_cache_key is not None,
                        "noise_shape": tuple(noise.shape) if noise is not None else None,
                        "num_steps": num_steps,
                        "has_timing": timing is not None,
                        "profile_nvtx": profile_nvtx,
                        "precompute_rope": precompute_rope,
                    },
                )
            )
            if timing is not None:
                timing["realtime_prefix_kv_cache_hit"] = True
            return torch.zeros(1, 2, 4, dtype=torch.float32)

        def sample_actions(self, **kwargs):
            del kwargs
            raise AssertionError("prefix KV cache hit should bypass sample_actions")

        def _unnormalize_actions(self, actions):
            return actions + 1.0

    fake_model = CacheHitModel()
    monkeypatch.setattr(Pi05Pipeline, "_resolve_model_dir", staticmethod(lambda _model: None))
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: ExplodingTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: fake_model)

    pipeline = Pi05Pipeline(
        od_config=OmniDiffusionConfig(
            model=None,
            model_class_name="Pi05Pipeline",
            dtype=torch.float32,
            model_config={
                **_LEROBOT_PI05_CFG,
                "image_resolution": [8, 8],
                "chunk_size": 2,
                "max_action_dim": 4,
                "max_state_dim": 3,
                "tokenizer_max_length": 24,
            },
        )
    )
    req = OmniDiffusionRequest(
        prompts=["ignored"],
        request_id="pi05-prefix-kv-cache-hit",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "prompt": "open drawer",
                    "state": [-1.0, 0.0, 1.0],
                    "images": {
                        "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                },
                "noise": np.zeros((1, 2, 4), dtype=np.float32),
                "num_inference_steps": 3,
                "return_timing": True,
                "profile_nvtx": True,
                "precompute_rope": True,
                "pi05_execution_backend": "realtime_triton_prefix_cache",
            }
        ),
    )

    out = pipeline.forward(req)

    assert out.error is None
    assert np.allclose(out.output["actions"], 1.0)
    timing = out.custom_output["policy_timing"]
    assert timing["realtime_prefix_kv_cache_hit_before_preprocess"] is True
    assert timing["preprocess_ms"] == 0.0
    assert fake_model.cache_calls[0][0] == "has_kv"
    assert fake_model.cache_calls[1] == (
        "hit",
        {
            "has_key": True,
            "noise_shape": (1, 2, 4),
            "num_steps": 3,
            "has_timing": True,
            "profile_nvtx": True,
            "precompute_rope": True,
        },
    )


def test_pi05_denoise_step_uses_direct_suffix_runner():
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction

    class FakeExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direct_calls = 0

        def forward(self, *args, **kwargs):
            raise AssertionError("denoise_step should not call generic pi0.5 forward")

        def forward_suffix_only(self, *, suffix_embs, **kwargs):
            del kwargs
            self.direct_calls += 1
            return suffix_embs + 1.0

    model = Pi05ForActionPrediction.__new__(Pi05ForActionPrediction)
    torch.nn.Module.__init__(model)
    model.action_horizon = 2
    model.action_dim = 3
    model.action_in_proj = torch.nn.Linear(3, 4)
    model.action_out_proj = torch.nn.Linear(4, 3)
    model.time_mlp_in = torch.nn.Linear(4, 4)
    model.time_mlp_out = torch.nn.Linear(4, 4)
    model.config = Pi05Config(chunk_size=2, max_action_dim=3)
    fake_expert = FakeExpert()
    model.paligemma_with_expert = fake_expert

    out = model.denoise_step(
        prefix_pad_masks=torch.ones(1, 5, dtype=torch.bool),
        past_key_values=[],
        x_t=torch.zeros(1, 2, 3),
        timestep=torch.ones(1),
        direct_suffix=True,
    )

    assert fake_expert.direct_calls == 1
    assert out.shape == (1, 2, 3)


def test_pi05_suffix_attention_no_cat_matches_cat_path():
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
        _suffix_attend,
        _suffix_attend_no_cat,
    )

    torch.manual_seed(0)
    query = torch.randn(1, 8, 3, 4)
    k_prefix = torch.randn(1, 1, 5, 4)
    v_prefix = torch.randn(1, 1, 5, 4)
    k_suffix = torch.randn(1, 1, 3, 4)
    v_suffix = torch.randn(1, 1, 3, 4)
    mask = torch.zeros(1, 1, 3, 8)
    mask[:, :, 1, 6:] = -10000.0

    ref = _suffix_attend(
        query,
        k_prefix,
        v_prefix,
        k_suffix,
        v_suffix,
        mask,
        num_kv_groups=8,
        scaling=0.5,
    )
    actual = _suffix_attend_no_cat(
        query,
        k_prefix,
        v_prefix,
        k_suffix,
        v_suffix,
        mask,
        scaling=0.5,
    )

    assert actual is not None
    assert torch.allclose(actual, ref, atol=1e-6, rtol=1e-6)


def test_pi05_realtime_fused_attention_matches_reference():
    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        is_available as realtime_triton_available,
    )

    if not realtime_triton_available():
        pytest.skip("realtime Triton attention requires CUDA and Triton")

    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        _attention_prefix_suffix_fused,
    )

    torch.manual_seed(0)
    rows_q = 6
    prefix_len = 5
    suffix_len = 2
    head_dim = 16
    total_len = prefix_len + suffix_len
    q = torch.randn(rows_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total_len, head_dim, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    prefix_mask = torch.tensor([1, 0, 1, 0, 1], device="cuda", dtype=torch.int32)

    _attention_prefix_suffix_fused[(rows_q,)](
        q,
        k,
        v,
        out,
        prefix_mask,
        rows_q,
        prefix_len,
        suffix_len,
        head_dim,
        0.5,
        block_m=1,
        block_n=16,
        block_d=16,
    )
    torch.accelerator.synchronize("cuda")

    logits = q.float() @ k.float().T * 0.5
    logits[:, :prefix_len] = torch.where(
        prefix_mask.bool()[None, :],
        logits[:, :prefix_len],
        torch.full_like(logits[:, :prefix_len], -float("inf")),
    )
    ref = torch.softmax(logits, dim=-1, dtype=torch.float32) @ v.float()

    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_pi05_realtime_split_attention_matches_reference():
    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        is_available as realtime_triton_available,
    )

    if not realtime_triton_available():
        pytest.skip("realtime Triton attention requires CUDA and Triton")

    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        _matmul_abt_scale,
        _matmul_small,
        _softmax_prefix_suffix,
    )

    torch.manual_seed(0)
    rows_q = 32
    prefix_len = 12
    valid_prefix_len = 9
    suffix_len = 4
    head_dim = 64
    total_len = prefix_len + suffix_len
    q = torch.randn(rows_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total_len, head_dim, device="cuda", dtype=torch.bfloat16)
    logits = torch.empty(rows_q, total_len, device="cuda", dtype=torch.float32)
    attn = torch.empty(rows_q, total_len, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    valid_prefix_len_tensor = torch.tensor([valid_prefix_len], device="cuda", dtype=torch.int32)

    _matmul_abt_scale[((rows_q + 31) // 32) * ((total_len + 31) // 32),](
        q,
        k,
        logits,
        rows_q,
        total_len,
        head_dim,
        0.5,
        block_m=32,
        block_n=32,
        block_k=64,
    )
    _softmax_prefix_suffix[((rows_q + 3) // 4,)](
        logits,
        rows_q,
        prefix_len,
        suffix_len,
        valid_prefix_len_tensor,
        attn,
        block_m=4,
        block_size=16,
    )
    _matmul_small[((rows_q + 31) // 32) * ((head_dim + 31) // 32),](
        attn,
        v,
        out,
        rows_q,
        total_len,
        head_dim,
        block_n=32,
        block_m=32,
        block_k=16,
    )
    torch.accelerator.synchronize("cuda")

    ref_logits = q.float() @ k.float().T * 0.5
    ref_logits[:, valid_prefix_len:prefix_len] = -float("inf")
    ref = torch.softmax(ref_logits, dim=-1, dtype=torch.float32) @ v.float()

    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_pi05_realtime_split_attention_rejects_oversized_softmax_window():
    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        _validate_prefix_suffix_softmax_window,
    )

    _validate_prefix_suffix_softmax_window(974, 50)
    with pytest.raises(ValueError, match="prefix_len \\+ suffix_len <= 1024"):
        _validate_prefix_suffix_softmax_window(975, 50)


def test_pi05_realtime_decoder_mlp_kernels_match_reference():
    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        is_available as realtime_triton_available,
    )

    if not realtime_triton_available():
        pytest.skip("realtime Triton MLP kernels require CUDA and Triton")

    from vllm_omni.diffusion.models.pi05.realtime_triton import (
        _matmul_small_gate,
        _matmul_small_res_gate,
    )

    torch.manual_seed(0)
    seq_len = 50

    # FFN gate/up tuned A100 launch: block_n=32, block_m=64, block_k=64.
    features = 128
    hidden = 128
    x = torch.randn(seq_len, features, device="cuda", dtype=torch.bfloat16) * 0.1
    gate_w = torch.randn(features, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    up_w = torch.randn(features, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty(seq_len, hidden, device="cuda", dtype=torch.bfloat16)

    _matmul_small_gate[((seq_len + 31) // 32, (hidden + 63) // 64)](
        x,
        gate_w,
        up_w,
        out,
        seq_len,
        features,
        hidden,
        block_n=32,
        block_m=64,
        block_k=64,
    )
    torch.accelerator.synchronize("cuda")

    gate = x.float() @ gate_w.float()
    up = x.float() @ up_w.float()
    ref = gate * torch.sigmoid(1.5957691216057308 * gate * (1 + 0.044715 * gate * gate))
    ref = ref * up
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

    # Attention O-proj launch: block_n=32, block_m=32, block_k=128.
    features = 128
    hidden = 64
    x = torch.randn(seq_len, features, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(features, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    residual = torch.randn(seq_len, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(seq_len, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty_like(residual)

    _matmul_small_res_gate[((seq_len + 31) // 32) * ((hidden + 31) // 32),](
        x,
        weight,
        out,
        residual,
        gate,
        seq_len,
        features,
        hidden,
        block_n=32,
        block_m=32,
        block_k=128,
    )
    torch.accelerator.synchronize("cuda")

    ref = residual.float() + (x.float() @ weight.float()) * gate.float()
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

    # FFN down tuned launch: block_n=16, block_m=64, block_k=256.
    features = 256
    hidden = 128
    x = torch.randn(seq_len, features, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(features, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    residual = torch.randn(seq_len, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(seq_len, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty_like(residual)

    _matmul_small_res_gate[((seq_len + 15) // 16) * ((hidden + 63) // 64),](
        x,
        weight,
        out,
        residual,
        gate,
        seq_len,
        features,
        hidden,
        block_n=16,
        block_m=64,
        block_k=256,
    )
    torch.accelerator.synchronize("cuda")

    ref = residual.float() + (x.float() @ weight.float()) * gate.float()
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_pi05_pipeline_warmup_without_obs_returns_zero_actions(monkeypatch):
    monkeypatch.setattr(Pi05Pipeline, "_resolve_model_dir", staticmethod(lambda _model: None))
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: _RecordingTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: torch.nn.Identity())

    pipeline = Pi05Pipeline(
        od_config=OmniDiffusionConfig(
            model=None,
            model_class_name="Pi05Pipeline",
            dtype=torch.float32,
            model_config={"chunk_size": 2, "max_action_dim": 4},
        )
    )
    req = OmniDiffusionRequest(
        prompts=["dummy run"],
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    out = pipeline.forward(req)

    assert out.output["actions"].shape == (2, 4)
    assert np.allclose(out.output["actions"], 0.0)


def test_pi05_pipeline_without_obs_does_not_treat_one_step_user_request_as_warmup(monkeypatch):
    monkeypatch.setattr(Pi05Pipeline, "_resolve_model_dir", staticmethod(lambda _model: None))
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: _RecordingTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: torch.nn.Identity())

    pipeline = Pi05Pipeline(
        od_config=OmniDiffusionConfig(
            model=None,
            model_class_name="Pi05Pipeline",
            dtype=torch.float32,
            model_config={"chunk_size": 2, "max_action_dim": 4},
        )
    )
    req = OmniDiffusionRequest(
        prompts=["real request missing obs"],
        request_id="pi05-user-request",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    out = pipeline.forward(req)

    assert out.output is None
    assert "requires sampling_params.extra_args['robot_obs']" in out.error


def test_pi05_pipeline_detects_sharded_safetensors_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(Pi05Pipeline, "_resolve_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(Pi05Pipeline, "_load_tokenizer", lambda self: _RecordingTokenizer())
    monkeypatch.setattr(Pi05Pipeline, "_initialize_model", lambda self: torch.nn.Identity())
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")

    pipeline = Pi05Pipeline(
        od_config=OmniDiffusionConfig(
            model=str(tmp_path),
            model_class_name="Pi05Pipeline",
            dtype=torch.float32,
        )
    )

    assert pipeline.has_real_checkpoint()
