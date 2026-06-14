# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pi0.5 e2e gates: checkpoint smoke, LeRobot parity, and OpenPI serving.

The default test collection does not download weights. Pass
``--pi05-e2e-ckpt`` and ``--pi05-e2e-tokenizer`` to exercise the real
checkpoint path. The OpenPI serving test injects the tokenizer through
``model_config.tokenizer`` in a generated deploy overlay.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

try:
    from tests.helpers.mark import hardware_test
    from tests.pi05 import openpi_client_helper

    _HAS_SERVING_DEPS = True
except Exception:  # noqa: BLE001
    _HAS_SERVING_DEPS = False

    def hardware_test(**_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap

    openpi_client_helper = None  # type: ignore[assignment]


_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None

ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200


def _device(pytestconfig: pytest.Config) -> str:
    configured = pytestconfig.getoption("pi05_e2e_device")
    return configured or ("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(pytestconfig: pytest.Config) -> torch.dtype:
    dtype_str = pytestconfig.getoption("pi05_e2e_dtype")
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]


def _dtype_str(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("pi05_e2e_dtype"))


def _num_steps(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("pi05_e2e_num_steps"))


def _require_checkpoint(pytestconfig: pytest.Config) -> tuple[str, str]:
    model_path = pytestconfig.getoption("pi05_e2e_ckpt")
    tokenizer_path = pytestconfig.getoption("pi05_e2e_tokenizer")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Set --pi05-e2e-ckpt to a local pi05_base checkpoint directory.")
    if not tokenizer_path or not Path(tokenizer_path).is_dir():
        pytest.skip("Set --pi05-e2e-tokenizer to a local PaliGemma tokenizer directory.")
    return str(model_path), str(tokenizer_path)


def _blank_robot_obs() -> dict:
    return {
        "prompt": "pick up the red block and place it in the bin",
        "state": np.zeros(STATE_DIM, dtype=np.float32),
        "images": {
            "observation.images.base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation.images.left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation.images.right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        },
    }


@pytest.mark.full_model
@pytest.mark.diffusion
def test_pi05_pipeline_checkpoint_smoke(pytestconfig: pytest.Config):
    model_path, tokenizer_path = _require_checkpoint(pytestconfig)
    num_steps = _num_steps(pytestconfig)
    od_config = OmniDiffusionConfig(
        model=model_path,
        model_class_name="Pi05Pipeline",
        dtype=_dtype(pytestconfig),
        model_config={
            "num_inference_steps": num_steps,
            "tokenizer": tokenizer_path,
            "tokenizer_max_length": MAX_TOKEN_LEN,
            "image_feature_keys": [
                "observation.images.base_0_rgb",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            ],
        },
    )
    pipeline = Pi05Pipeline(od_config=od_config)
    req = OmniDiffusionRequest(
        prompts=["pi05 checkpoint smoke"],
        request_id="pi05-checkpoint-smoke",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": _blank_robot_obs(),
                "noise": np.zeros((1, ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
                "num_inference_steps": num_steps,
            }
        ),
    )

    out = pipeline.forward(req)

    actions = out.output["actions"]
    assert actions.shape == (ACTION_HORIZON, ACTION_DIM)
    assert np.isfinite(actions).all()


@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed.")
@pytest.mark.full_model
@pytest.mark.diffusion
def test_pi05_vllm_omni_vs_lerobot(pytestconfig: pytest.Config):
    model_path, _ = _require_checkpoint(pytestconfig)
    device = _device(pytestconfig)
    num_steps = _num_steps(pytestconfig)
    try:
        from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LeRobot pi0.5 reference implementation is unavailable: {exc!r}")

    from vllm_omni.diffusion.models.pi05.config import Pi05Config
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction

    policy = PI05Policy.from_pretrained(model_path, strict=True)
    policy.to(device)
    policy.config.device = device
    policy.eval()
    pre, _ = make_pi05_pre_post_processors(config=policy.config, dataset_stats=_dummy_dataset_stats())

    cfg = Pi05Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=num_steps,
        tokenizer_max_length=MAX_TOKEN_LEN,
        dtype=_dtype_str(pytestconfig),
    )
    model = Pi05ForActionPrediction(cfg).to(device).eval()
    _load_weights(model, model_path)

    batch = pre(_create_dummy_batch(device))
    images, image_masks = policy._preprocess_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    noise = torch.zeros(1, ACTION_HORIZON, ACTION_DIM, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref = policy.model.sample_actions(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            noise=noise,
            num_steps=num_steps,
        )
        got = model.sample_actions(
            images=images,
            image_masks=image_masks,
            tokens=lang_tokens,
            masks=lang_masks,
            noise=noise,
            num_steps=num_steps,
        )

    assert torch.allclose(got.float(), ref.float(), atol=float(pytestconfig.getoption("pi05_parity_atol")))


def _load_weights(model, model_path: str) -> None:
    import safetensors.torch

    state = safetensors.torch.load_file(str(Path(model_path) / "model.safetensors"))
    model.load_weights(list(state.items()))


def _dummy_dataset_stats() -> dict:
    image_stats = {
        "mean": torch.zeros(3, 224, 224),
        "std": torch.ones(3, 224, 224),
        "q01": torch.zeros(3, 224, 224),
        "q99": torch.ones(3, 224, 224),
    }
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "q01": -torch.ones(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "q01": -torch.ones(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "images": {
            "base_0_rgb": image_stats,
            "left_wrist_0_rgb": image_stats,
            "right_wrist_0_rgb": image_stats,
        },
    }


def _create_dummy_batch(device: str) -> dict:
    return {
        "observation.state": torch.zeros(1, STATE_DIM, dtype=torch.float32, device=device),
        "action": torch.zeros(1, ACTION_HORIZON, ACTION_DIM, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.left_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["pick up the red block and place it in the bin"],
    }


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.skipif(not _HAS_SERVING_DEPS, reason="serving stack / OpenPI deps unavailable")
def test_pi05_openpi_online(omni_server):
    try:
        openpi_client_helper.require_dependencies()
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    result = openpi_client_helper.run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        prompt="pick up the red block and place it in the bin",
        session_id="pi05-online-e2e",
        num_steps=2,
    )

    openpi_client_helper.validate_session_result(result)
