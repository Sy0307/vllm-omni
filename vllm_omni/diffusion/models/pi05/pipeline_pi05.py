# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pi0.5 VLA pipeline for vllm-omni.

pi0.5 uses the same serving contract as pi0: the OpenPI realtime layer forwards
raw robot observations through ``sampling_params.extra_args["robot_obs"]`` and
the diffusion pipeline returns a continuous action chunk under
``DiffusionOutput.output["actions"]``. The pi0.5-specific preprocessing lives in
``processor_pi05``: robot state is normalized, discretized, and serialized into
the language prompt instead of being passed as a separate model tensor.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import build_model_inputs
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"


def _pi05_post_process(x):
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return _pi05_post_process


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


def _update_hash_with_array(hasher: hashlib._Hash, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        arr = value.detach()
        if arr.device.type != "cpu":
            arr = arr.cpu()
        np_arr = arr.contiguous().numpy()
    else:
        np_arr = np.asarray(value)
    np_arr = np.ascontiguousarray(np_arr)
    hasher.update(str(np_arr.dtype).encode("utf-8"))
    hasher.update(str(tuple(np_arr.shape)).encode("utf-8"))
    hasher.update(memoryview(np_arr).cast("B"))


def _update_hash_with_value(hasher: hashlib._Hash, value: Any) -> None:
    if value is None:
        hasher.update(b"<none>")
    elif isinstance(value, str):
        hasher.update(b"str:")
        hasher.update(value.encode("utf-8", errors="surrogatepass"))
    elif isinstance(value, bytes):
        hasher.update(b"bytes:")
        hasher.update(value)
    elif isinstance(value, (bool, int, float, np.number)):
        hasher.update(f"scalar:{value!r}".encode())
    elif isinstance(value, torch.Tensor | np.ndarray):
        hasher.update(b"array:")
        _update_hash_with_array(hasher, value)
    elif isinstance(value, dict):
        hasher.update(b"dict:{")
        for key in sorted(value):
            _update_hash_with_value(hasher, key)
            _update_hash_with_value(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, list | tuple):
        try:
            hasher.update(b"arraylike:")
            _update_hash_with_array(hasher, value)
        except Exception:  # noqa: BLE001
            hasher.update(b"seq:[")
            for item in value:
                _update_hash_with_value(hasher, item)
            hasher.update(b"]")
    else:
        try:
            hasher.update(b"arraylike:")
            _update_hash_with_array(hasher, value)
        except Exception:  # noqa: BLE001
            hasher.update(repr(value).encode("utf-8", errors="backslashreplace"))


@contextmanager
def _skip_hf_weight_initialization():
    from transformers.modeling_utils import PreTrainedModel

    original_initialize_weights = PreTrainedModel.initialize_weights
    original_reset_parameters = {
        cls: cls.reset_parameters
        for cls in (nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.Linear)
        if hasattr(cls, "reset_parameters")
    }
    PreTrainedModel.initialize_weights = lambda self: None
    for cls in original_reset_parameters:
        cls.reset_parameters = lambda self: None
    try:
        yield
    finally:
        PreTrainedModel.initialize_weights = original_initialize_weights
        for cls, reset_parameters in original_reset_parameters.items():
            cls.reset_parameters = reset_parameters


PI05_PIPELINE = PipelineConfig(
    model_type="pi05",
    model_arch="Pi05Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="Pi05Pipeline",
        ),
    ),
)


class Pi05Pipeline(nn.Module):
    """pi0.5 VLA pipeline: raw robot obs -> continuous action chunk."""

    EXTRA_BODY_PARAMS = frozenset(
        {
            "robot_obs",
            "session_id",
            "reset",
            "pi05_execution_backend",
            "execution_backend",
            "pi05_max_cameras",
            "pi05_realtime_max_cameras",
            "num_inference_steps",
            "noise",
            "return_timing",
            "profile_nvtx",
            "direct_suffix",
            "static_denoise",
            "cuda_graph_denoise",
            "cuda_graph_image_embed",
            "cuda_graph_prefix",
            "cuda_graph_prefix_denoise",
            "torch_compile_image_embed",
            "torch_compile_image_embed_fullgraph",
            "torch_compile_prefix",
            "torch_compile_prefix_fullgraph",
            "torch_compile_suffix",
            "torch_compile_suffix_fullgraph",
            "torch_compile_denoise_loop",
            "use_packed_prefix_qkv",
            "use_packed_prefix_mlp",
            "use_packed_qkv",
            "use_packed_mlp",
            "use_triton_qkv_rope",
            "use_no_cat_suffix_attn",
            "use_triton_final_head",
            "use_triton_final_head_euler",
            "use_realtime_triton_decoder",
            "use_realtime_triton_prefix_encoder",
            "use_realtime_image_embed_cache",
            "use_realtime_triton_prefix_emb_cache",
            "use_realtime_triton_prefix_kv_cache",
            "pi05_prefix_cache_key",
            "precompute_adarms",
            "precompute_rope",
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        self.tokenizer_source = str(self.config.tokenizer or self._resolve_tokenizer_source())
        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device()

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        if not model:
            return None
        if os.path.isdir(model):
            return model
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        if od_config.model_config:
            config = Pi05Config.from_model_config(dict(od_config.model_config))
            if self.model_dir and (not config.image_feature_keys or config.norm_stats is None):
                ckpt = Pi05Config.from_pretrained(self.model_dir)
                if not config.image_feature_keys:
                    config.image_feature_keys = ckpt.image_feature_keys
                    if not config.input_features:
                        config.input_features = ckpt.input_features
                if config.norm_stats is None:
                    config.norm_stats = ckpt.norm_stats
                    config.state_norm_stats = ckpt.state_norm_stats
            return config
        if self.model_dir:
            return Pi05Config.from_pretrained(self.model_dir)
        return Pi05Config()

    def _resolve_tokenizer_source(self) -> str:
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        dt = od_config.dtype
        if isinstance(dt, torch.dtype):
            return dt
        return getattr(torch, str(dt).split(".")[-1], torch.float32)

    @staticmethod
    def _resolve_device() -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "not initialized" not in message and "has not been initialized" not in message:
                raise
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_source)

    def has_real_checkpoint(self) -> bool:
        if not self.model_dir:
            return False
        model_dir = Path(self.model_dir)
        return any(
            (
                (model_dir / "model.safetensors").exists(),
                any(model_dir.glob("model-*-of-*.safetensors")),
                (model_dir / "pytorch_model.bin").exists(),
                any(model_dir.glob("pytorch_model-*-of-*.bin")),
            )
        )

    def _initialize_model(self) -> Pi05ForActionPrediction:
        has_checkpoint = self.has_real_checkpoint()
        if has_checkpoint:
            with _skip_hf_weight_initialization():
                model = Pi05ForActionPrediction(self.config)
            self._load_checkpoint(model)
        else:
            model = Pi05ForActionPrediction(self.config)
            logger.info("Pi05Pipeline: no model.safetensors under %s; using random init.", self.model_dir)
        model.to(device=self._device, dtype=self._torch_dtype)
        model.eval()
        return model

    def _load_checkpoint(self, model: Pi05ForActionPrediction) -> None:
        import safetensors.torch

        model_dir = Path(self.model_dir)
        if (model_dir / "model.safetensors").exists():
            paths = [model_dir / "model.safetensors"]
        else:
            paths = sorted(model_dir.glob("model-*-of-*.safetensors"))

        if paths:
            logger.info("Pi05Pipeline: loading pi0.5 weights from %s", ", ".join(str(path) for path in paths))
            state = {}
            for path in paths:
                state.update(safetensors.torch.load_file(str(path)))
        else:
            if (model_dir / "pytorch_model.bin").exists():
                paths = [model_dir / "pytorch_model.bin"]
            else:
                paths = sorted(model_dir.glob("pytorch_model-*-of-*.bin"))
            logger.info("Pi05Pipeline: loading pi0.5 PyTorch weights from %s", ", ".join(str(path) for path in paths))
            state = {}
            for path in paths:
                state.update(torch.load(path, map_location="cpu"))
        model.load_weights(list(state.items()))

    def load_weights(self, weights=()):
        count = sum(1 for _ in weights)
        if count:
            logger.debug("Pi05Pipeline: ignoring %d external load_weights item(s); model is loaded in __init__.", count)
        return None

    def _make_realtime_prefix_cache_key(
        self,
        *,
        robot_obs: dict[str, Any],
        input_max_cameras: int | None,
        execution_backend: str,
        external_key: Any,
    ) -> tuple[Any, ...]:
        hasher = hashlib.blake2b(digest_size=24)
        _update_hash_with_value(hasher, robot_obs.get("prompt", "") or "")
        _update_hash_with_value(hasher, robot_obs.get("state"))
        images = robot_obs.get("images")
        if isinstance(images, dict):
            _update_hash_with_value(hasher, images)
        else:
            raw_images = {key: value for key, value in robot_obs.items() if key not in {"prompt", "state", "metadata"}}
            _update_hash_with_value(hasher, raw_images)
        _update_hash_with_value(hasher, external_key)
        _update_hash_with_value(
            hasher,
            {
                "image_feature_keys": tuple(self.config.image_feature_keys or ()),
                "image_key_map": dict(self.config.image_key_map or {}),
                "image_resolution": tuple(self.config.image_resolution),
                "input_max_cameras": input_max_cameras,
                "max_cameras": self.config.max_cameras,
                "max_state_dim": self.config.max_state_dim,
                "state_norm_stats": self.config.state_norm_stats,
                "state_num_bins": self.config.state_num_bins,
                "tokenizer_max_length": self.config.tokenizer_max_length,
            },
        )
        return (
            "pi05-realtime-prefix-kv-v1",
            hasher.hexdigest(),
            str(self.model.action_in_proj.weight.dtype),
            str(self._device),
            execution_backend or "manual",
        )

    def _make_realtime_image_embed_cache_keys(
        self,
        *,
        robot_obs: dict[str, Any],
        input_max_cameras: int | None,
    ) -> list[tuple[Any, ...]]:
        raw_images = robot_obs.get("images")
        if not isinstance(raw_images, dict):
            raw_images = {key: value for key, value in robot_obs.items() if key not in {"prompt", "state", "metadata"}}
        key_map = self.config.image_key_map or {}
        canonical_images = {key_map.get(key, key): value for key, value in raw_images.items()}
        feature_keys = list(self.config.image_feature_keys or canonical_images.keys())
        max_cameras = int(input_max_cameras or self.config.max_cameras or len(feature_keys) or 1)
        cache_keys: list[tuple[Any, ...]] = []
        for feature_key in feature_keys[:max_cameras]:
            hasher = hashlib.blake2b(digest_size=24)
            _update_hash_with_value(hasher, feature_key)
            _update_hash_with_value(hasher, canonical_images.get(feature_key))
            _update_hash_with_value(
                hasher,
                {
                    "image_resolution": tuple(self.config.image_resolution),
                    "model_dtype": str(self.model.action_in_proj.weight.dtype),
                    "device": str(self._device),
                },
            )
            cache_keys.append(("pi05-realtime-image-emb-v1", hasher.hexdigest()))
        return cache_keys

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        del kwargs
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")
        return_timing = bool(extra_args.get("return_timing", False))
        profile_nvtx = bool(extra_args.get("profile_nvtx", False))
        direct_suffix = bool(extra_args.get("direct_suffix", False))
        static_denoise = bool(extra_args.get("static_denoise", False))
        cuda_graph_denoise = bool(extra_args.get("cuda_graph_denoise", False))
        cuda_graph_image_embed = bool(extra_args.get("cuda_graph_image_embed", False))
        cuda_graph_prefix = bool(extra_args.get("cuda_graph_prefix", False))
        cuda_graph_prefix_denoise = bool(extra_args.get("cuda_graph_prefix_denoise", False))
        torch_compile_image_embed = bool(extra_args.get("torch_compile_image_embed", False))
        torch_compile_image_embed_fullgraph = bool(extra_args.get("torch_compile_image_embed_fullgraph", False))
        torch_compile_prefix = bool(extra_args.get("torch_compile_prefix", False))
        torch_compile_prefix_fullgraph = bool(extra_args.get("torch_compile_prefix_fullgraph", False))
        torch_compile_suffix = bool(extra_args.get("torch_compile_suffix", False))
        torch_compile_suffix_fullgraph = bool(extra_args.get("torch_compile_suffix_fullgraph", False))
        torch_compile_denoise_loop = bool(extra_args.get("torch_compile_denoise_loop", False))
        use_packed_prefix_qkv = bool(extra_args.get("use_packed_prefix_qkv", False))
        use_packed_prefix_mlp = bool(extra_args.get("use_packed_prefix_mlp", False))
        use_packed_qkv = bool(extra_args.get("use_packed_qkv", False))
        use_packed_mlp = bool(extra_args.get("use_packed_mlp", False))
        use_triton_qkv_rope = bool(extra_args.get("use_triton_qkv_rope", False))
        use_no_cat_suffix_attn = bool(extra_args.get("use_no_cat_suffix_attn", False))
        use_triton_final_head = bool(extra_args.get("use_triton_final_head", False))
        use_triton_final_head_euler = bool(extra_args.get("use_triton_final_head_euler", False))
        use_realtime_triton_decoder = bool(extra_args.get("use_realtime_triton_decoder", False))
        use_realtime_triton_prefix_encoder = bool(extra_args.get("use_realtime_triton_prefix_encoder", False))
        use_realtime_image_embed_cache = bool(extra_args.get("use_realtime_image_embed_cache", False))
        use_realtime_triton_prefix_emb_cache = bool(extra_args.get("use_realtime_triton_prefix_emb_cache", False))
        use_realtime_triton_prefix_kv_cache = bool(extra_args.get("use_realtime_triton_prefix_kv_cache", False))
        precompute_adarms = bool(extra_args.get("precompute_adarms", False))
        precompute_rope = bool(extra_args.get("precompute_rope", False))
        execution_backend = (
            str(extra_args.get("pi05_execution_backend", extra_args.get("execution_backend", ""))).strip().lower()
        )
        if execution_backend:
            if execution_backend in ("safe", "default"):
                pass
            elif execution_backend in ("cuda_graph", "best_cuda_graph"):
                static_denoise = True
                cuda_graph_image_embed = True
                cuda_graph_prefix_denoise = True
                torch_compile_image_embed = True
                torch_compile_image_embed_fullgraph = True
                torch_compile_prefix = True
                torch_compile_suffix = True
                precompute_adarms = True
            elif execution_backend in (
                "realtime",
                "realtime_triton",
                "realtime_triton_prefix",
                "realtime_triton_prefix_image_cache",
                "realtime_triton_prefix_emb_cache",
                "realtime_triton_prefix_cache",
            ):
                static_denoise = True
                cuda_graph_image_embed = True
                cuda_graph_prefix_denoise = True
                torch_compile_image_embed = True
                torch_compile_image_embed_fullgraph = True
                torch_compile_prefix = True
                use_realtime_triton_decoder = True
                use_realtime_triton_prefix_encoder = execution_backend in (
                    "realtime_triton_prefix",
                    "realtime_triton_prefix_image_cache",
                    "realtime_triton_prefix_emb_cache",
                    "realtime_triton_prefix_cache",
                )
                use_realtime_image_embed_cache = execution_backend == "realtime_triton_prefix_image_cache"
                use_realtime_triton_prefix_emb_cache = execution_backend == "realtime_triton_prefix_emb_cache"
                use_realtime_triton_prefix_kv_cache = execution_backend == "realtime_triton_prefix_cache"
                precompute_adarms = True
            else:
                return DiffusionOutput(error=f"Unsupported pi05_execution_backend: {execution_backend!r}.")
        timing: dict[str, float] = {}
        total_t0 = time.perf_counter() if return_timing else 0.0
        if return_timing and execution_backend:
            timing["pi05_execution_backend"] = execution_backend

        if robot_obs is None:
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            num_steps = getattr(req.sampling_params, "num_inference_steps", None)
            if req.is_dummy_run() and prompt == "dummy run" and num_steps == 1:
                logger.info("Pi05Pipeline: dummy warmup request without robot_obs; returning zeros.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.max_action_dim),
                            dtype=np.float32,
                        )
                    },
                )
            return DiffusionOutput(error="Pi05Pipeline.forward requires sampling_params.extra_args['robot_obs'].")

        input_max_cameras = extra_args.get("pi05_max_cameras")
        if input_max_cameras is None and (
            execution_backend
            in (
                "realtime",
                "realtime_triton",
                "realtime_triton_prefix",
                "realtime_triton_prefix_image_cache",
                "realtime_triton_prefix_emb_cache",
                "realtime_triton_prefix_cache",
            )
            or use_realtime_triton_decoder
        ):
            input_max_cameras = extra_args.get(
                "pi05_realtime_max_cameras",
                len(self.config.image_feature_keys or []) or 1,
            )
        input_max_cameras_int = None if input_max_cameras is None else int(input_max_cameras)
        realtime_prefix_cache_key = None
        realtime_image_embed_cache_keys = None
        if use_realtime_image_embed_cache:
            if bool(extra_args.get("reset", False)):
                self.model.clear_realtime_image_embed_cache()
            realtime_image_embed_cache_keys = self._make_realtime_image_embed_cache_keys(
                robot_obs=robot_obs,
                input_max_cameras=input_max_cameras_int,
            )
        if use_realtime_triton_prefix_emb_cache or use_realtime_triton_prefix_kv_cache:
            if not use_realtime_triton_prefix_encoder:
                return DiffusionOutput(error="realtime prefix cache requires realtime_triton_prefix backend.")
            if bool(extra_args.get("reset", False)):
                self.model.clear_realtime_prefix_caches()
            realtime_prefix_cache_key = self._make_realtime_prefix_cache_key(
                robot_obs=robot_obs,
                input_max_cameras=input_max_cameras_int,
                execution_backend=execution_backend,
                external_key=extra_args.get("pi05_prefix_cache_key"),
            )

        noise = extra_args.get("noise")
        if noise is not None and not isinstance(noise, torch.Tensor):
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self._device)
        elif isinstance(noise, torch.Tensor):
            noise = noise.to(device=self._device, dtype=torch.float32)

        num_steps = extra_args.get("num_inference_steps")
        if (
            use_realtime_triton_prefix_kv_cache
            and realtime_prefix_cache_key is not None
            and self.model.has_realtime_prefix_kv_cache(realtime_prefix_cache_key)
        ):
            sample_timing: dict[str, object] | None = {} if return_timing else None
            if return_timing:
                timing["realtime_prefix_kv_cache_hit_before_preprocess"] = True
                _sync_if_cuda(self._device)
                sample_t0 = time.perf_counter()
            actions = self.model.sample_actions_from_realtime_prefix_kv_cache(
                realtime_prefix_cache_key=realtime_prefix_cache_key,
                noise=noise,
                num_steps=num_steps,
                timing=sample_timing,
                profile_nvtx=profile_nvtx,
                precompute_rope=precompute_rope,
            )
            if return_timing:
                _sync_if_cuda(self._device)
                timing["sample_actions_ms"] = (time.perf_counter() - sample_t0) * 1000.0
                if sample_timing is not None:
                    timing.update(sample_timing)
            postprocess_t0 = time.perf_counter() if return_timing else 0.0
            actions = self.model._unnormalize_actions(actions)
            actions_np = actions.squeeze(0).float().cpu().numpy()
            output: dict[str, object] = {"actions": actions_np}
            if return_timing:
                timing["preprocess_ms"] = 0.0
                timing["postprocess_ms"] = (time.perf_counter() - postprocess_t0) * 1000.0
                timing["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
                return DiffusionOutput(output=output, custom_output={"policy_timing": timing})
            return DiffusionOutput(output=output)
        if (
            use_realtime_triton_prefix_emb_cache
            and realtime_prefix_cache_key is not None
            and self.model.has_realtime_prefix_emb_cache(realtime_prefix_cache_key)
        ):
            sample_timing: dict[str, object] | None = {} if return_timing else None
            if return_timing:
                timing["realtime_prefix_emb_cache_hit_before_preprocess"] = True
                _sync_if_cuda(self._device)
                sample_t0 = time.perf_counter()
            actions = self.model.sample_actions_from_realtime_prefix_emb_cache(
                realtime_prefix_cache_key=realtime_prefix_cache_key,
                noise=noise,
                num_steps=num_steps,
                timing=sample_timing,
                profile_nvtx=profile_nvtx,
                precompute_rope=precompute_rope,
            )
            if return_timing:
                _sync_if_cuda(self._device)
                timing["sample_actions_ms"] = (time.perf_counter() - sample_t0) * 1000.0
                if sample_timing is not None:
                    timing.update(sample_timing)
            postprocess_t0 = time.perf_counter() if return_timing else 0.0
            actions = self.model._unnormalize_actions(actions)
            actions_np = actions.squeeze(0).float().cpu().numpy()
            output = {"actions": actions_np}
            if return_timing:
                timing["preprocess_ms"] = 0.0
                timing["postprocess_ms"] = (time.perf_counter() - postprocess_t0) * 1000.0
                timing["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
                return DiffusionOutput(output=output, custom_output={"policy_timing": timing})
            return DiffusionOutput(output=output)

        preprocess_t0 = time.perf_counter() if return_timing else 0.0
        model_inputs = build_model_inputs(
            robot_obs,
            self.config,
            self.tokenizer,
            self._device,
            max_cameras=input_max_cameras_int,
            return_metadata=bool(use_realtime_triton_prefix_encoder),
        )
        input_metadata = None
        if len(model_inputs) == 5:
            images, image_masks, lang_tokens, lang_masks, input_metadata = model_inputs
        else:
            images, image_masks, lang_tokens, lang_masks = model_inputs
        if return_timing and input_max_cameras is not None:
            timing["pi05_input_max_cameras"] = int(input_max_cameras)
        if return_timing:
            timing["pi05_input_image_count"] = len(images)
            timing["pi05_input_image_masks"] = [bool(mask.item()) for mask in image_masks]
        if return_timing:
            timing["preprocess_ms"] = (time.perf_counter() - preprocess_t0) * 1000.0

        sample_timing: dict[str, object] | None = {} if return_timing else None
        if return_timing:
            _sync_if_cuda(self._device)
            sample_t0 = time.perf_counter()
        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            tokens=lang_tokens,
            masks=lang_masks,
            noise=noise,
            num_steps=num_steps,
            timing=sample_timing,
            direct_suffix=direct_suffix,
            static_denoise=static_denoise,
            cuda_graph_denoise=cuda_graph_denoise,
            cuda_graph_image_embed=cuda_graph_image_embed,
            cuda_graph_prefix=cuda_graph_prefix,
            cuda_graph_prefix_denoise=cuda_graph_prefix_denoise,
            torch_compile_image_embed=torch_compile_image_embed,
            torch_compile_image_embed_fullgraph=torch_compile_image_embed_fullgraph,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            torch_compile_suffix=torch_compile_suffix,
            torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
            torch_compile_denoise_loop=torch_compile_denoise_loop,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            use_packed_qkv=use_packed_qkv,
            use_packed_mlp=use_packed_mlp,
            use_triton_qkv_rope=use_triton_qkv_rope,
            use_no_cat_suffix_attn=use_no_cat_suffix_attn,
            use_triton_final_head=use_triton_final_head,
            use_triton_final_head_euler=use_triton_final_head_euler,
            use_realtime_triton_decoder=use_realtime_triton_decoder,
            use_realtime_triton_prefix_encoder=use_realtime_triton_prefix_encoder,
            use_realtime_triton_prefix_emb_cache=use_realtime_triton_prefix_emb_cache,
            use_realtime_triton_prefix_kv_cache=use_realtime_triton_prefix_kv_cache,
            use_realtime_image_embed_cache=use_realtime_image_embed_cache,
            realtime_prefix_cache_key=realtime_prefix_cache_key,
            realtime_image_embed_cache_keys=realtime_image_embed_cache_keys,
            prefix_valid_len=(None if input_metadata is None else int(input_metadata["prefix_valid_len"])),
            prefix_masks_contiguous=(
                False if input_metadata is None else bool(input_metadata["prefix_masks_contiguous"])
            ),
            profile_nvtx=profile_nvtx,
            precompute_adarms=precompute_adarms,
            precompute_rope=precompute_rope,
        )
        if return_timing:
            _sync_if_cuda(self._device)
            timing["sample_actions_ms"] = (time.perf_counter() - sample_t0) * 1000.0
            if sample_timing is not None:
                timing.update(sample_timing)
        postprocess_t0 = time.perf_counter() if return_timing else 0.0
        actions = self.model._unnormalize_actions(actions)
        actions_np = actions.squeeze(0).float().cpu().numpy()
        output: dict[str, object] = {"actions": actions_np}
        if return_timing:
            timing["postprocess_ms"] = (time.perf_counter() - postprocess_t0) * 1000.0
            timing["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
            return DiffusionOutput(output=output, custom_output={"policy_timing": timing})
        return DiffusionOutput(output=output)
