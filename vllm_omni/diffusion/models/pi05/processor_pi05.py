# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preprocessing for the pi0.5 VLA model.

pi0.5 differs from pi0 in how robot state is consumed: state is normalized,
discretized, and serialized into the language prompt. The action expert then
denoises action tokens only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

PI05_IMAGE_SIZE = 224
PI05_NUM_IMAGE_TOKENS = 256
PI05_IMAGE_TOKEN_INDEX = 257152
PI05_MAX_CAMERAS = 3
PI05_MAX_TOKEN_LEN = 200
PI05_NUM_BINS = 256


def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi05ImageProcessor:
    def __init__(self, image_size: int = PI05_IMAGE_SIZE):
        self.image_size = image_size

    def preprocess_single(self, image: Any, *, device: torch.device | None = None) -> torch.Tensor:
        t = self._to_tensor(image, device=device)
        if t.ndim != 4:
            raise ValueError(f"Expected image tensor with shape (B,C,H,W), got {tuple(t.shape)}")
        if t.shape[1] not in (1, 3):
            raise ValueError(f"Expected image tensor channel dimension to be 1 or 3, got {t.shape[1]}")
        if t.shape[2] <= 0 or t.shape[3] <= 0:
            raise ValueError(f"Expected non-empty image height/width, got {tuple(t.shape)}")
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def _to_tensor(self, image: Any, *, device: torch.device | None = None) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return pil_image_to_tensor(image)
        if isinstance(image, np.ndarray):
            arr = image
            if arr.size == 0:
                raise ValueError(f"Expected non-empty image array, got shape {arr.shape}")
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                fast_uint8 = os.environ.get("PI05_FAST_UINT8_IMAGE_PREPROCESS", "0").lower() not in {
                    "0",
                    "false",
                    "no",
                }
                if (
                    fast_uint8
                    and device is not None
                    and device.type == "cuda"
                    and arr.dtype == np.uint8
                    and arr.shape[0] == self.image_size
                    and arr.shape[1] == self.image_size
                ):
                    return (
                        torch.as_tensor(arr, device=device)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(dtype=torch.float32)
                        .div_(255.0)
                        .mul_(2.0)
                        .sub_(1.0)
                    )
                if np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0:
                    arr = arr.astype(np.float32)
                    arr /= 255.0
                    arr *= 2.0
                    arr -= 1.0
                else:
                    arr = arr.astype(np.float32)
                return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return torch.as_tensor(arr, dtype=torch.float32)
        if isinstance(image, torch.Tensor):
            t = image
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t.to(dtype=torch.float32)
        raise TypeError(f"Unsupported image type for pi0.5 preprocessing: {type(image)}")

    def make_empty_image(self) -> torch.Tensor:
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


def pad_or_truncate_state(raw_state: Any, max_state_dim: int) -> np.ndarray:
    if raw_state is None:
        return np.zeros((max_state_dim,), dtype=np.float32)
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < max_state_dim:
        state = np.pad(state, (0, max_state_dim - state.shape[0]), mode="constant")
    elif state.shape[0] > max_state_dim:
        state = state[:max_state_dim]
    return state.astype(np.float32)


def _coerce_stat_vector(value: Any, max_state_dim: int, fill_value: float) -> np.ndarray:
    if value is None:
        return np.full((max_state_dim,), fill_value, dtype=np.float32)
    return pad_or_truncate_state(value, max_state_dim)


def normalize_state(
    raw_state: Any,
    *,
    max_state_dim: int,
    state_norm_stats: dict[str, Any] | None,
) -> np.ndarray:
    state = pad_or_truncate_state(raw_state, max_state_dim)
    if not state_norm_stats:
        return np.clip(state, -1.0, 1.0)

    stats = state_norm_stats
    mode = stats.get("mode")
    if mode is None:
        if "mean" in stats and "std" in stats:
            mode = "mean_std"
        elif "min" in stats and "max" in stats:
            mode = "min_max"
        elif "q01" in stats and "q99" in stats:
            mode = "quantile"
        elif "low" in stats and "high" in stats:
            mode = "quantile"

    if mode == "mean_std":
        mean = _coerce_stat_vector(stats.get("mean"), max_state_dim, 0.0)
        std = _coerce_stat_vector(stats.get("std"), max_state_dim, 1.0)
        std = np.where(np.abs(std) < 1e-6, 1.0, std)
        return np.clip((state - mean) / std, -1.0, 1.0)

    if mode == "min_max":
        vmin = _coerce_stat_vector(stats.get("min"), max_state_dim, -1.0)
        vmax = _coerce_stat_vector(stats.get("max"), max_state_dim, 1.0)
        denom = np.where(np.abs(vmax - vmin) < 1e-6, 1.0, vmax - vmin)
        return np.clip(2.0 * (state - vmin) / denom - 1.0, -1.0, 1.0)

    if mode == "quantile":
        low_key = "q01" if "q01" in stats else "low"
        high_key = "q99" if "q99" in stats else "high"
        low = _coerce_stat_vector(stats.get(low_key), max_state_dim, -1.0)
        high = _coerce_stat_vector(stats.get(high_key), max_state_dim, 1.0)
        denom = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
        return np.clip(2.0 * (state - low) / denom - 1.0, -1.0, 1.0)

    raise ValueError(f"Unsupported pi0.5 state_norm_stats mode: {mode!r}.")


def discretize_state(state: np.ndarray, *, num_bins: int = PI05_NUM_BINS) -> np.ndarray:
    clipped = np.clip(np.asarray(state, dtype=np.float32), -1.0, 1.0)
    bins = np.linspace(-1.0, 1.0, num_bins + 1, dtype=np.float32)[:-1]
    return np.clip(np.digitize(clipped, bins=bins) - 1, 0, num_bins - 1).astype(np.int64)


def build_pi05_prompt(
    *,
    task: str,
    state: Any,
    max_state_dim: int,
    state_norm_stats: dict[str, Any] | None,
    state_num_bins: int = PI05_NUM_BINS,
) -> str:
    cleaned_task = (task or "").strip().replace("_", " ").replace("\n", " ")
    normed_state = normalize_state(
        state,
        max_state_dim=max_state_dim,
        state_norm_stats=state_norm_stats,
    )
    state_bins = discretize_state(normed_state, num_bins=state_num_bins)
    state_str = " ".join(str(int(x)) for x in state_bins.tolist())
    return f"Task: {cleaned_task}, State: {state_str};\nAction: "


def tokenize_prompt(tokenizer, text: str, max_token_len: int = PI05_MAX_TOKEN_LEN):
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=max_token_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    return list(enc["input_ids"]), list(enc["attention_mask"])


def _compact_padding_to_right(ids: list[int], attn: list[int]) -> tuple[list[int], list[int]]:
    """Move live language tokens before padding for realtime prefix compression.

    The realtime prefix path later removes all language tokens with attention
    mask 0. Compacting them here preserves the live-token order while allowing
    the fast contiguous-prefix path to slice instead of boolean-index.
    """
    if len(ids) != len(attn):
        raise ValueError("input_ids and attention_mask must have the same length")
    live_ids = [token_id for token_id, keep in zip(ids, attn, strict=True) if int(keep)]
    pad_ids = [token_id for token_id, keep in zip(ids, attn, strict=True) if not int(keep)]
    if len(live_ids) == len(ids):
        return ids, attn
    return live_ids + pad_ids, [1] * len(live_ids) + [0] * (len(ids) - len(live_ids))


def _extract_images(robot_obs: dict, config) -> dict[str, Any]:
    images = robot_obs.get("images")
    if not isinstance(images, dict):
        images = {k: v for k, v in robot_obs.items() if _is_image_like(v)}
    key_map = getattr(config, "image_key_map", None) or {}
    return {key_map.get(k, k): v for k, v in images.items() if _is_image_like(v)}


def _is_image_like(value: Any) -> bool:
    if isinstance(value, Image.Image | torch.Tensor):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim >= 3
    if isinstance(value, list | tuple):
        try:
            return np.asarray(value).ndim >= 3
        except Exception:  # noqa: BLE001
            return False
    return False


def build_model_inputs(
    robot_obs: dict,
    config,
    tokenizer,
    device: torch.device,
    *,
    max_cameras: int | None = None,
    return_metadata: bool = False,
):
    image_size = int(config.image_resolution[0])
    img_proc = Pi05ImageProcessor(image_size=image_size)
    max_cameras = int(config.max_cameras if max_cameras is None else max_cameras)
    max_cameras = max(1, max_cameras)

    feature_keys = config.image_feature_keys or []
    if not feature_keys:
        provided = _extract_images(robot_obs, config)
        feature_keys = list(provided.keys())[:max_cameras]

    obs_images = _extract_images(robot_obs, config)

    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    valid_image_tokens = 0
    seen_image_padding = False
    image_prefix_contiguous = True
    for key in feature_keys[:max_cameras]:
        img = obs_images.get(key)
        if img is not None:
            tensor = img_proc.preprocess_single(img, device=device)
            if tensor.device != device:
                tensor = tensor.to(device=device)
            mask = True
            valid_image_tokens += PI05_NUM_IMAGE_TOKENS
            if seen_image_padding:
                image_prefix_contiguous = False
        else:
            tensor = img_proc.make_empty_image().to(device=device)
            mask = False
            seen_image_padding = True
        images.append(tensor)
        image_masks.append(torch.tensor([mask], dtype=torch.bool, device=device))

    while len(images) < max_cameras:
        images.append(img_proc.make_empty_image().to(device=device))
        image_masks.append(torch.tensor([False], dtype=torch.bool, device=device))
        seen_image_padding = True

    prompt = build_pi05_prompt(
        task=robot_obs.get("prompt", "") or "",
        state=robot_obs.get("state"),
        max_state_dim=config.max_state_dim,
        state_norm_stats=config.state_norm_stats,
        state_num_bins=config.state_num_bins,
    )
    ids, attn = tokenize_prompt(tokenizer, prompt, config.tokenizer_max_length)
    if return_metadata:
        ids, attn = _compact_padding_to_right(ids, attn)
    lang_tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lang_masks = torch.tensor([attn], dtype=torch.bool, device=device)

    if return_metadata:
        lang_valid_len = int(sum(int(x) for x in attn))
        lang_prefix_contiguous = all(int(x) == 1 for x in attn[:lang_valid_len]) and not any(
            int(x) for x in attn[lang_valid_len:]
        )
        # Prefix masks are contiguous only if no missing image block appears
        # before live language tokens. The realtime path can then trim padding
        # without a GPU-side sum/any synchronization.
        prefix_masks_contiguous = (
            image_prefix_contiguous and lang_prefix_contiguous and not (seen_image_padding and lang_valid_len > 0)
        )
        metadata = {
            "prefix_valid_len": valid_image_tokens + lang_valid_len,
            "prefix_masks_contiguous": prefix_masks_contiguous,
        }
        return images, image_masks, lang_tokens, lang_masks, metadata

    return images, image_masks, lang_tokens, lang_masks
