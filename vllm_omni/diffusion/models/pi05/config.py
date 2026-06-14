# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config surface for the pi0.5 VLA model in vllm-omni.

pi0.5 is intentionally configured like pi0: a small dataclass consumes the raw
LeRobot checkpoint ``config.json`` and keeps only runtime-relevant fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"


@dataclass
class Pi05Config:
    """pi0.5 VLA config.

    The important pi0.5-specific difference from pi0 is that state is
    discretized into language tokens by the processor; the model does not
    receive a separate state tensor.
    """

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    chunk_size: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32
    num_inference_steps: int = 10

    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer: str | None = None
    max_cameras: int = 3
    dtype: str = "float32"

    state_num_bins: int = 256
    state_norm_stats: dict | None = None
    norm_stats: dict | None = None

    min_period: float = 4e-3
    max_period: float = 4.0
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    image_feature_keys: list[str] | None = None
    image_key_map: dict[str, str] = field(default_factory=dict)
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        res = self.image_resolution
        if not isinstance(res, (tuple, list)) or len(res) != 2 or res[0] != res[1]:
            raise ValueError(f"pi0.5 expects a square image_resolution (H == W); got {res!r}.")
        self.image_resolution = (int(res[0]), int(res[1]))

        if self.image_feature_keys is None and self.input_features:
            self.image_feature_keys = [key for key in self.input_features if key.startswith(OBS_IMAGES + ".")]

        if self.state_norm_stats is None and isinstance(self.norm_stats, dict):
            self.state_norm_stats = self.norm_stats.get("state")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Pi05Config:
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            return cls()
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_model_config(raw)

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Pi05Config:
        if not model_config:
            return cls()

        raw = dict(model_config)
        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])

        allowed = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        return cls(**filtered)
