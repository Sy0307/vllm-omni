#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal OpenPI client for the pi0.5 VLA policy server."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    example_dir = str(Path(__file__).resolve().parent)
    removed = sys.path and sys.path[0] == example_dir
    if removed:
        sys.path.pop(0)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    finally:
        if removed:
            sys.path.insert(0, example_dir)
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("pi0.5 OpenPI example requires `openpi-client`.") from exc

logger = logging.getLogger("pi05_openpi_client")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_PROMPT = "pick up the red block and place it in the bin"

CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
STATE_DIM = 32


def _make_dummy_obs(prompt: str, session_id: str, image_size: int = 224) -> dict[str, Any]:
    obs: dict[str, Any] = {cam: np.zeros((image_size, image_size, 3), dtype=np.uint8) for cam in CAMERA_KEYS}
    obs["state"] = np.zeros(STATE_DIM, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _ws_uri(host: str, port: int, path: str) -> str:
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{port}{path}"


def _as_action_array(response: Any) -> np.ndarray:
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(f"Inference failed: {response.get('message', response)}")
    return np.asarray(response, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="pi0.5 OpenPI client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=2, help="Inference observations to send.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    policy = WebsocketClientPolicy(host=_ws_uri(args.host, args.port, args.path))
    metadata = dict(policy.get_server_metadata())
    logger.info("Server metadata: %s", metadata)

    session_id = str(uuid.uuid4())
    for step in range(args.num_steps):
        obs = _make_dummy_obs(args.prompt, session_id)
        actions = _as_action_array(policy.infer(obs))
        logger.info(
            "[step %d] actions shape=%s mean=%.4f std=%.4f",
            step,
            actions.shape,
            float(actions.mean()),
            float(actions.std()),
        )
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned non-finite actions.")
    logger.info("Done: pi0.5 OpenPI round-trip OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
