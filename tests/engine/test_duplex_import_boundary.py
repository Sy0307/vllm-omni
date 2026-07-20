# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stable_engine_imports_do_not_load_experimental_duplex() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import sys

import vllm_omni.engine.async_omni_engine
import vllm_omni.engine.orchestrator
import vllm_omni.entrypoints.async_omni

loaded = sorted(
    name
    for name in sys.modules
    if name == "vllm_omni.experimental.fullduplex"
    or name.startswith("vllm_omni.experimental.fullduplex.")
)
if loaded:
    raise SystemExit("stable imports loaded experimental duplex: " + ", ".join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
