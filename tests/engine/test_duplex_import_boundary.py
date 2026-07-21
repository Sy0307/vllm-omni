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


def test_stable_engine_does_not_expose_duplex_contract_modules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib.util

from vllm_omni.engine import messages

legacy_modules = (
    "vllm_omni.engine.duplex_contracts",
    "vllm_omni.engine.duplex_lease",
    "vllm_omni.engine.resumable",
)
present = [name for name in legacy_modules if importlib.util.find_spec(name) is not None]
if present:
    raise SystemExit("stable duplex modules still exist: " + ", ".join(present))

duplex_exports = sorted(name for name in vars(messages) if name.startswith("Duplex"))
if duplex_exports:
    raise SystemExit("stable messages still expose duplex contracts: " + ", ".join(duplex_exports))
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


def test_runtime_package_does_not_bundle_the_browser_demo() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "vllm_omni" / "experimental" / "fullduplex" / "web").exists()


def test_experimental_engine_uses_canonical_contract_module_names() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    engine_dir = repo_root / "vllm_omni" / "experimental" / "fullduplex" / "engine"

    assert (engine_dir / "contracts.py").is_file()
    assert (engine_dir / "lease.py").is_file()
    assert (engine_dir / "messages.py").is_file()
    assert not (engine_dir / "duplex_lease.py").exists()
