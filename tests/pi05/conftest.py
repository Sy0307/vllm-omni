# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pi05")
    group.addoption(
        "--pi05-e2e-ckpt",
        action="store",
        default=None,
        help="Local pi0.5 checkpoint directory for full-model tests.",
    )
    group.addoption(
        "--pi05-e2e-tokenizer",
        action="store",
        default=None,
        help="Local PaliGemma tokenizer directory used by pi0.5 tests.",
    )
    group.addoption(
        "--pi05-e2e-device",
        action="store",
        default=None,
        help="Device for pi0.5 full-model tests. Defaults to cuda when available.",
    )
    group.addoption(
        "--pi05-e2e-dtype",
        action="store",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Torch dtype for pi0.5 full-model tests.",
    )
    group.addoption(
        "--pi05-e2e-num-steps",
        action="store",
        type=int,
        default=1,
        help="Diffusion steps for pi0.5 full-model tests.",
    )
    group.addoption(
        "--pi05-parity-atol",
        action="store",
        type=float,
        default=1e-5,
        help="Absolute tolerance for LeRobot pi0.5 parity.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.function.__name__ != "test_pi05_openpi_online":
        return
    if "omni_server" not in metafunc.fixturenames:
        return

    model_path = metafunc.config.getoption("pi05_e2e_ckpt")
    tokenizer_path = metafunc.config.getoption("pi05_e2e_tokenizer")
    if not model_path or not tokenizer_path:
        param = pytest.param(
            None,
            marks=pytest.mark.skip(reason="Set --pi05-e2e-ckpt and --pi05-e2e-tokenizer to run online pi0.5 serving."),
        )
        metafunc.parametrize("omni_server", [param], indirect=True)
        return

    try:
        from tests.helpers.runtime import OmniServerParams
    except Exception as exc:  # noqa: BLE001
        param = pytest.param(None, marks=pytest.mark.skip(reason=f"serving stack unavailable: {exc!r}"))
        metafunc.parametrize("omni_server", [param], indirect=True)
        return

    deploy_config = _write_pi05_deploy_overlay(metafunc.config.rootpath, str(tokenizer_path))
    params = OmniServerParams(
        model=str(model_path),
        port=8095,
        server_args=[
            "--deploy-config",
            str(deploy_config),
            "--served-model-name",
            "pi05",
            "--enforce-eager",
            "--disable-log-stats",
        ],
    )
    metafunc.parametrize("omni_server", [params], indirect=True)


def _write_pi05_deploy_overlay(root: Path, tokenizer_path: str) -> Path:
    base_path = root / "vllm_omni" / "deploy" / "pi05.yaml"
    with open(base_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    stage = data["stages"][0]
    model_config = dict(stage.get("model_config") or {})
    model_config["tokenizer"] = tokenizer_path
    stage["model_config"] = model_config

    digest = hashlib.sha1(tokenizer_path.encode("utf-8")).hexdigest()[:12]
    out_dir = root / ".pytest_cache" / "pi05"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pi05_{digest}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path
