"""Tests for the fixed IndexTTS 2.5 Seed-TTS performance sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).parents[2] / "benchmarks" / "tts"),
)
import benchmark_indextts2_5 as benchmark  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_defaults_match_fixed_seed_tts_acceptance_workload():
    args = benchmark.parse_args(
        [
            "--model",
            "/models/indextts25",
            "--dataset-path",
            "/datasets/seed-tts",
        ]
    )

    assert args.concurrency == [4, 8, 16, 32]
    assert args.num_prompts == 500
    assert args.num_warmups == 5
    assert args.request_seed is None


def test_format_help_distinguishes_request_and_dataset_seeds(capsys):
    with pytest.raises(SystemExit, match="0"):
        benchmark.parse_args(["--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert (
        "Model sampling seed for reproducible requests; omitted by default so no request seed is sent."
    ) in help_text
    assert "Seed used only for Seed-TTS sample selection; does not seed model sampling." in help_text


def test_build_command_without_request_seed_omits_seed(tmp_path):
    args = benchmark.parse_args(
        [
            "--model",
            "/models/indextts25",
            "--dataset-path",
            "/datasets/seed-tts",
            "--output-dir",
            str(tmp_path),
            "--label",
            "optimized",
        ]
    )

    cmd = benchmark.build_command(args, concurrency=8)

    assert cmd[:5] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
    ]
    assert cmd[cmd.index("--tokenizer") + 1] == ("/models/indextts25/qwen0.6bemo4-merge")
    assert cmd[cmd.index("--dataset-name") + 1] == "seed-tts"
    assert cmd[cmd.index("--num-prompts") + 1] == "500"
    assert cmd[cmd.index("--max-concurrency") + 1] == "8"
    assert cmd[cmd.index("--request-rate") + 1] == "inf"
    assert cmd[cmd.index("--result-filename") + 1] == ("optimized_c8_n500.json")
    assert cmd[cmd.index("--extra-body") + 1] == ('{"extra_params":{"lang":"en","text_normalization":true}}')


def test_build_command_with_request_seed_preserves_reproducible_quality_mode(tmp_path):
    args = benchmark.parse_args(
        [
            "--model",
            "/models/indextts25",
            "--dataset-path",
            "/datasets/seed-tts",
            "--request-seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ]
    )

    cmd = benchmark.build_command(args, concurrency=8)

    assert cmd[cmd.index("--extra-body") + 1] == ('{"seed":42,"extra_params":{"lang":"en","text_normalization":true}}')
