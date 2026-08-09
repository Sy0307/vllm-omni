"""Fixed Seed-TTS performance sweep for IndexTTS 2.5.

The defaults are the release-validation workload used for this integration:
500 Seed-TTS Eval EN requests at concurrency 4/8/16/32.  The script only
drives an already-running server, so server recipe changes remain an explicit
A/B variable.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--locale", default="en")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
    )
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument(
        "--request-seed",
        type=int,
        default=None,
        help=("Model sampling seed for reproducible requests; omitted by default so no request seed is sent."),
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=0,
        help=("Seed used only for Seed-TTS sample selection; does not seed model sampling."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/indextts25-seed-perf"),
    )
    parser.add_argument("--label", default="indextts25")
    args = parser.parse_args(argv)
    if args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if args.num_warmups < 0:
        parser.error("--num-warmups cannot be negative")
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    return args


def build_command(
    args: argparse.Namespace,
    *,
    concurrency: int,
) -> list[str]:
    tokenizer = args.tokenizer or str(Path(args.model) / "qwen0.6bemo4-merge")
    extra_body_payload = {
        "extra_params": {
            "lang": args.locale,
            "text_normalization": True,
        },
    }
    if args.request_seed is not None:
        extra_body_payload = {"seed": args.request_seed, **extra_body_payload}
    extra_body = json.dumps(
        extra_body_payload,
        separators=(",", ":"),
    )
    result_filename = f"{args.label}_c{concurrency}_n{args.num_prompts}.json"
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--tokenizer",
        tokenizer,
        "--trust-remote-code",
        "--backend",
        "openai-audio-speech",
        "--endpoint",
        "/v1/audio/speech",
        "--dataset-name",
        "seed-tts",
        "--dataset-path",
        args.dataset_path,
        "--seed-tts-locale",
        args.locale,
        "--seed",
        str(args.dataset_seed),
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--extra-body",
        extra_body,
        "--percentile-metrics",
        "ttft,e2el,audio_rtf,audio_ttfp,audio_duration",
        "--metric-percentiles",
        "50,95,99",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(args.output_dir),
        "--result-filename",
        result_filename,
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for concurrency in args.concurrency:
        command = build_command(args, concurrency=concurrency)
        print(f"[IndexTTS 2.5 perf] {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
