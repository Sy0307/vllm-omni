# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Use a pinned external Seed-TTS client with vLLM's streaming HTTP protocol.

Set SEEDTTS_CLIENT_ROOT to the reference repository checkout. All benchmark
arguments are forwarded unchanged. BENCH_WALL_AUDIT_PATH optionally records
unrounded measured wall time after the client's timer stops.
"""

import json
import os
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    client_root = Path(os.environ["SEEDTTS_CLIENT_ROOT"]).resolve()
    if not (client_root / "benchmarks/eval/benchmark_tts_seedtts.py").is_file():
        raise ValueError("SEEDTTS_CLIENT_ROOT must contain the Seed-TTS reference client")
    # This script is an explicit bridge to a separately pinned benchmark package.
    sys.path.insert(0, str(client_root))
    from benchmarks.benchmarker.runner import BenchmarkRunner
    from benchmarks.tasks import tts

    original_payload = tts._build_tts_payload

    def payload(*args, **kwargs):
        value = original_payload(*args, **kwargs)
        if value.get("stream"):
            value["stream_format"] = "audio"
        ref = value.get("ref_audio")
        if isinstance(ref, str) and ref.startswith("/"):
            value["ref_audio"] = Path(ref).as_uri()
        if os.environ.get("BENCH_LANGUAGE"):
            value["language"] = os.environ["BENCH_LANGUAGE"]
        return value

    tts._build_tts_payload = payload
    audit_path = os.environ.get("BENCH_WALL_AUDIT_PATH")
    if audit_path:
        original_run = BenchmarkRunner.run

        async def audited_run(self, samples, send_fn):
            outputs = await original_run(self, samples, send_fn)
            if self.wall_clock_s <= 0:
                raise ValueError("The reference client returned a nonpositive measured wall time")
            with Path(audit_path).open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "completed_at_unix": time.time(),
                            "concurrency": self.config.max_concurrency,
                            "samples": len(samples),
                            "wall_clock_s": self.wall_clock_s,
                        }
                    )
                    + "\n"
                )
            return outputs

        BenchmarkRunner.run = audited_run
    runpy.run_module("benchmarks.eval.benchmark_tts_seedtts", run_name="__main__")


if __name__ == "__main__":
    main()
