"""VoxCPM2 performance benchmark with detailed profiling.

Measures RTF, per-step latency breakdown, and GPU utilization.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIGS_PATH = str(REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm2.yaml")
SAMPLE_RATE = 48_000

# Test sentences of varying length
TEST_TEXTS = {
    "short": "Hello, this is a test.",
    "medium": "The quick brown fox jumps over the lazy dog. This is a VoxCPM2 native AR synthesis example running on vLLM Omni with CUDA Graph optimization.",
    "long": (
        "In the heart of the ancient forest, where sunlight barely penetrated the thick canopy of leaves, "
        "there lived a community of creatures who had developed their own language over millennia. "
        "They communicated through a combination of sounds, gestures, and bioluminescent patterns that "
        "danced across their translucent skin. The elders among them could tell stories that lasted for days, "
        "weaving together the history of their species with the rhythms of the forest itself."
    ),
}


def extract_audio(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs") or multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"No audio key: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("Audio list is empty")
        return valid[-1]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def benchmark_single(engine, text: str, warmup: bool = False) -> dict:
    prompt = {"prompt": text}

    # GPU warmup
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    outputs = engine.generate([prompt])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    mm = outputs[0].outputs[0].multimodal_output
    audio = extract_audio(mm)
    duration = audio.numel() / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    return {
        "text_len": len(text),
        "audio_duration": duration,
        "inference_time": elapsed,
        "rtf": rtf,
        "peak_gpu_mem_gb": peak_mem,
        "audio": audio,
    }


def main():
    parser = FlexibleArgumentParser(description="VoxCPM2 benchmark")
    parser.add_argument("--model", type=str, default="openbmb/VoxCPM2")
    parser.add_argument("--stage-configs-path", type=str, default=DEFAULT_STAGE_CONFIGS_PATH)
    parser.add_argument("--text-length", type=str, default="medium", choices=["short", "medium", "long", "all"])
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="benchmark_output")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    parser.add_argument(
        "--deep-profile",
        action="store_true",
        help="Enable VOXCPM2_PROFILE=1 for sub-component breakdown (deferred-sync)",
    )
    args = parser.parse_args()

    if args.deep_profile:
        os.environ["VOXCPM2_PROFILE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
    )

    texts = TEST_TEXTS if args.text_length == "all" else {args.text_length: TEST_TEXTS[args.text_length]}

    for text_name, text in texts.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {text_name} ({len(text)} chars)")
        print(f"{'=' * 60}")

        # Warmup
        for i in range(args.warmup_runs):
            print(f"  Warmup {i + 1}/{args.warmup_runs}...")
            benchmark_single(engine, text, warmup=True)

        # Benchmark runs
        results = []
        for i in range(args.num_runs):
            print(f"  Run {i + 1}/{args.num_runs}...", end=" ")
            result = benchmark_single(engine, text)
            results.append(result)
            print(f"RTF={result['rtf']:.4f} ({result['inference_time']:.2f}s / {result['audio_duration']:.2f}s)")

            # Save audio from last run
            if i == args.num_runs - 1:
                sf.write(str(output_dir / f"{text_name}.wav"), result["audio"].numpy(), SAMPLE_RATE)

        # Summary
        rtfs = [r["rtf"] for r in results]
        times = [r["inference_time"] for r in results]
        durations = [r["audio_duration"] for r in results]
        print(f"\n  Summary ({text_name}):")
        print(f"    RTF:      {min(rtfs):.4f} (best) / {sum(rtfs) / len(rtfs):.4f} (avg)")
        print(f"    Latency:  {min(times):.3f}s (best) / {sum(times) / len(times):.3f}s (avg)")
        print(f"    Duration: {durations[0]:.2f}s")
        print(f"    GPU Mem:  {results[-1]['peak_gpu_mem_gb']:.2f} GB")

    # Run torch profiler if requested
    if args.profile:
        print(f"\n{'=' * 60}")
        print("Running torch profiler...")
        print(f"{'=' * 60}")
        text = TEST_TEXTS["medium"]

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            benchmark_single(engine, text)

        # Print key averages
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # Export chrome trace
        trace_path = str(output_dir / "voxcpm2_profile.json")
        prof.export_chrome_trace(trace_path)
        print(f"Trace saved to: {trace_path}")

    if args.deep_profile:
        print(f"\n{'=' * 60}")
        print("Deep profile results (sub-component breakdown)")
        print(f"{'=' * 60}")
        print("Note: VOXCPM2_PROFILE=1 was set. The per-step breakdown is logged")
        print("by the model every 20 steps. Check the engine logs above for the")
        print("detailed breakdown table. The final request summary is logged")
        print("at REQUEST DONE.")


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
