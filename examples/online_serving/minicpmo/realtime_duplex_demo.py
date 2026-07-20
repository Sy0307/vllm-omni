"""Minimal single-input MiniCPM-o 4.5 Realtime duplex demo.

Run this after starting the duplex server. Strict lifecycle, overlap, and
multi-session validation lives under ``tests/e2e/online_serving``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    RealtimeDuplexClient,
    build_realtime_url,
    read_pcm16_wav,
    wait_for,
    write_pcm16_wav,
)


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    input_pcm16 = read_pcm16_wav(Path(args.input_wav))
    if not input_pcm16:
        raise ValueError("input WAV has no audio")

    url = build_realtime_url(
        args.url,
        args.model,
        session_id=args.session_id,
    )
    async with RealtimeDuplexClient(url) as client:
        await client.configure(
            args.model,
            session_id=args.session_id,
            timeout_s=args.timeout_s,
        )
        before_done = client.events.count("response.done")
        before_listen = client.events.count("response.listen")
        await client.stream_pcm16(
            input_pcm16,
            chunk_ms=args.chunk_ms,
            realtime=not args.no_realtime_pacing,
        )
        commit_sent_at_s = time.monotonic()
        await client.commit()
        await wait_for(
            lambda: (
                client.events.count("response.done") > before_done
                or client.events.count("response.listen") > before_listen
            ),
            timeout_s=args.timeout_s,
            label="model speak/listen decision",
        )
        await client.acknowledge_playback()
        await client.close_session(timeout_s=args.timeout_s)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio = client.events.audio_bytes()
        first_text_at_s = client.events.first_received_at(
            "response.audio_transcript.delta",
            "response.output_text.delta",
            after_s=commit_sent_at_s,
        )
        first_audio_at_s = client.events.first_received_at(
            "response.audio.delta",
            after_s=commit_sent_at_s,
        )
        response_created_at_s = client.events.first_received_at(
            "response.created",
            after_s=commit_sent_at_s,
        )
        response_done_at_s = client.events.last_received_at("response.done")
        audio_duration_s = len(audio) / (client.events.output_sample_rate_hz * 2)
        response_generation_s = (
            response_done_at_s - response_created_at_s
            if response_done_at_s is not None and response_created_at_s is not None
            else None
        )
        transcript_deltas = [
            str(event.get("delta", ""))
            for event in client.events.events
            if event.get("type")
            in {
                "response.audio_transcript.delta",
                "response.output_text.delta",
            }
        ]
        response_id = client.events.response_ids[0] if client.events.response_ids else None
        timing = client.events.timing_summary(
            after_s=commit_sent_at_s,
            input_committed_at_s=commit_sent_at_s,
            response_id=response_id,
        )
        (output_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
            encoding="utf-8",
        )
        (output_dir / "output.pcm").write_bytes(audio)
        if audio:
            write_pcm16_wav(
                output_dir / "output.wav",
                audio,
                sample_rate_hz=client.events.output_sample_rate_hz,
            )

        result = {
            "ok": (
                client.events.count("session.created") > 0
                and client.events.count("session.closed") > 0
                and not client.events.errors()
                and (bool(audio) or not args.require_audio)
            ),
            "model_decision": "speak" if audio else "listen",
            "audio_bytes": len(audio),
            "output_sample_rate_hz": client.events.output_sample_rate_hz,
            "latency": {
                "ttft_ms": (
                    round((first_text_at_s - commit_sent_at_s) * 1000, 2) if first_text_at_s is not None else None
                ),
                "ttfp_ms": (
                    round((first_audio_at_s - commit_sent_at_s) * 1000, 2) if first_audio_at_s is not None else None
                ),
                "rtf": (
                    round(response_generation_s / audio_duration_s, 4)
                    if response_generation_s is not None and audio_duration_s > 0
                    else None
                ),
                "response_generation_ms": (
                    round(response_generation_s * 1000, 2) if response_generation_s is not None else None
                ),
                "text_stream_ms": (
                    round((response_done_at_s - first_text_at_s) * 1000, 2)
                    if response_done_at_s is not None and first_text_at_s is not None
                    else None
                ),
                "transcript_delta_count": len(transcript_deltas),
                "audio_duration_s": round(audio_duration_s, 3),
                "measurement_origin": "input_audio_buffer.commit send",
            },
            "timing": timing,
            "response_ids": client.events.response_ids,
            "transcript": "".join(transcript_deltas),
            "errors": client.events.errors(),
            "output_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--session-id")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--require-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
