"""Live MiniCPM-o Realtime acceptance probe for the two overlap modes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
    wait_for,
)


def _data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _target_done(
    events: list[dict[str, object]],
    response_id: str,
) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if event.get("type") == "response.done" and RealtimeEventCollector.response_id(event) == response_id
    ]


def _response_status(event: dict[str, object]) -> tuple[object, object]:
    response = event.get("response")
    if not isinstance(response, dict):
        return None, None
    details = response.get("status_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return response.get("status"), reason


def _event_time(
    events: list[dict[str, object]],
    event_type: str,
    *,
    after_index: int = 0,
) -> float | None:
    for event in events[after_index:]:
        if event.get("type") != event_type:
            continue
        received_at = event.get("_client_received_at_s")
        if isinstance(received_at, int | float):
            return float(received_at)
    return None


async def _open_client(args: argparse.Namespace, *, mode: str) -> RealtimeDuplexClient:
    session_id = f"server-vad-{mode}-{uuid.uuid4().hex}"
    client = RealtimeDuplexClient(
        build_realtime_url(
            args.url,
            args.model,
            autostart=False,
            session_id=session_id,
        )
    )
    await client.__aenter__()
    turn_detection = None
    if mode == "server_vad":
        turn_detection = {
            "type": "server_vad",
            "threshold": args.vad_threshold,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
            "min_speech_duration_ms": 96,
            "interrupt_response": True,
        }
    await client.configure(
        args.model,
        ref_audio=_data_url(Path(args.ref_audio)),
        session_id=session_id,
        temperature=0.0,
        turn_detection=turn_detection,
        timeout_s=args.timeout_s,
    )
    return client


async def _start_response(
    client: RealtimeDuplexClient,
    input_pcm16: bytes,
    *,
    timeout_s: float,
    trailing_silence_ms: int = 0,
) -> tuple[str, int]:
    cursor = len(client.events.events)
    await client.stream_pcm16(input_pcm16, chunk_ms=200, realtime=True)
    if trailing_silence_ms > 0:
        silence_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * trailing_silence_ms // 1000
        await client.stream_pcm16(bytes(silence_bytes), chunk_ms=200, realtime=True)
    await client.commit()
    await wait_for(
        lambda: any(event.get("type") == "response.created" for event in client.events.events[cursor:]),
        timeout_s=timeout_s,
        label="response.created",
    )
    created_index = next(
        index
        for index, event in enumerate(client.events.events[cursor:], start=cursor)
        if event.get("type") == "response.created"
    )
    response_id = RealtimeEventCollector.response_id(client.events.events[created_index])
    if not response_id:
        raise AssertionError("response.created did not include a response id")
    return response_id, created_index


async def _run_listen_only(
    args: argparse.Namespace,
    input_pcm16: bytes,
    interrupt_pcm16: bytes,
) -> dict[str, object]:
    client = await _open_client(args, mode="listen_only")
    try:
        response_id, created_index = await _start_response(
            client,
            input_pcm16,
            timeout_s=args.timeout_s,
            trailing_silence_ms=1600,
        )
        await client.stream_pcm16(interrupt_pcm16, chunk_ms=200, realtime=True)
        await wait_for(
            lambda: bool(_target_done(client.events.events, response_id)),
            timeout_s=args.timeout_s,
            label="listen_only target response.done",
        )
        terminals = _target_done(client.events.events, response_id)
        status, reason = _response_status(terminals[0])
        if len(terminals) != 1 or status == "cancelled":
            raise AssertionError(f"listen_only unexpectedly cancelled {response_id}: {status=}, {reason=}")
        return {
            "response_id": response_id,
            "status": status,
            "reason": reason,
            "speech_started_after_response": sum(
                event.get("type") == "input_audio_buffer.speech_started"
                for event in client.events.events[created_index + 1 :]
            ),
            "cancelled_terminals": sum(
                _response_status(event)[0] == "cancelled"
                for event in client.events.events[created_index + 1 :]
                if event.get("type") == "response.done"
            ),
            "events": client.events.events,
        }
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)


async def _run_server_vad(
    args: argparse.Namespace,
    input_pcm16: bytes,
    interrupt_pcm16: bytes,
) -> dict[str, object]:
    client = await _open_client(args, mode="server_vad")
    try:
        response_id, created_index = await _start_response(
            client,
            input_pcm16,
            timeout_s=args.timeout_s,
            trailing_silence_ms=1600,
        )
        interrupt_cursor = len(client.events.events)
        await client.stream_pcm16(interrupt_pcm16, chunk_ms=200, realtime=True)
        await wait_for(
            lambda: bool(_target_done(client.events.events, response_id)),
            timeout_s=args.timeout_s,
            label="server_vad target response.done",
        )
        terminals = _target_done(client.events.events, response_id)
        status, reason = _response_status(terminals[0])
        public_types = {event.get("type") for event in client.events.events[created_index + 1 :]}
        starts = sum(
            event.get("type") == "input_audio_buffer.speech_started"
            for event in client.events.events[interrupt_cursor:]
        )
        if len(terminals) != 1 or (status, reason) != ("cancelled", "turn_detected"):
            raise AssertionError(f"server_vad terminal mismatch: {status=}, {reason=}, count={len(terminals)}")
        if starts != 1:
            raise AssertionError(f"one interrupt utterance must emit one speech_started, got {starts}")
        if "audio.cancelled" in public_types or "conversation.item.truncated" in public_types:
            raise AssertionError(f"non-canonical public cancellation event: {public_types}")
        speech_started_at = _event_time(
            client.events.events,
            "input_audio_buffer.speech_started",
            after_index=interrupt_cursor,
        )
        done_at = next(
            (
                float(event["_client_received_at_s"])
                for event in terminals
                if isinstance(event.get("_client_received_at_s"), int | float)
            ),
            None,
        )
        return {
            "response_id": response_id,
            "status": status,
            "reason": reason,
            "speech_started_count": starts,
            "cancel_latency_ms": (
                round((done_at - speech_started_at) * 1000, 3)
                if done_at is not None and speech_started_at is not None
                else None
            ),
            "events": client.events.events,
        }
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)


async def _run_client_cancel(
    args: argparse.Namespace,
    input_pcm16: bytes,
) -> dict[str, object]:
    client = await _open_client(args, mode="listen_only")
    try:
        response_id, _ = await _start_response(client, input_pcm16, timeout_s=args.timeout_s)
        sent_at = time.monotonic()
        await client.send({"type": "response.cancel", "response_id": response_id})
        await wait_for(
            lambda: bool(_target_done(client.events.events, response_id)),
            timeout_s=args.timeout_s,
            label="client cancel response.done",
        )
        terminals = _target_done(client.events.events, response_id)
        status, reason = _response_status(terminals[0])
        if len(terminals) != 1 or (status, reason) != ("cancelled", "client_cancelled"):
            raise AssertionError(f"client cancel terminal mismatch: {status=}, {reason=}")
        done_at = terminals[0].get("_client_received_at_s")
        return {
            "response_id": response_id,
            "status": status,
            "reason": reason,
            "cancel_latency_ms": (
                round((float(done_at) - sent_at) * 1000, 3) if isinstance(done_at, int | float) else None
            ),
            "events": client.events.events,
        }
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)


async def _run(args: argparse.Namespace) -> dict[str, object]:
    input_pcm16 = read_pcm16_wav(Path(args.input_wav))
    interrupt = read_pcm16_wav(Path(args.interrupt_wav))
    interrupt_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * args.interrupt_duration_ms // 1000
    interrupt = interrupt[:interrupt_bytes]
    runners = {
        "listen_only": lambda: _run_listen_only(args, input_pcm16, interrupt),
        "server_vad": lambda: _run_server_vad(args, input_pcm16, interrupt),
        "client_cancel": lambda: _run_client_cancel(args, input_pcm16),
    }
    selected = list(runners) if args.scenario == "all" else [args.scenario]
    results: dict[str, dict[str, object]] = {}
    for name in selected:
        results[name] = await runners[name]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        events = result.pop("events")
        (output_dir / f"{name}.events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--interrupt-wav", required=True)
    parser.add_argument("--interrupt-duration-ms", type=int, default=4700)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--scenario",
        choices=("all", "listen_only", "server_vad", "client_cancel"),
        default="all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run(parse_args())), ensure_ascii=False, indent=2))
