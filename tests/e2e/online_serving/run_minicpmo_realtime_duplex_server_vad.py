"""Live MiniCPM-o Realtime probe for opt-in server-VAD hard interruption."""

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

_DELTA_EVENT_TYPES = {
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}


def _data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _response_status(event: dict[str, object]) -> tuple[object, object]:
    response = event.get("response")
    if not isinstance(response, dict):
        return None, None
    details = response.get("status_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return response.get("status"), reason


def _event_time(event: dict[str, object]) -> float | None:
    received_at = event.get("_client_received_at_s")
    return float(received_at) if isinstance(received_at, int | float) else None


def _target_events(
    events: list[dict[str, object]],
    response_id: str,
    *,
    event_type: str | None = None,
) -> list[tuple[int, dict[str, object]]]:
    return [
        (index, event)
        for index, event in enumerate(events)
        if RealtimeEventCollector.response_id(event) == response_id
        and (event_type is None or event.get("type") == event_type)
    ]


def summarize_server_vad_interrupt(
    events: list[dict[str, object]],
    *,
    target_response_id: str,
    interrupt_cursor: int,
) -> dict[str, object]:
    """Evaluate the public hard-interruption and post-fence event contract."""
    target_terminals = _target_events(events, target_response_id, event_type="response.done")
    target_terminal_index = target_terminals[0][0] if len(target_terminals) == 1 else None
    target_status, target_reason = (
        _response_status(target_terminals[0][1]) if len(target_terminals) == 1 else (None, None)
    )
    speech_started = [
        (index, event)
        for index, event in enumerate(events[interrupt_cursor:], start=interrupt_cursor)
        if event.get("type") == "input_audio_buffer.speech_started"
    ]
    speech_stopped = [
        (index, event)
        for index, event in enumerate(events[interrupt_cursor:], start=interrupt_cursor)
        if event.get("type") == "input_audio_buffer.speech_stopped"
    ]
    committed = [
        index
        for index, event in enumerate(events[interrupt_cursor:], start=interrupt_cursor)
        if event.get("type") == "input_audio_buffer.committed"
    ]
    cancelled_terminals = [
        event for event in events if event.get("type") == "response.done" and _response_status(event)[0] == "cancelled"
    ]
    stale_target_deltas = (
        [
            event
            for event in events[target_terminal_index + 1 :]
            if event.get("type") in _DELTA_EVENT_TYPES
            and RealtimeEventCollector.response_id(event) == target_response_id
        ]
        if target_terminal_index is not None
        else []
    )

    followup_ids: list[str] = []
    if target_terminal_index is not None:
        for event in events[target_terminal_index + 1 :]:
            if event.get("type") != "response.created":
                continue
            response_id = RealtimeEventCollector.response_id(event)
            if response_id and response_id != target_response_id and response_id not in followup_ids:
                followup_ids.append(response_id)
    completed_followups: list[str] = []
    followup_audio_delta_count = 0
    followup_transcript = ""
    for response_id in followup_ids:
        terminals = _target_events(events, response_id, event_type="response.done")
        if len(terminals) != 1 or _response_status(terminals[0][1])[0] not in {None, "completed"}:
            continue
        completed_followups.append(response_id)
        followup_audio_delta_count += len(_target_events(events, response_id, event_type="response.audio.delta"))
        for _, event in _target_events(events, response_id, event_type="response.audio_transcript.delta"):
            delta = event.get("delta")
            if isinstance(delta, str):
                followup_transcript += delta

    one_cancelled_terminal = (
        len(target_terminals) == 1
        and (target_status, target_reason) == ("cancelled", "turn_detected")
        and len(cancelled_terminals) == 1
    )
    no_post_fence_stale_deltas = target_terminal_index is not None and not stale_target_deltas
    interrupt_committed = bool(
        speech_started
        and speech_stopped
        and committed
        and speech_started[0][0] < speech_stopped[-1][0] <= committed[-1]
    )
    subsequent_response_ok = bool(completed_followups and followup_audio_delta_count > 0)
    public_types = {event.get("type") for event in events[interrupt_cursor:]}
    errors = [event for event in events if event.get("type") == "error"]
    started_at = _event_time(speech_started[0][1]) if speech_started else None
    done_at = _event_time(target_terminals[0][1]) if len(target_terminals) == 1 else None
    result = {
        "ok": bool(
            not errors
            and one_cancelled_terminal
            and no_post_fence_stale_deltas
            and interrupt_committed
            and subsequent_response_ok
            and "conversation.item.truncated" not in public_types
        ),
        "target_response_id": target_response_id,
        "target_status": target_status,
        "target_reason": target_reason,
        "cancelled_count": len(cancelled_terminals),
        "speech_started_count": len(speech_started),
        "speech_stopped_count": len(speech_stopped),
        "one_cancelled_terminal": one_cancelled_terminal,
        "no_post_fence_stale_deltas": no_post_fence_stale_deltas,
        "stale_target_delta_count": len(stale_target_deltas),
        "interrupt_committed": interrupt_committed,
        "completed_followup_count": len(completed_followups),
        "completed_followup_response_ids": completed_followups,
        "followup_audio_delta_count": followup_audio_delta_count,
        "followup_transcript": followup_transcript,
        "subsequent_response_ok": subsequent_response_ok,
        "error_count": len(errors),
        "cancel_latency_ms": round((done_at - started_at) * 1000, 3)
        if done_at is not None and started_at is not None
        else None,
    }
    return result


async def _open_client(args: argparse.Namespace) -> RealtimeDuplexClient:
    session_id = f"server-vad-hard-interrupt-{uuid.uuid4().hex}"
    client = RealtimeDuplexClient(
        build_realtime_url(
            args.url,
            args.model,
            autostart=False,
            session_id=session_id,
        )
    )
    await client.__aenter__()
    await client.configure(
        args.model,
        ref_audio=_data_url(Path(args.ref_audio)),
        session_id=session_id,
        temperature=args.temperature,
        turn_detection={
            "type": "server_vad",
            "threshold": args.vad_threshold,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
            "min_speech_duration_ms": 96,
            "interrupt_response": True,
        },
        timeout_s=args.timeout_s,
    )
    return client


async def _start_spoken_response(
    client: RealtimeDuplexClient,
    input_pcm16: bytes,
    *,
    timeout_s: float,
) -> str:
    cursor = len(client.events.events)
    await client.stream_pcm16(input_pcm16, chunk_ms=200, realtime=True)
    silence_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 1600 // 1000
    await client.stream_pcm16(bytes(silence_bytes), chunk_ms=200, realtime=True)
    await client.commit()
    await wait_for(
        lambda: any(event.get("type") == "response.created" for event in client.events.events[cursor:]),
        timeout_s=timeout_s,
        label="initial response.created",
    )
    created = next(event for event in client.events.events[cursor:] if event.get("type") == "response.created")
    response_id = RealtimeEventCollector.response_id(created)
    if not response_id:
        raise AssertionError("response.created did not include a response id")
    await wait_for(
        lambda: bool(_target_events(client.events.events, response_id, event_type="response.audio.delta")),
        timeout_s=timeout_s,
        label="initial response.audio.delta",
    )
    return response_id


async def run_server_vad_interrupt(args: argparse.Namespace) -> dict[str, object]:
    """Interrupt live model speech and require a clean, useful next response."""
    input_pcm16 = read_pcm16_wav(Path(args.input_wav))
    interrupt_pcm16 = read_pcm16_wav(Path(args.interrupt_wav))
    interrupt_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * args.interrupt_duration_ms // 1000
    interrupt_pcm16 = interrupt_pcm16[:interrupt_bytes]
    client = await _open_client(args)
    try:
        target_response_id = await _start_spoken_response(
            client,
            input_pcm16,
            timeout_s=args.timeout_s,
        )
        interrupt_cursor = len(client.events.events)
        await client.stream_pcm16(interrupt_pcm16, chunk_ms=args.chunk_ms, realtime=True)
        silence_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * args.trailing_silence_ms // 1000
        await client.stream_pcm16(bytes(silence_bytes), chunk_ms=args.chunk_ms, realtime=True)
        await wait_for(
            lambda: bool(_target_events(client.events.events, target_response_id, event_type="response.done")),
            timeout_s=args.timeout_s,
            label="cancelled response.done",
        )
        await client.commit()
        await wait_for(
            lambda: any(
                event.get("type") == "input_audio_buffer.committed" for event in client.events.events[interrupt_cursor:]
            ),
            timeout_s=args.timeout_s,
            label="interrupt input_audio_buffer.committed",
        )
        await wait_for(
            lambda: (
                summarize_server_vad_interrupt(
                    client.events.events,
                    target_response_id=target_response_id,
                    interrupt_cursor=interrupt_cursor,
                )["subsequent_response_ok"]
                is True
            ),
            timeout_s=args.timeout_s,
            label="completed response after server-VAD interruption",
        )
        result = summarize_server_vad_interrupt(
            client.events.events,
            target_response_id=target_response_id,
            interrupt_cursor=interrupt_cursor,
        )
        await client.acknowledge_playback()
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--interrupt-wav", required=True)
    parser.add_argument("--interrupt-duration-ms", type=int, default=4700)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--trailing-silence-ms", type=int, default=800)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    started_at = time.monotonic()
    summary = asyncio.run(run_server_vad_interrupt(parse_args()))
    summary["wall_time_s"] = round(time.monotonic() - started_at, 3)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
