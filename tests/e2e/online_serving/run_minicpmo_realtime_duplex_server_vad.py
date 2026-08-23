"""MiniCPM-o server-VAD hard-interruption acceptance driver."""

from __future__ import annotations

import base64
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


def _events(events: list[dict[str, object]], response_id: str, kind: str) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if event.get("type") == kind and RealtimeEventCollector.response_id(event) == response_id
    ]


async def run_server_vad_interrupt(args) -> dict[str, object]:
    initial = read_pcm16_wav(Path(args.input_wav))
    interrupt = read_pcm16_wav(Path(args.interrupt_wav))
    interrupt = interrupt[: PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 4700 // 1000]
    session_id = f"server-vad-hard-interrupt-{id(args)}"
    client = RealtimeDuplexClient(build_realtime_url(args.url, args.model, autostart=False, session_id=session_id))
    try:
        await client.__aenter__()
        await client.configure(
            args.model,
            ref_audio="data:audio/wav;base64," + base64.b64encode(Path(args.ref_audio).read_bytes()).decode(),
            session_id=session_id,
            turn_detection={
                "type": "server_vad",
                "interrupt_response": True,
            },
            timeout_s=args.timeout_s,
        )
        await client.stream_pcm16(initial + bytes(16_000 * 2 * 1600 // 1000), chunk_ms=200, realtime=True)
        await client.commit()
        await wait_for(
            lambda: any(e.get("type") == "response.created" for e in client.events.events),
            timeout_s=args.timeout_s,
            label="response.created",
        )
        target = next(e for e in client.events.events if e.get("type") == "response.created")
        target_id = RealtimeEventCollector.response_id(target)
        assert target_id
        await wait_for(
            lambda: _events(client.events.events, target_id, "response.audio.delta"),
            timeout_s=args.timeout_s,
            label="response.audio.delta",
        )
        cursor = len(client.events.events)
        await client.stream_pcm16(interrupt + bytes(16_000 * 2 * 800 // 1000), chunk_ms=args.chunk_ms, realtime=True)
        await wait_for(
            lambda: _events(client.events.events, target_id, "response.done"),
            timeout_s=args.timeout_s,
            label="cancelled response.done",
        )
        await client.commit()
        await wait_for(
            lambda: any(
                e.get("type") == "response.done" and RealtimeEventCollector.response_id(e) != target_id
                for e in client.events.events[cursor:]
            ),
            timeout_s=args.timeout_s,
            label="follow-up response.done",
        )
        events = client.events.events
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)

    done = _events(events, target_id, "response.done")
    done_index = events.index(done[0]) if len(done) == 1 else len(events)
    stale = (
        any(
            e.get("type") == "response.audio.delta" and RealtimeEventCollector.response_id(e) == target_id
            for e in events[done_index + 1 :]
        )
        if len(done) == 1
        else False
    )
    followup_audio = any(
        e.get("type") == "response.audio.delta" and RealtimeEventCollector.response_id(e) != target_id
        for e in events[cursor:]
    )
    ok = (
        len(done) == 1
        and isinstance(done[0].get("response"), dict)
        and done[0]["response"].get("status") == "cancelled"
        and done[0]["response"].get("status_details", {}).get("reason") == "turn_detected"
        and not stale
        and any(e.get("type") == "input_audio_buffer.committed" for e in events[cursor:])
        and followup_audio
        and not any(e.get("type") == "error" for e in events)
    )
    return {"ok": ok}
