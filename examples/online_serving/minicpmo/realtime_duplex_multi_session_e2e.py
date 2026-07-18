"""Concurrent MiniCPM-o Realtime duplex and resumable-session E2E driver."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets

from examples.online_serving.minicpmo.realtime_duplex_demo import (
    _url_with_model,
    run_demo,
)


def _with_resume_mode(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["resume"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _response_ids(result: dict[str, object]) -> set[str]:
    values = result.get("completed_response_ids")
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


def _validate_identity_isolation(results: list[dict[str, object]]) -> bool:
    seen: set[str] = set()
    for result in results:
        current = _response_ids(result)
        if current & seen:
            return False
        seen.update(current)
    return True


async def _receive_until(ws, event_type: str, *, timeout_s: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    async def receive() -> tuple[dict[str, object], list[dict[str, object]]]:
        events: list[dict[str, object]] = []
        while True:
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == event_type:
                return event, events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _resume_probe(
    args: argparse.Namespace,
    *,
    session_id: str,
    expect_expired: bool = False,
) -> dict[str, object]:
    url = _url_with_model(args.url, args.model, session_id=session_id)
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as first:
        await first.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": args.model,
                        "modalities": ["audio", "text"],
                        "extra_body": {"minicpmo45_native_duplex": True},
                    },
                }
            )
        )
        created, first_events = await _receive_until(first, "session.created", timeout_s=args.timeout_s)
        token = created.get("resume_token")
        incarnation = created.get("incarnation")
        generation = created.get("attachment_generation")
        if not isinstance(token, str) or not isinstance(incarnation, int):
            raise RuntimeError("session.created omitted resumable credentials")
        last_seq = max(
            (
                event.get("server_event_seq", 0)
                for event in first_events
                if isinstance(event.get("server_event_seq"), int)
            ),
            default=0,
        )

    delay_s = args.expire_after_s if expect_expired else args.resume_after_ms / 1000
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    resume_url = _with_resume_mode(url)
    async with websockets.connect(resume_url, max_size=64 * 1024 * 1024) as second:
        await second.send(
            json.dumps(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "incarnation": incarnation,
                    "resume_token": token,
                    "last_received_server_event_seq": last_seq,
                }
            )
        )
        if expect_expired:
            error, error_events = await _receive_until(second, "error", timeout_s=args.timeout_s)
            error_payload = error.get("error")
            code = error_payload.get("code") if isinstance(error_payload, dict) else error.get("code")
            return {
                "ok": code == "session_resume_expired",
                "session_id": session_id,
                "expired": True,
                "error_code": code,
                "event_count": len(error_events),
            }
        resumed, replay = await _receive_until(second, "session.resumed", timeout_s=args.timeout_s)
        rotated = resumed.get("resume_token")
        if not isinstance(rotated, str) or rotated == token:
            raise RuntimeError("session.resume did not rotate the resume token")
        await second.send(json.dumps({"type": "session.heartbeat"}))
        heartbeat, heartbeat_events = await _receive_until(
            second,
            "session.heartbeat_ack",
            timeout_s=args.timeout_s,
        )
        await second.send(json.dumps({"type": "session.close"}))
        closed, close_events = await _receive_until(second, "session.closed", timeout_s=args.timeout_s)

    replay_sequences = [event["server_event_seq"] for event in replay if isinstance(event.get("server_event_seq"), int)]
    return {
        "ok": (
            resumed.get("session_id") == session_id
            and isinstance(generation, int)
            and resumed.get("attachment_generation") == generation + 1
            and heartbeat.get("session_id") == session_id
            and closed.get("session_id") == session_id
            and replay_sequences == sorted(replay_sequences)
        ),
        "session_id": session_id,
        "initial_attachment_generation": generation,
        "resumed_attachment_generation": resumed.get("attachment_generation"),
        "replayed_event_count": len(replay_sequences),
        "heartbeat_event_count": len(heartbeat_events),
        "close_event_count": len(close_events),
        "token_rotated": True,
    }


def _demo_args(args: argparse.Namespace, index: int) -> SimpleNamespace:
    validation_mode = "response-required" if args.response_required else "model-policy"
    return SimpleNamespace(
        url=args.url,
        model=args.model,
        session_id=f"multi-{index}-{uuid.uuid4().hex}",
        input_wav=args.input_wav,
        turn_input_wav=list(args.turn_input_wav),
        output_dir=str(Path(args.output_dir) / f"session_{index:02d}"),
        output_audio_format="pcm16",
        chunk_ms=args.chunk_ms,
        realtime_input=args.realtime_input,
        first_turn_ms=args.first_turn_ms,
        turn_duration_ms=list(args.turn_duration_ms),
        first_turn_transcript=f"session {index} input",
        omit_transcript_hints=True,
        validation_mode=validation_mode,
        scenario="sequential",
        require_audio=args.response_required,
        require_distinct_inputs=False,
        expect_empty_turn=[],
        short_ack_ms=350,
        turns=args.turns,
        timeout_s=args.timeout_s,
        model_policy_settle_ms=args.model_policy_settle_ms,
    )


async def run_multi_session(args: argparse.Namespace) -> dict[str, object]:
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_result = None
    if args.disconnect_session_index is not None:
        if not 0 <= args.disconnect_session_index < args.sessions:
            raise ValueError("--disconnect-session-index is outside the session range")
        resume_result = await _resume_probe(
            args,
            session_id=f"resume-{args.disconnect_session_index}-{uuid.uuid4().hex}",
        )
    expiry_result = None
    if args.expire_session_index is not None:
        if not 0 <= args.expire_session_index < args.sessions:
            raise ValueError("--expire-session-index is outside the session range")
        expiry_result = await _resume_probe(
            args,
            session_id=f"expire-{args.expire_session_index}-{uuid.uuid4().hex}",
            expect_expired=True,
        )

    session_results = await asyncio.gather(
        *(run_demo(_demo_args(args, index)) for index in range(args.sessions)),
        return_exceptions=True,
    )
    failures = [repr(result) for result in session_results if isinstance(result, BaseException)]
    completed = [result for result in session_results if isinstance(result, dict)]
    identity_isolation_ok = _validate_identity_isolation(completed)
    result = {
        "ok": (
            not failures
            and len(completed) == args.sessions
            and all(item.get("ok") is True for item in completed)
            and identity_isolation_ok
            and (resume_result is None or resume_result.get("ok") is True)
            and (expiry_result is None or expiry_result.get("ok") is True)
        ),
        "session_count": args.sessions,
        "identity_isolation_ok": identity_isolation_ok,
        "resume": resume_result,
        "expiry": expiry_result,
        "failures": failures,
        "sessions": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime?duplex=1")
    parser.add_argument("--base-url", help="Deprecated alias; /v1/realtime is appended when supplied.")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--turn-input-wav", action="append", default=[])
    parser.add_argument("--output-dir", default="/tmp/minicpmo_pr3907_multi_session_e2e")
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument("--turn-duration-ms", type=int, action="append", default=[])
    parser.add_argument("--response-required", action="store_true")
    parser.add_argument("--disconnect-session-index", type=int)
    parser.add_argument("--resume-after-ms", type=int, default=1000)
    parser.add_argument("--expire-session-index", type=int)
    parser.add_argument("--expire-after-s", type=float, default=6.0)
    parser.add_argument("--model-policy-settle-ms", type=int, default=600)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.base_url:
        args.url = args.base_url.rstrip("/") + "/v1/realtime?duplex=1"
    return args


def main() -> None:
    result = asyncio.run(run_multi_session(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
