# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LIBERO closed-loop benchmark client for pi0.5 OpenPI serving.

This is a software-only robot benchmark path: LIBERO drives robosuite/MuJoCo
OffScreenRenderEnv and the policy is queried through
`/v1/realtime/robot/openpi`.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import math
import pathlib
import re
import statistics
import sys
import time
import uuid
from typing import Any

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional video dependency
    imageio = None

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:  # pragma: no cover - optional benchmark dependency
    benchmark = None
    get_libero_path = None
    OffScreenRenderEnv = None
    _LIBERO_IMPORT_ERROR = exc
else:
    _LIBERO_IMPORT_ERROR = None

try:
    example_dir = str(pathlib.Path(__file__).resolve().parent)
    removed = sys.path and sys.path[0] == example_dir
    if removed:
        sys.path.pop(0)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    finally:
        if removed:
            sys.path.insert(0, example_dir)
except ImportError as exc:  # pragma: no cover - optional serving dependency
    WebsocketClientPolicy = None
    _OPENPI_IMPORT_ERROR = exc
else:
    _OPENPI_IMPORT_ERROR = None


LOGGER = logging.getLogger("pi05_libero_client")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
PI05_IMAGE_SIZE = 224
PI05_ACTION_DIM = 32
LIBERO_ACTION_DIM = 7
DEFAULT_OPENPI_PATH = "/v1/realtime/robot/openpi"
CAMERA_BASE = "observation.images.base_0_rgb"
CAMERA_LEFT_WRIST = "observation.images.left_wrist_0_rgb"
CAMERA_RIGHT_WRIST = "observation.images.right_wrist_0_rgb"


class OpenPIBenchmarkClient:
    def __init__(self, *, host: str, port: int, path: str) -> None:
        if WebsocketClientPolicy is None:
            raise ImportError("pi0.5 LIBERO benchmark requires the `openpi-client` package.") from _OPENPI_IMPORT_ERROR
        self._uri = _openpi_uri(host, port, path)
        self._policy = WebsocketClientPolicy(host=self._uri)
        self.metadata = dict(self._policy.get_server_metadata())
        if not isinstance(self.metadata, dict):
            raise TypeError(f"Expected OpenPI metadata dict, got {type(self.metadata)!r}")

    def infer(self, obs: dict[str, Any]) -> tuple[dict[str, Any], float, float, float]:
        payload = dict(obs)
        payload["return_timing"] = True
        send_ts = time.time()
        t0 = time.perf_counter()
        decoded = self._policy.infer(payload)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        recv_ts = time.time()
        if isinstance(decoded, dict) and decoded.get("type") == "error":
            raise RuntimeError(str(decoded.get("message", decoded)))
        if not isinstance(decoded, dict) or "actions" not in decoded:
            decoded = {"actions": decoded, "policy_timing": {}, "server_timing": {}}
        return decoded, send_ts, recv_ts, roundtrip_ms

    def close(self) -> None:
        ws = getattr(self._policy, "_ws", None)
        if ws is not None:
            ws.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eval_libero(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pi0.5 OpenPI LIBERO closed-loop benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default=DEFAULT_OPENPI_PATH)
    parser.add_argument("--task-suite-name", default="libero_goal")
    parser.add_argument("--task", default="0", help='Task ids such as "0", "0-3", or "0,2,5".')
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=12)
    parser.add_argument("--resize-size", type=int, default=PI05_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="data/pi05_libero")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def _openpi_uri(host: str, port: int, path: str) -> str:
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{int(port)}{path}"


def eval_libero(args: argparse.Namespace) -> None:
    if benchmark is None or get_libero_path is None or OffScreenRenderEnv is None:
        raise ImportError(
            "LIBERO benchmark dependencies are missing. Install LIBERO, robosuite, and MuJoCo first."
        ) from _LIBERO_IMPORT_ERROR
    if int(args.replan_steps) <= 0:
        raise ValueError("--replan-steps must be positive.")
    if int(args.num_trials_per_task) <= 0:
        raise ValueError("--num-trials-per-task must be positive.")
    if args.no_video is False and imageio is None:
        raise ImportError("Video output requires `imageio`, or pass --no-video.")

    np.random.seed(int(args.seed))
    task_suite = benchmark.get_benchmark_dict()[str(args.task_suite_name)]()
    selected_task_ids = _parse_task_spec(args.task, task_suite.n_tasks)
    run_dir = _make_run_dir(args.output_dir, args.run_name)
    run_id = run_dir.name
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "task_suite_name": str(args.task_suite_name),
            "selected_task_ids": selected_task_ids,
            "num_trials_per_task": int(args.num_trials_per_task),
            "max_steps": None if args.max_steps is None else int(args.max_steps),
            "num_steps_wait": int(args.num_steps_wait),
            "replan_steps": int(args.replan_steps),
            "resize_size": int(args.resize_size),
            "seed": int(args.seed),
            "openpi_path": str(args.path),
            "latency_note": "batch=1 websocket calls; per-action latency divides by executed LIBERO actions.",
        },
    )

    client = OpenPIBenchmarkClient(host=args.host, port=args.port, path=args.path)
    try:
        LOGGER.info("OpenPI metadata: %s", client.metadata)
        records = []
        for task_id in selected_task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
            try:
                for episode_idx in range(int(args.num_trials_per_task)):
                    records.append(
                        _run_episode(
                            args=args,
                            client=client,
                            env=env,
                            init_state=init_states[episode_idx],
                            run_dir=run_dir,
                            run_id=run_id,
                            task_id=int(task_id),
                            episode_idx=int(episode_idx),
                            task_description=str(task_description),
                        )
                    )
                    _write_json(run_dir / "episode_log.json", records)
            finally:
                env.close()
        _write_summary(run_dir, records)
    finally:
        client.close()


def _run_episode(
    *,
    args: argparse.Namespace,
    client: OpenPIBenchmarkClient,
    env: Any,
    init_state: Any,
    run_dir: pathlib.Path,
    run_id: str,
    task_id: int,
    episode_idx: int,
    task_description: str,
) -> dict[str, Any]:
    max_steps = int(args.max_steps) if args.max_steps is not None else _default_max_steps(str(args.task_suite_name))
    session_id = str(uuid.uuid4())
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: collections.deque[dict[str, Any]] = collections.deque()
    replay_images: list[np.ndarray] = []
    trace_records: list[dict[str, Any]] = []
    infer_records: list[dict[str, Any]] = []
    done = False
    error = None
    env_step = 0
    infer_id = 0

    while env_step < max_steps + int(args.num_steps_wait):
        try:
            if env_step < int(args.num_steps_wait):
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                env_step += 1
                continue

            if not action_plan:
                pi05_obs, frame = _libero_obs_to_pi05_obs(
                    obs,
                    prompt=task_description,
                    session_id=session_id,
                    resize_size=int(args.resize_size),
                )
                replay_images.append(frame)
                response, send_ts, recv_ts, roundtrip_ms = client.infer(pi05_obs)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] < LIBERO_ACTION_DIM:
                    raise ValueError(f"Expected action chunk [H, >=7], got {chunk.shape!r}")
                exec_len = int(min(int(args.replan_steps), chunk.shape[0]))
                infer_record = _make_infer_record(
                    run_id=run_id,
                    task_suite_name=str(args.task_suite_name),
                    task_id=task_id,
                    episode_idx=episode_idx,
                    infer_id=infer_id,
                    env_step=env_step,
                    client_send_timestamp_s=send_ts,
                    client_recv_timestamp_s=recv_ts,
                    client_roundtrip_ms=roundtrip_ms,
                    policy_timing=_dict_or_empty(response.get("policy_timing")),
                    server_timing=_dict_or_empty(response.get("server_timing")),
                    chunk_actions=chunk,
                    chunk_exec_len=exec_len,
                )
                infer_records.append(infer_record)
                for offset in range(exec_len):
                    action_plan.append(
                        {
                            "action": chunk[offset, :LIBERO_ACTION_DIM].astype(np.float32, copy=True),
                            "infer_record": infer_record,
                            "action_offset_in_chunk": offset,
                        }
                    )
                infer_id += 1

            action_meta = action_plan.popleft()
            action = np.asarray(action_meta["action"], dtype=np.float32)
            obs, reward, done, _ = env.step(action.tolist())
            trace_records.append(
                {
                    "run_id": run_id,
                    "task_suite_name": str(args.task_suite_name),
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "env_step": env_step,
                    "infer_id": int(action_meta["infer_record"]["infer_id"]),
                    "action_offset_in_chunk": int(action_meta["action_offset_in_chunk"]),
                    "executed_action": action.tolist(),
                    "reward": float(reward),
                    "done_after_step": bool(done),
                    "client_roundtrip_ms": action_meta["infer_record"].get("client_roundtrip_ms"),
                    "sample_actions_ms": action_meta["infer_record"].get("sample_actions_ms"),
                    "policy_time_ms": action_meta["infer_record"].get("policy_time_ms"),
                    "serve_time_ms": action_meta["infer_record"].get("serve_time_ms"),
                }
            )
            if done:
                break
            env_step += 1
        except Exception as exc:
            LOGGER.exception("Episode failed")
            error = str(exc)
            break

    episode_dir = _episode_output_dir(
        run_dir=run_dir,
        task_description=task_description,
        suffix="success" if done else "failure",
        task_id=task_id,
        episode_idx=episode_idx,
    )
    trace_path = episode_dir / "trace.jsonl"
    infer_path = episode_dir / "infer.jsonl"
    _write_jsonl(trace_path, trace_records)
    _write_jsonl(infer_path, infer_records)
    video_path = None
    if not args.no_video:
        video_path = episode_dir / ("rollout_success.mp4" if done else "rollout_failure.mp4")
        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

    executed_actions = len(trace_records)
    sample_sum = _sum_present(record.get("sample_actions_ms") for record in infer_records)
    policy_sum = _sum_present(record.get("policy_time_ms") for record in infer_records)
    serve_sum = _sum_present(record.get("serve_time_ms") for record in infer_records)
    roundtrips = [float(record["client_roundtrip_ms"]) for record in infer_records]
    episode_record = {
        "run_id": run_id,
        "task_suite_name": str(args.task_suite_name),
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "success": bool(done),
        "failure_reason": error,
        "max_steps": max_steps,
        "num_steps_wait": int(args.num_steps_wait),
        "env_steps_taken": int(env_step),
        "infer_calls": len(infer_records),
        "executed_action_count": executed_actions,
        "client_roundtrip_mean_ms": _mean_or_none(roundtrips),
        "client_roundtrip_p50_ms": _quantile_or_none(roundtrips, 0.50),
        "client_roundtrip_p95_ms": _quantile_or_none(roundtrips, 0.95),
        "sample_actions_mean_ms": _mean_present(record.get("sample_actions_ms") for record in infer_records),
        "policy_time_mean_ms": _mean_present(record.get("policy_time_ms") for record in infer_records),
        "serve_time_mean_ms": _mean_present(record.get("serve_time_ms") for record in infer_records),
        "avg_sample_actions_per_action_ms": sample_sum / executed_actions if executed_actions else None,
        "avg_policy_time_per_action_ms": policy_sum / executed_actions if executed_actions else None,
        "avg_serve_time_per_action_ms": serve_sum / executed_actions if executed_actions else None,
        "trace_path": str(trace_path.relative_to(run_dir)),
        "infer_path": str(infer_path.relative_to(run_dir)),
        "video_path": None if video_path is None else str(video_path.relative_to(run_dir)),
    }
    LOGGER.info(
        "Episode task=%s ep=%s success=%s infer_calls=%s roundtrip_mean_ms=%s",
        task_id,
        episode_idx,
        bool(done),
        len(infer_records),
        episode_record["client_roundtrip_mean_ms"],
    )
    return episode_record


def _libero_obs_to_pi05_obs(
    obs: dict[str, Any],
    *,
    prompt: str,
    session_id: str,
    resize_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    base = _resize_with_pad_uint8(base, resize_size)
    wrist = _resize_with_pad_uint8(wrist, resize_size)
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    return (
        {
            CAMERA_BASE: base,
            CAMERA_LEFT_WRIST: wrist,
            CAMERA_RIGHT_WRIST: np.zeros_like(wrist),
            "state": state,
            "prompt": str(prompt),
            "session_id": str(session_id),
        },
        base,
    )


def _resize_with_pad_uint8(image: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {arr.shape!r}")
    pil = Image.fromarray(arr.astype(np.uint8, copy=False))
    width, height = pil.size
    scale = min(float(size) / float(width), float(size) / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil = pil.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (int(size), int(size)))
    canvas.paste(pil, ((int(size) - new_width) // 2, (int(size) - new_height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _get_libero_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    env.seed(int(seed))
    return env, str(task_description)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / np.linalg.norm(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    angle = 2.0 * math.acos(float(quat[3]))
    den = math.sqrt(max(1.0 - float(quat[3]) * float(quat[3]), 0.0))
    if den < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (angle / den)).astype(np.float32)


def _default_max_steps(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(task_suite_name, 300)


def _parse_task_spec(spec: str | None, num_tasks: int) -> list[int]:
    if spec is None or str(spec).strip() == "":
        return list(range(int(num_tasks)))
    task_ids: set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match:
            lo = int(match.group(1))
            hi = int(match.group(2))
            if lo > hi:
                raise ValueError(f"Invalid task range {token!r}: start > end.")
            task_ids.update(range(lo, hi + 1))
            continue
        if not token.isdigit():
            raise ValueError(f"Unsupported task selector token {token!r}.")
        task_ids.add(int(token))
    bad = [task_id for task_id in task_ids if task_id < 0 or task_id >= int(num_tasks)]
    if bad:
        raise ValueError(f"Task id(s) out of range: {bad}; valid range is [0, {int(num_tasks) - 1}].")
    return sorted(task_ids)


def _make_infer_record(
    *,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    infer_id: int,
    env_step: int,
    client_send_timestamp_s: float,
    client_recv_timestamp_s: float,
    client_roundtrip_ms: float,
    policy_timing: dict[str, Any],
    server_timing: dict[str, Any],
    chunk_actions: np.ndarray,
    chunk_exec_len: int,
) -> dict[str, Any]:
    policy_time_ms = _float_or_none(server_timing.get("policy_time_ms") or server_timing.get("infer_ms"))
    serve_time_ms = _float_or_none(server_timing.get("serve_time_ms"))
    return {
        "timestamp": float(client_recv_timestamp_s),
        "client_send_timestamp_s": float(client_send_timestamp_s),
        "client_recv_timestamp_s": float(client_recv_timestamp_s),
        "client_roundtrip_ms": float(client_roundtrip_ms),
        "server_recv_timestamp_s": _float_or_none(server_timing.get("server_recv_timestamp_s")),
        "server_response_timestamp_s": _float_or_none(server_timing.get("server_response_timestamp_s")),
        "run_id": run_id,
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "infer_id": int(infer_id),
        "env_step": int(env_step),
        "chunk_exec_len": int(chunk_exec_len),
        "chunk_actions": np.asarray(chunk_actions, dtype=np.float32).tolist(),
        "preprocess_ms": _float_or_none(policy_timing.get("preprocess_ms")),
        "sample_actions_ms": _float_or_none(policy_timing.get("sample_actions_ms")),
        "postprocess_ms": _float_or_none(policy_timing.get("postprocess_ms")),
        "total_ms": _float_or_none(policy_timing.get("total_ms")),
        "policy_time_ms": policy_time_ms,
        "serve_time_ms": serve_time_ms,
        "ws_unpack_ms": _float_or_none(server_timing.get("ws_unpack_ms")),
        "ws_pack_ms": _float_or_none(server_timing.get("ws_pack_ms")),
        "action_dim": int(chunk_actions.shape[1]),
        "action_horizon": int(chunk_actions.shape[0]),
    }


def _episode_output_dir(
    *,
    run_dir: pathlib.Path,
    task_description: str,
    suffix: str,
    task_id: int,
    episode_idx: int,
) -> pathlib.Path:
    task_segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_description.strip())[:120]
    path = run_dir / "episodes" / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_run_dir(output_dir: str, run_name: str | None) -> pathlib.Path:
    base_dir = pathlib.Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None or str(run_name).strip() == "":
        run_name = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_summary(run_dir: pathlib.Path, records: list[dict[str, Any]]) -> None:
    roundtrip_values = _present(record.get("client_roundtrip_mean_ms") for record in records)
    summary = {
        "episodes": len(records),
        "successes": sum(1 for record in records if record.get("success")),
        "client_roundtrip_episode_mean_ms": _mean_or_none(roundtrip_values),
        "client_roundtrip_episode_p50_ms": _quantile_or_none(roundtrip_values, 0.50),
        "client_roundtrip_episode_p95_ms": _quantile_or_none(roundtrip_values, 0.95),
        "infer_calls": sum(int(record.get("infer_calls", 0)) for record in records),
        "executed_action_count": sum(int(record.get("executed_action_count", 0)) for record in records),
    }
    _write_json(run_dir / "summary.json", summary)
    LOGGER.info("Summary: %s", summary)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _present(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _sum_present(values: Any) -> float:
    return float(sum(_present(values)))


def _mean_present(values: Any) -> float | None:
    return _mean_or_none(_present(values))


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _quantile_or_none(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return float(ordered[index])


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
