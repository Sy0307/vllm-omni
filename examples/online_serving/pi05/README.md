# pi0.5 VLA — OpenPI realtime serving

pi0.5 is a Vision-Language-Action model from Physical Intelligence, available
through LeRobot as `lerobot/pi05_base`. It consumes multi-camera images, a
language instruction, and robot proprioceptive state, then returns a continuous
action chunk. It does not emit text tokens.

The important runtime difference from pi0 is state handling: pi0.5 normalizes
and discretizes the state vector into the language prompt (`Task: ..., State:
...; Action:`), so the action expert denoises action tokens only.

## Install extras

The core `pip install -e .` does not include the OpenPI client used here:

- `openpi-client` (from the [openpi](https://github.com/Physical-Intelligence/openpi)
  repo: `pip install -e packages/openpi-client`), `websockets`, `msgpack`, `msgpack-numpy`

## Run the server

```bash
vllm serve lerobot/pi05_base --omni --port 8000 \
    --served-model-name pi05 \
    --deploy-config vllm_omni/deploy/pi05.yaml \
    --enforce-eager --disable-log-stats
```

## Run the client

```bash
python examples/online_serving/pi05/openpi_client.py --host 127.0.0.1 --port 8000 \
    --prompt "pick up the red block and place it in the bin"
```

The client sends a flat OpenPI observation:

```python
{
    "observation.images.base_0_rgb":        np.uint8[H, W, 3],
    "observation.images.left_wrist_0_rgb":  np.uint8[H, W, 3],
    "observation.images.right_wrist_0_rgb": np.uint8[H, W, 3],
    "state":   np.float32[state_dim],
    "prompt":  "pick up the red block",
    "session_id": "<uuid>",
}
```

Camera keys must match `vllm_omni/deploy/pi05.yaml` or be translated with
`model_config.image_key_map`.

## Run the LIBERO closed-loop benchmark

`libero_client.py` follows the software simulation setup used by
`dexmal/realtime-vla-flash`: LIBERO drives robosuite/MuJoCo
`OffScreenRenderEnv`, while vLLM-Omni serves the pi0.5 policy through the
OpenPI websocket endpoint.

```bash
python examples/online_serving/pi05/libero_client.py \
    --host 127.0.0.1 --port 8000 \
    --task-suite-name libero_goal --task 0 \
    --num-trials-per-task 1 --max-steps 50 --replan-steps 12 \
    --output-dir data/pi05_libero --run-name smoke --no-video
```

The benchmark writes:

- `manifest.json`
- `episode_log.json`
- `summary.json`
- per-episode `trace.jsonl` and `infer.jsonl`
- optional rollout videos unless `--no-video` is passed

Latency is batch=1 websocket latency. `infer.jsonl` records
`client_roundtrip_ms`, `policy_time_ms`, `serve_time_ms`, `ws_unpack_ms`,
`ws_pack_ms`, and pi0.5 pipeline timings such as `preprocess_ms`,
`sample_actions_ms`, `postprocess_ms`, and `total_ms`.

The public `lerobot/pi05_base` checkpoint is not a LIBERO-finetuned policy, so
success rate from this script is only a plumbing signal unless you serve a
LIBERO-finetuned pi0.5 checkpoint. The latency and end-to-end simulation path
are still valid for regression testing.
