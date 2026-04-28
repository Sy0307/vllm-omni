# Round 2 baseline validation

Date: 2026-04-29

## Lesson selection

- `.humanize/bitlesson.md`: read.
- Lesson ID: NONE. The file only contains the template and no reusable entries.
- `humanize:bitlesson-selector`: requested, but unavailable in this Claude session's skill list; treated as NONE.

## Remote environment

- Controller health: OK.
- GPU: 4x NVIDIA H20. At start, GPU 0/1 were already occupied by unrelated VLLM workers; this run used GPU 2 for benchmark attempts.
- Local target commit: `f71688215dab69ec0c218d78eefe726eba86494b`.
- Remote source commit available: `e0fc6e8c3a15a5f29296220d770d8d44acc953fc`.
- Remote validation worktree: `/home/admin/workspace/aop_lab/model_runner_v2/vllm-omni-worktrees/rlcr-r2-baseline-f7168821`.
- Remote validation commit after applying the local `e0fc6e8c..f7168821` patch: `585a842c81e0e58e0f699866214a281ed82db87f`.

## Pytest

Command run remotely:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/core/sched/test_acoustic_inner_loop.py
```

Result:

```text
....                                                                     [100%]
--- Running Summary
4 passed, 16 warnings in 0.33s
```

## Qwen3 TTS NV talker benchmark

Baseline settings:

- K: `1`, via `VLLM_OMNI_QWEN3_TTS_NV_INNER_LOOP_STEPS=1`.
- Async scheduling: disabled, via `VLLM_OMNI_BENCH_ASYNC_SCHEDULING=0`; logs confirmed `Asynchronous scheduling is disabled.`.
- Model path attempted: `/home/admin/workspace/aop_lab/model_runner_v2/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- Available remote Qwen3 TTS model dirs found: `Qwen3-TTS-0___6B-Instruct`, `Qwen3-TTS-12Hz-1___7B-Base`; no CustomVoice model dir was available under the shared Qwen model root.

First command attempted:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 VLLM_OMNI_BENCH_ASYNC_SCHEDULING=0 VLLM_OMNI_QWEN3_TTS_NV_INNER_LOOP_STEPS=1 python3 examples/online_serving/qwen3_tts_nv_triton/benchmark_model.py --model /home/admin/workspace/aop_lab/model_runner_v2/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base --num-requests 10 --concurrency 1 --num-warmups 1 --max-new-tokens 512 --gpu-memory-utilization 0.4 --stage-init-timeout 600 --config-name baseline_k1_async_off
```

Outcome: failed during stage initialization because Python imported the remote installed/source `vllm_omni` from `/home/admin/workspace/remote_workspace/vllm-omni-origin-main`, where `Qwen3TTSTalkerForConditionalGenerationNv` was not registered.

Second command attempted:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/admin/workspace/aop_lab/model_runner_v2/vllm-omni-worktrees/rlcr-r2-baseline-f7168821 PYTHONDONTWRITEBYTECODE=1 VLLM_OMNI_BENCH_ASYNC_SCHEDULING=0 VLLM_OMNI_QWEN3_TTS_NV_INNER_LOOP_STEPS=1 python3 examples/online_serving/qwen3_tts_nv_triton/benchmark_model.py --model /home/admin/workspace/aop_lab/model_runner_v2/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base --num-requests 10 --concurrency 1 --num-warmups 1 --max-new-tokens 512 --gpu-memory-utilization 0.4 --stage-init-timeout 600 --config-name baseline_k1_async_off
```

Outcome:

- Engine initialized successfully from the validation worktree.
- Logs confirmed async scheduling disabled.
- Warmup first request failed in model preprocess with:

```text
ValueError: Unsupported CustomVoice speaker: 'vivian' (known: <none>)
```

- The benchmark then attempted 10 measured requests after the warmup, but the orchestrator/engine was already dead from the same root cause.
- Completed requests: 0.
- Generated tokens: 0.
- E2E latency: unavailable.
- Token throughput: unavailable.
- Request throughput: unavailable.

Blocker: `benchmark_model.py` is hardwired for `task_type=["CustomVoice"]` and defaults to speaker `vivian`, while the available remote 1.7B Base checkpoint has no `spk_id` map for CustomVoice speakers. A valid baseline performance run needs the matching `Qwen3-TTS-12Hz-1.7B-CustomVoice` checkpoint or a benchmark path that supports the Base checkpoint input contract.

## Cleanup check

After the benchmark attempt, GPU 2 returned to 4 MiB usage. The only remaining GPU processes were pre-existing unrelated VLLM workers on GPU 0/1.
