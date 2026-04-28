# Round 1 Baseline: acoustic inner loop

## Scope

- Local repo: `/Users/sy03/review_3221/vllm-omni`
- Local HEAD: `c537c017a5733b1736996757677ea986d93fe709`
- Target baseline commits noted by coordinator:
  - `21556171 qwen3_tts_nv: add acoustic inner loop prototype`
  - `c537c017 qwen3_tts_nv: initialize RLCR goal tracker`

## BitLesson

- Read: `.humanize/bitlesson.md`
- Entries found: none; the file only contains the entry template.
- Selector attempt: `humanize:bitlesson-selector` was unavailable in this Claude session (`Unknown skill: humanize:bitlesson-selector`).
- Lesson ID: `NONE`

## Local validation

### py_compile

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  vllm_omni/core/sched/acoustic_inner_loop.py \
  vllm_omni/core/sched/omni_ar_scheduler.py \
  vllm_omni/worker/gpu_ar_model_runner.py \
  examples/online_serving/qwen3_tts_nv_triton/benchmark_model.py \
  tests/core/sched/test_acoustic_inner_loop.py
```

Result:

- Failed before running because local `python` is not configured by pyenv.
- Error: `pyenv: python: command not found`; available pyenv version listed: `3.11.8`.

Retry command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  vllm_omni/core/sched/acoustic_inner_loop.py \
  vllm_omni/core/sched/omni_ar_scheduler.py \
  vllm_omni/worker/gpu_ar_model_runner.py \
  examples/online_serving/qwen3_tts_nv_triton/benchmark_model.py \
  tests/core/sched/test_acoustic_inner_loop.py
```

Result: passed with no output.

### Unit test

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/core/sched/test_acoustic_inner_loop.py
```

Result: failed during pytest plugin import because local Python environment lacks `torch`.

Key error:

```text
ImportError: Error importing plugin "tests.helpers.fixtures.env": No module named 'torch'
```

## Remote GPU validation

Remote commands were run through the Chrome bridge helper with controller URL/token supplied only through environment variables. No token or controller URL is written here.

### Controller health

Command shape:

```bash
CHROME_REMOTE_GPU_BASE_URL=<env> CHROME_REMOTE_GPU_TOKEN=<env> \
python3 "$HOME/.claude/skills/chrome-remote-gpu/scripts/chrome_remote_gpu.py" health
```

Result:

```json
{"status":"ok","workspace":"/home/admin/workspace/remote_workspace","log_dir":"/tmp/remote_gpu_logs","tasks_in_memory":530}
```

### GPU status

Command shape:

```bash
CHROME_REMOTE_GPU_BASE_URL=<env> CHROME_REMOTE_GPU_TOKEN=<env> \
python3 "$HOME/.claude/skills/chrome-remote-gpu/scripts/chrome_remote_gpu.py" gpu
```

Result summary:

- Hardware: 4x NVIDIA H20, driver `580.82.07`, CUDA `13.0`.
- GPU 0 occupied by `VLLM::Worker`, about `83564MiB`.
- GPU 1 occupied by two `VLLM::Worker` processes, about `59260MiB` and `14416MiB`.
- GPU 2 and GPU 3 were mostly free (`4MiB` each) at observation time.

### Remote repo state

Command shape:

```bash
CHROME_REMOTE_GPU_BASE_URL=<env> CHROME_REMOTE_GPU_TOKEN=<env> \
python3 "$HOME/.claude/skills/chrome-remote-gpu/scripts/chrome_remote_gpu.py" exec \
  'git rev-parse HEAD && git status --short && git branch --show-current' \
  --cwd /home/admin/workspace/aop_lab/model_runner_v2/vllm-omni
```

Result:

- Remote main repo HEAD: `fd8255d0a08d74421bd998911821cd30f1171903`
- Remote main repo has many unrelated modified and untracked files.
- The remote git version rejected `git branch --show-current` with `error: unknown option 'show-current'`.

Follow-up command shape:

```bash
CHROME_REMOTE_GPU_BASE_URL=<env> CHROME_REMOTE_GPU_TOKEN=<env> \
python3 "$HOME/.claude/skills/chrome-remote-gpu/scripts/chrome_remote_gpu.py" exec \
  'git cat-file -t c537c017a5733b1736996757677ea986d93fe709 && git worktree list' \
  --cwd /home/admin/workspace/aop_lab/model_runner_v2/vllm-omni
```

Result:

```text
fatal: git cat-file: could not get object info
```

Additional ref inspection showed the configured remotes but neither short target commit resolved:

```text
error: malformed object name 21556171
error: malformed object name c537c017
```

Remote Python/torch check:

```bash
CHROME_REMOTE_GPU_BASE_URL=<env> CHROME_REMOTE_GPU_TOKEN=<env> \
python3 "$HOME/.claude/skills/chrome-remote-gpu/scripts/chrome_remote_gpu.py" exec \
  'python3 --version && python3 -c "import torch; print(torch.__version__)"' \
  --cwd /home/admin/workspace/aop_lab/model_runner_v2/vllm-omni
```

Result:

```text
Python 3.10.16
2.10.0+cu128
```

### Remote pytest

Not run against the baseline, because the remote repository does not contain target commit `c537c017a5733b1736996757677ea986d93fe709`, and the main remote repo is dirty and at unrelated HEAD `fd8255d0a08d74421bd998911821cd30f1171903`.

Running `tests/core/sched/test_acoustic_inner_loop.py` there would not validate the requested acoustic inner-loop baseline. I did not mutate the remote main repo to synthesize the missing commit.

### Qwen3 TTS NV talker benchmark

Not run.

Blocking reasons:

- Remote repo does not contain the requested baseline commit, so benchmark results would not correspond to this Round 1 baseline.
- Remote GPU 0 and GPU 1 were already occupied by VLLM worker processes from existing tasks. I did not stop unknown services.
- The benchmark script defaults `async_scheduling` to `True`, with override via `VLLM_OMNI_BENCH_ASYNC_SCHEDULING`; no safe baseline run was possible, so K/E2E/throughput/generated-token metrics are `N/A`.

Baseline benchmark fields:

| Field | Value |
| --- | --- |
| commit hash | `N/A` for remote benchmark; remote lacked baseline commit |
| K | `N/A` |
| async setting | benchmark default is `async_scheduling=True`; no run performed |
| E2E | `N/A` |
| throughput | `N/A` |
| generated tokens | `N/A` |

## Summary

- Local syntax validation passed with `python3 -m py_compile` for the changed Python files.
- Local pytest is blocked by missing `torch`.
- Remote controller and torch environment are healthy, but the remote repository is not synchronized to the baseline commit and is dirty; therefore remote pytest and benchmark were intentionally not treated as valid baseline evidence.
