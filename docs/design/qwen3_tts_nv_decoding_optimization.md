# Qwen3 TTS NV Decoding Step Optimization Notes

## Background

This work started from the performance review of PR 3221:

<https://github.com/vllm-project/vllm-omni/pull/3221>

The original request was not a normal code review. The goal was to deeply
investigate why Qwen3 TTS NV talker decoding is slow, use profiling and
comparative experiments to identify the root cause, and then implement and
benchmark a real optimization.

The initial context included:

- A remote GPU environment reachable through the Chrome remote GPU helper.
- A Qwen3 TTS NV talker model available on the remote machine.
- A PR comment pointing out decoding gaps.
- The option to use `vllm-torch-profiler-analysis` or other profiling methods.
- A requirement to compare performance before and after any proposed fix.

The important profile observation was that there are visible gaps between
decoding steps. The author also noted that this is a problem for both
`vllm-omni` main and the PR branch, with roughly `~2ms` overhead per decoding
step. That made the first hypothesis clear: the issue is not only a handful of
small GPU kernels, but also the fixed per-token cost of repeatedly going
through the vLLM engine, scheduler, worker, sampler, output processing, and the
next scheduling round.

## Initial Requirements

The initial requirements can be summarized as follows.

1. Investigate why decoding is slow.

   The work needed to be profile-driven. The requirement was to inspect decoding
   steps, compare experiments, and understand whether the slowdown came from
   small kernels, CPU/GPU gaps, scheduler overhead, sampler overhead, or model
   structure.

2. Find the root cause and try to fix it.

   The profile showed many small kernels, so kernel fusion was one possible
   direction. However, the deeper question was whether those kernels were the
   main bottleneck or only symptoms of a larger per-token execution structure.

3. Run real before-and-after benchmarks.

   The task explicitly required experiments, not just analysis. Any optimization
   needed to be measured against a baseline.

4. Do not use the wrong semantic shortcut.

   The final user direction was explicit: do not borrow vLLM text speculative
   decoding's repeated draft semantics. Instead, implement a real acoustic
   multi-token inner loop:

   - decode K acoustic tokens continuously;
   - advance KV/cache state each substep;
   - advance hidden state each substep;
   - advance code predictor state each substep;
   - preserve correctness instead of treating acoustic tokens as text speculative
     draft tokens.

5. Do not focus on small, certain optimizations first.

   The user explicitly said not to start with small optimizations such as:

   - `omit_hidden_output`;
   - skipping decode placeholder embedding copy;
   - benchmark-only environment switches.

   The main implementation should directly target multi-token acoustic decoding.

## Current Progress

The task has been advanced from a profiling suspicion to a measured structural
optimization prototype.

The implemented prototype is a direct acoustic inner loop. The scheduler
reserves multiple acoustic decode slots in one engine step, and the GPU model
runner then performs K sequential acoustic decode substeps inside the worker.
Each substep advances the real model state instead of using speculative decode
semantics.

The current implementation is intentionally conservative. It only enables the
inner loop for the target benchmark-like path:

- non-async scheduling;
- single running request;
- decode-only stage;
- no speculative tokens;
- no structured output;
- no logprobs;
- no prefix cache.

This restriction is deliberate. The current purpose is to prove the performance
direction on Qwen3 TTS NV talker first, not to immediately support every vLLM
scheduling mode.

## Code Changes

The current local implementation touches the following files:

- `vllm_omni/core/sched/acoustic_inner_loop.py`
  - Adds small helper functions for acoustic inner-loop slot calculation and
    early-stop correction.

- `vllm_omni/core/sched/omni_ar_scheduler.py`
  - Reserves extra output placeholders for acoustic inner-loop decoding.
  - Records `omni_acoustic_inner_loop_extra_slots` on scheduler output.
  - Corrects `num_output_placeholders` and `num_computed_tokens` when the inner
    loop generates fewer tokens than scheduled, for example due to early stop.

- `vllm_omni/worker/gpu_ar_model_runner.py`
  - Detects when the acoustic inner loop can run.
  - Runs K real sequential decode substeps inside one worker execution.
  - Updates input position, slot mapping, attention metadata, forward state,
    sampled token state, hidden state, and code predictor state per substep.
  - Returns multiple acoustic tokens in one `OmniModelRunnerOutput`.

- `examples/online_serving/qwen3_tts_nv_triton/benchmark_model.py`
  - Adds a small benchmark environment override for async scheduling control.

- `tests/core/sched/test_acoustic_inner_loop.py`
  - Adds unit coverage for slot reservation and correction helper behavior.

## Remote Model And Verification

The remote model used for the benchmark was:

```text
/home/admin/workspace/remote_workspace/pr3221_exp/Qwen3-TTS-12Hz-1.7B-Base-cv-view
```

The experiments were run on the remote GPU machine, primarily using GPU 2.

Verification completed:

- remote `pytest -q tests/core/sched/test_acoustic_inner_loop.py`: `4 passed`;
- remote `py_compile` for the changed Python files: passed;
- local `py_compile`: passed;
- local direct helper assertions: passed.

## Benchmark Results

### Inner-loop Prototype Benchmark

Benchmark configuration:

- `concurrency=1`;
- `num_requests=12`;
- `max_new_tokens=64`;
- total generated tokens: `674` for all compared runs;
- async scheduling disabled for the no-async baseline and inner-loop runs.

| Configuration | E2E mean | Token throughput | Relative to baseline |
|---|---:|---:|---:|
| baseline no-async | 521.50 ms | 107.62 tok/s | baseline |
| inner K=2 | 519.52 ms | 108.03 tok/s | +0.4% |
| inner K=3 | 507.70 ms | 110.54 tok/s | +2.7% |
| inner K=4 | 501.40 ms | 111.93 tok/s | +4.0% |
| inner K=6 | 491.26 ms | 114.24 tok/s | +6.1% |

The `ITL` metric should not be compared directly for these runs. With the inner
loop, one engine output can contain multiple acoustic tokens, so the benchmark's
original inter-token latency accounting no longer has the same meaning. The
more reliable metrics here are E2E latency, wall-clock duration, and token
throughput.

Two additional follow-up experiments were also tried:

1. Skipping the unused K-token prepare before entering the inner loop.

   Result for K=6: `491.32 ms`. This is effectively unchanged from `491.26 ms`.
   It shows that this extra prepare was not the main bottleneck.

2. Keeping sampled tokens on GPU and doing bookkeeping later.

   Result for K=6: `491.75 ms`. This also did not improve performance. The
   riskier code path was not kept.

## Current Conclusion

The direct acoustic inner loop is a correct direction and gives measurable
benefit, but the current runner-layer implementation only improves throughput by
about `6%` at K=6. This is below the rough `15-20%` gain one might expect if the
entire `~2ms` per-token gap were only scheduler/engine round-trip overhead.

The reason is that the current implementation removes only part of the outer
scheduler and engine round-trip. Inside the worker, each acoustic token still
goes through much of the generic single-token vLLM path:

- input preparation;
- batch execution and padding decision;
- slot mapping;
- attention metadata construction;
- model-specific preprocess;
- model forward;
- logits computation;
- vLLM sampler;
- bookkeeping;
- hidden and code predictor state update.

Therefore, the current bottleneck is no longer just the scheduler gap. A large
part of the remaining cost is the per-acoustic-token generic runner, sampler,
bookkeeping, and code predictor execution structure.

### Final Implementation Benchmark

A later remote validation round measured the final implementation at commit
`b82c03f8`, using `Qwen3-TTS-12Hz-1.7B-Base-cv-view` on an H20 GPU.

The final implementation includes the main structural changes proposed above:

- lightweight decode state mutation (AC2);
- greedy GPU-resident group-0 sampling (AC3);
- cached residual code prediction (AC4);
- graph-ready fixed-K loop (AC5).

Round 1 validation results:

| Configuration | E2E mean | Token throughput | Generated tokens |
|---|---:|---:|---:|
| K=1 baseline, no-async fallback | 784.32 ms | 75.19 tok/s | 708 |
| K=6 fast graph-ready, no-async | 660.98 ms | 89.21 tok/s | 708 |

Compared with the K=1 no-async fallback baseline, the K=6 fast graph-ready path
reduced E2E latency by `15.7%` and improved token throughput by `18.6%`.

This confirms that moving beyond the original runner-layer inner-loop prototype
is necessary for larger gains. The prototype benchmark remains useful as a
historical comparison: reserving multiple acoustic slots and looping in the
worker produced a measurable `+6.1%` throughput gain at K=6, while the final
fast path combines lighter state mutation, GPU-resident greedy sampling, cached
residual prediction, and fixed-K loop structure to reach the expected `15-20%`
range.

## Recommended Optimization Direction

The final goal is to make acoustic decoding steps as fast as possible. Based on
the current evidence, the next optimizations should be done in the following
order.

### 1. Move The Inner Loop Below The Generic Runner Path

The current structure is:

```text
engine step once
  Python runner loop K times
    generic vLLM single-token decode path
```

The target structure should be:

```text
engine step once
  Qwen3 NV acoustic fast loop K times
    minimal tensor update
    model forward
    sampler
    state update
```

This means adding a Qwen3 TTS NV specific decode fast path for the restricted
decode-only case. The fast path should avoid repeatedly running the full generic
runner preparation stack for every acoustic token.

The fast path should precompute or reuse as much as possible:

- decode-only batch shape;
- fixed request metadata;
- attention metadata template;
- slot mapping structure;
- padded shape decisions;
- model kwargs that do not change across substeps.

Inside each acoustic substep, it should update only the necessary state:

- current position;
- KV/cache write position;
- group-0 sampled token;
- `last_talker_hidden`;
- code predictor state;
- output token buffer.

This is the most direct continuation of the current prototype.

### 2. Move Group-0 Sampling Into The Model Or Graph Path

Residual groups `1..15` are already handled inside the NV talker/code predictor
path, but group-0 still goes through the vLLM sampler path. The profile still
shows sampler CPU cost and sampler kernels such as argmax or sort-like kernels.

For Qwen3 TTS acoustic decoding, especially under greedy or deterministic
sampling, a specialized group-0 sampler should be implemented:

- compute logits at the end of Qwen3 NV talker forward;
- run greedy `argmax` directly in the model-side path;
- keep the sampled group-0 token on GPU;
- feed it immediately into the next acoustic substep;
- copy all generated tokens back to CPU only once at the end of the K-step loop.

The first version should only support greedy sampling. That keeps the fast path
simple and measurable. Top-k, top-p, random sampling, penalties, and logprobs can
stay on the generic path until there is a clear need to support them.

This is likely more valuable than trying to fuse a few isolated small kernels,
because it removes a per-token framework boundary.

### 3. Make Code Predictor Incremental With KV Cache

This is probably the largest structural opportunity.

The residual code predictor still behaves like a small transformer that is
repeatedly run in a small prefill-like mode. This is expensive because the
sequence is tiny, the kernels are small, and the same context is repeatedly
processed.

The target design should make the code predictor a real incremental decoder:

- build code predictor cache during prefill;
- on each acoustic step, feed only the new group-0 or previous-code input;
- reuse cached state for residual group prediction;
- update cache incrementally;
- avoid repeated small transformer re-prefill.

This is a larger model-level change than the runner inner loop, but it is also
the most likely source of a bigger speedup.

### 4. Capture The Acoustic K-Step Loop As A CUDA Graph

Once the fast path is stable and most state stays GPU-resident, the next step is
to capture fixed K acoustic decoding loops as CUDA graphs, for example K=4,
K=6, or K=8.

The target graph would look like:

```text
for i in 0..K-1:
  talker decode one token
  sample group-0
  run residual code predictor
  update hidden/code state
  write KV/cache
```

Early stop makes graph capture harder because a graph cannot easily break out
dynamically. A practical solution is to run a fixed K steps with a valid-token
mask, then only return valid tokens to the scheduler. This is acceptable for an
initial performance fast path.

Graph capture should come after the sampler and code predictor path are moved
closer to the model/GPU side. Capturing the current generic Python loop would
not solve enough of the real overhead.

### 5. Reduce CPU Output And Hidden Payload Movement

The current runner still returns hidden states and multimodal/audio code payload
through CPU-facing output structures. The K-step loop already reduces how often
this happens, but there is more room:

- avoid returning hidden states when downstream does not need them;
- keep audio codes GPU-resident when the next stage can consume GPU tensors;
- defer CPU copies until the final API response boundary;
- avoid per-token hidden payload processing.

This may not dominate the talker-only benchmark, but it can matter for end-to-end
TTS pipelines.

## Suggested Execution Plan

The recommended path is:

1. Keep the current acoustic inner-loop prototype as the measured baseline fast
   path.

2. Implement a Qwen3 TTS NV decode-only fast path below the generic runner
   preparation stack.

3. Add greedy group-0 in-model sampling for the fast path.

4. Convert the residual code predictor to incremental decoding with cache.

5. Capture fixed K acoustic loops with CUDA graph after the state update path is
   mostly GPU-resident.

6. Only after these structural changes, revisit minor kernel fusion or small
   copy-elimination opportunities.

## Final Requirement

The requirement for this task is not merely to get a small benchmark improvement
or merge a narrow workaround. The requirement is to make Qwen3 TTS NV acoustic
decoding steps as fast as possible.

That means the final optimized path should minimize per-token framework
overhead, minimize CPU/GPU synchronization, keep acoustic decode state
GPU-resident, avoid repeatedly executing the generic vLLM single-token decode
path, and push group-0 sampling plus residual code prediction into a tightly
fused incremental loop.

In short: the end goal is to optimize the decoding steps as aggressively as
correctness allows, until the remaining cost is dominated by unavoidable model
math rather than scheduler, Python, sampler, bookkeeping, or repeated small
prefill overhead.

If remote GPU benchmarking is needed, use the existing chrome-remote-gpu workflow and provide credentials out of band.