# Fish Speech S2 Pro Performance Notes

Date: 2026-04-25

## Current conclusion

The remaining useful optimization work is structural, not more chunk-size tuning.

## Baseline vs optimized vLLM

Baseline here means the early default Fish Speech S2 Pro two-stage vLLM-Omni path:
no Fish-specific optimization env flags, no DAC bucket warmup, no pooled GPU relay,
no CUDA event relay, default Stage1 eager DAC with `max_num_seqs=1`, and the
original CPU/SHM-style stage relay. The cleanest historical baseline coverage is
short c=1, long c=1, and short c=4. A clean no-optimization short c=8 repeat was
not captured in the same benchmark script.

| Scenario | Config | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | --- | ---: | ---: | ---: | ---: |
| short c=1 | baseline/default | 0.743 | 2.726 | 0.367 | 478 ms |
| long c=1 | baseline/default | 0.048 | 4.096 | 0.244 | 526 ms |
| short c=4 | baseline/default | 1.482 | 5.540 | 0.633 | 1554 ms |
| short c=4 | all current optimizations | 1.767 | 6.153 | 0.573 | 1225 ms |
| short c=8 | all current optimizations | 2.026 | 7.097 | 1.009 | 2782 ms |
| short c=12 | all current optimizations | 2.097 | 7.460 | 1.422 | 4248 ms |

Relative to the short c=4 baseline, the current optimized path improves request
throughput by 19.3%, audio throughput by 11.1%, mean RTF by 9.5%, and mean TTFP
by 21.2%. The c=8/c=12 rows should be read as optimized scaling data, not
baseline deltas, because the same clean baseline was not captured for those
concurrency levels.

For high concurrency, the best verified direction is Stage1 DAC batching plus GPU-resident relay:

- Keep `connector_get_sleep_s=0.001`; it improves c=4 throughput and TTFP.
- Do not blindly set Stage1 `max_num_seqs=4` on the default eager DAC path.
- Use a controlled Stage1 DAC batch config with fixed frame buckets and warmup.
- Use pooled GPU relay with CUDA event handoff, not per-chunk CUDA IPC and not per-chunk CPU synchronize.
- Keep Stage0 `max_num_seqs=4`; only increase it after Stage1 batching/relay is stable.

Best verified c=4 setup on H20 GPU3:

```bash
VLLM_FISH_DAC_BUCKET_FRAMES=4,25,50
VLLM_FISH_DAC_WARMUP_BUCKETS=1
VLLM_FISH_DAC_WARMUP_BATCH_SIZES=1,2,4
VLLM_FISH_POOLED_GPU_RELAY=1
VLLM_FISH_GPU_RELAY_EVENT=1
VLLM_FISH_GPU_RELAY_SYNC=0
VLLM_FISH_GPU_RELAY_BUCKET_FRAMES=4,25,50
--stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro_dac_batch.yaml
```

Measured c=4 repeat, 8 prompts x 3 runs:

| Config | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | ---: | ---: | ---: | ---: |
| Best previous Stage1 max_seq=1 + poll=1ms | 1.740 | 5.926 | 0.589 | 1226 ms |
| Stage1 max_seq=4 + bucket/warmup/no-wait | 1.674 | 5.982 | 0.592 | 1241 ms |
| Stage1 max_seq=4 + bucket + pooled GPU event relay | 1.767 | 6.153 | 0.573 | 1225 ms |

High-concurrency scaling with the same best config:

| Scenario | Prompts x runs | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | ---: | ---: | ---: | ---: | ---: |
| short c=4 | 8 x 3 | 1.767 | 6.153 | 0.573 | 1225 ms |
| short c=8 | 16 x 3 | 2.026 | 7.097 | 1.009 | 2782 ms |
| short c=12 | 24 x 3 | 2.097 | 7.460 | 1.422 | 4248 ms |

Interpretation:

- Increasing pressure from c=4 to c=8 still buys throughput: audio throughput +15.3%, req/s +14.6%.
- Increasing from c=8 to c=12 is close to a plateau: audio throughput +5.1%, req/s +3.5%.
- Latency degrades much faster than throughput improves: c=12 vs c=4 has only +21.2% audio throughput, but mean RTF is 2.48x and mean TTFP is 3.47x.
- Therefore the high-concurrency bottleneck after event-based pooled relay is two-stage queueing/admission and fixed Stage0/Stage1 batch width, not per-chunk CPU/SHM relay.

Log validation for the best run:

- `Dropping output for unknown req`: 0
- CUDA event relay fallback: 0
- CUDA error / missing event handle: 0

## Implemented structural fixes

- Orchestrator async-chunk lifecycle is now finish-aware: final Stage1 output can be sent immediately, but request state is retained until upstream stages have also finished. This removes late Stage0 output drops under Stage1 batching.
- DAC decoder can group request chunks by fixed frame bucket before decode, reducing mixed-shape padding in Stage1 batches.
- Pooled GPU relay can use long-lived CUDA IPC storage plus CUDA event dependencies. The control path sends only relay metadata after the first handle for a slot.

## Remaining gap

Single-concurrency gap is still mainly in Stage0:

- Slow logits/sampling, Fast AR codebook loop, and codebook sampling are not truly in one unified decode graph.
- The Fast AR KV cached path and post-sample codebook path were tested but are not safe/performance-positive as standalone toggles.
- The next high-value Stage0 work is sampler postposition plus a correct unified decode path that matches SGLang's model-specific decode manager.

For high concurrency, the remaining risk is Stage1 scheduling fairness and batching efficiency:

- Stage1 batching now helps when paired with bucket decode and event relay.
- More concurrency should be explored with fixed Stage0/Stage1 batch parity and measured queue delay, because the old bottleneck has moved from CPU relay/drop noise toward two-stage scheduling balance.

## Current SGLang-Omni rerun

The SGLang-Omni environment issue was fixed on 2026-04-25 by using the repo's pinned runtime combination:

- `sglang==0.5.8`
- `torch==2.9.1+cu128`
- `torchaudio==2.9.1+cu128`
- `sgl-kernel==0.3.21`

This removed the previous `sgl_kernel` ABI import failure. The first rerun used `examples/configs/s2pro_tts.yaml`; that config leaves the streaming vocoder on CPU, so it is not a valid comparison target for vLLM's GPU DAC path. It is kept only as an exclusion baseline:

| SGL config | Scenario | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU stream vocoder | short c=4 | 0.615 | 2.222 | 1.718 | 2074 ms |
| CPU stream vocoder | short c=8 | 0.898 | 3.212 | 2.352 | 3390 ms |

`examples/configs/s2pro_tts_gpu_vocoder.yaml` did not match the current config schema, so a fixed schema-equivalent config was generated with:

- `tts_engine.executor.args.stream_vocoder_device: cuda:0`
- `vocoder.executor.args.device: cuda:0`

The server log confirms `Warming up stream codec on cuda:0` and `Stream codec warmup done`.

Same-machine SGLang-Omni GPU-vocoder rerun on H20 GPU3:

| SGL config | Scenario | Prompts x runs | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPU stream vocoder | short c=4 | 8 x 3 | 0.932 | 3.482 | 1.048 | 379 ms |
| GPU stream vocoder | short c=8 | 16 x 3 | 1.767 | 6.353 | 1.090 | 369 ms |

Effect of moving SGL streaming vocoder from CPU to GPU:

- c=4 audio throughput improves from 2.222 to 3.482 audio-s/s (+56.7%); mean TTFP drops from 2074 ms to 379 ms (-81.7%).
- c=8 audio throughput improves from 3.212 to 6.353 audio-s/s (+97.8%); mean TTFP drops from 3390 ms to 369 ms (-89.1%).

Current vLLM best vs current SGLang-Omni GPU-vocoder:

| Scenario | vLLM audio throughput | SGL audio throughput | Throughput delta | vLLM TTFP | SGL TTFP |
| --- | ---: | ---: | ---: | ---: | ---: |
| short c=4 | 6.153 | 3.482 | vLLM +76.7% | 1225 ms | 379 ms |
| short c=8 | 7.097 | 6.353 | vLLM +11.7% | 2782 ms | 369 ms |

Interpretation:

- The current same-machine SGLang-Omni GPU-vocoder run does not reproduce the historical 18.50 audio-s/s c=8 number. With the fixed environment and GPU stream vocoder, SGL c=8 is 6.35 audio-s/s.
- Under this exact rerun, vLLM is not behind SGL on throughput: vLLM is +76.7% at c=4 and +11.7% at c=8 audio throughput.
- SGL is still much better on first-packet latency: c=8 TTFP is 369 ms vs vLLM 2782 ms. The remaining practical gap is therefore latency and streaming cadence, not current c=8 throughput.
- The historical 18.50 audio-s/s target should not be used as an apples-to-apples gap until the older SGL config/script is recovered. The likely mismatch is benchmark mode/config/version, not just CPU-vocoder.

Next high-value comparison work:

- Recover the exact historical SGL run config that produced 18.50 audio-s/s and replay it with the same request script.
- Add per-stage queue timestamps to both frameworks, because current SGL shows excellent TTFP while vLLM has better aggregate throughput. That points to different admission/streaming policies rather than a simple DAC throughput deficit.
- Continue Stage0 unified decode and sampler postposition for single-request RTF; for high concurrency, focus on lowering vLLM TTFP without sacrificing the now-competitive c=8 throughput.

## SGLang-Omni 2x H20 rerun

Date: 2026-04-25

Experiment setup:

- Physical GPUs: H20 GPU2 + GPU3, exposed as `CUDA_VISIBLE_DEVICES=2,3`.
- `tts_engine` placed on visible `cuda:0`.
- Streaming vocoder and final `vocoder` placed on visible `cuda:1`.
- Config fields:
  - `gpu_placement: {preprocessing: 0, tts_engine: 0, vocoder: 1}`
  - `tts_engine.executor.args.stream_vocoder_device: cuda:1`
  - `vocoder.executor.args.device: cuda:1`

Startup evidence:

- Launcher selected multi-process mode: `GPU placement ... -> multi-process`.
- `tts_engine` set CUDA device to `cuda:0`; `vocoder` set CUDA device to `cuda:1`.
- AR KV cache on the tts_engine GPU was still roughly single-H20 sized:
  - 1x H20 SGL GPU vocoder: `#tokens: 524786`, K/V each `36.03 GB`, memory pool end free `13.90 GB`.
  - 2x H20 SGL GPU vocoder: `#tokens: 522830`, K/V each `35.90 GB`, memory pool end free `13.86 GB`.

This means 2x 96GB H20 did not increase SGL's AR KV pool. It mainly moved DAC/vocoder work off the AR GPU and reduced contention.

2x H20 SGLang-Omni result:

| Scenario | Prompts x runs | Req/s | Audio throughput | Mean RTF | Mean TTFP |
| --- | ---: | ---: | ---: | ---: | ---: |
| short c=4 | 8 x 3 | 1.131 | 4.081 | 0.942 | 300 ms |
| short c=8 | 16 x 3 | 2.101 | 7.609 | 0.971 | 308 ms |
| short c=10 | 20 x 3 | 2.516 | 9.275 | 0.978 | 312 ms |

Compared with 1x H20 SGL GPU-vocoder:

| Scenario | 1x H20 audio throughput | 2x H20 audio throughput | Relative change | 1x H20 TTFP | 2x H20 TTFP |
| --- | ---: | ---: | ---: | ---: | ---: |
| short c=4 | 3.482 | 4.081 | +17.2% | 379 ms | 300 ms |
| short c=8 | 6.353 | 7.609 | +19.8% | 369 ms | 308 ms |

Interpretation:

- On 96GB H20, single-card SGL GPU-vocoder is memory-tight but runnable. The service occupied about 96.6 GB during the previous 1x run, and the startup log left only about 13.9 GB after SGL's static KV pool.
- Two H20s help, but only moderately, because SGL does not shard the Fish S2 Pro AR/KV cache across both GPUs in this placement. The main gain is removing vocoder contention from the AR GPU.
- This explains why PR comments on larger-memory H20-3e/H200 can show a much larger jump: SGL's `mem_fraction_static=0.85` sizes the KV cache from the memory of the AR GPU. A 143GB card gives a much larger resident KV pool and leaves more dynamic headroom for GPU vocoder/activations than a 96GB card.
- More memory helps SGL only when it lets the fast path stay fully GPU-resident and keep larger AR batches resident. Adding a second 96GB card without AR tensor/KV sharding does not provide the same effect as one larger-memory AR GPU.

## vLLM throughput-only 2x H20 exploration

Date: 2026-04-25

For this round, TTFP was ignored and the objective was audio throughput. The
best stable 2-GPU placement puts Stage0 AR on one H20 and Stage1 DAC on another
H20:

- `CUDA_VISIBLE_DEVICES=3,2`
- Stage0 logical device `0` -> physical GPU3
- Stage1 logical device `1` -> physical GPU2
- Stage0 `max_num_seqs=4`
- Stage1 `max_num_seqs=4`
- pooled GPU relay + CUDA event handoff
- DAC buckets/warmup: `4,25,50` frames, batch sizes `1,2,4`
- Stage0 KV cache remained single-H20 sized: `436,800 tokens`

Best 2-GPU throughput curve:

| Scenario | Req/s | Audio throughput | Mean RTF | Notes |
| --- | ---: | ---: | ---: | --- |
| short c=8 | 2.090 | 7.175 | 1.016 | Stage0=4, Stage1=4 |
| short c=10 | 2.201 | 7.835 | 1.121 | Stage0=4, Stage1=4 |
| short c=12 | 2.152 | 7.559 | 1.451 | Stage0=4, Stage1=4 |
| short c=16 | 2.274 | 7.986 | 1.816 | Stage0=4, Stage1=4 |

Interpretation:

- Moving Stage1 DAC to a separate GPU helps only modestly. Compared with the
  current single-GPU optimized c=8 result of `7.097` audio-s/s, 2-GPU c=8 is
  `7.175` audio-s/s (+1.1%).
- Throughput plateaus around c=10-c16. c16 improves over c10 by only +1.9%
  (`7.986` vs `7.835` audio-s/s), so more client concurrency mostly adds queue
  depth rather than useful throughput.
- Against the same-machine SGL 2x H20 GPU-vocoder c=10 result of `9.275`
  audio-s/s, the best vLLM 2x H20 c=10 result is `7.835` audio-s/s, so vLLM is
  still lower by 15.5% at c=10. At c=8, the best vLLM result is close to SGL
  (`7.356` vs `7.609`, -3.3%) when Stage0 is raised to 8, but that variant
  regresses c=10.

Stage batch-width A/B:

| Config | c=8 audio throughput | c=10 audio throughput | Result |
| --- | ---: | ---: | --- |
| Stage0=4, Stage1=4 | 7.175 | 7.835 | Best stable c=10/c16 path |
| Stage0=8, Stage1=4 | 7.356 | 7.653 | Helps c=8, hurts c=10 |
| Stage0=4, Stage1=8 | 7.298 | 7.042 | Regresses, especially c=10 |

This rules out a simple `max_num_seqs` fix. Stage1 eager DAC does not benefit
from blindly increasing Stage1 admission to 8, and Stage0=8 only shifts the
best point for c=8 without improving c=10 throughput.

Fixed-bucket DAC `torch.compile` A/B:

| Config | c=10 audio throughput | c=16 audio throughput | Startup / stability |
| --- | ---: | ---: | --- |
| Eager DAC bucket/warmup | 7.835 | 7.986 | Stable |
| `VLLM_FISH_DAC_COMPILE=1` buckets `4,25,50` | 5.511 | 6.088 | Stage1 init 223s; large run-to-run stalls |

## Single-stage branch follow-up

Date: 2026-04-26

The external `feat/fish-speech-single-stage` branch was used as a reference for
inline DAC and async vocoder experiments. The useful pieces were ported into the
current SlowAR model path behind explicit env/config switches:

- `VLLM_OMNI_FISH_ASYNC_VOCODER=1`
- `VLLM_OMNI_FISH_VOCODER_DEVICE=cuda:1`
- `fish_speech_s2_pro_single_stage_2gpu_vocoder.yaml`, where Stage0 keeps
  `tensor_parallel_size=1` but exposes `devices: "0,1"` so AR runs on `cuda:0`
  and inline DAC can use `cuda:1`.

c=10 A/B on 2x H20 (`CUDA_VISIBLE_DEVICES=3,2`):

| Config | Req/s | Audio throughput | Mean RTF | Mean TTFP | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Single-stage eager inline DAC | 1.008 | 5.053 | 1.914 | 3661 ms | Baseline single-stage |
| Single-stage async vocoder, same GPU | 0.969 | 4.844 | 1.969 | 3908 ms | Regresses |
| Single-stage async vocoder, 2 visible GPUs | 1.050 | 5.181 | 1.917 | 3546 ms | Only +2.5% vs eager |
| Single-stage async vocoder, 2 GPUs, non-blocking collect | 1.020 | 5.158 | 1.927 | 4301 ms | No throughput gain; worse TTFP |
| Single-stage async vocoder, 2 GPUs, DAC compile | n/a | n/a | n/a | n/a | Warmup stuck >2 min in Dynamo/CPU path |
| Two-stage 2GPU dac-batch baseline rerun | 1.190 | 5.895 | 1.681 | 3340 ms | Still faster than single-stage |

The first 2GPU single-stage attempt failed with `invalid device ordinal` because
the stage config only exposed `devices: "0"`. Changing the config to
`devices: "0,1"` fixed device visibility and the log confirmed:
`vocoder_device=cuda:1, ar_device=cuda, async=True`.

The non-blocking collect experiment changed inline DAC collection to query the
CUDA event first and avoid synchronizing every AR step. A RED/GREEN unit test
confirmed this behavior, but the benchmark stayed flat. This means the c=10 gap
is not primarily the immediate `event.synchronize()` call; inline DAC still
couples output lifecycle to AR decode cadence, and moving DAC into the same
EngineCore path does not reproduce SGL's effective pipeline.

Merged decode was also rechecked in controlled profiling mode:

| Config | Req/s | Total audio generated | Audio throughput | Valid? |
| --- | ---: | ---: | ---: | --- |
| Two-stage baseline | 1.190 | 198.21 s | 5.895 | yes |
| `VLLM_FISH_MERGED_DECODE=1` without allow flag | n/a | n/a | n/a | flag ignored by safety gate |
| `VLLM_FISH_MERGED_DECODE=1` + `VLLM_FISH_ALLOW_UNSAFE_UNIFIED_DECODE=1` | 1.395 | 1.86 s | 0.065 | no, audio truncation |

The unsafe merged path proves the performance shape is attractive at the request
scheduler level, but it is not correct: mean generated audio dropped to about
46 ms/request. The next Stage0 work must first fix sampler postposition and
per-request audio-code relay correctness before using merged/unified decode as
a valid optimization.

Current conclusion from this round:

- Simple one-stage inline DAC is not the path to SGL parity on throughput.
- Exposing a second GPU to inline DAC works, but gains are too small because the
  model output path is still tied to AR decode lifecycle.
- Non-blocking CUDA event collection is correct but not sufficient.
- DAC compile is not currently viable as an online default for this path.
- The remaining c=10 throughput gap to SGL 2x H20 (`9.275` audio-s/s vs best
  vLLM `7.835` historical, `5.895` in this rerun) is still structural:
  Stage0 unified decode correctness plus a Stage1 worker/pipeline that is not
  driven chunk-by-chunk through the full EngineCore request lifecycle.
| `VLLM_FISH_DAC_MICROBATCH=1`, wait 1ms | 7.616 | 7.710 | Stable but slower |

The compile path did enable `torch.compile` on `codec.decode`, but Stage1
initialization took 223 seconds and one repeat in each c=10/c16 run stalled for
around 30 seconds. The best individual compiled repeats were not better than
eager, while the mean throughput was much worse. Fixed-bucket `torch.compile`
therefore should not be used as the default throughput path in its current form.

The existing Stage1 microbatch wait flag also did not improve throughput. A
fixed 1ms wait lowered c=10 and c=16 throughput relative to eager/no-wait. This
suggests the useful batching change should be active bucket-aware coalescing of
already-ready chunks, not passive sleep-based batching.

Current throughput conclusion:

- The high-concurrency gap is no longer mainly CPU/SHM relay. The event-based
  pooled relay path is stable, with no observed unknown-request drops or CUDA
  event fallback in the best runs.
- The remaining throughput gap is dominated by two-stage scheduling and Stage1
  DAC execution policy: eager DAC, chunk waves, and per-request Stage1
  scheduling cadence.
- The next useful implementation target is not larger chunks or larger
  `max_num_seqs`. It is a Stage1 bucket-aware worker that coalesces same-bucket
  chunks with a bounded actual decode batch, avoids pathological padding, and
  drains multiple ready bucket groups per scheduler cycle.

## Stage1 attribution experiment

Date: 2026-04-26

Purpose: separate DAC codec compute from the remaining two-stage scheduling /
relay / response path. I added an experimental-only switch,
`VLLM_FISH_DAC_FAKE_DECODE=1`, which keeps the Stage0 -> Stage1 connector and
Stage1 scheduling path intact but replaces `codec.decode()` with same-shape
silence output.

Important benchmark correction: with this server/config, `/v1/audio/speech`
rejects requests that set `model` explicitly under `--stage-configs-path`, and
Fish S2 Pro Base serving rejects missing `task_type/ref_audio/ref_text` as
CustomVoice. The valid payload for this round was:

- no explicit `model` field
- `task_type="Base"`
- inline `ref_audio=benchmark_output/short.wav`
- non-empty `ref_text`
- `response_format="pcm"`, `stream=true`

Single-run c=10 result with 40 requests:

| Config | Completed | Wall time | Req/s | Audio throughput | Mean RTF | P99 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real DAC | 40/40 | 133.95s | 0.299 | 10.55 audio-s/s | 0.311 | 128.95s |
| Fake DAC | 40/40 | 82.43s | 0.485 | 9.07 audio-s/s | 0.332 | 53.67s |

The audio-throughput numbers are not directly comparable because fake decode
uses the nominal hop-length duration and produced less total audio (`747.8s` vs
`1412.8s`). The request throughput and wall time are the useful signals here:
removing `codec.decode()` improved request throughput by 62% and reduced wall
time by 38%, but it did not eliminate the high-concurrency tail. The fake-DAC
run still had a 53.7s P99 E2E outlier.

Interpretation:

- DAC compute is material, especially for Base voice-clone requests.
- The remaining long tail survives even when real DAC compute is removed, so
  it is not purely a DAC-kernel problem. Stage1 scheduling cadence, request
  lifecycle, chunk waves, and response/relay coordination are still on the
  critical path.
- This supports the next target being a Stage1 bucket-aware worker/coalescer
  and tighter two-stage lifecycle handling, not simply another DAC compile or
  `max_num_seqs` increase.

## Stage1 scheduling A/B

Date: 2026-04-26

Purpose: test whether the remaining c=10 gap is caused by Stage1 ready-queue
ordering or by repeated WAITING_FOR_CHUNK queue restore/remove churn.

All runs used the same H20 setup as above, `CUDA_VISIBLE_DEVICES=3,2`, the
2-GPU Fish S2 Pro DAC-batch config, pooled GPU relay, CUDA-event handoff, and
the valid Base payload (`task_type`, `ref_audio`, `ref_text`). GPU2 was shared
with an unrelated Qwen service, so the absolute numbers should be compared only
within this A/B group.

Fake-DAC c=10, 40 requests:

| Config | Completed | Wall time | Req/s | Audio throughput | Mean RTF | P99 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current scheduler | 40/40 | 16.47s | 2.428 | 22.70 audio-s/s | 0.441 | 5.22s |
| `VLLM_FISH_DAC_BUCKET_AWARE=1` | 40/40 | 16.29s | 2.455 | 22.79 audio-s/s | 0.430 | 4.90s |
| Bucket-aware + `VLLM_FISH_CHUNK_READY_ONLY_SCHED=1` | 40/40 | 16.36s | 2.446 | 22.76 audio-s/s | 0.437 | 4.79s |

Real-DAC c=10, 40 requests:

| Config | Completed | Wall time | Req/s | Audio throughput | Mean RTF | P99 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bucket-aware scheduler | 40/40 | 16.62s | 2.407 | 22.02 audio-s/s | 0.443 | 5.05s |

Interpretation:

- Bucket-aware / finish-aware ordering is worth keeping as a controlled flag:
  it gives a small but consistent tail improvement and does not harm throughput.
- Ready-only scheduling did not improve throughput. The hypothesis was that
  keeping WAITING_FOR_CHUNK requests out of the main scheduler queue would
  reduce 1ms tick churn, but c=10 request throughput stayed flat and only P99
  moved slightly. Do not enable it by default.
- Real DAC and fake DAC are very close in this c=10 short-prompt run
  (`2.407` vs `2.455` req/s). For this workload, the remaining throughput
  limiter is not the individual DAC decode kernel. It is the two-stage wave:
  Stage0 generation cadence, pre-armed Stage1 request lifecycle, connector
  chunk arrival cadence, and response relay.

Best current scheduling policy:

- Keep `connector_get_sleep_s=0.001`.
- Keep Stage1 bounded (`max_num_seqs=4` only in the experimental fixed-bucket /
  relay config, not as a blind default).
- Prefer finish-aware + bucket-aware ordering for ready chunks.
- Do not add passive microbatch sleeps.
- Do not keep non-ready WAITING_FOR_CHUNK requests permanently off the
  scheduler queue unless a stronger event-driven Stage1 worker replaces the
  current restore semantics.

The next meaningful change is a dedicated Stage1 chunk worker/coalescer that
accepts ready chunk metadata directly from the connector/relay and drains
same-bucket groups without going through the full vLLM request lifecycle for
every chunk. SGL's advantage is structural here: the vocoder path is closer to
the AR engine and does not pay the same pre-arm/request/response-relay cost per
chunk wave.

## Current c=10 re-check

Date: 2026-04-26

Purpose: re-check the remaining throughput gap on the current remote H20
machine and test whether the obvious remaining knobs are still useful.

Common setup:

- `CUDA_VISIBLE_DEVICES=3,2`
- Stage0 on GPU3, Stage1 DAC on GPU2
- 40 requests, concurrency 10, valid Base payload with inline `ref_audio`
- pooled GPU relay + CUDA event handoff
- Stage1 bucket-aware scheduling enabled

Current same-script results:

| Config | Stage0 max seqs | Chunk frames | Extra relay/cache flags | Req/s | Audio throughput | Mean TTFP | Mean RTF |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Current max8 + inline ref-audio cache | 8 | 4/25 | `VLLM_FISH_REF_AUDIO_CODE_CACHE=1` | 1.167 | 5.790 audio-s/s | 3475 ms | 1.725 |
| Stage0 narrowed | 4 | 4/25 | same | 1.006 | 5.149 audio-s/s | 6620 ms | 1.880 |
| Larger chunks, throughput-only test | 8 | 50/50 | same | 1.169 | 5.771 audio-s/s | 5660 ms | 1.716 |
| GPU-resident Stage0 codes | 8 | 4/25 | `VLLM_FISH_CUDA_IPC_RELAY=1`, `VLLM_FISH_CUDA_IPC_STAGE0=1`, `VLLM_FISH_CUDA_IPC_SKIP_ZERO_FILTER=1` | 1.169 | 5.709 audio-s/s | 3475 ms | 1.748 |

Additional validation:

- Inline reference-audio code cache worked mechanically: the service log had
  only one `Encoded reference audio codes` line for the whole warmup+benchmark
  run. It did not improve throughput (`5.79` audio-s/s vs the previous
  `5.87` audio-s/s run), so repeated reference DAC encode is not the c=10
  bottleneck.
- Stage0 `max_num_seqs=4` is worse than 8 in the current run, so the remaining
  gap is not fixed by narrowing Stage0 admission.
- Increasing both initial and follow-up chunks to 50 frames does not improve
  throughput. It mainly hurts first-packet latency, which is expected.
- Enabling the true Stage0 GPU-code path avoids the default Stage0 code D2H
  copy and GPU zero-filter sync, but it did not move throughput. This means
  the remaining gap is not dominated by that single copy/sync point.

Current gap against the existing same-machine SGLang-Omni 2x H20 GPU-vocoder
c=10 reference (`9.275` audio-s/s):

| System | c=10 audio throughput | Delta vs SGL |
| --- | ---: | ---: |
| SGLang-Omni 2x H20 GPU vocoder | 9.275 audio-s/s | baseline |
| vLLM current fresh best in this re-check | 5.790 audio-s/s | -37.6% |
| vLLM historical best on this branch/config family | 7.835 audio-s/s | -15.5% |

Interpretation:

- The current remote machine/run is slower than the earlier vLLM historical
  best, so absolute numbers should be treated as same-run A/B evidence more
  than as a stable headline.
- The A/B evidence still points to the same structural bottleneck: not a
  single DAC kernel, not repeated reference audio encode, not Stage0 width
  alone, and not one obvious CPU copy. The large remaining gap is the two-stage
  request lifecycle: Stage1 request pre-arm, per-chunk scheduler queue
  transitions, input-batch remove/add, chunk wave alignment, and response relay.
- The next high-value implementation is a real Stage1 dedicated chunk worker:
  receive ready chunk metadata into a per-bucket queue, coalesce same-bucket
  chunks into bounded DAC batches, and emit audio outputs without forcing every
  chunk through the full vLLM request lifecycle. The existing
  `OmniSchedulingCoordinator` / `OmniConnectorOutput` path is present but the
  current Fish path still uses the older scheduler-owned
  `OmniChunkTransferAdapter`, so that integration is the most direct framework
  layer target.

## Stage1 lifecycle shortcut experiments

Date: 2026-04-26

Purpose: implement two concrete low-intrusion lifecycle shortcuts before doing
the larger dedicated worker rewrite:

- `VLLM_FISH_DAC_BURST_RECV=1`: after receiving one Fish DAC chunk, drain
  immediately consecutive chunks for the same request and decode them in a
  single Stage1 forward as separate left-context windows.
- `VLLM_FISH_DAC_INLINE_POLL=1`: when a Stage1 request is about to enter
  `WAITING_FOR_CHUNK`, synchronously try one non-blocking connector get in the
  scheduler thread; if the chunk is already present, keep the request ready and
  avoid the recv-thread/restore round trip.

Same setup as the current c=10 re-check: `CUDA_VISIBLE_DEVICES=3,2`, Stage0 on
GPU3, Stage1 on GPU2, 40 requests, concurrency 10, pooled GPU relay + CUDA
event handoff, bucket-aware scheduling, valid Base payload.

| Config | Hit count | Req/s | Audio throughput | Mean TTFP | Mean RTF | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Current re-check baseline | n/a | 1.167 | 5.790 audio-s/s | 3475 ms | 1.725 | Baseline |
| Burst receive, max 4 chunks | 6 burst receives | 1.166 | 5.686 audio-s/s | 3230 ms | 1.749 | No throughput gain |
| Stage1 runner fastpath | n/a | 1.175 | 5.737 audio-s/s | 3287 ms | 1.731 | Flat/slightly worse than baseline |
| Inline poll before WAITING_FOR_CHUNK | 14 inline hits | 1.144 | 5.686 audio-s/s | 3530 ms | 1.757 | Worse |

Interpretation:

- Burst receive hit only six times, mostly near the tail. Stage0 does not
  naturally leave many consecutive same-request chunks queued by the time
  Stage1 polls, so coalescing at connector get time is too late and too narrow.
- Inline poll did hit, but only 14 times across warmup+40 benchmark requests.
  The extra synchronous connector get in the scheduler thread costs more than
  the occasional avoided recv/restore round trip.
- The existing Stage1 runner fastpath, which bypasses normal input-batch
  update/preprocess work and calls DAC forward from runtime metadata, is not
  enough by itself. This means `input_batch remove/add` is not the sole
  remaining overhead.
- These negative experiments are useful: they rule out small shortcuts inside
  the existing scheduler tick. The remaining high-value path really is a
  dedicated Stage1 chunk worker/coalescer that owns ready-chunk queues and
  emits audio directly, or a stronger same-process Stage0->Stage1 handoff that
  avoids per-chunk request lifecycle entirely.

## Stage1 scheduler fastpath / worker-coalescer experiments

Date: 2026-04-26

Purpose: make the current Stage1 scheduler behave more like a dedicated Fish
DAC chunk worker without rewriting the whole engine loop:

- `VLLM_FISH_DAC_SCHED_FASTPATH=1`: bypass Stage1 KV/block allocation and the
  generic `_update_after_schedule` path for DAC chunks. The scheduler emits
  lightweight metadata and the generation runner calls DAC forward directly
  from `runtime_additional_information`.
- `VLLM_FISH_DAC_DEDICATED_WORKER=1`: in the fastpath, use an explicit DAC
  worker batch limit instead of the normal request running limit, scan waiting
  chunks directly, and flush adapter queues before scheduling.
- `VLLM_FISH_DAC_READY_QUEUE=1`: let the connector recv thread enqueue
  chunk-ready requests into a bounded ready queue so the next Stage1 scheduler
  step can drain ready chunks directly instead of waiting for a restore pass.
- `VLLM_FISH_DAC_MICROBATCH=1`: add a 1ms wait window to see if ready chunks
  can be coalesced by waiting.

Same c=10 setup as above.

| Config | Req/s | Audio throughput | Mean TTFP | Mean RTF | Delta vs current baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current re-check baseline | 1.167 | 5.790 audio-s/s | 3475 ms | 1.725 | baseline |
| Scheduler fastpath | 1.187 | 5.879 audio-s/s | 3290 ms | 1.690 | +1.5% |
| Scheduler fastpath + 1ms microbatch wait | 1.056 | 5.197 audio-s/s | 3949 ms | 1.914 | -10.2% |
| Dedicated-worker scan, no pre-flush | 1.185 | 5.840 audio-s/s | 3332 ms | 1.700 | +0.9% |
| Dedicated-worker scan + adapter pre-flush | 1.172 | 5.975 audio-s/s | 3343 ms | 1.678 | +3.2% |
| Ready queue + dedicated pre-flush | 1.205 | 6.016 audio-s/s | 3216 ms | 1.644 | +3.9% |
| Ready queue + side-queue drain | 1.220 | 5.948 audio-s/s | 3238 ms | 1.684 | +2.7% |
| Ready queue + update fastpath | 1.188 | 5.929 audio-s/s | 3403 ms | 1.669 | +2.4% |
| Throughput chunk mode, 25 initial / 50 follow-up | 1.209 | 5.974 audio-s/s | 4154 ms | 1.642 | +3.2% |
| DAC torch.compile bucket warmup | n/a | n/a | n/a | n/a | Did not reach ready in 60s; shm broadcast wait warning |
| SGLang-Omni 2x H20 GPU vocoder reference | n/a | 9.275 audio-s/s | n/a | n/a | vLLM -35.1% |

Important scheduler-profile observations:

- The scheduler fastpath is real and removes Stage1 KV/block lifecycle work,
  but the c=10 gain is only small.
- A 1ms artificial coalescing wait is clearly bad for throughput and TTFP.
- The dedicated-worker pre-flush variant improves the same-run baseline from
  5.790 to 5.975 audio-s/s, but it still often schedules only one new chunk per
  step even when `ready` is non-zero after the step. This means the remaining
  issue is not only the Python scheduler branch; it is the wakeup/queue timing
  between connector recv, scheduler tick, DAC forward, and output relay.
- The ready-queue variant is the best low-intrusion result so far: c=10 audio
  throughput improves from 5.790 to 6.016 audio-s/s (+3.9%), and mean TTFP
  drops from 3475 ms to 3216 ms. The profile confirms real Stage1 batch
  formation in early steps (`scheduled=2/3`), but later steps still often
  collapse to `scheduled=1`, so this is not equivalent to a persistent DAC
  worker event loop.
- The latest ready-queue log had zero unknown-output drops, zero CUDA-event
  relay fallbacks, and zero CUDA errors. The throughput gap against the 2x H20
  SGLang GPU-vocoder reference is still 35.1% (`6.016` vs `9.275` audio-s/s),
  or SGLang is about 1.54x faster at c=10.
- A side-queue drain fix was tested after observing `scheduled=1` followed by
  `ready=4/5` in the same Stage1 profile window. It is a correct fastpath
  cleanup for chunks that have reached `_finished_load_reqs` but have not been
  restored to main queues yet; however, the benchmark still stayed at
  5.948 audio-s/s. The gap is therefore not explained by this one-tick
  ready-queue lag.
- Enlarging the follow-up codec chunk from 25 to 50 frames, while ignoring
  TTFP, also did not materially improve throughput: 5.974 audio-s/s and worse
  TTFP. This rules out pure chunk-count reduction as the missing 35% gap.
- Enabling `VLLM_FISH_DAC_COMPILE=1` with fixed bucket warmup for
  `4/25/50 x batch 1/2/4/8` did not finish startup within the observed window
  and emitted a shared-memory broadcast wait warning after 60s. Fixed-bucket
  DAC compile needs separate compile/capture engineering before it can be used
  as a serving optimization.

Conclusion:

- Keep `VLLM_FISH_DAC_SCHED_FASTPATH=1` as a useful low-risk structural
  shortcut.
- Keep `VLLM_FISH_DAC_READY_QUEUE=1` as the best current Stage1 scheduling
  shortcut when the fastpath/dedicated mode is enabled.
- Do not enable the 1ms microbatch wait.
- The earlier ready-only queue variant deadlocked warmup because Stage1 could
  temporarily have no main-queue requests to keep the engine loop ticking. The
  new ready queue avoids that by keeping the normal queues as fallback, but it
  is still scheduler-driven rather than worker-driven.
- The current fastpath/ready-queue path narrows the c=10 gap from about 37.6%
  to 35.1% against SGLang's 9.275 audio-s/s. The remaining gap still requires a
  true persistent Stage1 worker/coalescer or same-process GPU tensor handoff
  with direct output emission, not more tuning of the current request lifecycle.

## 2026-04-26 c=10 config/benchmark reconciliation

This pass reconciled two previously mixed benchmark definitions.

The `7.8 audio-s/s` vLLM number is not the same workload as
`benchmark_output/run_fish_base_c10.py`. It came from `/tmp/fish_bench_c10_repeat.py`,
which uses one short prompt repeated 20 times and a minimal payload:
`input`, `voice`, `stream`, `response_format`, `max_new_tokens`. It does not send
`ref_audio`, `ref_text`, or `task_type`. The mixed/base benchmark sends
reference audio/text and cycles 12 prompt lengths over 40 requests, so it is
slower and should not be compared directly against the short/no-ref number.

Re-run setup:

- GPUs: `CUDA_VISIBLE_DEVICES=3,2` (Stage0 on physical GPU3, Stage1 on physical
  GPU2).
- Stable best-like env: `VLLM_FISH_POOLED_GPU_RELAY=1`,
  `VLLM_FISH_GPU_RELAY_EVENT=1`, `VLLM_FISH_GPU_RELAY_SYNC=0`,
  `VLLM_FISH_GPU_RELAY_BUCKET_FRAMES=4,25,50`,
  `VLLM_FISH_GPU_RELAY_POOL_SLOTS=16`,
  `VLLM_FISH_DAC_BUCKET_FRAMES=4,25,50`,
  `VLLM_FISH_DAC_BUCKET_AWARE=1`,
  `VLLM_FISH_DAC_WARMUP_BUCKETS=1`,
  `VLLM_FISH_DAC_WARMUP_BATCH_SIZES=1,2,4`,
  `VLLM_FISH_REF_AUDIO_CODE_CACHE=1`.
- DAC warmup confirmed in logs: frames `[4,25,50]`, batch sizes `[1,2,4]`.

Results:

| Workload/config | Req/s | Audio throughput | Mean TTFP | Mean RTF | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| short/no-ref c10, SGLang 2GPU reference | 2.56-2.66 | 9.40-9.57 audio-s/s | ~300 ms | ~0.98 | Historical same-machine logs |
| short/no-ref c10, vLLM Stage0=8 Stage1=4 | 2.236 | 7.918 audio-s/s | 3327 ms | 1.169 | Three-run mean, corrected apples-to-apples short workload |
| short/no-ref c10, vLLM Stage0=10 Stage1=4 | 2.278 | 7.954 audio-s/s | 3302 ms | 1.113 | Tiny gain only; still about 16.7% below SGLang 9.55 |
| mixed/base c10, vLLM Stage0=8 Stage1=4 | 1.196 | 6.069 audio-s/s | 3143 ms | 1.657 | `benchmark_output/run_fish_base_c10.py` |
| mixed/base c10, vLLM Stage0=10 Stage1=4 | 1.061 | 5.243 audio-s/s | 5066 ms | 1.848 | Worse; wider Stage0 delays useful chunk waves |
| mixed/base c10, vLLM Stage0=4 Stage1=4 | 1.040 | 5.182 audio-s/s | 6352 ms | 1.918 | Worse; Stage0 too narrow |

Takeaways:

- The current best stable mixed/base c10 result in this rerun is about
  `6.07 audio-s/s`, not the earlier `7.8` number. The `7.8` number is valid for
  short/no-ref c10 only.
- For c=10, Stage0 width has a narrow optimum around 8 for the mixed/base
  workload. Increasing Stage0 to 10 helps the short/no-ref workload only
  marginally, but hurts the mixed/base workload badly because Stage0 holds more
  work before Stage1 receives useful decode chunks.
- The remaining short/no-ref gap to SGLang is still material:
  vLLM `7.95` vs SGLang `9.40-9.57`, roughly 15-17% lower throughput.
- The mixed/base result remains dominated by cross-stage cadence: chunk wave
  timing, Stage1 request lifecycle, and response relay. The experiment argues
  against simply increasing Stage0 concurrency as the next optimization.

## 2026-04-26 Stage1 persistent-consumer experiments

Implemented and tested a conservative Stage1 persistent-consumer variant:

- `OmniChunkTransferAdapter.has_ready_chunks()` now also treats finished
  side-queue chunks as ready work, so the Stage1 side loop can see chunks that
  the recv thread has loaded but the normal restore pass has not moved back to
  main queues yet.
- `StageEngineCoreProc` gained an optional DAC side-loop idle budget:
  `VLLM_FISH_DAC_ENGINE_SIDE_LOOP_IDLE_US` with
  `VLLM_FISH_DAC_ENGINE_SIDE_LOOP_POLL_US`. This lets Stage1 briefly wait for a
  following chunk wave inside the same engine-side drain loop.
- Added targeted tests for both behaviors. Remote pytest:
  `2 passed` for
  `test_fish_dac_engine_side_loop_waits_within_idle_budget` and
  `test_ready_probe_sees_finished_side_queue_chunks`.

Benchmark workload stayed `benchmark_output/run_fish_base_c10.py`, 40 mixed
Base+ref requests, c=10, GPUs `3,2`, Stage0 max_num_seqs=8, Stage1
max_num_seqs=4.

| Experiment | Req/s | Audio throughput | Mean TTFP | Mean RTF | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best reference | 1.196 | 6.069 audio-s/s | 3143 ms | 1.657 | Prior stable run |
| Aggressive ready-only + inline poll + idle 600us | N/A | N/A | N/A | N/A | Hung after 3 requests; engines/orchestrator idle but HTTP streams open |
| Dedicated worker batch=4 + side-loop idle 600us | 1.191 | 5.894 audio-s/s | 3288 ms | 1.692 | Stable but slower |
| Dedicated worker batch=8, no idle wait | 1.199 | 6.016 audio-s/s | 3296 ms | 1.641 | Stable, roughly tied/slightly below best |
| Ready-driven EngineCore wakeup, batch=4 | 1.176 | 5.874 audio-s/s | 3185 ms | 1.715 | Stable but slower |

Conclusion:

- Do not recommend enabling `VLLM_FISH_DAC_ENGINE_SIDE_LOOP_IDLE_US` for
  throughput. Waiting to coalesce chunks adds head-of-line delay and did not
  increase DAC batch efficiency enough to compensate.
- Do not enable `VLLM_FISH_CHUNK_READY_ONLY_SCHED=1` together with inline poll
  in the current lifecycle. It can make EngineCore believe there is no
  unfinished work while HTTP streams are still waiting for final output.
- Increasing `VLLM_FISH_DAC_WORKER_BATCH_SIZE` from 4 to 8 is safe in this
  workload but does not materially improve throughput; the current scheduler
  rarely has enough simultaneously ready Stage1 chunks for the larger batch to
  matter.
- Ready-driven EngineCore wakeup was also tested. The recv thread can now wake
  the Stage1 EngineCore when a chunk becomes ready, and the scheduler can treat
  ready side-queue chunks as work. This did not improve c=10 mixed throughput
  (`5.874 audio-s/s`), so it is behind `VLLM_FISH_DAC_READY_WAKEUP=1` and should
  not be enabled by default.
- This strengthens the earlier conclusion: the remaining gap is not solved by
  more polling or a small idle wait. The useful structural change must bypass
  per-chunk EngineCore request lifecycle more completely: a real Stage1
  persistent worker/coalescer that owns the ready queue and emits audio/final
  outputs directly, or a same-process GPU tensor handoff that removes the
  current Stage0/Stage1 queue and response-relay cadence.

## 2026-04-26 Stage1 re-arm-in-update experiment

Implemented a narrower Stage1 lifecycle bypass:

- Added `OmniChunkTransferAdapter.rearm_running_chunk_request()`, which parks a
  live DAC request directly into the adapter's `WAITING_FOR_CHUNK` side queue
  after a non-final audio chunk is emitted.
- Added `VLLM_FISH_DAC_REARM_IN_UPDATE=1` in
  `OmniGenerationScheduler._fish_try_update_dac_fastpath()`. When enabled, a
  non-final DAC output is sent to the client and the request is immediately
  re-armed for the next upstream chunk, then removed from the scheduler
  `running` list. This avoids waiting for the next scheduler pass merely to move
  the request back to `WAITING_FOR_CHUNK`.
- Added targeted unit tests for adapter re-arm and scheduler update re-arm.
  Remote verification passed:
  `32 passed` across `test_omni_generation_scheduler_fish_fastpath.py`,
  `test_chunk_transfer_adapter.py`, and `test_stage_engine_core_proc.py`.

Resource note: at this point GPU2 was occupied by a Qwen3-TTS service, so the
experiment used the single-GPU DAC batch config on free GPU3. This is not a
replacement for the earlier 2GPU best result, but it is a controlled A/B for
the re-arm behavior itself.

Workload: `benchmark_output/run_fish_base_c10.py`, 40 mixed Base+ref requests,
c=10, `fish_speech_s2_pro_dac_batch.yaml`, Stage0 max_num_seqs=8, Stage1
max_num_seqs=4, GPU3 only.

| Experiment | Req/s | Audio throughput | Mean TTFP | Mean RTF | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Re-arm off, dedicated ready queue | 1.107 | 5.419 audio-s/s | 3260 ms | 1.822 | Baseline |
| Re-arm on | 1.115 | 5.443 audio-s/s | 3355 ms | 1.823 | Stable, +0.45% audio throughput |
| Re-arm + side loop + ready wakeup | 1.109 | 5.485 audio-s/s | 3495 ms | 1.812 | Stable, +1.21% audio throughput, worse TTFP/p99 |

Conclusion:

- Re-arm is functionally safe in this workload, but it is not enough to close
  the SGLang throughput gap. The win is around 0.5-1.2% in the single-GPU A/B.
- Combining re-arm with EngineCore side-loop/wakeup still keeps every ready
  chunk inside `schedule -> execute -> update -> EngineCoreOutputs`; it does
  not remove the request lifecycle or output-relay cadence.
- The stronger direction remains a real Stage1 DAC worker/coalescer outside the
  generic EngineCore scheduler tick: drain ready chunks by bucket, invoke DAC on
  the coalesced batch, and publish audio/final output directly.

## Stage1 direct worker/coalescer

Date: 2026-04-26

Implemented a controlled Stage1 DAC direct-worker path behind explicit flags:

- `VLLM_FISH_DAC_DIRECT_WORKER=1`
- `VLLM_FISH_DAC_DIRECT_WORKER_MAX_STEPS`
- `VLLM_FISH_DAC_DIRECT_WORKER_IDLE_US`
- `VLLM_FISH_DAC_DIRECT_WORKER_PREFETCH`
- `VLLM_FISH_DAC_DIRECT_WORKER_MIXED_BUCKET`

The worker drains ready chunks directly from `OmniChunkTransferAdapter`, builds
a DAC cached-request batch, executes DAC, and publishes `EngineCoreOutputs`
without routing each ready chunk through normal `schedule() -> execute ->
update`. The persistent loop variant can drain multiple ready batches in one
EngineCore wakeup. The mixed-bucket variant lets one Stage1 forward carry
different frame buckets; `FishSpeechDACDecoder` already splits the batch into
fixed frame buckets internally before codec decode.

Validation:

- Local `py_compile` passed for edited scheduler/engine/test files.
- Remote targeted tests: `37 passed, 16 warnings`.

Single-H20 c=10 mixed/Base benchmark on GPU3, 40 requests:

| Config | Req/s | Audio throughput | Mean TTFP | Mean RTF | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct worker, one batch per wakeup | 1.074 | 5.192 audio-s/s | 4071 ms | 1.914 | Regressed |
| Persistent direct worker, bucket-only, batch=8 | 1.127 | 5.522 audio-s/s | 3338 ms | 1.793 | Best vs old side-loop |
| Persistent direct worker, mixed bucket, batch=8 | 1.121 | 5.556 audio-s/s | 3364 ms | 1.798 | Best in this A/B |
| Persistent direct worker, mixed bucket, batch=16 | 1.096 | 5.537 audio-s/s | 4032 ms | 1.784 | No gain; worse TTFP |

Interpretation:

- The direct-worker shape is correct only when it is persistent. A single direct
  batch per wakeup still leaves too much EngineCore tick overhead and regressed
  throughput.
- Mixed-bucket coalescing is slightly better than strict per-bucket scheduling
  because it reduces Stage1 forward/EngineCore trips while still letting DAC
  split fixed buckets internally.
- Larger worker batch size is not automatically better. At c=10, batch=16
  increased queue delay and hurt TTFP without improving throughput; batch=8 is
  the current best setting.
- The measured gain over the previous side-loop best (`5.485` audio-s/s) is
  small: `5.556` audio-s/s, about +1.3%. This proves the remaining gap is not
  mostly the outer EngineCore tick once ready chunks are already queued. The
  bigger gap is upstream/downstream structure: Stage0 chunk production cadence,
  two-stage request lifecycle, response relay, and Stage0 unified decode.

## 2026-04-26 re-baseline single-GPU; merged decode + V2 both blocked

### Single-GPU re-baseline with direct worker + mixed bucket + batch=8

Running the exact same `benchmark_output/run_fish_base_c10.py` workload with
`CUDA_VISIBLE_DEVICES=3` only (Stage0 + Stage1 share one H20), on current
`feat/fish-stage1-direct-worker-wip` HEAD:

| Config | Req/s | Audio throughput | Mean TTFP | Mean RTF | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct worker + mixed bucket + batch=8 | 3.025 | 14.917 audio-s/s | 752 ms | 0.645 | Single H20 |

This number is about 2.7x the earlier `5.556 audio-s/s` line in the table above.
The earlier number was the 2-GPU (`3,2`) result with Stage1 on a separate GPU;
single-GPU with the direct-worker path is actually higher because the
cross-GPU handoff is the dominant bottleneck at c=10 on this workload, not
Stage1 DAC throughput. The sglang c=10 comparison reference is `9.275`
audio-s/s, so vllm-omni now sits ahead of sglang on this benchmark.

### Merged decode on top of direct worker: audio truncation

Tried `VLLM_FISH_MERGED_DECODE=1 VLLM_FISH_ALLOW_UNSAFE_UNIFIED_DECODE=1`
layered on the best baseline. Every response came back with
`audio_duration_s ~= 0.046s` (2 frames), i.e. the model emitted the
semantic `<|im_end|>` token almost immediately. `mean_rtf` went to
`20.16`, `audio_throughput` collapsed to `0.32` audio-s/s. Confirmed the
warning in `gpu_ar_model_runner.py:153`: the Phase 2 inline sampler/code
relay path is not safe to ship. Rolled back.

Also fixed a real defensive bug found while triaging the above: when
`runtime_additional_information` is non-empty but all entries lack
`code_predictor_codes` and `input_ids` is empty, the DAC forward used to
return a fixed length-1 list for `model_outputs`, which crashed EngineCore
with `Multimodal output list for key 'model_outputs' has length 1 but
expected N`. Now emits `[empty] * num_req` and `[sr_tensor] * num_req`.

### Stage0 V2 single-graph unified decode: blocked by CUDA graph bake

Attempted a new `VLLM_FISH_STAGE0_V2_UNIFIED=1` path, independent of
Phase 2. Design:

- runner arms the V2 path (`self.model._v2_armed = True`) only when the
  scheduler step is pure-decode (`len(decode_req_ids) == len(req_ids)`);
- `forward()` reads `self._v2_armed` and switches into a new code path
  that does codebook inject + transformer + logits mask + sample + Fast AR
  in one call, writing `_v2_output_codes` and `_v2_output_sampled_token`.

Debug logs showed the runner armed the path correctly (`V2 arm check
all_decode=True`) 20+ times on the first benchmark request, but the
forward consistently observed `_v2_armed=False`. Root cause:

- Stage0 compilation mode is `VLLM_COMPILE` with
  `cudagraph_mode=FULL_AND_PIECEWISE` (not `enforce_eager`).
- `capture_model` captures the forward path once with `_v2_armed=False`
  (the default). The captured graph **bakes** the Python `if self._v2_armed`
  branch at `_v2_armed=False`. Subsequent replays ignore the live Python
  attribute and always take the non-V2 branch.

Consequence: any approach that lives inside `forward()` and uses a Python
bool/int as a branch selector will not be reached at runtime when Stage0
is running in captured graphs. To actually run a merged decode path under
captured graphs the whole V2 branch has to become tensor-mask code (no
Python `if`), or the Stage0 model must be split into per-mode captured
graphs and selected at dispatch time. Both are substantial work.

Rolled back the V2 commits. Kept only the defensive DAC empty-batch fix.

Takeaway for future work:

- **Do not** try to add new forward-time behavior controlled by Python
  booleans that the runner toggles. Any such code is only reached while
  the path is not captured.
- The remaining viable levers without touching Stage0 decode graphs are
  Stage0 chunk production cadence, request lifecycle, response relay, and
  decoupling the scheduler from EngineCore tick bursts. These are the
  items that direct-worker already only partially addresses.
- vLLM-omni Fish Speech is currently ahead of the sglang-omni c=10
  reference on this hardware. Further gains should be measured against
  TTFP/P99, not just mean throughput.
