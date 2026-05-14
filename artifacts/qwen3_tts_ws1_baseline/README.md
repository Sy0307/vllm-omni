# Qwen3-TTS WS1 Baseline

Baseline scope:
- Config fixed at Stage0 max_num_seqs=64 and Stage1 max_num_seqs=10.
- Existing initial_codec_chunk_frames=1 is kept.
- Existing Code2Wav exact-length batching is kept.
- No WS1 Stage0 slot runner is enabled.

Primary workload:
- Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
- Task: voice_clone
- Concurrency: 64
- Num prompts: 256 for stable run, 128 for quick A/B
- Warmups: 2, excluded from steady-state SLA

Metrics:
- median / p99 TTFT
- median / p99 audio TTFP
- median / p99 E2EL
- median / p99 audio RTF
- audio throughput
- request throughput
- failed request count

Validated WS1 result:
- Change: batched Stage0 Base voice_clone preprocessing for tokenizer ids,
  ref-audio normalization, and same-sample-rate ref_code encoding.
- Remote result:
  `/home/admin/workspace/remote_workspace/qwen3_stage0_slot_runner_ab_20260514_1840/results_20260514_190000/ab_summary.json`
- Workload: 2x H20, GPU pair `0,1`, concurrency 64, prompts 256,
  warmups 2, Stage0 `max_num_seqs=64`, Stage1 `max_num_seqs=10`.
- Correctness smoke: new and old both completed 256 requests, failed requests
  0 in the benchmark log, with nonzero audio output (`1078.00s` new,
  `1076.96s` old).

| Metric | New | Old | Delta |
| --- | ---: | ---: | ---: |
| Audio throughput | 28.8289 | 25.5383 | +12.89% |
| Request throughput | 6.8462 | 6.0706 | +12.78% |
| Median audio RTF | 2.1888 | 2.4626 | -11.12% |
| Median audio TTFP ms | 1573.48 | 1634.63 | -3.74% |
| P99 audio TTFP ms | 5032.01 | 7319.04 | -31.25% |
| Median E2EL ms | 8746.86 | 9949.19 | -12.08% |
