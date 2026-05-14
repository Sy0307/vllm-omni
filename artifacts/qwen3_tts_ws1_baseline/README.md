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
