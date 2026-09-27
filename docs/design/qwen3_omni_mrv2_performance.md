# Qwen3-Omni MRv2 performance

This model adaptation builds on #8184's existing stage runtime and output
contracts. It adds Talker CPU metadata parsing, sampled-token embedding
handoff, an optional first-frame decoder, length-grouped Code2Wav graph
buckets, and opt-in incremental short-KV code prediction. The H200 profile
also enables conservative fused sampling and bounded cuDNN algorithm search.
The shared predictor keeps the upstream re-prefill helper and its default
sampling behavior; short KV is explicitly selected by the deployment profile.

Use `vllm_omni/deploy/qwen3_omni_moe_mrv2_h200.yaml` for the experimental H200
profile. Preserve a control with `code_predictor_fused_sampling=false` and
`codec_cudnn_benchmark=false` when C64 P99 is the primary requirement.
cuDNN search restores process settings on exit and has startup cost. Kernel
arithmetic is not bitwise-equivalent; the first-frame decoder uses additional
model memory. Full-duplex performance is not established by turn-mode tests.

Historical measurements (2026-09-26) use GSM8K-derived EN1088 short prompts,
**not Seed-TTS Eval**. Short KV improved repeated C64 throughput from 94.53 to
108.45 audio-s/s (+14.7%). Relative to an already-short-KV control, fused
sampling plus cuDNN search improved C32 4096-request throughput 83.469 to
92.382 and first-audio P50 135.82 to 128.10 ms. Two C64 starts had throughput
changes -2.59% and +10.08%, and P99 changes -3.80% and +23.36%; neither C64
throughput nor P99 has a reliable uniform gain. P50 improved 7.13% and 4.91%.
Do not add sequential percentages or present kernel speedups as service gains.
Long mathematical replies were retained. 128 English quality prompts showed
no clear WER/UTMOS regression, but are not a noninferiority test. The split PR
needs its own correctness validation; these are historical performance results.
