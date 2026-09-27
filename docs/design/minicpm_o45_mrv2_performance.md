# MiniCPM-o 4.5 turn-mode MRv2 performance

The three-stage turn pipeline uses stage-level MRv2 contracts from #8184.
Thinker emits live latent metadata outside graph replay; Talker keeps codec
history and EOS decisions on device; Code2Wav batches compatible CFM/HiFT work.
The H200 profile enables block compilation, graph I/O reuse, tiled FP32
attention, channels-last convolutions and a 4/16/8 stage-capacity split.
Duplex continues to use its existing V1 path; turn-mode measurements do not
establish duplex performance.

The opt-in H200 YAML is `vllm_omni/deploy/minicpmo_4_5_turn_mrv2_h200.yaml`.
The generic MRv2 profile keeps 4/8/8 capacities. Talker 16 reserves 4 GiB KV
instead of 2 GiB. Warm representative shapes before serving; lazy compilation
and graph capture can create large first-use latency. Shared HiFT ISTFT code
also has consumers outside MiniCPM, so its regression coverage matters.

Historical same-worktree measurements (2026-09-26) used GSM8K-derived EN1088
short prompts, **not Seed-TTS Eval**. Tiled attention plus channels-last improved
C16 throughput 36.585 to 43.292 audio-s/s and C32 39.837 to 46.350, compared
with a baseline already using CFM compilation. One start per condition.
Separately, repeated Talker 8/16 tests at C32 yielded 45.833/45.677 audio-s/s,
first-audio P50 703.00/659.21 ms and P99 2032.90/1697.99 ms. Do not add these
percentages or call them the total gain over upstream. C16 pooled P99 improved
only 3.41%, with one round regressing. Quality spot checks used 128 English
prompts; unchanged WER does not establish zero quality cost. The split PR
needs its own correctness validation; these are historical performance results.
