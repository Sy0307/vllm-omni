# MiniCPM-o 4.5 turn-mode MRv2 performance

The three-stage turn pipeline uses stage-level MRv2 contracts from #8184.
Thinker emits live latent metadata outside graph replay; Talker keeps codec
history and EOS decisions on device; Code2Wav batches compatible CFM/HiFT work.
The H200 profile enables block compilation, graph I/O reuse, tiled FP32
attention, channels-last convolutions and a 4/16/8 stage-capacity split.
Duplex continues to use its existing V1 path; turn-mode measurements do not
establish duplex performance.
The separate opt-in `minicpmo_4_5_duplex_h200.yaml` profile enables native V1
graph I/O reuse, retained graph caches and TF32 Flow GEMMs. TF32 changes
rounding; validate quality for the target workload. Block compilation stays off.

The opt-in H200 YAML is `vllm_omni/deploy/minicpmo_4_5_turn_mrv2_h200.yaml`.
The generic MRv2 profile keeps 4/8/8 capacities. Talker 16 reserves 4 GiB KV
instead of 2 GiB. Warm representative shapes before serving; lazy compilation
and graph capture can create large first-use latency. Shared HiFT ISTFT code
also has consumers outside MiniCPM, so its regression coverage matters.

The existing V1 turn profile retains its original capacities and scheduling defaults.
