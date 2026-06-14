# Pi05 Realtime Performance Notes

This file records the validated pi0.5 realtime paths and rejected experiments so
future changes do not repeat low-value optimization work.

## Benchmark Shape

- GPU: A100-SXM4-80GB
- Batch size: 1
- Views: 1
- Flow steps: 10
- Action chunk: 50
- Model dtype: bfloat16
- Wall-clock metric: no `return_timing`; timing mode adds synchronization and
  is only used for component attribution.

## Validated Latency

| Backend | Cache condition | p50 | Mean | Action diff |
| --- | --- | ---: | ---: | --- |
| `realtime_triton_prefix` | no cache, new image | 26.82 ms | 26.84 ms | baseline |
| `realtime_triton_prefix_image_cache` | exact image-embedding hit | 23.98 ms | 23.98 ms | max 0, mean 0 |
| `realtime_triton_prefix_emb_cache` | exact prefix-embedding hit | 23.61 ms | 23.68 ms | max 0, mean 0 |
| `realtime_triton_prefix_cache` | exact full prefix-KV hit | 13.78 ms | 13.80 ms | max 0, mean 0 |

The ordinary non-cache path is now roughly 26.8-27.0 ms. The <25 ms path is an
exact cache-hit optimization, not a claim that every new frame runs below 25 ms.

## Cache Semantics

The realtime cache layers are exact and conservative:

1. Full prefix-KV cache: reuses the final prefix encoder K/V only when the raw
   image, state, prompt, masks, positions, dtype, device, and backend key match.
2. Prefix-embedding cache: reuses built prefix embeddings/masks/positions but
   still reruns the prefix encoder and decoder.
3. Image-embedding cache: reuses per-camera SigLIP image embeddings but still
   reruns language embedding, prefix encoder, and decoder.

Misses fall back to the next lower cache layer or the full equivalent path. The
implementation must not reuse stale prefix K/V across changed observations,
because prefix self-attention lets image, state, and language tokens affect later
prefix K/V.

## Optimizations Kept

- bfloat16 inference for the deployment config.
- Realtime decoder with fixed-shape buffers and CUDA Graph replay.
- Realtime prefix encoder that writes prefix K/V directly into decoder buffers.
- Left-padding compaction for realtime metadata, turning non-contiguous language
  padding into a contiguous valid prefix slice.
- Exact cache fallback layers for prefix K/V, prefix embeddings, and image
  embeddings.

## Rejected Or Non-Promoted Experiments

| Experiment | Result | Reason |
| --- | --- | --- |
| Fused decoder attention | Slower in full graph | Split QK/softmax/AV keeps better GEMM efficiency for the current batch=1 shape. |
| AdaRMS/RMSNorm + QKV + RoPE fusion | Slower | Repeated row norm work across output tiles outweighed saved HBM traffic. |
| Simple Triton prefix MLP | Slower than torch/cuBLAS path | Prefix MLP uses large GEMMs; simple Triton matmul could not beat cuBLAS/Inductor. |
| Packed prefix QKV / packed prefix MLP toggles | No stable e2e win | Diff was zero, but p50 stayed around the no-cache baseline. |
| Decoder tile sweeps | Mostly neutral or slower | O-proj, QKV/RoPE, and FFN block changes did not produce stable e2e gains. |
| Scalar-mask softmax variants | Correct but slower or too small | Did not reduce no-timing wall latency. |
| GPU uint8 image preprocessing | Faster but nonzero drift | Action diff was nonzero; keep out of no-regression fast path. |
| cuBLASLt fixed-algo wrapper | Not pursued | Requires native extension/rebuild for modest expected gain. |
| One-shot fused FFN | Not pursued | A naive kernel would recompute gate/up activation for each down output tile. |

## Remaining Hotspots

Fresh nsys profiles after image-cache hit show the remaining non-cache hotspot
mix is dominated by prefix MLP and decoder FFN/QKV/O-proj kernels. Further
ordinary-path gains likely require a larger fused FFN implementation or a
different execution backend; small Python/Triton tile sweeps have not been
enough.
