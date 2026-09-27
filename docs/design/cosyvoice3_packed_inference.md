# CosyVoice3 packed inference

## Scope and dependencies

These experimental Hopper profiles optimize the existing CosyVoice3 integration.
They reuse the stage runtime, same-device replicas and private CUDA MPS from
PR #8184, and the HiFT/F0 weight-normalization folding from #7871. Neither is a new
model-specific optimization in this change. Shared-memory wakeup scoping and
streaming metadata preservation are separate runtime prerequisites.

Packed DiT is adapted from SGLang-Omni commit
`127f34b57446a3cb5588ec987da0771a3be675be` under Apache-2.0. The adapted attention
boundary uses vLLM FA3. This is an implementation adaptation, not an original
attention algorithm.

## Data flow

The processor caches reference conditioning in the existing bounded speaker
cache, using waveform content, shape, sample rate, dtype, model and speaker
backend. Text is processed independently. Returned tensors own their storage.
The talker assembles live request conditioning outside graph replay and keeps
prefill/decode row boundaries explicit. Random RAS uses batch tensor operations;
only rejected requests consume a second per-request RNG draw.

The codec groups requests by finalization state, packs valid Flow frames and
retains all ten Euler steps. Full responses use full-context attention and
batched HiFT. Streaming uses chunk-causal attention and request-owned GPU HiFT
mel/phase state. Sorting or grouping restores original request order before
returning results. Position noise grows in fixed blocks without modifying its
existing prefix or the global RNG state.

## Configuration and compatibility

`COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS=1` selects the full-response path;
`COSYVOICE3_PACKED_STREAMING=1` selects streaming. Both default to disabled and
require Hopper. Existing paths remain available on other GPU architectures.
The high-concurrency full-response YAML uses two AR and two codec replicas on
one GPU, each with capacity 32. It is a C64 option, not a universal default.
Chunked prefill is disabled in these profiles because prompt embedding layout
requires complete prompt boundaries.

Optional speaker TensorRT may coexist with packed Torch Flow. Packed mode must
not replace its Torch estimator with a TensorRT Flow estimator. Invalid optional
backend combinations should fail at initialization rather than silently select
a different model path.

An explicit HF override, `cosyvoice3_sampling_mode=standard`, selects ordinary
sampling for controlled comparisons. Default `ras` preserves existing behavior.
Standard mode keeps all 200 control logits separate, stops on every control ID,
and applies repetition penalties only to generated tokens. Set temperature, top-p, top-k and repetition penalty explicitly in the
stage-0 deployment `default_sampling_params`. Top-level Speech API fields do
not reliably override these values; verify the effective worker configuration. Identical
parameters do not imply identical RNG trajectories across inference engines.

## Quality, measurement and limitations

Packed attention, batched GEMMs and GPU FP64 F0 are not bitwise-equivalent to the
default path. Validate WER and complete-audio speaker similarity for the target
language and voices; no zero-quality-cost claim is made. The streaming and
full-response profiles are distinct configurations. First audio means the first
nonempty PCM payload, not response completion. Report payload size and playback
underrun separately, excluding single-payload responses from underrun summaries.

Measure Seed-TTS Eval with a frozen checkpoint, software environment, concurrency,
input order, effective sampling policy, warmup and independent restarts. Separate
reference-cache misses from hits; warmup does not imply all reference audio is
cached. Publish failed requests and output durations alongside throughput.
A cross-framework native-configuration comparison does not establish a matched
quality improvement or isolate the gain attributable to this PR.

Full Euler CUDA Graph experiments are intentionally excluded: component
correctness did not establish an end-to-end benefit for the selected profiles.
