# Higgs Audio v3 MRV2 validation

This comparison measures the existing vLLM-Omni Higgs V1 serving path against the
MRV2 deployment profile in this change. It compares complete serving
configurations, including scheduling, attention backend and CUDA graph coverage;
it does not attribute the improvement to one kernel.

## Frozen protocol

- Baseline: vLLM-Omni `817f5d0e4d12751020a2ed8d7886b821eb605467`.
- Candidate: the production Python changes in this PR, including the FULL graph
  capture query-length fix. Production file hashes accompany the result artifact.
- One H200, physical GPU 6; CPU affinity `48-95,144-191`. Both servers run
  sequentially on the same GPU. GPU ASR scoring runs separately from timing. Small CPU-only ASR diagnostics
  used cores 0-3, outside the serving CPU affinity.
- vLLM 0.30.0, PyTorch 2.13.0 / CUDA 13.0, Transformers 5.14.1, Triton 3.7.1.
- Same `bosonai/higgs-audio-v3-tts-4b` weights, full Seed-TTS EN1088, plain TTS.
- C32 and C64, 1088 measured requests per concurrency, 64 excluded warmup requests.
- Three independent starts per side, ordered before/after, after/before,
  before/after. All planned rounds are retained.
- Both AR and codec capacities are 64, prefix cache is disabled, MPS is enabled,
  and stage memory fractions are 0.60/0.25.
- Sampling: seed 42, temperature 1, top-p 0.95, top-k 50, repetition penalty 1,
  maximum 2048 tokens. The same seed does not imply identical generated audio
  across different batching and numerical execution paths.
- PCM16 mono 24 kHz, first 8 frames (320 ms / 15,360 bytes), following 25 frames,
  left context 8 frames, no right holdback. The actual first payload is checked.

`configs/higgs_v3_v1_matched.yaml` starts from the baseline's default profile and
normalizes only capacity, prefix caching, MPS and output chunk protocol. It keeps
V1 eager execution, synchronous scheduling and FLASHINFER with TRT-LLM attention
disabled. The candidate uses
`vllm_omni/deploy/higgs_multimodal_qwen3_mrv2_h200.yaml`: MRV2 asynchronous
scheduling, FLASH_ATTN, FULL backbone graphs, model-owned sampling/state and
exact codec graphs. These configuration differences are part of the measured
change, not hidden baseline overrides.

## Reproduction

The client bridge `seedtts_reference_client.py` uses the pinned Seed-TTS evaluator
from <https://github.com/sgl-project/sglang-omni>, commit
`7dc8909e78534a07400e86f781a969437ecc70e3`. This is benchmark provenance; the main
performance comparison is between two vLLM-Omni versions. Install that client's
benchmark dependencies in a separate environment. The bridge only adapts HTTP
stream/reference fields and records unrounded wall time after measurement; it
keeps dataset loading, scheduling, warmup and metrics in the reference client.
The recorded experiments used equivalent frozen transport and timer wrappers.

Create baseline and candidate worktrees and provide local model and dataset paths.
For each side and each independent start, run this server command, with `SOURCE`
set to that worktree and `CONFIG` set to its profile listed above:

```bash
export MODEL=/path/to/higgs-audio-v3-tts-4b
export META=/path/to/seedtts_testset/en/meta.lst
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1
export PYTHONPATH="$SOURCE"
taskset -c 48-95,144-191 python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --omni --host 127.0.0.1 --port 19202 --trust-remote-code \
  --deploy-config "$CONFIG" --stage-init-timeout 1800 --init-timeout 2400 \
  --allowed-local-media-path /path/to/seedtts_testset \
  --disable-log-stats --disable-uvicorn-access-log
```

Check GPU availability before starting. Choose a permitted GPU on your own
machine; the physical device selection happens before CUDA remapping. Use the
same device and CPU assignment for both sides. Wait for server readiness, then
run the client in its environment, giving every start a new output directory:

```bash
export SEEDTTS_CLIENT_ROOT=/path/to/pinned-reference-client
export BENCH_WALL_AUDIT_PATH="$OUTPUT_DIR-wall.jsonl"
python "$PR_WORKTREE/benchmarks/tts/seedtts_reference_client.py" \
  --generate-only --use-existing-server --skip-gpu-cleanup \
  --base-url http://127.0.0.1:19202 --model "$MODEL" --meta "$META" --lang en \
  --concurrencies 32,64 --repeats 1 --warmup 64 --max-new-tokens 2048 \
  --temperature 1 --top-p 0.95 --top-k 50 --repetition-penalty 1 --seed 42 \
  --output-dir "$OUTPUT_DIR" --disable-tqdm --stream \
  --initial-codec-chunk-frames 8 --no-ref-audio
```

Stop the owned server after both concurrency levels, then start the next arm.
Pool throughput as total emitted audio seconds divided by total measured wall
seconds, rather than averaging per-start throughput. Pool per-request latency
records for mean and percentiles. Preserve failures, audio durations and first
payload sizes. Report the individual-start range as well as pooled results.
These are repeated EN1088 runs, not an hour-long soak test.

## Clone capture regression

The faulty intermediate MRV2 candidate generated exactly 81.6 seconds for 14 of
128 C1 clone requests. Diagnostics reproduced three failures in the first 32 rows
and recorded `finish_reason=length`, `num_output_tokens=2048`. ASR long-audio
admission failures initially hid some of these bad outputs; serial rescoring of
the same WAVs exposed the generation defect.

For FULL mixed-prefill graphs, a dummy batch can divide 256 tokens among 64
requests while replay assigns over 200 tokens to one request. Capturing the
attention launch with the dummy per-request query length is unsafe. The fix uses
the full token bucket as the capture bound only when no explicit bound exists,
without mutating the original batch or changing runtime/explicit bounds.

The six new unit cases fail on the unfixed code for the three unconstrained
capture cases and all pass with the fix. The H200 regression suite passes 278
cases. Shadow prefill comparison improved worst observed hidden-state cosine
from 0.07891 to 0.999981. The shadow pass changes KV state and is used only for
numerical comparison, never for quality or performance claims.

The capture-fix candidate's full clone EN1088 C1/C32 validation preserved the 2048-token limit and all
optimization switches: 1088/1088 successful requests per level, no 81.6-second
outputs, maximum durations 8.92/8.88 seconds. All saved WAVs were scored with
Whisper large-v3 and the official Seed-TTS normalizer, with zero skipped samples.
WER was 1.552%/1.668%. On the same original 128 C1 rows, WER fell from 25.380% to
0.784%; the original 14 long outputs now lasted 2.32–8.92 seconds.

This does not establish perfect synthesis quality. A separate 4.32-second C32
sample had WER 100%; both the service evaluator and a Transformers CPU run of the
same Whisper large-v3 weights transcribed it as “Wow.” A different model,
OpenAI Whisper-small on CPU with beam size 5 and no target-text prompt, recovered
the complete correct sentence from the same WAV. This is an ASR disagreement,
not evidence that the synthesizer only said “Wow.” The original large-v3 score
remains in the aggregate; it was not replaced with the more favorable transcript.
Speaker similarity and human listening were not evaluated.

## Measured before / after

| Concurrency | Version | Audio-s/s | Requests/s | Mean first audio (ms) | P95 first audio (ms) | Mean completion (s) |
|---|---|---:|---:|---:|---:|---:|
| C32 | Main / V1 | 46.39 | 10.77 | 409.60 | 467.98 | 2.714 |
| C32 | PR / MRV2 | 197.88 | 45.44 | 118.08 | 135.48 | 0.692 |
| C64 | Main / V1 | 71.89 | 16.50 | 551.19 | 692.60 | 3.563 |
| C64 | PR / MRV2 | 277.86 | 63.71 | 168.89 | 253.40 | 0.973 |

- C32: throughput **+326.6%**, mean first audio **-71.2%**, P95 first audio **-71.0%**. Individual-start audio-s/s: main 50.21, 40.88, 49.41; PR 197.52, 197.35, 198.77.
- C64: throughput **+286.5%**, mean first audio **-69.4%**, P95 first audio **-63.4%**. Individual-start audio-s/s: main 75.54, 67.28, 73.41; PR 279.47, 273.87, 280.32.

All 13,056 timed requests succeeded. Baseline round 2 includes 61.64-second and
30.12-second outputs; all rounds and outliers are retained. The baseline has
substantially higher WER in this configuration, so this is an end-to-end
code/configuration comparison, not equal-quality kernel attribution.

| Concurrency | Main WER | PR WER | Paired difference 95% CI (percentage points) |
|---|---:|---:|---|
| C32 | 7.238% | 1.818% | [-6.174, -4.685] |
| C64 | 12.671% | 1.759% | [-11.974, -9.805] |

Quality covers the first prespecified paired start (4352 WAVs, zero skips), with
10000 paired bootstrap resamples by text. The final PR tree also repeated 128
clone rows at C1/C32: 256 successes, no long outputs or skipped scores, official
WER 0.784%/0.899%. All 278 regression tests passed on that tree. Detailed settings,
per-start measurements, source/weight hashes and telemetry are in
[evidence/higgs_v3_mrv2_h200_20260928.json](evidence/higgs_v3_mrv2_h200_20260928.json).

## Scope cleanup after the measured revision

The performance and audio-quality measurements above belong to commit
`0e679d523d419470bacd6046d9ceb6c4e151a589`. A subsequent cleanup removes the
experimental batched RNG and optional scheduler-level deferred-code cache;
both were disabled in the measured profile. Per-request torch RNG, sampler
capture parity checks, owned output snapshots and codec output copies remain.
The legacy and native whole-utterance adapters now share codebook de-delay,
invalid-code substitution and final-frame trimming. Streaming windows are
unchanged. These changes are covered by regression tests; the historical
measurements are not presented as a fresh benchmark of the cleanup revision.
