# MRv2 performance evidence and current validation

Historical results below belong to archived experimental trees, not the current
PR head. They are retained as evidence for optimization decisions. Combining
numbers from different configurations would misstate the performance of the
shipped defaults.

## Historical Qwen3-TTS measurements

Qwen3-TTS-12Hz-1.7B-Base, vLLM 0.29, one H200 with both stages on the same
GPU, SeedTTS EN 1088, C64, three full warmup rounds followed by two measured
rounds. Output limit 1024, dataset seed 42, no per-request generation seed.
Audio throughput is total generated audio duration divided by wall time.

| Archived experiment | Audio-s/s | Included in current default? |
| --- | ---: | --- |
| B1 baseline, no MPS | 75.10 | Different archived source/configuration |
| copylite experiment, no MPS | 88.15 | Unsafe general shallow copy not imported; current PR retains owned snapshots |
| API cache capacity 1024 | 88.65 | Compact cache/capacity changes imported; this is not an isolated cache speedup claim |
| Compact cache + B2 | 103.52 | B2 remains opt-in; numerical/quality qualification incomplete |
| Compact cache + B1 + MPS | 117.05 | MPS is an external runtime configuration, not enabled by YAML |
| Compact cache + B2 + MPS, final repeat | 147.57 | Experimental configuration, not the default PR throughput |

Each row has two full measured rounds. The selected rounds completed 1088/1088
requests each and had no audio near the 1024-token cap. Other historical
cachefull/EDF/wide arms did contain 81.84-second outputs; their HTTP success
counts must not be treated as proof of complete or equivalent speech.
Historical B2/B4 numerical differences and incomplete ASR/SIM evaluation remain
reasons not to enable those configurations by default.

The archived TTS configuration uses a Talker prefill budget of 512, Code2Wav
max_num_seqs 10 and a 72-frame reference codec context. The standard YAML uses
32768 and 64 respectively and does not specify that reference-context cap.
Concurrency, warmup coverage, per-request seed, source tree and these settings
must be matched before diagnosing a regression against historical numbers.

## Historical Qwen3-Omni measurements

A separate archived vLLM 0.29 experiment used two H200 GPUs, native single-client
SeedTTS text input, full reference workload warmup, and two measured rounds of
2176 requests per concurrency. Stage max_num_seqs was 64.

| Concurrency | V1 audio-s/s | V2 audio-s/s | V1 TTFP p95 (ms) | V2 TTFP p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| C64 | 82.98 | 115.04 | 757.6 | 911.6 |
| C128 | 84.39 | 113.85 | 3757.9 | 2923.6 |

These are archived runner comparisons, not current-head results. C64 tail
latency worsened. Sampled ASR WER was 2.15% for V1 versus 3.08% for V2, with
additional speech still observed; quality equivalence was not established.
MTP prefix computation and output-copy/batchview experiments are not rolled
into these numbers or enabled as new defaults by this integration.

## Current source verification

Source commit: `b0ecf406a3a901e021e33faeb305c9a97e03c10d`.
All 1544 files under `vllm_omni` matched the remote source by SHA-256.
Environment: H200, vLLM 0.29.0, PyTorch 2.13.0+cu130, driver 580.173.02.

- Configuration tests: 365 passed, 3 skipped. All changed-file pre-commit checks,
  including mypy, passed for the default-profile update.
- Earlier integration regression: 2116 CPU and 100 CUDA tests passed. Broader
  historical mypy/DCO issues remain; these counts are not a merge-ready verdict.
- CustomVoice and VoiceDesign MRv2 checks: each 16/16 nonempty audio, 8 at C1
  and 8 at C4. These use `seed-tts-text` and reduced sequence/capture limits.
  An initial CustomVoice run used the wrong `seed-tts` dataset, whose Base-task
  fields overwrite the supplied task; that run returned 400 and was not counted.
- C16/64-request V1/V2 screening completed, but its finite workload and eight
  repeated probe warmups are insufficient to characterize steady-state C64
  performance. It is not used as the PR performance conclusion.

## C64 verification is paused on a generation failure

The current standard MRv2 YAML was tested with SeedTTS EN 1088, C64, output
limit 1024, no MPS and no per-request generation seed. The intended protocol
was three full warmups and two measured rounds, matching the historical load.

The first warmup completed 1087 requests and failed one: request index 48 did
not emit codec EOS within 1024 tokens. The server raised
`Qwen3TTSCodecLimitError`; the client observed an incomplete streaming transfer.
This was a normal short input sentence. No corresponding CUDA or runner crash
was found, but the cause of missing EOS has not been established.

The automation stopped immediately. Later warmups, measured rounds, and the
historical-configuration remeasurement did not run. The observed 74.68 audio-s/s
is a failed warmup, **not a valid current-head performance result**. Historical
cap-like outputs occurred at different request indices and do not establish
that this failure has the same cause. Further performance qualification is
paused pending investigation; no implementation change is justified yet.

## Evidence

[Machine-readable evidence](mrv2_performance_evidence_20260917.json) records
historical per-run aggregates, archived configuration text, report/provenance
hashes, software and dataset identity, and the current failed warmup's command,
configuration and result hash. Full logs and per-request data are retained in
the validation bundle `pr4582-defaults-20260917`.

Configuration tests require project test dependencies and vLLM 0.29, without
model weights:

```bash
python -m pytest tests/config/test_config_factory.py -q
python -m pytest tests/config/test_config_factory.py -m "core_model and cpu" --run-level core_model -q
```
