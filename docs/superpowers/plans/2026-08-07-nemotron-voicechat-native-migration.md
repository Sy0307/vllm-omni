# Nemotron VoiceChat Native Full-Duplex Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task by task. Every
> behavior change follows RED/GREEN verification, and GPU claims require a
> pinned H20 run.

**Goal:** Replace the NVIDIA Speech runtime prototype with a deployable
vLLM-Omni-native Nemotron VoiceChat Full-Duplex pipeline built on PR #5842.

**Architecture:** Keep the experimental Full-Duplex Session, fence,
idempotent append, typed event, and resource-lease contracts in vLLM-Omni.
Move the model execution path to #5842's native Thinker, Talker, and Code2Wav
Stages so production serving never imports or starts NVIDIA Speech. The Speech
checkout remains reference-only for weight provenance, algorithm comparison,
and parity fixtures.

**Tech Stack:** Python 3.12, vLLM V1, vLLM-Omni, PyTorch, FastConformer,
NemotronH, EAR-TTS, RVQ-VAE, pytest, OpenAI Realtime WebSocket, NVIDIA H20.

---

## 1. Scope and pinned base

The model base is vLLM-Omni PR #5842:

- PR: `vllm-project/vllm-omni#5842`
- Reviewed head: `13a4bf5fa120d27ce650ce1715b5d20b45692647`
- Native Stage 0: `NemotronVoiceChatThinkerForConditionalGeneration`
- Native Stage 1: `NemotronVoiceChatTalkerForConditionalGeneration`
- Native Stage 2: `NemotronVoiceChatCode2Wav`
- Offline deploy: `vllm_omni/deploy/nemotron_labs_voicechat.yaml`

The Full-Duplex architecture work is developed independently so reviewers can
separate shared runtime changes from the 12,000-line model contribution. The
native model follow-up is stacked on #5842 until it merges, then rebased onto
`main` without copying #5842's diff into the Full-Duplex PR.

### Required final property

The final server process and all Stage workers must run from the vLLM-Omni
repository and model checkpoint only. They must not import:

- `nemotron_voicechat_vllm_omni` from the Speech checkout;
- `nemo.collections.speechlm2.inference`;
- a nested Speech runtime pool;
- a second `AsyncOmniEngine` constructed by a serving adapter.

The `nemo_vendored` package in #5842 is allowed: it is dependency-stripped,
Apache-2.0 code stored and imported inside vLLM-Omni. It is not an external
runtime dependency.

## 2. One branch, nine reviewable modules

All work is preserved on one integration branch:
`dev/full_duplex_enhancement`. The branch may use multiple signed-off commits,
but the nine module boundaries below must remain visible in commit messages,
tests, and the final change description. Pushing this branch does not imply
that every module is complete or that a pull request has been opened.

| Module | Boundary | Current branch state |
| --- | --- | --- |
| 1. Processor and payload failure semantics | Surface Stage processor exceptions; never convert failure into an empty payload or finish marker | Implemented in committed history; native Nemotron regression remains planned |
| 2. Typed model events and ledger | Immutable listen/speak/transcript/function events, epoch filtering, duplicate/gap handling | Core implementation and focused tests present |
| 3. Input/output identity and ordering | Separate `input_seq` from `output_id/output_seq`; remove turn identity from transport | Core and MiniCPM migration present |
| 4. Realtime event projection | Map accepted typed events to response, audio, transcript, and function-call wire records | Implementation and boundary tests present; native Nemotron projection planned |
| 5. Model-neutral input controller and runtime data plane | Remove MiniCPM fields from generic serving; let each model own append/commit/rollback policy | Generic projection hooks are in progress |
| 6. Existing model migrations | Keep MiniCPM and PersonaPlex behavior while moving them onto the model-neutral contracts | MiniCPM and PersonaPlex changes and tests are present; final regression pass remains |
| 7. Resource lease and execution profile | Atomic reserve/rollback/release plus optional prewarm and step-latency budget | Implementation and saga tests are in the working checkpoint |
| 8. #5842-native Nemotron three-Stage streaming | Incremental Thinker, EAR-TTS Talker, and cached Code2Wav inside vLLM-Omni | Detailed implementation plan only; no native code or native E2E claim yet |
| 9. Acceptance, Demo, and performance cleanup | Single-Session Engine E2E, Realtime E2E, two-Session follow-up, latency and cleanup evidence | Speech-backed protocol evidence exists; #5842-native acceptance remains planned |

Recommended commit grouping on this branch follows the same order. A module
may use a RED-test commit followed by a GREEN implementation commit; unrelated
modules must not be combined merely to reduce commit count.

## 3. What the current prototype proves

The external prototype remains valid evidence for model-neutral behavior:

- one upstream append is exactly 80 ms / 1280 float32 samples at 16 kHz;
- append reservation supports commit and rollback;
- `operation_id` replay is idempotent;
- `input_seq` and output correlation are monotonic;
- epoch changes reject stale output;
- transcript and function channels require typed, independently sequenced
  events;
- a Session-scoped lease can own model GPU state;
- Realtime `session.updated`, cancellation, response completion, and Demo
  protocol shapes are known.

It does not prove the final native model path. Previous H20 results used
Speech-owned perception, EAR-TTS, and codec execution and therefore cannot be
reported as #5842 native three-Stage E2E.

## 4. Ownership after migration

| State or operation | Final owner |
| --- | --- |
| Session/incarnation/epoch | Full-Duplex control plane |
| operation id and input sequence | Full-Duplex control plane |
| 80 ms PCM validation/reservation | Nemotron serving adapter |
| waveform/mel carry and perception cache | native Thinker request state |
| vLLM KV and Mamba state | vLLM scheduler/worker |
| previous text/function token | native Thinker request state |
| EAR-TTS KV and previous code | native Talker request state |
| RVQ-VAE causal convolution cache | native Code2Wav request state |
| transcript/function parsing | Nemotron event projector |
| output id and output sequence | typed Full-Duplex event ledger |
| OpenAI response/item/call ids | Realtime serving layer |

Opaque model state is never serialized into a Stage payload. Cross-Stage
payloads contain only immutable frame deltas and correlation metadata.

## 5. Target data flow

```text
Realtime input_audio_buffer.append (1280 PCM samples)
  -> Nemotron input reservation
  -> append/resume the same Stage-0 request
  -> native Thinker computes one stable acoustic frame
  -> NemotronH executes one frame-locked step
  -> text token + function token + source_input_seq
  -> resume the same Stage-1 Talker request
  -> EAR-TTS emits one [1, 31] codec-code delta
  -> resume the same Stage-2 Code2Wav request
  -> RVQ-VAE emits one cached 22.05 kHz PCM delta
  -> typed transcript/function/assistant events
  -> OpenAI Realtime projection
```

The ordinary vLLM scheduler remains the execution scheduler. The control plane
submits one resumable unit per accepted input frame; no standalone
`OmniDuplexScheduler` is introduced in this migration.

## 6. Implementation tasks

### Task 1: Stabilize the #5842 offline base

**Files:**

- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_thinker.py`
- Modify: `tests/model_executor/models/test_nemotron_voicechat_shapes.py`
- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_code2wav.py`
- Modify: `tests/model_executor/stage_input_processors/test_nemotron_voicechat.py`

- [ ] Add a RED frame-count case using the checkpoint's
  `causal_downsampling=true` geometry and a duration known to differ by one
  frame under the non-causal formula.
- [ ] Compute convolution padding as `(kernel - 1) + (stride - 1)` for causal
  downsampling and retain the existing symmetric formula for non-causal
  configurations.
- [ ] Add a RED missing-code test and remove the Code2Wav fallback that decodes
  placeholder `input_ids` as codec codes.
- [ ] Run the existing offline shape, registration, Stage processor, and model
  load tests before any Full-Duplex behavior is added.

This task is a prerequisite because a frame-count mismatch currently fails
prefill for ordinary input durations and would make 80 ms sequence accounting
untrustworthy.

### Task 2: Add a model-specific Full-Duplex plugin descriptor

**Files:**

- Create: `vllm_omni/experimental/fullduplex/nemotron_voicechat/__init__.py`
- Create: `vllm_omni/experimental/fullduplex/nemotron_voicechat/runtime.py`
- Create: `vllm_omni/experimental/fullduplex/nemotron_voicechat/serving_adapter.py`
- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/pipeline.py`
- Create: `vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml`
- Test: `tests/e2e/features/fullduplex/nemotron_voicechat/test_runtime_contract.py`

- [ ] Add RED tests requiring `duplex_control_enabled`, the Nemotron runtime
  extension path, and the Nemotron serving adapter path.
- [ ] Configure Stage 0 for exactly one generated token per 80 ms append with
  greedy sampling and `ignore_eos=true`.
- [ ] Validate that the append payload is float32 PCM with exactly 1280 finite
  samples and preserve `source_input_seq` in request-owned metadata.
- [ ] Keep the offline YAML unchanged. The duplex YAML enables `async_chunk`,
  one Session initially, persistent resumable requests, and Stage connector
  settings required for delta transport.

No Nemotron behavior is added to the generic Realtime adapter or ordinary
non-Duplex pipeline configuration.

### Task 3: Make the native Thinker accept one 80 ms append

**Files:**

- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_thinker.py`
- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemo_vendored/perception.py`
- Test: `tests/model_executor/models/test_nemotron_voicechat_streaming_thinker.py`

- [ ] Define request-owned state containing the waveform/mel carry,
  perception state, previous text token, previous function token, and last
  accepted `source_input_seq`.
- [ ] For the first correctness implementation, reproduce the reference
  rolling-buffer algorithm inside the native Thinker: append exactly one
  frame, run perception on the bounded context, and select the last stable
  acoustic embedding. This removes the Speech runtime immediately while
  preserving known behavior.
- [ ] Address the frame by request position plus `source_input_seq`; never use
  `decode_step += 1` as the source of truth, so retry/recompute cannot consume
  an extra acoustic frame.
- [ ] Prefix the system-prompt embeddings only on the first accepted frame.
  Rollback must leave first-frame prefill available for retry.
- [ ] Emit only the current function token. Remove per-step CPU concatenation
  of the cumulative function timeline from the hot path.
- [ ] Clean all request-owned tensors in `on_requests_finished` and verify a
  replacement Session starts from empty state.

After correctness is proven, a separate performance change may call the
vendored Conformer `cache_last_channel/cache_last_time` path. Cache adoption
requires frame-by-frame equivalence and is not mixed into the first native
correctness patch.

### Task 4: Stream Thinker output into the native Talker

**Files:**

- Modify: `vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py`
- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_talker.py`
- Test: `tests/model_executor/stage_input_processors/test_nemotron_voicechat.py`
- Test: `tests/model_executor/models/test_nemotron_voicechat_streaming_talker.py`

- [ ] Add RED tests that an unfinished Stage-0 stream can emit one immutable
  text/function delta at a segment boundary.
- [ ] Replace the offline-only full timeline handoff in duplex mode with
  `{text_token, function_token, source_input_seq}`. Keep the existing offline
  processor for non-Duplex inference.
- [ ] Initialize EAR-TTS warmup once per request, not once per resumed
  segment. `set_init_inputs()` output must be snapshotted into request-owned
  state before another Session can initialize.
- [ ] On every resumed input, execute exactly one
  `infer_codes_one_step()` call and emit one `[1, 31]` code delta.
- [ ] Replace the cumulative `codes_rows` plus `torch.cat` output with the
  current row. Preserve EAR-TTS KV and previous code until Session cleanup.
- [ ] Keep the first acceptance configuration at `max_num_seqs=1`; lift it to
  two only after isolated per-request state passes interleaving tests.

### Task 5: Incrementally decode Code2Wav

**Files:**

- Modify: `vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py`
- Modify: `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_code2wav.py`
- Test: `tests/model_executor/models/test_nemotron_voicechat_streaming_code2wav.py`

- [ ] Add RED tests that two sequential code deltas reuse one codec cache and
  return audio deltas rather than cumulative waveforms.
- [ ] Store one vendored `CausalConv1dCache` per request and call
  `audio_codec.decode(..., cache=cache, flush=False)` for ordinary frames.
- [ ] Carry text/function/source correlation metadata through Stage 1 without
  putting OpenAI ids into the Stage payload.
- [ ] Flush and delete codec state on close/abort/expiry. A missing or stale
  code delta must fail explicitly and cannot decode placeholder ids.
- [ ] Verify PCM is finite float32 at 22.05 kHz and that concatenated deltas
  have a sane duration relative to accepted frame count.

### Task 6: Bind native model state to the existing lease lifecycle

**Files:**

- Create: `vllm_omni/experimental/fullduplex/nemotron_voicechat/resources.py`
- Modify: `vllm_omni/experimental/fullduplex/nemotron_voicechat/runtime.py`
- Test: `tests/e2e/features/fullduplex/nemotron_voicechat/test_resource_lifecycle.py`

- [ ] Declare bounded Stage capacity and prewarm sizes through the existing
  optional execution profile.
- [ ] Reserve one opaque Session handle before the first Stage request and
  release it after Stage cleanup completes.
- [ ] Advance epoch without reusing stale output state. Session close,
  startup failure, Stage failure, expiry, and cancellation rollback must be
  idempotent.
- [ ] Keep KV, perception, EAR-TTS, and codec tensors in their native worker
  request maps. The lease coordinates lifetime and capacity; it does not
  serialize or clone those tensors through the control plane.

### Task 7: Project native outputs to Realtime

**Files:**

- Modify: `vllm_omni/experimental/fullduplex/nemotron_voicechat/serving_adapter.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/runtime_bridge.py`
- Test: `tests/e2e/features/fullduplex/nemotron_voicechat/test_serving_adapter.py`
- Test: `tests/e2e/features/fullduplex/openai/test_runtime_adapter_boundary.py`

- [ ] Decode text and function deltas with model-owned cursors and emit typed
  `DuplexUserTranscriptDelta` and function-call events.
- [ ] Emit assistant audio as typed speak chunks; text EOS closes the current
  speaking span but does not close the logical Session.
- [ ] Allocate Realtime response, item, and call ids only in the serving
  layer. Stage payloads and model events remain transport neutral.
- [ ] Preserve `session.updated` ordering, cancellation acknowledgement, and
  stale-epoch filtering already validated by the architecture branch.

RNNT transcript support is accepted only when the checkpoint's native RNNT
state and tokenizer are loaded inside vLLM-Omni. Until then the server must
omit transcript events rather than return an empty transcript as a success
claim.

## 7. Verification gates

### Local and CPU contract checks

```bash
ruff check <changed-python-files>
ruff format --check <changed-python-files>
git diff --check
pytest -q tests/model_executor/models/test_nemotron_voicechat_shapes.py
pytest -q tests/model_executor/stage_input_processors/test_nemotron_voicechat.py
pytest -q tests/e2e/features/fullduplex/nemotron_voicechat
pytest -q tests/e2e/features/fullduplex/engine
pytest -q tests/e2e/features/fullduplex/openai/test_runtime_adapter_boundary.py
```

### H20 single-Session acceptance

Run on one idle H20 with an exact branch SHA and checkpoint revision:

1. Start the native three-Stage duplex YAML without the Speech repository on
   `PYTHONPATH`.
2. Open one Engine Full-Duplex Session.
3. Pace at least 40 consecutive 1280-sample appends at 80 ms.
4. Verify one stable request identity per Stage and monotonic input/output
   sequence correlation.
5. Require nonempty finite 22.05 kHz PCM, no missing/stale frame errors, and
   complete state cleanup after close.
6. Compare text/function tokens and output audio against the reference path
   on the same fixture. Report differences instead of reducing the gate to
   nonempty audio.

Realtime WebSocket E2E follows only after Engine E2E passes. Two-Session
admission follows only after the single-Session native state is correct.

## 8. Risk controls and compatibility

- The new pipeline behavior is selected only by the duplex deployment and
  plugin; #5842 offline single-turn inference remains unchanged.
- Stage payload changes are Nemotron-specific. Shared Full-Duplex files change
  only when a second-consumer contract test proves the need.
- The initial bounded rolling perception buffer trades compute for migration
  speed and parity. It is a known performance cost, not hidden as an
  optimization.
- Native Talker and codec state can retain large GPU tensors. Cleanup tests
  and post-close memory evidence are required before multi-Session support.
- A Draft PR must not claim native Full-Duplex E2E until the Speech checkout is
  absent from the server environment and the three vLLM-Omni Stages are seen
  in startup/runtime logs.

## 9. Expected follow-up diff boundary

The native model follow-up is expected to modify approximately:

- 700–1,200 lines in Nemotron model and Stage processor files;
- 400–700 lines in `experimental/fullduplex/nemotron_voicechat`;
- 300–600 lines of focused tests and H20 drivers;
- fewer than 150 lines in shared Full-Duplex/OpenAI code unless a new generic
  contract is independently justified.

The external Speech plugin package, nested runtime pool, and Speech-owned
Realtime execution are explicitly excluded from the vLLM-Omni PR.
