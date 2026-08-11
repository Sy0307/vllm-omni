# Nemotron VoiceChat 11B: vLLM-Omni and Full-Duplex Implementation Plan

> **Historical plan — superseded for model integration.** This document
> records the external NVIDIA Speech runtime prototype that was used to
> validate the Full-Duplex Session/control-plane contracts. It is not the
> final model-integration direction. The authoritative migration plan is
> `docs/superpowers/plans/2026-08-07-nemotron-voicechat-native-migration.md`,
> which uses vLLM-Omni PR #5842 as the native Thinker/Talker/Code2Wav base and
> removes the external Speech runtime from the deployed path.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the experimental Full-Duplex architecture with Nemotron
VoiceChat as a second model-native consumer through both Engine and Realtime
WebSocket E2E.

**Architecture:** Reuse the existing fenced Session, idempotent append,
resumable request, persistent KV, and ordinary vLLM scheduler. Add only the
four contracts required by the second consumer: model-owned input control,
typed transcript/function events, a resource-lease rollback saga, and an
optional execution profile. Keep all new framework contracts under
`vllm_omni/experimental/fullduplex`; keep Nemotron model policy in the external
Speech plugin.

**Tech Stack:** Python 3.12, asyncio, vLLM V1, vLLM-Omni, PyTorch, msgspec,
pytest, OpenAI Realtime WebSocket, NVIDIA H20.

---

> Status: implementation in progress. The simple external plugin, shared
> Full-Duplex contracts, Nemotron Engine adapter, and Realtime adapter exist.
> Engine single/dual Session and the main Realtime lifecycle scenarios have
> run on H20. One gap found by the real Realtime run remains before final
> acceptance: first-append system-prompt/tool prefill. The Nemotron adapter
> contract requires each upstream append to contain exactly one 80 ms frame;
> it rejects partial or multi-frame input instead of adding a generic drain.

## Goal

Adapt `nv-community/NVIDIA-NemotronLabs-VoiceChat-11B` in two observable
milestones:

1. A real, simple vLLM-Omni LLM adaptation that can replace the public
   Speech wrapper's native Nemotron-H call while keeping its perception,
   fusion, RNNT, EarTTS, and codec behavior intact.
2. A Full-Duplex integration that reuses the existing long-lived Session,
   fence, resumable Stage-0 request, incremental append, persistent KV, and
   cleanup lifecycle.

The final result must process paced 80 ms PCM continuously, preserve one
request/KV identity until Session close, emit assistant text/audio plus the
RNNT transcript and function-call channel, survive speaking-span EOS without
tearing down the Session, cancel without stale output, and work with one and
two concurrent Sessions on H20.

## Pinned sources

- vLLM-Omni implementation worktree:
  `/Users/sy03/.config/superpowers/worktrees/vllm-omni/nemotron-voicechat-duplex`
- vLLM-Omni branch: `feat/nemotron-voicechat-duplex`
- Starting HEAD: `0b7acfc4a1c83449c171b92d8861eda1498f1b46`
- Base `origin/main`: `b4581b293beb837a2936b16c0cad6d2834fed9f6`
- NVIDIA Speech implementation worktree:
  `/Users/sy03/.config/superpowers/worktrees/Speech/nemotron-voicechat-vllm-omni`
- Speech branch: `feat/nemotron-voicechat-vllm-omni`
- Speech source HEAD: `14c77efb8110ee46eebdc50a3b15ee6d2c1a3878`
- ModelScope model: `nv-community/NVIDIA-NemotronLabs-VoiceChat-11B`

The vLLM-Omni branch already contains earlier Full-Duplex work. Nemotron's
own diff must always be reported separately from the existing 52-file branch
diff.

## Confirmed model contract

- Input PCM is 16 kHz.
- One execution frame is 80 ms, exactly 1280 samples.
- Each frame produces one Nemotron-H step.
- The fused LLM input uses current user-acoustic embedding, previous assistant
  token, and previous function token.
- Text and function heads are separate outputs.
- User transcription comes from the RNNT path in the released checkpoint;
  `use_separate_asr_head=false`.
- EarTTS consumes one assistant token per frame and produces codec codes.
- Audio output is 22.05 kHz.
- Text EOS ends the current assistant speaking span, not the logical Session.
- Perception, RNNT, LLM KV, EarTTS, and codec states live until close, abort,
  expiry, or failure.
- The function-call path may temporarily advance faster than the 80 ms audio
  clock while injecting a tool result; this is model policy, not a generic
  scheduler rule.

## Architecture decision

### Shared engine primitive

MiniCPM and Nemotron use the same runtime primitive:

```text
long-lived Session and fence
  -> append incremental model input
  -> resume the same Stage-0 request
  -> execute with max_tokens=1
  -> retain request identity and KV/model state
  -> wait for the next input unit
```

The difference is only the model unit and frequency: MiniCPM buffers roughly
one second; Nemotron submits one 80 ms frame. No standalone Duplex scheduler
will be added before H20 data shows the existing append/resume path misses the
frame budget.

### Simple adaptation

The simple path is intentionally hybrid:

```text
Speech native perception/fusion/RNNT
  -> one-stage vLLM-Omni Nemotron-H plugin
  -> Speech native EarTTS/codec
```

It uses a persistent resumable request but does not require the Duplex control
plane, OpenAI Realtime mapping, or multi-stage EarTTS. It exists to prove the
weight mapping, fused-embedding contract, text token, function token, KV
continuity, and native-vs-adapted behavior before framework integration.

The public Speech branch's `CustomInputSpec` path is not used. Official vLLM
0.26 does not provide that NVIDIA-private ABI. vLLM-Omni already transports
per-request tensors through `model_intermediate_buffer` and invokes the
model's `preprocess` hook.

### Full-Duplex path

```text
Realtime PCM ingress
  -> Nemotron session adapter (frame reservation, perception, fusion, RNNT)
  -> existing Duplex control plane (fence, input_seq, operation id, lease)
  -> existing Stage-0 append/resume, max_tokens=1
  -> Nemotron-H text/function outputs
  -> EarTTS streaming state and codec decode
  -> typed model events
  -> OpenAI Realtime text/audio/transcript/function mapping
```

Perception/fusion/RNNT and codec remain model adapter responsibilities. They
must not become generic Full-Duplex concepts. EarTTS is first validated native
behind the simple adapter; it moves to a downstream stage only after Stage-0
equivalence is established. This avoids debugging weight conversion, LLM
state, TTS state, and control-plane lifecycle simultaneously.

## Ownership and invariants

| State | Owner | Required invariant |
| --- | --- | --- |
| Session/incarnation/epoch | Duplex control plane | Reject stale append and cleanup |
| operation id/input sequence | Duplex control plane | Retry is idempotent; order is monotonic |
| 80 ms PCM reservation | Nemotron session adapter | Commit exactly once or roll back |
| perception/RNNT cache | Nemotron session adapter | One isolated state per Session |
| Stage-0 request/KV | vLLM scheduler/worker | Same internal request until close |
| previous text/function token | Nemotron model request state | Advance only after accepted output |
| EarTTS/codec cache | Nemotron model adapter/stage | Isolated by fenced Session identity |
| speaking-span state | Nemotron model policy | EOS closes span, never Session |
| output id/output sequence | typed event bridge | Monotonic per assistant output |
| Realtime response id | serving layer | Never transported as a stage payload |

No mutable live `Request` object may cross a background processor boundary.
Every queued per-frame handoff must contain an immutable snapshot of the
frame sequence, fused embedding or token, function token, and boundary.

## Planned file boundary

### NVIDIA Speech repository

- Add `nemotron_voicechat_vllm_omni/nemotron_voicechat_llm.py` for the
  lightweight external vLLM-Omni model class. Keeping the plugin outside
  `nemo.collections.speechlm2` prevents spawned workers from importing the
  unrelated Lightning training stack through `speechlm2/__init__.py`.
- Add `nemotron_voicechat_vllm_omni/model_interface.py`
  for a `ModelInterface` implementation backed by `AsyncOmniEngine`.
- Add `nemotron_voicechat_vllm_omni/pipeline.py`
  for the one-stage simple pipeline, and extend it only if EarTTS becomes a
  downstream stage.
- Add `nemotron_voicechat_vllm_omni/register.py`
  and a `vllm_omni.general_plugins` entry point so spawned stage processes
  register the external architecture.
- Update
  `nemo/collections/speechlm2/inference/vllm/scripts/convert_nemotronllm_checkpoint.py`
  to emit the plugin architecture and vLLM-Omni config without
  `custom_input_specs/custom_outputs`.
- Update
  `nemo/collections/speechlm2/inference/model_wrappers/model_factory.py`
  with a distinct `vllm_omni_llm` backend; do not silently change existing
  `native` or NVIDIA-private `vllm_llm` behavior.
- Update
  `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
  only where request lifecycle/output extraction requires it.
- Add focused tests under
  `tests/collections/speechlm2/inference/vllm_omni/`.

### vLLM-Omni repository

- Add model-specific runtime under
  `vllm_omni/experimental/fullduplex/nemotron_voicechat/` only after the
  simple adapter passes.
- Add model-specific tests under
  `tests/experimental/fullduplex/nemotron_voicechat/` and H20 scenarios under
  `tests/e2e/features/fullduplex/nemotron_voicechat/`.
- Extend typed model events with transcript/function events only if RED tests
  prove current assistant text/audio events cannot express them.
- Add an immutable processor snapshot or a stage prewarm-input hook only when
  the Nemotron Stage-1 RED test proves it is required.
- Do not add a generic ingress enum, a generic frame clock, a new scheduler,
  or Nemotron metadata to ordinary non-Duplex request paths.

## Task 1: Baseline and source provenance

1. Create isolated remote worktrees for the pinned Speech and vLLM-Omni refs.
2. Recheck GPU occupancy immediately before every launch and use only an idle
   H20.
3. Download the ModelScope snapshot into a shared model directory.
4. Record ModelScope revision, every file size, and SHA-256 of
   `model.safetensors`.
5. Run the public native inference script on its official/example audio and
   save:
   - input WAV and configuration;
   - text token ids/text;
   - function token ids/text;
   - incremental RNNT transcript;
   - output WAV;
   - per-frame perception/LLM/EarTTS/codec timings.

No adapted output is accepted without this native reference.

## Task 2: RED tests for the simple model contract

Add tests that fail before implementation and prove:

1. Conversion drops NVIDIA-private `custom_input_specs` and writes the external
   architecture name.
2. Weight mapping loads the Nemotron-H backbone, text head, embeddings, and
   function head; unrelated perception/RNNT/TTS keys are ignored explicitly.
3. `preprocess` replaces placeholder token embeddings with a per-request
   `combined_embeds` tensor of matching row count/dtype/device.
4. Missing, stale, wrong-rank, wrong-row-count, or wrong-hidden-size embeddings
   fail the request rather than returning a PAD/finish marker.
5. `make_omni_output` computes a per-request function token from the correct
   request row using `request_token_spans`.
6. Mixed prefill/decode and two-request batches never route request 0's
   embedding or function output to request 1.

Run the tests before implementation and preserve the RED failure output.

## Task 3: GREEN simple model/plugin implementation

Implement the smallest external class around vLLM's supported
`NemotronHForCausalLM`:

- `has_preprocess = True`;
- `have_multimodal_outputs = True`;
- standard vLLM sampler remains the canonical text sampler;
- `preprocess` consumes only the current request's fused embedding;
- `make_omni_output` emits function token/logits as side output;
- state is request-local through runner-owned `model_intermediate_buffer`;
- no global current request, live mutable `Request`, or request-0 fallback.

Register the plugin in orchestrator and spawned Stage workers. The converter
must stream/filter the 44 GB safetensors file without holding two full copies
in host memory.

## Task 4: GREEN simple streaming ModelInterface

Implement `VllmOmniNemotronModel(ModelInterface)`:

1. Start one one-stage `AsyncOmniEngine` per configured backend.
2. On the first call, submit a resumable request with placeholder token ids and
   the current fused embedding.
3. On later calls, use `add_streaming_update()` for the same request id.
4. Set Stage 0 `max_tokens=1`, `ignore_eos=true`, and the configured sampling
   behavior.
5. Wait for exactly one accepted output for the submitted input sequence.
6. Return `predicted_token` plus `function_predicted_token`; RNNT remains the
   public wrapper's native output.
7. Abort removes the request and all backend state.
8. Never translate engine/model exceptions into a PAD token or successful
   finish.

Focused tests must cover first submit, subsequent update, exactly-one-output,
timeout, duplicate output, stale output, engine failure, and abort.

## Task 5: H20 simple-adapter equivalence gate

With identical input, prompt, speaker, sampling seed/settings, and checkpoint:

1. Run public native inference.
2. Run `vllm_omni_llm` with native perception/RNNT/EarTTS/codec.
3. Compare per-frame text and function ids for greedy decoding.
4. Compare RNNT transcript equality (it is the same native component).
5. Check output audio is nonempty, finite, correct sample rate/duration, and
   does not contain stale chunks.
6. Report first-token and per-frame latency; do not claim quality equivalence
   from nonempty audio alone.

The Full-Duplex work does not start until the model loads and at least one
native-vs-adapted sample passes this gate, or a precisely explained numerical
kernel difference is isolated with token-level evidence.

## Task 6: Typed transcript and function channels

**Files:**

- Modify: `vllm_omni/experimental/fullduplex/engine/model_events.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/__init__.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/runtime_bridge.py`
- Test: `tests/e2e/features/fullduplex/engine/test_duplex_model_events.py`
- Test: `tests/e2e/features/fullduplex/openai/test_runtime_adapter_boundary.py`

- [ ] **Step 1: Write RED event-ordering tests.** Add immutable
  `DuplexUserTranscriptDelta`, `DuplexFunctionCallStart`,
  `DuplexFunctionCallDelta`, and `DuplexFunctionCallEnd` examples. Assert
  epoch filtering, channel-local monotonic sequence, duplicate suppression,
  gaps as protocol errors, and independence from assistant `output_id`.
- [ ] **Step 2: Run the two focused files and record the expected import or
  ordering failures.**
- [ ] **Step 3: Implement a `DuplexSideChannelLedger`.** It owns only the
  current fence, transcript sequence, and active/completed function call
  sequences. It must not allocate OpenAI response ids or mutate Session state.
- [ ] **Step 4: Project accepted events to Realtime.** Transcript events map to
  `conversation.item.input_audio_transcription.delta/completed`; function
  events map to function-call item and argument delta/done records. The serving
  layer creates wire ids; the engine events remain wire-protocol neutral.
- [ ] **Step 5: Run focused tests, Ruff, and `git diff --check`.**

## Task 7: Atomic model-resource lease saga

**Files:**

- Create: `vllm_omni/experimental/fullduplex/engine/resource_lease.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/contracts.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/duplex_control_plane.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/duplex_runtime.py`
- Test: `tests/e2e/features/fullduplex/engine/test_duplex_resource_lease.py`
- Test: `tests/e2e/features/fullduplex/engine/test_duplex_control_plane.py`

- [ ] **Step 1: Write RED saga tests.** Use three recording providers and
  assert reserve order A/B/C, reverse rollback B/A when C fails, reverse
  release C/B/A on close, idempotent retry after partial release, and no
  logical admission-slot reuse before cleanup completes.
- [ ] **Step 2: Define the narrow protocol.** Each provider declares
  `provider_id`, async `reserve(fence, session_config, runtime_config)`, async
  `release(handle, abort)`, and optional async `prewarm(batch_sizes)`.
- [ ] **Step 3: Implement `DuplexResourceLeaseCoordinator`.** Store opaque
  handles only in the coordinator; never serialize them into payloads or
  expose them through Realtime.
- [ ] **Step 4: Integrate open/close/expiry/open-rollback.** Reserve after
  logical Session creation and before Stage request reservation. Preserve
  pending cleanup for retry and finalize the Session only after provider and
  Stage cleanup both succeed. Cancellation changes epoch but does not destroy
  Session-scoped resources.
- [ ] **Step 5: Run focused control-plane and lease suites.** Existing
  extensions without providers must retain byte-for-byte result behavior.

## Task 8: Optional execution profile and deadline metrics

**Files:**

- Modify: `vllm_omni/experimental/fullduplex/engine/contracts.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/duplex_runtime.py`
- Modify: `vllm_omni/experimental/fullduplex/engine/duplex_control_plane.py`
- Test: `tests/e2e/features/fullduplex/engine/test_duplex_runtime.py`
- Test: `tests/e2e/features/fullduplex/engine/test_duplex_control_plane.py`

- [ ] **Step 1: Write RED validation and timing tests.** Reject duplicate or
  non-positive prewarm sizes and non-positive/non-finite latency budgets.
  Prove a late step increments metrics but still returns its correct output.
- [ ] **Step 2: Add immutable `DuplexExecutionProfile`.** Default profile has
  no prewarm sizes and no deadline, so existing models and ordinary requests
  are unchanged.
- [ ] **Step 3: Prewarm once before first reservation.** The resource
  coordinator invokes provider prewarm in declaration order and caches
  successful completion.
- [ ] **Step 4: Record append latency.** Expose count, p50/p95/max and deadline
  misses through an in-process snapshot and structured log fields. Do not
  reorder work or fail correct late steps.
- [ ] **Step 5: Run the full Full-Duplex engine suite.**

## Task 9: Nemotron Engine runtime extension

**Files in the NVIDIA Speech worktree:**

- Create: `nemotron_voicechat_vllm_omni/duplex/__init__.py`
- Create: `nemotron_voicechat_vllm_omni/duplex/runtime.py`
- Create: `nemotron_voicechat_vllm_omni/duplex/resources.py`
- Modify: `nemotron_voicechat_vllm_omni/pipeline.py`
- Modify: `nemotron_voicechat_vllm_omni/model_interface.py`
- Test: `tests/collections/speechlm2/inference/vllm_omni/test_duplex_runtime.py`

- [ ] **Step 1: Write RED append-plan tests.** One accepted frame must produce
  one placeholder token plus immutable `combined_embeds`, `max_tokens=1`, and
  the caller's `source_input_seq`; wrong mode/rank/row count fails explicitly.
- [ ] **Step 2: Add `NemotronVoiceChatDuplexRuntimeExtension`.** It declares
  `APPEND_AUDIO_CHUNK`, preserves the one-stage sampling type, and never owns
  Session/fence/operation id allocation.
- [ ] **Step 3: Add a bounded resource provider.** Reserve isolated
  perception/RNNT/TTS/codec Session slots and release them idempotently. The
  provider owns no logical Session TTL.
- [ ] **Step 4: Configure the external pipeline.** Set
  `duplex_control_enabled=True`, the runtime-extension path, the serving-
  adapter path, and an execution profile with batch sizes `(1, 2)` and the
  80 ms observational budget.
- [ ] **Step 5: Remove diagnostic environment switches and force the verified
  plugin compile option `cudagraph_mode=NONE` while preserving unrelated
  caller compile options.**

## Task 10: H20 Engine-level Full-Duplex E2E

**Files:**

- Create in Speech:
  `tests/collections/speechlm2/inference/vllm_omni/run_duplex_engine_e2e.py`
- Create in vLLM-Omni:
  `tests/e2e/features/fullduplex/nemotron_voicechat/test_engine_contract.py`

- [ ] **Step 1: Run contract RED tests with a fake Stage port.** Verify
  operation-id replay, monotonic `input_seq`, stale fence rejection, one-step
  append, speaking EOS without close, and reverse cleanup.
- [ ] **Step 2: On an idle H20, run a real one-Session engine scenario.** Use
  `open_duplex_session -> append_duplex_input -> try_get_output -> close` with
  paced 1280-sample-derived embeddings. Record one stable Stage request id and
  cumulative token/KV continuity.
- [ ] **Step 3: Run two concurrent Sessions.** Assert distinct request ids,
  independent token/function streams, bounded progress skew, and complete GPU
  state release.
- [ ] **Step 4: Record exact SHA, GPU, model revision, logs, p50/p95/max step
  latency, deadline misses, and cleanup evidence.** Engine E2E is the first
  architecture gate; do not start serving claims before it passes.

## Task 11: Model-owned Realtime input controller

**Files:**

- Modify: `vllm_omni/experimental/fullduplex/openai/runtime_adapter.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/session_runner.py`
- Modify: `vllm_omni/experimental/fullduplex/openai/runtime_bridge.py`
- Modify: `vllm_omni/experimental/fullduplex/minicpmo45/serving_adapter.py`
- Test: `tests/e2e/features/fullduplex/openai/test_runtime_adapter_boundary.py`
- Test: `tests/e2e/features/fullduplex/openai/test_websocket_actor.py`

- [ ] **Step 1: Write RED second-adapter tests.** A minimal non-MiniCPM adapter
  with opaque Session state must append, commit, cancel, and close without
  defining MiniCPM audio-buffer or continuation fields.
- [ ] **Step 2: Define `DuplexInputController`, immutable input commands,
  reservations, and effects.** Reservations own commit/rollback; effects
  describe append, clear, and response-boundary work without calling handler
  internals.
- [ ] **Step 3: Move MiniCPM direct field accesses behind its controller.** Its
  behavior and existing E2E fixtures remain unchanged.
- [ ] **Step 4: Remove MiniCPM imports from generic runner and bridge.** Generic
  modules execute controller effects and typed events only.
- [ ] **Step 5: Run import-boundary, MiniCPM, PersonaPlex, and generic Realtime
  regression suites.**

## Task 12: Nemotron Realtime serving adapter

**Files in the NVIDIA Speech worktree:**

- Create: `nemotron_voicechat_vllm_omni/duplex/input.py`
- Create: `nemotron_voicechat_vllm_omni/duplex/session.py`
- Create: `nemotron_voicechat_vllm_omni/duplex/serving_adapter.py`
- Create: `nemotron_voicechat_vllm_omni/duplex/data_plane.py`
- Modify:
  `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
- Test: `tests/collections/speechlm2/inference/vllm_omni/test_duplex_serving.py`

- [ ] **Step 1: Write RED frame-boundary tests.** Each upstream append must
  contain exactly one 1280-sample PCM frame. Partial and multi-frame appends
  fail closed; failed reservations roll back; duplicate operation ids do not
  repeat perception/model work; the next frame is rejected until the prior
  output has been projected.
- [ ] **Step 2: Implement isolated Session model state.** It owns perception,
  RNNT, previous text/function token, EarTTS, codec, transcript cursor, and
  function-call parser state. No state is stored on the adapter singleton.
- [ ] **Step 3: Implement controller and projector.** The controller converts
  PCM to fused embeddings and emits transcript events. The projector consumes
  Stage text/function outputs, advances EarTTS/codec, and emits typed assistant
  and function events. Text EOS emits `SPEAK_END` and keeps the Session open.
- [ ] **Step 4: Bind the adapter to the engine client without constructing a
  nested `AsyncOmniEngine`.** The Speech wrapper may expose reusable
  perception/TTS helpers, but engine Session/KV ownership remains in the outer
  Full-Duplex control plane.
- [ ] **Step 5: Run focused Speech and vLLM-Omni serving suites.**

## Task 13: Realtime system-prompt and tool prefill

**Files in NVIDIA Speech:**

- Modify: `nemotron_voicechat_vllm_omni/duplex/serving_adapter.py`
- Modify: `nemotron_voicechat_vllm_omni/duplex/session.py`
- Modify: `nemotron_voicechat_vllm_omni/duplex/runtime.py`
- Modify: `nemotron_voicechat_vllm_omni/duplex/realtime_e2e.py`
- Test: `tests/collections/speechlm2/inference/vllm_omni/test_duplex_runtime.py`
- Test: `tests/collections/speechlm2/inference/vllm_omni/test_duplex_serving.py`
- Test:
  `tests/collections/speechlm2/inference/vllm_omni/test_duplex_realtime_e2e_driver.py`

- [ ] **Step 1: Write RED prompt-rendering tests.** Use a Realtime config with
  `instructions`, `realtime_tools`, and `realtime_tool_choice`. Assert the
  adapter renders the official Nemotron `<AVAILABLE_TOOLS>` and `<TOOLCALL>`
  contract without copying Realtime ids into model input. Reject malformed
  tools and unsupported tool-choice values as runtime-config errors.
- [ ] **Step 2: Write RED first-append prefill tests.** The reserved native
  Session prepares prompt embeddings and exact matching token ids once. The
  first frame payload prepends those rows; later frames contain one embedding
  row and one placeholder token. Retry/rollback of the first frame must not
  mark the prefill as consumed.
- [ ] **Step 3: Extend only the Nemotron append payload.** Add optional
  `prompt_token_ids` beside `combined_embeds`. The runtime accepts multiple
  embedding rows only when prompt ids are present, match every leading row,
  and end with the single frame placeholder. Existing one-row payloads remain
  unchanged.
- [ ] **Step 4: Reuse the released model prompt encoder.** Render with the
  model-specific template, then call the existing wrapper helper to prepare
  embeddings once per leased Session. Keep prompt tensors in model-owned
  Session state; generic Full-Duplex and Realtime state store no Nemotron
  tokenizer or prompt tensor.
- [ ] **Step 5: Strengthen the Realtime E2E client.** Send the
  `generate_random_number` tool and matching instructions. Require a nonempty
  call id, ordered function argument delta/done events, and the expected tool
  name. Absence of function events fails the scenario rather than being
  reported as informational.
- [ ] **Step 6: Run focused Speech tests, Ruff, format check, and
  `git diff --check`.**

## Task 14: H20 Realtime WebSocket E2E

**Files:**

- Create in Speech:
  `tests/collections/speechlm2/inference/vllm_omni/run_duplex_realtime_e2e.py`
- Create in vLLM-Omni:
  `tests/e2e/online_serving/nemotron_voicechat_realtime_duplex.py`

- [ ] **Step 1: Start the exact final server on an idle H20 and verify the
  advertised adapter/runtime descriptor.**
- [ ] **Step 2: Send paced PCM16 over `/v1/realtime`.** Verify transcript
  deltas, function-channel ordering, assistant text/audio, and Session reuse
  across speaking EOS.
- [ ] **Step 3: Cancel during queued audio.** Require a new epoch and no old-
  epoch text, function, or audio after the cancellation acknowledgement.
- [ ] **Step 4: Run one and two concurrent WebSocket Sessions.** Check
  isolation, fairness, bounded input memory, resource release, and admission of
  a replacement Session.
- [ ] **Step 5: Validate output WAVs.** Require nonempty finite 22.05 kHz audio,
  sane duration/RMS/peak, no long unintended silence, no clipping burst, and
  no final truncation. Save audio plus event/timing traces.
- [ ] **Step 6: Run ordinary non-Duplex smoke tests.** Confirm the optional
  contracts do not change default scheduling or output behavior.

## Verification commands

Every local shell command, including the Chrome bridge helper invocation, must
begin with `rtk`. Commands sent as the H20 payload use the remote machine's
native tools because `rtk` is not installed there. Every remote command must
target the isolated worktree. Exact selectors may be refined as tests are
added, but the final report must include the commands, SHA, GPU id,
pass/fail counts, and log paths.

```bash
git diff --check
ruff check <changed-python-files>
ruff format --check <changed-python-files>
pytest -q tests/collections/speechlm2/inference/vllm_omni
pytest -q tests/experimental/fullduplex/nemotron_voicechat
pytest -q tests/e2e/features/fullduplex/nemotron_voicechat
```

## Completion matrix

| Requirement | Required evidence | Current state |
| --- | --- | --- |
| ModelScope provenance | revision, sizes, model SHA-256 | Missing |
| Public native inference | tokens, transcript, function, WAV, timings | Missing |
| Simple model/plugin | focused GREEN tests | Missing |
| Simple real H20 inference | load + nonempty correct outputs | Missing |
| Native/adapted comparison | per-frame token/function/audio evidence | Missing |
| Full-Duplex Session reuse | request/KV identity trace | Missing |
| 80 ms paced E2E | timing and sequence trace | Missing |
| EOS without teardown | later span on same Session | Missing |
| RNNT transcript | typed monotonic deltas | Missing |
| Function-call channel | protocol trace | Missing |
| Cancel/no stale audio | acknowledged cancellation trace | Missing |
| Two Sessions | isolation/fairness/cleanup trace | Missing |
| Non-Duplex regressions | focused framework test matrix | Missing |
| Final architecture audit | actual gaps, risks, diff boundary | Missing |

The implementation is complete only when every row has direct current-state
evidence. Static tests cannot substitute for H20 inference or E2E.
