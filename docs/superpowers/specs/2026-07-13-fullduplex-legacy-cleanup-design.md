# Full-Duplex Legacy Cleanup Design

## Goal

Remove obsolete compatibility implementations from the experimental full-duplex
runtime without changing the verified MiniCPM-o 4.5 scheduler data-plane
behavior or the generic non-native chat fallback used by other models.

The cleanup must leave one implementation for each active responsibility and
must reject incomplete runtime providers explicitly instead of guessing legacy
model methods.

## Current Problems

### Duplicate Realtime audio implementations

`openai/realtime_session.py` delegates audio conversion to `openai/audio.py`,
but retains complete codec implementations after unconditional returns. Those
blocks are unreachable. The protocol also supports three query-controlled event
families: current `response.audio.*`, opt-in `response.output_audio.*`, and a
legacy group containing `response.text.*`, `response.cancelled`, and
`response.output_item.created`. No repository caller enables the legacy or
output event switches.

### Reflection-based worker runtime adapter

`engine/worker.py` can adapt arbitrary `prepare/prefill/generate` method names
into a duplex runtime. The only production provider is the MiniCPM provider,
whose Stage0 and Stage1 runtimes already implement the complete
`open_duplex_session`, `append_duplex_input`, `signal_duplex_turn`, and
`close_duplex_session` contract. The reflection adapter is exercised only by
adapter-specific tests and can bypass the scheduler-owned runtime boundary.

### MiniCPM no-op and fail-open paths

The private Stage0 embedding helper accepts `new_speech` but never reads it.
The caller and tests still pass it, which makes the contract misleading. The
Stage0 reference-audio decoder also catches every exception and silently drops
invalid reference audio even though session opening should fail loudly.

### Unreferenced bridge script

`official_demo_bridge_worker.py` is not referenced by documentation, tests,
launch scripts, or runtime entry points. It is a standalone pre-runtime bridge
that duplicates the supported Realtime demo path.

### Combined-runtime compatibility export

`minicpmo45/runtime.py` only re-exports the split Stage0 and Stage1 runtimes
under the former combined-runtime module path. Production code does not import
it; only tests retained the old path.

## Target Design

### Realtime protocol

- Emit exactly one public audio lifecycle: `response.audio.delta`,
  `response.audio.done`, `response.audio_transcript.delta`, and
  `response.audio_transcript.done`.
- Keep internal duplex events named `response.output_audio.delta`; projection
  to the public Realtime event remains centralized in
  `NativeRealtimeSessionProtocol`.
- Call `openai/audio.py` conversion functions directly from protocol and
  serving code.
- Remove legacy/output event query flags, duplicate wrapper methods, duplicate
  codec bodies, and imports used only by those bodies.

### Worker runtime contract

Every registered native duplex provider must return a target implementing all
four lifecycle methods:

1. `open_duplex_session`
2. `append_duplex_input`
3. `signal_duplex_turn`
4. `close_duplex_session`

The worker validates this contract at open. Missing methods are a loud open
error. Append, signal, and close call the explicit methods directly. There is
no prepare/prefill/generate reflection adapter, loaded-model signal fallback,
or stop/reset/cleanup close fallback.

Provider registration remains supported, so future models can implement the
same explicit contract without modifying shared worker code.

### MiniCPM Stage0

- Remove the unused `new_speech` parameter from the private embedding helper
  and its caller. Previous-terminator reinjection remains append-driven, which
  matches the implementation.
- Decode resolved reference audio through the shared worker helper without a
  broad exception handler. Invalid payloads fail session opening.
- Import split runtimes from `stage0` and `stage1` directly and remove the
  old combined-runtime compatibility export.

## Explicitly Retained Boundaries

- `core/`, typed fences, reducer/runtime ports, and JoyVL adapters are retained
  as the model-independent runtime contract.
- MiniCPM `force_listen` is retained as the documented short-commit/overlap
  safety override; it is active behavior, not dead compatibility code.
- Generic non-native chat streaming remains available to avoid changing other
  models.
- Audio format support for PCM, WAV, G.711 u-law, and G.711 A-law remains in
  `openai/audio.py`.
- Automatic/VAD barge-in, multi-session concurrency, scheduler-native append,
  and bounded KV remain outside the capability claim.

## Verification

1. RED tests prove legacy event switches no longer create extra event families,
   incomplete providers fail at open, and invalid reference audio is not
   silently accepted.
2. Vulture reports no 100-percent-confidence unreachable or unused code in the
   cleaned package.
3. Ruff check, Ruff format, compileall, and `git diff --check` pass.
4. The expanded remote H20 full-duplex matrix passes.
5. The pinned no-hint, real-time paced, three-turn response-required E2E passes
   with independent playback cursors and committed follow-up history.
