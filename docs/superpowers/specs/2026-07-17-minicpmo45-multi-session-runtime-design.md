# MiniCPM-o 4.5 Native Duplex Multi-Session Runtime Design

Date: 2026-07-17

Status: Approved architecture; implementation pending

## 1. Objective

Extend the MiniCPM-o 4.5 native duplex runtime from one admitted session to an
unbounded number of logical sessions on the same Stage0/Stage1 replica pair.
The runtime must isolate model, scheduler, protocol, playback, and reconnect
state for every session while leaving physical concurrency and memory admission
to the existing schedulers.

"Unbounded logical sessions" does not mean unbounded resident GPU requests.
`max_num_seqs`, KV-cache capacity, preemption, recomputation, and device memory
continue to bound concurrent execution. The serving layer must not impose a
separate fixed session-count limit.

The public Realtime adapter remains WebSocket-based, but the runtime contract
must be transport-neutral. Correctness must be proven both below the transport
through the engine API and through two concurrent public WebSocket sessions.

## 2. Non-goals

This design does not add:

- automatic or VAD-driven barge-in;
- scheduler-native media append beyond the existing resumable adapter;
- a guarantee that all sessions remain GPU-resident;
- bounded long-session KV or an unlimited-context claim;
- cross-API-head resume, shared external session storage, or replica migration;
- resume after server process restart;
- production latency or fairness SLOs;
- streaming video input, which requires a separate design.

Session lease is deliberately separate from KV lease. The implementation may
set `supports_session_lease=true`; it must not change
`supports_core_kv_lease` or `supports_kv_lease` to true without independent KV
residency and eviction evidence.

## 3. Approved product contract

### 3.1 Capacity and scheduling

- There is no application-level maximum number of logical duplex sessions.
- Each session has independent identity, input, response, playback, history,
  lease, Stage0 state, and Stage1 state.
- The vLLM scheduler owns finite active-request admission. Requests beyond the
  active capacity wait or are preempted according to the existing scheduler.
- A resource failure is explicit. The runtime must not silently discard input,
  context, output, or a session.
- One session cannot enqueue unbounded input or output replay data. Per-session
  byte and turn limits provide backpressure without imposing a session-count
  limit.

### 3.2 Lease defaults

The pipeline owns typed configuration with these defaults:

```yaml
duplex_session:
  idle_ttl_s: 300
  disconnect_grace_s: 30
  reaper_interval_s: 5
  resume_replay_ttl_s: 60
  resume_replay_max_bytes_per_session: 8388608
  max_pending_input_bytes_per_session: 16777216
  max_pending_turns_per_session: 4
```

`idle_ttl_s: null` disables idle expiry. Zero and negative values are invalid;
they are not alternate spellings for disabled behavior.

Authenticated append, commit, signal, playback acknowledgement, heartbeat,
attach, and resume activity refresh the lease. Routed model output also
refreshes the lease while a response is active. A session cannot expire in the
middle of an accepted append transaction or active response.

Disconnect grace and idle TTL serve different purposes. During
`disconnect_grace_s`, an already active response may continue and its output is
journaled for replay. If the client has not resumed when grace ends, that
response is terminated explicitly so an orphan cannot consume GPU indefinitely.
The session identity, Stage0 context, and acknowledged history remain resumable
until `idle_ttl_s` expires. A successful resume cancels both pending deadlines.

### 3.3 Resume contract

- `session.created` returns a server-generated 256-bit opaque `resume_token`.
- The server stores only a digest of the token and never logs the plaintext.
- A reconnect sends `session.resume` with `session_id`, `resume_token`, and the
  last received `server_event_seq`.
- A successful resume preserves incarnation, fence, Stage0 KV, protocol
  history, playback ledgers, and response state.
- The token rotates after every successful resume.
- Only one transport attachment can own a session. A valid new attachment
  atomically replaces the old attachment; the old connection receives
  `session.replaced` and is closed.
- Expired, closed, mismatched, or stale-incarnation sessions cannot resume.
- Recreating an expired session ID creates a new incarnation. All output from
  the previous incarnation is stale.

Resume authentication belongs to the serving adapter. Plaintext tokens and
transport attachment objects must never enter scheduler, worker, or model
messages.

### 3.4 Output replay

Every server event for a resumable session receives a monotonically increasing
`server_event_seq`. The serving layer keeps a bounded per-session journal until
events are acknowledged or expire.

On resume, the new attachment receives all retained events after the client's
`last_received_server_event_seq`. Replaying protocol events must not replay
domain transitions: response-scoped playback cursors and history commits remain
idempotent.

If the requested sequence is older than the retained journal, or the journal
would exceed its byte bound, the active response is terminated explicitly and
the client receives `session.resync_required`. The session and already
acknowledged history remain valid for later turns. The runtime must never
silently skip missing audio or transcript deltas.

## 4. Architecture and ownership

The design extends existing owners rather than creating a second session state
machine.

```text
Realtime/WebSocket or future transport
        |
        v
DuplexSessionAttachmentRegistry             serving-only
  - resume token digest and rotation
  - one active transport attachment
  - bounded event journal
  - protocol input/response/playback/history ledgers
        |
        | authenticated typed control calls
        v
DuplexControlClient                          transport-neutral client
        |
        v
DuplexControlPlane                           engine lifecycle authority
  - DuplexSessionRuntimeManager
  - fence/incarnation and lease generation
  - monotonic last activity / expiry
  - scheduler request bindings
  - periodic reaper and cleanup transaction
        |
        +--> Stage0 resumable request and per-session audio processor state
        |
        `--> Stage1 per-turn talker and per-response Token2Wav state
```

### 4.1 Engine lifecycle authority

`DuplexSessionRuntimeManager` remains the only engine-side owner of session
lifecycle. `DuplexSessionRuntimeState` gains typed lease state:

- `lease_generation`;
- `last_activity_monotonic`;
- `detached_at_monotonic`;
- `expires_at_monotonic`;
- active-operation accounting needed to prevent mid-transaction expiry.

It exposes state transitions rather than public mutable fields:

```text
open(fence, lease_config)
touch(fence, activity)
detach(fence)
resume(fence, expected_lease_generation)
begin_operation(fence, operation_id)
end_operation(fence, operation_id)
close(fence, reason)
collect_expired(now)
```

All time comparisons use a monotonic clock. Wall-clock timestamps may be added
to observability events but never decide expiry.

`DuplexControlPlane` gains typed touch/resume messages and a periodic reaper.
Expiry uses the same cleanup transaction as explicit close:

1. atomically mark the runtime session closing/expired;
2. invalidate the current fence and lease generation;
3. collect submitted and reserved request IDs;
4. abort submitted requests and release reserved bindings;
5. invoke model-owned request/session cleanup hooks;
6. remove the runtime session;
7. publish a lifecycle result for serving projections.

Close and expiry are idempotent. Concurrent close, resume, and reaper actions
must select exactly one terminal transition.

### 4.2 Serving projection and attachment ownership

The current serving `DuplexSession` remains the aggregate root for Realtime
protocol transitions and its four ledgers. It no longer independently decides
that runtime resources are open or closed. Open, resume, close, and expiry
projections follow successful control-plane results.

Transport/security state moves behind focused components:

- `DuplexResumeCredential`: token generation, constant-time verification,
  digest storage, and rotation;
- `DuplexTransportAttachment`: attachment generation and replacement;
- `DuplexEventJournal`: sequence allocation, byte accounting, acknowledgement,
  replay, expiry, and overflow;
- `DuplexSessionAttachmentRegistry`: atomic create/attach/resume/detach/close
  coordination keyed by session identity.

No component outside `DuplexSession` may directly mutate input, response,
playback, or conversation ledgers. Journal replay emits recorded payloads; it
does not call the session reducer again.

Disconnect no longer runs the current unconditional close path. It detaches the
transport, keeps the runtime session alive for the configured grace/TTL, and
lets model output continue into the event journal. Explicit close still cancels
the active response and releases all resources immediately.

### 4.3 Stable control surface

The stable engine message layer adds typed operations for:

- `touch_duplex_session`;
- `resume_duplex_session`;
- lifecycle expiry results.

Append and signal already touch lease state as part of their accepted
transactions. Heartbeat and playback acknowledgement use `touch` without
creating scheduler work. A resume operation contains session identity, fence,
and expected lease generation; it never contains the resume token.

The correlated RPC router continues to route request/response control results.
Unsolicited lifecycle events use a dedicated lifecycle sink and must not be
inserted into an arbitrary pending RPC waiter.

## 5. MiniCPM-o 4.5 state isolation

### 5.1 Stage0 streaming processor

The current `_MiniCPMO45Stage0SessionState` already isolates audio buffers,
context embeddings, context token IDs, prepared append state, and audio model
KV. It does not isolate the Hugging Face processor's mutable streaming mel
buffer.

Each Stage0 session therefore receives a shallow processor view that shares
immutable tokenizer/feature-extractor objects but owns an independent deep copy
of `_streaming_mel_processor`. Streaming configuration and reset occur only on
that session-local view. `process_audio_streaming(reset=False)` is never called
through the shared processor after session creation.

The session-local processor is created lazily on first audio append and removed
by the model cleanup hook. Creating or closing session B cannot call reset on
session A's processor state.

The Stage0 state key is a typed identity containing session ID and incarnation,
not a lossy request-ID substring. Fence validation rejects stale epoch/turn
updates before they can mutate session state.

### 5.2 Stage0 KV and scheduler pressure

The existing resumable scheduler request remains the owner of Stage0 model KV.
Multiple logical sessions can be waiting while only `max_num_seqs` requests run.
No serving semaphore serializes complete sessions.

Remote RED tests must determine whether the current vLLM preemption path can
recompute a parked resumable request without losing append position or model
context. If it cannot, the implementation must add a model-owned hibernation
snapshot/replay path before claiming scheduler-managed unbounded sessions. It
is not acceptable to disable preemption, pin every session's KV, or silently
restart a conversation.

### 5.3 Stage1 talker and Token2Wav

The Stage1 key becomes a typed stream identity:

```text
(session_id, incarnation, epoch, turn_id, response_seq)
```

`_TalkerTurnState` owns:

- talker `past_key_values` and text cursor;
- pending audio tokens and transcript text;
- prompt WAV identity;
- Token2Wav `stream_cache`;
- Token2Wav `hift_cache_dict`;
- continuity and consumed-token cursors.

The shared tokenizer/vocoder object currently exposes mutable cache fields.
The implementation wraps each vocoder call in a minimal model-local critical
section:

1. install the selected response state's caches;
2. run one Token2Wav streaming window;
3. capture updated caches back into that response state;
4. clear the shared object's cache fields before releasing the lock.

This lock protects one non-reentrant model call. It must not cover a complete
response, turn, session, scheduler wait, or WebSocket operation. Thus Stage1
may interleave windows from different responses without corrupting continuity.

Closing one response removes only its state. Global Token2Wav reset is forbidden
while another response state exists.

## 6. Backpressure and fairness

The serving layer enforces per-session pending-input limits from typed config.
It returns `input_backpressure` before accepting an event that would exceed the
byte or pending-turn bound. Accepted PCM is never truncated or discarded.

Backpressure for one session does not prevent other sessions from appending or
receiving output. The implementation does not add a duplex-specific global
admission queue. Scheduler waiting order and preemption remain the source of
compute fairness in this checkpoint.

Per-session output journals have independent byte accounting. Journal overflow
terminates only the affected response and produces `session.resync_required`.

## 7. Configuration and capabilities

The fixed `max_native_duplex_sessions` admission field and its serving checks
are removed. Before deletion, repository-wide tests must prove that no other
pipeline relies on the field; any remaining compatibility-only reference is
removed rather than retained as a dormant false capability. A client cannot
reintroduce a capacity limit or override server lease/buffer limits through
`session.update` or `extra_body`.

The MiniCPM native capability response becomes:

```text
supports_multi_session=true
supports_multi_session_same_replica=true
supports_session_lease=true
supports_session_resume=true
session_admission_mode="scheduler_managed"
supports_kv_lease=false
supports_core_kv_lease=false
```

The first four positive fields become true only after the required H20 engine
and public-protocol E2E tests pass on the exact final code tree.

## 8. Protocol behavior

New client events:

- `session.resume`;
- `session.heartbeat`;
- `session.event_ack`.

New server events:

- `session.resumed` with the rotated resume token;
- `session.heartbeat_ack`;
- `session.replaced`;
- `session.expired` when an attached client can still receive it;
- `session.resync_required` for unrecoverable journal gaps.

All errors include a stable code and the relevant session ID. Token values are
excluded from `repr`, structured logs, exceptions, and debug dumps.

## 9. Failure semantics

- Invalid token: reject resume without touching lease or the existing
  attachment.
- Concurrent valid resume: exactly one attachment generation wins; losing
  attempts receive `session_resume_conflict`.
- Resume after expiry: return `session_expired`; do not recreate implicitly.
- Recoverable scheduler resource error: fail the affected operation explicitly
  while keeping the session if its context remains valid.
- Context-invalidating scheduler/preemption failure: close the affected session
  with a typed reason; never continue with partial context.
- Engine fatal error: close all affected sessions and revoke credentials.
- Reaper/close race: one terminal cleanup transaction, idempotent followers.
- Late output: rejected by incarnation/fence before protocol or journal commit.

## 10. Test and validation requirements

All pytest, runtime, E2E, ASR, and audio-quality validation for this checkout
runs on the remote H20. The local Mac is limited to editing, static checks, git,
and sync preparation.

### 10.1 RED/GREEN unit and integration tests

- runtime manager accepts multiple identities and isolates leases/resources;
- monotonic touch, detach, resume, expiry, and terminal races;
- token hashing, constant-time validation, rotation, and takeover;
- event sequence, ACK, replay, TTL, byte bound, and resync overflow;
- input byte/turn backpressure without cross-session blocking;
- two Stage0 sessions alternate chunks without processor or audio-KV leakage;
- stale Stage0 fence cannot mutate the current session;
- two Stage1 responses alternate Token2Wav windows without cache crossover;
- closing one response/session preserves the other;
- ordinary non-duplex paths do not allocate duplex state;
- capability and configuration serialization.

### 10.2 Engine E2E

Using one Stage0/Stage1 replica pair:

1. open session A and B through the transport-neutral engine client;
2. append alternating distinct PCM chunks;
3. commit different semantic prompts and secrets;
4. observe independent model decisions and output fences;
5. close/expire A while B continues;
6. verify B retains its own context and emits no A transcript/audio;
7. verify all A scheduler/model state is removed.

### 10.3 Public two-WebSocket E2E

- two concurrent real-time-paced 16 kHz WAV inputs without transcript hints;
- both sessions remain active on the same replica pair;
- response IDs, transcript, audio, playback cursors, and history never cross;
- disconnect and resume one session during an active response;
- replay missing response events and rotate the token;
- the other session continues uninterrupted;
- expire one detached session and reject its old token/fence;
- output WAV files are valid 24 kHz mono audio;
- Whisper ASR is non-empty and semantically matches each session's transcript.

Model-policy tests do not require one response per physical input. Separate
response-required fixtures enforce exact spoken-response lifecycle where exact
cardinality is required.

### 10.4 Pressure matrix

Run 2, 4, 8, 16, and 32 logical sessions on H20, subject to safe remote GPU
availability. Record scheduler waiting, preemption/recompute, TTFT, output
completion, OOM, stale output, and cross-session contamination. A tested scale
is evidence for that deployment, not a code-level maximum.

### 10.5 Completion evidence

Completion requires all of:

- local changed-file Ruff, format, Python 3.10/3.11 compile, and diff checks;
- local/remote SHA-256 parity for every changed runtime/test file;
- remote affected pytest matrix;
- remote engine two-session E2E;
- remote public two-WebSocket reconnect/TTL E2E;
- remote pressure matrix with explicit failed scales, if any;
- valid output WAV metadata and ASR evidence;
- no traceback, engine death, leaked service process, or occupied test GPU;
- capability statements updated to match exactly what final evidence proves.

## 11. Implementation order

1. Add RED tests for typed lease state and control messages.
2. Implement runtime-manager touch/resume/reaper and correlated lifecycle
   cleanup.
3. Add RED tests for serving credentials, attachment takeover, journal, and
   backpressure.
4. Implement serving composition and replace disconnect-close behavior.
5. Add RED tests for Stage0 processor isolation and Stage1 vocoder cache
   interleaving.
6. Implement model state isolation and cleanup hooks.
7. Remove fixed native session admission and update typed configuration and
   capabilities.
8. Run remote focused suites, then the affected matrix.
9. Run engine and public E2E, pressure tests, WAV checks, and ASR.
10. Fix every discovered correctness defect and repeat the relevant RED/GREEN
    and E2E evidence on the final tree.
