# Stage Payload Contract

## Scope

This contract governs ordinary data-plane messages between two pipeline
stages. It is shared by full-payload and async-chunk connectors and is not a
Full-Duplex lifecycle API. Abort, cancellation, session admission, and resource
leases remain on their reliable control planes.

## Ownership

The transport and model stores have deliberately different owners:

- `StagePayloadEnvelope` is the immutable cross-stage transfer message. Its
  route, identity, request-global sequence, payload, and boundary cannot be
  changed after enqueue. Contained tensors are read-only by contract.
- `StagePayloadAccumulator` is destination-local transport state. It validates
  the edge schema and atomically applies an accepted envelope exactly once.
- `model_intermediate_buffer` is mutable runner/model state. Connector receive
  threads do not mutate it directly.
- `additional_information` is a compatibility projection for existing model
  inputs. It is not the long-term inter-stage transport owner.

## Wire contract

`StagePayloadEnvelope` contains:

- `schema_version` (currently exactly `1`);
- one directed `StageRoute`;
- stable external correlation identity plus the source Stage's internal ID;
- optional Full-Duplex `session_fence`;
- request-global `chunk_seq`;
- optional `OmniPayloadStruct` data;
- `NONE`, `SEGMENT_END`, or `STREAM_END` boundary.

Payload and boundary are independent. The last data chunk may also close a
segment or stream, and a boundary-only envelope is valid. An empty envelope
with `boundary=NONE` is invalid; processors use `NoPayloadYet` instead. Abort is
not a data boundary.

Sequence numbers count emitted envelopes, including boundary-only emissions.
`NoPayloadYet` consumes no sequence. A duplicate sequence is idempotently
ignored with no second payload projection or scheduler wake-up. A gap is a
terminal contract failure. Segment boundaries do not reset the sequence.

## Edge schema

The source edge owns one `StagePayloadSchema`; configuration resolution derives
the same schema path as `stage_payload_input_schema` for its unique destination.
The derived input field is not a second source of truth.

Every emitted field must be declared as:

- `DELTA`: append a true incremental tensor/list to destination transport
  state;
- `SNAPSHOT`: replace a cumulative snapshot;
- `REPLACE`: replace an opaque value or a self-contained processing window.

Tensor rank, dtype, and concatenation dimension are validated before connector
I/O and again before destination state commits. All fields in one envelope are
applied atomically. For example, MiniCPM-o 4.5 `codes.audio` is `REPLACE`, not
`DELTA`: every emission includes codec left-context overlap and must reach
Code2Wav as one self-contained window.

## Processor ABI and errors

A migrated processor accepts exactly one `StagePayloadBuildContext` and returns
`NoPayloadYet` or `PayloadEmission`. The context names normalized model output,
request-carried payload, and raw model output separately; a model-native raw
mapping that cannot become `OmniPayloadStruct` remains available as
`raw_output`.

There is no `PayloadError` result. Undeclared exceptions fail the request through
`StageTransferFailure`; they are never converted to an empty finish marker.
Schema/identity/sequence errors are deterministic and are not retried.
Connector transport errors retain bounded retries and fail the scheduler-local
internal request ID after exhaustion. External IDs are correlation data only.

## Compatibility window

For one release, receivers accept a legacy raw payload and legacy processors
run through an explicit adapter mode without signature inspection. Only the
legacy bridge interprets `meta.finished`, `meta.is_segment_finished`, and
`meta.override_keys` as generic transport/merge compatibility fields.
`stream_finished`, `last_chunk`, and model `chunk_seq` remain model metadata
when an explicit edge schema declares them.

A raw legacy Mapping stays untyped and follows the receiver's pre-envelope
projection rules. It does not gain envelope identity, sequence, deduplication,
or edge-schema guarantees; those guarantees begin when the producer emits a
`StagePayloadEnvelope`.

The legacy chunk adapter also reproduces its old transport-owned flag writes:
the real non-resumable request terminal state overwrites `meta.finished`, while
an explicitly false `is_segment_finished` may suppress an inferred segment
boundary. This prevents model-level codec turn completion from closing an
ongoing Full-Duplex stream.

Compatibility code must be removed only after all active producer and receiver
pairs use envelopes, explicit schemas, and the context-only processor ABI.
