# MiniCPM-o Realtime Terminal Release Design

## Goal

Make completed speech responses end less abruptly without changing model
generation, chunking, initial buffering, or underrun recovery.

## Approved behavior

- Apply a 30 ms linear fade only to the final buffered PCM frames after a
  terminal `drain`.
- Emit 20 ms of explicit zero-valued post-roll before reporting
  `playback-drained`.
- Keep the existing 5 ms fade for playback start and underrun recovery.
- Do not count the synthetic post-roll in `playedMs`; playback acknowledgement
  continues to describe model-generated audio only.
- Reset terminal fade and post-roll state on `clear` and after a completed
  drain.

## Implementation boundary

The change is confined to
`examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js`.
The browser event flow in `app.js`, the 1000 ms initial/rebuffer threshold, the
backend, and the model are unchanged.

The worklet keeps separate frame counts for:

- the existing 5 ms transition used by start/underrun behavior;
- the new 30 ms terminal fade window;
- the remaining 20 ms terminal silence.

`playback-drained` is sent only when both the PCM queue and terminal silence are
empty.

## Validation

At a synthetic 1000 Hz sample rate, a constant 100-frame signal must:

1. remain at full scale before the final 30-frame window;
2. decrease monotonically through the terminal window and end at zero;
3. withhold `playback-drained` until exactly 20 zero frames have been rendered;
4. report `playback-drained` once, with `playedMs` excluding post-roll.

The complete realtime-web static suite and both JavaScript syntax checks must
remain green. A remote audio-required E2E must still produce non-empty audio,
`response.audio.done`, `response.done`, and no protocol errors.
