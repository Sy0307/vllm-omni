# MiniCPM-o Realtime UI Motion Design

## Summary

Refresh the MiniCPM-o 4.5 Realtime browser demo without changing its
low-contrast brutalist character or its audio behavior. The final design uses:

- the **Ash Indigo** color system;
- an editorial system-font typography stack;
- the **Duplex Rail** microphone and session-status component;
- **Quiet Response** motion that reacts only to meaningful state changes; and
- a client-observed generation timer attached to every assistant reply.

The Event Log remains collapsed by default. The existing 1-second playback
buffering, underrun recovery, tail draining, terminal fade, WebSocket protocol,
and backend behavior remain unchanged.

## Goals

1. Make the microphone signal and current duplex state immediately legible.
2. Give the page restrained movement without adding decorative clutter.
3. Improve typography while avoiding external font or CDN dependencies.
4. Record the streaming generation duration of every assistant response.
5. Preserve current audio, session, prompt, camera, and diagnostics behavior.
6. Keep the page suitable for a screen-recorded promotional video.

## Non-goals

- Do not change the Realtime protocol or backend event schema.
- Do not change input chunking, output chunking, playback buffering, or
  playback acknowledgement.
- Do not add analytics, telemetry, local storage, cookies, or persistence
  across page reloads.
- Do not display an estimated user-perceived first-response latency when the
  continuous full-duplex input has no reliable utterance-end timestamp.
- Do not add external font downloads, gradients, bright accents, rounded
  cards, glass effects, or decorative ambient animation.
- Do not expand the Event Log by default.

## Typography

Use three related system-font roles:

```css
--font-display:
  "Avenir Next", "Helvetica Neue", system-ui, sans-serif;
--font-interface:
  -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
  "Hiragino Sans GB", sans-serif;
--font-data:
  "SFMono-Regular", "Cascadia Code", Menlo, Consolas, monospace;
```

- Display: page title and section headings, with a tighter but less extreme
  letter spacing than the current Arial treatment.
- Interface: conversation text, buttons, selects, and explanatory copy.
- Data: statuses, timers, microphone metadata, role labels, and Event Log.

The stack must degrade cleanly on non-macOS clients. No font asset is added.

## Ash Indigo Color System

The page uses the following semantic tokens:

```css
--ash-page: #e4e6ea;
--ash-surface: #f1f2f4;
--ash-ink: #353b46;
--ash-muted: #69707c;
--ash-line: #606875;
--ash-control: #c4c8d0;
--ash-primary: #939faf;
--ash-title-accent: #68778f;
--ash-wave: #8390a3;
--ash-online: #979fa8;
--ash-online-bg: #d5d9df;
--ash-active: #9a8f9c;
--ash-warning: #aea493;
--ash-error: #aa9196;
--ash-footer: #58616d;
--ash-footer-ink: #edf0f3;
```

All controls remain square and bordered. The palette must stay low saturation:
state changes use token substitution, not bright semantic colors.

## Duplex Rail

Replace the existing horizontal Mic fill bar and separate runtime-status list
with a single compact `Duplex Rail`.

### Structure

1. **Microphone glyph**
   - Square bordered tile.
   - Uses an inline accessible SVG; no icon dependency.
   - Background reflects open, muted, and offline state.

2. **Signal area**
   - Header: `Your microphone`.
   - Metadata: input sample rate and current session timer.
   - A row of narrow vertical bars driven by the real PCM input envelope.
   - Bars use Ash Indigo wave and inactive colors.

3. **Duplex state**
   - Primary label reuses the existing model state:
     `Listening`, `Speaking`, `Connecting`, or `Idle`.
   - Secondary label compresses mic/playback state:
     `Mic open`, `Mic muted`, `Buffering`, or `Playing`.

### Signal behavior

- `updateMeter(pcm)` continues sampling PCM data rather than creating a
  decorative waveform.
- Convert the sampled peak to a normalized level in `[0, 1]`.
- Activate an integer number of bars from left to right.
- Apply a short attack and slower release so the signal is readable without
  flicker.
- Offline and muted states immediately settle to zero.
- The bar count and DOM updates stay small enough to avoid affecting capture or
  playback scheduling.

### Responsive behavior

- Desktop: glyph, signal, and duplex state share one row.
- Narrow viewports: duplex state moves below the signal area while remaining
  inside the same bordered rail.
- Runtime DOM IDs remain available to current JavaScript and tests.

## Quiet Response Motion

Motion communicates state and hierarchy; it is not ambient decoration.

### Motion inventory

- Page frame: one 650–700 ms fade-and-rise on initial load.
- Online indicator: low-frequency 1.8-second opacity/scale breath.
- Conversation turns: 650 ms fade and 8 px rise when first created.
- Buttons: 1 px press displacement and matching shadow reduction.
- Duplex signal: responds to live PCM input; no motion when there is no signal.
- Live transcript cursor: preserve the existing restrained blink.

Do not animate layout dimensions, conversation scrolling, the Event Log, or the
whole page continuously.

### Reduced motion

Under `prefers-reduced-motion: reduce`:

- disable page and turn entrance animation;
- disable status breathing;
- disable button transitions;
- retain instantaneous microphone level changes because they are functional
  feedback, not decorative motion.

## Per-response Latency Timing

### Definition

The UI labels its browser-side turn-transition measurement `TTFT` and displays
it together with the full response duration. In this demo, TTFT has the
following concrete client-observed definition:

```text
VAD end             = client PCM energy stays below the end threshold for 420 ms
Speaking            = the UI switches its duplex state to Speaking
full-response start = browser receives input_audio_buffer.committed
fully complete      = browser receives response.done
```

The VAD arms only after at least 160 ms of speech. It uses RMS hysteresis with a
higher speech-start threshold and a lower speech-end threshold. The timestamp
starts when the 420 ms silence decision is confirmed, not at the first quiet
sample. If no valid local VAD end precedes Speaking, the metric says
`unavailable` instead of presenting a misleading zero.
An unconsumed endpoint expires after 12 seconds, preventing a later response
from reusing stale speech timing.

If a response begins without a preceding committed event, full-response timing
uses the first response event as a defensive fallback start. These values are
client observations rather than server-only model-generation metrics.

### Display states

- Waiting for state change: `TTFT · waiting / Responding · 0.8s`
- Speaking: `TTFT · 0.6s / Responding · 3.8s`
- Completed: `TTFT · 0.6s / Fully completed · 8.2s`
- Cancelled/interrupted: `TTFT · 0.6s / Interrupted · 2.6s`
- No valid VAD endpoint: `TTFT · unavailable / Failed · 1.4s`

The metadata appears below the assistant text in enlarged data typography. It
does not create a separate dashboard row.

### State ownership

- Track timing state by `response_id`.
- Start timing on `response.created`.
- Track local VAD continuously from real microphone PCM without changing or
  gating the uploaded audio.
- Consume the latest valid VAD-end timestamp when the UI enters Speaking.
- If an implementation receives a response-associated visible output before
  `response.created`, create a fallback timer at that first visible event and
  replace its ownership when the response ID becomes available.
- Update the visible active value at no more than 10 Hz.
- Freeze and clear the active timer on `response.done`.
- A `response.listen` decision creates no assistant row and no timer.
- Clearing or stopping the session cancels timer updates.
- Timing history lives only in the current conversation DOM and in-memory
  state; page reload clears it.

## Conversation Presentation

- Preserve the editorial row layout rather than introducing chat bubbles.
- Keep role labels in the left column and content in the right column.
- Attach timing metadata to assistant rows only.
- Preserve incremental transcript updates and current scroll-to-latest
  behavior.
- A timer must not cause the conversation container to scroll on every update.

## Accessibility

- Keep existing `aria-live` behavior for conversation content.
- Mark continuously updating timer text `aria-hidden="true"` while a response
  is active to prevent repetitive announcements.
- On completion, expose one stable accessible label such as
  `Response completed in 8.2 seconds`.
- Give the Duplex Rail an accessible name and expose the current microphone
  and duplex state through stable text.
- Preserve keyboard focus outlines and native button/select semantics.
- All iconography must have text equivalents.

## Files in Scope

- `examples/online_serving/minicpmo/realtime_web/app/index.html`
- `examples/online_serving/minicpmo/realtime_web/app/static/styles.css`
- `examples/online_serving/minicpmo/realtime_web/app/static/app.js`
- `tests/examples/test_minicpmo_realtime_web_static.py`

AudioWorklet files, the FastAPI proxy, Realtime backend code, deployment YAML,
and documentation outside this design record are out of scope.

## Validation

### Static and unit checks

- HTML contains the Duplex Rail and response-timing anchors.
- CSS contains the Ash Indigo tokens, typography roles, Quiet Response motion,
  and reduced-motion handling.
- JavaScript syntax checks pass.
- Static demo tests cover:
  - Event Log default collapsed state;
  - 1-second playback buffering contract;
  - Duplex Rail state anchors;
  - response-ID timing lifecycle;
  - completed/interrupted timing labels; and
  - no persistence or telemetry addition.

### Runtime checks

- Start the existing MiniCPM-o Realtime backend unchanged.
- Restart only the demo frontend after updating the remote worktree.
- Confirm the browser shows Ash Indigo, Duplex Rail, and collapsed Event Log.
- Complete a real audio-required WebSocket response through the frontend proxy.
- Confirm:
  - microphone bars respond to live input;
  - Listening/Speaking/Muted state transitions are correct;
  - each assistant reply receives exactly one timing record;
  - the timer freezes at `response.done`;
  - the final sub-second audio tail still plays; and
  - no new WebSocket or playback error is logged.

## Delivery

- Commit implementation on `codex/main-brutalist-demo-20260730`.
- Push the updated branch to the existing `sy0307` remote.
- Fast-forward the isolated remote worktree to the exact pushed SHA.
- Keep the MiniCPM backend running on its current H20.
- Restart only the known frontend task on port `7867`.
- Return the unchanged public Demo URL and the exact implementation SHA.
