# MiniCPM-o Realtime UI Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved Ash Indigo MiniCPM-o Realtime page with Duplex Rail microphone feedback, Quiet Response motion, and per-assistant-response generation timing.

**Architecture:** Keep the existing static HTML/CSS/JavaScript application and Realtime protocol unchanged. Replace the runtime-row presentation with a Duplex Rail, drive its bars from the existing PCM capture callback, and add a response-ID keyed browser timer that annotates existing assistant turns. Preserve all playback and WebSocket behavior.

**Tech Stack:** HTML5, CSS custom properties and media queries, browser DOM APIs, Web Audio PCM capture, OpenAI Realtime-style WebSocket events, pytest static contract tests, Node syntax checks.

---

## File Map

- Modify `examples/online_serving/minicpmo/realtime_web/app/index.html`
  - Owns the Duplex Rail semantic structure and stable DOM anchors.
- Modify `examples/online_serving/minicpmo/realtime_web/app/static/styles.css`
  - Owns Ash Indigo tokens, typography, Duplex Rail layout, Quiet Response
    motion, response-timing metadata, responsive behavior, and reduced motion.
- Modify `examples/online_serving/minicpmo/realtime_web/app/static/app.js`
  - Owns PCM-to-bar rendering, compressed duplex state, and response timing.
- Modify `tests/examples/test_minicpmo_realtime_web_static.py`
  - Locks the approved static and lifecycle contracts without changing backend
    or audio-worklet tests.

## Task 1: Add failing contracts for the approved design

**Files:**

- Modify: `tests/examples/test_minicpmo_realtime_web_static.py`
- Test: `tests/examples/test_minicpmo_realtime_web_static.py`

- [ ] **Step 1: Rename the layout test and add Duplex Rail assertions**

Replace the first test with:

```python
def test_page_exposes_ash_indigo_duplex_rail_and_collapsed_log():
    html = (APP_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'class="app-shell"' in html
    assert 'id="callButton"' in html
    assert 'id="muteButton"' in html
    assert 'id="connectionState"' in html
    assert 'id="microphoneRail"' in html
    assert 'id="micBars"' in html
    assert 'id="modelState"' in html
    assert 'id="playbackState"' in html
    assert 'id="sessionTimer"' in html
    assert 'id="conversation"' in html
    assert 'id="promptEditor"' in html
    assert 'id="toggleLogButton"' in html
    assert 'aria-controls="eventLogPanel"' in html
    assert 'aria-expanded="false"' in html
    assert re.search(r'id="eventLogPanel"[^>]*\bhidden\b', html)
    assert 'id="eventLog"' in html
    assert "<details" not in html
    assert "Automatic barge-in" not in html
    assert "Server VAD" not in html
```

- [ ] **Step 2: Replace Fog Blue style expectations with Ash Indigo, motion, and typography contracts**

Replace `test_stylesheet_uses_fog_blue_tokens_and_responsive_shell` with:

```python
def test_stylesheet_uses_ash_indigo_typography_motion_and_responsive_shell():
    source = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

    assert "--ash-page: #e4e6ea;" in source
    assert "--ash-control: #c4c8d0;" in source
    assert "--ash-primary: #939faf;" in source
    assert '--font-display: "Avenir Next"' in source
    assert "--font-interface:" in source
    assert "--font-data:" in source
    assert ".duplex-rail" in source
    assert ".mic-bar.is-active" in source
    assert ".turn-response-meta" in source
    assert "@keyframes page-arrive" in source
    assert "@keyframes status-breathe" in source
    assert "@keyframes turn-arrive" in source
    assert "@media (prefers-reduced-motion: reduce)" in source
    assert "@media (max-width: 760px)" in source
    assert "[hidden]" in source
```

- [ ] **Step 3: Add response-timing and signal-rendering contracts**

Add:

```python
def test_client_tracks_pcm_bars_and_per_response_generation_duration():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "const micBars = Array.from" in source
    assert "function updateMeter(pcm)" in source
    assert "classList.toggle('is-active'" in source
    assert "const responseTimings = new Map();" in source
    assert "function startResponseTiming(responseId)" in source
    assert "function finishResponseTiming(responseId, status)" in source
    assert "performance.now()" in source
    assert "Responding ·" in source
    assert "Completed ·" in source
    assert "Interrupted ·" in source
    assert "aria-hidden" in source
    assert "localStorage" not in source
    assert "sendBeacon" not in source
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/examples/test_minicpmo_realtime_web_static.py
```

Expected: the existing tests pass, while the three changed/new design tests fail
because `microphoneRail`, Ash Indigo tokens, and response-timing functions do
not exist.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/examples/test_minicpmo_realtime_web_static.py
git commit -m "test(minicpmo): define realtime UI motion contracts"
```

## Task 2: Implement Ash Indigo, typography, Duplex Rail, and Quiet Response

**Files:**

- Modify: `examples/online_serving/minicpmo/realtime_web/app/index.html`
- Modify: `examples/online_serving/minicpmo/realtime_web/app/static/styles.css`
- Test: `tests/examples/test_minicpmo_realtime_web_static.py`

- [ ] **Step 1: Replace the runtime row with the Duplex Rail**

Replace the current `.runtime-row` block in `index.html` with:

```html
<section
  id="microphoneRail"
  class="duplex-rail"
  aria-label="Microphone and duplex status"
>
  <div class="mic-glyph" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none">
      <rect x="8" y="3" width="8" height="12" rx="4"></rect>
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M8.5 21h7"></path>
    </svg>
  </div>
  <div class="mic-signal">
    <div class="mic-signal-heading">
      <strong>Your microphone</strong>
      <span><span id="captureRate">16 kHz</span> · <span id="sessionTimer">00:00</span></span>
    </div>
    <div id="micBars" class="mic-bars" aria-hidden="true">
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
      <i class="mic-bar"></i><i class="mic-bar"></i><i class="mic-bar"></i>
    </div>
  </div>
  <div class="duplex-state">
    <strong id="modelState">Idle</strong>
    <span id="playbackState">Mic closed</span>
  </div>
</section>
```

- [ ] **Step 2: Replace the token and base typography declarations**

At the beginning of `styles.css`, use:

```css
:root {
  color-scheme: light;
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
  --font-display: "Avenir Next", "Helvetica Neue", system-ui, sans-serif;
  --font-interface: -apple-system, BlinkMacSystemFont, "Segoe UI",
    "PingFang SC", "Hiragino Sans GB", sans-serif;
  --font-data: "SFMono-Regular", "Cascadia Code", Menlo, Consolas, monospace;
  color: var(--ash-ink);
  background: var(--ash-page);
  font-family: var(--font-interface);
  font-synthesis: none;
}
```

Apply this exact token mapping throughout the existing stylesheet:

```text
--fog-page         -> --ash-page
--fog-surface      -> --ash-surface
--fog-ink          -> --ash-ink
--fog-muted        -> --ash-muted
--fog-line         -> --ash-line
--fog-control      -> --ash-control
--fog-primary      -> --ash-primary
--fog-active       -> --ash-active
--fog-online       -> --ash-online
--fog-warning      -> --ash-warning
--fog-error        -> --ash-error
```

Set `h1`, `h2`, and `.section-heading` to `var(--font-display)`. Set `.eyebrow`,
`.status`, `.prompt-control`, `.prompt-editor`, `.turn-role`, `.policy-label`,
`.log-meta`, `.log-toggle`, `.log-toolbar`, `.event-log`, and Duplex Rail
metadata to `var(--font-data)`.

- [ ] **Step 3: Add Duplex Rail styles**

Add:

```css
.duplex-rail {
  min-height: 70px;
  padding: 13px 16px;
  border-bottom: 1px solid color-mix(in srgb, var(--ash-line) 45%, var(--ash-surface));
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  align-items: center;
  gap: 14px;
}

.mic-glyph {
  width: 38px;
  height: 38px;
  border: 1px solid var(--ash-line);
  background: var(--ash-online-bg);
  display: grid;
  place-items: center;
}

.mic-glyph svg {
  width: 18px;
  height: 18px;
  stroke: var(--ash-line);
  stroke-width: 1.7;
}

.mic-signal-heading {
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--ash-muted);
  font: 750 8px/1 var(--font-data);
  text-transform: uppercase;
}

.mic-signal-heading strong {
  color: var(--ash-ink);
  font-size: 10px;
}

.mic-bars {
  height: 18px;
  display: flex;
  align-items: center;
  gap: 3px;
  overflow: hidden;
}

.mic-bar {
  width: 4px;
  height: 4px;
  background: color-mix(in srgb, var(--ash-wave) 32%, var(--ash-surface));
  transition: height 90ms ease, background-color 90ms ease, opacity 90ms ease;
}

.mic-bar.is-active {
  background: var(--ash-wave);
  opacity: 1;
}

.duplex-state {
  min-width: 92px;
  padding-left: 14px;
  border-left: 1px solid color-mix(in srgb, var(--ash-line) 45%, var(--ash-surface));
  text-align: right;
}

.duplex-state strong,
.duplex-state span {
  display: block;
  font-family: var(--font-data);
  text-transform: uppercase;
}

.duplex-state strong {
  color: var(--ash-title-accent);
  font-size: 10px;
}

.duplex-state span {
  margin-top: 6px;
  color: var(--ash-muted);
  font-size: 7px;
}
```

- [ ] **Step 4: Add Quiet Response motion and reduced-motion overrides**

Add:

```css
.app-shell { animation: page-arrive 680ms cubic-bezier(.2,.75,.25,1) both; }
.status-online::before { animation: status-breathe 1.8s ease-in-out infinite; }
.turn { animation: turn-arrive 650ms cubic-bezier(.2,.75,.25,1) both; }

button:not(:disabled) {
  transition: transform 90ms ease, box-shadow 90ms ease, background-color 140ms ease;
}

button:not(:disabled):active {
  transform: translate(1px, 1px);
  box-shadow: 1px 1px 0 var(--ash-line);
}

@keyframes page-arrive {
  from { opacity: 0; transform: translateY(8px); box-shadow: 0 0 0 transparent; }
}

@keyframes status-breathe {
  50% { opacity: .42; transform: scale(.82); }
}

@keyframes turn-arrive {
  from { opacity: 0; transform: translateY(8px); }
}

@media (prefers-reduced-motion: reduce) {
  .app-shell,
  .status-online::before,
  .turn {
    animation: none;
  }

  button:not(:disabled),
  .mic-bar {
    transition: none;
  }
}
```

Update the existing `@media (max-width: 760px)` rule so `.duplex-rail` becomes
two columns and `.duplex-state` spans both columns below the signal.

- [ ] **Step 5: Run focused tests**

```bash
python3 -m pytest -q tests/examples/test_minicpmo_realtime_web_static.py
```

Expected: Ash Indigo, typography, HTML, and CSS assertions pass; response
timing JavaScript assertions still fail.

- [ ] **Step 6: Commit the visual component**

```bash
git add \
  examples/online_serving/minicpmo/realtime_web/app/index.html \
  examples/online_serving/minicpmo/realtime_web/app/static/styles.css
git commit -m "feat(minicpmo): refresh realtime voice interface"
```

## Task 3: Implement PCM bars and per-response timing

**Files:**

- Modify: `examples/online_serving/minicpmo/realtime_web/app/static/app.js`
- Modify: `examples/online_serving/minicpmo/realtime_web/app/static/styles.css`
- Test: `tests/examples/test_minicpmo_realtime_web_static.py`

- [ ] **Step 1: Bind Duplex Rail elements and timing state**

Replace `meterFill` with:

```javascript
const microphoneRail = document.getElementById('microphoneRail');
const captureRateLabel = document.getElementById('captureRate');
const micBars = Array.from(document.querySelectorAll('#micBars .mic-bar'));
```

Add alongside the session state:

```javascript
const RESPONSE_TIMER_INTERVAL_MS = 100;
const responseTimings = new Map();
let responseTimer = null;
let smoothedMicLevel = 0;
```

- [ ] **Step 2: Render the real PCM level as bars**

Replace `updateMeter` with:

```javascript
function renderMicLevel(level) {
  const activeCount = Math.round(Math.max(0, Math.min(1, level)) * micBars.length);
  micBars.forEach((bar, index) => {
    const position = index / Math.max(1, micBars.length - 1);
    bar.classList.toggle('is-active', index < activeCount);
    bar.style.height = `${Math.round(4 + position * 14)}px`;
  });
}

function updateMeter(pcm) {
  let peak = 0;
  for (let index = 0; index < pcm.length; index += 8) {
    peak = Math.max(peak, Math.abs(pcm[index]));
  }
  const nextLevel = Math.min(1, (peak / 32768) * 1.5);
  smoothedMicLevel = nextLevel > smoothedMicLevel
    ? nextLevel
    : Math.max(0, smoothedMicLevel * .82);
  renderMicLevel(muted ? 0 : smoothedMicLevel);
}
```

Call `renderMicLevel(0)` when stopping or muting. In `openSocket`, set
`captureRateLabel.textContent = `${Math.round(captureRate / 1000)} kHz``.

- [ ] **Step 3: Add response timing DOM and lifecycle helpers**

Add:

```javascript
function formatResponseDuration(durationMs) {
  return `${(Math.max(0, durationMs) / 1000).toFixed(1)}s`;
}

function ensureResponseMeta(turn) {
  if (turn.responseMeta) return turn.responseMeta;
  const meta = document.createElement('div');
  meta.className = 'turn-response-meta';
  meta.setAttribute('aria-hidden', 'true');
  turn.text.appendChild(meta);
  turn.responseMeta = meta;
  return meta;
}

function updateResponseTimers() {
  const now = performance.now();
  for (const timing of responseTimings.values()) {
    if (timing.finishedAt !== null) continue;
    ensureResponseMeta(timing.turn).textContent =
      `Responding · ${formatResponseDuration(now - timing.startedAt)}`;
  }
}

function startResponseTiming(responseId) {
  if (!responseId || responseTimings.has(responseId)) return;
  const turn = ensureTurn('assistant');
  const timing = {
    responseId,
    startedAt: performance.now(),
    finishedAt: null,
    turn,
  };
  turn.responseId = responseId;
  responseTimings.set(responseId, timing);
  ensureResponseMeta(turn).textContent = 'Responding · 0.0s';
  if (responseTimer === null) {
    responseTimer = window.setInterval(updateResponseTimers, RESPONSE_TIMER_INTERVAL_MS);
  }
}

function finishResponseTiming(responseId, status = 'completed') {
  const timing = responseTimings.get(responseId);
  if (!timing || timing.finishedAt !== null) return;
  timing.finishedAt = performance.now();
  const duration = formatResponseDuration(timing.finishedAt - timing.startedAt);
  const label = status === 'cancelled'
    ? 'Interrupted'
    : status === 'failed' ? 'Failed' : 'Completed';
  const meta = ensureResponseMeta(timing.turn);
  meta.textContent = `${label} · ${duration}`;
  meta.removeAttribute('aria-hidden');
  meta.setAttribute('aria-label', `Response ${label.toLowerCase()} in ${duration}`);
  if ([...responseTimings.values()].every((entry) => entry.finishedAt !== null)) {
    window.clearInterval(responseTimer);
    responseTimer = null;
  }
}

function clearResponseTimers() {
  if (responseTimer !== null) window.clearInterval(responseTimer);
  responseTimer = null;
  responseTimings.clear();
}
```

Update `ensureTurn` so assistant turns preserve the timing node:

```javascript
const turn = { row, text, value: '', responseMeta: null, responseId: null };
```

Update `addTranscript` and `finishTranscript` so they write assistant text into
a dedicated text node before the metadata element instead of assigning
`textContent` on the container.

- [ ] **Step 4: Wire Realtime events**

Change event handling to:

```javascript
case 'response.created':
case 'response.speak':
  beginAssistant(responseId);
  startResponseTiming(responseId);
  break;
case 'response.audio.delta':
  startResponseTiming(responseId);
  currentResponseId = responseId || currentResponseId;
  assistantActive = true;
  setModel('Speaking');
  setPlayback('Buffering');
  decodeAudioDelta(event)
    .then((decoded) => feedPlayback(decoded, responseId))
    .catch((error) => appendLog(`audio decode failed: ${error.message || error}`, true));
  break;
case 'response.done': {
  const status = event.response && event.response.status
    ? event.response.status
    : 'completed';
  finishTranscript('assistant');
  finishResponseTiming(responseId, status);
  if (!responseHasAudio) requestPlaybackDrain(responseId);
  break;
}
```

Call `clearResponseTimers()` and `renderMicLevel(0)` during `stopSession`.
Use this state helper from `setPlayback`, `toggleMute`, and session cleanup:

```javascript
function setMicDetail(label) {
  playbackState.textContent = label;
}

function syncMicDetail() {
  if (!running) setMicDetail('Mic closed');
  else if (muted) setMicDetail('Mic muted');
  else setMicDetail('Mic open');
}
```

`setPlayback('Buffering')` and `setPlayback('Playing')` still override the
secondary label while output audio is active. `playbackDrained` calls
`syncMicDetail()` instead of writing `Idle`.

- [ ] **Step 5: Add response metadata styles**

Add:

```css
.turn-response-meta {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ash-muted);
  font: 750 8px/1 var(--font-data);
  text-transform: uppercase;
}

.turn-response-meta::before {
  content: "";
  width: 14px;
  height: 1px;
  background: color-mix(in srgb, var(--ash-line) 58%, var(--ash-surface));
}

.turn-live .turn-response-meta::before {
  height: 5px;
  border: 1px solid var(--ash-line);
  background: var(--ash-primary);
  animation: response-pulse 900ms ease-in-out infinite;
}

@keyframes response-pulse {
  50% { opacity: .42; transform: scaleX(.65); transform-origin: left; }
}
```

Disable `response-pulse` in the existing reduced-motion media query.

- [ ] **Step 6: Run tests and syntax checks**

```bash
python3 -m pytest -q tests/examples/test_minicpmo_realtime_web_static.py
node --check examples/online_serving/minicpmo/realtime_web/app/static/app.js
node --check examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js
git diff --check
```

Expected: all focused tests pass, both Node checks exit zero, and diff check is
clean.

- [ ] **Step 7: Commit behavior**

```bash
git add \
  examples/online_serving/minicpmo/realtime_web/app/static/app.js \
  examples/online_serving/minicpmo/realtime_web/app/static/styles.css \
  tests/examples/test_minicpmo_realtime_web_static.py
git commit -m "feat(minicpmo): add duplex signal and reply timing"
```

## Task 4: Remote validation and Demo refresh

**Files:**

- No source changes expected.

- [ ] **Step 1: Run the complete focused remote suite**

Run on the isolated H20 worktree:

```bash
python3 -m pytest -q tests/examples/test_minicpmo_realtime_web_static.py
node --check examples/online_serving/minicpmo/realtime_web/app/static/app.js
node --check examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js
git diff --check origin/main...HEAD
```

Expected: all static tests pass, JavaScript syntax is valid, and diff check is
clean.

- [ ] **Step 2: Push and verify the exact branch SHA**

```bash
git push sy0307 codex/main-brutalist-demo-20260730
git rev-parse HEAD
git ls-remote sy0307 refs/heads/codex/main-brutalist-demo-20260730
```

Expected: the local and remote SHAs match.

- [ ] **Step 3: Update the clean remote worktree**

Run in the isolated remote worktree:

```bash
test -z "$(git status --porcelain)"
git fetch sy0307 codex/main-brutalist-demo-20260730
demo_target_sha="$(git rev-parse FETCH_HEAD)"
git reset --hard "$demo_target_sha"
test "$(git rev-parse HEAD)" = "$demo_target_sha"
```

Expected: the cleanliness checks exit zero and the detached remote worktree
points at the exact fetched branch SHA.

- [ ] **Step 4: Restart only the known frontend task**

Keep the existing MiniCPM backend on GPU 2 running. Stop only the known
Realtime web frontend task, then launch:

```bash
python3 -m examples.online_serving.minicpmo.realtime_web \
  --port 7867 \
  --ws-backend ws://127.0.0.1:28907 \
  --model /home/admin/workspace/aop_lab/modelscope_models/openbmb/MiniCPM-o-4_5 \
  --ref-audio /home/admin/workspace/aop_lab/modelscope_models/openbmb/MiniCPM-o-4_5/assets/HT_ref_audio.wav
```

Expected: `/healthz` returns `ok`; page, CSS, and JavaScript return HTTP 200.

- [ ] **Step 5: Run a real audio-required WebSocket smoke**

Use `realtime_duplex_demo.py` through the frontend proxy with the existing
response-required input and reference audio. Require audio output.

Expected:

- `ok: true`
- `model_decision: speak`
- `response.done_count: 1`
- non-empty audio
- no error events
- a final sub-second tail chunk is present when produced

- [ ] **Step 6: Inspect the browser**

Confirm:

- Ash Indigo palette and editorial typography are visible;
- Duplex Rail responds to microphone input;
- Listening, Speaking, Playing, and Muted states are legible;
- Event Log is collapsed by default;
- assistant timing transitions from `Responding` to `Completed`; and
- the public Demo URL remains unchanged.
