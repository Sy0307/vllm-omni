# MiniCPM-o Realtime Terminal Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the final 5 ms drain fade with a terminal-only 30 ms fade and 20 ms silent post-roll.

**Architecture:** Keep the worklet's existing 5 ms start/underrun transition unchanged. Add independent terminal fade and silence frame budgets; delay `playback-drained` until the silence budget is consumed while keeping `playedMs` limited to generated PCM.

**Tech Stack:** Browser AudioWorklet JavaScript, Node VM regression harness, pytest.

---

### Task 1: Define the terminal release contract

**Files:**
- Modify: `tests/examples/test_minicpmo_realtime_web_static.py`

- [ ] **Step 1: Replace the existing terminal drain regression with the approved contract**

Use a 1000 Hz synthetic worklet, enqueue 100 constant frames, drain, and assert
that the last 30 frames fade monotonically to zero. Assert that no
`playback-drained` message is sent until two additional 10-frame zero render
blocks consume the 20 ms post-roll, and that `playedMs` remains 100.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
python3 -m pytest -q --noconftest tests/examples/test_minicpmo_realtime_web_static.py -k terminal_release
```

Expected: FAIL because the current worklet still uses a 5 ms terminal fade and
reports drain immediately.

### Task 2: Implement the terminal-only release

**Files:**
- Modify: `examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js`

- [ ] **Step 1: Add independent terminal frame budgets**

Keep `fadeFrames` at 5 ms. Add a 30 ms terminal fade window, a 20 ms terminal
silence budget, and a remaining-silence counter.

- [ ] **Step 2: Apply the budgets only during drain**

Initialize the terminal fade and post-roll when a drain has buffered or already
played response audio. Consume post-roll as zero output after the PCM queue
empties. Do not increment `playedFrames` for silence.

- [ ] **Step 3: Delay and reset drain completion**

Make `notifyIfDrained()` wait for the post-roll counter. Reset both terminal
budgets on `clear` and after sending the single drain notification.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run:

```bash
python3 -m pytest -q --noconftest tests/examples/test_minicpmo_realtime_web_static.py -k terminal_release
```

Expected: `1 passed`.

### Task 3: Verify and deploy

**Files:**
- Verify: `tests/examples/test_minicpmo_realtime_web_static.py`
- Verify: `examples/online_serving/minicpmo/realtime_web/app/static/app.js`
- Verify: `examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js`

- [ ] **Step 1: Run the complete local checks**

```bash
python3 -m pytest -q --noconftest tests/examples/test_minicpmo_realtime_web_static.py
node --check examples/online_serving/minicpmo/realtime_web/app/static/app.js
node --check examples/online_serving/minicpmo/realtime_web/app/static/playback_worklet.js
git diff --check
```

Expected: all checks pass.

- [ ] **Step 2: Commit and push the exact branch**

Commit the spec, plan, regression, and implementation, then push
`codex/main-brutalist-demo-20260730` to `sy0307`.

- [ ] **Step 3: Update only the remote frontend**

Reset the clean remote worktree to the pushed SHA. Keep backend PID `941773`
running, restart only the frontend on port 7867, and confirm `/healthz`.

- [ ] **Step 4: Run remote checks and audio-required E2E**

Run the same static/Node checks remotely, followed by
`realtime_duplex_demo.py` through `ws://127.0.0.1:7867/v1/realtime?duplex=1`
using `response_required_16k.wav`, `HT_ref_audio.wav`, and `--require-audio`.

Expected: `ok: true`, `model_decision: speak`, non-empty audio,
`response.audio.done`, `response.done`, and no errors.
