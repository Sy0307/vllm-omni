import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples" / "online_serving" / "minicpmo" / "realtime_web"
APP_ROOT = ROOT / "app"
STATIC_ROOT = APP_ROOT / "static"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def test_client_toggles_event_log_and_custom_prompt_visibility():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "function setLogExpanded(expanded)" in source
    assert "eventLogPanel.hidden = !expanded;" in source
    assert "toggleLogButton.setAttribute('aria-expanded', String(expanded));" in source
    assert "expanded ? 'Hide event log' : 'Show event log'" in source
    assert "function syncPromptEditorVisibility()" in source
    assert "promptEditor.hidden = promptPreset.value !== 'custom';" in source


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


def test_duplex_rail_preserves_active_playback_when_mute_state_changes():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "let playbackStatus = 'Idle';" in source
    assert "playbackStatus = label;" in source
    assert "if (playbackStatus !== 'Idle')" in source
    assert "setMicDetail(playbackStatus);" in source


def test_client_uses_proxy_relative_realtime_url_and_model_policy_session():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "new URL(config.realtimePath, window.location.href)" in source
    assert "url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'" in source
    assert "url.searchParams.set('autostart', '0')" in source
    assert "url.searchParams.set('minicpmo45_native_duplex', '1')" in source
    assert "auto_response: true" in source
    assert "input_audio_buffer.append" in source
    assert "input_audio_buffer.commit" not in source
    assert "playback.ack" in source
    assert "event.event || event" in source
    assert "response.audio.delta" in source
    assert "response.audio_transcript.delta" in source
    assert "conversation.item.input_audio_transcription" in source
    assert "force_barge_in" not in source
    assert "server_vad" not in source
    assert "type: 'response.create'" not in source
    assert 'type: "response.create"' not in source


def test_web_server_requires_ref_audio_for_audio_output_session():
    source = (ROOT / "server.py").read_text(encoding="utf-8")

    ref_audio_arg = re.search(
        r"parser\.add_argument\(\s*\"--ref-audio\",(?P<body>.*?)\n\s*\)",
        source,
        re.DOTALL,
    )
    assert ref_audio_arg is not None
    assert "required=True" in ref_audio_arg.group("body")
    assert "Optional reference voice" not in ref_audio_arg.group("body")


def test_client_sends_ref_audio_in_realtime_session_contract():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "if (config.refAudio) session.ref_audio = config.refAudio;" in source
    assert "extraBody.ref_audio" not in source


def test_client_has_transactional_cleanup_and_visible_event_logging():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "async function stopSession" in source
    assert "type: 'session.close'" in source
    assert "waitForSessionClosed" in source
    assert "SESSION_CLOSE_TIMEOUT_MS" in source
    assert "case 'session.closed':" in source
    assert "track.stop()" in source
    assert "clearInterval(sendTimer)" in source
    assert "appendEventLog(event)" in source
    assert "stopSession({ terminal: false })" in source


def test_client_keeps_microphone_upload_active_during_assistant_playback():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    upload_gate = re.search(r"function microphoneUploadEnabled\(\) \{(?P<body>.*?)\n  \}", source, re.DOTALL)
    begin_assistant = re.search(r"function beginAssistant\(responseId\) \{(?P<body>.*?)\n  \}", source, re.DOTALL)

    assert upload_gate is not None
    assert "return running && !muted;" in upload_gate.group("body")
    assert "assistantActive" not in upload_gate.group("body")
    assert begin_assistant is not None
    assert "pendingCapture = []" not in begin_assistant.group("body")


def test_audio_worklets_define_capture_and_playback_processors():
    capture = (STATIC_ROOT / "pcm_worklet.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "registerProcessor('fullduplex-pcm-capture'" in capture
    assert "Int16Array" in capture
    assert "registerProcessor('fullduplex-pcm-playback'" in playback
    assert "playback-drained" in playback
    assert "clear" in playback


def test_audio_worklet_urls_use_the_static_asset_version():
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '"appVersion": app_version' in server
    assert 'STATIC_DIR / "playback_worklet.js"' in server
    assert 'STATIC_DIR / "pcm_worklet.js"' in server
    assert "staticAssetUrl('static/playback_worklet.js')" in app
    assert "staticAssetUrl('static/pcm_worklet.js')" in app


def test_playback_worklet_buffers_first_one_second_and_reports_underruns():
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "INITIAL_PLAYBACK_BUFFER_MS = 1000" in app
    assert "initialBufferMs" in app
    assert "responseId" in app
    assert "playback-underrun" in app
    assert "underrunMs" in app
    assert "sampleRate * 1.0" in playback
    assert "initialBufferFrames" in playback
    assert "playback-underrun" in playback
    assert "underrunFrames" in playback
    assert "underrunMs" in playback


def test_playback_worklet_waits_before_playing_and_rebuffers_after_underrun():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        global.sampleRate = 1000;
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = { onmessage: null, postMessage: () => {} };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor();
        const render = () => {
          const output = new Float32Array(100);
          processor.process([], [[output]]);
          return output;
        };
        const assert = (condition, message) => {
          if (!condition) throw new Error(message);
        };

        const first = new Int16Array(999);
        first.fill(16384);
        processor.handleMessage({
          type: 'audio',
          pcm: first,
          responseId: 'response-1',
          initialBufferMs: 1000,
        });
        assert(!processor.started, 'sub-second first delta must remain buffered');
        assert(render().every((sample) => sample === 0), 'sub-second first delta must stay silent');

        const firstRemainder = new Int16Array(1);
        firstRemainder.fill(16384);
        processor.handleMessage({
          type: 'audio',
          pcm: firstRemainder,
          responseId: 'response-1',
          initialBufferMs: 1000,
        });
        const firstPlayback = render();
        assert(
          firstPlayback.some((sample) => sample !== 0),
          'playback must start as soon as one second is buffered',
        );

        for (let index = 0; index < 9; index += 1) render();
        const underrun = render();
        assert(!processor.started, 'an empty queue must return to buffering');
        assert(underrun[underrun.length - 1] === 0, 'underrun boundary must fade to zero');

        const resumed = new Int16Array(999);
        resumed.fill(8192);
        processor.handleMessage({
          type: 'audio',
          pcm: resumed,
          responseId: 'response-1',
          initialBufferMs: 1000,
        });
        assert(render().every((sample) => sample === 0), 'sub-second resume must stay buffered');

        const resumedRemainder = new Int16Array(1);
        resumedRemainder.fill(8192);
        processor.handleMessage({
          type: 'audio',
          pcm: resumedRemainder,
          responseId: 'response-1',
          initialBufferMs: 1000,
        });
        const resumedPlayback = render();
        assert(
          resumedPlayback.some((sample) => sample !== 0),
          'playback must resume as soon as one second is rebuffered',
        );
        assert(
          Math.abs(resumedPlayback[0]) < Math.abs(resumedPlayback[50]),
          'resumed playback must fade in instead of jumping from silence',
        );

        const tailProcessor = new Processor();
        const tail = new Int16Array(300);
        tail.fill(4096);
        tailProcessor.handleMessage({
          type: 'audio',
          pcm: tail,
          responseId: 'response-tail',
          initialBufferMs: 1000,
        });
        const renderTail = () => {
          const output = new Float32Array(100);
          tailProcessor.process([], [[output]]);
          return output;
        };
        assert(renderTail().every((sample) => sample === 0), 'short final tail must wait before drain');
        tailProcessor.handleMessage({ type: 'drain', responseId: 'response-tail' });
        assert(
          renderTail().some((sample) => sample !== 0),
          'drain must immediately release a final tail shorter than one second',
        );
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "playback_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_playback_worklet_uses_30ms_terminal_release_and_20ms_post_roll():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        global.sampleRate = 1000;
        const messages = [];
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = {
              onmessage: null,
              postMessage: (message) => messages.push(message),
            };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor();
        const pcm = new Int16Array(100);
        pcm.fill(16384);
        processor.handleMessage({
          type: 'audio',
          pcm,
          responseId: 'response-terminal',
          initialBufferMs: 400,
        });
        processor.handleMessage({
          type: 'drain',
          responseId: 'response-terminal',
        });

        const output = new Float32Array(100);
        processor.process([], [[output]]);
        const assert = (condition, message) => {
          if (!condition) throw new Error(message);
        };

        assert(output.some((sample) => sample !== 0), 'terminal audio must still play');
        assert(output[output.length - 1] === 0, 'terminal drain must fade the final sample to zero');
        assert(
          output[75] < output[70],
          'terminal drain must begin fading across the final 30 ms',
        );
        const tail = Array.from(output.slice(-30), Math.abs);
        assert(
          tail.every((sample, index) => index === 0 || sample <= tail[index - 1]),
          'terminal drain fade must decrease monotonically',
        );
        assert(
          messages.filter((message) => message.type === 'playback-drained').length === 0,
          'terminal drain must wait for the explicit post-roll',
        );

        const firstPostRoll = new Float32Array(10);
        processor.process([], [[firstPostRoll]]);
        assert(
          firstPostRoll.every((sample) => sample === 0),
          'terminal post-roll must contain only silence',
        );
        assert(
          messages.filter((message) => message.type === 'playback-drained').length === 0,
          'terminal drain must retain the full 20 ms post-roll',
        );

        const secondPostRoll = new Float32Array(10);
        processor.process([], [[secondPostRoll]]);
        const drained = messages.filter((message) => message.type === 'playback-drained');
        assert(
          drained.length === 1,
          'terminal drain must be reported exactly once after the 20 ms post-roll',
        );
        assert(
          drained[0].playedMs === 100,
          'synthetic post-roll must not increase acknowledged model-audio duration',
        );
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "playback_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )
