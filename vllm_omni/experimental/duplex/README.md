# Experimental Duplex Runtime

This package contains the unstable session-oriented MiniCPM-o duplex runtime
and its OpenAI Realtime adapter. Public APIs, event details, and internal state
contracts may change without compatibility guarantees.

The current MiniCPM-o 4.5 checkpoint covers:

- model-owned listen/speak decisions;
- automatic response creation after committed speech;
- clean multi-turn, no-barge audio conversations;
- response audio and transcript lifecycle events.

The following are not completion claims for this checkpoint:

- VAD-driven or explicit serving-side barge-in;
- bounded/windowed KV for minute-scale sessions;
- production concurrency and multi-session capacity.

Core scheduler, orchestrator, and model-runner files contain only the hooks
needed to call this package. New MiniCPM duplex behavior should remain under
this namespace until its contracts are stable enough to graduate.
