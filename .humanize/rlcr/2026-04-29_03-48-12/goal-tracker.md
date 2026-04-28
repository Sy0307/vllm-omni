# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
Make Qwen3 TTS NV acoustic decoding steps as fast as correctness allows by replacing per-token generic framework execution with a tightly scoped acoustic decode fast path that minimizes scheduler, Python, sampler, bookkeeping, CPU/GPU synchronization, and repeated small prefill overhead until remaining latency is dominated by unavoidable model computation.

Source plan: docs/design/qwen3_tts_nv_decoding_optimization.md

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

1. The existing measured acoustic inner-loop prototype remains available as the baseline fast path, with unit coverage for slot reservation and early-stop correction still passing.
2. The restricted Qwen3 TTS NV decode-only path avoids re-running the full generic single-token runner preparation stack for every acoustic substep while preserving KV/cache position, hidden state, sampled token, and code predictor state correctness across K steps.
3. The fast path supports greedy group-0 sampling without the vLLM sampler boundary, keeps sampled group-0 tokens GPU-resident during the K-step loop, and copies generated tokens to CPU only at the loop boundary needed by scheduler output.
4. The residual code predictor uses incremental cached decoding for acoustic substeps instead of repeatedly processing the same tiny context in prefill-like mode.
5. Fixed-K acoustic decode loops can be captured or prepared for CUDA graph capture once state updates are GPU-resident, with early-stop handled through valid-token masking or equivalent scheduler-visible correction.
6. Benchmarks on the Qwen3 TTS NV talker path report before/after E2E latency and token throughput for the same request configuration, and no claimed improvement relies on incomparable ITL accounting.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Preserve and verify current acoustic inner-loop baseline | AC1, AC6 | pending | coding | claude | Keep current prototype measurable before deeper rewrites. |
| Design Qwen3 TTS NV decode-only fast path below generic runner preparation | AC2 | pending | analyze | codex | Identify reusable metadata and per-step mutable state. |
| Implement minimal per-substep runner path for decode-only acoustic loop | AC2 | pending | coding | claude | Update only position, slot mapping/cache write state, hidden/code state, and output buffers per substep. |
| Move greedy group-0 sampling into the fast model-side path | AC3 | pending | coding | claude | Limit first version to deterministic greedy sampling. |
| Analyze residual code predictor incremental-cache interface | AC4 | pending | analyze | codex | Define cache construction during prefill and one-step residual prediction updates. |
| Implement incremental cached residual code prediction | AC4 | pending | coding | claude | Avoid repeated tiny-context prefill behavior for residual groups. |
| Prepare fixed-K acoustic loop for CUDA graph capture | AC5 | pending | analyze | codex | Validate fixed-shape constraints and early-stop masking strategy. |
| Benchmark structural fast paths against baseline | AC6 | pending | coding | claude | Compare E2E latency and token throughput under matching request settings. |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
