# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full-Duplex step-latency observations without scheduler policy."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplexStepLatencySnapshot:
    count: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    deadline_misses: int


class DuplexStepLatencyMetrics:
    def __init__(self) -> None:
        self._samples_ms: list[float] = []
        self._deadline_misses = 0

    def record(self, latency_ms: float, *, budget_ms: float | None) -> bool:
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("duplex step latency must be a non-negative finite number")
        self._samples_ms.append(float(latency_ms))
        missed = budget_ms is not None and latency_ms > budget_ms
        if missed:
            self._deadline_misses += 1
        return missed

    @staticmethod
    def _percentile(samples: list[float], percentile: float) -> float | None:
        if not samples:
            return None
        index = max(0, math.ceil(percentile * len(samples)) - 1)
        return samples[index]

    def snapshot(self) -> DuplexStepLatencySnapshot:
        samples = sorted(self._samples_ms)
        return DuplexStepLatencySnapshot(
            count=len(samples),
            p50_ms=self._percentile(samples, 0.50),
            p95_ms=self._percentile(samples, 0.95),
            max_ms=samples[-1] if samples else None,
            deadline_misses=self._deadline_misses,
        )


__all__ = ["DuplexStepLatencyMetrics", "DuplexStepLatencySnapshot"]
