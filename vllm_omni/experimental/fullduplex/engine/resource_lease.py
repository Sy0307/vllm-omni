# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Atomic model-resource leases for experimental Full-Duplex Sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


class DuplexResourceLeaseProvider(Protocol):
    """Model-owned allocator participating in the Session open/close saga."""

    provider_id: str

    async def reserve(
        self,
        fence: DuplexFence,
        *,
        session_config: Mapping[str, object],
        runtime_config: Mapping[str, object],
    ) -> object: ...

    async def release(self, handle: object, *, abort: bool) -> None: ...


@dataclass(slots=True)
class _LeaseEntry:
    provider: DuplexResourceLeaseProvider
    handle: object


@dataclass(slots=True)
class _LeaseBundle:
    fence: DuplexFence
    entries: list[_LeaseEntry] = field(default_factory=list)
    transition_target: DuplexFence | None = None
    transitioned_provider_ids: set[str] = field(default_factory=set)


class DuplexResourceLeaseRollbackError(RuntimeError):
    """A reserve failed and at least one compensating release also failed."""

    def __init__(self, reserve_error: BaseException, rollback_error: BaseException) -> None:
        self.reserve_error = reserve_error
        self.rollback_error = rollback_error
        super().__init__(
            "duplex resource reservation failed and rollback remains pending: "
            f"reserve={reserve_error!r}, rollback={rollback_error!r}"
        )


class DuplexResourceLeaseCoordinator:
    """Reserve providers in order and release opaque handles in reverse order."""

    def __init__(self, providers: Sequence[DuplexResourceLeaseProvider] = ()) -> None:
        normalized = tuple(providers)
        provider_ids: list[str] = []
        for provider in normalized:
            provider_id = getattr(provider, "provider_id", None)
            if not isinstance(provider_id, str) or not provider_id.strip():
                raise TypeError("duplex resource provider must declare a non-empty provider_id")
            if not callable(getattr(provider, "reserve", None)):
                raise TypeError(f"duplex resource provider {provider_id!r} must define reserve()")
            if not callable(getattr(provider, "release", None)):
                raise TypeError(f"duplex resource provider {provider_id!r} must define release()")
            provider_ids.append(provider_id)
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("duplex resource provider_id values must be unique")

        self._providers = normalized
        self._bundles: dict[tuple[str, int], _LeaseBundle] = {}
        self._prewarm_lock = asyncio.Lock()
        self._prewarmed_provider_ids: set[str] = set()
        self._prewarm_batch_sizes: tuple[int, ...] | None = None

    @property
    def providers(self) -> tuple[DuplexResourceLeaseProvider, ...]:
        return self._providers

    @staticmethod
    def _key(fence: DuplexFence) -> tuple[str, int]:
        if not isinstance(fence, DuplexFence):
            raise TypeError("fence must be a DuplexFence")
        return fence.session_id, fence.incarnation

    def has_lease(self, fence: DuplexFence) -> bool:
        return self._key(fence) in self._bundles

    async def prewarm(self, batch_sizes: tuple[int, ...]) -> None:
        if not isinstance(batch_sizes, tuple):
            raise TypeError("prewarm batch sizes must be a tuple")
        if any(type(size) is not int or size <= 0 for size in batch_sizes):
            raise ValueError("prewarm batch sizes must contain positive integers")
        if len(batch_sizes) != len(set(batch_sizes)):
            raise ValueError("prewarm batch sizes must not contain duplicates")

        async with self._prewarm_lock:
            if self._prewarm_batch_sizes is not None and self._prewarm_batch_sizes != batch_sizes:
                raise ValueError("duplex resource providers were already prewarmed for different batch sizes")
            self._prewarm_batch_sizes = batch_sizes
            for provider in self._providers:
                if provider.provider_id in self._prewarmed_provider_ids:
                    continue
                prewarm = getattr(provider, "prewarm", None)
                if callable(prewarm):
                    await prewarm(batch_sizes)
                self._prewarmed_provider_ids.add(provider.provider_id)

    async def reserve(
        self,
        fence: DuplexFence,
        *,
        session_config: Mapping[str, object],
        runtime_config: Mapping[str, object],
    ) -> None:
        key = self._key(fence)
        if key in self._bundles:
            raise RuntimeError(f"duplex resources are already reserved for {fence.session_id!r}")
        bundle = _LeaseBundle(fence=fence)
        self._bundles[key] = bundle
        try:
            for provider in self._providers:
                handle = await provider.reserve(
                    fence,
                    session_config=session_config,
                    runtime_config=runtime_config,
                )
                bundle.entries.append(_LeaseEntry(provider=provider, handle=handle))
        except BaseException as reserve_error:
            try:
                await self.release(fence, abort=True)
            except BaseException as rollback_error:
                raise DuplexResourceLeaseRollbackError(reserve_error, rollback_error) from reserve_error
            raise

    async def release(self, fence: DuplexFence, *, abort: bool) -> None:
        key = self._key(fence)
        bundle = self._bundles.get(key)
        if bundle is None:
            return
        while bundle.entries:
            entry = bundle.entries[-1]
            await entry.provider.release(entry.handle, abort=abort)
            bundle.entries.pop()
        self._bundles.pop(key, None)

    async def advance_epoch(
        self,
        cancelled_fence: DuplexFence,
        next_fence: DuplexFence,
    ) -> None:
        key = self._key(cancelled_fence)
        if self._key(next_fence) != key or next_fence.epoch <= cancelled_fence.epoch:
            raise ValueError("next_fence must advance the same Session incarnation")
        bundle = self._bundles.get(key)
        if bundle is None:
            return
        if bundle.fence == next_fence:
            return
        if bundle.fence != cancelled_fence:
            raise RuntimeError(
                f"duplex resource lease epoch mismatch: expected={cancelled_fence}, actual={bundle.fence}"
            )
        if bundle.transition_target not in {None, next_fence}:
            raise RuntimeError("duplex resource lease already has a different pending epoch transition")
        bundle.transition_target = next_fence
        for entry in bundle.entries:
            provider_id = entry.provider.provider_id
            if provider_id in bundle.transitioned_provider_ids:
                continue
            advance_epoch = getattr(entry.provider, "advance_epoch", None)
            if callable(advance_epoch):
                await advance_epoch(
                    entry.handle,
                    cancelled_fence=cancelled_fence,
                    next_fence=next_fence,
                )
            bundle.transitioned_provider_ids.add(provider_id)
        bundle.fence = next_fence
        bundle.transition_target = None
        bundle.transitioned_provider_ids.clear()


__all__ = [
    "DuplexResourceLeaseCoordinator",
    "DuplexResourceLeaseProvider",
    "DuplexResourceLeaseRollbackError",
]
