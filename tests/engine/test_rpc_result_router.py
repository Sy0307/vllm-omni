import queue

import pytest

from vllm_omni.engine.messages import (
    CollectiveRPCResultMessage,
    DuplexControlResultMessage,
    ErrorMessage,
)
from vllm_omni.engine.rpc_result_router import RpcResultRouter
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _duplex_result(control_id: str) -> DuplexControlResultMessage:
    return DuplexControlResultMessage(
        control_id=control_id,
        fence=DuplexFence("sid"),
        operation="append",
        session_id="sid",
        ok=True,
        stage_results=[],
    )


def test_rpc_result_router_routes_out_of_order_results_by_correlation_id():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    first = router.register(("duplex", "first"))
    second = router.register(("duplex", "second"))

    source.put(_duplex_result("second"))
    source.put(_duplex_result("first"))

    assert first.get(timeout=1).control_id == "first"
    assert second.get(timeout=1).control_id == "second"
    router.close()


def test_rpc_result_router_drops_only_the_late_result_after_unregister():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    expired = router.register(("duplex", "expired"))
    active = router.register(("collective", "active"))
    router.unregister(("duplex", "expired"), expired)

    source.put(_duplex_result("expired"))
    source.put(
        CollectiveRPCResultMessage(
            rpc_id="active",
            method="health",
            stage_ids=[0],
            results=["ok"],
        )
    )

    assert active.get(timeout=1).rpc_id == "active"
    assert expired.empty()
    router.close()


def test_rpc_result_router_broadcasts_fatal_errors_to_pending_waiters():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    duplex = router.register(("duplex", "one"))
    collective = router.register(("collective", "two"))

    source.put(ErrorMessage(error="orchestrator failed", fatal=True))

    assert duplex.get(timeout=1).error == "orchestrator failed"
    assert collective.get(timeout=1).error == "orchestrator failed"
    with pytest.raises(RuntimeError, match="orchestrator failed"):
        router.register(("duplex", "after-failure"))
    router.close()


def test_rpc_result_router_does_not_broadcast_uncorrelated_nonfatal_errors():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    waiter = router.register(("duplex", "active"))

    source.put(ErrorMessage(error="request failed", fatal=False, request_id="other"))
    source.put(_duplex_result("active"))

    assert waiter.get(timeout=1).control_id == "active"
    router.close()


def test_rpc_result_router_close_unblocks_waiters_and_stops_consumer():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    waiter = router.register(("duplex", "pending"))

    router.close()

    result = waiter.get(timeout=1)
    assert isinstance(result, ErrorMessage)
    assert result.fatal is True
    assert result.error == "RPC result router closed"
    assert not router._thread.is_alive()
    with pytest.raises(RuntimeError, match="router is closed"):
        router.register(("duplex", "after-close"))
