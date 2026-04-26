"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import os
import signal
import time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import (
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.engine.core import EngineCoreProc, EngineCoreRequestType
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_engine_zmq_addresses,
)
from vllm.v1.utils import shutdown

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fish_install_dac_ready_wakeup()

    def _fish_dac_engine_side_loop_enabled(self) -> bool:
        if os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP", "0") != "1":
            return False
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        return (
            getattr(model_config, "model_stage", "") == "dac_decoder"
            and bool(getattr(model_config, "async_chunk", False))
        )

    def _fish_dac_direct_worker_enabled(self) -> bool:
        if os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER", "0") != "1":
            return False
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        return (
            getattr(model_config, "model_stage", "") == "dac_decoder"
            and bool(getattr(model_config, "async_chunk", False))
        )

    def _fish_dac_ready_wakeup_enabled(self) -> bool:
        return (
            os.environ.get("VLLM_FISH_DAC_READY_WAKEUP", "0") == "1"
            and (
                self._fish_dac_engine_side_loop_enabled()
                or self._fish_dac_direct_worker_enabled()
            )
        )

    def _fish_dac_has_ready_side_work(self) -> bool:
        scheduler = getattr(self, "scheduler", None)
        probe = getattr(scheduler, "fish_dac_has_ready_work", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            logger.debug("Fish DAC side-loop readiness probe failed", exc_info=True)
            return False

    def _fish_input_queue_empty(self) -> bool:
        input_queue = getattr(self, "input_queue", None)
        if input_queue is None:
            return True
        try:
            return bool(input_queue.empty())
        except Exception:
            logger.debug("Fish DAC input queue probe failed", exc_info=True)
            return False

    def _fish_install_dac_ready_wakeup(self) -> None:
        if not self._fish_dac_ready_wakeup_enabled():
            return
        scheduler = getattr(self, "scheduler", None)
        adapter = getattr(scheduler, "chunk_transfer_adapter", None)
        register = getattr(adapter, "set_ready_callback", None)
        if not callable(register):
            return
        register(self._fish_dac_ready_wakeup)

    def _fish_dac_ready_wakeup(self) -> None:
        input_queue = getattr(self, "input_queue", None)
        if input_queue is None:
            return
        try:
            input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        except Exception:
            logger.debug("Fish DAC ready wakeup failed", exc_info=True)

    def has_work(self) -> bool:
        if (
            self._fish_dac_ready_wakeup_enabled()
            and self._fish_dac_has_ready_side_work()
        ):
            return True
        return super().has_work()

    def _process_engine_step(self) -> bool:
        """Run one or more Stage1 DAC steps before returning to the busy loop.

        The default vLLM loop performs one schedule/execute/update cycle per
        outer EngineCore tick.  For Fish DAC chunks this leaves throughput on
        the table because the recv thread can make the next chunk ready
        immediately after the current decode.  In side-loop mode, Stage1 keeps
        consuming already-ready DAC chunks and pushes each EngineCoreOutputs
        object directly to the normal output queue.
        """
        if self._fish_dac_direct_worker_enabled():
            handled, model_executed = self._fish_dac_process_direct_worker()
            if handled:
                return model_executed

        if not self._fish_dac_engine_side_loop_enabled():
            return super()._process_engine_step()

        max_steps = max(
            1,
            int(os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_MAX_STEPS", "8") or 8),
        )
        wait_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_WAIT_US", "0") or 0) / 1_000_000.0,
        )
        idle_budget_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_IDLE_US", "0") or 0) / 1_000_000.0,
        )
        poll_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_POLL_US", "100") or 100) / 1_000_000.0,
        )
        profile = os.environ.get("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_PROFILE", "0") == "1"

        model_executed_any = False
        saw_empty_step = False
        steps = 0
        outputs_sent = 0
        idle_polls = 0

        while steps < max_steps:
            outputs, model_executed = self.step_fn()
            steps += 1
            model_executed_any = model_executed_any or model_executed
            saw_empty_step = saw_empty_step or not model_executed

            for output in outputs.items() if outputs else ():
                self.output_queue.put_nowait(output)
                outputs_sent += 1

            self.post_step(model_executed)

            if not model_executed:
                break
            if steps >= max_steps:
                break
            if not self.input_queue.empty():
                break
            if self._fish_dac_has_ready_side_work():
                continue
            if idle_budget_s > 0 and self.scheduler.has_unfinished_requests():
                deadline = time.monotonic() + idle_budget_s
                ready_after_idle = False
                while time.monotonic() < deadline and self.input_queue.empty():
                    if poll_s > 0:
                        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
                    idle_polls += 1
                    if self._fish_dac_has_ready_side_work():
                        ready_after_idle = True
                        break
                if ready_after_idle:
                    continue
            if wait_s > 0 and self.scheduler.has_unfinished_requests():
                time.sleep(wait_s)
                if self._fish_dac_has_ready_side_work():
                    continue
            break

        if saw_empty_step and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        if profile and (steps > 1 or outputs_sent):
            logger.info(
                "Fish DAC engine side loop: steps=%d model_executed=%s outputs=%d "
                "idle_polls=%d has_more_ready=%s",
                steps,
                model_executed_any,
                outputs_sent,
                idle_polls,
                self._fish_dac_has_ready_side_work(),
            )

        return model_executed_any

    def _fish_dac_process_direct_worker(self) -> tuple[bool, bool]:
        scheduler = getattr(self, "scheduler", None)
        schedule_worker = getattr(scheduler, "fish_dac_worker_schedule", None)
        update_worker = getattr(scheduler, "fish_dac_worker_update", None)
        if not callable(schedule_worker) or not callable(update_worker):
            return False, False

        max_steps = max(
            1,
            int(os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER_MAX_STEPS", "8") or 8),
        )
        idle_budget_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER_IDLE_US", "0") or 0) / 1_000_000.0,
        )
        poll_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER_POLL_US", "100") or 100) / 1_000_000.0,
        )
        profile = os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER_PROFILE", "0") == "1"

        handled_any = False
        model_executed_any = False
        outputs_sent = 0
        idle_polls = 0
        steps = 0

        while steps < max_steps:
            scheduler_output = schedule_worker()
            if scheduler_output is None:
                if not handled_any:
                    return False, False
                if (
                    idle_budget_s <= 0
                    or not self._fish_input_queue_empty()
                    or not self.scheduler.has_unfinished_requests()
                ):
                    break
                deadline = time.monotonic() + idle_budget_s
                became_ready = False
                while time.monotonic() < deadline and self._fish_input_queue_empty():
                    if poll_s > 0:
                        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
                    idle_polls += 1
                    if self._fish_dac_has_ready_side_work():
                        became_ready = True
                        break
                if became_ready:
                    continue
                break

            handled_any = True
            model_executed = scheduler_output.total_num_scheduled_tokens > 0
            if not model_executed:
                break

            future = self.model_executor.execute_model(
                scheduler_output,
                non_block=True,
            )
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                model_output = future.result()
                if model_output is None:
                    model_output = self.model_executor.sample_tokens(grammar_output)

            self._process_aborts_queue()
            outputs = update_worker(scheduler_output, model_output)
            for output in outputs.items() if outputs else ():
                self.output_queue.put_nowait(output)
                outputs_sent += 1
            self.post_step(model_executed)
            model_executed_any = True
            steps += 1

            if steps >= max_steps:
                break
            if not self._fish_input_queue_empty():
                break
            if self._fish_dac_has_ready_side_work():
                continue
            if idle_budget_s <= 0 or not self.scheduler.has_unfinished_requests():
                break

        if profile and handled_any:
            logger.info(
                "Fish DAC direct worker: steps=%d model_executed=%s outputs=%d "
                "idle_polls=%d has_more_ready=%s",
                steps,
                model_executed_any,
                outputs_sent,
                idle_polls,
                self._fish_dac_has_ready_side_work(),
            )

        return handled_any, model_executed_any

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process."""
        shutdown_requested = False
        maybe_register_config_serialize_by_value()

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: StageEngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config

            set_process_title(f"StageEngineCoreProc_DP{dp_rank}")
            decorate_logs()

            # the current vllm-omni does not support data parallelism,
            # so we set the data parallel size to 1.
            # [TODO] support data parallelism in the future.
            # https://github.com/vllm-project/vllm-omni/issues/984
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            parallel_config.data_parallel_index = dp_rank

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            if engine_core is not None:
                engine_core.shutdown()


def spawn_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
) -> tuple[EngineZmqAddresses, BaseProcess, str]:
    """Spawn a *StageEngineCoreProc* subprocess without performing the handshake.

    Must be called while the correct device env vars are set (e.g. under
    the stage-launch lock).  Call ``complete_stage_handshake`` afterwards.

    Returns ``(addresses, process, handshake_address)``.
    """
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageEngineCoreProc.run_stage_core,
        name="StageEngineCoreProc",
        kwargs={
            "vllm_config": vllm_config,
            "local_client": True,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "dp_rank": 0,
            "local_dp_rank": 0,
        },
    )
    proc.start()
    return addresses, proc, handshake_address


def complete_stage_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Perform the HELLO/INIT/READY handshake with an already-spawned proc.

    On failure the process is terminated before re-raising.
    """
    try:
        _perform_handshake(proc, handshake_address, addresses, vllm_config, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Run the HELLO / INIT / READY handshake with the subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        identity, msg = _recv(poller, handshake_socket, proc, "HELLO", handshake_timeout)
        if msg.get("status") != "HELLO":
            raise RuntimeError(f"Expected HELLO, got: {msg}")

        init_payload = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={},
        )
        handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

        identity, msg = _recv(poller, handshake_socket, proc, "READY", handshake_timeout)
        if msg.get("status") != "READY":
            raise RuntimeError(f"Expected READY, got: {msg}")
        num_gpu_blocks = msg.get("num_gpu_blocks")
        if num_gpu_blocks is not None:
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks


def _recv(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    proc: BaseProcess,
    expected: str,
    timeout_s: int = 600,
) -> tuple[bytes, dict]:
    """Wait for one handshake message; raise if the process dies first."""
    timeout_ms = timeout_s * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected} from StageEngineCoreProc after {timeout_s}s. "
                f"This typically indicates model loading or initialization is taking too long. "
                f"Consider increasing `stage_init_timeout` for large models."
            )
        if handshake_socket in events:
            identity, raw = handshake_socket.recv_multipart()
            return identity, msgspec.msgpack.decode(raw)
        if proc.exitcode is not None:
            raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")
