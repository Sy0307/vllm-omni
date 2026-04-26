import os
import time
from collections import defaultdict

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.outputs import OmniModelRunnerOutput

logger = init_logger(__name__)


class OmniGenerationScheduler(VLLMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        self._fish_dac_microbatch = (
            os.environ.get("VLLM_FISH_DAC_MICROBATCH", "0") == "1"
            and getattr(model_config, "model_stage", "") == "dac_decoder"
        )
        self._fish_dac_microbatch_target = max(
            1,
            int(os.environ.get("VLLM_FISH_DAC_MICROBATCH_TARGET", "4") or 4),
        )
        self._fish_dac_microbatch_wait_s = max(
            0.0,
            float(os.environ.get("VLLM_FISH_DAC_MICROBATCH_WAIT_MS", "1.0") or 0.0) / 1000.0,
        )
        self._fish_dac_bucket_aware = (
            os.environ.get("VLLM_FISH_DAC_BUCKET_AWARE", "0") == "1"
            and getattr(model_config, "model_stage", "") == "dac_decoder"
        )
        self._fish_dac_sched_fastpath = (
            os.environ.get("VLLM_FISH_DAC_SCHED_FASTPATH", "0") == "1"
            and getattr(model_config, "model_stage", "") == "dac_decoder"
            and getattr(model_config, "async_chunk", False)
        )
        self._fish_dac_sched_fastpath_profile = (
            os.environ.get("VLLM_FISH_DAC_SCHED_FASTPATH_PROFILE", "0") == "1"
            and self._fish_dac_sched_fastpath
        )
        self._fish_dac_dedicated_worker = (
            os.environ.get("VLLM_FISH_DAC_DEDICATED_WORKER", "0") == "1"
            and self._fish_dac_sched_fastpath
        )
        self._fish_dac_worker_batch_size = max(
            1,
            int(
                os.environ.get(
                    "VLLM_FISH_DAC_WORKER_BATCH_SIZE",
                    str(max(1, self.max_num_running_reqs)),
                )
                or max(1, self.max_num_running_reqs)
            ),
        )
        self._fish_dac_sched_profile = (
            os.environ.get("VLLM_FISH_DAC_SCHED_PROFILE", "0") == "1"
            and getattr(model_config, "model_stage", "") == "dac_decoder"
        )
        self._fish_dac_update_fastpath = (
            os.environ.get("VLLM_FISH_DAC_UPDATE_FASTPATH", "0") == "1"
            and self._fish_dac_sched_fastpath
        )
        self._fish_dac_rearm_in_update = (
            os.environ.get("VLLM_FISH_DAC_REARM_IN_UPDATE", "0") == "1"
            and self._fish_dac_update_fastpath
        )
        self._fish_dac_direct_worker = (
            os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER", "0") == "1"
            and self._fish_dac_sched_fastpath
            and self._fish_dac_update_fastpath
        )
        self._fish_dac_direct_worker_mixed_bucket = (
            os.environ.get("VLLM_FISH_DAC_DIRECT_WORKER_MIXED_BUCKET", "0") == "1"
            and self._fish_dac_direct_worker
        )
        self._fish_dac_direct_worker_prefetch = max(
            1,
            int(
                os.environ.get(
                    "VLLM_FISH_DAC_DIRECT_WORKER_PREFETCH",
                    str(max(1, self._fish_dac_worker_batch_size * 2)),
                )
                or max(1, self._fish_dac_worker_batch_size * 2)
            ),
        )
        self._fish_dac_sched_profile_interval = max(
            1,
            int(os.environ.get("VLLM_FISH_DAC_SCHED_PROFILE_INTERVAL", "200") or 200),
        )
        self._fish_dac_bucket_frames = self._parse_fish_dac_bucket_frames()
        self._fish_dac_sched_steps = 0
        self._fish_dac_fastpath_steps = 0

    def _fish_count_ready_dac_chunks(self) -> tuple[int, int]:
        ready = 0
        live = 0
        for request in self.running:
            if request.request_id not in self.requests:
                continue
            live += 1
            if len(request.prompt_token_ids) > request.num_computed_tokens:
                ready += 1
        for request in self.waiting:
            if request.request_id not in self.requests:
                continue
            live += 1
            if len(request.prompt_token_ids) > 0:
                ready += 1
        return ready, live

    @staticmethod
    def _parse_fish_dac_bucket_frames() -> list[int]:
        spec = os.environ.get("VLLM_FISH_DAC_BUCKET_FRAMES", "4,25,50").strip()
        try:
            return sorted(int(part.strip()) for part in spec.split(",") if part.strip())
        except ValueError:
            logger.warning_once("Ignoring invalid VLLM_FISH_DAC_BUCKET_FRAMES=%r", spec)
            return [4, 25, 50]

    def _fish_dac_ready_tokens(self, request: Request) -> int:
        if request.request_id not in self.requests:
            return 0
        ready_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
        if ready_tokens > 0:
            return ready_tokens
        if (
            self.chunk_transfer_adapter is not None
            and request.request_id in self.chunk_transfer_adapter.finished_requests
        ):
            return 1
        return 0

    def fish_dac_has_ready_work(self) -> bool:
        """Return whether Stage1 can run a DAC chunk without waiting.

        This is intentionally a non-mutating probe used by the Stage1 engine
        side loop.  Queue restoration and chunk metadata transfer still happen
        inside ``schedule()`` so the normal scheduler invariants stay in one
        place.
        """
        if not self._fish_dac_sched_fastpath or self.chunk_transfer_adapter is None:
            return False
        if hasattr(self.chunk_transfer_adapter, "has_ready_chunks") and self.chunk_transfer_adapter.has_ready_chunks():
            return True
        for request in self.running:
            if self._fish_dac_ready_tokens(request) > 0:
                return True
        for request in self.waiting:
            if self._fish_dac_ready_tokens(request) > 0:
                return True
        return False

    def has_requests(self) -> bool:
        if (
            os.environ.get("VLLM_FISH_DAC_READY_WAKEUP", "0") == "1"
            and self.fish_dac_has_ready_work()
        ):
            return True
        return super().has_requests()

    def _fish_dac_frame_bucket(self, request: Request) -> int:
        info = getattr(request, "additional_information", None)
        token_count = None
        if isinstance(info, dict):
            next_len = info.get("next_stage_prompt_len")
            if isinstance(next_len, int) and next_len > 0:
                token_count = next_len
        if token_count is None:
            token_count = max(
                len(request.prompt_token_ids) - request.num_computed_tokens,
                len(request.prompt_token_ids),
            )
        frames = max(1, int(token_count + 9) // 10)
        for bucket in self._fish_dac_bucket_frames:
            if frames <= bucket:
                return bucket
        return frames

    def _fish_reorder_ready_waiting(self, selected: list[Request]) -> None:
        if not selected:
            return
        selected_ids = {request.request_id for request in selected}
        if hasattr(self.waiting, "remove_requests"):
            self.waiting.remove_requests(selected)
        else:
            for request in selected:
                try:
                    self.waiting.remove(request)
                except ValueError:
                    pass
        if hasattr(self.waiting, "prepend_requests"):
            self.waiting.prepend_requests(selected)
        else:
            for request in reversed(selected):
                self.waiting.insert(0, request)
        if self._fish_dac_sched_profile:
            logger.debug("Stage1 bucket-aware waiting reorder selected=%s", selected_ids)

    def _fish_bucket_aware_reorder(self) -> None:
        if not self._fish_dac_bucket_aware or self.chunk_transfer_adapter is None:
            return

        def score(request: Request) -> tuple[int, int, int]:
            is_terminal = int(
                request.request_id in self.chunk_transfer_adapter.finished_requests
            )
            return (
                is_terminal,
                self._fish_dac_frame_bucket(request),
                -len(request.prompt_token_ids),
            )

        ready_running = [
            request
            for request in self.running
            if self._fish_dac_ready_tokens(request) > 0
        ]
        if ready_running:
            grouped: dict[int, list[Request]] = defaultdict(list)
            for request in ready_running:
                grouped[self._fish_dac_frame_bucket(request)].append(request)
            preferred_bucket = max(
                grouped.items(),
                key=lambda item: (
                    any(
                        r.request_id in self.chunk_transfer_adapter.finished_requests
                        for r in item[1]
                    ),
                    len(item[1]),
                ),
            )[0]
            preferred_ids = {request.request_id for request in grouped[preferred_bucket]}
            terminal_ids = set(self.chunk_transfer_adapter.finished_requests)
            self.running.sort(
                key=lambda request: (
                    request.request_id not in terminal_ids,
                    request.request_id not in preferred_ids,
                    -self._fish_dac_ready_tokens(request),
                )
            )

        if not self.waiting:
            return

        ready_waiting = [
            request
            for request in list(self.waiting)
            if self._fish_dac_ready_tokens(request) > 0
        ]
        if not ready_waiting:
            return

        grouped_waiting: dict[int, list[Request]] = defaultdict(list)
        for request in ready_waiting:
            grouped_waiting[self._fish_dac_frame_bucket(request)].append(request)
        preferred_bucket = max(
            grouped_waiting.items(),
            key=lambda item: (
                any(
                    r.request_id in self.chunk_transfer_adapter.finished_requests
                    for r in item[1]
                ),
                len(item[1]),
            ),
        )[0]
        selected = sorted(
            grouped_waiting[preferred_bucket],
            key=score,
            reverse=True,
        )
        self._fish_reorder_ready_waiting(selected)

    def _fish_log_sched_profile(
        self,
        num_scheduled_tokens: dict[str, int],
        scheduled_new_reqs: list[Request],
        scheduled_running_reqs: list[Request],
    ) -> None:
        if not self._fish_dac_sched_profile:
            return
        self._fish_dac_sched_steps += 1
        if self._fish_dac_sched_steps > 10 and self._fish_dac_sched_steps % self._fish_dac_sched_profile_interval:
            return

        ready_buckets: dict[int, int] = defaultdict(int)
        for request in list(self.running) + list(self.waiting):
            if self._fish_dac_ready_tokens(request) > 0:
                ready_buckets[self._fish_dac_frame_bucket(request)] += 1
        scheduled_buckets: dict[int, int] = defaultdict(int)
        for request in scheduled_new_reqs + scheduled_running_reqs:
            scheduled_buckets[self._fish_dac_frame_bucket(request)] += 1

        logger.info(
            "Stage1 DAC sched profile: step=%d scheduled=%d new=%d running=%d tokens=%d "
            "ready_buckets=%s scheduled_buckets=%s waiting=%d running_queue=%d",
            self._fish_dac_sched_steps,
            len(num_scheduled_tokens),
            len(scheduled_new_reqs),
            len(scheduled_running_reqs),
            sum(num_scheduled_tokens.values()),
            dict(sorted(ready_buckets.items())),
            dict(sorted(scheduled_buckets.items())),
            len(self.waiting),
            len(self.running),
        )

    def _fish_make_empty_cached_request_data(self) -> OmniCachedRequestData:
        return OmniCachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
            prompt_token_ids={},
            additional_information={},
        )

    def _fish_make_dac_fastpath_output(
        self,
        *,
        scheduled_new_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        cached_prompt_token_ids: dict[str, list[int]],
        cached_additional_information: dict[str, dict | None],
    ) -> SchedulerOutput:
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        empty_block_ids = tuple([] for _ in self.kv_cache_config.kv_cache_groups)

        new_reqs_data = [
            OmniNewRequestData.from_request(
                req,
                empty_block_ids,
                getattr(req, "_all_token_ids", None) if self.use_v2_model_runner else None,
            )
            for req in scheduled_new_reqs
        ]

        cached_reqs_data = OmniCachedRequestData(
            req_ids=[req.request_id for req in scheduled_running_reqs],
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in scheduled_running_reqs],
            all_token_ids={},
            new_block_ids=[None for _ in scheduled_running_reqs],
            num_computed_tokens=[
                req.num_computed_tokens for req in scheduled_running_reqs
            ],
            num_output_tokens=[
                req.num_output_tokens for req in scheduled_running_reqs
            ],
            prompt_token_ids=cached_prompt_token_ids,
            additional_information=cached_additional_information,
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=None,
        )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.postprocess_scheduler_output(
                scheduler_output
            )

        return scheduler_output

    def fish_dac_worker_schedule(self) -> SchedulerOutput | None:
        """Build a DAC batch directly from the ready-chunk side queue.

        This is the Stage1 worker/coalescer path. It bypasses the generic
        request lifecycle in ``schedule()`` for chunk-ready Fish DAC work:
        ready chunks are drained from the adapter, grouped by decode bucket,
        and returned as a cached-request batch for the DAC runner fastpath.
        """
        if not self._fish_dac_direct_worker or self.chunk_transfer_adapter is None:
            return None
        if self._pause_state == PauseState.PAUSED_ALL:
            return None

        self.kv_cache_manager.new_step_starts()

        scheduled_req_limit = self._fish_dac_worker_batch_size
        drain_limit = max(scheduled_req_limit, self._fish_dac_direct_worker_prefetch)
        ready_chunks = self.chunk_transfer_adapter.drain_ready_chunks(drain_limit)
        if not ready_chunks:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting,
                self.running,
            )
            ready_chunks = self.chunk_transfer_adapter.drain_ready_chunks(drain_limit)
            if not ready_chunks:
                return None

        grouped: dict[int, list[Request]] = defaultdict(list)
        valid_chunks: list[Request] = []
        leftovers: list[Request] = []
        for request in ready_chunks:
            if request.request_id not in self.requests:
                continue
            if self._fish_dac_ready_tokens(request) <= 0:
                if request.request_id in self.chunk_transfer_adapter.finished_requests:
                    if len(request.prompt_token_ids) <= request.num_computed_tokens:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                else:
                    continue
            valid_chunks.append(request)
            grouped[self._fish_dac_frame_bucket(request)].append(request)

        if not grouped:
            return None

        selected_ids: set[str] = set()
        scheduled_bucket: int | str = "mixed"
        if getattr(self, "_fish_dac_direct_worker_mixed_bucket", False):
            ordered_chunks = sorted(
                valid_chunks,
                key=lambda req: (
                    req.request_id not in self.chunk_transfer_adapter.finished_requests,
                    self._fish_dac_frame_bucket(req),
                ),
            )
            selected = ordered_chunks[:scheduled_req_limit]
            selected_ids.update(req.request_id for req in selected)
        else:
            preferred_bucket, selected_bucket_reqs = max(
                grouped.items(),
                key=lambda item: (
                    any(
                        req.request_id in self.chunk_transfer_adapter.finished_requests
                        for req in item[1]
                    ),
                    len(item[1]),
                ),
            )
            scheduled_bucket = preferred_bucket
            selected = selected_bucket_reqs[:scheduled_req_limit]
            selected_ids.update(req.request_id for req in selected)
        for bucket, bucket_reqs in grouped.items():
            for request in bucket_reqs:
                if request.request_id not in selected_ids:
                    leftovers.append(request)
        if leftovers and hasattr(self.chunk_transfer_adapter, "prepend_ready_chunks"):
            self.chunk_transfer_adapter.prepend_ready_chunks(leftovers)

        token_budget = self.max_num_scheduled_tokens
        scheduled_timestamp = time.monotonic()
        scheduled_running_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}

        for request in selected:
            if token_budget <= 0:
                break
            required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
            if required_tokens <= 0:
                if request.request_id in self.chunk_transfer_adapter.finished_requests:
                    if len(request.prompt_token_ids) <= request.num_computed_tokens:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
                else:
                    continue
            num_new_tokens = min(required_tokens, token_budget)
            if num_new_tokens <= 0:
                continue

            request.status = RequestStatus.RUNNING
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = max(0, request.num_computed_tokens)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            cached_additional_information[request.request_id] = getattr(
                request,
                "additional_information",
                None,
            )
            request.num_computed_tokens += num_new_tokens
            scheduled_running_reqs.append(request)
            token_budget -= num_new_tokens

        if not num_scheduled_tokens:
            return None

        scheduler_output = self._fish_make_dac_fastpath_output(
            scheduled_new_reqs=[],
            scheduled_running_reqs=scheduled_running_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            cached_prompt_token_ids=cached_prompt_token_ids,
            cached_additional_information=cached_additional_information,
        )

        self._fish_dac_fastpath_steps += 1
        if self._fish_dac_sched_fastpath_profile and (
            self._fish_dac_fastpath_steps <= 10
            or self._fish_dac_fastpath_steps % self._fish_dac_sched_profile_interval == 0
        ):
            logger.info(
                "Stage1 DAC direct worker: step=%d bucket=%s scheduled=%d "
                "tokens=%d leftovers=%d waiting=%d running_queue=%d",
                self._fish_dac_fastpath_steps,
                scheduled_bucket,
                len(num_scheduled_tokens),
                sum(num_scheduled_tokens.values()),
                len(leftovers),
                len(self.waiting),
                len(self.running),
            )

        return scheduler_output

    def fish_dac_worker_update(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        outputs = self._fish_try_update_dac_fastpath(
            scheduler_output,
            model_runner_output,
        )
        if outputs is None:
            return {}
        return outputs

    def _fish_try_schedule_dac_fastpath(self) -> SchedulerOutput | None:
        if not self._fish_dac_sched_fastpath or self.chunk_transfer_adapter is None:
            return None

        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
        if self._fish_dac_dedicated_worker:
            self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
        if self._fish_dac_microbatch and self._fish_dac_microbatch_wait_s > 0:
            ready, live = self._fish_count_ready_dac_chunks()
            if 0 < ready < self._fish_dac_microbatch_target and live > ready:
                time.sleep(self._fish_dac_microbatch_wait_s)
                self.chunk_transfer_adapter.process_pending_chunks(
                    self.waiting,
                    self.running,
                )
                if self._fish_dac_dedicated_worker:
                    self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
                    self.chunk_transfer_adapter.process_pending_chunks(
                        self.waiting,
                        self.running,
                    )
        self._fish_bucket_aware_reorder()

        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}
        scheduled_req_limit = (
            self._fish_dac_worker_batch_size
            if self._fish_dac_dedicated_worker
            else self.max_num_running_reqs
        )

        if self._fish_dac_dedicated_worker and token_budget > 0:
            ready_chunks = self.chunk_transfer_adapter.drain_ready_chunks(
                scheduled_req_limit
            )
            if ready_chunks:
                waiting_remove: list[Request] = []
                running_ids = {request.request_id for request in self.running}
                for request in ready_chunks:
                    if (
                        token_budget <= 0
                        or len(num_scheduled_tokens) >= scheduled_req_limit
                    ):
                        break
                    if request.request_id not in self.requests:
                        continue
                    if request.request_id in num_scheduled_tokens:
                        continue

                    required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
                    if required_tokens <= 0:
                        if request.request_id in self.chunk_transfer_adapter.finished_requests:
                            if len(request.prompt_token_ids) <= request.num_computed_tokens:
                                request.prompt_token_ids.append(0)
                                try:
                                    request._all_token_ids.append(0)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
                        else:
                            continue

                    num_new_tokens = min(required_tokens, token_budget)
                    if num_new_tokens <= 0:
                        continue

                    request.status = RequestStatus.RUNNING
                    if self.log_stats:
                        request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                    if request.num_cached_tokens < 0:
                        request.num_cached_tokens = (
                            request.num_computed_tokens
                            if request.request_id in running_ids
                            else 0
                        )
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens

                    if request.request_id in running_ids:
                        cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
                        cached_additional_information[request.request_id] = getattr(
                            request,
                            "additional_information",
                            None,
                        )
                        scheduled_running_reqs.append(request)
                    else:
                        self.running.append(request)
                        running_ids.add(request.request_id)
                        waiting_remove.append(request)
                        scheduled_new_reqs.append(request)

                if waiting_remove:
                    if hasattr(self.waiting, "remove_requests"):
                        self.waiting.remove_requests(waiting_remove)
                    else:
                        for request in waiting_remove:
                            try:
                                self.waiting.remove(request)
                            except ValueError:
                                pass

        req_index = 0
        while (
            req_index < len(self.running)
            and token_budget > 0
            and len(num_scheduled_tokens) < scheduled_req_limit
        ):
            request = self.running[req_index]
            if request.request_id not in self.requests:
                req_index += 1
                continue
            if request.request_id in num_scheduled_tokens:
                req_index += 1
                continue

            required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
            if required_tokens <= 0:
                if request.request_id in self.chunk_transfer_adapter.finished_requests:
                    if len(request.prompt_token_ids) <= request.num_computed_tokens:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    required_tokens = len(request.prompt_token_ids) - request.num_computed_tokens
                else:
                    req_index += 1
                    continue

            num_new_tokens = min(required_tokens, token_budget)
            if num_new_tokens <= 0:
                req_index += 1
                continue

            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            cached_additional_information[request.request_id] = getattr(
                request,
                "additional_information",
                None,
            )
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        if self._fish_dac_dedicated_worker:
            selected_waiting: list[tuple[Request, int]] = []
            stale_waiting: list[Request] = []
            for request in list(self.waiting):
                if (
                    token_budget <= 0
                    or len(num_scheduled_tokens) >= scheduled_req_limit
                    or self._pause_state != PauseState.UNPAUSED
                ):
                    break
                if request.request_id not in self.requests:
                    stale_waiting.append(request)
                    continue
                if request.request_id in num_scheduled_tokens:
                    continue
                if len(request.prompt_token_ids) == 0:
                    if request.request_id in self.chunk_transfer_adapter.finished_requests:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    else:
                        continue

                required_tokens = max(len(request.prompt_token_ids), 1)
                num_new_tokens = min(required_tokens, token_budget)
                if num_new_tokens <= 0:
                    break
                selected_waiting.append((request, num_new_tokens))
                token_budget -= num_new_tokens

            to_remove = stale_waiting + [request for request, _ in selected_waiting]
            if to_remove:
                if hasattr(self.waiting, "remove_requests"):
                    self.waiting.remove_requests(to_remove)
                else:
                    for request in to_remove:
                        try:
                            self.waiting.remove(request)
                        except ValueError:
                            pass

            for request, num_new_tokens in selected_waiting:
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = 0
                num_scheduled_tokens[request.request_id] = num_new_tokens
                scheduled_new_reqs.append(request)
        else:
            skipped_waiting_requests = create_request_queue(self.policy)
            while (
                self.waiting
                and token_budget > 0
                and len(num_scheduled_tokens) < scheduled_req_limit
                and len(self.running) < self.max_num_running_reqs
                and self._pause_state == PauseState.UNPAUSED
            ):
                request = self.waiting.peek_request()
                if request.request_id not in self.requests:
                    self.waiting.pop_request()
                    continue

                if len(request.prompt_token_ids) == 0:
                    if request.request_id in self.chunk_transfer_adapter.finished_requests:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                required_tokens = max(len(request.prompt_token_ids), 1)
                num_new_tokens = min(required_tokens, token_budget)
                if num_new_tokens <= 0:
                    break

                request = self.waiting.pop_request()
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = 0
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                scheduled_new_reqs.append(request)

            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        if not num_scheduled_tokens:
            self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            return self._fish_make_dac_fastpath_output(
                scheduled_new_reqs=[],
                scheduled_running_reqs=[],
                num_scheduled_tokens={},
                cached_prompt_token_ids={},
                cached_additional_information={},
            )

        scheduler_output = self._fish_make_dac_fastpath_output(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            cached_prompt_token_ids=cached_prompt_token_ids,
            cached_additional_information=cached_additional_information,
        )

        for request in scheduled_new_reqs + scheduled_running_reqs:
            request.num_computed_tokens += num_scheduled_tokens[request.request_id]

        try:
            self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
        finally:
            pass

        self._fish_dac_fastpath_steps += 1
        if self._fish_dac_sched_fastpath_profile and (
            self._fish_dac_fastpath_steps <= 10
            or self._fish_dac_fastpath_steps % self._fish_dac_sched_profile_interval == 0
        ):
            ready, live = self._fish_count_ready_dac_chunks()
            logger.info(
                "Stage1 DAC sched fastpath: step=%d scheduled=%d new=%d running=%d "
                "tokens=%d ready=%d live=%d waiting=%d running_queue=%d "
                "dedicated=%s batch_limit=%d",
                self._fish_dac_fastpath_steps,
                len(num_scheduled_tokens),
                len(scheduled_new_reqs),
                len(scheduled_running_reqs),
                sum(num_scheduled_tokens.values()),
                ready,
                live,
                len(self.waiting),
                len(self.running),
                self._fish_dac_dedicated_worker,
                scheduled_req_limit,
            )

        return scheduler_output

    def _fish_try_update_dac_fastpath(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs] | None:
        if (
            getattr(self, "_fish_dac_update_fastpath", False) is not True
            or self.chunk_transfer_adapter is None
        ):
            return None

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        if not num_scheduled_tokens:
            return {}

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        pooler_outputs = model_runner_output.pooler_output
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        rearmed_running_reqs: set[Request] = set()

        for req_id in num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            req_index = model_runner_output.req_id_to_index.get(req_id)
            pooler_output = (
                pooler_outputs[req_index]
                if pooler_outputs is not None and req_index is not None
                else None
            )
            status_before_stop = request.status
            stopped = False
            finish_reason = None
            kv_transfer_params = None

            if (
                req_id in self.chunk_transfer_adapter.finished_requests
                and request.num_computed_tokens >= len(request.prompt_token_ids)
            ):
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if stopped:
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                    self.chunk_transfer_adapter.cleanup(
                        request.request_id,
                        getattr(request, "external_req_id", None),
                    )
                if status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_running_reqs.add(request)

            if pooler_output is not None or stopped:
                num_cached = request.num_cached_tokens
                if num_cached < 0:
                    num_cached = 0
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finish_reason=finish_reason,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=num_cached,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
                if (
                    not stopped
                    and pooler_output is not None
                    and getattr(self, "_fish_dac_rearm_in_update", False)
                    and hasattr(
                        self.chunk_transfer_adapter,
                        "rearm_running_chunk_request",
                    )
                ):
                    if self.chunk_transfer_adapter.rearm_running_chunk_request(
                        request
                    ):
                        rearmed_running_reqs.add(request)

        if stopped_running_reqs or rearmed_running_reqs:
            self.running = remove_all(
                self.running,
                stopped_running_reqs | rearmed_running_reqs,
            )
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        return engine_core_outputs

    def schedule(self) -> SchedulerOutput:
        """Diffusion fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """
        fish_dac_fastpath_output = self._fish_try_schedule_dac_fastpath()
        if fish_dac_fastpath_output is not None:
            return fish_dac_fastpath_output

        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running_reqs: list[Request] = []
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}

        # Temporary queue: preserve waiting order, do not disturb non-diffusion requests
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
            if self._fish_dac_microbatch and self._fish_dac_microbatch_wait_s > 0:
                ready, live = self._fish_count_ready_dac_chunks()
                if 0 < ready < self._fish_dac_microbatch_target and live > ready:
                    time.sleep(self._fish_dac_microbatch_wait_s)
                    self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
            self._fish_bucket_aware_reorder()

        # OMNI: Track requests that are already finished (e.g., marked by connector)
        # These should be removed from running and not scheduled
        already_finished_reqs: set[Request] = set()
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                already_finished_reqs.add(request)
                req_index += 1
                continue

            num_computed_tokens = request.num_computed_tokens
            required_tokens = len(request.prompt_token_ids) - num_computed_tokens
            # async_chunk: don't schedule placeholder tokens when no new chunk is available.
            if required_tokens <= 0:
                if (
                    self.chunk_transfer_adapter is not None
                    and request.request_id in self.chunk_transfer_adapter.finished_requests
                ):
                    # Upstream may finish with no terminal tokens; append one pad token so we can emit FINISHED.
                    if len(request.prompt_token_ids) <= num_computed_tokens:
                        request.prompt_token_ids.append(0)
                        try:
                            request._all_token_ids.append(0)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    required_tokens = len(request.prompt_token_ids) - num_computed_tokens
                else:
                    req_index += 1
                    continue
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = num_computed_tokens
            cached_additional_information[request.request_id] = getattr(request, "additional_information", None)
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        # OMNI: Remove already finished requests from running queue
        if already_finished_reqs:
            self.running = remove_all(self.running, already_finished_reqs)

        # Fast path selection and scheduling (treat all as diffusion requests,
        # independent of pooling_params)
        while (
            self.waiting
            and token_budget > 0
            and len(self.running) < self.max_num_running_reqs
            and self._pause_state == PauseState.UNPAUSED
        ):
            request = self.waiting.peek_request()
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                # Pop the finished request from waiting queue and don't schedule it
                self.waiting.pop_request()
                continue
            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

            # async_chunk: wait for the first upstream chunk (don't start with placeholders).
            if self.chunk_transfer_adapter is not None and len(request.prompt_token_ids) == 0:
                if request.request_id in self.chunk_transfer_adapter.finished_requests:
                    request.prompt_token_ids.append(0)
                    try:
                        request._all_token_ids.append(0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                else:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            # Uniformly treat as diffusion. A feature flag can be added later
            # via config or request tag.

            # Allocate all input tokens for the request in one shot
            # (allocate 1 placeholder if zero)
            required_tokens = max(len(request.prompt_token_ids), 1)
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break

            # Officially schedule this request
            request = self.waiting.pop_request()
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = 0
            token_budget -= num_new_tokens
            scheduled_new_reqs.append(request)

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            if self.chunk_transfer_adapter:
                # Don't fall back: base scheduler doesn't handle async_chunk
                # requests with empty prompt_token_ids.
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            else:
                res = super().schedule()
                return res

        # Compute common prefix blocks (aligned with v1)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request.request_id)

        # Assemble SchedulerOutput (align with v0.14.0)
        if self.use_v2_model_runner:
            # No resumed reqs in fast path; pass prefill_token_ids for new reqs.
            new_reqs_data = [
                OmniNewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    getattr(req, "_all_token_ids", None),
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                OmniNewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]
        # No running/resumed reqs scheduled in our fast path
        cached_reqs_data = self._make_cached_request_data(
            running_reqs=scheduled_running_reqs,
            resumed_reqs=[],
            num_scheduled_tokens=num_scheduled_tokens,
            spec_decode_tokens=scheduled_spec_decode_tokens,
            req_to_new_blocks=req_to_new_blocks,
        )

        cached_reqs_data = OmniCachedRequestData(
            req_ids=cached_reqs_data.req_ids,
            resumed_req_ids=cached_reqs_data.resumed_req_ids,
            new_token_ids=cached_reqs_data.new_token_ids,
            all_token_ids=cached_reqs_data.all_token_ids,
            new_block_ids=cached_reqs_data.new_block_ids,
            num_computed_tokens=cached_reqs_data.num_computed_tokens,
            num_output_tokens=cached_reqs_data.num_output_tokens,
            prompt_token_ids=cached_prompt_token_ids,
            additional_information=cached_additional_information,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        self._fish_log_sched_profile(
            num_scheduled_tokens,
            scheduled_new_reqs,
            scheduled_running_reqs,
        )

        # Record the request ids scheduled in this step (v0.14.0 behavior).
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        # EC Connector: package metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        self._update_after_schedule(scheduler_output)

        try:
            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]

            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output)

        except Exception:
            # If anything goes wrong, leave the original output unchanged
            logger.exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
        finally:
            # Ensure chunk-waiting requests are restored even on error,
            # otherwise they are permanently orphaned in the adapter's
            # internal deques and never scheduled again.
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)

        return scheduler_output

    """
    Scheduler for the diffusion model.
    This scheduler is modified to stop the request immediately for the diffusion model.
    This is because the diffusion model can generate the final image/audio in one step.
    Note: This is just a minimal modification to the original scheduler,
    and there should be some further efforts to optimize the scheduler.
    The original scheduler is still used for the AR model.
    """

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update the scheduler state based on the model runner output.

        This method is modified to stop the request immediately for the diffusion model.
        """
        fish_dac_fastpath_outputs = OmniGenerationScheduler._fish_try_update_dac_fastpath(
            self,
            scheduler_output,
            model_runner_output,
        )
        if fish_dac_fastpath_outputs is not None:
            return fish_dac_fastpath_outputs

        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        # Merge connector-side stats (align with v0.14.0)
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and getattr(kv_connector_output, "invalid_block_ids", None):
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Request may already be finished (e.g., aborted during
                # execution / pipeline parallelism / async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            status_before_stop = request.status
            finish_reason = None
            routed_experts = None

            # Diffusion request: completes in one step; mark finished and free resources
            if (
                request.status == RequestStatus.FINISHED_STOPPED
                or (self.chunk_transfer_adapter is None and request.num_computed_tokens >= request.num_prompt_tokens)
                or (
                    self.chunk_transfer_adapter is not None
                    and request.request_id in self.chunk_transfer_adapter.finished_requests
                    and request.num_computed_tokens >= len(request.prompt_token_ids)
                )
            ):
                request.status = RequestStatus.FINISHED_STOPPED
                # Optional: set a stop_reason for front-end clarity
                # (does not affect protocol)
                request.stop_reason = request.stop_reason  # or "generation_done"
                stopped = True

            if stopped:
                routed_experts = self._get_routed_experts(request)
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                    if self.chunk_transfer_adapter is not None:
                        self.chunk_transfer_adapter.cleanup(
                            request.request_id,
                            getattr(request, "external_req_id", None),
                        )
                if status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_running_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                # NOTE: structured_output_request should not be None if
                # use_structured_output, we have check above, so safe to ignore
                # type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]  # noqa: E501
                    req_id, new_token_ids
                )

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                num_cached = request.num_cached_tokens
                if num_cached < 0:
                    logger.warning("Negative num_cached_tokens (%d) for request %s, clamping to 0", num_cached, req_id)
                    num_cached = 0
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=num_cached,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                num_cached = request.num_cached_tokens
                if num_cached < 0:
                    logger.warning(
                        "Negative num_cached_tokens (%d) for request %s, clamping to 0", num_cached, request.request_id
                    )
                    num_cached = 0
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=num_cached,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup(
                        request.request_id,
                        getattr(request, "external_req_id", None),
                    )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Collect and publish KV cache events (align with v0.14.0)
        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs
