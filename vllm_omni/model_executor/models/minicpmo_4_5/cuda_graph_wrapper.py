# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class HiFTGraphWrapper:
    def __init__(self, token2wav, connector_config, capture_batch_sizes):
        self.decode_fn = token2wav.hift.inference
        self.graph_fn = token2wav.hift._inference_pre_istft
        self.finalize_fn = token2wav.hift._finalize_decode
        self.codec_chunk_frames = connector_config["codec_chunk_frames"]
        self.initial_codec_chunk_frames = int(connector_config.get("initial_codec_chunk_frames", 0))
        self.codec_left_context_frames = connector_config["codec_left_context_frames"]
        lookahead_layer = getattr(token2wav.flow.encoder, "pre_lookahead_layer", None)
        pre_lookahead_len = getattr(lookahead_layer, "pre_lookahead_len", None)
        self.pre_lookahead_len = int(pre_lookahead_len) if pre_lookahead_len is not None else 3
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.mel_frames = int(token2wav.hift.conv_pre.in_channels)
        self.flow_upsample_rate = int(getattr(token2wav.flow, "token_mel_ratio", 2))
        self.capture_bucket_size, self.capture_source_cache_len = self.derive_capture_bucket_size()
        self.capture_batch_sizes = capture_batch_sizes
        self.graph: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.static_speech_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_magnitude_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_phase_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        parameter = next(token2wav.hift.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.max_lazy_graphs = int(connector_config.get("hift_max_lazy_graphs", 8))
        if self.max_lazy_graphs < 0:
            raise ValueError("MiniCPM-o hift_max_lazy_graphs must be >= 0")
        self.lazy_graph_count = 0

    def derive_capture_bucket_size(self):
        chunk_mel_frames = (
            self.codec_chunk_frames + self.codec_left_context_frames - self.pre_lookahead_len
        ) * self.flow_upsample_rate

        frames = [chunk_mel_frames, chunk_mel_frames + self.mel_cache_len]
        cache_lengths = [0, self.source_cache_len]
        if self.initial_codec_chunk_frames and self.initial_codec_chunk_frames != self.codec_chunk_frames:
            first_frames = (
                self.initial_codec_chunk_frames + self.codec_left_context_frames - self.pre_lookahead_len
            ) * self.flow_upsample_rate
            if first_frames <= self.mel_cache_len:
                raise ValueError("MiniCPM-o initial codec chunk must emit audio beyond the HiFT mel cache")
            frames.append(first_frames)
            cache_lengths.append(0)
        return frames, cache_lengths

    def capture(self):
        for batch_size in self.capture_batch_sizes:
            for mel_frames, source_cache_len in zip(
                self.capture_bucket_size,
                self.capture_source_cache_len,
                strict=True,
            ):
                self._capture(batch_size, mel_frames, source_cache_len)

    def _capture(
        self,
        batch_size: int,
        mel_frames: int,
        source_cache_len: int,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot capture HiFT graph during an active stream capture")

        key = (batch_size, mel_frames, source_cache_len)

        if key in self.graph:
            return

        static_mel = torch.zeros(batch_size, self.mel_frames, mel_frames, device=self.device, dtype=self.dtype)
        static_source_cache = torch.zeros(batch_size, 1, source_cache_len, device=self.device, dtype=self.dtype)
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                warmup_outputs = self.graph_fn(static_mel, static_source_cache)
        current_stream.wait_stream(warmup_stream)
        del warmup_outputs

        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            static_magnitude_output, static_phase_output, static_cache_source_output = self.graph_fn(
                static_mel,
                static_source_cache,
            )

        self.graph[key] = graph
        self.static_speech_inputs[key] = static_mel
        self.static_cache_source_inputs[key] = static_source_cache

        self.static_magnitude_outputs[key] = static_magnitude_output
        self.static_phase_outputs[key] = static_phase_output
        self.static_cache_source_outputs[key] = static_cache_source_output
        logger.info("Captured HiFT CUDA Graph for shape %s", key)

    def replay(self, speech_feat, cache_source):
        if torch.cuda.is_current_stream_capturing():
            logger.info("Falling back to eager HiFT inference during an active stream capture")
            return self.decode_fn(speech_feat, cache_source)

        batch_size = speech_feat.shape[0]
        num_frames = speech_feat.shape[2]
        cache_source_len = cache_source.shape[2]
        target_b = next((b for b in sorted(self.capture_batch_sizes) if b >= batch_size), None)

        if target_b is None:
            logger.info("Falling back to eager HiFT inference for unsupported batch size %d", batch_size)
            return self.decode_fn(speech_feat, cache_source)

        key = (target_b, num_frames, cache_source_len)

        if key not in self.graph:
            if self.lazy_graph_count >= self.max_lazy_graphs:
                logger.info("Falling back to eager HiFT inference after reaching the lazy Graph limit")
                return self.decode_fn(speech_feat, cache_source)
            logger.info("Lazily capturing HiFT CUDA Graph for shape %s", key)
            self._capture(*key)
            self.lazy_graph_count += 1

        static_speech_inputs = self.static_speech_inputs[key].zero_()
        static_speech_inputs[:batch_size].copy_(speech_feat)
        static_cache_sources = self.static_cache_source_inputs[key].zero_()
        static_cache_sources[:batch_size].copy_(cache_source)

        self.graph[key].replay()
        static_magnitude_output = self.static_magnitude_outputs[key]
        static_phase_output = self.static_phase_outputs[key]
        static_cache_source_output = self.static_cache_source_outputs[key]
        cache_source = static_cache_source_output[:batch_size].clone()
        speech = self.finalize_fn(static_magnitude_output[:batch_size], static_phase_output[:batch_size]).clone()
        return speech, cache_source


def _tensor_signature(value: torch.Tensor | None) -> tuple:
    if value is None:
        return (None,)
    return tuple(value.shape), str(value.dtype), str(value.device)


_DTYPE_MAP = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.bool)}


def _memory_snapshot(device: torch.device) -> tuple[int, int] | None:
    """(allocated, reserved) bytes, or None when the device cannot report them.

    Capture draws on the caching allocator, so these are the numbers that say
    what a capture cost. Free device memory is not: the allocator serves a
    capture out of memory it has already reserved, which is most of the device
    on a normally configured worker.
    """
    if device.type != "cuda":
        return None
    try:
        return int(torch.accelerator.memory_allocated(device)), int(torch.accelerator.memory_reserved(device))
    except Exception:
        return None


def _format_memory_delta(before: tuple[int, int] | None, after: tuple[int, int] | None) -> str:
    if before is None or after is None:
        return ""
    mib = 1024 * 1024
    return (
        f" [allocated {after[0] / mib:.1f} MiB (+{(after[0] - before[0]) / mib:.1f}), "
        f"reserved {after[1] / mib:.1f} MiB (+{(after[1] - before[1]) / mib:.1f})]"
    )


def _tensors_from_key(key: tuple) -> tuple[torch.Tensor, ...]:
    """Rebuild zero tensors from a cache key (shape, dtype, device tuples).

    Raises ``KeyError`` for a dtype the key cannot round-trip, so the caller
    can fall back to eager instead of capturing wrong-dtype static buffers.
    """
    tensors = []
    for shape, dtype_str, device_str in key[1:]:
        dtype = _DTYPE_MAP[dtype_str]
        device = torch.device(device_str)
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(tensors)


class CFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the CFM DiT estimator.

    Captures one blocks_forward_chunk call (in_proj -> DiT blocks -> final_layer)
    as the graph target. The 10-step Euler loop stays in Python, replaying
    the graph 10 times per decode.

    Graphs are retired a whole generation at a time rather than one at a time.
    Every capture shares one private memory pool, so a retired graph's blocks
    return to that pool while its live peers still hold those addresses in
    their recorded kernel arguments, and the next replay reads memory that now
    belongs to something else. Retiring the whole generation bounds the cache
    without ever leaving a live graph behind a freed one.

    Cache misses capture. A capture failure disables the wrapper for the rest
    of the process, while a key whose static buffers cannot be rebuilt sends
    only that one shape eager. Outputs are cloned after replay to prevent
    streaming cache corruption.
    """

    cache_policy: str = "flush"
    optimized_io: bool = False
    _compiled_graph_fn = None
    _graph_pool = None

    def __init__(
        self,
        graph_fn,
        *,
        max_graphs: int = 32,
        cache_policy: str = "flush",
        optimized_io: bool = False,
        compile_estimator: bool = False,
        compile_mode: str = "blocks",
        attention_backend: str = "sdpa",
        channels_last: bool = False,
    ) -> None:
        self.graph_fn = graph_fn
        self.max_graphs = int(max_graphs)
        if cache_policy not in ("flush", "retain"):
            raise ValueError("CFM graph cache_policy must be 'flush' or 'retain'")
        self.cache_policy = cache_policy
        self.optimized_io = optimized_io
        self.device = next(graph_fn.__self__.parameters()).device
        # A non-positive budget means "no graphs", the same as
        # `enable_cfm_graph: false`. Clamping to 1 would instead build a
        # one-entry cache that flushes on every new shape.
        self.enabled = self.max_graphs > 0
        if compile_mode not in ("full", "blocks"):
            raise ValueError("CFM compile_mode must be 'full' or 'blocks'")
        if attention_backend not in ("sdpa", "tiled_fp32"):
            raise ValueError("CFM attention_backend must be 'sdpa' or 'tiled_fp32'")
        if (attention_backend != "sdpa" or channels_last) and (not compile_estimator or compile_mode != "blocks"):
            raise ValueError("CFM optimized attention/convolution requires block compilation")
        if compile_estimator and self.enabled and self.device.type == "cuda":
            # CUDA graphs remain owned here; nested Inductor graph pools have
            # incompatible lifetimes with shape-cache retirement.
            self._compiled_graph_fn = self._compile_graph_fn(graph_fn, compile_mode, attention_backend, channels_last)
        self._cache: dict[tuple, tuple] = {}
        # Shapes whose key cannot round-trip: eager for those, keep the rest.
        self._unsupported: set[tuple] = set()
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "flushes": 0,
            "eager": 0,
        }

    @staticmethod
    def _compile_graph_fn(graph_fn, mode, attention_backend="sdpa", channels_last=False):
        def compile_function(function):
            return torch.compile(
                function,
                dynamic=True,
                fullgraph=True,
                options={"epilogue_fusion": False, "triton.cudagraphs": False},
            )

        if mode == "full":
            return compile_function(graph_fn)
        estimator = graph_fn.__self__
        if attention_backend == "tiled_fp32" or channels_last:
            from functools import partial

            from .cfm_attention import block_forward_optimized

            if next(estimator.parameters()).dtype != torch.float32:
                raise ValueError("CFM optimized attention/convolution requires float32 estimator weights")
            blocks = []
            for block in estimator.blocks:
                weights = None
                if channels_last:
                    layers = (block.conv.block[1], block.conv.block[6])
                    if any(
                        layer.kernel_size != (3,)
                        or layer.stride != (1,)
                        or layer.dilation != (1,)
                        or layer.groups != 1
                        or layer.padding != (0,)
                        for layer in layers
                    ):
                        raise ValueError("CFM channels_last requires the native 3-tap causal convolutions")
                    # Inference-only wrapper is constructed after weight loading.
                    # Do not replace module methods or mutate checkpoint weights.
                    weights = tuple(
                        layer.weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last)
                        for layer in layers
                    )
                blocks.append(
                    compile_function(
                        partial(
                            block_forward_optimized, block, attention_backend=attention_backend, conv_weights=weights
                        )
                    )
                )
        else:
            blocks = [compile_function(block.forward_chunk) for block in estimator.blocks]
        final_layer = compile_function(estimator.final_layer)

        def blocks_forward(x, time, mask, cnn_cache, att_cache, cnn_out, att_out):
            # Keep cache writes outside the compiled blocks. Whole-estimator
            # compilation functionalizes these writes into a large, expensive
            # cat/copy kernel. The outer CUDA graph still records every launch.
            x = estimator.in_proj(x.transpose(1, 2))
            for index, block in enumerate(blocks):
                x, cnn, att = block(x, time, cnn_cache[index], att_cache[index], mask)
                cnn_out[index].copy_(cnn)
                att_out[index, :, :, : att.shape[2], :].copy_(att)
            return final_layer(x, time).transpose(1, 2)

        # Original modules stay intact for eager fallback and other callers.
        return blocks_forward

    def stats_snapshot(self) -> dict[str, int]:
        """Bounded cumulative telemetry for the graph cache."""
        return {**self._stats, "cache_size": len(self._cache)}

    def _call_graph_fn(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        function = self._compiled_graph_fn or self.graph_fn
        return function(args[0], args[1], args[6], args[2], args[3], args[4], args[5])

    def _eager(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._stats["eager"] += 1
        with torch.no_grad():
            # Unknown shapes beyond a retained cache must not trigger another
            # compile; this is also the fallback after compilation failure.
            result = self.graph_fn(inputs[0], inputs[1], inputs[6], inputs[2], inputs[3], inputs[4], inputs[5])
        return result, inputs[4], inputs[5]

    def _flush(self) -> None:
        """Retire every captured graph at once.

        The sync keeps a replay from being in flight when the graphs go, and
        the explicit ``reset()`` tears each one down here rather than whenever
        Python drops the last reference. Nothing process-wide is touched: the
        cuBLAS workspace is shared with graphs this wrapper does not own.
        """
        if not self._cache:
            return
        torch.accelerator.synchronize(self.device)
        for entry in self._cache.values():
            entry[2].reset()
        self._cache.clear()
        self._graph_pool = None
        self._stats["flushes"] += 1
        logger.info("CFM graph cache flushed; stats=%s", self.stats_snapshot())

    def _disable(self, reason: str, key: tuple) -> None:
        logger.warning("Disabling CFM CUDA graphs (%s) for shape=%s; using eager", reason, key, exc_info=True)
        self.enabled = False
        self._flush()

    def _capture(self, key: tuple, inputs: tuple | None = None) -> tuple | None:
        """Capture a CUDA graph for the given key. Returns None on failure.

        ``inputs`` should be the real tensors for this key whenever they are
        available: the static buffers must be built from the values the graph
        will actually run with. Zero-filled placeholders are only a fallback --
        capturing with them bakes the placeholder branch into the graph (e.g. an
        attention mask that looks like "nothing is masked").
        """
        try:
            static_inputs = (
                tuple(None if tensor is None else tensor.detach().clone() for tensor in inputs)
                if inputs is not None
                else _tensors_from_key(key)
            )
        except KeyError:
            # An unsupported dtype is a property of this shape alone, so run it
            # eager and keep the other shapes on graphs.
            logger.warning("CFM graph key carries an unsupported dtype: %s; using eager", key)
            self._unsupported.add(key)
            return None

        memory_before = _memory_snapshot(self.device)
        try:
            # Warmup runs the same kernels as the capture, so a fault here
            # leaves the same dirty capture-stream and pool state, and must be
            # handled the same way.
            current_stream = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(3):
                    warmup_output = self._call_graph_fn(static_inputs)
            current_stream.wait_stream(warmup_stream)
            del warmup_output

            graph = CUDAGraph()
            if self._compiled_graph_fn is not None:
                # Allocate after compiler warmup, which can clear allocator
                # state. This wrapper owns and retires the entire pool.
                if self._graph_pool is None:
                    self._graph_pool = torch.cuda.graph_pool_handle()
                pool = self._graph_pool
            else:
                pool = current_platform.get_global_graph_pool()
            with torch.no_grad(), torch.cuda.graph(graph, pool=pool):
                static_output = self._call_graph_fn(static_inputs)
        except Exception:
            # A failed capture can leave the capture stream current and the
            # allocator still routing into the graph pool, so there is no safe
            # way to keep capturing afterwards.
            self._disable("capture failed", key)
            return None

        self._stats["captures"] += 1
        logger.info(
            "Captured CFM CUDA Graph for shape %s (cache=%d/%d, stats=%s)%s",
            key,
            len(self._cache) + 1,
            self.max_graphs,
            self.stats_snapshot(),
            _format_memory_delta(memory_before, _memory_snapshot(self.device)),
        )
        return (static_inputs, static_output, graph)

    def replay(
        self,
        estimator_input: torch.Tensor,
        time_emb: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        borrow_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = (estimator_input, time_emb, cnn_cache, att_cache, cnn_out, att_out, attn_mask)
        self._stats["calls"] += 1

        if not self.enabled or torch.cuda.is_current_stream_capturing() or estimator_input.device.type != "cuda":
            return self._eager(inputs)

        key = ("estimator_step",) + tuple(_tensor_signature(v) for v in inputs)
        if key in self._unsupported:
            return self._eager(inputs)
        entry = self._cache.get(key)

        if entry is None:
            if len(self._cache) >= self.max_graphs:
                if self.cache_policy == "retain":
                    # Preserve the live graph pool; a cold shape runs eagerly.
                    return self._eager(inputs)
                self._flush()
            entry = self._capture(key, inputs)
            if entry is None:
                return self._eager(inputs)
            self._cache[key] = entry
        else:
            self._stats["hits"] += 1

        static_inputs, static_output, graph = entry
        for index, (static, current) in enumerate(zip(static_inputs, inputs, strict=True)):
            if self.optimized_io and index in (4, 5):
                # The estimator fully overwrites these output buffers. Copying
                # their uninitialized caller values wastes memory bandwidth.
                continue
            if static is not None:
                static.copy_(current)
        graph.replay()
        if self.optimized_io and borrow_outputs:
            # Valid only until the next replay. The Euler loop consumes the
            # estimate and copies each step's cache before invoking us again.
            return static_output, static_inputs[4], static_inputs[5]
        return (
            static_output.detach().clone(),
            static_inputs[4].detach().clone(),
            static_inputs[5].detach().clone(),
        )
