"""Compatibility imports for the consolidated full-duplex MiniCPM runtime."""

from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)
from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import MiniCPMO45Stage1DuplexRuntime

__all__ = [
    "MiniCPMO45Stage0DuplexRuntime",
    "MiniCPMO45Stage1DuplexRuntime",
    "_MiniCPMO45Stage0SessionState",
]
