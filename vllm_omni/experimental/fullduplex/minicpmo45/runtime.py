"""Compatibility exports for tests and integrations using the old combined runtime."""

from .stage0 import MiniCPMO45Stage0DuplexRuntime, _MiniCPMO45Stage0SessionState
from .stage1 import MiniCPMO45Stage1DuplexRuntime

__all__ = [
    "MiniCPMO45Stage0DuplexRuntime",
    "MiniCPMO45Stage1DuplexRuntime",
    "_MiniCPMO45Stage0SessionState",
]
