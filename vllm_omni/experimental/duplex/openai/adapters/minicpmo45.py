"""Compatibility imports for the consolidated MiniCPM serving adapter."""

from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45NativeDuplexServingAdapter
from vllm_omni.experimental.fullduplex.minicpmo45.input import MiniCPMO45PcmAppendBuffer

__all__ = ["MiniCPMO45NativeDuplexServingAdapter", "MiniCPMO45PcmAppendBuffer"]
