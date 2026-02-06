"""CLI helpers for vLLM-Omni entrypoints."""

# To ensure patch imports work properly, disable unused import checks
# ruff: noqa: E402, F401
# isort: off
try:
    from vllm_omni.benchmarks.patch import patch
except ModuleNotFoundError as e:
    # Serving should not require the optional benchmark dependency `pydub`.
    if e.name != "pydub":
        raise
# isort: on

from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

from .serve import OmniServeCommand

__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]
