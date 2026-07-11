"""Compatibility imports for the consolidated MiniCPM worker provider."""

from vllm_omni.experimental.fullduplex.minicpmo45.worker import (
    PassiveNativeDuplexStage,
    is_passive_native_duplex_stage,
    maybe_load_minicpmo_native_duplex_target,
    patch_minicpmo_remote_config,
    patch_minicpmo_transformers_compat,
)

__all__ = [
    "PassiveNativeDuplexStage",
    "is_passive_native_duplex_stage",
    "maybe_load_minicpmo_native_duplex_target",
    "patch_minicpmo_remote_config",
    "patch_minicpmo_transformers_compat",
]
