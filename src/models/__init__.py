"""
SPOD Models Module
================
"""

from .spod import (
    FDAConfig,
    TDSConfig,
    SPODConfig,
    compute_fda_frame_weights,
    build_tds_subspaces,
    decompose_hidden_state,
    enhance_hidden_state,
)

from .video_io import load_video, find_local_video, estimate_vision_seq_len

from .model_config import (
    ModelConfig,
    MODEL_CONFIGS,
    get_model_config,
    list_supported_models,
    get_arch_model_ids,
)

__all__ = [
    # Core SPOD
    "FDAConfig",
    "TDSConfig",
    "SPODConfig",
    "compute_fda_frame_weights",
    "build_tds_subspaces",
    "decompose_hidden_state",
    "enhance_hidden_state",
    # Video I/O
    "load_video",
    "find_local_video",
    "estimate_vision_seq_len",
    # Model config
    "ModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "list_supported_models",
    "get_arch_model_ids",
]
