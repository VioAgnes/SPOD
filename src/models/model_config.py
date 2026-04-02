"""
Model Configurations for SPOD
============================

Defines layer ranges, dimensions, and other hyperparameters for each supported model.
All parameters are aligned with the paper's methodology.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for a specific video LLM."""
    
    # Model identifier
    model_id: str
    
    # Model architecture type
    arch: str  # "qwen2_vl", "videoLlama2", "videoLlava"
    
    # Total number of layers in the LLM backbone
    num_layers: int
    
    # Hidden dimension
    hidden_dim: int
    
    # FDA layer range: layers 8-16 (paper: Stage 2 region)
    fda_layer_start: int
    fda_layer_end: int
    
    # TDS extraction layer: layer 22 (paper: deep layer for inter-frame differences)
    tds_deep_layer: int
    
    # Subspace dimensions (paper: r=4, q=4)
    temporal_dims: int  # r: dimension of temporal subspace V_r
    prior_dims: int     # q: dimension of prior subspace P_q
    
    # Cross-attention fusion weight (paper: alpha=0.5)
    fda_alpha: float = 0.5
    
    # Temporal smoothing window size
    smooth_window: int = 3
    
    # Default frames to sample per video
    default_num_frames: int = 16
    
    # HuggingFace model path
    hf_path: str = ""


# =============================================================================
# Supported Models (aligned with paper)
# =============================================================================

MODEL_CONFIGS = {
    # ===== Qwen2.5-VL Models =====
    # Qwen2.5-VL-7B (72 layers)
    "Qwen/Qwen2.5-VL-7B-Instruct": ModelConfig(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        arch="qwen2_vl",
        num_layers=72,
        hidden_dim=3584,
        fda_layer_start=8,
        fda_layer_end=16,
        tds_deep_layer=22,
        temporal_dims=4,
        prior_dims=4,
        fda_alpha=0.5,
        smooth_window=3,
        default_num_frames=16,
        hf_path="Qwen/Qwen2.5-VL-7B-Instruct",
    ),
    
    # ===== VideoLlama2 Models =====
    # VideoLlama2-7B: adjust layers proportionally (32 layers total)
    # Qwen2.5-VL has 72 layers, FDA at 8-16 => relative position ~22% to 44% into Stage 2
    # VideoLlama2 has 32 layers, so FDA at ~7-14
    "LanguageBind/VideoLlama2-7B": ModelConfig(
        model_id="LanguageBind/VideoLlama2-7B",
        arch="videoLlama2",
        num_layers=32,
        hidden_dim=4096,
        fda_layer_start=7,
        fda_layer_end=14,
        tds_deep_layer=10,  # Proportionally: 22/72 * 32 ≈ 10
        temporal_dims=4,
        prior_dims=4,
        fda_alpha=0.5,
        smooth_window=3,
        default_num_frames=16,
        hf_path="LanguageBind/VideoLlama2-7B",
    ),
    
    # ===== VideoLlava Models =====
    # Video-LLaVA-7B: proportional adjustment (32 layers)
    "LanguageBind/VideoLlava-7B": ModelConfig(
        model_id="LanguageBind/VideoLlava-7B",
        arch="videoLlava",
        num_layers=32,
        hidden_dim=4096,
        fda_layer_start=7,
        fda_layer_end=14,
        tds_deep_layer=10,  # Same proportional adjustment
        temporal_dims=4,
        prior_dims=4,
        fda_alpha=0.5,
        smooth_window=3,
        default_num_frames=16,
        hf_path="LanguageBind/VideoLlava-7B",
    ),
}


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a model by ID.
    
    Args:
        model_id: HuggingFace model ID or local path
        
    Returns:
        ModelConfig for the model
        
    Raises:
        ValueError: If model is not supported
    """
    # Exact match
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    
    # Partial match
    for config_id, config in MODEL_CONFIGS.items():
        model_name = config_id.split("/")[-1].lower()
        if model_name in model_id.lower() or model_id.lower() in model_name:
            return config
    
    # Short name matching
    short_names = {
        "videollama2": "LanguageBind/VideoLlama2-7B",
        "videollava": "LanguageBind/VideoLlava-7B",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5vl7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    for key, config_id in short_names.items():
        if key in model_id.lower():
            return MODEL_CONFIGS[config_id]
    
    # Default fallback for Qwen2.5-VL
    if "qwen" in model_id.lower() and "vl" in model_id.lower():
        return MODEL_CONFIGS["Qwen/Qwen2.5-VL-7B-Instruct"]
    
    raise ValueError(
        f"Model '{model_id}' not supported. "
        f"Supported models: {list(MODEL_CONFIGS.keys())}"
    )


def list_supported_models() -> list[str]:
    """List all supported model IDs."""
    return list(MODEL_CONFIGS.keys())


def get_arch_model_ids(arch: str) -> list[str]:
    """Get all model IDs for a specific architecture."""
    return [
        model_id for model_id, config in MODEL_CONFIGS.items()
        if config.arch == arch
    ]
