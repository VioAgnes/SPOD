"""
TDS Subspace Analysis — Temporal Delta Subspace Validation
====================================================

Validates that the Temporal Delta Subspace has low intrinsic dimensionality
by analyzing effective rank and variance explained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.spod import build_tds_subspaces


def compute_svd_cumulative_variance(S: np.ndarray) -> dict:
    """Compute SVD cumulative variance explained from singular values.

    Args:
        S: Singular values from np.linalg.svd(..., compute_uv=False).

    Returns:
        Dict with effective rank and top-k variance ratios.
    """
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumsum = np.cumsum(variance_explained)
    p_k = variance_explained + 1e-12
    p_k = p_k / p_k.sum()
    effective_rank = float(np.exp(-np.sum(p_k * np.log(p_k))))

    k_values = [1, 3, 5, 10]
    top_k_ratios = {k: float(cumsum[min(k - 1, len(cumsum) - 1)]) for k in k_values}

    return {
        "explained_variance_ratio_cum": cumsum.tolist(),
        "singular_values": S.tolist(),
        "effective_rank": effective_rank,
        "top_k_ratios": top_k_ratios,
    }


def compute_delta_direction_consistency(delta_matrix: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity of normalized delta vectors.

    Values close to 0 indicate that delta vectors are near-orthogonal,
    capturing diverse types of temporal changes.
    """
    if delta_matrix.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(delta_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    delta_normalized = delta_matrix / norms
    cos_sim = np.dot(delta_normalized, delta_normalized.T)
    n = cos_sim.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return float(cos_sim[mask].mean())


def analyze_tds_subspace(
    hidden_states: tuple,
    vision_seq_len: int,
    num_frames: int,
    frame_weights: torch.Tensor | None = None,
    deep_layer: int = 22,
    temporal_dims: int = 16,
    prior_dims: int = 8,
    layer_start: int = 9,
    layer_end: int = 28,
) -> dict:
    """Analyze the structure of the Temporal Delta Subspace.

    Validates TDS properties:
    - Effective rank of delta matrix (should be << temporal_dims)
    - Top-k variance explained (should saturate quickly)
    - Delta direction consistency (should be near 0)

    Args:
        hidden_states: Per-layer hidden state tuple.
        vision_seq_len: Number of vision tokens.
        num_frames: Number of sampled video frames (T).
        frame_weights: FDA weights; if None, recomputed automatically.
        deep_layer: Deep layer near Stage 2/3 transition (l*).
        temporal_dims: Dimension r of temporal subspace V_r.
        prior_dims: Dimension q of non-temporal subspace P_q.
        layer_start/end: Range of layers for FDA.

    Returns:
        Dict with TDS analysis metrics.
    """
    if frame_weights is None:
        from ..models.spod import compute_fda_frame_weights
        frame_weights = compute_fda_frame_weights(
            hidden_states=hidden_states,
            vision_seq_len=vision_seq_len,
            num_frames=num_frames,
            layer_start=layer_start,
            layer_end=layer_end,
            alpha=0.7,
            smooth_window=3,
        )

    tokens_per_frame = max(1, vision_seq_len // num_frames)
    device = hidden_states[0].device
    compute_dtype = torch.float32

    # Extract hidden states at deep layer l*
    H_deep = hidden_states[deep_layer].squeeze(0).float()
    w_hat = frame_weights.to(compute_dtype)

    # Per-frame hidden states
    frame_hidden = []
    for t in range(num_frames):
        start = t * tokens_per_frame
        end = min(start + tokens_per_frame, vision_seq_len)
        if start >= vision_seq_len:
            start = max(0, vision_seq_len - 1)
            end = vision_seq_len
        frame_hidden.append(H_deep[start:end].mean(dim=0))

    frame_hidden = torch.stack(frame_hidden, dim=0)

    # Compute weighted inter-frame differences
    deltas = []
    for t in range(1, num_frames):
        delta = w_hat[t] * (frame_hidden[t] - frame_hidden[t - 1])
        deltas.append(delta)

    if len(deltas) == 0:
        return {
            "effective_rank": 0.0,
            "svd_cumulative": {},
            "delta_direction_consistency": 0.0,
            "delta_magnitude_mean": 0.0,
            "delta_magnitude_std": 0.0,
            "temporal_dims_used": 0,
        }

    # Delta matrix Delta in R^{(T-1) x d}
    Delta = torch.stack(deltas, dim=0).cpu().numpy().astype(np.float32)
    delta_mags = np.linalg.norm(Delta, axis=1)

    # SVD analysis
    _, S, _ = np.linalg.svd(Delta, full_matrices=False)
    svd_info = compute_svd_cumulative_variance(S)
    delta_dir_consistency = compute_delta_direction_consistency(Delta)

    r_used = min(temporal_dims, Delta.shape[0], Delta.shape[1])
    V_r_np, _ = build_tds_subspaces(
        hidden_states=hidden_states,
        vision_seq_len=vision_seq_len,
        num_frames=num_frames,
        frame_weights=frame_weights,
        deep_layer=deep_layer,
        temporal_dims=temporal_dims,
        prior_dims=prior_dims,
    )
    V_r_np_loaded = V_r_np.cpu().numpy()

    return {
        "effective_rank": svd_info["effective_rank"],
        "svd_cumulative": svd_info,
        "delta_direction_consistency": delta_dir_consistency,
        "delta_magnitude_mean": float(delta_mags.mean()),
        "delta_magnitude_std": float(delta_mags.std()),
        "temporal_dims_used": r_used,
        "V_r_shape": list(V_r_np_loaded.shape),
    }
