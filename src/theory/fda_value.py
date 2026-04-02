"""
FDA Signal Value Analysis — FDA-Selected vs. Random Frame Comparison
============================================================

Validates that FDA-selected frames contribute more to temporal reasoning
than uniformly random frames by comparing semantic coherence and delta metrics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.spod import compute_fda_frame_weights


def analyze_fda_frame_contribution(
    hidden_states: tuple,
    vision_seq_len: int,
    num_frames: int,
    layer_start: int = 9,
    layer_end: int = 28,
    deep_layer: int = 22,
    temporal_dims: int = 16,
    device: torch.device | None = None,
) -> dict:
    """Compare FDA-selected vs random frames on temporal metrics.

    Compares high-distinctiveness frames (top-K by FDA) against randomly
    selected frames using:
    - Semantic Coherence (SC): Mean pairwise cosine similarity
    - Delta Semantic Coherence (Delta-SC): Cosine similarity of delta vectors
    - Delta Magnitude: Mean L2 norm of frame deltas

    Args:
        hidden_states: Per-layer hidden state tuple.
        vision_seq_len: Number of vision tokens.
        num_frames: Number of sampled video frames (T).
        layer_start/end: FDA layer range (L).
        deep_layer: Deep layer for comparison (l*).
        temporal_dims: Passed to TDS (not used in P2 alone).
        device: Target device.

    Returns:
        Dict with FDA vs Random comparison metrics.
    """
    device = hidden_states[0].device
    tokens_per_frame = max(1, vision_seq_len // num_frames)
    K = max(1, num_frames // 4)

    # FDA weights and Top-K selection
    fda_weights = compute_fda_frame_weights(
        hidden_states=hidden_states,
        vision_seq_len=vision_seq_len,
        num_frames=num_frames,
        layer_start=layer_start,
        layer_end=layer_end,
        alpha=0.7,
        smooth_window=3,
        device=device,
    )
    _, fda_topk_indices = torch.topk(fda_weights, k=K)
    fda_topk_indices = fda_topk_indices.cpu().numpy().tolist()

    # Random selection (with seed for reproducibility)
    rng = np.random.RandomState(42)
    random_indices = sorted(
        rng.choice(num_frames, size=K, replace=False).tolist()
    )
    # Minimize overlap
    overlap = set(fda_topk_indices) & set(random_indices)
    if len(overlap) > K // 2:
        random_indices = [i for i in random_indices if i not in overlap]
        remaining = [
            i for i in range(num_frames)
            if i not in fda_topk_indices and i not in random_indices
        ]
        random_indices.extend(remaining[:K - len(random_indices)])

    # Per-layer frame representations
    def get_frame_hiddens(lidx: int) -> torch.Tensor:
        H = hidden_states[lidx].squeeze(0).float()
        frame_hs = []
        for t in range(num_frames):
            s = t * tokens_per_frame
            e = min(s + tokens_per_frame, vision_seq_len)
            if s >= vision_seq_len:
                s = max(0, vision_seq_len - 1)
                e = vision_seq_len
            frame_hs.append(H[s:e].mean(dim=0))
        return torch.stack(frame_hs, dim=0)

    def compute_metrics(indices: list[int], lidx: int) -> dict:
        H_layer = get_frame_hiddens(lidx)
        H_sel = H_layer[indices]

        # Semantic Coherence: mean pairwise cosine similarity
        H_norm = F.normalize(H_sel, p=2, dim=1)
        sc = torch.mm(H_norm, H_norm.T).triu(diagonal=1).mean().item()

        # Delta Semantic Coherence and Delta Magnitude
        if len(indices) >= 2:
            sorted_idx = sorted(indices)
            deltas = []
            for j in range(1, len(sorted_idx)):
                delta = H_sel[sorted_idx[j]] - H_sel[sorted_idx[j - 1]]
                deltas.append(delta)
            deltas = torch.stack(deltas, dim=0)

            delta_norms = F.normalize(deltas, p=2, dim=1)
            if delta_norms.shape[0] > 1:
                delta_sc = (
                    torch.mm(delta_norms, delta_norms.T).triu(diagonal=1).mean().item()
                )
            else:
                delta_sc = 0.0
            delta_mag = deltas.norm(p=2, dim=1).mean().item()
        else:
            delta_sc = 0.0
            delta_mag = 0.0

        return {"sc": sc, "delta_sc": delta_sc, "delta_mag": delta_mag}

    # Use deep layer for comparison
    fda_metrics = compute_metrics(fda_topk_indices, deep_layer)
    rand_metrics = compute_metrics(random_indices, deep_layer)

    return {
        "fda_topk_indices": fda_topk_indices,
        "random_indices": random_indices,
        "fda_semantic_coherence": fda_metrics["sc"],
        "random_semantic_coherence": rand_metrics["sc"],
        "fda_delta_sc": fda_metrics["delta_sc"],
        "random_delta_sc": rand_metrics["delta_sc"],
        "fda_delta_magnitude": fda_metrics["delta_mag"],
        "random_delta_magnitude": rand_metrics["delta_mag"],
        "fda_wins_sc": int(fda_metrics["sc"] > rand_metrics["sc"]),
        "fda_wins_delta_sc": int(fda_metrics["delta_sc"] < rand_metrics["delta_sc"]),
    }
