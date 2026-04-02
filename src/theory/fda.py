"""
FDA Signal Analysis — Frame Distinctiveness Validation
=================================================

Validates that intermediate-layer hidden states encode per-frame distinctiveness
signals (phi_i) through pairwise cosine similarity variance.

This corresponds to the FDA analysis in the methodology paper.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def analyze_fda_signal(
    hidden_states: tuple,
    vision_seq_len: int,
    num_frames: int,
    layer_start: int = 9,
    layer_end: int = 28,
    smooth_window: int = 3,
) -> dict:
    """Analyze FDA signals from intermediate-layer hidden states.

    Computes per-frame distinctiveness scores phi_i by:
    1. Computing pairwise cosine similarity matrix at each layer
    2. Computing variance of similarities per frame (phi_i^l)
    3. Aggregating across layers and smoothing

    Args:
        hidden_states: Per-layer hidden state tuple, each shape (1, seq_len, d).
        vision_seq_len: Number of vision tokens.
        num_frames: Number of sampled video frames (T).
        layer_start/end: Range of layers for FDA analysis (L).
        smooth_window: Temporal smoothing window size.

    Returns:
        Dict with keys:
          phi_t_avg: Cross-layer mean distinctiveness scores per frame.
          phi_t_std: Layer-wise std of phi_i.
          per_layer_phi: Dict of per-layer phi_i values.
          high_distinctiveness_frames: Frame indices above median distinctiveness.
          low_distinctiveness_frames: Frame indices below/equal median.
          peak_indices: Local maxima (potential semantic boundaries).
          layer_high_ratio: Ratio of high-phi frames per layer.
          variance_ratio: High-phi mean / Low-phi mean per layer.
          num_layers_analyzed: Number of layers used.
          layer_range: [layer_start, effective_layer_end].
    """
    device = hidden_states[0].device
    tokens_per_frame = max(1, vision_seq_len // num_frames)

    # Per-layer distinctiveness phi_i^l
    per_layer_phi = {}
    per_layer_variance_ratio = {}

    for layer_idx in range(layer_start, min(layer_end + 1, len(hidden_states))):
        H = hidden_states[layer_idx].squeeze(0).float()
        
        # Per-frame hidden states
        frame_hidden_states = []
        for t in range(num_frames):
            start = t * tokens_per_frame
            end = min(start + tokens_per_frame, vision_seq_len)
            if start >= vision_seq_len:
                start = max(0, vision_seq_len - 1)
                end = vision_seq_len
            frame_h = H[start:end].mean(dim=0)
            frame_hidden_states.append(frame_h)

        frame_hidden_states = torch.stack(frame_hidden_states, dim=0)
        
        # Normalize and compute similarity matrix
        H_tilde = F.normalize(frame_hidden_states, p=2, dim=1)
        S = torch.mm(H_tilde, H_tilde.T)
        
        # Variance of similarities: phi_i = Var(S[i, :])
        row_means = S.mean(dim=1, keepdim=True)
        phi_i = ((S - row_means) ** 2).mean(dim=1)
        
        per_layer_phi[layer_idx] = phi_i.cpu().numpy()

        # High/Low distinctiveness variance ratio
        sorted_phi = torch.sort(phi_i).values
        top_k = max(1, num_frames // 4)
        high_phi_mean = sorted_phi[-top_k:].mean().item()
        low_phi_mean = sorted_phi[:top_k].mean().item()
        per_layer_variance_ratio[layer_idx] = (
            high_phi_mean / (low_phi_mean + 1e-8)
        )

    # Cross-layer average phi_bar_i
    all_phi = np.array([per_layer_phi[l] for l in sorted(per_layer_phi.keys())])
    phi_t_avg = all_phi.mean(axis=0)
    phi_t_std = all_phi.std(axis=0)

    # Temporal smoothing
    if smooth_window > 1 and num_frames > 1:
        phi_smooth = np.copy(phi_t_avg)
        half = smooth_window // 2
        for t in range(num_frames):
            indices = [i for i in range(t - half, t + half + 1)
                     if 0 <= i < num_frames]
            phi_smooth[t] = phi_t_avg[indices].mean()
        phi_t_avg = phi_smooth

    # High / Low distinctiveness classification
    phi_threshold = np.median(phi_t_avg)
    high_distinctiveness_frames = np.where(phi_t_avg > phi_threshold)[0].tolist()
    low_distinctiveness_frames = np.where(phi_t_avg <= phi_threshold)[0].tolist()

    # Peak detection — local maxima
    peak_indices = []
    if num_frames > 2:
        phi_diff = np.diff(phi_t_avg)
        zero_crossings = np.where(np.diff(np.sign(phi_diff)) < 0)[0]
        for zc in zero_crossings:
            if 1 <= zc < num_frames - 1:
                peak_indices.append(int(zc + 1))

    # Layer-wise high-phi ratio
    layer_high_ratio = {}
    for layer_idx in sorted(per_layer_phi.keys()):
        phi_layer = per_layer_phi[layer_idx]
        threshold = np.median(phi_layer)
        ratio = (phi_layer > threshold).sum() / len(phi_layer)
        layer_high_ratio[str(layer_idx)] = float(ratio)

    return {
        "phi_t_avg": phi_t_avg.tolist(),
        "phi_t_std": phi_t_std.tolist(),
        "per_layer_phi": {str(k): v.tolist() for k, v in per_layer_phi.items()},
        "high_distinctiveness_frames": high_distinctiveness_frames,
        "low_distinctiveness_frames": low_distinctiveness_frames,
        "peak_indices": peak_indices,
        "layer_high_ratio": layer_high_ratio,
        "variance_ratio": {str(k): v for k, v in per_layer_variance_ratio.items()},
        "num_layers_analyzed": len(per_layer_phi),
        "layer_range": [layer_start, min(layer_end, len(hidden_states) - 1)],
    }
