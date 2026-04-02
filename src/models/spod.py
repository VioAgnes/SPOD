"""
SPOD: Semantic Prior-Orthogonal Decomposition for Video-LLMs
==========================================================

FDA - Frame Distinctiveness Analysis
    Identifies temporally salient frames by measuring hidden-state discriminability
    across middle layers using pairwise cosine similarity variance.

TDS - Temporal Delta Subspace
    Constructs an orthogonal temporal subspace (V_r) and non-temporal subspace (P_q)
    from weighted inter-frame hidden-state differences.

Subspace Decomposition and Enhancement
    Decomposes hidden states via orthogonal projections onto temporal, non-temporal,
    and residual subspaces, enabling independent scaling for temporal signal recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD


# ----------------------------------------------------------------------
# Default hyperparameters (per-subset tuned)
# ----------------------------------------------------------------------

@dataclass
class FDAConfig:
    """FDA (Frame Distinctiveness Analysis) hyper-parameters.
    
    Paper defaults: layers 8-16, alpha=0.5 (cross-attention fusion weight).
    """
    layer_range: tuple = (8, 16)   # FDA layer range (Stage 2 region)
    alpha: float = 0.5            # Cross-attention fusion weight
    smooth_window: int = 3        # Smoothing window size


@dataclass
class TDSConfig:
    """TDS (Temporal Delta Subspace) hyper-parameters.
    
    Paper defaults: deep_layer=22, r=4, q=4.
    """
    deep_layer: int = 22          # Layer for inter-frame difference extraction
    temporal_dims: int = 4         # Dimension r of temporal subspace V_r
    prior_dims: int = 4           # Dimension q of non-temporal subspace P_q


@dataclass
class SPODConfig:
    """Combined SPOD hyper-parameters with enhancement scaling factors.
    
    Paper defaults: gamma_T=1.2, gamma_P=0.5, gamma_R=1.0.
    """
    fda: FDAConfig = field(default_factory=FDAConfig)
    tds: TDSConfig = field(default_factory=TDSConfig)
    
    # Enhancement scaling factors (gamma values from paper)
    gamma_temporal: float = 1.2    # gamma_T: enhance temporal evidence
    gamma_prior: float = 0.5       # gamma_P: suppress non-temporal component
    gamma_residual: float = 1.0    # gamma_R: preserve residual unchanged

    def for_subset(self, subset: Literal["perception", "comprehension", "adaptation", "all"]) -> tuple:
        return self.fda, self.tds


# ----------------------------------------------------------------------
# Phase 1: FDA - Frame Distinctiveness Analysis
# ----------------------------------------------------------------------

def compute_cross_attention_weights(
    attentions: tuple | None,
    mm_token_type_ids: torch.Tensor | None,
    vision_seq_len: int,
    num_frames: int,
    layer_start: int = 9,
    layer_end: int = 28,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract cross-modal attention as per-frame textual relevance scores (a_i).

    Cross-attention from text tokens to video tokens provides textual relevance
    signal for weight refinement (Eq. in methodology).
    """
    if mm_token_type_ids is None or vision_seq_len <= 0:
        device = device or torch.device("cpu")
        n = max(1, num_frames)
        return torch.ones(n, device=device) / n

    device = device or torch.device("cpu")
    mm_flat = mm_token_type_ids.flatten().long()
    vis_pos = (mm_flat == 1).nonzero(as_tuple=True)[0]
    txt_pos = (mm_flat == 0).nonzero(as_tuple=True)[0]

    if len(vis_pos) == 0:
        n = max(1, num_frames)
        return torch.ones(n, device=device) / n

    num_vis = len(vis_pos)
    tokens_per_frame = max(1, num_vis // num_frames)
    vis_pos_t = vis_pos.to(device)
    txt_pos_t = txt_pos.to(device)

    fda_layer_range = range(layer_start, min(layer_end + 1, len(attentions) if attentions else 0))

    # Strategy 1: text->vision cross-attention
    cross_attn_available = (
        attentions is not None
        and len(attentions) > 0
        and attentions[layer_start] is not None
        and len(txt_pos) > 0
    )

    if cross_attn_available:
        cross_per_layer = []
        for lidx in fda_layer_range:
            attn = attentions[lidx][0].to(device)
            # Text queries attending to vision keys
            txt_q_vis_k = attn[:, txt_pos_t][:, :, vis_pos_t]
            per_vis = txt_q_vis_k.mean(dim=(0, 1))
            cross_per_layer.append(per_vis)

        if cross_per_layer:
            a_vis = torch.stack(cross_per_layer, dim=0).mean(dim=0)
        else:
            a_vis = None
    else:
        a_vis = None

    # Strategy 2: fallback to vision->vision self-attention
    if a_vis is None and len(vis_pos) >= 2:
        vis_to_vis_per_layer = []
        for lidx in fda_layer_range:
            if attentions is None or attentions[lidx] is None:
                continue
            attn = attentions[lidx][0].to(device)
            v2v = attn[:, vis_pos_t][:, :, vis_pos_t]
            per_vis = v2v.mean(dim=(0, 1))
            vis_to_vis_per_layer.append(per_vis)

        if vis_to_vis_per_layer:
            a_vis = torch.stack(vis_to_vis_per_layer, dim=0).mean(dim=0)

    # Ultimate fallback: uniform
    if a_vis is None:
        n = max(1, num_frames)
        return torch.ones(n, device=device) / n

    # Aggregate per vision token -> per frame (a_i)
    a_t_raw = torch.zeros(num_frames, device=device)
    for f in range(num_frames):
        start = f * tokens_per_frame
        end = min(start + tokens_per_frame, num_vis)
        if start < num_vis:
            a_t_raw[f] = a_vis[start:end].mean()

    a_t = a_t_raw / (a_t_raw.sum() + 1e-8)
    if a_t.isnan().any() or a_t.le(0).all():
        return torch.ones(num_frames, device=device) / num_frames

    return a_t


def compute_fda_frame_weights(
    hidden_states: tuple,
    vision_seq_len: int,
    num_frames: int,
    layer_start: int = 9,
    layer_end: int = 28,
    alpha: float = 0.7,
    smooth_window: int = 3,
    device: torch.device | None = None,
    attentions: tuple | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """FDA: Compute per-frame temporal salience weights.

    Frame Distinctiveness Analysis identifies temporally salient frames by:
    1. Computing pairwise cosine similarity matrix S at each layer
    2. Computing variance of similarities (phi_i) per frame
    3. Smoothing and aggregating across layers (phi_i_smooth)
    4. Softmax normalization to get frame weights (w_i)
    5. Attention-level refinement (w_hat_i) if available

    Args:
        hidden_states: Tuple of per-layer hidden tensors, shape (1, seq_len, d).
        vision_seq_len: Number of vision tokens in the sequence.
        num_frames: Number of sampled video frames (T).
        layer_start/end: Range of intermediate layers for FDA (L).
        alpha: Balance between hidden-state and cross-attention weights.
        smooth_window: Temporal smoothing window size.
        device: Target device.
        attentions: Optional per-layer attention tensors.
        mm_token_type_ids: Token type tensor (1=vision, 0=text).

    Returns:
        Refined frame weights tensor w_hat, shape (num_frames,), summing to 1.
    """
    if vision_seq_len <= 0 or num_frames <= 0:
        device = device or torch.device("cpu")
        n = max(1, num_frames)
        return torch.ones(n, device=device) / n

    device = hidden_states[0].device
    tokens_per_frame = max(1, vision_seq_len // num_frames)
    
    # Layer-wise variance phi_i^l for each frame
    phi_per_layer = []

    for layer_idx in range(layer_start, min(layer_end + 1, len(hidden_states))):
        H = hidden_states[layer_idx].squeeze(0).float()
        
        # Compute per-frame hidden states H_i
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
        
        # Normalize hidden states (H_tilde_i = H_i / ||H_i||)
        H_tilde = F.normalize(frame_hidden_states, p=2, dim=1)
        
        # Pairwise cosine similarity matrix S
        S = torch.mm(H_tilde, H_tilde.T)
        
        # Variance of similarities in row i: phi_i = Var(S[i, :])
        # Using maximum likelihood estimator (biased)
        row_means = S.mean(dim=1, keepdim=True)
        phi_i = ((S - row_means) ** 2).mean(dim=1)
        phi_i = torch.clamp(phi_i, min=1e-12)
        phi_per_layer.append(phi_i)

    if not phi_per_layer:
        return torch.ones(num_frames, device=device) / num_frames

    # Aggregate across layers: mean_phi_i = (1/|L|) sum_l phi_i^l
    phi_avg = torch.stack(phi_per_layer, dim=0).mean(dim=0)

    # Smoothing window of size 3 (Eq. smooth_phi)
    if smooth_window > 1 and num_frames > 1:
        phi_smooth = phi_avg.clone()
        half = smooth_window // 2
        for t in range(num_frames):
            indices = [i for i in range(t - half, t + half + 1)
                     if 0 <= i < num_frames]
            phi_smooth[t] = phi_avg[indices].mean()
        phi_avg = phi_smooth

    # Softmax normalization to get frame weights w_i (Eq. frame_weights)
    phi_std = phi_avg.std()
    if phi_std < 1e-6:
        w_i = torch.ones(num_frames, device=device) / num_frames
    else:
        w_i = torch.softmax(phi_avg / (phi_std + 1e-8), dim=0)

    # Attention-level refinement (Eq. refined_weights)
    if attentions is not None and mm_token_type_ids is not None:
        a_i = compute_cross_attention_weights(
            attentions=attentions,
            mm_token_type_ids=mm_token_type_ids,
            vision_seq_len=vision_seq_len,
            num_frames=num_frames,
            layer_start=layer_start,
            layer_end=layer_end,
            device=device,
        )
    else:
        a_i = torch.ones(num_frames, device=device) / num_frames

    # Final refined weight: w_hat_i = alpha * w_i + (1-alpha) * a_i
    w_hat = alpha * w_i + (1 - alpha) * a_i
    w_hat = w_hat / (w_hat.sum() + 1e-8)

    if w_hat.isnan().any() or (w_hat <= 0).any():
        return torch.ones(num_frames, device=device) / num_frames

    return w_hat


# ----------------------------------------------------------------------
# Phase 2: TDS - Temporal Delta Subspace
# ----------------------------------------------------------------------

def _safe_randomized_svd(M: np.ndarray, n_components: int) -> np.ndarray:
    """Robust SVD with fallback to QR decomposition."""
    try:
        svd = TruncatedSVD(
            n_components=n_components, algorithm="randomized",
            n_iter=10, random_state=42
        )
        _ = svd.fit(M)
        return svd.components_.T
    except Exception:
        pass
    try:
        _, _, Vh = np.linalg.svd(M, full_matrices=False)
        return Vh[:n_components].T
    except Exception:
        pass
    d_dim = M.shape[1]
    Q = np.random.randn(d_dim, n_components).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    return Q


def build_tds_subspaces(
    hidden_states: tuple,
    vision_seq_len: int,
    num_frames: int,
    frame_weights: torch.Tensor,
    deep_layer: int = 22,
    temporal_dims: int = 16,
    prior_dims: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TDS: Build temporal subspace V_r and non-temporal subspace P_q.

    Temporal Delta Subspace construction:
    1. Extract hidden states at deep layer l*
    2. Compute weighted inter-frame differences: delta_i = w_hat_i * (H_i - H_{i-1})
    3. Apply truncated SVD to get temporal subspace V_r
    4. Project text states onto orthogonal complement and SVD for P_q

    Args:
        hidden_states: Per-layer hidden state tuple.
        vision_seq_len: Number of vision tokens in the sequence.
        num_frames: Number of sampled video frames (T).
        frame_weights: FDA refined weights w_hat, shape (num_frames,).
        deep_layer: Deep layer near Stage 2/3 transition (l*).
        temporal_dims: Dimension r of temporal subspace V_r.
        prior_dims: Dimension q of non-temporal subspace P_q.

    Returns:
        (V_r, P_q) where V_r in R^{d x r}, P_q in R^{d x q}.
    """
    raw_dtype = hidden_states[0].dtype
    compute_dtype = torch.float32
    device = hidden_states[0].device
    tokens_per_frame = max(1, vision_seq_len // num_frames)

    # Extract hidden states at deep layer l*
    H_deep = hidden_states[deep_layer].squeeze(0).float()
    w_hat = frame_weights.to(compute_dtype)

    # Compute per-frame hidden states H_i^{(l*)}
    frame_hidden = []
    for t in range(num_frames):
        start = t * tokens_per_frame
        end = min(start + tokens_per_frame, vision_seq_len)
        if start >= vision_seq_len:
            start = max(0, vision_seq_len - 1)
            end = vision_seq_len
        frame_hidden.append(H_deep[start:end].mean(dim=0))

    frame_hidden = torch.stack(frame_hidden, dim=0)

    # Compute weighted inter-frame differences: delta_i = w_hat_i * (H_i - H_{i-1})
    deltas = []
    for t in range(1, num_frames):
        delta = w_hat[t] * (frame_hidden[t] - frame_hidden[t - 1])
        deltas.append(delta)

    if len(deltas) == 0:
        d = frame_hidden.shape[-1]
        V_r = torch.eye(d, temporal_dims, device=device, dtype=compute_dtype)
        P_q = torch.zeros(d, prior_dims, device=device, dtype=compute_dtype)
        return V_r.to(raw_dtype), P_q.to(raw_dtype)

    # Build delta matrix Delta in R^{(T-1) x d}
    Delta = torch.stack(deltas, dim=0).cpu().numpy().astype(np.float32)
    d_dim = Delta.shape[1]
    
    # Truncated SVD: Delta = U * Sigma * V^T, V_r = V^{(1:r)}
    r = min(temporal_dims, Delta.shape[0], d_dim)
    V_r_np = _safe_randomized_svd(Delta, r)
    V_r = torch.from_numpy(V_r_np).to(device=device, dtype=compute_dtype)

    # Non-temporal subspace construction from text tokens
    # Project text hidden states onto orthogonal complement of V_r
    H_full = hidden_states[deep_layer].squeeze(0).float()
    text_hidden = H_full[vision_seq_len:] if vision_seq_len < H_full.shape[0] else H_full[-1:, :]
    text_np = text_hidden.cpu().numpy().astype(np.float32)

    if text_np.shape[0] < 2:
        P_np = np.random.randn(d_dim, prior_dims).astype(np.float32)
        P_np, _ = np.linalg.qr(P_np)
        return V_r.to(raw_dtype), torch.from_numpy(P_np).to(
            device=device, dtype=compute_dtype
        ).to(raw_dtype)

    # T_perp = T * (I - V_r * V_r^T)
    I_np = np.eye(d_dim, dtype=np.float32)
    proj_Vr = V_r_np @ V_r_np.T
    orthogonal_np = I_np - proj_Vr
    T_perp_np = text_np @ orthogonal_np

    # SVD on T_perp: T_perp = U' * Sigma' * V'^T, P_q = V'^{(1:q)}
    q = min(prior_dims, T_perp_np.shape[0], T_perp_np.shape[1])
    try:
        pca = PCA(n_components=q, random_state=42)
        pca.fit(T_perp_np)
        P_np = pca.components_.T
    except Exception:
        try:
            _, _, Vh_p = np.linalg.svd(T_perp_np, full_matrices=False)
            P_np = Vh_p[:q].T
        except Exception:
            P_np = np.random.randn(d_dim, q).astype(np.float32)
            P_np, _ = np.linalg.qr(P_np)

    P_q = torch.from_numpy(P_np).to(device=device, dtype=compute_dtype)
    return V_r.to(raw_dtype), P_q.to(raw_dtype)


# ----------------------------------------------------------------------
# Phase 3: Subspace Decomposition and Enhancement
# ----------------------------------------------------------------------

def decompose_hidden_state(
    h: torch.Tensor,
    V_r: torch.Tensor,
    P_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose hidden state via orthogonal projections.

    Given hidden state h, decompose into three mutually orthogonal components:
    - h_T: temporal component (projection onto V_r)
    - h_P: non-temporal/prior component (projection onto P_q)
    - h_R: residual component (orthogonal to both)

    Args:
        h: Hidden state tensor, shape (..., d).
        V_r: Temporal subspace basis, shape (d, r).
        P_q: Non-temporal subspace basis, shape (d, q).

    Returns:
        (h_T, h_P, h_R) - temporal, prior, and residual components.
    """
    # Temporal component: h_T = V_r * V_r^T * h
    if V_r is not None:
        h_T = torch.matmul(h, V_r)  # (..., d) -> (..., r)
        h_T = torch.matmul(h_T, V_r.T)  # (..., r) -> (..., d)
    else:
        h_T = torch.zeros_like(h)

    # Prior component: h_P = P_q * P_q^T * h
    if P_q is not None:
        h_P = torch.matmul(h, P_q)  # (..., d) -> (..., q)
        h_P = torch.matmul(h_P, P_q.T)  # (..., q) -> (..., d)
    else:
        h_P = torch.zeros_like(h)

    # Residual component: h_R = h - h_T - h_P
    h_R = h - h_T - h_P

    return h_T, h_P, h_R


def enhance_hidden_state(
    h: torch.Tensor,
    V_r: torch.Tensor,
    P_q: torch.Tensor,
    gamma_T: float = 1.0,
    gamma_P: float = 0.5,
    gamma_R: float = 1.0,
) -> torch.Tensor:
    """Enhance hidden state by scaling temporal and prior components.

    h_hat = gamma_T * h_T + gamma_P * h_P + gamma_R * h_R

    Args:
        h: Original hidden state, shape (..., d).
        V_r: Temporal subspace basis, shape (d, r).
        P_q: Non-temporal subspace basis, shape (d, q).
        gamma_T: Temporal scaling factor (default: 1.0).
        gamma_P: Prior scaling factor (default: 0.5).
        gamma_R: Residual scaling factor (default: 1.0).

    Returns:
        Enhanced hidden state h_hat, same shape as h.
    """
    h_T, h_P, h_R = decompose_hidden_state(h, V_r, P_q)
    h_enhanced = gamma_T * h_T + gamma_P * h_P + gamma_R * h_R
    return h_enhanced


def apply_spod_to_logits(
    hidden_states: tuple,
    V_r: torch.Tensor,
    P_q: torch.Tensor,
    lm_head: torch.nn.Linear,
    gamma_T: float = 1.0,
    gamma_P: float = 0.5,
    gamma_R: float = 1.0,
) -> torch.Tensor:
    """Apply SPOD enhancement to logits from final hidden state.

    This function decomposes the final hidden state and applies scaling
    before passing through the language model head.

    Args:
        hidden_states: Tuple of per-layer hidden states.
        V_r: Temporal subspace basis.
        P_q: Non-temporal subspace basis.
        lm_head: Language model head (linear layer).
        gamma_T: Temporal scaling factor.
        gamma_P: Prior scaling factor.
        gamma_R: Residual scaling factor.

    Returns:
        Enhanced logits tensor.
    """
    # Get final hidden state (h_final)
    h_final = hidden_states[-1][:, -1, :]  # (batch, d)

    # Decompose and enhance
    h_enhanced = enhance_hidden_state(
        h_final, V_r, P_q,
        gamma_T=gamma_T, gamma_P=gamma_P, gamma_R=gamma_R
    )

    # Apply language model head
    logits = torch.matmul(h_enhanced, lm_head.weight.t())
    if lm_head.bias is not None:
        logits = logits + lm_head.bias

    return logits.unsqueeze(1)  # (batch, 1, vocab_size)
