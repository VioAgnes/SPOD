"""Numerical helpers for orthogonal evidence decomposition."""

import functools
import torch


def precise_projection(fn):

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    return wrapped


def svd_basis(x, rank):
    _, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    n = min(rank, int((s > s[0].clamp_min(1e-12) * 1e-05).sum()))
    assert n > 0
    u = torch.linalg.qr(vh[:n].T.contiguous(), mode="reduced").Q
    error = float((u.T @ u - torch.eye(n, device=u.device)).abs().max())
    assert error < 0.0001, error
    return u


def decompose(x, vt, vp):
    hT = x @ vt @ vt.T
    hP = x @ vp @ vp.T
    hR = x - hT - hP
    return (hT, hP, hR)
