"""
SPOD Theory Module
=================
"""

from .fda import analyze_fda_signal
from .fda_value import analyze_fda_frame_contribution
from .tds import analyze_tds_subspace, compute_svd_cumulative_variance

__all__ = [
    "analyze_fda_signal",
    "analyze_fda_frame_contribution",
    "analyze_tds_subspace",
    "compute_svd_cumulative_variance",
]
