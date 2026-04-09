"""
AL9999 Primary Model Factory.

Two-stage parameter search:
- CUSUM calibration (rate → k lookup)
- Lightweight scoring (90 combos: Recall/CPR/Coverage/Lift)
- Deep scoring (Top-20: Uniqueness/Turnover/Regime/OOS)
- Composite scoring → Top-5 candidates for Meta Model
"""

from .cusum_calibrator import calibrate_cusum_rates
from .param_grid import (
    expand_deep_param_grid,
    generate_lightweight_param_grid,
    generate_param_grid,
)
from .lightweight_scorer import compute_lightweight_metrics, compute_all_lightweight_metrics
from .deep_scorer import compute_deep_metrics, compute_all_deep_metrics
from .scorer import compute_composite_score, get_top_candidates
from .runner import run_primary_factory

__all__ = [
    "calibrate_cusum_rates",
    "generate_lightweight_param_grid",
    "expand_deep_param_grid",
    "generate_param_grid",
    "compute_lightweight_metrics",
    "compute_all_lightweight_metrics",
    "compute_deep_metrics",
    "compute_all_deep_metrics",
    "compute_composite_score",
    "get_top_candidates",
    "run_primary_factory",
]
