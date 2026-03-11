"""Tests for Alpha158 FFD features module."""
import numpy as np
import pandas as pd
import pytest


def test_compute_ffd_base_basic():
    """Test basic FFD base series computation."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Check return types
    assert isinstance(ffd_series, pd.Series)
    assert isinstance(optimal_d, float)

    # Check optimal_d is in valid range
    assert 0.0 <= optimal_d <= 1.0

    # Check series length (should be <= input due to FFD lag)
    assert len(ffd_series) <= len(close)
    assert len(ffd_series) > 0

    # Check no NaN in result
    assert ffd_series.isna().sum() == 0


def test_compute_ffd_base_already_stationary():
    """Test FFD returns d=0 for already stationary series."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    stationary = pd.Series(np.random.randn(200))

    ffd_series, optimal_d = compute_ffd_base(stationary)

    # Should return d=0 or very small
    assert optimal_d < 0.1


def test_compute_ffd_base_custom_params():
    """Test FFD with custom threshold and d_step."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(
        close,
        thres=1e-5,
        d_step=0.1
    )

    assert isinstance(ffd_series, pd.Series)
    assert 0.0 <= optimal_d <= 1.0


def test_compute_ffd_base_preserves_index():
    """Test FFD preserves datetime index."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='T')
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Index should be preserved (though first few rows may be dropped)
    assert isinstance(ffd_series.index, pd.DatetimeIndex)
