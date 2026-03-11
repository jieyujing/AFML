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
    dates = pd.date_range('2023-01-01', periods=200, freq='min')
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Index should be preserved (though first few rows may be dropped)
    assert isinstance(ffd_series.index, pd.DatetimeIndex)


# ============== compute_ffd_volatility tests ==============

def test_compute_ffd_volatility_basic():
    """Test FFD volatility feature computation."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100), name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5, 10, 20])

    # Check all expected columns exist
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' in result.columns
    assert 'ffd_vol_ewm_5' in result.columns
    assert 'ffd_vol_ewm_10' in result.columns
    assert 'ffd_vol_ewm_20' in result.columns

    # Check data types
    assert all(result.dtypes == np.float64)


def test_compute_ffd_volatility_custom_spans():
    """Test FFD volatility with custom spans."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100))

    result = compute_ffd_volatility(ffd_series, spans=[3, 7])

    assert 'ffd_vol_std_3' in result.columns
    assert 'ffd_vol_std_7' in result.columns
    assert 'ffd_vol_ewm_3' in result.columns
    assert 'ffd_vol_ewm_7' in result.columns


def test_compute_ffd_volatility_preserves_index():
    """Test FFD volatility preserves datetime index."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='min')
    ffd_series = pd.Series(np.random.randn(100), index=dates, name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5])

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 100
