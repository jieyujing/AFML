import pytest
import numpy as np
from afmlkit.feature.core.structural_break.adf import (
    _ols_coefficients,
    _adf_regression_core,
    adf_test,
    adf_test_rolling,
)


def test_ols_coefficients_basic():
    """Test OLS regression with known coefficients."""
    x = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
    y = np.array([2, 5, 8, 11, 14], dtype=np.float64)
    coef, residuals = _ols_coefficients(x, y)
    assert np.allclose(coef, [2, 3], atol=1e-10)


def test_adf_regression_core_trend():
    """Test ADF regression on mean-reverting series with trend."""
    np.random.seed(42)
    # Generate mean-reverting series (stationary around trend)
    n = 100
    y = np.zeros(n, dtype=np.float64)
    y[0] = 100.0
    for i in range(1, n):
        y[i] = 0.5 * (100 + 0.1 * i) + 0.5 * y[i-1] + np.random.randn() * 0.5

    t_stat, p_value = _adf_regression_core(y, max_lag=0, trend=True)
    # For stationary series, t_stat should be significantly negative
    assert t_stat < 0
    assert 0 <= p_value <= 1


def test_adf_regression_core_random_walk():
    """Test ADF on random walk (unit root present)."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 100
    t_stat, p_value = _adf_regression_core(y, max_lag=0, trend=False)
    assert abs(t_stat) < 2.0


def test_adf_test_basic():
    """Test public adf_test function."""
    np.random.seed(42)
    # Use random walk (unit root) as test case
    y = np.cumsum(np.random.randn(50)) + 100
    t_stat, p_value, lag = adf_test(y, max_lag=4, trend=True)
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)
    assert isinstance(lag, int)
    assert not np.isnan(t_stat)
    assert -10 < t_stat < 10


def test_adf_test_rolling():
    """Test rolling ADF calculation."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(200)) + 100
    window = 50
    t_stats = adf_test_rolling(y, window=window, max_lag=2, trend=True)
    assert len(t_stats) == len(y)
    assert np.all(np.isnan(t_stats[:window-1]))
    assert np.all(np.isfinite(t_stats[window:]))
