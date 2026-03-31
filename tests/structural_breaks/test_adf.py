import pytest
import numpy as np
from afmlkit.feature.core.structural_break.adf import (
    _ols_coefficients,
    _adf_regression_core,
    adf_test,
    adf_test_rolling,
    schwert_maxlag,
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


def test_schwert_maxlag():
    """Test Schwert formula for maxlag calculation."""
    # Known values from Schwert formula: int(12 * (n/100)**(1/4))
    assert schwert_maxlag(100) == 12      # 12 * 1.0 = 12
    assert schwert_maxlag(25) == 8        # 12 * 0.7071 = 8.49 -> int = 8
    assert schwert_maxlag(1000) == 21     # 12 * 1.7783 = 21.34 -> int = 21
    assert schwert_maxlag(50) == 10       # 12 * 0.8409 = 10.09 -> int = 10

    # Edge case: minimum maxlag
    assert schwert_maxlag(10) >= 1


def test_compute_aic():
    """Test AIC calculation."""
    from afmlkit.feature.core.structural_break.adf import _compute_aic

    # AIC = n * log(rss/n) + 2 * k
    # Case 1: n=100, k=3, rss=10
    # AIC = 100 * log(10/100) + 2*3 = 100 * (-2.3026) + 6 = -223.26
    n_obs = 100
    n_params = 3
    rss = 10.0
    aic = _compute_aic(n_obs, n_params, rss)
    expected = 100.0 * np.log(10.0 / 100.0) + 2.0 * 3.0
    assert np.isclose(aic, expected, rtol=0.01)

    # Case 2: Perfect fit (rss very small)
    aic2 = _compute_aic(50, 2, 1e-10)
    assert aic2 < 0  # Small RSS -> negative AIC


def test_build_adf_design_matrix_no_lag():
    """Test design matrix construction with lag=0."""
    from afmlkit.feature.core.structural_break.adf import _build_adf_design_matrix

    # Simple price series
    y = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype=np.float64)

    # lag=0, trend=False: X = [constant, y_{t-1}], dy = diff(y)
    X, dy, gamma_idx = _build_adf_design_matrix(y, lag=0, trend=False)

    # Expected:
    # dy = [1.0, 1.0, 1.0, 1.0, 1.0] (diff of y)
    # y_lag = [100, 101, 102, 103, 104]
    # X[:, 0] = 1 (constant)
    # X[:, 1] = y_lag
    assert X.shape[0] == 5  # n - 1 observations
    assert X.shape[1] == 2  # constant + y_lag
    assert gamma_idx == 1   # gamma is at index 1
    assert np.allclose(dy, np.diff(y))


def test_build_adf_design_matrix_with_lag():
    """Test design matrix construction with lag=2."""
    from afmlkit.feature.core.structural_break.adf import _build_adf_design_matrix

    y = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], dtype=np.float64)

    # lag=2, trend=True: X = [constant, trend, y_{t-1}, dy_{t-1}, dy_{t-2}]
    X, dy, gamma_idx = _build_adf_design_matrix(y, lag=2, trend=True)

    # n_obs = len(y) - 1 - lag = 8 - 1 - 2 = 5
    assert X.shape[0] == 5  # observations after lag adjustment
    assert X.shape[1] == 5  # constant + trend + y_lag + 2 lagged diffs
    assert gamma_idx == 2   # gamma (y_{t-1}) is at index 2


def test_adf_regression_with_lag_basic():
    """Test ADF regression with lagged terms."""
    from afmlkit.feature.core.structural_break.adf import _adf_regression_with_lag

    np.random.seed(42)
    # Generate mean-reverting series
    n = 50
    y = np.zeros(n, dtype=np.float64)
    y[0] = 100.0
    for i in range(1, n):
        y[i] = 0.8 * 100.0 + 0.2 * y[i-1] + np.random.randn() * 0.5

    # Test with lag=0 (basic DF)
    t_stat, p_value, rss, n_params, n_obs, success = _adf_regression_with_lag(y, lag=0, trend=False)
    assert success == True
    assert n_obs > 0
    assert n_params >= 2
    assert not np.isnan(t_stat)

    # Test with lag=2
    t_stat2, p_value2, rss2, n_params2, n_obs2, success2 = _adf_regression_with_lag(y, lag=2, trend=False)
    assert success2 == True
    assert n_params2 == n_params + 2  # 2 more lag terms
