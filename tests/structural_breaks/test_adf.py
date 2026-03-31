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


def test_select_lag_by_aic():
    """Test AIC-based lag selection."""
    from afmlkit.feature.core.structural_break.adf import _select_lag_by_aic

    np.random.seed(42)
    # Generate series with some serial correlation
    n = 100
    y = np.zeros(n, dtype=np.float64)
    y[0] = 100.0
    # AR(1) process with coefficient 0.5
    for i in range(1, n):
        y[i] = 0.5 * y[i-1] + 50.0 + np.random.randn() * 2.0

    best_lag, t_stat, p_value, best_aic = _select_lag_by_aic(y, max_lag=5, trend=False)

    assert 0 <= best_lag <= 5
    assert not np.isnan(t_stat)
    assert not np.isnan(best_aic)

    # The chosen lag should have the lowest AIC among all tested
    # We can verify by checking that AIC decreases or stays similar
    # as we increase lag from 0 to optimal


def test_adf_test_full_returns_tuple():
    """Test adf_test_full returns 6-element tuple."""
    from afmlkit.feature.core.structural_break.adf import adf_test_full

    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 100

    result = adf_test_full(y, max_lag=None, trend=True)

    assert len(result) == 6
    adf_stat, pvalue, used_lag, nobs, critical_values, icbest = result

    assert isinstance(adf_stat, float)
    assert isinstance(pvalue, float)
    assert isinstance(used_lag, int)
    assert isinstance(nobs, int)
    assert isinstance(critical_values, dict)
    assert isinstance(icbest, float)

    # Critical values should have 1%, 5%, 10%
    assert '1%' in critical_values
    assert '5%' in critical_values
    assert '10%' in critical_values


def test_adf_test_full_small_sample():
    """Test adf_test_full handles small sample."""
    from afmlkit.feature.core.structural_break.adf import adf_test_full

    y = np.array([100.0, 101.0, 102.0], dtype=np.float64)  # n=3 < 10

    result = adf_test_full(y)
    adf_stat, pvalue, used_lag, nobs, critical_values, icbest = result

    assert np.isnan(adf_stat)
    assert used_lag == 0


def test_adf_test_auto_lag():
    """Test adf_test with automatic lag selection."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 100

    # Auto lag (max_lag=None)
    t_stat_auto, p_value_auto, lag_auto = adf_test(y, max_lag=None, trend=True)

    # Fixed lag (max_lag=0)
    t_stat_fixed, p_value_fixed, lag_fixed = adf_test(y, max_lag=0, trend=True)

    assert isinstance(lag_auto, int)
    assert isinstance(lag_fixed, int)
    assert lag_fixed == 0
    # Auto lag may differ from 0
    assert lag_auto >= 0
    # Results should be finite
    assert np.isfinite(t_stat_auto)
    assert np.isfinite(t_stat_fixed)


def test_adf_test_small_sample_returns_nan():
    """Test adf_test returns NaN for small sample."""
    y = np.array([100.0, 101.0, 102.0], dtype=np.float64)

    t_stat, p_value, lag = adf_test(y)
    assert np.isnan(t_stat)
    assert np.isnan(p_value)
    assert lag == 0


def test_adf_vs_statsmodels_comparison():
    """Compare AFMLKit ADF results with statsmodels."""
    from statsmodels.tsa.stattools import adfuller
    from afmlkit.feature.core.structural_break.adf import adf_test_full

    np.random.seed(42)

    # Test 1: Random walk (unit root)
    y1 = np.cumsum(np.random.randn(100)) + 100

    sm_result1 = adfuller(y1, autolag='AIC', regression='ct')
    ak_result1 = adf_test_full(y1, trend=True)

    # Allow 10% relative tolerance for t-stat (different implementations)
    assert np.isclose(ak_result1[0], sm_result1[0], rtol=0.10)
    # Lag selection should be similar (may differ slightly due to implementation)
    assert abs(ak_result1[2] - sm_result1[2]) <= 2

    # Test 2: Mean-reverting series (stationary)
    y2 = np.zeros(100, dtype=np.float64)
    y2[0] = 100.0
    for i in range(1, 100):
        y2[i] = 0.3 * y2[i-1] + 70.0 + np.random.randn() * 1.0

    sm_result2 = adfuller(y2, autolag='AIC', regression='c')
    ak_result2 = adf_test_full(y2, trend=False)

    # Stationary series: t-stat should be significantly negative
    assert ak_result2[0] < sm_result2[4]['5%']  # Reject unit root at 5%
    assert np.isclose(ak_result2[0], sm_result2[0], rtol=0.10)
