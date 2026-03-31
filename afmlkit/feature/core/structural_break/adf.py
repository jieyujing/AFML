"""
ADF (Augmented Dickey-Fuller) test base implementation.

Provides foundation for SADF, QADF, CADF tests.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple, Union, Optional
import pandas as pd

from afmlkit.utils.log import get_logger

logger = get_logger(__name__)


# MacKinnon approximate critical values for p-value interpolation
# Format: sample_size -> (1%, 5%, 10%)


def schwert_maxlag(n: int) -> int:
    """
    Calculate optimal maxlag using Schwert formula.

    Formula: int(12 * (n/100)**(1/4))

    Reference: Schwert, G.W. (1989). "Tests for Unit Roots: A Monte Carlo Investigation"

    :param n: Sample size
    :returns: Maximum lag for ADF test
    """
    if n < 10:
        return 1
    return int(12.0 * (n / 100.0) ** 0.25)


@njit(nogil=True)
def _compute_aic(n_obs: int, n_params: int, rss: float) -> float:
    """
    Calculate Akaike Information Criterion (AIC).

    AIC = n * log(rss/n) + 2 * k

    :param n_obs: Number of observations
    :param n_params: Number of parameters (including constant)
    :param rss: Residual sum of squares
    :returns: AIC value (lower is better)
    """
    if n_obs <= 0 or rss <= 0:
        return np.inf
    return float(n_obs) * np.log(rss / float(n_obs)) + 2.0 * float(n_params)


@njit(nogil=True)
def _build_adf_design_matrix(
    y: NDArray[np.float64],
    lag: int,
    trend: bool
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """
    Build design matrix for ADF regression with lagged differences.

    Model: dy_t = alpha + beta*t + gamma*y_{t-1} + sum(delta_i * dy_{t-i}) + epsilon

    :param y: Price series
    :param lag: Number of lagged difference terms (0 for basic DF)
    :param trend: Include time trend term
    :returns: (X, dy, gamma_idx)
              - X: Design matrix (n_obs, n_features)
              - dy: Target vector (first difference)
              - gamma_idx: Index of gamma coefficient (y_{t-1}) in beta
    """
    n = len(y)

    # Compute first difference
    dy = np.diff(y)  # length n-1

    # Number of usable observations after accounting for lag
    n_obs = n - 1 - lag
    if n_obs <= 0:
        # Return empty arrays if insufficient data
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), 0

    # Number of features: constant + (trend) + y_lag + lag terms
    n_features = 1 + (1 if trend else 0) + 1 + lag

    # Build design matrix
    X = np.ones((n_obs, n_features), dtype=np.float64)

    col_idx = 0

    # Column 0: constant (already set to 1)
    col_idx = 1

    # Column 1 (if trend): time trend
    if trend:
        for t in range(n_obs):
            X[t, col_idx] = float(t)
        col_idx += 1

    # Column: y_{t-1} (lagged level)
    gamma_idx = col_idx
    # y_lag starts at index lag+1 (need y[lag] to be the first y_{t-1})
    # For observation t = lag+1 (first usable), y_{t-1} = y[lag]
    y_lag_start = lag  # First index of y_lag for first observation
    for i in range(n_obs):
        X[i, col_idx] = y[y_lag_start + i]
    col_idx += 1

    # Columns: lagged differences dy_{t-1}, dy_{t-2}, ..., dy_{t-lag}
    # dy array has indices 0 to n-2 (dy[0] = y[1] - y[0])
    # For observation at time t = lag+1+obs_idx:
    #   dy_{t-1} = dy[t-1-1] = dy[t-2] = dy[lag-1+obs_idx]
    #   dy_{t-2} = dy[t-2-1] = dy[t-3] = dy[lag-2+obs_idx]
    for lag_i in range(lag):
        for i in range(n_obs):
            # dy_{t-lag_i-1} for observation i
            # t = lag + 1 + i, so dy_{t-lag_i-1} = dy[lag - lag_i + i]
            X[i, col_idx + lag_i] = dy[lag - lag_i - 1 + i]

    # Target vector: dy for usable observations
    # First usable dy is at index lag (dy[lag] = y[lag+1] - y[lag])
    dy_target = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        dy_target[i] = dy[lag + i]

    return X, dy_target, gamma_idx


@njit(nogil=True)
def _adf_regression_with_lag(
    y: NDArray[np.float64],
    lag: int,
    trend: bool
) -> Tuple[float, float, float, int, int, bool]:
    """
    ADF regression with lagged difference terms.

    :param y: Price series
    :param lag: Number of lagged differences
    :param trend: Include time trend
    :returns: (t_statistic, p_value, rss, n_params, n_obs, success)
              - success: True if calculation succeeded (non-singular matrix)
    """
    # Build design matrix
    X, dy, gamma_idx = _build_adf_design_matrix(y, lag, trend)

    n_obs = X.shape[0]
    if n_obs < 5:
        return np.nan, np.nan, np.nan, 0, 0, False

    n_params = X.shape[1]

    # Check for collinearity
    XtX = X.T @ X
    det_XtX = np.linalg.det(XtX)
    if np.abs(det_XtX) < 1e-15:
        return np.nan, np.nan, np.nan, n_params, n_obs, False

    # OLS estimation
    beta, residuals = _ols_coefficients(X, dy)

    # Get gamma coefficient (unit root test)
    gamma_hat = beta[gamma_idx]

    # Compute t-statistic
    rss = np.sum(residuals ** 2)
    df = n_obs - n_params
    sigma_sq = rss / df if df > 0 else rss

    try:
        XtX_inv = np.linalg.inv(XtX)
    except:
        return np.nan, np.nan, np.nan, n_params, n_obs, False

    var_gamma = sigma_sq * XtX_inv[gamma_idx, gamma_idx]

    if var_gamma <= 0 or np.isnan(var_gamma) or np.isinf(var_gamma):
        return np.nan, np.nan, np.nan, n_params, n_obs, False

    se_gamma = np.sqrt(var_gamma)
    t_stat = gamma_hat / se_gamma

    if np.isnan(t_stat) or np.isinf(t_stat):
        return np.nan, np.nan, np.nan, n_params, n_obs, False

    # Approximate p-value
    p_value = _approx_p_value(t_stat, n_obs, trend)

    return t_stat, p_value, rss, n_params, n_obs, True


def _select_lag_by_aic(
    y: NDArray[np.float64],
    max_lag: int,
    trend: bool
) -> Tuple[int, float, float, float]:
    """
    Select optimal lag using AIC criterion.

    Python layer iterates through lags, calls Numba core,
    and returns the lag with minimum AIC.

    :param y: Price series
    :param max_lag: Maximum lag to test
    :param trend: Include time trend
    :returns: (best_lag, t_stat, p_value, best_aic)
    """
    best_lag = 0
    best_aic = np.inf
    best_t_stat = np.nan
    best_p_value = np.nan

    for lag in range(max_lag + 1):
        t_stat, p_value, rss, n_params, n_obs, success = _adf_regression_with_lag(y, lag, trend)

        if not success or np.isnan(t_stat):
            continue

        aic = _compute_aic(n_obs, n_params, rss)

        if aic < best_aic:
            best_aic = aic
            best_lag = lag
            best_t_stat = t_stat
            best_p_value = p_value

    return best_lag, best_t_stat, best_p_value, best_aic


def _get_critical_values(n_obs: int, trend: bool) -> dict:
    """
    Get critical values from MacKinnon table with interpolation.

    :param n_obs: Number of observations
    :param trend: Whether trend was included
    :returns: dict with '1%', '5%', '10%' critical values
    """
    if trend:
        table = _MACKINNON_WITH_TREND
    else:
        table = _MACKINNON_NO_TREND

    # Interpolate based on sample size
    sizes = [25, 50, 100, 250, 500, 1000]

    if n_obs < 25:
        return {'1%': table[25][0], '5%': table[25][1], '10%': table[25][2]}
    elif n_obs >= 1000:
        return {'1%': table[1000][0], '5%': table[1000][1], '10%': table[1000][2]}
    else:
        # Linear interpolation
        for i in range(len(sizes) - 1):
            if sizes[i] <= n_obs < sizes[i + 1]:
                n1, n2 = sizes[i], sizes[i + 1]
                w = (n_obs - n1) / (n2 - n1)
                cv1 = table[n1]
                cv2 = table[n2]
                return {
                    '1%': cv1[0] * (1 - w) + cv2[0] * w,
                    '5%': cv1[1] * (1 - w) + cv2[1] * w,
                    '10%': cv1[2] * (1 - w) + cv2[2] * w,
                }
        return {'1%': table[1000][0], '5%': table[1000][1], '10%': table[1000][2]}


# MacKinnon approximate critical values for p-value interpolation
# Format: sample_size -> (1%, 5%, 10%)
# Python dict versions for public API
_MACKINNON_WITH_TREND = {
    25: (-4.38, -3.60, -3.24),
    50: (-4.15, -3.50, -3.18),
    100: (-4.04, -3.45, -3.15),
    250: (-3.99, -3.43, -3.13),
    500: (-3.98, -3.42, -3.13),
    1000: (-3.96, -3.41, -3.12),
}

_MACKINNON_NO_TREND = {
    25: (-3.75, -3.00, -2.63),
    50: (-3.58, -2.93, -2.60),
    100: (-3.51, -2.89, -2.58),
    250: (-3.46, -2.88, -2.57),
    500: (-3.44, -2.87, -2.57),
    1000: (-3.43, -2.86, -2.57),
}

# Numba-compatible numpy array versions
# Rows: [25, 50, 100, 250, 500, 1000] sample sizes
# Columns: [1%, 5%, 10%] critical values
_MACKINNON_WITH_TREND_ARR = np.array([
    [-4.38, -3.60, -3.24],  # n=25
    [-4.15, -3.50, -3.18],  # n=50
    [-4.04, -3.45, -3.15],  # n=100
    [-3.99, -3.43, -3.13],  # n=250
    [-3.98, -3.42, -3.13],  # n=500
    [-3.96, -3.41, -3.12],  # n=1000
], dtype=np.float64)

_MACKINNON_NO_TREND_ARR = np.array([
    [-3.75, -3.00, -2.63],  # n=25
    [-3.58, -2.93, -2.60],  # n=50
    [-3.51, -2.89, -2.58],  # n=100
    [-3.46, -2.88, -2.57],  # n=250
    [-3.44, -2.87, -2.57],  # n=500
    [-3.43, -2.86, -2.57],  # n=1000
], dtype=np.float64)

# Sample sizes for index lookup
_MACKINNON_SIZES = np.array([25, 50, 100, 250, 500, 1000], dtype=np.int64)


@njit(nogil=True)
def _ols_coefficients(
    X: NDArray[np.float64],
    y: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    OLS regression: y = X @ beta + epsilon.

    Uses normal equations: beta = (X'X)^-1 X'y

    :param X: Design matrix (n_obs, n_features)
    :param y: Target vector (n_obs,)
    :returns: Tuple of (coefficients, residuals)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    # Solve using simple linear algebra (Numba-compatible)
    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    residuals = y - y_hat

    return beta, residuals


@njit(nogil=True)
def _approx_p_value(t_stat: float, n: int, trend: bool) -> float:
    """
    Approximate p-value using MacKinnon critical values.

    Linear interpolation between critical value tables.
    Uses numpy arrays for Numba compatibility.
    """
    # Select appropriate table (numpy array version)
    if trend:
        table = _MACKINNON_WITH_TREND_ARR
    else:
        table = _MACKINNON_NO_TREND_ARR

    # Sample sizes for index lookup
    sizes = _MACKINNON_SIZES

    # Determine which bracket n falls into and get critical values
    if n < 25:
        # Use first row (n=25)
        c1, c5, c10 = table[0, 0], table[0, 1], table[0, 2]
    elif n >= 1000:
        # Use last row (n=1000)
        c1, c5, c10 = table[5, 0], table[5, 1], table[5, 2]
    else:
        # Find bracket and interpolate
        for i in range(len(sizes) - 1):
            if sizes[i] <= n < sizes[i + 1]:
                # Linear interpolation
                n1, n2 = sizes[i], sizes[i + 1]
                w = float(n - n1) / float(n2 - n1)
                # Get critical values from rows i and i+1
                c1 = table[i, 0] * (1 - w) + table[i + 1, 0] * w
                c5 = table[i, 1] * (1 - w) + table[i + 1, 1] * w
                c10 = table[i, 2] * (1 - w) + table[i + 1, 2] * w
                break
        else:
            # Fallback to last row
            c1, c5, c10 = table[5, 0], table[5, 1], table[5, 2]

    # Map t-statistic to approximate p-value
    if t_stat < c1:
        return 0.01
    elif t_stat < c5:
        return 0.05
    elif t_stat < c10:
        return 0.10
    else:
        return 0.20


@njit(nogil=True)
def _adf_regression_core(
    y: NDArray[np.float64],
    max_lag: int,
    trend: bool = True
) -> Tuple[float, float]:
    """
    Core ADF regression without lagged differences (simplest case).

    Model: dy_t = alpha + beta*t + gamma*y_{t-1} + epsilon_t

    Returns t-statistic for gamma (test of unit root).

    :param y: Price series (will be differenced internally)
    :param max_lag: Number of lagged differences (0 for basic DF)
    :param trend: Include time trend term
    :returns: (t_statistic, p_value_approx)
    """
    n = len(y)
    if n < 10:
        return np.nan, np.nan

    # Compute log differences
    dy = np.diff(y)
    y_lag = y[:-1]

    # Build design matrix
    n_obs = len(dy)
    if trend:
        # [constant, trend, y_lag]
        X = np.ones((n_obs, 3), dtype=np.float64)
        X[:, 1] = np.arange(n_obs, dtype=np.float64)
        X[:, 2] = y_lag
        gamma_idx = 2
    else:
        # [constant, y_lag]
        X = np.ones((n_obs, 2), dtype=np.float64)
        X[:, 1] = y_lag
        gamma_idx = 1

    # Check for collinearity: if X'X is singular, matrix inversion fails
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except:
        return np.nan, np.nan

    # Check if matrix is singular (determinant near zero)
    det_XtX = np.linalg.det(XtX)
    if np.abs(det_XtX) < 1e-15:
        return np.nan, np.nan

    # OLS estimation
    beta, residuals = _ols_coefficients(X, dy)

    # Compute t-statistic for gamma (unit root coefficient)
    gamma_hat = beta[gamma_idx]

    # Variance of residuals
    rss = np.sum(residuals ** 2)
    df = n_obs - X.shape[1]
    sigma_sq = rss / df if df > 0 else rss

    var_gamma = sigma_sq * XtX_inv[gamma_idx, gamma_idx]

    if var_gamma <= 0 or np.isnan(var_gamma) or np.isinf(var_gamma):
        return np.nan, np.nan

    se_gamma = np.sqrt(var_gamma)
    t_stat = gamma_hat / se_gamma

    if np.isnan(t_stat) or np.isinf(t_stat):
        return np.nan, np.nan

    # Approximate p-value using MacKinnon interpolation
    p_value = _approx_p_value(t_stat, n_obs, trend)

    return t_stat, p_value


def adf_test(
    y: Union[pd.Series, NDArray[np.float64]],
    max_lag: Optional[int] = None,
    trend: bool = True
) -> Tuple[float, float, int]:
    """
    ADF test with automatic lag selection.

    :param y: Price series (raw prices, not returns)
    :param max_lag: None for automatic selection (Schwert + AIC), int for fixed lag
    :param trend: Include time trend in regression
    :returns: (t_statistic, p_value, selected_lag)
    """
    try:
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64)

        n = len(y)

        # Check sample size
        if n < 10:
            logger.warning(f"ADF: 样本量过小 ({n})，返回 NaN")
            return (np.nan, np.nan, 0)

        # Calculate Schwert maxlag if not provided
        if max_lag is None:
            max_lag = schwert_maxlag(n)

        # Truncate maxlag if too large
        if max_lag > n // 2:
            logger.warning(f"ADF: 滞后截断 ({max_lag} -> {n//2})")
            max_lag = n // 2

        # If max_lag is specified as int > 0, still use AIC within that range
        # This matches statsmodels behavior: autolag with maxlag constraint
        best_lag, t_stat, p_value, best_aic = _select_lag_by_aic(y, max_lag, trend)

        if np.isnan(t_stat):
            logger.warning("ADF: 设计矩阵奇异或计算失败")
            return (np.nan, np.nan, best_lag)

        logger.debug(f"ADF: 选择滞后={best_lag}, AIC={best_aic:.2f}")
        return (t_stat, p_value, best_lag)

    except Exception as e:
        logger.error(f"ADF: 未捕获错误 - {e}")
        return (np.nan, np.nan, 0)


def adf_test_rolling(
    y: Union[pd.Series, NDArray[np.float64]],
    window: int,
    max_lag: int = 12,
    trend: bool = True
) -> NDArray[np.float64]:
    """
    Rolling ADF test statistic calculation.

    **Note**: Currently only implements basic DF test (lag=0). The max_lag
    parameter is accepted for API consistency but is not used.

    :param y: Price series
    :param window: Rolling window size
    :param max_lag: Maximum lag for ADF (default 12, not currently used)
    :param trend: Include time trend
    :returns: Array of t-statistics (NaN for first window-1 observations)
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)

    n = len(y)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n + 1):
        window_data = y[i - window:i]
        t_stat, _ = _adf_regression_core(window_data, max_lag=0, trend=trend)
        result[i - 1] = t_stat

    return result


def adf_test_full(
    y: Union[pd.Series, NDArray[np.float64]],
    max_lag: Optional[int] = None,
    trend: bool = True
) -> Tuple[float, float, int, int, dict, float]:
    """
    ADF test with full results (statsmodels-compatible format).

    :param y: Price series (raw prices, not returns)
    :param max_lag: None for automatic selection (Schwert + AIC), int for fixed lag
    :param trend: Include time trend in regression
    :returns: (adf_stat, pvalue, used_lag, nobs, critical_values, icbest)
              - adf_stat: t-statistic for gamma coefficient
              - pvalue: approximate p-value
              - used_lag: lag used in test
              - nobs: number of observations used
              - critical_values: dict {'1%': ..., '5%': ..., '10%': ...}
              - icbest: best AIC value
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)

    n = len(y)

    # Check sample size
    if n < 10:
        logger.warning(f"ADF: 样本量过小 ({n}), 返回 NaN")
        return (np.nan, np.nan, 0, n, _get_critical_values(n, trend), np.nan)

    # Calculate Schwert maxlag if not provided
    if max_lag is None:
        max_lag = schwert_maxlag(n)

    # Truncate maxlag if too large
    if max_lag > n // 2:
        logger.warning(f"ADF: 滞后截断 ({max_lag} -> {n//2})")
        max_lag = n // 2

    # AIC lag selection
    best_lag, t_stat, p_value, best_aic = _select_lag_by_aic(y, max_lag, trend)

    if np.isnan(t_stat):
        logger.warning("ADF: 设计矩阵奇异或计算失败")
        return (np.nan, np.nan, best_lag, n, _get_critical_values(n, trend), np.nan)

    # Get actual n_obs from the regression
    _, _, _, _, n_obs, _ = _adf_regression_with_lag(y, best_lag, trend)

    logger.debug(f"ADF: 选择滞后={best_lag}, AIC={best_aic:.2f}")

    critical_values = _get_critical_values(n_obs, trend)

    return (t_stat, p_value, best_lag, n_obs, critical_values, best_aic)
