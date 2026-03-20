"""
ADF (Augmented Dickey-Fuller) test base implementation.

Provides foundation for SADF, QADF, CADF tests.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple, Union
import pandas as pd


# MacKinnon approximate critical values for p-value interpolation
# Format: sample_size -> (1%, 5%, 10%)
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
    """
    # Select appropriate table
    if trend:
        table = _MACKINNON_WITH_TREND
    else:
        table = _MACKINNON_NO_TREND

    # Find appropriate sample size bracket
    sizes = np.array([25, 50, 100, 250, 500, 1000], dtype=np.int64)

    # Determine which bracket n falls into
    if n < 25:
        crit_vals = table[25]
    elif n >= 1000:
        crit_vals = table[1000]
    else:
        # Find bracket
        for i in range(len(sizes) - 1):
            if sizes[i] <= n < sizes[i + 1]:
                # Linear interpolation
                n1, n2 = sizes[i], sizes[i + 1]
                w = float(n - n1) / float(n2 - n1)
                cv1 = np.array(table[int(n1)], dtype=np.float64)
                cv2 = np.array(table[int(n2)], dtype=np.float64)
                crit_vals_arr = cv1 * (1 - w) + cv2 * w
                # Convert back to tuple
                crit_vals = (float(crit_vals_arr[0]), float(crit_vals_arr[1]), float(crit_vals_arr[2]))
                break
        else:
            crit_vals = table[1000]

    # Map t-statistic to approximate p-value
    c1, c5, c10 = crit_vals

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
    max_lag: int = 12,
    trend: bool = True
) -> Tuple[float, float, int]:
    """
    Single ADF test on a price series.

    **Note**: Currently only implements basic DF test (lag=0). Full ADF with
    automatic lag selection is a future enhancement. The max_lag parameter
    is accepted for API consistency but is not used.

    :param y: Price series (raw prices, not returns)
    :param max_lag: Maximum lag for augmented DF (default 12, not currently used)
    :param trend: Include time trend in regression
    :returns: (t_statistic, p_value, selected_lag)
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)

    selected_lag = 0

    t_stat, p_value = _adf_regression_core(y, max_lag=selected_lag, trend=trend)

    return float(t_stat), float(p_value), int(selected_lag)


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
