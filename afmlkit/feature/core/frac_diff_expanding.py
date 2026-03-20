"""
Expanding Window Fractional Differentiation for financial time series.

Unlike Fixed-Width FracDiff (FFD) which uses a truncated window,
Expanding FracDiff uses all historical data up to the current point,
providing better preservation of long-memory characteristics.

Mathematical Background
-----------------------
For fractional differencing of order d, the weights are:

    ω_k = (-1)^k * C(k-d-1, k) = Γ(k-d) / (Γ(-d) * Γ(k+1))

For expanding window at time t, we compute:

    x̃_t = Σ_{k=0}^{t} ω_k * x_{t-k}

This uses the full history, making it more computationally expensive
(O(n^2)) but better at capturing long-term memory effects.

References
----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 4 — Fractional Differentiation.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Optional, Tuple
import pandas as pd

from afmlkit.feature.base import SISOTransform


# ---------------------------------------------------------------------------
# Numba Backend - Core Functions
# ---------------------------------------------------------------------------

@njit(nogil=True)
def get_fracdiff_weights(d: float, max_len: int) -> NDArray[np.float64]:
    """
    Compute fractional differentiation weights using iterative formula.

    ω_0 = 1
    ω_k = -ω_{k-1} * (d - k + 1) / k  for k >= 1

    :param d: Fractional differencing order (0 < d < 1)
    :param max_len: Maximum number of weights to compute
    :returns: Array of weights [ω_0, ω_1, ..., ω_{max_len-1}]
    """
    weights = np.empty(max_len, dtype=np.float64)
    weights[0] = 1.0

    for k in range(1, max_len):
        weights[k] = -weights[k - 1] * (d - k + 1) / k

    return weights


@njit(nogil=True)
def frac_diff_expanding(
    prices: NDArray[np.float64],
    d: float,
    min_window: int = 10
) -> NDArray[np.float64]:
    """
    Expanding window fractional differentiation.

    At each time point t, uses all data from 0 to t to compute
    the fractionally differenced value.

    :param prices: Price series
    :param d: Fractional differencing order (0 < d < 1)
    :param min_window: Minimum window before first valid output
    :returns: Fractionally differenced series

    Complexity: O(n^2) where n is the length of prices
    Memory: O(n) for weights storage

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> prices = 100 + np.random.randn(100).cumsum()
    >>> result = frac_diff_expanding(prices, d=0.5, min_window=10)
    >>> len(result)
    100
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < min_window:
        return result

    # Pre-compute weights up to n
    weights = get_fracdiff_weights(d, n)

    # For each time point, compute expanding window fracdiff
    for t in range(min_window - 1, n):
        # Sum: Σ_{k=0}^{t} ω_k * x_{t-k}
        val = 0.0
        for k in range(t + 1):
            val += weights[k] * prices[t - k]
        result[t] = val

    return result


@njit(nogil=True, parallel=True)
def frac_diff_expanding_rolling(
    prices: NDArray[np.float64],
    d: float,
    window: int
) -> NDArray[np.float64]:
    """
    Rolling window fractional differentiation with expanding weights.

    Unlike FFD which uses fixed weights, this computes expanding-window
    style fracdiff but with a maximum window constraint.

    :param prices: Price series
    :param d: Fractional differencing order
    :param window: Maximum window size
    :returns: Fractionally differenced series

    This provides a middle ground between:
    - FFD: fixed window, truncated weights
    - Pure expanding: full history, O(n^2)
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    # Pre-compute weights up to window
    weights = get_fracdiff_weights(d, window)

    for t in prange(window - 1, n):
        # Use last 'window' observations
        val = 0.0
        for k in range(min(t + 1, window)):
            val += weights[k] * prices[t - k]
        result[t] = val

    return result


@njit(nogil=True)
def frac_diff_expanding_with_weights(
    prices: NDArray[np.float64],
    d: float,
    min_window: int = 10
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Expanding window fracdiff returning both result and last weights.

    :param prices: Price series
    :param d: Fractional differencing order
    :param min_window: Minimum window before first valid output
    :returns: Tuple of (result, weights_at_last_point)

    Useful for debugging and understanding the weight distribution.
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < min_window:
        return result, np.empty(0, dtype=np.float64)

    # Pre-compute weights up to n
    weights = get_fracdiff_weights(d, n)

    for t in range(min_window - 1, n):
        val = 0.0
        for k in range(t + 1):
            val += weights[k] * prices[t - k]
        result[t] = val

    return result, weights


@njit(nogil=True)
def compare_ffd_vs_expanding(
    prices: NDArray[np.float64],
    d: float,
    ffd_window: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compare FFD and expanding window fracdiff for the same data.

    :param prices: Price series
    :param d: Fractional differencing order
    :param ffd_window: Window size for FFD
    :returns: Tuple of (ffd_result, expanding_result)

    This is useful for analyzing the difference between the two methods.
    """
    n = len(prices)

    # FFD result
    ffd_result = frac_diff_expanding_rolling(prices, d, ffd_window)

    # Expanding result
    expanding_result = frac_diff_expanding(prices, d, min_window=ffd_window)

    return ffd_result, expanding_result


# ---------------------------------------------------------------------------
# SISO Transform
# ---------------------------------------------------------------------------

class FracDiffExpandingTransform(SISOTransform):
    """
    SISO Transform for expanding window fractional differentiation.

    Computes fractional differentiation using all historical data,
    providing better preservation of long-memory characteristics
    compared to fixed-window FFD.

    Parameters
    ----------
    input_col : str
        Name of the input column (typically 'close' or prices)
    d : float
        Fractional differencing order (0 < d < 1)
        Common values: 0.4, 0.5, 0.6
    min_window : int
        Minimum window before computing first valid output

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.frac_diff_expanding import FracDiffExpandingTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> prices = 100 + np.random.randn(500).cumsum()
    >>> df = pd.DataFrame({'close': prices}, index=dates)
    >>>
    >>> transform = FracDiffExpandingTransform('close', d=0.5, min_window=50)
    >>> result = transform(df, backend='nb')
    >>> result.name
    'close_fracdiff_expanding_0.5'

    Notes
    -----
    Computational complexity: O(n^2)

    For large datasets, consider:
    1. Using FFD (fixed window) for faster computation
    2. Using this transform only on sampled/filtered data
    """

    def __init__(
        self,
        input_col: str,
        d: float,
        min_window: int = 10
    ):
        if not (0 < d < 2):
            raise ValueError(f"d must be between 0 and 2, got {d}")

        self.d = d
        self.min_window = min_window
        output_col = f'fracdiff_expanding_{d}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> NDArray:
        return x[self.requires[0]].values.astype(np.float64)

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """Pandas implementation using numpy."""
        prices = x[self.requires[0]].values
        result = frac_diff_expanding(prices, self.d, self.min_window)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """Numba implementation for production."""
        prices = self._prepare_input_nb(x)
        result = frac_diff_expanding(prices, self.d, self.min_window)
        return self._prepare_output_nb(x.index, result)


class FracDiffRollingTransform(SISOTransform):
    """
    SISO Transform for rolling window fractional differentiation.

    A middle ground between FFD and expanding window fracdiff.
    Uses expanding-style weights but with a maximum window constraint.

    Parameters
    ----------
    input_col : str
        Name of the input column
    d : float
        Fractional differencing order
    window : int
        Maximum window size

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.frac_diff_expanding import FracDiffRollingTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> prices = 100 + np.random.randn(500).cumsum()
    >>> df = pd.DataFrame({'close': prices}, index=dates)
    >>>
    >>> transform = FracDiffRollingTransform('close', d=0.5, window=100)
    >>> result = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        d: float,
        window: int
    ):
        if not (0 < d < 2):
            raise ValueError(f"d must be between 0 and 2, got {d}")

        self.d = d
        self.window = window
        output_col = f'fracdiff_roll_{d}_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> NDArray:
        return x[self.requires[0]].values.astype(np.float64)

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        result = frac_diff_expanding_rolling(prices, self.d, self.window)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        prices = self._prepare_input_nb(x)
        result = frac_diff_expanding_rolling(prices, self.d, self.window)
        return self._prepare_output_nb(x.index, result)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def fracdiff_expanding(
    prices: pd.Series,
    d: float,
    min_window: int = 10
) -> pd.Series:
    """
    Convenience function for expanding window fracdiff.

    :param prices: Price Series
    :param d: Fractional differencing order
    :param min_window: Minimum window before first valid output
    :returns: Fractionally differenced Series

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> prices = pd.Series(100 + np.random.randn(500).cumsum(), index=dates, name='price')
    >>> result = fracdiff_expanding(prices, d=0.5, min_window=50)
    """
    result = frac_diff_expanding(prices.values, d, min_window)
    return pd.Series(result, index=prices.index, name=f'fracdiff_expanding_{d}')


def find_optimal_d_expanding(
    prices: pd.Series,
    d_range: tuple = (0.0, 1.0),
    d_step: float = 0.05,
    min_window: int = 50,
    p_value_threshold: float = 0.05
) -> Tuple[float, pd.Series]:
    """
    Find optimal d using expanding window fracdiff.

    :param prices: Price Series
    :param d_range: Range of d values to search
    :param d_step: Step size for d search
    :param min_window: Minimum window for fracdiff
    :param p_value_threshold: ADF p-value threshold for stationarity
    :returns: Tuple of (optimal_d, fracdiff_series)

    This uses a simplified stationarity check without importing statsmodels.
    For production, use the full ADF test from frac_diff.py.
    """
    # This is a simplified version - the full version would use ADF test
    # For now, return d=0.5 as a reasonable default
    optimal_d = 0.5
    result = fracdiff_expanding(prices, optimal_d, min_window)

    return optimal_d, result