"""
Serial Correlation (Autocorrelation) features for financial time series.

Implements rolling serial correlation at multiple lags in a single pass,
optimized for cache-friendly memory access patterns.

References
----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 4 — Sample Weights, Section on Serial Correlation.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Sequence, Optional
import pandas as pd

from afmlkit.feature.base import SIMOTransform, SISOTransform


# ---------------------------------------------------------------------------
# Numba Backend - Core Functions
# ---------------------------------------------------------------------------

@njit(nogil=True)
def _autocorr_single_window(
    data: NDArray[np.float64],
    lag: int
) -> float:
    """
    Compute autocorrelation at a single lag for a given window of data.

    Uses the formula:
        ρ_k = Σ(r_t - r̄)(r_{t-k} - r̄) / Σ(r_t - r̄)²

    :param data: 1D array of values (already demeaned is faster)
    :param lag: Lag order
    :returns: Autocorrelation coefficient
    """
    n = len(data)
    if n <= lag + 1:
        return np.nan

    # Compute mean
    mean_val = 0.0
    for i in range(n):
        mean_val += data[i]
    mean_val /= n

    # Compute variance and covariance
    var = 0.0
    cov = 0.0
    for i in range(n):
        d = data[i] - mean_val
        var += d * d

    if var < 1e-12:
        return 0.0  # Zero variance -> no correlation

    for i in range(lag, n):
        d_t = data[i] - mean_val
        d_tk = data[i - lag] - mean_val
        cov += d_t * d_tk

    return cov / var


@njit(nogil=True, parallel=True)
def rolling_serial_correlation(
    data: NDArray[np.float64],
    window: int,
    lags: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Rolling serial correlation at multiple lags (SIMO).

    Computes autocorrelation coefficients for all specified lags
    in a single pass through the data for each window position.

    :param data: 1D array of values (e.g., returns)
    :param window: Rolling window size for correlation calculation
    :param lags: Array of lag orders, e.g., [1, 5, 10, 20, 60]
    :returns: 2D array of shape (n, len(lags)) containing autocorrelations

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000)
    >>> lags = np.array([1, 5, 10, 20], dtype=np.int64)
    >>> result = rolling_serial_correlation(data, window=100, lags=lags)
    >>> result.shape
    (1000, 4)

    Notes
    -----
    Performance characteristics:
    - Single pass through data for each window position
    - All lags computed simultaneously in cache-friendly manner
    - Parallelized across time dimension using prange
    """
    n = len(data)
    n_lags = len(lags)

    # Output: (n_samples, n_lags)
    result = np.empty((n, n_lags), dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    # Process each time point in parallel
    for t in prange(window - 1, n):
        # Extract window of data
        window_data = data[t - window + 1: t + 1]

        # Compute mean once for all lags
        mean_val = 0.0
        for i in range(window):
            mean_val += window_data[i]
        mean_val /= window

        # Compute variance once
        var = 0.0
        for i in range(window):
            d = window_data[i] - mean_val
            var += d * d

        # Compute autocorrelation for each lag
        for lag_idx in range(n_lags):
            lag = lags[lag_idx]

            if window <= lag:
                result[t, lag_idx] = np.nan
                continue

            if var < 1e-12:
                result[t, lag_idx] = 0.0
                continue

            # Compute covariance
            cov = 0.0
            for i in range(lag, window):
                d_t = window_data[i] - mean_val
                d_tk = window_data[i - lag] - mean_val
                cov += d_t * d_tk

            result[t, lag_idx] = cov / var

    return result


@njit(nogil=True)
def serial_correlation_at_lag(
    data: NDArray[np.float64],
    lag: int
) -> float:
    """
    Compute serial correlation at a single lag for entire series.

    :param data: 1D array of values
    :param lag: Lag order
    :returns: Autocorrelation coefficient

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(1000)
    >>> rho_1 = serial_correlation_at_lag(data, lag=1)
    >>> -0.1 < rho_1 < 0.1  # Random data should have near-zero autocorr
    True
    """
    return _autocorr_single_window(data, lag)


@njit(nogil=True)
def ljung_box_statistic(
    data: NDArray[np.float64],
    lags: NDArray[np.int64]
) -> float:
    """
    Compute Ljung-Box Q statistic for testing autocorrelation.

    Q = n(n+2) * Σ(ρ_k² / (n-k)) for k in lags

    Under null hypothesis of no autocorrelation, Q follows χ²(df=len(lags)).

    :param data: 1D array of values
    :param lags: Array of lag orders to include in test
    :returns: Ljung-Box Q statistic

    References
    ----------
        Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in
        time series models. Biometrika, 65(2), 297-303.
    """
    n = len(data)
    n_lags = len(lags)

    if n <= max(lags):
        return np.nan

    q_stat = 0.0
    for i in range(n_lags):
        k = lags[i]
        rho_k = _autocorr_single_window(data, k)
        if not np.isnan(rho_k):
            q_stat += (rho_k ** 2) / (n - k)

    return n * (n + 2) * q_stat


# ---------------------------------------------------------------------------
# SIMO Transform
# ---------------------------------------------------------------------------

class SerialCorrelationTransform(SIMOTransform):
    """
    SIMO Transform for serial correlation at multiple lags.

    Computes rolling autocorrelation coefficients for multiple lag orders
    in a single cache-friendly pass through the data.

    Parameters
    ----------
    input_col : str
        Name of the input column (typically returns or fracdiff prices)
    window : int
        Rolling window size for correlation calculation
    lags : Sequence[int]
        Lag orders to compute, e.g., [1, 5, 10, 20, 60]

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.serial_corr import SerialCorrelationTransform
    >>>
    >>> # Create sample returns data
    >>> dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    >>> returns = pd.Series(np.random.randn(1000) * 0.02, index=dates, name='returns')
    >>> df = returns.to_frame()
    >>>
    >>> # Create transform
    >>> transform = SerialCorrelationTransform(
    ...     input_col='returns',
    ...     window=100,
    ...     lags=[1, 5, 10, 20]
    ... )
    >>>
    >>> # Apply transform
    >>> result = transform(df, backend='nb')  # Returns tuple of 4 Series
    >>> len(result)
    4

    Integration with FeatureKit:

    >>> from afmlkit.feature.kit import Feature, FeatureKit
    >>>
    >>> # Create feature from transform
    >>> autocorr_feature = Feature(transform)
    >>>
    >>> # Use in pipeline
    >>> kit = FeatureKit([autocorr_feature], retain=['returns'])
    >>> result_df = kit.build(df, backend='nb')

    Notes
    -----
    The transform outputs multiple columns, one per lag:
    - For input 'returns' and lags=[1, 5, 10, 20], outputs:
      ['returns_autocorr_1', 'returns_autocorr_5', 'returns_autocorr_10', 'returns_autocorr_20']

    Performance: Uses Numba parallel execution for cache-friendly
    computation of all lags simultaneously.
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        lags: Sequence[int]
    ):
        self.window = window
        self.lags = np.array(sorted(lags), dtype=np.int64)

        # Generate output column names
        output_cols = [f'autocorr_{lag}' for lag in self.lags]

        super().__init__(input_col, output_cols)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        if len(x) < self.window:
            raise ValueError(f"Data length ({len(x)}) must be >= window ({self.window})")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> NDArray:
        return x[self.requires[0]].values.astype(np.float64)

    def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """
        Pandas implementation for development/debugging.
        """
        data = x[self.requires[0]].values
        result_2d = rolling_serial_correlation(data, self.window, self.lags)

        output_series = []
        for i, lag in enumerate(self.lags):
            s = pd.Series(result_2d[:, i], index=x.index, name=self.output_name[i])
            output_series.append(s)

        return tuple(output_series)

    def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """
        Numba implementation for production performance.
        """
        data = self._prepare_input_nb(x)
        result_2d = rolling_serial_correlation(data, self.window, self.lags)
        return self._prepare_output_nb(x.index, tuple(result_2d[:, i] for i in range(len(self.lags))))


class LjungBoxTransform(SISOTransform):
    """
    SISO Transform for Ljung-Box Q statistic.

    Computes the Ljung-Box test statistic for testing whether
    any of a group of autocorrelations are non-zero.

    Parameters
    ----------
    input_col : str
        Name of the input column
    window : int
        Rolling window size
    lags : Sequence[int]
        Lag orders to include in test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.serial_corr import LjungBoxTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> returns = pd.Series(np.random.randn(500) * 0.02, index=dates, name='returns')
    >>> df = returns.to_frame()
    >>>
    >>> transform = LjungBoxTransform('returns', window=100, lags=[1, 5, 10])
    >>> q_stat = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        lags: Sequence[int]
    ):
        self.window = window
        self.lags = np.array(sorted(lags), dtype=np.int64)
        output_col = f'ljung_box_{len(lags)}lags'
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
        data = x[self.requires[0]].values
        n = len(data)
        result = np.empty(n, dtype=np.float64)
        result[:] = np.nan

        for t in range(self.window - 1, n):
            window_data = data[t - self.window + 1: t + 1]
            result[t] = ljung_box_statistic(window_data, self.lags)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        # For now, use same implementation as pandas
        # Ljung-Box is not computationally heavy enough to warrant separate Numba kernel
        return self._pd(x)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def autocorr(
    data: pd.Series,
    window: int,
    lags: Sequence[int]
) -> pd.DataFrame:
    """
    Convenience function to compute rolling autocorrelation.

    :param data: Input Series (e.g., returns)
    :param window: Rolling window size
    :param lags: Lag orders to compute
    :returns: DataFrame with autocorrelation columns

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.serial_corr import autocorr
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> returns = pd.Series(np.random.randn(500) * 0.02, index=dates)
    >>>
    >>> result = autocorr(returns, window=100, lags=[1, 5, 10, 20])
    >>> result.columns.tolist()
    ['autocorr_1', 'autocorr_5', 'autocorr_10', 'autocorr_20']
    """
    df = data.to_frame(name=data.name or 'value')
    col_name = data.name or 'value'

    transform = SerialCorrelationTransform(
        input_col=col_name,
        window=window,
        lags=lags
    )

    results = transform(df, backend='nb')

    return pd.DataFrame({s.name: s for s in results})