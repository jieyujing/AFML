"""
SADF (Supremum Augmented Dickey-Fuller) test for bubble detection.

Detects explosive behavior by taking supremum of ADF statistics across
expanding windows.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import SISOTransform
from afmlkit.feature.core.structural_break.adf import _adf_regression_core


@njit(nogil=True)
def _sadf_core(
    log_prices: NDArray[np.float64],
    min_window: int,
    max_window: int,
    max_lag: int,
    trend: bool
) -> NDArray[np.float64]:
    """
    SADF core computation.

    For each time t, computes ADF statistics for all windows from min_window
to current window (or max_window), and returns the supremum.

    :param log_prices: Log price series
    :param min_window: Minimum window size for ADF
    :param max_window: Maximum window size (use -1 for expanding to t)
    :param max_lag: Maximum lag for ADF regression
    :param trend: Include time trend in ADF
    :returns: SADF statistics (NaN for first min_window observations)
    """
    n = len(log_prices)
    result = np.full(n, np.nan, dtype=np.float64)

    for t in range(min_window, n):
        current_max = min(t, max_window) if max_window > 0 else t
        max_adf = -np.inf

        for window in range(min_window, current_max + 1):
            start = t - window + 1
            if start < 0:
                continue

            window_data = log_prices[start:t + 1]
            adf_t, _ = _adf_regression_core(window_data, max_lag, trend)

            if not np.isnan(adf_t) and adf_t > max_adf:
                max_adf = adf_t

        if max_adf > -np.inf:
            result[t] = max_adf

    return result


def sadf_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    min_window: int = 20,
    max_window: int = 100,
    max_lag: int = 0,
    trend: bool = True,
    use_log: bool = True,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    SADF test for bubble detection.

    :param prices: Price series
    :param min_window: Minimum window size (default 20)
    :param max_window: Maximum window size (default 100, None for expanding)
    :param max_lag: ADF lag order (default 0)
    :param trend: Include time trend (default True)
    :param use_log: Apply log transform to prices (default True, recommended)
    :param use_numba: Use Numba backend (default True)
    :returns: SADF statistic series
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    prices_arr = np.asarray(prices, dtype=np.float64)

    if use_log:
        prices_arr = np.log(prices_arr)

    max_w = max_window if max_window is not None else -1

    result = _sadf_core(prices_arr, min_window, max_w, max_lag, trend)

    if is_pandas:
        return pd.Series(result, index=index, name='sadf')
    return result


class SADFTest(SISOTransform):
    """
    SADF Transform for FeatureKit pipeline.

    Detects explosive price behavior (bubbles) using supremum ADF test.

    :param input_col: Input price column name (default 'close')
    :param min_window: Minimum window for ADF calculation (default 20)
    :param max_window: Maximum window (default 100, None for expanding)
    :param max_lag: ADF lag order (default 0)
    :param trend: Include time trend (default True)
    :param use_log: Apply log transform (default True)
    """

    def __init__(
        self,
        input_col: str = 'close',
        min_window: int = 20,
        max_window: Optional[int] = 100,
        max_lag: int = 0,
        trend: bool = True,
        use_log: bool = True
    ):
        super().__init__(input_col, 'sadf')
        self.min_window = min_window
        self.max_window = max_window
        self.max_lag = max_lag
        self.trend = trend
        self.use_log = use_log

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        if self.use_log:
            prices = np.log(prices)

        max_w = self.max_window if self.max_window is not None else -1
        result = _sadf_core(prices, self.min_window, max_w, self.max_lag, self.trend)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)
