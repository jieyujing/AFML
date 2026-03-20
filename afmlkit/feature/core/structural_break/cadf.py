"""
CADF (Conditional ADF) test - expected ADF given it exceeds quantile threshold.

Quantifies average bubble strength when SADF > QADF threshold.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import MISOTransform
from afmlkit.feature.core.structural_break.sadf import _sadf_core
from afmlkit.feature.core.structural_break.qadf import _rolling_quantile_core


@njit(nogil=True)
def _conditional_expectation_core(
    adf_values: NDArray[np.float64],
    quantile_values: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Conditional expectation: E[ADF_t | ADF_t > QADF_{t-L:t}].

    :param adf_values: ADF/SADF values
    :param quantile_values: Rolling quantile (threshold) values
    :param window: Window size (should match QADF window)
    :returns: Conditional expectation values
    """
    n = len(adf_values)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        if adf_values[i] > quantile_values[i]:
            window_start = max(0, i - window + 1)
            count = 0
            total = 0.0

            for j in range(window_start, i + 1):
                if adf_values[j] > quantile_values[j]:
                    total += adf_values[j]
                    count += 1

            if count > 0:
                result[i] = total / count

    return result


def cadf_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    sadf_values: Optional[NDArray[np.float64]] = None,
    qadf_values: Optional[NDArray[np.float64]] = None,
    min_window: int = 20,
    max_window: int = 100,
    quantile_window: int = 20,
    quantile: float = 0.95,
    max_lag: int = 0
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    CADF test - conditional expectation of ADF given threshold exceedance.

    :param prices: Price series (if sadf_values not provided)
    :param sadf_values: Pre-computed SADF values (optional)
    :param qadf_values: Pre-computed QADF values (optional)
    :param min_window: SADF minimum window
    :param max_window: SADF maximum window
    :param quantile_window: QADF rolling window
    :param quantile: QADF quantile level
    :param max_lag: ADF lag order
    :returns: CADF series
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    if sadf_values is None:
        prices_arr = np.asarray(prices, dtype=np.float64)
        if np.any(prices_arr <= 0):
            raise ValueError("Price values must be positive (greater than 0)")
        log_prices = np.log(prices_arr)
        sadf_arr = _sadf_core(log_prices, min_window, max_window, max_lag, True)
    else:
        sadf_arr = np.asarray(sadf_values, dtype=np.float64)

    if qadf_values is None:
        qadf_arr = _rolling_quantile_core(sadf_arr, quantile_window, quantile)
    else:
        qadf_arr = np.asarray(qadf_values, dtype=np.float64)

    result = _conditional_expectation_core(sadf_arr, qadf_arr, quantile_window)

    if is_pandas:
        return pd.Series(result, index=index, name='cadf')
    return result


class CADFTest(MISOTransform):
    """
    CADF Transform for FeatureKit pipeline.

    Computes conditional expected SADF when it exceeds QADF threshold.

    :param min_window: SADF minimum window (default 20)
    :param max_window: SADF maximum window (default 100)
    :param quantile_window: QADF rolling window (default 20)
    :param quantile: QADF quantile level (default 0.95)
    :param sadf_values: Pre-computed SADF values (optional)
    :param qadf_values: Pre-computed QADF values (optional)
    """

    def __init__(
        self,
        min_window: int = 20,
        max_window: int = 100,
        quantile_window: int = 20,
        quantile: float = 0.95,
        sadf_values: Optional[NDArray[np.float64]] = None,
        qadf_values: Optional[NDArray[np.float64]] = None
    ):
        super().__init__(['close'], 'cadf')
        self.min_window = min_window
        self.max_window = max_window
        self.quantile_window = quantile_window
        self.quantile = quantile
        self.sadf_values = sadf_values
        self.qadf_values = qadf_values

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        if self.sadf_values is not None and self.qadf_values is not None:
            result = _conditional_expectation_core(
                self.sadf_values,
                self.qadf_values,
                self.quantile_window
            )
        else:
            prices = x[self.requires[0]].values
            if np.any(prices <= 0):
                raise ValueError("Price values must be positive (greater than 0)")
            result = cadf_test(
                prices,
                min_window=self.min_window,
                max_window=self.max_window,
                quantile_window=self.quantile_window,
                quantile=self.quantile
            )
            # cadf_test returns ndarray when input is ndarray
            if hasattr(result, 'values'):
                result = result.values

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)
