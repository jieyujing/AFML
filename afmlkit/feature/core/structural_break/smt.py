"""
Sub-Martingale and Super-Martingale tests for trend detection.

S_t = sum(w_i * r_i) / sqrt(sum(w_i^2))

where w_i = lambda^(t-i) are exponential weights.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional, Tuple
import pandas as pd

from afmlkit.feature.base import SISOTransform, SIMOTransform


@njit(nogil=True)
def _sub_martingale_core(
    prices: NDArray[np.float64],
    decay: float,
    window_size: int
) -> NDArray[np.float64]:
    """
    Sub-martingale test core computation.

    H0: E[P_{t+1} | P_t, ...] <= P_t (not an uptrend)

    Formula: S_t = sum(w_i * r_i) / sqrt(sum(w_i^2))

    :param prices: Price series
    :param decay: Decay factor (0-1)
    :param window_size: Window size (-1 for expanding)
    :returns: Sub-martingale statistics
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    # Compute returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(0.0, returns)

    for t in range(1, n):
        if window_size > 0:
            start = max(0, t - window_size + 1)
        else:
            start = 0

        weighted_sum = 0.0
        weight_sq_sum = 0.0

        for i in range(start, t + 1):
            if i == 0:
                continue
            w = decay ** (t - i)
            weighted_sum += w * returns[i]
            weight_sq_sum += w * w

        if weight_sq_sum > 1e-12:
            result[t] = weighted_sum / np.sqrt(weight_sq_sum)

    return result


@njit(nogil=True)
def _super_martingale_core(
    prices: NDArray[np.float64],
    decay: float,
    window_size: int
) -> NDArray[np.float64]:
    """
    Super-martingale test - negative of sub-martingale.

    H0: E[P_{t+1} | P_t, ...] >= P_t (not a downtrend)
    """
    return -_sub_martingale_core(prices, decay, window_size)


def sub_martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    Sub-martingale test for uptrend detection.

    **Note**: The first element is always NaN because the first return cannot
    be computed (no previous price point).

    :param prices: Price series
    :param decay: Exponential decay factor (default 0.95)
    :param window: Rolling window (None for expanding)
    :returns: Sub-martingale statistics
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    arr = np.asarray(prices, dtype=np.float64)
    window_size = window if window is not None else -1

    result = _sub_martingale_core(arr, decay, window_size)

    if is_pandas:
        return pd.Series(result, index=index, name='sub_martingale')
    return result


def super_martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    Super-martingale test for downtrend detection.

    **Note**: The first element is always NaN because the first return cannot
    be computed (no previous price point).

    :param prices: Price series
    :param decay: Exponential decay factor (default 0.95)
    :param window: Rolling window (None for expanding)
    :returns: Super-martingale statistics
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    arr = np.asarray(prices, dtype=np.float64)
    window_size = window if window is not None else -1

    result = _super_martingale_core(arr, decay, window_size)

    if is_pandas:
        return pd.Series(result, index=index, name='super_martingale')
    return result


def martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None
) -> Tuple[Union[pd.Series, NDArray], Union[pd.Series, NDArray]]:
    """
    Combined sub and super-martingale tests.

    **Note**: The first element of both results is always NaN because the
    first return cannot be computed (no previous price point).

    :returns: Tuple of (sub_martingale, super_martingale)
    """
    sub = sub_martingale_test(prices, decay, window)
    sup = super_martingale_test(prices, decay, window)
    return sub, sup


class SubMartingaleTest(SISOTransform):
    """
    Sub-martingale Transform for uptrend detection.

    **Note**: The first output value is always NaN because the first return
    cannot be computed (no previous price point).

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, 'sub_martingale')
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        if np.any(prices <= 0):
            raise ValueError("Price values must be positive (greater than 0)")
        window_size = self.window if self.window is not None else -1
        result = _sub_martingale_core(prices, self.decay, window_size)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class SuperMartingaleTest(SISOTransform):
    """
    Super-martingale Transform for downtrend detection.

    **Note**: The first output value is always NaN because the first return
    cannot be computed (no previous price point).

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, 'super_martingale')
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        if np.any(prices <= 0):
            raise ValueError("Price values must be positive (greater than 0)")
        window_size = self.window if self.window is not None else -1
        result = _super_martingale_core(prices, self.decay, window_size)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class MartingaleTest(SIMOTransform):
    """
    Combined martingale test Transform - outputs both sub and super.

    Produces two columns: ['{col}_sub_martingale', '{col}_super_martingale']

    **Note**: The first output value of both series is always NaN because the
    first return cannot be computed (no previous price point).

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, ['sub_martingale', 'super_martingale'])
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        prices = x[self.requires[0]].values
        if np.any(prices <= 0):
            raise ValueError("Price values must be positive (greater than 0)")
        window_size = self.window if self.window is not None else -1

        sub = _sub_martingale_core(prices, self.decay, window_size)
        sup = _super_martingale_core(prices, self.decay, window_size)

        return (
            pd.Series(sub, index=x.index, name=self.output_name[0]),
            pd.Series(sup, index=x.index, name=self.output_name[1])
        )

    def _nb(self, x: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return self._pd(x)
