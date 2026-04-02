"""
Trend indicators for financial time series analysis.
"""
from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nogil=True)
def _supertrend_core(
    close: NDArray[np.float64],
    atr_values: NDArray[np.float64],
    atr_period: int,
    multiplier: float,
) -> Tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64], NDArray[np.float64]]:
    """
    Core Numba kernel for SuperTrend calculation.

    Algorithm:
    1. Base upper = close + multiplier * ATR
    2. Base lower = close - multiplier * ATR
    3. Final bands adjust based on trend:
       - Uptrend (direction=+1): lower band acts as support (can only move up)
       - Downtrend (direction=-1): upper band acts as resistance (can only move down)
    4. Trend reversal when price crosses active band

    :param close: Array of close prices
    :param atr_values: Pre-computed ATR values
    :param atr_period: ATR calculation period (for warmup skip)
    :param multiplier: ATR multiplier for band width
    :returns: (trend_line, direction, upper_band, lower_band)
    """
    n = len(close)

    # Initialize output arrays
    trend_line = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int8)  # 0: undefined, +1: uptrend, -1: downtrend
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)

    # Skip warmup period
    for i in range(atr_period, n):
        if np.isnan(atr_values[i]) or np.isnan(close[i]):
            continue

        # Basic bands
        base_upper = close[i] + multiplier * atr_values[i]
        base_lower = close[i] - multiplier * atr_values[i]

        # First valid point: initialize with basic bands
        if i == atr_period or np.isnan(trend_line[i - 1]):
            upper_band[i] = base_upper
            lower_band[i] = base_lower
            # Determine initial direction based on price relative to midpoint
            mid = (base_upper + base_lower) / 2.0
            if close[i] >= mid:
                direction[i] = 1  # uptrend
                trend_line[i] = lower_band[i]
            else:
                direction[i] = -1  # downtrend
                trend_line[i] = upper_band[i]
            continue

        # Get previous values
        prev_upper = upper_band[i - 1]
        prev_lower = lower_band[i - 1]
        prev_direction = direction[i - 1]

        # Dynamic band adjustment
        # Lower band: can only move up during uptrend (acts as support)
        if base_lower > prev_lower or prev_direction == -1:
            lower_band[i] = base_lower
        else:
            lower_band[i] = prev_lower

        # Upper band: can only move down during downtrend (acts as resistance)
        if base_upper < prev_upper or prev_direction == 1:
            upper_band[i] = base_upper
        else:
            upper_band[i] = prev_upper

        # Trend direction determination
        if prev_direction == 1:  # Previous uptrend
            if close[i] < lower_band[i]:  # Price breaks below support
                direction[i] = -1  # Switch to downtrend
                trend_line[i] = upper_band[i]
            else:
                direction[i] = 1  # Continue uptrend
                trend_line[i] = lower_band[i]

        elif prev_direction == -1:  # Previous downtrend
            if close[i] > upper_band[i]:  # Price breaks above resistance
                direction[i] = 1  # Switch to uptrend
                trend_line[i] = lower_band[i]
            else:
                direction[i] = -1  # Continue downtrend
                trend_line[i] = upper_band[i]

        else:  # Previous undefined (shouldn't happen after warmup)
            mid = (upper_band[i] + lower_band[i]) / 2.0
            if close[i] >= mid:
                direction[i] = 1
                trend_line[i] = lower_band[i]
            else:
                direction[i] = -1
                trend_line[i] = upper_band[i]

    return trend_line, direction, upper_band, lower_band


def supertrend(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate SuperTrend indicator.

    SuperTrend is a trend-following indicator that uses ATR to determine
    dynamic support/resistance levels. It's particularly useful for:
    - Identifying trend direction
    - Setting trailing stop-losses
    - Generating buy/sell signals

    :param high: Array of high prices
    :param low: Array of low prices
    :param close: Array of close prices
    :param atr_period: ATR calculation period (default 10)
    :param multiplier: ATR multiplier for band width (default 3.0)
    :returns: (trend_line, direction, upper_band, lower_band)
              - trend_line: Active trend line value (lower in uptrend, upper in downtrend)
              - direction: +1 (uptrend/buy), -1 (downtrend/sell)
              - upper_band: Final upper band (resistance in downtrend)
              - lower_band: Final lower band (support in uptrend)

    Example:
        >>> from afmlkit.feature.core.trend import supertrend
        >>> trend_line, direction, upper, lower = supertrend(high, low, close, 10, 3.0)
        >>> # direction == 1 means uptrend (price above trend_line)
        >>> # direction == -1 means downtrend (price below trend_line)
    """
    from afmlkit.feature.core.volatility import atr

    # Calculate ATR
    atr_values = atr(high, low, close, atr_period, ema_based=True, normalize=False)

    # Call Numba kernel
    return _supertrend_core(close, atr_values, atr_period, multiplier)


@njit
def adx_core(high, low, close, length):
    """
    Calculate the Average Directional Index (ADX) with specified length.

    The ADX is a technical analysis indicator used to determine the strength of a trend.
    Higher values indicate a stronger trend (regardless of direction).

    :param high: Array of high prices
    :param low: Array of low prices
    :param close: Array of close prices
    :param length: Period for ADX calculation
    :return: ADX indicator values
    """
    size = len(high)

    # Arrays to store True Range, +DM, -DM
    tr = np.zeros(size)
    plus_dm = np.zeros(size)
    minus_dm = np.zeros(size)

    # Calculate True Range, +DM, -DM
    for i in range(1, size):
        # True Range = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

        # Directional Movement
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]

        # +DM
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0

        # -DM
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0

    # Initialize smoothed values using Wilder's smoothing method
    smoothed_tr = np.zeros(size)
    smoothed_plus_dm = np.zeros(size)
    smoothed_minus_dm = np.zeros(size)

    # First calculation is simple average
    if size >= length + 1:
        smoothed_tr[length] = np.sum(tr[1:length+1])
        smoothed_plus_dm[length] = np.sum(plus_dm[1:length+1])
        smoothed_minus_dm[length] = np.sum(minus_dm[1:length+1])

    # Apply Wilder's smoothing for subsequent values
    for i in range(length + 1, size):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / length) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / length) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / length) + minus_dm[i]

    # Calculate +DI, -DI
    plus_di = np.zeros(size)
    minus_di = np.zeros(size)
    for i in range(length, size):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * (smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di[i] = 100 * (smoothed_minus_dm[i] / smoothed_tr[i])

    # Calculate DX
    dx = np.zeros(size)
    for i in range(length, size):
        if (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))

    # Calculate ADX
    adx = np.zeros(size)

    # First ADX is simple average of DX
    if size >= 2*length:
        adx[2*length-1] = np.mean(dx[length:2*length])

    # Apply Wilder's smoothing for subsequent ADX values
    for i in range(2*length, size):
        adx[i] = ((adx[i-1] * (length-1)) + dx[i]) / length

    return adx
