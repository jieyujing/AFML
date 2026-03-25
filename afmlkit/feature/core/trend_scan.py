"""
Trend Scanning (t-value based) forward-looking labeling model.

Implements the Trend Scan algorithm from Marcos López de Prado's
*Machine Learning for Asset Managers* (MLAM). For each event timestamp,
forward-scans through multiple window lengths, fitting OLS regressions
and selecting the window with the maximum absolute t-statistic.
This is strictly a Labeling method and contains future information.

This module is split into:
    - ``_trend_scan_core``: Numba ``@njit`` compiled kernel for raw array processing.
    - ``trend_scan_labels``: Python/Pandas frontend that handles index alignment
      and returns a structured DataFrame with future timestamp ``t1``.

References
----------
    López de Prado, M. (2020). *Machine Learning for Asset Managers*.
    Cambridge University Press. Chapter 3.5 — Trend Scanning.
"""

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray
from typing import Tuple

from afmlkit.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numba Backend
# ---------------------------------------------------------------------------

@njit(nogil=True)
def _trend_scan_core(
    prices: NDArray[np.float64],
    event_indices: NDArray[np.int64],
    L_windows: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Core Numba kernel for Trend Scanning.

    For each event located at ``prices[event_indices[i]]``, the function
    forward-scans through every window length in *L_windows*, computes an
    OLS regression slope t-statistic using closed-form mathematics (no
    external library calls), and selects the window whose slope has the
    maximum absolute t-value.

    Parameters
    ----------
    prices : 1-D float64 array
        Full, continuous price series (e.g. close prices). Must be a plain
        NumPy array — all Pandas / DatetimeIndex alignment must be completed
        *before* calling this function.
    event_indices : 1-D int64 array
        Positional indices into *prices* where events occurred (e.g. from
        CUSUM filter). Each index ``event_indices[i]`` marks the point *from
        which* the forward window is evaluated. (Contains future information).
    L_windows : 1-D int64 array
        Sorted array of candidate window lengths to scan (e.g.
        ``[10, 20, 30, …, 100]``). Each element ``L`` means "use the next
        ``L`` price observations starting at the event".

    Returns
    -------
    t_values : 1-D float64 array, length = len(event_indices)
        The signed t-statistic of the slope for the optimal window.
    sides : 1-D int64 array, length = len(event_indices)
        ``+1`` (uptrend) or ``-1`` (downtrend), derived from ``sign(t_value)``.
        Zero-variance events default to ``side = 0``.
    best_windows : 1-D int64 array, length = len(event_indices)
        The window length ``L`` that yielded the maximum ``|t_value|``.

    Notes
    -----
    *   The OLS model fitted in each window is ``y = a + b·x`` where
        ``x = [0, 1, …, L-1]`` and ``y = prices[idx : idx+L]``.
    *   A closed-form expression for ``Sxx = n(n² − 1) / 12`` is used
        (valid for consecutive integer regressors) to avoid a second pass.
    *   Zero-variance guards (``ε = 1e-12``) protect against flat-price
        segments that would cause division by zero.
    *   Windows with fewer than 3 observations are skipped because the
        t-statistic denominator ``(n − 2)`` would be ≤ 0.
    """
    EPSILON = 1e-12
    n_events = len(event_indices)

    # Pre-allocate output
    t_values = np.zeros(n_events, dtype=np.float64)
    sides = np.zeros(n_events, dtype=np.int64)
    best_windows = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        idx = event_indices[i]
        best_abs_t = 0.0
        best_t = 0.0
        best_L = np.int64(0)

        for w in range(len(L_windows)):
            L = L_windows[w]
            start = idx

            # --- Guard: not enough future history ---
            if start + L > len(prices):
                continue

            n = L  # number of data points in this window
            if n < 3:
                continue  # t-stat requires n > 2

            # --- OLS closed-form ---
            # Regressor: x = [0, 1, ..., n-1]
            x_mean = (n - 1) / 2.0
            Sxx = n * (n * n - 1) / 12.0  # sum (x_i - x_mean)^2

            # Compute y_mean
            y_sum = 0.0
            for j in range(n):
                y_sum += prices[start + j]
            y_mean = y_sum / n

            # Compute Sxy, Syy in a single pass
            Sxy = 0.0
            Syy = 0.0
            for j in range(n):
                dx = j - x_mean
                dy = prices[start + j] - y_mean
                Sxy += dx * dy
                Syy += dy * dy

            # Slope
            if Sxx < EPSILON:
                continue
            b = Sxy / Sxx

            # Zero y-variance guard: if Syy ≈ 0, the series is flat
            # → b ≈ 0, t_value = 0 (no trend signal). Skip this window.
            if Syy < EPSILON:
                continue

            # Residual sum of squares: RSS = Syy - b * Sxy
            RSS = Syy - b * Sxy
            if RSS < 0.0:
                RSS = 0.0  # numerical guard

            # Standard error of the slope
            SE_b_sq = RSS / ((n - 2) * Sxx)
            if SE_b_sq < EPSILON:
                # Non-flat but perfectly linear (RSS ≈ 0, b ≠ 0).
                # The slope is exact → t-value is conceptually ±∞.
                abs_t_candidate = 1e12
                t_candidate = 1e12 if b >= 0.0 else -1e12
            else:
                SE_b = np.sqrt(SE_b_sq)
                t_candidate = b / SE_b
                abs_t_candidate = abs(t_candidate)

            if abs_t_candidate > best_abs_t:
                best_abs_t = abs_t_candidate
                best_t = t_candidate
                best_L = L

        # --- Write results for this event ---
        t_values[i] = best_t
        if best_t > 0.0:
            sides[i] = np.int64(1)
        elif best_t < 0.0:
            sides[i] = np.int64(-1)
        else:
            sides[i] = np.int64(0)
        best_windows[i] = best_L

    return t_values, sides, best_windows


# ---------------------------------------------------------------------------
# Python / Pandas Frontend
# ---------------------------------------------------------------------------

def trend_scan_labels(
    price_series: pd.Series,
    t_events: pd.DatetimeIndex,
    L_windows: list | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute Trend Scan labels for a set of event timestamps.

    This is the high-level wrapper that:

    1. Validates and aligns Pandas indices between *price_series* and
       *t_events*.
    2. Converts everything to fast NumPy arrays (float64 / int64).
    3. Dispatches to the Numba kernel :func:`_trend_scan_core`.
    4. Formats the output into a DataFrame indexed by the valid subset
       of *t_events*.

    Parameters
    ----------
    price_series : pd.Series
        Close prices with a ``DatetimeIndex``. The index must be monotonically
        increasing and should cover at least ``max(L_windows)`` observations
        before the first event.
    t_events : pd.DatetimeIndex
        Discrete event timestamps emitted by the CUSUM filter.  Events that
        fall outside *price_series* will be silently dropped.
    L_windows : list[int] | np.ndarray | None, optional
        Candidate forward-looking window lengths.  Defaults to
        ``[10, 20, 30, …, 100]`` if *None*.

    Returns
    -------
    pd.DataFrame
        Indexed by the (aligned) subset of *t_events* with columns:

        ``t1``
            The exact future timestamp (Datetime) that marks the end of the
            optimal trend window.
        ``t_value``
            Signed t-statistic (float64) of the OLS slope in the optimal
            window.
        ``side``
            ``+1`` (up) or ``-1`` (down), i.e. ``sign(t_value)``.
            ``0`` if the segment had zero variance.

    Raises
    ------
    ValueError
        If *price_series* does not have a ``DatetimeIndex``, or if no valid
        events remain after alignment.

    Examples
    --------
    >>> from afmlkit.feature.core.trend_scan import trend_scan_labels
    >>> trend_df = trend_scan_labels(close_prices, t_events, L_windows=[10, 20, 50, 100])
    >>> trend_df.head()
                         t1    t_value  side
    2024-01-15 10:03:00  50   3.2145     1
    2024-01-15 11:47:00  20  -2.0011    -1
    """
    # ---- Input validation ----
    if not isinstance(price_series.index, pd.DatetimeIndex):
        raise ValueError(
            "price_series must have a DatetimeIndex. "
            f"Got {type(price_series.index).__name__}."
        )

    if L_windows is None:
        L_windows = list(range(10, 101, 10))
    L_windows_arr = np.asarray(L_windows, dtype=np.int64)

    if len(L_windows_arr) == 0:
        raise ValueError("L_windows must contain at least one window length.")

    # ---- Index alignment ----
    # Keep only events that appear in the price index
    valid_events = t_events[t_events.isin(price_series.index)]
    if len(valid_events) == 0:
        raise ValueError(
            "No events in t_events could be aligned with price_series index. "
            "Check that both share the same DatetimeIndex."
        )

    event_positions = price_series.index.get_indexer(valid_events)
    mask = event_positions >= 0
    event_positions = event_positions[mask].astype(np.int64)
    valid_events = valid_events[mask]

    if len(event_positions) == 0:
        raise ValueError("All event positions resolved to -1 after alignment.")

    # ---- Prepare numeric arrays ----
    prices = price_series.values.astype(np.float64)

    logger.info(
        "Running Trend Scan: %d events × %d windows (max L=%d)...",
        len(event_positions),
        len(L_windows_arr),
        int(L_windows_arr.max()),
    )

    # ---- Dispatch to Numba kernel ----
    t_values, sides, best_windows = _trend_scan_core(
        prices, event_positions, L_windows_arr
    )

    # ---- Format output DataFrame ----
    t1_timestamps = []
    for i in range(len(event_positions)):
        L = best_windows[i]
        if L == 0:
            t1_timestamps.append(pd.NaT)
        else:
            t1_idx = event_positions[i] + L - 1
            t1_timestamps.append(price_series.index[t1_idx])

    result = pd.DataFrame(
        {
            "t1": t1_timestamps,
            "t_value": t_values,
            "side": sides.astype(np.int8),
        },
        index=valid_events,
    )

    # Log summary statistics
    n_positive = int((result["side"] == 1).sum())
    n_negative = int((result["side"] == -1).sum())
    n_zero = int((result["side"] == 0).sum())
    logger.info(
        "Trend Scan complete: %d events → %d up / %d down / %d flat",
        len(result),
        n_positive,
        n_negative,
        n_zero,
    )

    return result
