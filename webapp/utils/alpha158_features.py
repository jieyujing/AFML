"""
Alpha158 Features - FFD-transformed feature engineering.

This module implements Qlib Alpha158-style features using Fractional Differentiation
(FFD) to create stationary, memory-preserving feature series.

Key design principles:
1. Apply FFD only to log_price to get stationary base series X_tilde
2. Use X_tilde directly for momentum features (no returns calculation)
3. Feed X_tilde into SMA/EMA/Rank operations (they don't increase differencing order)
4. Volume features use raw data (volume is already stationary)

All features use 'ffd_*' prefix for clear identification.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.volatility import ewms


# Default configuration
DEFAULT_VOLATILITY_SPANS = [5, 10, 20]
DEFAULT_MA_WINDOWS = [5, 10, 20]
DEFAULT_EMA_WINDOWS = [5, 10]
DEFAULT_RANK_WINDOW = 20
DEFAULT_FFD_THRES = 1e-4
DEFAULT_FFD_D_STEP = 0.05


def compute_ffd_base(
    close: pd.Series,
    thres: float = DEFAULT_FFD_THRES,
    d_step: float = DEFAULT_FFD_D_STEP
) -> Tuple[pd.Series, float]:
    """
    Compute FFD base series and return optimal d*.

    Applies fractional differentiation to log(close) using the Fixed-Width Window
    method. Automatically searches for the minimum d that makes the series stationary.

    Args:
        close: Closing price series (can be raw or log-transformed)
        thres: FFD weight truncation threshold (default: 1e-4)
        d_step: Step size for d search (default: 0.05)

    Returns:
        tuple: (ffd_log_price Series, optimal d value)

    Example:
        >>> close = pd.Series([100, 101, 102, ...])
        >>> ffd_series, d_star = compute_ffd_base(close)
    """
    # Ensure input is float64 for numerical stability
    close = close.astype(np.float64)

    # Compute log price if close is in raw form (heuristic: all values > 0)
    if close.min() > 0:
        log_price = np.log(close)
    else:
        log_price = close.copy()

    # Remove any existing NaN before FFD
    log_price = log_price.dropna()

    # Find optimal d* using ADF test
    optimal_d = optimize_d(log_price, thres=thres, d_step=d_step)

    # Apply FFD with optimal d
    log_price_named = pd.Series(
        log_price.values,
        index=log_price.index,
        name="log_close"
    )
    ffd_series = frac_diff_ffd(log_price_named, d=optimal_d, thres=thres)

    # Rename for clarity
    ffd_series.name = "ffd_log_price"

    return ffd_series, optimal_d


def compute_ffd_volatility(
    ffd_series: pd.Series,
    spans: List[int] = None
) -> pd.DataFrame:
    """
    Compute volatility features based on FFD series.

    Calculates:
    - Rolling standard deviation (ffd_vol_std_{span})
    - EWM volatility (ffd_vol_ewm_{span})

    Args:
        ffd_series: FFD-transformed series (typically ffd_log_price)
        spans: List of span values for volatility calculation (default: [5, 10, 20])

    Returns:
        DataFrame: Contains ffd_vol_std_* and ffd_vol_ewm_* columns
    """
    if spans is None:
        spans = DEFAULT_VOLATILITY_SPANS

    result = pd.DataFrame(index=ffd_series.index)
    ffd_values = ffd_series.values.astype(np.float64)

    for span in spans:
        # Rolling standard deviation
        std_col = f"ffd_vol_std_{span}"
        result[std_col] = pd.Series(ffd_values, index=ffd_series.index).rolling(window=span).std()

        # EWM volatility (using log returns of FFD series)
        ewm_col = f"ffd_vol_ewm_{span}"
        ewm_vol = ewms(ffd_values, span)
        result[ewm_col] = pd.Series(ewm_vol, index=ffd_series.index)

    return result
