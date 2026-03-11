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


def compute_ffd_ma(
    ffd_series: pd.Series,
    windows: List[int] = None,
    ema_windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute moving average features based on FFD series.

    Calculates:
    - Simple moving average (ffd_ma_{window})
    - Exponential moving average (ffd_ema_{window})

    Args:
        ffd_series: FFD-transformed series
        windows: List of window sizes for SMA (default: [5, 10, 20])
        ema_windows: List of window sizes for EMA (default: same as windows)

    Returns:
        DataFrame: Contains ffd_ma_* and ffd_ema_* columns
    """
    if windows is None:
        windows = DEFAULT_MA_WINDOWS

    if ema_windows is None:
        ema_windows = windows

    result = pd.DataFrame(index=ffd_series.index)
    ffd_values = ffd_series.values.astype(np.float64)

    # Simple Moving Averages
    for window in windows:
        ma_col = f"ffd_ma_{window}"
        ma_values = sma(ffd_values, window=window)
        result[ma_col] = pd.Series(ma_values, index=ffd_series.index)

    # Exponential Moving Averages
    for window in ema_windows:
        ema_col = f"ffd_ema_{window}"
        ema_values = ewma(ffd_values, span=window)
        result[ema_col] = pd.Series(ema_values, index=ffd_series.index)

    return result


def compute_ffd_rank(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    rank_window: int = DEFAULT_RANK_WINDOW
) -> pd.DataFrame:
    """
    Compute rolling rank features (percentile rank within rolling window).

    This implements temporal rank: where does current value sit relative to
    the past `rank_window` observations? Single-stock alternative to cross-sectional rank.

    Args:
        df: DataFrame containing base features
        feature_cols: List of feature columns to rank (default: all ffd_* columns)
        rank_window: Rolling window size for percentile rank (default: 20)

    Returns:
        DataFrame: Original columns + ffd_rank_{feature}_{window} columns
    """
    result = df.copy()

    if feature_cols is None:
        # Default: rank all ffd_* prefixed columns
        feature_cols = [col for col in df.columns if col.startswith('ffd_')]

    for col in feature_cols:
        if col not in df.columns:
            continue

        rank_col = f"ffd_rank_{col}_{rank_window}"
        # Percentile rank: 0 = lowest in window, 1 = highest in window
        result[rank_col] = df[col].rolling(window=rank_window, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

    return result


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume and price-based features.

    These features use raw OHLCV data (not FFD-transformed) because:
    - Volume is already a stationary flow variable
    - Price levels are needed for VWAP, amount calculations

    Calculates:
    - ffd_vwap: Volume-weighted average price (rolling 5-period)
    - ffd_amount: Dollar amount (close * volume)
    - ffd_amplification: Price range relative to open

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame: Contains ffd_vwap, ffd_amount, ffd_amplification columns
    """
    result = pd.DataFrame(index=df.index)

    # Amount (dollar volume) - only requires close and volume
    if 'close' in df.columns and 'volume' in df.columns:
        result['ffd_amount'] = df['close'] * df['volume']

    # VWAP - rolling 5-period volume-weighted average price (requires high/low/close/volume)
    if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        result['ffd_vwap'] = (typical_price * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()

        # Amplification (range / open) - requires open
        if 'open' in df.columns:
            result['ffd_amplification'] = (df['high'] - df['low']) / df['open']

    return result
