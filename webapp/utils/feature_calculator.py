# webapp/utils/feature_calculator.py
"""特征计算器 - 实现与 scripts/feature_engineering.py 一致的特征计算逻辑"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

# afmlkit core imports
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.volatility import (
    ewms,
    parkinson_range,
    atr,
    bollinger_percent_b,
    variance_ratio_1_4_core,
)
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.momentum import rsi_wilder, roc, stoch_k
from afmlkit.feature.core.correlation import rolling_price_volume_correlation

# Optional Alpha158 features
# from webapp.utils.alpha158_features import compute_alpha158_features


# ── Hyper-parameters ──────────────────────────────────────────────────
VOL_SPANS = [10, 50, 100]  # EWM volatility windows
EMA_SHORT_SPAN = 12  # Short EMA for crossover
EMA_LONG_SPAN = 26  # Long EMA
RSI_WINDOW = 14  # RSI lookback
FRACDIFF_THRES = 1e-4  # FFD weight truncation threshold
FRACDIFF_D_STEP = 0.05  # Step size for d optimisation


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算对数收益率"""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_volatility_features(
    df: pd.DataFrame, spans: List[int] = None
) -> pd.DataFrame:
    spans = spans or VOL_SPANS
    df = df.copy()

    log_ret = df["log_return"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64) if "high" in df.columns else None
    low = df["low"].values.astype(np.float64) if "low" in df.columns else None

    # EWM Volatility
    for span in spans:
        col_name = f"vol_ewm_{span}"
        df[col_name] = ewms(log_ret, span)

    # Parkinson volatility
    if high is not None and low is not None:
        df["vol_parkinson"] = parkinson_range(high, low)

    # ATR
    if high is not None and low is not None:
        df["vol_atr_14"] = atr(high, low, close, window=14)

    # Bollinger %B
    df["vol_bb_pct_b_20"] = bollinger_percent_b(close, window=20, num_std=2.0)

    # Variance Ratio
    df["trend_variance_ratio_20"] = variance_ratio_1_4_core(
        close, window=20, ddof=1, ret_type="log"
    )

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64) if "high" in df.columns else None
    low = df["low"].values.astype(np.float64) if "low" in df.columns else None
    volume = df["volume"].values.astype(np.float64) if "volume" in df.columns else None
    log_close = np.log(close)

    # EMA short/long
    ema_short = ewma(close, EMA_SHORT_SPAN)
    ema_long = ewma(close, EMA_LONG_SPAN)
    df["ema_short"] = ema_short
    df["ema_long"] = ema_long

    # Log price distance to EMAs
    df["log_dist_ema_short"] = log_close - np.log(ema_short)
    df["log_dist_ema_long"] = log_close - np.log(ema_long)

    # EMA difference
    df["ema_diff"] = ema_short - ema_long

    # RSI (Wilder)
    df["rsi_14"] = rsi_wilder(close, RSI_WINDOW)

    # ROC
    df["mom_roc_10"] = roc(close, period=10)

    # Stochastic %K
    if high is not None and low is not None:
        df["mom_stoch_k_14"] = stoch_k(close, low, high, length=14)

    # Price-Volume Correlation
    if volume is not None:
        df["corr_pv_10"] = rolling_price_volume_correlation(close, volume, window=10)

    # Amihud Illiquidity
    if "log_return" in df.columns and volume is not None:
        df["liq_amihud"] = np.abs(df["log_return"]) / (df["volume"] * df["close"]) * 1e6

    # Relative Volume
    if volume is not None:
        vol_sma_20 = sma(volume, window=20)
        df["vol_rel_20"] = volume / vol_sma_20

    return df


def compute_fracdiff_features(
    df: pd.DataFrame,
    thres: float = FRACDIFF_THRES,
    d_step: float = FRACDIFF_D_STEP,
) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    log_price = np.log(df["close"])
    log_price_series = pd.Series(log_price, index=df.index, name="log_close")

    optimal_d = optimize_d(log_price_series, thres=thres, d_step=d_step)
    ffd_series = frac_diff_ffd(log_price_series, d=optimal_d, thres=thres)
    ffd_series.name = "ffd_log_price"
    df["ffd_log_price"] = ffd_series

    return df, optimal_d


def purge_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values."""
    return df.dropna()


def compute_all_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """计算所有特征

    Args:
        df: 输入数据 DataFrame
        config: 特征配置

    Returns:
        Tuple[特征 DataFrame, 元数据字典]
    """
    config = config or {}
    metadata = {}

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    df = df.sort_index()
    df = compute_log_returns(df)

    vol_spans = config.get("volatility", {}).get("spans", VOL_SPANS)
    df = compute_volatility_features(df, spans=vol_spans)
    df = compute_momentum_features(df)

    fracdiff_config = config.get("fractional_diff", {})
    thres = fracdiff_config.get("threshold", FRACDIFF_THRES)
    d_step = fracdiff_config.get("d_step", FRACDIFF_D_STEP)
    df, optimal_d = compute_fracdiff_features(df, thres=thres, d_step=d_step)
    metadata["optimal_d"] = optimal_d

    # --- Alpha158 features (optional) ---
    alpha158_config = config.get("alpha158", {})
    if alpha158_config.get("enabled", False):
        from webapp.utils.alpha158_features import compute_alpha158_features

        alpha158_df, alpha158_meta = compute_alpha158_features(
            df, config=alpha158_config
        )

        # Merge features (with ffd_* prefix to avoid collisions)
        for col in alpha158_df.columns:
            if col not in df.columns:
                df[col] = alpha158_df[col]

        # Update metadata
        metadata["alpha158_enabled"] = True
        metadata["alpha158_columns"] = alpha158_df.columns.tolist()
        metadata["alpha158_optimal_d"] = alpha158_meta.get("optimal_d")

    n_before = len(df)
    df = purge_nan_rows(df)
    metadata["rows_before_clean"] = n_before
    metadata["rows_after_clean"] = len(df)
    metadata["rows_dropped"] = n_before - len(df)

    metadata["final_shape"] = df.shape
    metadata["feature_columns"] = list(df.columns)

    if "bin" in df.columns:
        metadata["label_distribution"] = df["bin"].value_counts().to_dict()

    return df, metadata
