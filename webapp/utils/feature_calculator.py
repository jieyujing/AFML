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


def align_features_with_cusum(
    features_df: pd.DataFrame,
    cusum_data: Optional[pd.DataFrame] = None,
    cusum_path: Optional[str] = None,
    label_cols: List[str] = None
) -> pd.DataFrame:
    """将特征与 CUSUM 采样数据对齐

    Args:
        features_df: 特征 DataFrame
        cusum_data: CUSUM 采样数据 DataFrame（优先使用）
        cusum_path: CUSUM 采样数据文件路径（fallback）
        label_cols: 需要保留的标签列

    Returns:
        对齐后的 DataFrame
    """
    # 优先使用 DataFrame 输入，fallback 到文件路径
    if cusum_data is not None:
        labels_df = cusum_data.copy()
    elif cusum_path is not None:
        labels_df = pd.read_csv(cusum_path, index_col=0, parse_dates=True)
    else:
        raise ValueError("必须提供 cusum_data 或 cusum_path")

    labels_df = labels_df.sort_index()

    if label_cols is None:
        label_cols = ["bin", "t1", "avg_uniqueness", "return_attribution"]
        label_cols = [c for c in label_cols if c in labels_df.columns]

    # Try exact index intersection first
    common_idx = features_df.index.intersection(labels_df.index)

    # If no exact match, try date-based matching (truncate time component)
    if len(common_idx) == 0:
        # Create date-only index for matching
        features_dates = features_df.index.normalize()
        labels_dates = labels_df.index.normalize()

        # Find matching dates
        common_dates = features_dates.intersection(labels_dates)

        if len(common_dates) > 0:
            # Map back to original indices
            features_mask = features_dates.isin(common_dates)
            labels_mask = labels_dates.isin(common_dates)

            features_filtered = features_df[features_mask]
            labels_filtered = labels_df[labels_mask]

            # Use merge_asof for time-based alignment
            features_reset = features_filtered.reset_index()
            labels_reset = labels_filtered.reset_index()

            # Rename timestamp column for merge_asof
            features_reset = features_reset.rename(columns={'index': 'timestamp'})
            labels_reset = labels_reset.rename(columns={'index': 'timestamp'})

            # Merge on closest timestamp (backward direction: use CUSUM labels for previous event)
            aligned = pd.merge_asof(
                features_reset.sort_values('timestamp'),
                labels_reset.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1D')
            )

            aligned = aligned.set_index('timestamp')
            return aligned
        else:
            # No common dates at all - return empty DataFrame with warning
            return features_df.iloc[:0].copy()

    # Original exact match logic
    aligned_features = features_df.loc[common_idx].copy()
    aligned = aligned_features.join(labels_df[label_cols], how="inner")

    return aligned


def purge_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values."""
    return df.dropna()


def compute_all_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    cusum_data: Optional[pd.DataFrame] = None,
    cusum_path: Optional[str] = None,
    align_to_cusum: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """计算所有特征

    Args:
        df: 输入数据 DataFrame
        config: 特征配置
        cusum_data: CUSUM 采样数据 DataFrame（优先使用）
        cusum_path: CUSUM 采样数据文件路径（fallback，向后兼容）
        align_to_cusum: 是否对齐到 CUSUM 数据

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

    if align_to_cusum:
        if cusum_data is not None or cusum_path is not None:
            df = align_features_with_cusum(df, cusum_data=cusum_data, cusum_path=cusum_path)
            metadata["aligned_to_cusum"] = True
        else:
            metadata["aligned_to_cusum"] = False
    else:
        metadata["aligned_to_cusum"] = False

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
