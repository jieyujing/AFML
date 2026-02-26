"""
Feature Engineering Pipeline — AFML Workflow Stage
===================================================

Implements the full AFML feature engineering pipeline:
  1. Load continuous Dollar Bars
  2. Compute features on the continuous series:
     - Multi-window EWM volatility (spans 10, 50, 100)
     - Momentum / structural metrics (log returns, EMA crossover distances, RSI)
     - Fractional Differentiation (FFD) with auto-optimised d
  3. Align continuous features to discrete CUSUM event timestamps
  4. Join with TBM labels / sample weights
  5. Purge NaN rows & save pristine feature matrix

Usage
-----
    uv run python scripts/feature_engineering.py
"""

import numpy as np
import pandas as pd

# ── afmlkit core functions ────────────────────────────────────────────
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.volatility import (
    ewms, parkinson_range, atr, bollinger_percent_b, variance_ratio_1_4_core
)
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.momentum import rsi_wilder, roc, stoch_k
from afmlkit.feature.core.correlation import rolling_price_volume_correlation

# ── Paths ─────────────────────────────────────────────────────────────
DOLLAR_BARS_PATH = "outputs/dollar_bars/dollar_bars_freq20.csv"
CUSUM_SAMPLED_PATH = "outputs/dollar_bars/cusum_sampled_bars.csv"
OUTPUT_PATH = "outputs/dollar_bars/feature_matrix.csv"

# ── Hyper-parameters ──────────────────────────────────────────────────
VOL_SPANS = [10, 50, 100]          # EWM volatility windows
EMA_SHORT_SPAN = 12                # Short EMA for MACD-like crossover
EMA_LONG_SPAN = 26                 # Long EMA
RSI_WINDOW = 14                    # RSI lookback
FRACDIFF_THRES = 1e-4              # FFD weight truncation threshold
FRACDIFF_D_STEP = 0.05             # Step size for d optimisation


def load_dollar_bars(path: str) -> pd.DataFrame:
    """Task 2.1 — Load continuous Dollar Bars and set DatetimeIndex."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"[1/6] Loaded continuous Dollar Bars: {len(df)} rows, "
          f"range {df.index[0]} → {df.index[-1]}")
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from close prices."""
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.2 — Multi-window EWM volatility features + range/structural metrics."""
    log_ret = df["log_return"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    # --- EWM Volatility ---
    for span in VOL_SPANS:
        col_name = f"vol_ewm_{span}"
        df[col_name] = ewms(log_ret, span)

    # --- Range-based Volatility (Parkinson) ---
    df["vol_parkinson"] = parkinson_range(high, low)

    # --- ATR (Average True Range) ---
    df["vol_atr_14"] = atr(high, low, close, window=14)

    # --- Bollinger %B ---
    df["vol_bb_pct_b_20"] = bollinger_percent_b(close, window=20, num_std=2.0)

    # --- Variance Ratio (detecting trend vs noise) ---
    # window=20, ddof=1, log returns
    df["trend_variance_ratio_20"] = variance_ratio_1_4_core(close, window=20, ddof=1, ret_type="log")

    print(f"[2/6] Volatility & Structural features computed: "
          + ", ".join([f"vol_ewm_{s}" for s in VOL_SPANS]
                     + ["vol_parkinson", "vol_atr_14", "vol_bb_pct_b_20", "trend_variance_ratio_20"]))
    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.3 — Momentum and structural metric features."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    log_close = np.log(close)

    # --- EMA short / long ---
    ema_short = ewma(close, EMA_SHORT_SPAN)
    ema_long = ewma(close, EMA_LONG_SPAN)
    df["ema_short"] = ema_short
    df["ema_long"] = ema_long

    # --- Log price distance to EMAs ---
    df["log_dist_ema_short"] = log_close - np.log(ema_short)
    df["log_dist_ema_long"] = log_close - np.log(ema_long)

    # --- MACD-like: difference of short & long EMA ---
    df["ema_diff"] = ema_short - ema_long

    # --- RSI (Wilder) ---
    df["rsi_14"] = rsi_wilder(close, RSI_WINDOW)

    # --- Roc (Rate of Change) ---
    df["mom_roc_10"] = roc(close, period=10)

    # --- Stochastic %K ---
    df["mom_stoch_k_14"] = stoch_k(close, low, high, length=14)

    # --- Price-Volume Correlation ---
    df["corr_pv_10"] = rolling_price_volume_correlation(close, volume, window=10)

    # --- Amihud Illiquidity ( |Return| / Volume ) ---
    # We use log_return for scale consistency
    df["liq_amihud"] = np.abs(df["log_return"]) / (df["volume"] * df["close"]) * 1e6

    # --- Relative Volume (Volume / SMA(Volume, 20)) ---
    vol_sma_20 = sma(volume, window=20)
    df["vol_rel_20"] = volume / vol_sma_20

    print(f"[3/6] Momentum & Liquidity features computed: ema_short, ema_long, "
          "log_dist_ema_short, log_dist_ema_long, ema_diff, rsi_14, mom_roc_10, "
          "mom_stoch_k_14, corr_pv_10, liq_amihud, vol_rel_20")
    return df


def compute_fracdiff_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Task 1.1 + 1.2 + 2.4 — Fractional Differentiation (FFD)
    with auto-optimised d on the log price series.
    """
    log_price = np.log(df["close"])

    # --- Optimise d via ADF test ---
    optimal_d = optimize_d(log_price, thres=FRACDIFF_THRES, d_step=FRACDIFF_D_STEP)
    print(f"[4/6] Optimal fractional differencing order d = {optimal_d}")

    # --- Apply FFD ---
    ffd_series = frac_diff_ffd(log_price, d=optimal_d, thres=FRACDIFF_THRES)
    ffd_series.name = "ffd_log_price"

    # Verify FFD — the output should be shorter by (width - 1) rows
    n_dropped = len(df) - len(ffd_series)
    print(f"     FFD window width dropped {n_dropped} initial rows "
          f"(FFD series length: {len(ffd_series)})")

    # Merge back — NaN for early rows that lack enough history
    df["ffd_log_price"] = ffd_series
    return df, optimal_d


def align_with_labels(
    features_df: pd.DataFrame,
    labels_path: str,
) -> pd.DataFrame:
    """
    Tasks 3.1 – 3.3 — Load CUSUM labels/weights, align features
    using exact DatetimeIndex, and join.
    """
    # Task 3.1 — Load labels + weights
    labels_df = pd.read_csv(labels_path, parse_dates=["timestamp"])
    labels_df = labels_df.set_index("timestamp").sort_index()
    print(f"[5/6] Loaded CUSUM sampled labels: {len(labels_df)} events")

    # Task 3.2 — Strict DatetimeIndex alignment
    # Slice the continuous features at exactly the CUSUM event timestamps.
    common_idx = features_df.index.intersection(labels_df.index)
    aligned_features = features_df.loc[common_idx].copy()

    # Task 3.3 — Join labels & weights columns
    label_cols = ["bin", "t1", "avg_uniqueness", "return_attribution"]
    # Only keep columns that actually exist in labels_df
    label_cols = [c for c in label_cols if c in labels_df.columns]
    aligned = aligned_features.join(labels_df[label_cols], how="inner")

    print(f"     Aligned features-to-labels: {len(aligned)} rows "
          f"(label columns: {label_cols})")
    return aligned


def purge_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Task 3.4 — Drop rows with NaN feature values from window init."""
    n_before = len(df)
    df_clean = df.dropna()
    n_dropped = n_before - len(df_clean)
    print(f"     NaN purge: dropped {n_dropped} rows → {len(df_clean)} clean rows remain")
    return df_clean


def main():
    print("=" * 70)
    print(" AFML Feature Engineering Pipeline")
    print("=" * 70)

    # ── Stage 1: Load & compute continuous features ───────────────────
    df = load_dollar_bars(DOLLAR_BARS_PATH)
    df = compute_log_returns(df)
    df = compute_volatility_features(df)
    df = compute_momentum_features(df)
    df, optimal_d = compute_fracdiff_features(df)

    # ── Stage 2: Align with CUSUM labels & purge ──────────────────────
    aligned = align_with_labels(df, CUSUM_SAMPLED_PATH)
    clean = purge_nan_rows(aligned)

    # ── Stage 3: Summary & save ───────────────────────────────────────
    # Task 4.1 — Log final summary
    print("\n" + "=" * 70)
    print(f" SUMMARY")
    print(f"   Optimal FFD d  : {optimal_d}")
    print(f"   Final shape    : {clean.shape}")
    print(f"   Feature columns: {[c for c in clean.columns]}")
    print(f"   Label dist     :")
    if "bin" in clean.columns:
        vc = clean["bin"].value_counts().sort_index()
        for label, count in vc.items():
            print(f"     bin={int(label):+d} : {count}  ({count/len(clean)*100:.1f}%)")
    print("=" * 70)

    # Task 4.2 — Save feature matrix
    clean.to_csv(OUTPUT_PATH)
    print(f"\n✓ Saved feature matrix → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
