"""
Triple Barrier Labeling for Financial Machine Learning

This module implements the Triple Barrier Method from 'Advances in Financial Machine Learning'
by Marcos López de Prado. The method generates labels by setting three barriers:
1. Upper barrier (profit-taking)
2. Lower barrier (stop-loss)
3. Vertical barrier (time limit)

The label is determined by which barrier is touched first.
"""

import pandas as pd
import numpy as np
import subprocess
import sys
from typing import Optional, Union

from afml import TripleBarrierLabeler


def get_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Calculate volatility using exponential weighted moving average of log returns.

    This is used to set dynamic barrier widths that adapt to market conditions.
    When applied to dollar bars, this represents volatility in intrinsic time.

    Args:
        close: Series of prices (time, dollar, or volume bars)
        span: Span for EMA calculation (default: 100)

    Returns:
        Series of volatility estimates
    """
    # Calculate log returns
    returns = np.log(close / close.shift(1))

    # Calculate EMA of absolute returns as volatility proxy
    volatility = returns.ewm(span=span).std()

    return volatility


def get_cusum_events(
    close: pd.Series, threshold: Union[float, pd.Series]
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM Filter (Snippet 2.4).

    The CUSUM filter is a quality-control method, designed to detect a shift in the
    mean value of a measured quantity away from a target value. It is used to
    downsample a time series to only those events where a significant shift has occurred.

    Args:
        close: Series of close prices
        threshold: Threshold for event detection. Can be fixed float or dynamic Series (volatility).
                   If dynamic, it should be the estimated daily volatility.

    Returns:
        DatetimeIndex of detected events
    """
    t_events = []
    s_pos = 0
    s_neg = 0

    # Calculate log returns to match volatility units
    diff = np.log(close / close.shift(1))

    # Get timestamps
    times = diff.index

    # Handle dynamic threshold
    if isinstance(threshold, pd.Series):
        # Align threshold to timestamps (reindex to ensure we have values)
        thresh = threshold.reindex(times).ffill()
    else:
        thresh = pd.Series(threshold, index=times)

    # Iterate through returns
    # Note: iterating rows in pandas is slow, but this is the standard implementation reference
    for i in range(1, len(times)):
        t = times[i]
        ret = diff.iloc[i]
        h = thresh.iloc[i]

        # Skip if invalid data
        if pd.isna(ret) or pd.isna(h):
            continue

        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)

        if s_neg < -h:
            s_neg = 0
            t_events.append(t)
        elif s_pos > h:
            s_pos = 0
            t_events.append(t)

    return pd.DatetimeIndex(t_events)


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: list,
    molecule: Optional[pd.DatetimeIndex] = None,
    t1: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None,
    target: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Apply the Triple Barrier Method to generate labels.

    This is the core labeling function. For each event (bar), it sets three barriers:
    - Upper barrier at close * (1 + pt_sl[0])
    - Lower barrier at close * (1 - pt_sl[1])
    - Vertical barrier at time t1

    Args:
        close: Series of close prices with datetime index
        events: DatetimeIndex of events to label (typically all bar timestamps)
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
               e.g., [2, 2] means symmetric 2-sigma barriers
        molecule: Subset of events to process (for parallel processing)
        t1: Series with vertical barrier timestamps (max holding period)
        side: Series with position side (1 for long, -1 for short, 0 for both)
        target: Series of target values (volatility) for barrier width. If None, it will be calculated.

    Returns:
        DataFrame with columns:
        - t1: timestamp when barrier was touched
        - trgt: target (barrier width)
        - side: position side
        - ret: return at barrier touch
        - label: {-1, 0, 1} indicating which barrier was hit first
    """
    # 1. Prepare events
    if molecule is None:
        molecule = events

    # Filter events to process
    events_ = events.intersection(molecule)

    # 2. Get target (volatility-based barrier width)
    if target is None:
        trgt = get_volatility(close)
    else:
        trgt = target
    trgt = trgt.reindex(events_, method="ffill")

    # 3. Set vertical barrier (time limit)
    if t1 is None:
        # Default: 1 day holding period
        t1 = pd.Series(pd.NaT, index=events_)
        for idx in events_:
            # Find next day's first bar
            future_dates = close.index[close.index > idx]
            if len(future_dates) > 0:
                # Set to next day (or max 1 day ahead)
                next_day = idx + pd.Timedelta(days=1)
                valid_dates = future_dates[future_dates <= next_day]
                if len(valid_dates) > 0:
                    t1.loc[idx] = valid_dates[-1]
                else:
                    t1.loc[idx] = future_dates[0]

    # 4. Set side (position direction)
    if side is None:
        # Default: both sides (long and short)
        side_ = pd.Series(1.0, index=trgt.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(trgt.index, method="ffill")
        pt_sl_ = pt_sl[:2]

    # 5. Apply barriers
    out = pd.DataFrame(index=events_)
    out["t1"] = t1
    out["trgt"] = trgt
    out["side"] = side_

    # For each event, find which barrier is hit first
    for loc in events_:
        # Get the barrier timestamps
        df0 = close[loc:]  # Path from event onwards

        if pd.isna(out.loc[loc, "t1"]):
            # If no vertical barrier, use end of data
            df0 = df0[: close.index[-1]]
        else:
            # Clip to vertical barrier
            df0 = df0[: out.loc[loc, "t1"]]

        if len(df0) <= 1:
            # Not enough data to evaluate
            out.loc[loc, "ret"] = 0
            out.loc[loc, "label"] = 0
            continue

        # Calculate returns from entry point
        ret = (df0 / close[loc] - 1) * out.loc[loc, "side"]

        # Upper barrier (profit-taking)
        upper = pt_sl_[0] * out.loc[loc, "trgt"]
        # Lower barrier (stop-loss)
        lower = -pt_sl_[1] * out.loc[loc, "trgt"]

        # Find first touch
        touch_upper = ret[ret >= upper]
        touch_lower = ret[ret <= lower]

        if len(touch_upper) > 0 and len(touch_lower) > 0:
            # Both barriers touched - use earliest
            if touch_upper.index[0] < touch_lower.index[0]:
                out.loc[loc, "t1"] = touch_upper.index[0]
                out.loc[loc, "ret"] = ret.loc[touch_upper.index[0]]
                out.loc[loc, "label"] = 1  # Profit
            else:
                out.loc[loc, "t1"] = touch_lower.index[0]
                out.loc[loc, "ret"] = ret.loc[touch_lower.index[0]]
                out.loc[loc, "label"] = -1  # Loss
        elif len(touch_upper) > 0:
            # Only upper barrier touched
            out.loc[loc, "t1"] = touch_upper.index[0]
            out.loc[loc, "ret"] = ret.loc[touch_upper.index[0]]
            out.loc[loc, "label"] = 1
        elif len(touch_lower) > 0:
            # Only lower barrier touched
            out.loc[loc, "t1"] = touch_lower.index[0]
            out.loc[loc, "ret"] = ret.loc[touch_lower.index[0]]
            out.loc[loc, "label"] = -1
        else:
            # Vertical barrier hit (timeout)
            out.loc[loc, "ret"] = ret.iloc[-1]
            out.loc[loc, "label"] = np.sign(ret.iloc[-1])

    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list,
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_days: Optional[int] = None,
    vertical_barrier_bars: Optional[int] = None,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    High-level function to generate labeled events using Triple Barrier Method.

    This is the main entry point for labeling. It combines all steps:
    1. Set vertical barriers (either by days or by bar count)
    2. Apply triple barrier method
    3. Filter events by minimum return threshold

    Args:
        close: Series of close prices with datetime index
        t_events: DatetimeIndex of events to label
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
        target: Series of target values (volatility) for barrier width
        min_ret: Minimum return threshold to keep event (filter noise)
        num_threads: Number of threads for parallel processing (not implemented yet)
        vertical_barrier_days: Number of days for vertical barrier (Time-based)
        vertical_barrier_bars: Number of bars for vertical barrier (Intrinsic-time based)
        side: Series indicating position side (1=long, -1=short, None=both)

    Returns:
        DataFrame with labeled events
    """
    # 1. Get vertical barrier timestamps (t1)
    if vertical_barrier_bars is not None:
        # Intrinsic-time barrier: Look ahead N bars
        indices = close.index.get_indexer(t_events)
        t1_indices = indices + vertical_barrier_bars
        # Clip to data length
        t1_indices = np.clip(t1_indices, 0, len(close) - 1)
        t1 = pd.Series(close.index[t1_indices], index=t_events)
    elif vertical_barrier_days is not None:
        # Standard time barrier: Look ahead N days
        t1 = close.index.searchsorted(
            t_events + pd.Timedelta(days=vertical_barrier_days)
        )
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=t_events[: t1.shape[0]])
    else:
        t1 = None

    # 2. Apply triple barrier
    events = apply_triple_barrier(
        close=close,
        events=t_events,
        pt_sl=pt_sl,
        t1=t1,
        side=side,
        target=target,
    )

    # 3. Filter by minimum return
    if min_ret > 0:
        events = events[events["ret"].abs() >= min_ret]

    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Convert continuous returns to discrete bins (labels).

    This function takes the raw events from triple barrier and creates
    final labels suitable for classification:
    - 1: Positive return (profit)
    - 0: Near-zero return (neutral)
    - -1: Negative return (loss)

    Args:
        events: DataFrame from apply_triple_barrier
        close: Series of close prices

    Returns:
        DataFrame with bin labels
    """
    # 1. Align prices
    events_ = events.dropna(subset=["t1"])

    # 2. Create bins
    bins = events_[["ret", "label"]].copy()

    # 3. Optional: Adjust labels based on return magnitude
    # For now, we use the label from barrier touch directly
    bins["bin"] = events_["label"]

    return bins


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Drop rare labels to avoid training on noise.

    In financial data, some labels may be very rare (e.g., extreme events).
    This function removes labels that appear less than min_pct of the time.

    Args:
        events: DataFrame with labels
        min_pct: Minimum percentage threshold (default: 5%)

    Returns:
        Filtered DataFrame
    """
    # Count label frequencies
    label_counts = events["label"].value_counts(normalize=True)

    # Find rare labels
    rare_labels = label_counts[label_counts < min_pct].index

    # Filter
    if len(rare_labels) > 0:
        print(f"Dropping rare labels: {list(rare_labels)} (< {min_pct * 100}%)")
        events = events[~events["label"].isin(rare_labels)]

    return events


def main():
    """
    Example usage of Triple Barrier Labeling on dollar bars using OO approach.
    """
    print("=" * 80)
    print("Triple Barrier Labeling - AFML (OO Implementation)")
    print("=" * 80)

    # 1. Load dollar bars
    print("\n1. Loading dollar bars...")
    try:
        df = pd.read_csv("dynamic_dollar_bars.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(
            "Error: 'dynamic_dollar_bars.csv' not found. Please run process_bars.py first."
        )
        return

    print(f"   Loaded {len(df)} bars")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Using OO TripleBarrierLabeler
    print("\n2. Using TripleBarrierLabeler...")
    labeler = TripleBarrierLabeler(
        pt_sl=[1.0, 1.0], vertical_barrier_bars=12, min_ret=0.001, volatility_span=100
    )

    # Fit labeler with close prices
    print("   Fitting volatility model...")
    labeler.fit(df["close"])

    # Generate CUSUM events
    cusum_vol_multiplier = 2
    print(f"   Generating CUSUM events (k={cusum_vol_multiplier})...")
    cusum_events = labeler.get_cusum_events(
        df["close"], threshold=labeler.volatility_ * cusum_vol_multiplier
    )
    print(f"   Selected {len(cusum_events)} events from {len(df)} bars")
    print(f"   Sampling ratio: {len(cusum_events) / len(df) * 100:.2f}%")

    # Apply triple barrier
    print("\n3. Applying Triple Barrier...")
    events = labeler.label(
        close=df["close"],
        events=cusum_events,
    )
    print(f"   Generated {len(events)} labeled events")

    # 6. Analyze labels
    print("\n4. Label distribution:")
    label_dist = events["label"].value_counts(normalize=True).sort_index()
    for label, pct in label_dist.items():
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        print(f"   {label_name:>8} ({label:>2}): {pct * 100:>5.2f}%")

    # 7. Statistics
    print("\n5. Return statistics:")
    if not events.empty:
        print(f"   Mean return: {events['ret'].mean() * 100:.3f}%")
        print(f"   Std return: {events['ret'].std() * 100:.3f}%")
        print(f"   Min return: {events['ret'].min() * 100:.3f}%")
        print(f"   Max return: {events['ret'].max() * 100:.3f}%")
    else:
        print("   No events labeled.")

    # 8. Save labeled data
    print("\n6. Saving labeled events...")
    output_file = "labeled_events.csv"
    events.to_csv(output_file)
    print(f"   ✓ Saved to: {output_file}")

    # 9. Create final dataset
    print("\n7. Creating final dataset...")
    df_labeled = df.loc[events.index].copy()
    df_labeled["label"] = events["label"]
    df_labeled["ret"] = events["ret"]
    df_labeled["barrier_time"] = events["t1"]
    df_labeled = df_labeled.dropna(subset=["label"])

    final_output = "dollar_bars_labeled.csv"
    df_labeled.to_csv(final_output)
    print(f"   ✓ Saved to: {final_output}")
    print(f"   ✓ Total labeled bars: {len(df_labeled)}")

    print("\n" + "=" * 80)
    print("✓ Triple Barrier Labeling Complete!")
    print("=" * 80)

    # 10. Auto-run visualization
    print("\n8. Running visualization...")
    try:
        subprocess.run([sys.executable, "src/visualize_labels.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"   Error running visualization: {e}")
    except FileNotFoundError:
        try:
            subprocess.run(
                ["uv", "run", "python", "src/visualize_labels.py"], check=True
            )
        except Exception as e:
            print(f"   Error running visualization via uv: {e}")

    print("\nNext steps:")
    print("  1. Feature engineering (technical indicators, microstructure)")
    print("  2. Sample weights (uniqueness, time decay)")
    print("  3. Cross-validation (purged k-fold)")
    print("  4. Model training")

    return events, df_labeled


if __name__ == "__main__":
    main()


#   1. 采样与过滤器 (CUSUM Filter)
#    * 采样率 18.77% (737 events / 3927 bars):
#        * 在 $2\sigma$ ($k=2$) 的阈值下，你过滤掉了 81% 的“噪音”Bar。
#        * 这是一个非常理想的比例。它意味着你只在价格发生显著偏移（$2\sigma$）时才进行
#          交易决策，这极大地提高了数据的信噪比。
#        * 结论: cusum_vol_multiplier = 2.0
#          是一个非常稳健且保守的设置，适合捕捉中短期趋势。

#   2. 标签分布 (Label Distribution)
#    * 均衡性:
#        * Loss: 51.97%
#        * Profit: 48.03%
#        * 几乎是 50/50 的分布。这对于金融数据来说是非常好的信号。它表明：
#            1. 你的 Barrier 设置是对称且公平的。
#            2. Dollar Bars
#               有效地去除了时间采样的微观噪音，使价格游走更接近随机游走（Random
#               Walk），符合有效市场假说。
#        * 机器学习启示:
#          你的数据集类别非常平衡，不需要进行过采样（Over-sampling）或欠采样（Under-sa
#          mpling），可以直接用于训练分类器（如 Random Forest, XGBoost）。

#   3. 回报统计 (Return Statistics)
#    * 均值: 0.007% (接近 0)。
#    * 中位数: -0.408%。
#    * 这再次印证了你在一个相对有效的市场上进行采样。如果均值显著不为0，说明存在简单的
#      动量或均值回归机会（或者标签存在前视偏差 Look-ahead Bias）。
#    * 获利能力:
#        * Profit Mean: 0.954%
#        * Loss Mean: -0.869%
#        * 盈利交易的平均收益略高于亏损交易的平均亏损。虽然差异不大，但在高频/中频交易
#          中，结合高胜率模型，这微小的 edge 是可以被放大的。

#   4. 持仓时间 (Holding Period)
#    * 平均: 28.19 小时。
#    * 最大: 647.25 小时（约 27 天）。
#    * Bar 视角: 你设置了 vertical_barrier_bars = 50。
#    * 这说明大部分交易（由于触碰止盈或止损）在远早于 50 个 Bar 的时限前就结束了。这是
#      Triple Barrier
#      的核心优势：路径依赖。它不会傻傻地等到时间结束，而是在价格触及边界时立即反应。
