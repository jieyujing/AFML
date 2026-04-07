"""Deep scorer for Primary Model Factory - Uniqueness, Turnover, Regime, OOS."""

import numpy as np
import pandas as pd
from afmlkit.sampling import cusum_filter, avg_uniqueness_of_sample


def compute_deep_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combo: pd.Series,
    pt_sl: float = 1.0,
    test_ratio: float = 0.30,
) -> dict:
    """
    Compute deep metrics for Top-N candidate combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combo: Single combo Series.
    :param pt_sl: TBM pt/sl in sigma.
    :param test_ratio: Fraction for OOS test split.
    :returns: dict with keys: [combo_id, uniqueness, turnover, regime_stability,
                               oos_recall, oos_unreliable, low_info].
    """
    combo_id = combo['combo_id']
    cusum_rate = combo['cusum_rate']
    fast = int(combo['fast'])
    slow = int(combo['slow'])
    vertical_bars = int(combo['vertical_bars'])

    # Get k value
    k_row = k_lookup[k_lookup['rate'] == cusum_rate]
    if len(k_row) == 0:
        raise ValueError(f"No k value found for rate {cusum_rate}")
    k = k_row['k'].iloc[0]

    # Compute log returns and CUSUM events
    close = bars['close'].values
    log_returns = np.diff(np.log(close))
    event_indices = cusum_filter(log_returns, np.array([k]))

    # Convert to timestamps
    bars_index = bars.index
    cusum_events_ts = bars_index[event_indices]

    # Compute DMA signals
    fast_ma = bars['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars['close'].ewm(span=slow, adjust=False).mean()

    # ============ 1. Uniqueness ============
    uniqueness, low_info = _compute_uniqueness(bars_index, event_indices, vertical_bars)

    # ============ 2. Turnover ============
    turnover = _compute_turnover(bars_index, cusum_events_ts)

    # ============ 2b. Avg Inter-Event Time ============
    avg_inter_event_time = _compute_avg_inter_event_time(cusum_events_ts)

    # ============ 3. Regime Stability ============
    regime_stability = _compute_regime_stability(
        bars, trend_labels, cusum_events_ts, fast_ma, slow_ma
    )

    # ============ 4. OOS Recall ============
    oos_recall, oos_unreliable = _compute_oos_recall(
        bars, trend_labels, k, fast, slow, test_ratio
    )

    return {
        'combo_id': combo_id,
        'uniqueness': uniqueness,
        'turnover': turnover,
        'avg_inter_event_time': avg_inter_event_time,
        'regime_stability': regime_stability,
        'oos_recall': oos_recall,
        'oos_unreliable': oos_unreliable,
        'low_info': low_info,
    }


def _compute_uniqueness(
    bars_index: pd.DatetimeIndex,
    event_indices: np.ndarray,
    vertical_bars: int,
) -> tuple[float, bool]:
    """Compute average uniqueness of events.

    Uniqueness measures how independent events are from each other.
    When events overlap (same vertical window), uniqueness is low.

    :param bars_index: Bars datetime index.
    :param event_indices: CUSUM event positions in bars array.
    :param vertical_bars: TBM vertical barrier bar count.
    :returns: (avg_uniqueness, low_info_flag)
    """
    if len(event_indices) == 0:
        return 0.0, True  # low_info = True

    # event_starts: position where each event starts on the bars time grid
    event_starts = event_indices.astype(np.int64)

    # event_ends: position where each event ends (start + vertical_bars)
    # Clip to bars length to avoid out-of-bounds
    event_ends = np.minimum(event_starts + vertical_bars, len(bars_index) - 1).astype(np.int64)

    # sampled_indices: all events are "sampled" for uniqueness calculation
    sampled_indices = np.arange(len(event_indices), dtype=np.int64)

    # Use avg_uniqueness_of_sample to compute average uniqueness
    try:
        avg_u = avg_uniqueness_of_sample(
            event_starts=event_starts,
            event_ends=event_ends,
            sampled_indices=sampled_indices,
        )
        uniqueness = float(avg_u) if np.isfinite(avg_u) else 0.0
        low_info = uniqueness == 0
    except Exception:
        uniqueness = 0.0
        low_info = True

    return uniqueness, low_info


def _compute_turnover(
    bars_index: pd.DatetimeIndex,
    cusum_events_ts: pd.DatetimeIndex,
) -> float:
    """Compute turnover burden (annual candidate count × concurrency)."""
    if len(cusum_events_ts) == 0:
        return 0.0

    # Annual candidate count
    n_years = (bars_index[-1] - bars_index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    annual_candidates = len(cusum_events_ts) / n_years

    # Event concurrency (average events per day)
    daily_events = cusum_events_ts.to_series().groupby(cusum_events_ts.date).size()
    avg_concurrency = daily_events.mean() if len(daily_events) > 0 else 1.0

    # Turnover = annual_candidates × avg_concurrency (normalized)
    # Higher = more burden
    turnover = annual_candidates * avg_concurrency

    return turnover


def _compute_avg_inter_event_time(
    cusum_events_ts: pd.DatetimeIndex,
) -> float:
    """
    Compute average time between consecutive events.

    Important for understanding signal frequency:
    - Low avg_inter_event_time: signals cluster together (may indicate noisy periods)
    - High avg_inter_event_time: signals spread out (more independent)

    :param cusum_events_ts: CUSUM event timestamps.
    :returns: Average inter-event time in hours.
    """
    if len(cusum_events_ts) < 2:
        return 0.0

    # Sort events by time
    sorted_events = cusum_events_ts.sort_values()

    # Compute time differences in nanoseconds (int64)
    time_diffs_ns = np.diff(sorted_events.astype(np.int64))

    if len(time_diffs_ns) == 0:
        return 0.0

    # Return average in hours (ns → hours = / 1e9 / 3600)
    avg_ns = float(np.mean(time_diffs_ns))
    avg_hours = avg_ns / 1e9 / 3600.0

    return avg_hours


def _compute_regime_stability(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    cusum_events_ts: pd.DatetimeIndex,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
) -> float:
    """Compute recall stability across volatility regimes."""
    if len(cusum_events_ts) == 0:
        return 0.0

    # Compute rolling volatility
    vol = bars['close'].pct_change().ewm(span=20).std()
    vol_median = vol.median()

    # Split into high/low volatility regimes
    high_vol_mask = vol > vol_median
    low_vol_mask = vol <= vol_median

    # Compute DMA signals
    dma_signal = pd.Series(0, index=bars.index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    # Filter trend events
    trend_events = trend_labels[trend_labels['side'] != 0]

    recalls = []
    for regime_mask in [high_vol_mask, low_vol_mask]:
        # Get events in this regime
        regime_events = cusum_events_ts[regime_mask.loc[cusum_events_ts].fillna(False)]

        # Match with trend events
        matched = regime_events.intersection(trend_events.index)
        if len(matched) == 0:
            continue

        # Compute recall for this regime
        tp = sum(
            1 for ts in matched
            if dma_signal.loc[ts] == trend_events.loc[ts, 'side']
        )
        regime_recall = tp / len(matched) if len(matched) > 0 else 0.0
        recalls.append(regime_recall)

    if len(recalls) < 2:
        return 0.0

    # Stability = 1 - std(recall_per_regime) (higher = more stable)
    stability = 1.0 - np.std(recalls)
    return max(0.0, min(1.0, stability))


def _compute_oos_recall(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k: float,
    fast: int,
    slow: int,
    test_ratio: float,
) -> tuple[float, bool]:
    """Compute OOS recall using time-based split."""
    # Time-based split: last test_ratio as OOS
    split_idx = int(len(bars) * (1 - test_ratio))

    bars_test = bars.iloc[split_idx:]
    trend_test = trend_labels[trend_labels.index >= bars_test.index[0]]

    if len(trend_test) < 10:
        return 0.0, True  # oos_unreliable

    # Compute CUSUM events on test data
    close_test = bars_test['close'].values
    log_returns_test = np.diff(np.log(close_test))
    event_indices_test = cusum_filter(log_returns_test, np.array([k]))

    if len(event_indices_test) == 0:
        return 0.0, False

    # Compute DMA on test data
    fast_ma = bars_test['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars_test['close'].ewm(span=slow, adjust=False).mean()
    dma_signal = pd.Series(0, index=bars_test.index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    # Get CUSUM event timestamps
    cusum_events_test = bars_test.index[event_indices_test]

    # Match with trend events
    trend_events_test = trend_test[trend_test['side'] != 0]
    matched = cusum_events_test.intersection(trend_events_test.index)

    if len(matched) == 0:
        return 0.0, False

    # Compute OOS recall
    tp = sum(
        1 for ts in matched
        if dma_signal.loc[ts] == trend_events_test.loc[ts, 'side']
    )
    fn = len(trend_events_test) - len(matched)
    oos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return oos_recall, False


def compute_all_deep_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combos: pd.DataFrame,
    pt_sl: float = 1.0,
    test_ratio: float = 0.30,
) -> pd.DataFrame:
    """
    Compute deep metrics for multiple combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combos: DataFrame of Top-N combos.
    :param pt_sl: TBM pt/sl in sigma.
    :param test_ratio: Fraction for OOS test split.
    :returns: DataFrame with deep metrics for each combo.
    """
    results = []
    for _, combo in combos.iterrows():
        metrics = compute_deep_metrics(bars, trend_labels, k_lookup, combo, pt_sl, test_ratio)
        results.append(metrics)

    return pd.DataFrame(results)