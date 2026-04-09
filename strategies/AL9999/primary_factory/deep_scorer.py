"""Deep scorer for Primary Model Factory - Uniqueness, Turnover, Regime, OOS."""

import numpy as np
import pandas as pd
from afmlkit.sampling import avg_uniqueness_of_sample, cusum_filter

from strategies.AL9999.config import COMMISSION_RATE, EXIT_CONFIG, SLIPPAGE_POINTS


def compute_deep_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combo: pd.Series,
    pt_sl: float = 1.0,
    test_ratio: float = 0.30,
    trend_labels_dict: dict | None = None,
) -> dict:
    """
    Compute deep metrics for Top-N candidate combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Default trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combo: Single combo Series.
    :param pt_sl: TBM pt/sl in sigma.
    :param test_ratio: Fraction for OOS test split.
    :param trend_labels_dict: Optional mapping {cusum_rate: trend_labels_df}.
    :returns: dict with keys: [combo_id, uniqueness, turnover, regime_stability,
                               oos_recall, oos_unreliable, low_info].
    """
    combo_id = combo['combo_id']
    base_combo_id = combo.get('base_combo_id', combo_id)
    cusum_rate = combo['cusum_rate']
    fast = int(combo['fast'])
    slow = int(combo['slow'])
    vertical_bars = int(combo['vertical_bars'])

    active_trend_labels = trend_labels
    if trend_labels_dict is not None:
        active_trend_labels = trend_labels_dict.get(cusum_rate, trend_labels)

    k_row = k_lookup[k_lookup['rate'] == cusum_rate]
    if len(k_row) == 0:
        raise ValueError(f"No k value found for rate {cusum_rate}")
    k = k_row['k'].iloc[0]

    close = bars['close'].values
    log_returns = np.diff(np.log(close))
    event_indices = cusum_filter(log_returns, np.array([k]))

    bars_index = bars.index
    cusum_events_ts = bars_index[event_indices]

    fast_ma = bars['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars['close'].ewm(span=slow, adjust=False).mean()

    uniqueness, low_info = _compute_uniqueness(bars_index, event_indices, vertical_bars)
    turnover = _compute_turnover(bars_index, cusum_events_ts)
    avg_inter_event_time = _compute_avg_inter_event_time(cusum_events_ts)
    regime_stability = _compute_regime_stability(
        bars, active_trend_labels, cusum_events_ts, fast_ma, slow_ma
    )
    oos_recall, oos_unreliable = _compute_oos_recall(
        bars, active_trend_labels, k, fast, slow, test_ratio
    )
    trade_metrics = _compute_trade_metrics(
        bars=bars,
        k=k,
        fast=fast,
        slow=slow,
        test_ratio=test_ratio,
        exit_config=EXIT_CONFIG,
        commission_rate=COMMISSION_RATE,
        slippage_points=SLIPPAGE_POINTS,
    )

    return {
        'combo_id': combo_id,
        'base_combo_id': base_combo_id,
        'uniqueness': uniqueness,
        'turnover': turnover,
        'avg_inter_event_time': avg_inter_event_time,
        'regime_stability': regime_stability,
        'oos_recall': oos_recall,
        'oos_unreliable': oos_unreliable,
        'low_info': low_info,
        'net_pnl': trade_metrics['net_pnl'],
        'sharpe': trade_metrics['sharpe'],
        'mdd': trade_metrics['mdd'],
        'trade_count': trade_metrics['trade_count'],
    }


def _compute_uniqueness(
    bars_index: pd.DatetimeIndex,
    event_indices: np.ndarray,
    vertical_bars: int,
) -> tuple[float, bool]:
    """Compute average uniqueness of events."""
    if len(event_indices) == 0:
        return 0.0, True

    event_starts = event_indices.astype(np.int64)
    event_ends = np.minimum(event_starts + vertical_bars, len(bars_index) - 1).astype(np.int64)
    sampled_indices = np.arange(len(event_indices), dtype=np.int64)

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

    n_years = (bars_index[-1] - bars_index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    annual_candidates = len(cusum_events_ts) / n_years

    daily_events = cusum_events_ts.to_series().groupby(cusum_events_ts.date).size()
    avg_concurrency = daily_events.mean() if len(daily_events) > 0 else 1.0

    turnover = annual_candidates * avg_concurrency
    return turnover


def _compute_avg_inter_event_time(
    cusum_events_ts: pd.DatetimeIndex,
) -> float:
    """Compute average time between consecutive events in hours."""
    if len(cusum_events_ts) < 2:
        return 0.0

    sorted_events = cusum_events_ts.sort_values()
    time_diffs_ns = np.diff(sorted_events.astype(np.int64))
    if len(time_diffs_ns) == 0:
        return 0.0

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

    vol = bars['close'].pct_change().ewm(span=20).std()
    vol_median = vol.median()

    high_vol_mask = vol > vol_median
    low_vol_mask = vol <= vol_median

    dma_signal = pd.Series(0, index=bars.index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    trend_events = trend_labels[trend_labels['side'] != 0]

    recalls = []
    for regime_mask in [high_vol_mask, low_vol_mask]:
        regime_events = cusum_events_ts[regime_mask.loc[cusum_events_ts].fillna(False)]
        matched = regime_events.intersection(trend_events.index)
        if len(matched) == 0:
            continue

        tp = sum(
            1 for ts in matched
            if dma_signal.loc[ts] == trend_events.loc[ts, 'side']
        )
        regime_recall = tp / len(matched) if len(matched) > 0 else 0.0
        recalls.append(regime_recall)

    if len(recalls) < 2:
        return 0.0

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
    split_idx = int(len(bars) * (1 - test_ratio))

    bars_test = bars.iloc[split_idx:]
    trend_test = trend_labels[trend_labels.index >= bars_test.index[0]]

    if len(trend_test) < 10:
        return 0.0, True

    close_test = bars_test['close'].values
    log_returns_test = np.diff(np.log(close_test))
    event_indices_test = cusum_filter(log_returns_test, np.array([k]))

    if len(event_indices_test) == 0:
        return 0.0, False

    fast_ma = bars_test['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars_test['close'].ewm(span=slow, adjust=False).mean()
    dma_signal = pd.Series(0, index=bars_test.index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    cusum_events_test = bars_test.index[event_indices_test]
    trend_events_test = trend_test[trend_test['side'] != 0]
    matched = cusum_events_test.intersection(trend_events_test.index)

    if len(matched) == 0:
        return 0.0, False

    tp = sum(
        1 for ts in matched
        if dma_signal.loc[ts] == trend_events_test.loc[ts, 'side']
    )
    fn = len(trend_events_test) - len(matched)
    oos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return oos_recall, False


def _compute_trade_metrics(
    bars: pd.DataFrame,
    k: float,
    fast: int,
    slow: int,
    test_ratio: float,
    exit_config: dict,
    commission_rate: float,
    slippage_points: float,
) -> dict:
    """Compute minimal cost-adjusted trade metrics on the OOS segment."""
    split_idx = int(len(bars) * (1 - test_ratio))
    bars_test = bars.iloc[split_idx:].copy()
    if len(bars_test) < max(slow + 2, 10):
        return {'net_pnl': 0.0, 'sharpe': 0.0, 'mdd': 0.0, 'trade_count': 0}

    open_col = 'open' if 'open' in bars_test.columns else 'close'
    close = bars_test['close'].values
    log_returns = np.diff(np.log(close))
    event_indices = cusum_filter(log_returns, np.array([k]))
    if len(event_indices) == 0:
        return {'net_pnl': 0.0, 'sharpe': 0.0, 'mdd': 0.0, 'trade_count': 0}

    fast_ma = bars_test['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars_test['close'].ewm(span=slow, adjust=False).mean()
    dma_signal = pd.Series(-1, index=bars_test.index, dtype=np.int8)
    dma_signal[fast_ma > slow_ma] = 1

    event_timestamps = bars_test.index[event_indices]
    if len(event_timestamps) == 0:
        return {'net_pnl': 0.0, 'sharpe': 0.0, 'mdd': 0.0, 'trade_count': 0}

    exit_type = exit_config.get('type', 'reverse_signal')
    fixed_hold_bars = int(exit_config.get('fixed_hold_bars', 20))
    max_hold_bars = int(exit_config.get('time_max_hold_bars', 60))

    trades = []
    position = 0
    entry_loc = None
    entry_price = None

    for event_no, ts in enumerate(event_timestamps):
        loc = bars_test.index.get_loc(ts)
        if isinstance(loc, slice):
            loc = loc.start
        loc = int(loc)
        side = int(dma_signal.loc[ts])
        close_price = float(bars_test.iloc[loc]['close'])

        if position != 0 and entry_loc is not None and entry_price is not None:
            exit_triggered = False
            exit_price = None

            if exit_type == 'reverse_signal' and side * position < 0:
                exit_triggered = True
                exit_price = float(bars_test.iloc[loc][open_col])
            elif exit_type == 'fixed_hold' and (loc - entry_loc) >= fixed_hold_bars:
                exit_triggered = True
                exit_price = float(bars_test.iloc[loc][open_col])
            elif exit_type == 'time_based' and (loc - entry_loc) >= max_hold_bars:
                exit_triggered = True
                exit_price = float(bars_test.iloc[loc][open_col])

            if event_no == len(event_timestamps) - 1:
                exit_triggered = True
                exit_price = close_price

            if exit_triggered and exit_price is not None:
                gross_pnl = (exit_price - entry_price) * position
                cost = (entry_price + exit_price) * commission_rate + slippage_points * 2.0
                trades.append(gross_pnl - cost)
                position = 0
                entry_loc = None
                entry_price = None

        if position == 0 and side != 0:
            position = side
            entry_loc = loc
            entry_price = float(bars_test.iloc[loc][open_col])

    if not trades:
        return {'net_pnl': 0.0, 'sharpe': 0.0, 'mdd': 0.0, 'trade_count': 0}

    pnl_series = pd.Series(trades, dtype=np.float64)
    net_pnl = float(pnl_series.sum())
    pnl_std = float(pnl_series.std(ddof=0))
    sharpe = float(pnl_series.mean() / pnl_std) if pnl_std > 0 else 0.0
    cum_pnl = pnl_series.cumsum()
    mdd = float((cum_pnl - cum_pnl.cummax()).min()) if len(cum_pnl) > 0 else 0.0

    return {
        'net_pnl': net_pnl,
        'sharpe': sharpe,
        'mdd': mdd,
        'trade_count': int(len(pnl_series)),
    }


def compute_all_deep_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combos: pd.DataFrame,
    pt_sl: float = 1.0,
    test_ratio: float = 0.30,
    trend_labels_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Compute deep metrics for multiple combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Default trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combos: DataFrame of Top-N combos.
    :param pt_sl: TBM pt/sl in sigma.
    :param test_ratio: Fraction for OOS test split.
    :param trend_labels_dict: Optional mapping {cusum_rate: trend_labels_df}.
    :returns: DataFrame with deep metrics for each combo.
    """
    results = []
    for _, combo in combos.iterrows():
        metrics = compute_deep_metrics(
            bars,
            trend_labels,
            k_lookup,
            combo,
            pt_sl,
            test_ratio,
            trend_labels_dict=trend_labels_dict,
        )
        results.append(metrics)

    return pd.DataFrame(results)
