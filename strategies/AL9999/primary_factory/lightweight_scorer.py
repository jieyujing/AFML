"""Lightweight scorer for Primary Model Factory."""

import numpy as np
import pandas as pd
from afmlkit.sampling import cusum_filter


def compute_lightweight_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combo: pd.Series,
    pt_sl: float = 1.0,
    trend_labels_dict: dict | None = None,
) -> dict:
    """
    Compute lightweight metrics for a single parameter combo.

    :param bars: Dollar bars DataFrame with 'close' column.
    :param trend_labels: Default trend scanning labels (index=timestamp, cols=[side, t_value, t1]).
    :param k_lookup: CUSUM rate→k mapping (cols=[rate, k]).
    :param combo: Single combo Series with [combo_id, cusum_rate, fast, slow, vertical_bars].
    :param pt_sl: TBM pt/sl in sigma units (default 1.0).
    :param trend_labels_dict: Optional mapping {cusum_rate: trend_labels_df}.
    :returns: dict with keys: [combo_id, recall, cpr, coverage, lift,
                               n_candidates, n_true_positives, n_events, base_rate].
    """
    combo_id = combo['combo_id']
    cusum_rate = combo['cusum_rate']
    fast = int(combo['fast'])
    slow = int(combo['slow'])

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
    cusum_events = bars_index[event_indices]

    fast_ma = bars['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars['close'].ewm(span=slow, adjust=False).mean()

    dma_signal = pd.Series(0, index=bars_index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    trend_events = active_trend_labels[active_trend_labels['side'] != 0].copy()

    candidate_timestamps = cusum_events.intersection(trend_events.index)

    n_candidates = len(candidate_timestamps)
    n_events = len(trend_events)

    if n_candidates == 0:
        return {
            'combo_id': combo_id,
            'recall': 0.0,
            'cpr': 0.0,
            'coverage': 0.0,
            'lift': 0.0,
            'n_candidates': 0,
            'n_true_positives': 0,
            'n_events': n_events,
            'base_rate': 1.0 if n_events > 0 else 0.0,
        }

    true_positives = 0
    for ts in candidate_timestamps:
        dma_side = dma_signal.loc[ts]
        trend_side = trend_events.loc[ts, 'side']
        if dma_side == trend_side:
            true_positives += 1

    fn = n_events - n_candidates
    recall = true_positives / (true_positives + fn) if (true_positives + fn) > 0 else 0.0
    cpr = true_positives / n_candidates if n_candidates > 0 else 0.0
    coverage = n_candidates / n_events if n_events > 0 else 0.0

    n_positive = (trend_events['side'] == 1).sum()
    n_negative = (trend_events['side'] == -1).sum()
    base_rate = max(n_positive, n_negative) / len(trend_events) if len(trend_events) > 0 else 0.5

    lift = cpr / base_rate if base_rate > 0 else 0.0

    return {
        'combo_id': combo_id,
        'recall': recall,
        'cpr': cpr,
        'coverage': coverage,
        'lift': lift,
        'n_candidates': n_candidates,
        'n_true_positives': true_positives,
        'n_events': n_events,
        'base_rate': base_rate,
    }


def compute_all_lightweight_metrics(
    bars: pd.DataFrame,
    trend_labels: pd.DataFrame,
    k_lookup: pd.DataFrame,
    combos: pd.DataFrame,
    pt_sl: float = 1.0,
    trend_labels_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Compute lightweight metrics for all parameter combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Default trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combos: DataFrame of all combos (from param_grid).
    :param pt_sl: TBM pt/sl in sigma units.
    :param trend_labels_dict: Optional mapping {cusum_rate: trend_labels_df}.
    :returns: DataFrame with lightweight metrics for each combo.
    """
    results = []
    for _, combo in combos.iterrows():
        metrics = compute_lightweight_metrics(
            bars,
            trend_labels,
            k_lookup,
            combo,
            pt_sl,
            trend_labels_dict=trend_labels_dict,
        )
        results.append(metrics)

    return pd.DataFrame(results)
