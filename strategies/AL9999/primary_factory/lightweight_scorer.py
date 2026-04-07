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
) -> dict:
    """
    Compute lightweight metrics for a single parameter combo.

    :param bars: Dollar bars DataFrame with 'close' column.
    :param trend_labels: Trend scanning labels (index=timestamp, cols=[side, t_value, t1]).
    :param k_lookup: CUSUM rate→k mapping (cols=[rate, k]).
    :param combo: Single combo Series with [combo_id, cusum_rate, fast, slow, vertical_bars].
    :param pt_sl: TBM pt/sl in sigma units (default 1.0).
    :returns: dict with keys: [combo_id, recall, cpr, coverage, lift,
                               n_candidates, n_true_positives, n_events, base_rate].
    """
    combo_id = combo['combo_id']
    cusum_rate = combo['cusum_rate']
    fast = int(combo['fast'])
    slow = int(combo['slow'])
    vertical_bars = int(combo['vertical_bars'])

    # Step 1: Get k value from lookup table
    k_row = k_lookup[k_lookup['rate'] == cusum_rate]
    if len(k_row) == 0:
        raise ValueError(f"No k value found for rate {cusum_rate}")
    k = k_row['k'].iloc[0]

    # Step 2: Apply CUSUM filter to get event indices
    close = bars['close'].values
    log_returns = np.diff(np.log(close))
    event_indices = cusum_filter(log_returns, np.array([k]))

    # Convert to timestamps
    bars_index = bars.index
    cusum_events = bars_index[event_indices]

    # Step 3: Compute DMA signals at CUSUM events
    # DMA: fast_ma > slow_ma → long (+1), else → short (-1)
    fast_ma = bars['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = bars['close'].ewm(span=slow, adjust=False).mean()

    # Get DMA signal at each CUSUM event
    dma_signal = pd.Series(0, index=bars_index)
    dma_signal[fast_ma > slow_ma] = 1
    dma_signal[fast_ma <= slow_ma] = -1

    # Step 4: Match DMA side with trend side
    # Filter trend_labels to only non-zero side events
    trend_events = trend_labels[trend_labels['side'] != 0].copy()

    # Get candidate events (CUSUM events that match trend events by timestamp)
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

    # Calculate true positives: DMA side matches trend side
    true_positives = 0
    for ts in candidate_timestamps:
        dma_side = dma_signal.loc[ts]
        trend_side = trend_events.loc[ts, 'side']
        if dma_side == trend_side:
            true_positives += 1

    # Step 5: Compute metrics
    # Recall = TP / (TP + FN) = TP / total_trend_events_matched
    # For our definition: Recall = TP / n_candidates (how many candidates we got right)
    # But spec says: Recall = TP / (TP + FN)
    # TP = candidates with matching side, FN = trend events not captured by CUSUM
    fn = n_events - n_candidates  # Trend events not captured
    recall = true_positives / (true_positives + fn) if (true_positives + fn) > 0 else 0.0

    # CPR = positive candidates / all candidates
    # "positive" here means candidates where DMA side matches trend side
    cpr = true_positives / n_candidates if n_candidates > 0 else 0.0

    # Coverage = candidates / all events
    coverage = n_candidates / n_events if n_events > 0 else 0.0

    # Base rate = max(class imbalance)
    # This is the accuracy of always predicting the majority class
    # If data is 50/50, base_rate = 0.5 (random baseline)
    # If data is 70% positive, base_rate = 0.7
    n_positive = (trend_events['side'] == 1).sum()
    n_negative = (trend_events['side'] == -1).sum()
    base_rate = max(n_positive, n_negative) / len(trend_events) if len(trend_events) > 0 else 0.5

    # Lift = CPR / base_rate
    # Lift > 1 means our candidates are better than random majority guessing
    # Lift < 1 means we're worse than random
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
) -> pd.DataFrame:
    """
    Compute lightweight metrics for all parameter combos.

    :param bars: Dollar bars DataFrame.
    :param trend_labels: Trend scanning labels.
    :param k_lookup: CUSUM rate→k mapping.
    :param combos: DataFrame of all combos (from param_grid).
    :param pt_sl: TBM pt/sl in sigma units.
    :returns: DataFrame with lightweight metrics for each combo.
    """
    results = []
    for _, combo in combos.iterrows():
        metrics = compute_lightweight_metrics(bars, trend_labels, k_lookup, combo, pt_sl)
        results.append(metrics)

    return pd.DataFrame(results)