"""Composite scorer for Primary Model Factory - weighted score with rank-percentile z."""

import numpy as np
import pandas as pd


def compute_composite_score(
    lightweight_df: pd.DataFrame,
    deep_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute weighted composite score for all combos.

    :param lightweight_df: All combos with lightweight metrics.
    :param deep_df: Top-N combos with deep metrics.
    :param weights: Score component weights (default: recall=0.45, lift=0.20, cpr=0.15,
                   turnover=-0.10, uniqueness=0.10).
    :returns: DataFrame with all combos scored and ranked.
              Columns include: combo_id, score, rank, and all metric columns.
    """
    if weights is None:
        weights = {
            'recall': 0.45,
            'lift': 0.20,
            'cpr': 0.15,
            'turnover': -0.10,
            'uniqueness': 0.10,
        }

    # Merge lightweight and deep metrics
    # Deep metrics only exist for Top-N combos
    merged = lightweight_df.copy()

    # Add deep metric columns if not present
    for col in ['uniqueness', 'turnover', 'regime_stability', 'oos_recall']:
        if col not in merged.columns:
            merged[col] = np.nan
    for col in ['oos_unreliable', 'low_info']:
        if col not in merged.columns:
            merged[col] = False

    # Fill deep metrics from deep_df
    for _, deep_row in deep_df.iterrows():
        combo_id = deep_row['combo_id']
        mask = merged['combo_id'] == combo_id
        for col in ['uniqueness', 'turnover', 'regime_stability', 'oos_recall', 'oos_unreliable', 'low_info']:
            if col in deep_row:
                merged.loc[mask, col] = deep_row[col]

    # For non-Top-N combos, fill deep metrics with median
    for col in ['uniqueness', 'turnover', 'regime_stability', 'oos_recall']:
        if col in deep_df.columns:
            median_val = deep_df[col].median()
            merged[col] = merged[col].fillna(median_val)

    # Handle edge cases
    # If CPR = 0, set Lift = 0
    merged.loc[merged['cpr'] == 0, 'lift'] = 0.0

    # Compute rank-percentile z-scores for each metric
    n = len(merged)
    if n <= 1:
        merged['score'] = 0.0
        merged['rank'] = 1
        return merged

    # Compute z-scores using rank percentile: z = (rank - 1) / (n - 1)
    # For turnover, higher is worse, so we invert (lower z = better)
    metrics_for_score = ['recall', 'lift', 'cpr', 'turnover', 'uniqueness']

    z_scores = {}
    for metric in metrics_for_score:
        if metric not in merged.columns:
            z_scores[metric] = np.zeros(n)
            continue

        values = merged[metric].values
        if np.isnan(values).all() or np.std(values) == 0:
            z_scores[metric] = np.zeros(n)
            continue

        # Rank (1 = best for recall/lift/cpr/uniqueness, 1 = worst for turnover)
        if metric == 'turnover':
            # Lower turnover is better, so rank ascending (lowest = rank 1)
            ranks = pd.Series(values).rank(ascending=True, method='average').values
        else:
            # Higher is better for recall/lift/cpr/uniqueness
            ranks = pd.Series(values).rank(ascending=False, method='average').values

        # z-score: invert so higher z = better performance
        # rank 1 → z = 1 (best), rank n → z = 0 (worst)
        z = 1.0 - (ranks - 1) / (n - 1)
        z_scores[metric] = z

    # Compute composite score
    scores = np.zeros(n)
    for metric, weight in weights.items():
        if metric in z_scores:
            scores += weight * z_scores[metric]

    # Handle case where all metrics have zero variance
    if np.std(scores) == 0:
        scores = np.zeros(n)

    merged['score'] = scores

    # Assign rank (higher score = better = rank 1)
    merged['rank'] = merged['score'].rank(ascending=False, method='min').astype(int)

    # Sort by rank
    merged = merged.sort_values('rank').reset_index(drop=True)

    return merged


def get_top_candidates(
    scored_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Get top-N candidates from scored DataFrame.

    :param scored_df: DataFrame with composite scores.
    :param top_n: Number of top candidates to return.
    :returns: DataFrame with top-N candidates.
    """
    return scored_df.head(top_n).copy()