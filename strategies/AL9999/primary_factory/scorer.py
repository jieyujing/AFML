"""Composite scorer for Primary Model Factory - weighted score with rate-normalized z."""

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
    :param weights: Score component weights (default: effective_recall=0.45,
                   turnover=-0.10, uniqueness=0.10).
    :returns: DataFrame with all combos scored and ranked.
              Columns include: combo_id, score, rank, and all metric columns.
    """
    if weights is None:
        weights = {
            'effective_recall': 0.45,
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

    # Compute EffectiveRecall = Recall × Lift
    # This combines recall quality with signal strength
    merged['effective_recall'] = merged['recall'] * merged['lift']

    # Extract cusum_rate from combo_id for rate-normalization
    merged['cusum_rate'] = merged['combo_id'].str.extract(r'rate=([0-9.]+)')[0].astype(float)

    n = len(merged)
    if n <= 1:
        merged['score'] = 0.0
        merged['rank'] = 1
        return merged

    # ========================================
    # Rate-normalized z-score for EffectiveRecall
    # ========================================
    # Within each rate, compute rank percentile z-score
    # This compares "who does best within the same sampling regime"
    merged['effective_recall_rate_z'] = 0.0

    # Check if we have valid rate information
    has_rates = merged['cusum_rate'].notna().any()

    if has_rates:
        # Rate-normalized scoring
        for rate in merged['cusum_rate'].dropna().unique():
            rate_mask = merged['cusum_rate'] == rate
            rate_indices = merged[rate_mask].index.tolist()
            rate_values = merged.loc[rate_indices, 'effective_recall'].values

            n_rate = len(rate_values)
            if n_rate <= 1:
                merged.loc[rate_indices, 'effective_recall_rate_z'] = 0.5
                continue

            # Check if all values are the same
            if np.std(rate_values) == 0:
                merged.loc[rate_indices, 'effective_recall_rate_z'] = 0.5
                continue

            # Rank within rate (higher is better)
            ranks = pd.Series(rate_values).rank(ascending=False, method='average').values

            # z-score: rank 1 → z = 1 (best), rank n → z = 0 (worst)
            z = 1.0 - (ranks - 1) / (n_rate - 1)
            merged.loc[rate_indices, 'effective_recall_rate_z'] = z

        # Fill NaN rates with global median z
        nan_mask = merged['cusum_rate'].isna()
        if nan_mask.any():
            merged.loc[nan_mask, 'effective_recall_rate_z'] = merged.loc[~nan_mask, 'effective_recall_rate_z'].median()
    else:
        # No rate info: use global z-score
        values = merged['effective_recall'].values
        if np.std(values) == 0:
            merged['effective_recall_rate_z'] = 0.5
        else:
            ranks = pd.Series(values).rank(ascending=False, method='average').values
            z = 1.0 - (ranks - 1) / (n - 1)
            merged['effective_recall_rate_z'] = z

    # ========================================
    # Global z-scores for Turnover and Uniqueness
    # ========================================
    z_scores = {}

    # Turnover: lower is better
    values = merged['turnover'].values
    if np.isnan(values).all() or np.std(values) == 0:
        z_scores['turnover'] = np.full(n, 0.5)
    else:
        ranks = pd.Series(values).rank(ascending=True, method='average').values
        z = 1.0 - (ranks - 1) / (n - 1)
        z_scores['turnover'] = z

    # Uniqueness: higher is better
    values = merged['uniqueness'].values
    if np.isnan(values).all() or np.std(values) == 0:
        z_scores['uniqueness'] = np.full(n, 0.5)
    else:
        ranks = pd.Series(values).rank(ascending=False, method='average').values
        z = 1.0 - (ranks - 1) / (n - 1)
        z_scores['uniqueness'] = z

    # ========================================
    # Compute composite score
    # ========================================
    scores = np.zeros(n)

    # EffectiveRecall (rate-normalized)
    scores += weights.get('effective_recall', 0.45) * merged['effective_recall_rate_z'].values

    # Turnover (global)
    scores += weights.get('turnover', -0.10) * z_scores['turnover']

    # Uniqueness (global)
    scores += weights.get('uniqueness', 0.10) * z_scores['uniqueness']

    # Handle case where all scores are the same
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