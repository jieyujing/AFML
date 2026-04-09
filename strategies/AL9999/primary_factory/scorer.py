"""Composite scorer for Primary Model Factory - weighted score with rate-normalized z."""

import numpy as np
import pandas as pd


MIN_OOS_RECALL = 0.01
MIN_TRADE_COUNT = 10
UNVERIFIED_PENALTY = 1.0
UNRELIABLE_PENALTY = 0.75
LOW_INFO_PENALTY = 0.50
LOW_TRADE_COUNT_PENALTY = 0.50
LOW_OOS_PENALTY = 0.50


def _collapse_deep_metrics(deep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse expanded deep metrics to one representative row per base combo.

    Preference order prioritizes OOS quality and reliability before IS trade quality.
    """
    if 'base_combo_id' not in deep_df.columns:
        return deep_df.copy()

    collapsed = deep_df.copy()
    collapsed['selected_deep_combo_id'] = collapsed['combo_id']

    if 'oos_unreliable' not in collapsed.columns:
        collapsed['oos_unreliable'] = False
    if 'low_info' not in collapsed.columns:
        collapsed['low_info'] = False
    if 'trade_count' not in collapsed.columns:
        collapsed['trade_count'] = 0
    if 'oos_recall' not in collapsed.columns:
        collapsed['oos_recall'] = 0.0

    collapsed = collapsed.sort_values(
        [
            'base_combo_id',
            'oos_unreliable',
            'low_info',
            'oos_recall',
            'trade_count',
            'sharpe',
            'net_pnl',
            'mdd',
            'uniqueness',
        ],
        ascending=[True, True, True, False, False, False, False, False, False],
    )
    collapsed = collapsed.drop_duplicates(subset=['base_combo_id'], keep='first').reset_index(drop=True)
    return collapsed


def _rank_to_unit_interval(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Convert a vector into rank-based [0, 1] scores."""
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5)
    if np.isnan(values).all() or np.nanstd(values) == 0:
        return np.full(n, 0.5)

    ranks = pd.Series(values).rank(ascending=not higher_is_better, method='average').values
    z = 1.0 - (ranks - 1) / (n - 1)
    return z



def compute_composite_score(
    lightweight_df: pd.DataFrame,
    deep_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute weighted composite score for scored combos.

    Only combos with deep metrics are considered verified and eligible for final
    ranking. Reliability and minimum-information checks are applied as penalties.
    """
    if weights is None:
        weights = {
            'effective_recall': 0.45,
            'turnover': 0.10,
            'uniqueness': 0.10,
            'sharpe': 0.20,
            'net_pnl': 0.10,
            'mdd': 0.05,
        }

    if 'base_combo_id' in deep_df.columns:
        deep_collapsed = _collapse_deep_metrics(deep_df)
        lightweight_base = lightweight_df.rename(columns={'combo_id': 'base_combo_id'}).copy()
        merged = deep_collapsed.merge(
            lightweight_base,
            on='base_combo_id',
            how='left',
            suffixes=('', '_light'),
        )
        merged['combo_id'] = merged['base_combo_id']

        for col in [
            'recall', 'cpr', 'coverage', 'lift', 'n_candidates',
            'n_true_positives', 'n_events', 'base_rate', 'effective_recall'
        ]:
            light_col = f'{col}_light'
            if col not in merged.columns and light_col in merged.columns:
                merged[col] = merged[light_col]

        drop_cols = [
            col for col in merged.columns
            if col.endswith('_light') and col != 'base_combo_id_light'
        ]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)
    else:
        merged = lightweight_df.merge(deep_df, on='combo_id', how='left', suffixes=('', '_deep'))

    if 'cpr' in merged.columns:
        merged.loc[merged['cpr'] == 0, 'lift'] = 0.0

    if 'effective_recall' not in merged.columns:
        merged['effective_recall'] = merged['recall'] * merged['lift']

    merged['cusum_rate'] = merged['combo_id'].str.extract(r'rate=([0-9.]+)')[0].astype(float)
    merged['is_verified'] = merged['sharpe'].notna()

    for col, default in [
        ('oos_unreliable', False),
        ('low_info', False),
        ('trade_count', 0),
        ('oos_recall', 0.0),
        ('uniqueness', np.nan),
        ('turnover', np.nan),
        ('net_pnl', np.nan),
        ('sharpe', np.nan),
        ('mdd', np.nan),
    ]:
        if col not in merged.columns:
            merged[col] = default

    verified = merged[merged['is_verified']].copy()
    unverified = merged[~merged['is_verified']].copy()

    if len(verified) == 0:
        merged['score'] = -UNVERIFIED_PENALTY
        merged['rank'] = merged['score'].rank(ascending=False, method='min').astype(int)
        merged = merged.sort_values('rank').reset_index(drop=True)
        return merged

    verified['effective_recall_rate_z'] = 0.0
    has_rates = verified['cusum_rate'].notna().any()

    if has_rates:
        for rate in verified['cusum_rate'].dropna().unique():
            rate_mask = verified['cusum_rate'] == rate
            rate_indices = verified[rate_mask].index.tolist()
            rate_values = verified.loc[rate_indices, 'effective_recall'].values
            verified.loc[rate_indices, 'effective_recall_rate_z'] = _rank_to_unit_interval(
                rate_values, higher_is_better=True
            )

        nan_mask = verified['cusum_rate'].isna()
        if nan_mask.any():
            verified.loc[nan_mask, 'effective_recall_rate_z'] = verified.loc[
                ~nan_mask, 'effective_recall_rate_z'
            ].median()
    else:
        verified['effective_recall_rate_z'] = _rank_to_unit_interval(
            verified['effective_recall'].values, higher_is_better=True
        )

    z_scores = {
        'turnover': _rank_to_unit_interval(verified['turnover'].values, higher_is_better=False),
        'uniqueness': _rank_to_unit_interval(verified['uniqueness'].values, higher_is_better=True),
        'sharpe': _rank_to_unit_interval(verified['sharpe'].values, higher_is_better=True),
        'net_pnl': _rank_to_unit_interval(verified['net_pnl'].values, higher_is_better=True),
        'mdd': _rank_to_unit_interval(verified['mdd'].values, higher_is_better=True),
    }

    scores = np.zeros(len(verified))
    scores += weights.get('effective_recall', 0.45) * verified['effective_recall_rate_z'].values
    scores += weights.get('turnover', 0.10) * z_scores['turnover']
    scores += weights.get('uniqueness', 0.10) * z_scores['uniqueness']
    scores += weights.get('sharpe', 0.20) * z_scores['sharpe']
    scores += weights.get('net_pnl', 0.10) * z_scores['net_pnl']
    scores += weights.get('mdd', 0.05) * z_scores['mdd']

    penalties = np.zeros(len(verified))
    oos_unreliable_mask = verified['oos_unreliable'].astype(bool).values
    low_info_mask = verified['low_info'].astype(bool).values
    low_trade_count_mask = verified['trade_count'].fillna(0).values < MIN_TRADE_COUNT
    low_oos_mask = verified['oos_recall'].fillna(0.0).values < MIN_OOS_RECALL

    penalties += np.where(oos_unreliable_mask, UNRELIABLE_PENALTY, 0.0)
    penalties += np.where(low_info_mask, LOW_INFO_PENALTY, 0.0)
    penalties += np.where(low_trade_count_mask, LOW_TRADE_COUNT_PENALTY, 0.0)
    penalties += np.where(low_oos_mask, LOW_OOS_PENALTY, 0.0)

    verified['score'] = scores - penalties
    verified['eligibility_reason'] = 'verified'
    verified.loc[oos_unreliable_mask, 'eligibility_reason'] = 'oos_unreliable'
    verified.loc[low_info_mask, 'eligibility_reason'] = 'low_info'
    verified.loc[low_trade_count_mask, 'eligibility_reason'] = 'low_trade_count'
    verified.loc[low_oos_mask, 'eligibility_reason'] = 'low_oos_recall'

    if len(unverified) > 0:
        unverified = unverified.copy()
        unverified['effective_recall_rate_z'] = np.nan
        unverified['score'] = -UNVERIFIED_PENALTY
        unverified['eligibility_reason'] = 'unverified_no_deep_metrics'

    merged = pd.concat([verified, unverified], ignore_index=True, sort=False)
    merged['rank'] = merged['score'].rank(ascending=False, method='min').astype(int)
    merged = merged.sort_values(['is_verified', 'score'], ascending=[False, False]).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    return merged



def get_top_candidates(
    scored_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Get top-N verified candidates from scored DataFrame.
    """
    if 'is_verified' not in scored_df.columns:
        return scored_df.head(top_n).copy()

    eligible = scored_df[scored_df['is_verified']].copy()
    if len(eligible) == 0:
        return scored_df.head(top_n).copy()
    return eligible.head(top_n).copy()
