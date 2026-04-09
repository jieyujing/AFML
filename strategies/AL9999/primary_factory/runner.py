"""Runner - main orchestrator for Primary Model Factory."""

import os
from typing import Optional

import pandas as pd

from afmlkit.feature.core.trend_scan import trend_scan_labels
from afmlkit.sampling import cusum_filter

from .cusum_calibrator import calibrate_cusum_rates
from .deep_scorer import compute_all_deep_metrics
from .lightweight_scorer import compute_all_lightweight_metrics
from .param_grid import expand_deep_param_grid, generate_lightweight_param_grid
from .scorer import compute_composite_score, get_top_candidates


def _generate_trend_labels_per_rate(
    bars: pd.DataFrame,
    calibration_df: pd.DataFrame,
    cusum_rates: list[float],
    l_windows: list[int],
    fallback_trend_labels: Optional[pd.DataFrame] = None,
) -> dict[float, pd.DataFrame]:
    """
    Generate one trend_labels DataFrame per CUSUM rate.

    Each rate uses its calibrated k to create its own CUSUM event set, then
    trend_scan_labels is applied on that event set. If generation fails for a
    rate, fallback_trend_labels is used when provided.
    """
    trend_labels_dict: dict[float, pd.DataFrame] = {}
    prices = bars['close']
    close = prices.values
    log_returns = pd.Series(close).pipe(lambda s: s).values
    log_returns = pd.Series(close).apply(lambda _: 0).values  # placeholder to keep shape explicit
    log_returns = None

    import numpy as np

    log_returns = np.diff(np.log(close))

    for rate in cusum_rates:
        k_row = calibration_df[calibration_df['rate'] == rate]
        if len(k_row) == 0:
            if fallback_trend_labels is not None:
                trend_labels_dict[rate] = fallback_trend_labels.copy()
                continue
            raise ValueError(f"No calibrated k found for cusum_rate={rate}")

        k = float(k_row['k'].iloc[0])

        try:
            event_indices = cusum_filter(log_returns, pd.Series([k]).to_numpy())
            t_events = bars.index[event_indices]
            trend_df = trend_scan_labels(
                price_series=prices,
                t_events=t_events,
                L_windows=l_windows,
            )
            trend_labels_dict[rate] = trend_df
        except Exception:
            if fallback_trend_labels is not None:
                trend_labels_dict[rate] = fallback_trend_labels.copy()
            else:
                raise

    return trend_labels_dict


def run_primary_factory(
    bars_path: str,
    trend_labels_path: str,
    output_dir: str,
    config: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full Primary Model Factory pipeline.

    :param bars_path: Path to dollar bars parquet file.
    :param trend_labels_path: Path to trend_labels parquet.
    :param output_dir: Output directory for results.
    :param config: PRIMARY_FACTORY_CONFIG dict (uses default if None).
    :returns: tuple of (scoring_final_df, top_candidates_df, calibration_df).
    """
    if config is None:
        from strategies.AL9999.config import PRIMARY_FACTORY_CONFIG
        config = PRIMARY_FACTORY_CONFIG

    cusum_rates = config.get('cusum_rates', [0.05, 0.10, 0.15])
    fast_windows = config.get('fast_windows', [5, 8, 10, 12, 15])
    slow_windows = config.get('slow_windows', [20, 30, 40, 50, 60])
    vertical_bars = config.get('vertical_bars', [10, 20, 30])
    pt_sl = config.get('pt_sl', 1.0)
    top_n_lightweight = config.get('top_n_lightweight', 20)
    top_n_final = config.get('top_n_final', 5)
    test_ratio = config.get('oos_test_ratio', 0.30)
    trend_windows = config.get('trend_windows', [10, 20, 30])
    score_weights = config.get(
        'score_weights',
        {
            'effective_recall': 0.45,
            'turnover': -0.10,
            'uniqueness': 0.10,
            'sharpe': 0.20,
            'net_pnl': 0.10,
            'mdd': 0.05,
        },
    )
    k_search_min = config.get('k_search_min', 0.1)
    k_search_max = config.get('k_search_max', 10.0)
    k_tolerance = config.get('k_tolerance', 1e-4)

    output_path = os.path.join(output_dir, 'primary_search')
    os.makedirs(output_path, exist_ok=True)

    print('[Step 0] Loading data...')
    bars = pd.read_parquet(bars_path)
    trend_labels = pd.read_parquet(trend_labels_path)
    print(f'  Bars: {len(bars)} rows')
    print(f'  Trend labels: {len(trend_labels)} rows')

    print('\n[Step 1] CUSUM calibration...')
    calibration_df = calibrate_cusum_rates(
        bars,
        target_rates=cusum_rates,
        k_min=k_search_min,
        k_max=k_search_max,
        tol=k_tolerance,
    )
    print(f'  Calibrated {len(calibration_df)} rates')
    calibration_path = os.path.join(output_path, 'cusum_calibration.parquet')
    calibration_df.to_parquet(calibration_path)
    print(f'  Saved: {calibration_path}')

    print('\n[Step 1b] Generating per-rate trend labels...')
    trend_labels_dict = _generate_trend_labels_per_rate(
        bars=bars,
        calibration_df=calibration_df,
        cusum_rates=cusum_rates,
        l_windows=trend_windows,
        fallback_trend_labels=trend_labels,
    )
    print(f'  Generated/loaded trend labels for {len(trend_labels_dict)} rates')

    print('\n[Step 2] Generating parameter grid...')
    lightweight_combos_df = generate_lightweight_param_grid(
        cusum_rates=cusum_rates,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
    )
    print(f'  Generated {len(lightweight_combos_df)} lightweight combinations')

    print('\n[Step 3] Computing lightweight metrics...')
    lightweight_df = compute_all_lightweight_metrics(
        bars=bars,
        trend_labels=trend_labels,
        trend_labels_dict=trend_labels_dict,
        k_lookup=calibration_df,
        combos=lightweight_combos_df,
        pt_sl=pt_sl,
    )
    lightweight_df['effective_recall'] = lightweight_df['recall'] * lightweight_df['lift']
    print(f'  Computed metrics for {len(lightweight_df)} combos')

    lightweight_path = os.path.join(output_path, 'scoring_lightweight.csv')
    lightweight_df.to_csv(lightweight_path, index=False)
    print(f'  Saved: {lightweight_path}')

    print(f'\n[Step 4] Selecting Top-{top_n_lightweight} for deep scoring...')
    top_lightweight = lightweight_df.nlargest(top_n_lightweight, 'effective_recall')
    top_combo_ids = top_lightweight['combo_id'].tolist()
    top_lightweight_combos = lightweight_combos_df[
        lightweight_combos_df['combo_id'].isin(top_combo_ids)
    ]
    top_combos = expand_deep_param_grid(
        lightweight_combos=top_lightweight_combos,
        vertical_bars=vertical_bars,
    )
    print(f'  Expanded to {len(top_combos)} deep-scoring combos')

    print('\n[Step 5] Computing deep metrics...')
    deep_df = compute_all_deep_metrics(
        bars=bars,
        trend_labels=trend_labels,
        trend_labels_dict=trend_labels_dict,
        k_lookup=calibration_df,
        combos=top_combos,
        pt_sl=pt_sl,
        test_ratio=test_ratio,
    )
    print(f'  Computed deep metrics for {len(deep_df)} combos')

    deep_path = os.path.join(output_path, 'scoring_deep.csv')
    deep_df.to_csv(deep_path, index=False)
    print(f'  Saved: {deep_path}')

    print('\n[Step 6] Computing composite score...')
    scored_df = compute_composite_score(
        lightweight_df=lightweight_df,
        deep_df=deep_df,
        weights=score_weights,
    )

    final_path = os.path.join(output_path, 'scoring_final.csv')
    scored_df.to_csv(final_path, index=False)
    print(f'  Saved: {final_path}')

    print(f'\n[Step 7] Selecting Top-{top_n_final} candidates (stratified by CUSUM rate)...')
    top_candidates = get_top_candidates(scored_df, top_n=top_n_final)

    covered_rates = set(top_candidates['cusum_rate'].unique())
    all_rates = set(cusum_rates)

    lightweight_ranked = lightweight_df.copy()
    lightweight_ranked['cusum_rate'] = (
        lightweight_ranked['combo_id'].str.extract(r'rate=([0-9.]+)')[0].astype(float)
    )

    missing_rates = all_rates - covered_rates
    additional_base_combos = []
    if missing_rates:
        print(f'  Missing rates in Top-{top_n_final}: {missing_rates}')
        for rate in sorted(missing_rates):
            rate_best = lightweight_ranked[lightweight_ranked['cusum_rate'] == rate].head(1)
            if len(rate_best) > 0:
                combo_id = rate_best['combo_id'].iloc[0]
                additional_base_combos.append(combo_id)

        if additional_base_combos:
            print(
                f'  Computing deep metrics for {len(additional_base_combos)} additional lightweight bases...'
            )
            additional_lightweight_df = lightweight_combos_df[
                lightweight_combos_df['combo_id'].isin(additional_base_combos)
            ]
            additional_combos_df = expand_deep_param_grid(
                lightweight_combos=additional_lightweight_df,
                vertical_bars=vertical_bars,
            )
            additional_deep = compute_all_deep_metrics(
                bars=bars,
                trend_labels=trend_labels,
                trend_labels_dict=trend_labels_dict,
                k_lookup=calibration_df,
                combos=additional_combos_df,
                pt_sl=pt_sl,
                test_ratio=test_ratio,
            )
            deep_df = pd.concat([deep_df, additional_deep], ignore_index=True)

            deep_df.to_csv(deep_path, index=False)
            print(f'  Updated deep metrics saved: {len(deep_df)} total candidates')

            scored_df = compute_composite_score(
                lightweight_df=lightweight_df,
                deep_df=deep_df,
                weights=score_weights,
            )

            scored_df.to_csv(final_path, index=False)
            top_candidates = get_top_candidates(scored_df, top_n=top_n_final)

    covered_rates = set(top_candidates['cusum_rate'].unique())
    missing_rates = all_rates - covered_rates
    if missing_rates:
        for rate in sorted(missing_rates):
            rate_best = scored_df[scored_df['cusum_rate'] == rate].head(1)
            if len(rate_best) > 0:
                top_candidates = pd.concat([top_candidates, rate_best], ignore_index=True)

    top_candidates = top_candidates.sort_values('score', ascending=False).reset_index(drop=True)
    top_candidates['rank'] = range(1, len(top_candidates) + 1)

    candidates_path = os.path.join(output_path, 'top_candidates.parquet')
    top_candidates.to_parquet(candidates_path)
    print(f'  Saved: {candidates_path}')

    print('\n' + '=' * 60)
    print('  Primary Model Factory Complete')
    print('=' * 60)
    print(f'  Output directory: {output_path}')
    print(f'  Top candidates ({len(top_candidates)} total, stratified by rate):')
    for _, row in top_candidates.iterrows():
        print(f"    {row['rank']}. {row['combo_id']}")
        print(f"       Score={row['score']:.4f}, Recall={row['recall']:.4f}, Lift={row['lift']:.4f}")

    return scored_df, top_candidates, calibration_df
