"""Runner - main orchestrator for Primary Model Factory."""

import os
import pandas as pd
from typing import Optional

from .cusum_calibrator import calibrate_cusum_rates
from .param_grid import generate_param_grid
from .lightweight_scorer import compute_all_lightweight_metrics
from .deep_scorer import compute_all_deep_metrics
from .scorer import compute_composite_score, get_top_candidates


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
    # Default config
    if config is None:
        from strategies.AL9999.config import PRIMARY_FACTORY_CONFIG
        config = PRIMARY_FACTORY_CONFIG

    # Extract config values
    cusum_rates = config.get('cusum_rates', [0.05, 0.10, 0.15])
    fast_windows = config.get('fast_windows', [5, 8, 10, 12, 15])
    slow_windows = config.get('slow_windows', [20, 30, 40, 50, 60])
    vertical_bars = config.get('vertical_bars', [10, 20, 30])
    pt_sl = config.get('pt_sl', 1.0)
    top_n_lightweight = config.get('top_n_lightweight', 20)
    top_n_final = config.get('top_n_final', 5)
    test_ratio = config.get('oos_test_ratio', 0.30)
    score_weights = config.get('score_weights', {
        'recall': 0.45, 'lift': 0.20, 'cpr': 0.15,
        'turnover': -0.10, 'uniqueness': 0.10,
    })
    k_search_min = config.get('k_search_min', 0.1)
    k_search_max = config.get('k_search_max', 10.0)
    k_tolerance = config.get('k_tolerance', 1e-4)

    # Create output directory
    output_path = os.path.join(output_dir, 'primary_search')
    os.makedirs(output_path, exist_ok=True)

    # Step 0: Load data
    print("[Step 0] Loading data...")
    bars = pd.read_parquet(bars_path)
    trend_labels = pd.read_parquet(trend_labels_path)
    print(f"  Bars: {len(bars)} rows")
    print(f"  Trend labels: {len(trend_labels)} rows")

    # Step 1: CUSUM calibration
    print("\n[Step 1] CUSUM calibration...")
    calibration_df = calibrate_cusum_rates(
        bars,
        target_rates=cusum_rates,
        k_min=k_search_min,
        k_max=k_search_max,
        tol=k_tolerance,
    )
    print(f"  Calibrated {len(calibration_df)} rates")
    calibration_path = os.path.join(output_path, 'cusum_calibration.parquet')
    calibration_df.to_parquet(calibration_path)
    print(f"  Saved: {calibration_path}")

    # Step 2: Generate parameter grid
    print("\n[Step 2] Generating parameter grid...")
    combos_df = generate_param_grid(
        cusum_rates=cusum_rates,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
        vertical_bars=vertical_bars,
    )
    print(f"  Generated {len(combos_df)} valid combinations")

    # Step 3: Compute lightweight metrics
    print("\n[Step 3] Computing lightweight metrics...")
    lightweight_df = compute_all_lightweight_metrics(
        bars=bars,
        trend_labels=trend_labels,
        k_lookup=calibration_df,
        combos=combos_df,
        pt_sl=pt_sl,
    )
    print(f"  Computed metrics for {len(lightweight_df)} combos")

    # Save lightweight results
    lightweight_path = os.path.join(output_path, 'scoring_lightweight.csv')
    lightweight_df.to_csv(lightweight_path, index=False)
    print(f"  Saved: {lightweight_path}")

    # Step 4: Get Top-N for deep scoring
    print(f"\n[Step 4] Selecting Top-{top_n_lightweight} for deep scoring...")
    # Sort by recall to get top candidates
    top_lightweight = lightweight_df.nlargest(top_n_lightweight, 'recall')
    top_combo_ids = top_lightweight['combo_id'].tolist()
    top_combos = combos_df[combos_df['combo_id'].isin(top_combo_ids)]
    print(f"  Selected {len(top_combos)} combos")

    # Step 5: Compute deep metrics
    print("\n[Step 5] Computing deep metrics...")
    deep_df = compute_all_deep_metrics(
        bars=bars,
        trend_labels=trend_labels,
        k_lookup=calibration_df,
        combos=top_combos,
        pt_sl=pt_sl,
        test_ratio=test_ratio,
    )
    print(f"  Computed deep metrics for {len(deep_df)} combos")

    # Save deep results
    deep_path = os.path.join(output_path, 'scoring_deep.csv')
    deep_df.to_csv(deep_path, index=False)
    print(f"  Saved: {deep_path}")

    # Step 6: Compute composite score
    print("\n[Step 6] Computing composite score...")
    scored_df = compute_composite_score(
        lightweight_df=lightweight_df,
        deep_df=deep_df,
        weights=score_weights,
    )

    # Save final scoring
    final_path = os.path.join(output_path, 'scoring_final.csv')
    scored_df.to_csv(final_path, index=False)
    print(f"  Saved: {final_path}")

    # Step 7: Get top candidates
    print(f"\n[Step 7] Selecting Top-{top_n_final} candidates...")
    top_candidates = get_top_candidates(scored_df, top_n=top_n_final)

    # Save top candidates
    candidates_path = os.path.join(output_path, 'top_candidates.parquet')
    top_candidates.to_parquet(candidates_path)
    print(f"  Saved: {candidates_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Primary Model Factory Complete")
    print("=" * 60)
    print(f"  Output directory: {output_path}")
    print(f"  Top-{top_n_final} candidates:")
    for i, row in top_candidates.iterrows():
        print(f"    {row['rank']}. {row['combo_id']}")
        print(f"       Score={row['score']:.4f}, Recall={row['recall']:.4f}, Lift={row['lift']:.4f}")

    return scored_df, top_candidates, calibration_df