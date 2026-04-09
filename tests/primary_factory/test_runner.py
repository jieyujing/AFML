"""Tests for runner module."""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.AL9999.primary_factory.runner import run_primary_factory


def test_runner_smoke():
    """Smoke test for the full pipeline."""
    np.random.seed(42)
    n = 500
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    trend_labels = pd.DataFrame({
        'side': [1] * 5 + [-1] * 5,
        't_value': [2.0] * 10,
        't1': pd.date_range('2020-01-03', periods=10, freq='50h'),
    }, index=pd.date_range('2020-01-01', periods=10, freq='50h'))

    with tempfile.TemporaryDirectory() as tmpdir:
        bars_path = os.path.join(tmpdir, 'bars.parquet')
        trend_path = os.path.join(tmpdir, 'trend.parquet')
        bars.to_parquet(bars_path)
        trend_labels.to_parquet(trend_path)

        scoring, top5, calib = run_primary_factory(
            bars_path, trend_path, tmpdir,
            config={
                'cusum_rates': [0.05],
                'fast_windows': [5, 10],
                'slow_windows': [20, 30],
                'pt_sl': 1.0,
                'vertical_bars': [10],
                'top_n_lightweight': 3,
                'top_n_final': 2,
                'oos_test_ratio': 0.30,
                'score_weights': {
                    'recall': 0.45, 'lift': 0.20, 'cpr': 0.15,
                    'turnover': -0.10, 'uniqueness': 0.10,
                },
                'k_search_min': 0.001,  # Lower for synthetic data
                'k_search_max': 10.0,
                'k_tolerance': 1e-4,
            }
        )

        assert len(top5) == 2
        assert len(scoring) > 0
        assert len(calib) == 1
        assert scoring['combo_id'].nunique() == len(scoring)
        assert 'selected_deep_combo_id' in scoring.columns
        assert os.path.exists(os.path.join(tmpdir, 'primary_search'))
        assert os.path.exists(os.path.join(tmpdir, 'primary_search', 'top_candidates.parquet'))


def test_runner_output_structure():
    """Test that all expected output files are created."""
    np.random.seed(42)
    n = 300
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    trend_labels = pd.DataFrame({
        'side': [1] * 3 + [-1] * 3,
        't_value': [2.0] * 6,
        't1': pd.date_range('2020-01-03', periods=6, freq='h'),
    }, index=pd.date_range('2020-01-01', periods=6, freq='24h'))

    with tempfile.TemporaryDirectory() as tmpdir:
        bars_path = os.path.join(tmpdir, 'bars.parquet')
        trend_path = os.path.join(tmpdir, 'trend.parquet')
        bars.to_parquet(bars_path)
        trend_labels.to_parquet(trend_path)

        run_primary_factory(
            bars_path, trend_path, tmpdir,
            config={
                'cusum_rates': [0.05],
                'fast_windows': [5],
                'slow_windows': [20],
                'vertical_bars': [10],
                'top_n_lightweight': 1,
                'top_n_final': 1,
                'k_search_min': 0.001,
            }
        )

        # Check all expected files exist
        expected_files = [
            'cusum_calibration.parquet',
            'scoring_lightweight.csv',
            'scoring_deep.csv',
            'scoring_final.csv',
            'top_candidates.parquet',
        ]

        output_path = os.path.join(tmpdir, 'primary_search')
        for f in expected_files:
            assert os.path.exists(os.path.join(output_path, f)), f"Missing: {f}"
