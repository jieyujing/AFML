"""Tests for lightweight scorer module."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the module using importlib
_module_path = (
    PROJECT_ROOT
    / "strategies"
    / "AL9999"
    / "primary_factory"
    / "lightweight_scorer.py"
)
_spec = importlib.util.spec_from_file_location("lightweight_scorer", _module_path)
_module = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_module)  # type: ignore
compute_lightweight_metrics = _module.compute_lightweight_metrics
compute_all_lightweight_metrics = _module.compute_all_lightweight_metrics


def test_lightweight_scorer_single_combo():
    """Test lightweight metrics computation for a single combo."""
    np.random.seed(42)
    n = 500
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    # Simple trend labels: every 50th bar is a trend event
    trend_events = pd.DataFrame({
        'side': [1] * 5 + [-1] * 5,
        't_value': [2.0] * 10,
        't1': pd.date_range('2020-01-03', periods=10, freq='h'),
    }, index=pd.date_range('2020-01-01', periods=10, freq='50h'))

    k_lookup = pd.DataFrame({'rate': [0.05], 'k': [1.0]})
    combo = pd.Series({
        'combo_id': 'test_combo',
        'cusum_rate': 0.05,
        'fast': 10,
        'slow': 30,
        'vertical_bars': 20,
    })

    result = compute_lightweight_metrics(bars, trend_events, k_lookup, combo)

    assert 'combo_id' in result
    assert 'recall' in result
    assert 'cpr' in result
    assert 'coverage' in result
    assert 'lift' in result
    assert 0 <= result['recall'] <= 1
    assert 0 <= result['cpr'] <= 1
    assert result['lift'] >= 0


def test_lightweight_scorer_empty_candidates():
    """Test when CUSUM produces no events matching trend labels."""
    np.random.seed(42)
    n = 100
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    # Trend labels at very specific times that won't match CUSUM events
    trend_events = pd.DataFrame({
        'side': [1] * 3,
        't_value': [2.0] * 3,
        't1': pd.date_range('2025-01-01', periods=3, freq='h'),  # Future dates
    }, index=pd.date_range('2025-01-01', periods=3, freq='h'))

    # Use high k to produce very few CUSUM events
    k_lookup = pd.DataFrame({'rate': [0.01], 'k': [10.0]})
    combo = pd.Series({
        'combo_id': 'empty_test',
        'cusum_rate': 0.01,
        'fast': 5,
        'slow': 20,
        'vertical_bars': 10,
    })

    result = compute_lightweight_metrics(bars, trend_events, k_lookup, combo)

    # Should handle gracefully with zero candidates
    assert result['n_candidates'] == 0
    assert result['recall'] == 0.0
    assert result['cpr'] == 0.0
    assert result['coverage'] == 0.0


def test_compute_all_lightweight_metrics():
    """Test computing metrics for multiple combos."""
    np.random.seed(42)
    n = 300
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    trend_events = pd.DataFrame({
        'side': [1] * 5 + [-1] * 5,
        't_value': [2.0] * 10,
        't1': pd.date_range('2020-01-03', periods=10, freq='h'),
    }, index=pd.date_range('2020-01-01', periods=10, freq='24h'))

    k_lookup = pd.DataFrame({
        'rate': [0.05, 0.10],
        'k': [1.0, 0.5],
    })

    combos = pd.DataFrame([
        {'combo_id': 'combo_1', 'cusum_rate': 0.05, 'fast': 5, 'slow': 20, 'vertical_bars': 10},
        {'combo_id': 'combo_2', 'cusum_rate': 0.10, 'fast': 10, 'slow': 30, 'vertical_bars': 20},
    ])

    result = compute_all_lightweight_metrics(bars, trend_events, k_lookup, combos)

    assert len(result) == 2
    assert 'combo_id' in result.columns
    assert 'recall' in result.columns
    assert 'cpr' in result.columns
    assert all(result['recall'] >= 0)
    assert all(result['recall'] <= 1)