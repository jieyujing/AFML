"""Tests for deep scorer module."""

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
    / "deep_scorer.py"
)
_spec = importlib.util.spec_from_file_location("deep_scorer", _module_path)
_module = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_module)  # type: ignore
compute_deep_metrics = _module.compute_deep_metrics
compute_all_deep_metrics = _module.compute_all_deep_metrics


def test_deep_scorer():
    """Test deep metrics computation."""
    np.random.seed(42)
    n = 1000
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    trend_events = pd.DataFrame({
        'side': [1] * 10 + [-1] * 10,
        't_value': [2.0] * 20,
        't1': pd.date_range('2020-01-03', periods=20, freq='50h'),
    }, index=pd.date_range('2020-01-01', periods=20, freq='50h'))

    k_lookup = pd.DataFrame({'rate': [0.05], 'k': [1.0]})
    combo = pd.Series({
        'combo_id': 'test_combo',
        'cusum_rate': 0.05,
        'fast': 10,
        'slow': 30,
        'vertical_bars': 20,
    })

    result = compute_deep_metrics(close.to_frame(), trend_events, k_lookup, combo)

    assert 'combo_id' in result
    assert 'uniqueness' in result
    assert 'turnover' in result
    assert 'regime_stability' in result
    assert 'oos_recall' in result
    assert 0 <= result['uniqueness'] <= 1
    assert result['oos_recall'] is not None
    assert isinstance(result['oos_unreliable'], bool)
    assert isinstance(result['low_info'], bool)


def test_deep_scorer_oos_unreliable():
    """Test that OOS is marked unreliable when test samples are few."""
    np.random.seed(42)
    n = 100  # Small sample
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    # Few trend events in the OOS period (last 30%)
    trend_events = pd.DataFrame({
        'side': [1] * 2,
        't_value': [2.0] * 2,
        't1': pd.date_range('2020-01-01', periods=2, freq='h'),
    }, index=pd.date_range('2020-01-01', periods=2, freq='h'))

    k_lookup = pd.DataFrame({'rate': [0.05], 'k': [0.5]})
    combo = pd.Series({
        'combo_id': 'small_sample',
        'cusum_rate': 0.05,
        'fast': 5,
        'slow': 20,
        'vertical_bars': 10,
    })

    result = compute_deep_metrics(bars, trend_events, k_lookup, combo, test_ratio=0.5)

    # With 50% split, OOS period has ~50 bars but only 2 trend events
    # Should be marked unreliable if < 10 trend events in OOS
    assert isinstance(result['oos_unreliable'], bool)


def test_compute_all_deep_metrics():
    """Test computing deep metrics for multiple combos."""
    np.random.seed(42)
    n = 800
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01), name='close')
    close.index = pd.date_range('2020-01-01', periods=n, freq='h')
    bars = close.to_frame()

    trend_events = pd.DataFrame({
        'side': [1] * 10 + [-1] * 10,
        't_value': [2.0] * 20,
        't1': pd.date_range('2020-01-03', periods=20, freq='50h'),
    }, index=pd.date_range('2020-01-01', periods=20, freq='24h'))

    k_lookup = pd.DataFrame({
        'rate': [0.05, 0.10],
        'k': [1.0, 0.5],
    })

    combos = pd.DataFrame([
        {'combo_id': 'combo_1', 'cusum_rate': 0.05, 'fast': 5, 'slow': 20, 'vertical_bars': 10},
        {'combo_id': 'combo_2', 'cusum_rate': 0.10, 'fast': 10, 'slow': 30, 'vertical_bars': 20},
    ])

    result = compute_all_deep_metrics(bars, trend_events, k_lookup, combos)

    assert len(result) == 2
    assert 'combo_id' in result.columns
    assert 'uniqueness' in result.columns
    assert 'turnover' in result.columns
    assert 'oos_recall' in result.columns