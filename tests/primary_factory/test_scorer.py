"""Tests for composite scorer module."""

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
    / "scorer.py"
)
_spec = importlib.util.spec_from_file_location("scorer", _module_path)
_module = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_module)  # type: ignore
compute_composite_score = _module.compute_composite_score
get_top_candidates = _module.get_top_candidates


def test_composite_score():
    """Test composite score computation."""
    np.random.seed(42)

    # Simulate 10 combos with lightweight metrics
    lightweight = pd.DataFrame({
        'combo_id': [f'combo_{i}' for i in range(10)],
        'recall': np.random.rand(10) * 0.5 + 0.3,
        'cpr': np.random.rand(10) * 0.3 + 0.2,
        'coverage': np.random.rand(10),
        'lift': np.random.rand(10) * 2 + 1,
    })

    deep = lightweight.head(5).copy()
    deep['uniqueness'] = np.random.rand(5) * 0.5 + 0.3
    deep['turnover'] = np.random.rand(5) * 50 + 10
    deep['oos_recall'] = np.random.rand(5) * 0.5 + 0.2
    deep['regime_stability'] = np.random.rand(5) * 0.3 + 0.5
    deep['net_pnl'] = np.random.randn(5) * 10
    deep['sharpe'] = np.random.rand(5)
    deep['mdd'] = -np.random.rand(5)
    deep['trade_count'] = np.random.randint(5, 20, size=5)

    result = compute_composite_score(lightweight, deep)

    assert 'score' in result.columns
    assert 'rank' in result.columns
    assert len(result) == 10
    assert result['rank'].tolist() == list(range(1, 11))
    # Score should be highest for rank 1
    assert result.loc[result['rank'] == 1, 'score'].values[0] >= result['score'].max()


def test_composite_score_single_row():
    """Test composite score with single combo."""
    lightweight = pd.DataFrame({
        'combo_id': ['only_one'],
        'recall': [0.5],
        'cpr': [0.3],
        'coverage': [0.4],
        'lift': [1.5],
    })

    deep = lightweight.copy()
    deep['uniqueness'] = [0.4]
    deep['turnover'] = [20.0]
    deep['oos_recall'] = [0.3]
    deep['regime_stability'] = [0.6]
    deep['net_pnl'] = [1.0]
    deep['sharpe'] = [0.5]
    deep['mdd'] = [-0.1]
    deep['trade_count'] = [4]

    result = compute_composite_score(lightweight, deep)

    assert len(result) == 1
    assert result['rank'].iloc[0] == 1


def test_get_top_candidates():
    """Test getting top-N candidates."""
    scored = pd.DataFrame({
        'combo_id': [f'combo_{i}' for i in range(10)],
        'score': np.arange(10, 0, -1),  # Descending scores
        'rank': list(range(1, 11)),
    })

    top5 = get_top_candidates(scored, top_n=5)

    assert len(top5) == 5
    assert top5['rank'].tolist() == [1, 2, 3, 4, 5]


def test_composite_score_weights():
    """Test that weights are applied correctly."""
    # Create two combos: one with high recall, one with high lift
    lightweight = pd.DataFrame({
        'combo_id': ['high_recall', 'high_lift'],
        'recall': [0.9, 0.3],  # high_recall wins on recall
        'cpr': [0.3, 0.3],
        'coverage': [0.5, 0.5],
        'lift': [1.0, 3.0],  # high_lift wins on lift
    })

    deep = lightweight.copy()
    deep['uniqueness'] = [0.5, 0.5]
    deep['turnover'] = [20.0, 20.0]
    deep['oos_recall'] = [0.5, 0.5]
    deep['regime_stability'] = [0.5, 0.5]
    deep['net_pnl'] = [1.0, 1.0]
    deep['sharpe'] = [0.3, 0.3]
    deep['mdd'] = [-0.1, -0.1]
    deep['trade_count'] = [10, 10]

    # With default effective_recall scoring, high_recall should win
    result = compute_composite_score(lightweight, deep)

    assert result['rank'].iloc[0] == 1  # high_recall should be rank 1
    assert result['combo_id'].iloc[0] == 'high_recall'


def test_composite_score_prefers_better_trade_quality_when_signal_quality_ties():
    """Trade quality should break ties when signal quality is the same."""
    lightweight = pd.DataFrame({
        'combo_id': ['weaker_trade', 'better_trade'],
        'recall': [0.6, 0.6],
        'cpr': [0.5, 0.5],
        'coverage': [0.5, 0.5],
        'lift': [1.2, 1.2],
    })

    deep = pd.DataFrame({
        'combo_id': ['weaker_trade', 'better_trade'],
        'uniqueness': [0.5, 0.5],
        'turnover': [20.0, 20.0],
        'oos_recall': [0.4, 0.4],
        'regime_stability': [0.6, 0.6],
        'net_pnl': [1.0, 5.0],
        'sharpe': [0.1, 0.8],
        'mdd': [-2.0, -0.5],
        'trade_count': [12, 12],
    })

    result = compute_composite_score(lightweight, deep)

    assert result.iloc[0]['combo_id'] == 'better_trade'


def test_composite_score_penalizes_unverified_combos():
    """Combos without deep metrics should rank below verified combos."""
    lightweight = pd.DataFrame({
        'combo_id': ['verified_combo', 'unverified_combo'],
        'recall': [0.4, 0.9],
        'cpr': [0.5, 0.9],
        'coverage': [0.4, 0.9],
        'lift': [1.5, 3.0],
    })

    deep = pd.DataFrame({
        'combo_id': ['verified_combo'],
        'uniqueness': [0.6],
        'turnover': [15.0],
        'oos_recall': [0.3],
        'regime_stability': [0.7],
        'net_pnl': [2.0],
        'sharpe': [0.8],
        'mdd': [-0.2],
        'trade_count': [20],
        'oos_unreliable': [False],
        'low_info': [False],
    })

    result = compute_composite_score(lightweight, deep)
    verified = result[result['combo_id'] == 'verified_combo'].iloc[0]
    unverified = result[result['combo_id'] == 'unverified_combo'].iloc[0]

    assert bool(verified['is_verified']) is True
    assert bool(unverified['is_verified']) is False
    assert verified['score'] > unverified['score']
    assert unverified['eligibility_reason'] == 'unverified_no_deep_metrics'


def test_composite_score_prefers_lower_turnover_when_other_metrics_tie():
    """Lower turnover should now help, not hurt, score."""
    lightweight = pd.DataFrame({
        'combo_id': ['high_turnover', 'low_turnover'],
        'recall': [0.6, 0.6],
        'cpr': [0.5, 0.5],
        'coverage': [0.5, 0.5],
        'lift': [1.5, 1.5],
    })

    deep = pd.DataFrame({
        'combo_id': ['high_turnover', 'low_turnover'],
        'uniqueness': [0.5, 0.5],
        'turnover': [50.0, 10.0],
        'oos_recall': [0.3, 0.3],
        'regime_stability': [0.7, 0.7],
        'net_pnl': [1.0, 1.0],
        'sharpe': [0.4, 0.4],
        'mdd': [-0.2, -0.2],
        'trade_count': [20, 20],
        'oos_unreliable': [False, False],
        'low_info': [False, False],
    })

    result = compute_composite_score(lightweight, deep)
    assert result.iloc[0]['combo_id'] == 'low_turnover'


def test_composite_score_applies_reliability_penalties():
    """Poor OOS reliability and low info should trigger explicit penalties."""
    lightweight = pd.DataFrame({
        'combo_id': ['clean_combo', 'penalized_combo'],
        'recall': [0.6, 0.6],
        'cpr': [0.5, 0.5],
        'coverage': [0.5, 0.5],
        'lift': [1.5, 1.5],
    })

    deep = pd.DataFrame({
        'combo_id': ['clean_combo', 'penalized_combo'],
        'uniqueness': [0.5, 0.5],
        'turnover': [20.0, 20.0],
        'oos_recall': [0.3, 0.0],
        'regime_stability': [0.7, 0.7],
        'net_pnl': [1.0, 1.0],
        'sharpe': [0.4, 0.4],
        'mdd': [-0.2, -0.2],
        'trade_count': [20, 5],
        'oos_unreliable': [False, True],
        'low_info': [False, True],
    })

    result = compute_composite_score(lightweight, deep)
    clean = result[result['combo_id'] == 'clean_combo'].iloc[0]
    penalized = result[result['combo_id'] == 'penalized_combo'].iloc[0]

    assert clean['score'] > penalized['score']
    assert penalized['eligibility_reason'] in {
        'oos_unreliable', 'low_info', 'low_trade_count', 'low_oos_recall'
    }


def test_get_top_candidates_prefers_verified_rows():
    """Top candidates should only include verified rows when available."""
    scored = pd.DataFrame({
        'combo_id': ['verified_a', 'verified_b', 'unverified_x'],
        'score': [0.8, 0.7, 10.0],
        'rank': [1, 2, 3],
        'is_verified': [True, True, False],
    })

    top2 = get_top_candidates(scored, top_n=2)

    assert top2['combo_id'].tolist() == ['verified_a', 'verified_b']
