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

    # With default weights (recall=0.45, lift=0.20), high_recall should win
    result = compute_composite_score(lightweight, deep)

    assert result['rank'].iloc[0] == 1  # high_recall should be rank 1
    assert result['combo_id'].iloc[0] == 'high_recall'