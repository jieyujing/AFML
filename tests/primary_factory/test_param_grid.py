"""Tests for parameter grid generator module."""

import importlib.util
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the module using importlib to avoid packaging issues
_module_path = (
    PROJECT_ROOT
    / "strategies"
    / "AL9999"
    / "primary_factory"
    / "param_grid.py"
)
_spec = importlib.util.spec_from_file_location("param_grid", _module_path)
_param_grid_module = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_param_grid_module)  # type: ignore
expand_deep_param_grid = _param_grid_module.expand_deep_param_grid
generate_lightweight_param_grid = _param_grid_module.generate_lightweight_param_grid


def test_generate_lightweight_param_grid():
    """Test lightweight grid generation without vertical-bar expansion."""
    grid = generate_lightweight_param_grid(
        cusum_rates=[0.05, 0.10],
        fast_windows=[5, 10],
        slow_windows=[20, 30],
    )

    # 过滤后: 2 rates × (5<20, 5<30, 10<20, 10<30) = 8
    assert len(grid) == 8
    # Verify fast < slow constraint
    assert all(grid['fast'] < grid['slow'])
    # Verify combo_id uniqueness
    assert grid['combo_id'].nunique() == 8
    # Verify expected pairs (all 4 pairs are valid)
    expected_pairs = {(5, 20), (5, 30), (10, 20), (10, 30)}
    actual_pairs = set(zip(grid['fast'], grid['slow']))
    assert actual_pairs == expected_pairs
    assert "vertical_bars" not in grid.columns


def test_generate_lightweight_param_grid_columns():
    """Test that lightweight output DataFrame has correct columns."""
    grid = generate_lightweight_param_grid(
        cusum_rates=[0.05],
        fast_windows=[5],
        slow_windows=[20],
    )

    expected_columns = ['combo_id', 'cusum_rate', 'fast', 'slow']
    assert list(grid.columns) == expected_columns


def test_generate_lightweight_param_grid_empty():
    """Test empty result when no valid combinations."""
    grid = generate_lightweight_param_grid(
        cusum_rates=[0.05],
        fast_windows=[20, 30],  # All >= slow
        slow_windows=[10],
    )

    assert len(grid) == 0
    assert list(grid.columns) == ['combo_id', 'cusum_rate', 'fast', 'slow']


def test_generate_lightweight_param_grid_sorting():
    """Test that results are sorted by rate, fast, slow."""
    grid = generate_lightweight_param_grid(
        cusum_rates=[0.15, 0.05],  # Unsorted input
        fast_windows=[15, 5],  # Unsorted input
        slow_windows=[30, 20],  # Unsorted input
    )

    # Verify sorting by cusum_rate first
    rates = grid['cusum_rate'].tolist()
    assert rates == sorted(rates)

    # For same rate, verify sorting by fast
    for rate in grid['cusum_rate'].unique():
        subset = grid[grid['cusum_rate'] == rate]
        fasts = subset['fast'].tolist()
        assert fasts == sorted(fasts)


def test_expand_deep_param_grid():
    """Test deep grid expansion from top lightweight combos."""
    lightweight = generate_lightweight_param_grid(
        cusum_rates=[0.05],
        fast_windows=[5],
        slow_windows=[20],
    )

    deep_grid = expand_deep_param_grid(
        lightweight_combos=lightweight,
        vertical_bars=[10, 20, 30],
    )

    assert len(deep_grid) == 3
    assert set(deep_grid['vertical_bars']) == {10, 20, 30}
    assert deep_grid['base_combo_id'].nunique() == 1
    assert all(deep_grid['combo_id'].str.contains("_vb="))
