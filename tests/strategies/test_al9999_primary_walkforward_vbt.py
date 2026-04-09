import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "05b_primary_walkforward_vbt.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_primary_walkforward_vbt", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_candidate_comparison_table_ranks_by_median_sharpe_then_worst_case():
    module = _load_module()

    wf_results = pd.DataFrame([
        {"combo_id": "a", "fold_id": 1, "sharpe": 0.5, "total_pnl": 10.0, "max_dd": -0.1},
        {"combo_id": "a", "fold_id": 2, "sharpe": 0.3, "total_pnl": 8.0, "max_dd": -0.2},
        {"combo_id": "b", "fold_id": 1, "sharpe": 0.4, "total_pnl": 12.0, "max_dd": -0.1},
        {"combo_id": "b", "fold_id": 2, "sharpe": 0.4, "total_pnl": 12.0, "max_dd": -0.1},
    ])
    cost_results = pd.DataFrame([
        {"combo_id": "a", "commission_mult": 1.0, "slippage_mult": 1.0, "sharpe": 0.2, "total_pnl": 5.0},
        {"combo_id": "a", "commission_mult": 2.0, "slippage_mult": 3.0, "sharpe": 0.0, "total_pnl": 1.0},
        {"combo_id": "b", "commission_mult": 1.0, "slippage_mult": 1.0, "sharpe": 0.3, "total_pnl": 6.0},
        {"combo_id": "b", "commission_mult": 2.0, "slippage_mult": 3.0, "sharpe": 0.2, "total_pnl": 2.0},
    ])

    summary = module.build_candidate_comparison_table(wf_results, cost_results)

    assert summary.iloc[0]["combo_id"] == "b"
    assert "wf_median_sharpe" in summary.columns
    assert "cost_worst_sharpe" in summary.columns
    assert "final_rank" in summary.columns


def test_generate_walk_forward_slices_produces_non_overlapping_test_windows():
    module = _load_module()

    index = pd.date_range("2024-01-01", periods=120, freq="h")
    slices = module.generate_walk_forward_slices(index=index, n_splits=3, train_ratio=0.6, test_ratio=0.2)

    assert len(slices) == 3
    test_windows = [(slc["test_start"], slc["test_end"]) for slc in slices]
    assert test_windows[0][1] <= test_windows[1][0]
    assert test_windows[1][1] <= test_windows[2][0]


def test_build_event_driven_side_from_events_changes_with_event_density():
    module = _load_module()

    bars = pd.DataFrame(
        {"close": [10, 11, 12, 11, 10, 9, 10, 11]},
        index=pd.date_range("2024-01-01", periods=8, freq="h"),
    )

    sparse = module.build_event_driven_side_from_events(
        bars=bars,
        event_indices=[1, 5],
        fast=2,
        slow=3,
    )
    dense = module.build_event_driven_side_from_events(
        bars=bars,
        event_indices=[1, 3, 5, 7],
        fast=2,
        slow=3,
    )

    assert sparse.nunique() <= dense.nunique()
    assert sparse.ne(dense).any()
