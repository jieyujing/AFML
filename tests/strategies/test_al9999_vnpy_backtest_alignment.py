from pathlib import Path
import importlib
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_load_filter_first_selection_strict_missing_artifacts_raises(tmp_path, monkeypatch):
    module = importlib.import_module("strategies.AL9999.12_run_vnpy_cta_backtest")
    monkeypatch.setattr(module, "FEATURES_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        module.load_filter_first_selection(strict=True)


def test_compare_with_research_uses_filter_first_trade_ledger_and_parity_snapshot(tmp_path, monkeypatch):
    module = importlib.import_module("strategies.AL9999.12_run_vnpy_cta_backtest")
    monkeypatch.setattr(module, "FEATURES_DIR", str(tmp_path))

    pd.DataFrame(
        [
            {
                "selected_threshold": 0.51,
                "side_mode": "both_with_short_penalty",
                "baseline_oos_n": 74,
                "selected_oos_n": 55,
                "trade_shrinkage": 0.2567,
                "scheme_used": "mda_positive",
            }
        ]
    ).to_parquet(tmp_path / "filter_first_selection.parquet")
    pd.DataFrame(
        [
            {"threshold": 0.50, "oos_dsr": 0.90, "oos_sharpe": 7.2, "trade_shrinkage": 0.12},
            {"threshold": 0.51, "oos_dsr": 0.96, "oos_sharpe": 10.2, "trade_shrinkage": 0.26},
        ]
    ).to_parquet(tmp_path / "filter_first_threshold_report.parquet")
    pd.DataFrame({"Metric": ["DSR"], "Combined (OOS)": [0.96]}).to_parquet(tmp_path / "backtest_stats.parquet")
    pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-01-05 09:30:00", "2026-01-05 10:00:00"]),
            "exit_time": pd.to_datetime(["2026-01-05 09:40:00", "2026-01-05 10:10:00"]),
            "side": [1, -1],
            "net_pnl": [12.0, -2.0],
        }
    ).to_parquet(tmp_path / "filter_first_combined_trades.parquet")

    vnpy_trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-01-05 09:30:00", "2026-01-05 10:00:00"]),
            "exit_time": pd.to_datetime(["2026-01-05 09:40:00", "2026-01-05 10:10:00"]),
            "side": [1, -1],
            "net_pnl": [10.0, -1.0],
        }
    )
    vnpy_stats = {"total_trade_count": 2, "sharpe_ratio": 1.1}

    report = module.compare_with_research(
        vnpy_stats=vnpy_stats,
        vnpy_trades=vnpy_trades,
        mode="filter_first",
        strict_filter_first_artifacts=True,
    )

    assert report["parity_mode"] == "filter_first"
    assert report["research_trade_path"].endswith("filter_first_combined_trades.parquet")
    assert report["filter_first_selection"]["selected_threshold"] == 0.51
    assert report["filter_first_selection"]["side_mode"] == "both_with_short_penalty"
