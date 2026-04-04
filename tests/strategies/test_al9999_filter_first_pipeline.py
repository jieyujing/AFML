import importlib.util
from pathlib import Path
import sys

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_combined_backtest_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "10_combined_backtest.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_combined_backtest", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_meta_signal_paths_prefers_selected_scheme(tmp_path):
    module = _load_combined_backtest_module()

    models_dir = tmp_path / "models"
    features_dir = tmp_path / "features"
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # baseline files
    (models_dir / "meta_oof_signals.parquet").touch()
    (models_dir / "meta_holdout_signals.parquet").touch()
    # preferred files
    (models_dir / "meta_oof_signals_mda_positive.parquet").touch()
    (models_dir / "meta_holdout_signals_mda_positive.parquet").touch()

    (features_dir / "selected_features.json").write_text(
        '{"best_scheme":"mda_positive","features":[]}',
        encoding="utf-8",
    )

    module.FEATURES_DIR = str(features_dir)
    oof_path, holdout_path, scheme = module._resolve_meta_signal_paths(str(models_dir))

    assert scheme == "mda_positive"
    assert oof_path.endswith("meta_oof_signals_mda_positive.parquet")
    assert holdout_path.endswith("meta_holdout_signals_mda_positive.parquet")


def test_threshold_report_generation_and_feasible_selection():
    module = _load_combined_backtest_module()

    idx = pd.date_range("2026-01-01 09:00:00", periods=12, freq="min")
    bars = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(12)],
            "high": [101.0 + i for i in range(12)],
            "low": [99.0 + i for i in range(12)],
            "close": [100.0 + i for i in range(12)],
            "volume": [1] * 12,
        },
        index=idx,
    )
    signals = pd.DataFrame(
        {
            "side": [1, 1, -1, 1, -1, 1],
            "touch_idx": [4, 6, 7, 8, 9, 10],
            "meta_prob": [0.52, 0.63, 0.51, 0.66, 0.58, 0.70],
        },
        index=[idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]],
    )
    signals["meta_pred"] = 1

    report, best, _baseline_oos_n = module._build_filter_first_threshold_report(
        signals=signals,
        bars=bars,
        oos_start=idx[6],
        threshold_grid=[0.50, 0.60],
        side_mode="both",
        short_penalty_delta=0.0,
        guard_cfg={"enabled": True, "min_hold_bars": 1, "cooldown_bars": 0, "reverse_confirmation_delta": 0.0},
        baseline_threshold=0.50,
        shrinkage_min=0.10,
        shrinkage_max=0.95,
    )

    assert "trade_shrinkage" in report.columns
    assert set(report["threshold"].tolist()) == {0.5, 0.6}
    assert best is not None
    assert 0.10 <= best["trade_shrinkage"] <= 0.95


def test_main_persists_threshold_report_and_selected_outputs(tmp_path, monkeypatch):
    module = _load_combined_backtest_module()

    features_dir = tmp_path / "features"
    figures_dir = tmp_path / "figures"
    bars_dir = tmp_path / "bars"
    models_dir = tmp_path / "models"
    features_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # tbm and bars minimal fixtures
    idx = pd.date_range("2026-01-01 09:00:00", periods=10, freq="min")
    tbm = pd.DataFrame(
        {
            "side": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
            "touch_idx": [3, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            "pnl": [10.0] * 10,
            "entry_price": [100.0] * 10,
            "exit_price": [101.0] * 10,
        },
        index=idx,
    )
    tbm.to_parquet(features_dir / "tbm_results.parquet")
    bars = pd.DataFrame(
        {
            "open": np.linspace(100.0, 109.0, 10),
            "high": np.linspace(101.0, 110.0, 10),
            "low": np.linspace(99.0, 108.0, 10),
            "close": np.linspace(100.0, 109.0, 10),
            "volume": [1] * 10,
        },
        index=idx,
    )
    bars.to_parquet(bars_dir / "dollar_bars_target4.parquet")

    module.FEATURES_DIR = str(features_dir)
    module.FIGURES_DIR = str(figures_dir)
    module.BARS_DIR = str(bars_dir)
    module.META_MODEL_CONFIG = {"precision_threshold": 0.5}
    module.FILTER_FIRST_CONFIG = {
        "threshold_grid": [0.5, 0.6],
        "shrinkage_min": 0.0,
        "shrinkage_max": 1.0,
        "execution_guard": {
            "enabled": False,
            "min_hold_bars": 0,
            "cooldown_bars": 0,
            "reverse_confirmation_delta": 0.0,
        },
        "short_penalty_delta": 0.0,
        "side_mode": "both",
    }

    def _fake_load_honest_meta_signals(models_dir: str, precision_threshold: float):
        meta = pd.DataFrame(
            {
                "meta_pred": [1] * len(idx),
                "meta_prob": [0.55, 0.62, 0.52, 0.65, 0.51, 0.63, 0.57, 0.64, 0.58, 0.66],
            },
            index=idx,
        )
        return meta, idx[5], "test_scheme"

    monkeypatch.setattr(module, "load_honest_meta_signals", _fake_load_honest_meta_signals)

    # Skip heavy plotting side effects in this integration test.
    monkeypatch.setattr(module.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)

    module.main()

    assert (features_dir / "filter_first_threshold_report.parquet").exists()
    assert (features_dir / "filter_first_selection.parquet").exists()
    assert (features_dir / "filter_first_primary_trades.parquet").exists()
    assert (features_dir / "filter_first_combined_trades.parquet").exists()
