import importlib.util
from pathlib import Path

import pandas as pd


def _load_threshold_optimizer_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "threshold_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_threshold_optimizer", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_best_threshold_respects_shrinkage_and_ranks_by_oos_dsr():
    module = _load_threshold_optimizer_module()
    df = pd.DataFrame(
        {
            "threshold": [0.50, 0.51, 0.52],
            "oos_dsr": [0.60, 0.70, 0.90],
            "oos_sharpe": [1.0, 1.2, 0.9],
            "trade_shrinkage": [0.10, 0.20, 0.40],
        }
    )

    best = module.select_best_threshold(df, shrinkage_min=0.15, shrinkage_max=0.30)
    assert best is not None
    assert best["threshold"] == 0.51
