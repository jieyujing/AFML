import importlib.util
import math
from pathlib import Path

import pandas as pd


def _load_rf_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "04_rf_primary_model.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_rf_primary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_avg_uniqueness_from_t1_returns_fraction():
    rf_module = _load_rf_module()
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    t1 = pd.Series(
        [
            idx[1],
            idx[2],
            idx[3],
            idx[4],
            idx[4],
        ],
        index=idx,
    )

    avg_u = rf_module.compute_avg_uniqueness_from_t1(idx, t1)
    assert 0.0 < avg_u <= 1.0
    assert math.isfinite(avg_u)


def test_split_train_holdout_keeps_last_months_as_holdout():
    rf_module = _load_rf_module()
    idx = pd.date_range("2024-01-01", periods=16, freq="30D")
    X = pd.DataFrame({"feat_rsi_14": range(16)}, index=idx)
    y = pd.Series([1, -1] * 8, index=idx)
    t1 = pd.Series(idx + pd.Timedelta(days=5), index=idx)

    train_pack, holdout_pack, holdout_start = rf_module.split_train_holdout(
        X, y, t1, holdout_months=6
    )

    X_train, _, _, _ = train_pack
    X_holdout, _, _, _ = holdout_pack
    assert len(X_train) > 0
    assert len(X_holdout) > 0
    assert X_train.index.max() < holdout_start
    assert X_holdout.index.min() >= holdout_start
