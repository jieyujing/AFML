import importlib.util
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


def test_build_signal_frame_contains_required_columns():
    rf_module = _load_rf_module()
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    result = rf_module.build_signal_frame(
        index=idx,
        trend_side=pd.Series([1, -1, 1], index=idx),
        probs=pd.Series([0.8, 0.3, 0.55], index=idx),
        t1=pd.Series(idx + pd.Timedelta(days=2), index=idx),
        is_holdout=pd.Series([False, False, True], index=idx),
    )

    assert list(result.columns) == [
        "side",
        "y_prob",
        "y_pred",
        "trend_side",
        "t1",
        "is_holdout",
    ]
    assert set(result["side"].unique()) == {-1, 1}
    assert result["y_prob"].between(0, 1).all()


def test_build_model_uses_avg_u_for_bagging_max_samples():
    rf_module = _load_rf_module()
    model = rf_module.build_model(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_samples=0.37,
    )
    assert model.n_estimators == 200
    assert model.max_samples == 0.37
