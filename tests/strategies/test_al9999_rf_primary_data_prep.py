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


def test_select_feature_columns_by_prefix():
    rf_module = _load_rf_module()
    cols = [
        "feat_rsi_14",
        "feat_roc_5",
        "feat_microstructure_20",
        "raw_value",
    ]
    selected = rf_module.select_feature_columns(cols, ["feat_rsi_", "feat_roc_"])
    assert selected == ["feat_rsi_14", "feat_roc_5"]


def test_prepare_rf_dataset_filters_side_zero_and_low_t_value():
    rf_module = _load_rf_module()
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {
            "feat_rsi_14": [1.0, 2.0, 3.0, 4.0],
            "feat_roc_5": [0.1, 0.2, 0.3, 0.4],
            "feat_microstructure_20": [5, 6, 7, 8],
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "side": [1, 0, -1, 1],
            "t_value": [2.0, 3.0, 0.2, -1.5],
            "t1": idx + pd.Timedelta(days=2),
        },
        index=idx,
    )

    X, y, t1, meta = rf_module.prepare_rf_dataset(
        features=features,
        labels=labels,
        feature_prefixes=["feat_rsi_", "feat_roc_"],
        min_t_value=1.0,
    )

    assert list(X.columns) == ["feat_rsi_14", "feat_roc_5"]
    assert list(y.tolist()) == [1, 1]
    assert set(meta["trend_side"].tolist()) == {1}
    assert (meta["abs_t_value"] >= 1.0).all()
    assert len(t1) == 2
