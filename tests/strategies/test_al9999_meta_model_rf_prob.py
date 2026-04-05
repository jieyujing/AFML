import importlib.util
from pathlib import Path

import pandas as pd


def _load_meta_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "07_meta_model.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_meta_model", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_rf_probability_adds_rf_prob_feature():
    meta_module = _load_meta_module()
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame({"feat_rsi_14": [1.0, 2.0, 3.0]}, index=idx)
    rf = pd.DataFrame({"y_prob": [0.7, 0.4, 0.8]}, index=idx)

    merged = meta_module.merge_rf_probability(features, rf)
    assert "rf_prob" in merged.columns
    assert merged["rf_prob"].tolist() == [0.7, 0.4, 0.8]


def test_merge_rf_probability_fills_missing_with_half():
    meta_module = _load_meta_module()
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame({"feat_rsi_14": [1.0, 2.0, 3.0]}, index=idx)
    rf = pd.DataFrame({"y_prob": [0.7]}, index=idx[:1])

    merged = meta_module.merge_rf_probability(features, rf)
    assert merged["rf_prob"].tolist() == [0.7, 0.5, 0.5]
