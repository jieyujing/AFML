import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_primary_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "04_ma_primary_model.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_ma_primary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_rf_signals_reads_rf_signal_file(tmp_path):
    primary_module = _load_primary_module()
    bars_idx = pd.date_range("2024-01-01", periods=5, freq="D")
    bars = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=bars_idx)
    events = pd.DataFrame({"price": [1.1, 2.2, 4.4]}, index=bars_idx[[0, 1, 3]])
    rf_df = pd.DataFrame(
        {
            "side": [1, -1, 1],
            "y_prob": [0.7, 0.2, 0.8],
        },
        index=events.index,
    )
    path = tmp_path / "rf_primary_signals.parquet"
    rf_df.to_parquet(path)

    signals = primary_module.generate_rf_signals(bars, events, str(path))

    assert list(signals["side"]) == [1, -1, 1]
    assert list(signals["idx"]) == [0, 1, 3]


def test_generate_rf_signals_raises_when_file_missing(tmp_path):
    primary_module = _load_primary_module()
    bars = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    events = pd.DataFrame({"price": [1.0]}, index=bars.index)
    with pytest.raises(FileNotFoundError):
        primary_module.generate_rf_signals(bars, events, str(tmp_path / "missing.parquet"))
