import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "strategies" / "cta" / "01_market_timing_selection.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location("cta_market_timing_selection", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_minute_data(n: int = 2400) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 09:00:00", periods=n, freq="min")
    x = np.linspace(0.0, 20.0, n)
    close = 18000.0 + np.cumsum(np.sin(x) * 0.8 + np.cos(x * 0.35) * 0.4 + 0.05)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    volume = 150.0 + np.abs(np.sin(x * 0.8) * 80.0)
    open_interest = 100000.0 + np.linspace(0.0, 3000.0, n)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "open_interest": open_interest,
        }
    )


def test_build_dollar_bars_from_minute_generates_bars():
    module = _load_module()
    minute_df = _build_minute_data()

    bars = module.build_dollar_bars_from_minute(
        minute_df=minute_df,
        target_daily_bars=20,
        ewma_span=10,
        contract_multiplier=5.0,
    )

    assert len(bars) > 0
    assert bars.index.is_monotonic_increasing
    assert {"open", "high", "low", "close", "volume", "dollar_volume", "n_ticks"}.issubset(
        bars.columns
    )
    assert bars["dollar_volume"].min() > 0


def test_calibrate_cusum_rates_hits_target_rate():
    module = _load_module()
    minute_df = _build_minute_data(3000)
    bars = module.build_dollar_bars_from_minute(
        minute_df=minute_df,
        target_daily_bars=15,
        ewma_span=15,
        contract_multiplier=5.0,
    )

    result = module.calibrate_cusum_rates_on_bars(
        bars=bars,
        target_rates=[0.10],
        k_min=1e-6,
        k_max=0.05,
        tol=0.02,
        max_iter=60,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["k"] > 0
    assert 0 < row["actual_rate"] < 1
    assert abs(row["actual_rate"] - row["rate"]) <= 0.05


def test_compute_market_timing_metrics_has_expected_fields():
    module = _load_module()
    minute_df = _build_minute_data(3200)
    bars = module.build_dollar_bars_from_minute(
        minute_df=minute_df,
        target_daily_bars=18,
        ewma_span=10,
        contract_multiplier=5.0,
    )
    calibration = module.calibrate_cusum_rates_on_bars(
        bars=bars,
        target_rates=[0.05],
        k_min=1e-6,
        k_max=0.05,
        tol=0.03,
        max_iter=60,
    )
    k = float(calibration.loc[0, "k"])
    event_indices = module.compute_event_indices_from_bars(bars=bars, k=k)

    metrics = module.compute_market_timing_metrics(
        bars=bars,
        event_indices=event_indices,
        overlap_window_bars=20,
        eval_horizons=[5, 10],
        random_seed=42,
    )

    expected_keys = {
        "n_bars",
        "n_events",
        "events_per_1000_bars",
        "mean_bars_between_events",
        "median_bars_between_events",
        "event_overlap_ratio",
        "event_absret_adv_h5",
        "event_absret_adv_h10",
        "event_dir_adv_h5",
        "event_dir_adv_h10",
    }
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["event_overlap_ratio"] <= 1.0
