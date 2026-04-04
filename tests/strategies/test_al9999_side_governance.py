import importlib.util
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_backtest_utils_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "backtest_utils.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_backtest_utils", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _load_dsr_validation_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "08_dsr_validation.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_dsr_validation", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_bars(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:00:00", periods=n, freq="min")
    price = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": price,
            "high": [p + 1 for p in price],
            "low": [p - 1 for p in price],
            "close": price,
            "volume": [1] * n,
        },
        index=idx,
    )


def test_side_mode_long_only_filters_short_entries():
    module = _load_backtest_utils_module()
    bars = _make_bars()

    signals = pd.DataFrame(
        {
            "side": [1, -1],
            "touch_idx": [4, 7],
            "meta_pred": [1, 1],
            "meta_prob": [0.70, 0.70],
        },
        index=[bars.index[0], bars.index[1]],
    )

    both_trades = module.rolling_backtest(signals, bars, use_meta_filter=True, side_mode="both")
    long_only_trades = module.rolling_backtest(
        signals,
        bars,
        use_meta_filter=True,
        side_mode="long_only",
    )

    assert len(both_trades) >= 1
    assert len(long_only_trades) >= 1
    assert (long_only_trades["side"] == -1).sum() == 0


def test_both_with_short_penalty_uses_higher_threshold_for_short():
    module = _load_combined_backtest_module()
    signals = pd.DataFrame(
        {
            "side": [1, -1, -1],
            "meta_prob": [0.55, 0.53, 0.57],
        }
    )

    out = module._apply_threshold_to_signals(
        signals=signals,
        threshold=0.52,
        side_mode="both_with_short_penalty",
        short_penalty_delta=0.03,
    )

    assert out.loc[0, "meta_pred"] == 1
    assert out.loc[1, "meta_pred"] == 0
    assert out.loc[2, "meta_pred"] == 1


def test_dsr_validation_filter_first_diagnostics_has_required_fields():
    module = _load_dsr_validation_module()
    primary = pd.DataFrame({"net_pnl": [1.0, 2.0, -1.0], "side": [1, -1, 1]})
    combined = pd.DataFrame({"net_pnl": [2.0, -0.5], "side": [1, -1]})

    diag = module.compute_filter_first_diagnostics(
        primary_trades=primary,
        combined_trades=combined,
        side_mode="both_with_short_penalty",
        shrinkage_min=0.10,
        shrinkage_max=0.60,
    )

    assert diag["side_mode"] == "both_with_short_penalty"
    assert "short_contribution_ratio" in diag
    assert "full_trade_shrinkage" in diag
    assert "shrinkage_pass" in diag
