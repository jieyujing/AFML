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


def test_min_hold_blocks_early_reverse():
    module = _load_backtest_utils_module()
    bars = _make_bars()

    signals = pd.DataFrame(
        {
            "side": [1, -1],
            "touch_idx": [6, 7],
            "meta_pred": [1, 1],
            "meta_prob": [0.70, 0.70],
        },
        index=[bars.index[0], bars.index[1]],
    )

    no_guard = module.rolling_backtest(signals, bars, use_meta_filter=True)
    with_guard = module.rolling_backtest(
        signals,
        bars,
        use_meta_filter=True,
        guard_enabled=True,
        min_hold_bars=3,
    )

    assert (no_guard["exit_reason"] == "reverse_signal").sum() == 1
    assert (with_guard["exit_reason"] == "reverse_signal").sum() == 0


def test_cooldown_blocks_reentry_after_close():
    module = _load_backtest_utils_module()
    bars = _make_bars()

    signals = pd.DataFrame(
        {
            "side": [1, -1],
            "touch_idx": [1, 7],
            "meta_pred": [1, 1],
            "meta_prob": [0.70, 0.70],
        },
        index=[bars.index[0], bars.index[2]],
    )

    no_guard = module.rolling_backtest(signals, bars, use_meta_filter=True)
    with_cooldown = module.rolling_backtest(
        signals,
        bars,
        use_meta_filter=True,
        guard_enabled=True,
        cooldown_bars=2,
    )

    assert len(no_guard) == 2
    assert len(with_cooldown) == 1


def test_reverse_confirmation_delta_requires_higher_prob():
    module = _load_backtest_utils_module()
    bars = _make_bars()

    low_prob_signals = pd.DataFrame(
        {
            "side": [1, -1],
            "touch_idx": [6, 7],
            "meta_pred": [1, 1],
            "meta_prob": [0.70, 0.55],
        },
        index=[bars.index[0], bars.index[1]],
    )
    high_prob_signals = low_prob_signals.copy()
    high_prob_signals.loc[high_prob_signals.index[1], "meta_prob"] = 0.65

    low_prob_result = module.rolling_backtest(
        low_prob_signals,
        bars,
        use_meta_filter=True,
        guard_enabled=True,
        reverse_confirmation_delta=0.10,
        entry_threshold=0.50,
    )
    high_prob_result = module.rolling_backtest(
        high_prob_signals,
        bars,
        use_meta_filter=True,
        guard_enabled=True,
        reverse_confirmation_delta=0.10,
        entry_threshold=0.50,
    )

    assert (low_prob_result["exit_reason"] == "reverse_signal").sum() == 0
    assert (high_prob_result["exit_reason"] == "reverse_signal").sum() == 1
