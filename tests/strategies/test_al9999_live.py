from pathlib import Path
import sys
from types import SimpleNamespace
import importlib

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategies.AL9999.live_runtime import (
    Al9999LiveRuntime,
    ContractResolver,
    DollarBar,
    LiveSignalModel,
    PositionState,
    SignalSnapshot,
)
from strategies.AL9999.vnpy_strategy import Al9999CtaStrategy, BarData


class DummyModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.full((len(X), 2), 0.0)
        probs[:, 1] = 0.8
        probs[:, 0] = 0.2
        return probs


def _minute_bar(dt, open_, high, low, close, volume, open_interest=1000):
    return {
        "datetime": pd.Timestamp(dt),
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
        "open_interest": float(open_interest),
    }


def _runtime(model_path=None):
    feature_config = {
        "momentum": {"enabled": True, "rsi_windows": [2], "roc_periods": [1], "stoch_k_length": 2},
        "volatility": {"enabled": True, "ewm_spans": [2], "hl_windows": [2]},
        "trend": {"enabled": False},
        "reversion": {"enabled": False},
        "time_cues": {"enabled": False},
        "open_interest": {"enabled": True, "windows": [2]},
        "theil_imbalance": {"enabled": False},
        "microstructure": {"enabled": False},
        "cross_ma": {"enabled": False},
        "entropy": {"enabled": False},
        "serial_corr": {"enabled": False},
        "structural_break": {"enabled": False},
        "trend_scan": {"enabled": False},
    }
    return Al9999LiveRuntime(
        target_daily_bars=2,
        ewma_span=2,
        fracdiff_threshold=1e-4,
        fracdiff_d=0.0,
        cusum_window=2,
        cusum_multiplier=0.0,
        primary_span=2,
        feature_config=feature_config,
        tbm_config={
            "target_ret_col": "feat_ewm_vol_2",
            "profit_loss_barriers": (1.5, 1.0),
            "vertical_barrier_bars": 2,
            "min_ret": 0.0,
        },
        model=LiveSignalModel(model_path=model_path),
        contract_resolver=ContractResolver({"AL9999": "AL2506.SHFE"}),
    )


def test_contract_resolver_supports_manual_override():
    resolver = ContractResolver({"AL9999": "AL2506.SHFE"}, manual_overrides={"AL9999": "AL2507.SHFE"})

    assert resolver.resolve_research_symbol("AL9999") == "AL2507.SHFE"


def test_live_signal_model_enters_safe_mode_when_model_missing(tmp_path):
    model = LiveSignalModel(model_path=tmp_path / "missing.pkl")

    pred = model.predict(pd.DataFrame([{"feat_a": 1.0}]))

    assert pred.meta_pred == 0
    assert pred.meta_prob == 0.0
    assert pred.safe_mode is True


def test_runtime_emits_signal_snapshot_and_round_trips_state(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    runtime = _runtime(model_path=model_path)

    rows = [
        _minute_bar("2026-01-05 09:00:00", 100, 101, 99, 100, 800_000),
        _minute_bar("2026-01-05 09:01:00", 100, 102, 99, 101, 900_000),
        _minute_bar("2026-01-05 09:02:00", 101, 103, 100, 102, 1_100_000),
        _minute_bar("2026-01-05 09:03:00", 102, 104, 101, 103, 1_200_000),
        _minute_bar("2026-01-05 09:04:00", 103, 105, 102, 104, 1_300_000),
        _minute_bar("2026-01-05 09:05:00", 104, 106, 103, 105, 1_300_000),
    ]

    latest = None
    for row in rows:
        new_bars = runtime.update_minute_bar(row)
        for dollar_bar in new_bars:
            latest = runtime.on_dollar_bar(dollar_bar)

    assert isinstance(latest, SignalSnapshot)
    assert latest.entry_allowed is True
    assert latest.meta_pred == 1
    assert latest.live_symbol == "AL2506.SHFE"
    assert latest.target_tp_price > latest.bar_close
    assert latest.target_sl_price < latest.bar_close

    state_path = tmp_path / "state.json"
    runtime.save_state(state_path)
    restored = _runtime(model_path=model_path)
    restored.load_state(state_path)

    assert restored.last_signal is not None
    assert restored.last_signal.meta_pred == latest.meta_pred
    assert restored.last_signal.live_symbol == latest.live_symbol


def test_runtime_handles_short_history_without_cusum_crash(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    runtime = _runtime(model_path=model_path)
    runtime.cusum_multiplier = 3.0

    rows = [
        _minute_bar("2026-01-05 09:00:00", 100, 101, 99, 100, 800_000),
        _minute_bar("2026-01-05 09:01:00", 100, 102, 99, 101, 900_000),
        _minute_bar("2026-01-05 09:02:00", 101, 103, 100, 102, 1_100_000),
    ]

    latest = None
    for row in rows:
        for dollar_bar in runtime.update_minute_bar(row):
            latest = runtime.on_dollar_bar(dollar_bar)

    assert latest is None


def test_runtime_replay_accepts_bars_without_bar_index(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    runtime = _runtime(model_path=model_path)

    runtime.dollar_bars = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2026-01-05")] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_100.0, 1_200.0],
            "dollar_volume": [505_000.0, 561_000.0, 618_000.0],
            "open_interest": [1_000.0, 1_002.0, 1_005.0],
            "n_ticks": [10, 11, 12],
        },
        index=pd.DatetimeIndex(
            [
                "2026-01-05 09:00:00",
                "2026-01-05 09:10:00",
                "2026-01-05 09:20:00",
            ],
            name="timestamp",
        ),
    )

    bar = DollarBar(
        timestamp=pd.Timestamp("2026-01-05 09:20:00"),
        trading_date=pd.Timestamp("2026-01-05"),
        open=102.0,
        high=104.0,
        low=101.0,
        close=103.0,
        volume=1_200.0,
        dollar_volume=618_000.0,
        open_interest=1_005.0,
        n_ticks=12,
        bar_index=2,
    )

    snapshot = runtime.on_dollar_bar(bar)

    assert snapshot is not None
    assert snapshot.bar_index == 2


def test_runtime_append_dollar_bar_preserves_prebuilt_bar_fields(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    runtime = _runtime(model_path=model_path)

    appended = runtime.append_dollar_bar(
        {
            "timestamp": pd.Timestamp("2026-01-05 09:30:00"),
            "trading_date": pd.Timestamp("2026-01-05"),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 104.0,
            "volume": 1000.0,
            "dollar_volume": 520000.0,
            "open_interest": 1200.0,
            "n_ticks": 12,
            "bar_index": 7,
        }
    )

    assert appended.bar_index == 7
    assert runtime.dollar_bars.loc[pd.Timestamp("2026-01-05 09:30:00"), "dollar_volume"] == 520000.0


def test_cta_strategy_supports_prebuilt_dollar_bar_mode(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    strategy = Al9999CtaStrategy(
        cta_engine=None,
        strategy_name="al_live",
        vt_symbol="AL2506.SHFE",
        setting={
            "fixed_size": 1,
            "model_path": str(model_path),
            "research_symbol": "AL9999",
            "symbol_mapping": {"AL9999": "AL2506.SHFE"},
            "emit_orders": False,
            "input_bar_mode": "dollar",
        },
    )

    strategy.on_bar(
        BarData(
            datetime=pd.Timestamp("2026-01-05 09:30:00"),
            open_price=100.0,
            high_price=104.0,
            low_price=99.0,
            close_price=103.0,
            volume=10,
            open_interest=1000.0,
        )
    )

    assert len(strategy.runtime.dollar_bars) == 1


def test_cta_strategy_exits_on_take_profit(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    strategy = Al9999CtaStrategy(
        cta_engine=None,
        strategy_name="al_live",
        vt_symbol="AL2506.SHFE",
        setting={
            "fixed_size": 1,
            "model_path": str(model_path),
            "research_symbol": "AL9999",
            "symbol_mapping": {"AL9999": "AL2506.SHFE"},
            "emit_orders": False,
        },
    )

    strategy.position_state = PositionState(
        direction=1,
        entry_price=100.0,
        entry_bar_idx=3,
        tp_price=103.0,
        sl_price=98.0,
        expire_bar_idx=6,
        entry_signal_ts=pd.Timestamp("2026-01-05 09:03:00"),
    )
    strategy.runtime_state["dollar_bar_count"] = 4

    bar = BarData(
        datetime=pd.Timestamp("2026-01-05 09:04:00"),
        open_price=101.0,
        high_price=103.5,
        low_price=100.5,
        close_price=103.0,
        volume=10,
    )

    reason = strategy.evaluate_exit(bar)

    assert reason == "take_profit"
    assert strategy.position_state is None
    assert strategy.runtime_state["last_exit_reason"] == "take_profit"


def test_cta_strategy_flips_on_opposite_signal(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    joblib.dump(DummyModel(), model_path)
    strategy = Al9999CtaStrategy(
        cta_engine=None,
        strategy_name="al_live",
        vt_symbol="AL2506.SHFE",
        setting={
            "fixed_size": 1,
            "model_path": str(model_path),
            "research_symbol": "AL9999",
            "symbol_mapping": {"AL9999": "AL2506.SHFE"},
            "emit_orders": False,
        },
    )
    strategy.position_state = PositionState(
        direction=1,
        entry_price=100.0,
        entry_bar_idx=2,
        tp_price=103.0,
        sl_price=98.0,
        expire_bar_idx=5,
        entry_signal_ts=pd.Timestamp("2026-01-05 09:02:00"),
    )
    strategy.runtime_state["dollar_bar_count"] = 3

    snapshot = SignalSnapshot(
        timestamp=pd.Timestamp("2026-01-05 09:03:00"),
        bar_index=3,
        primary_side=-1,
        meta_prob=0.8,
        meta_pred=1,
        entry_allowed=True,
        target_tp_price=95.0,
        target_sl_price=102.0,
        expire_bar_idx=5,
        bar_close=99.0,
        feature_row={"feat_ewm_vol_2": 0.02},
        live_symbol="AL2506.SHFE",
        safe_mode=False,
        diagnostics={"event_detected": True},
    )

    strategy.process_signal(snapshot)

    assert strategy.position_state is not None
    assert strategy.position_state.direction == -1
    assert strategy.runtime_state["last_exit_reason"] == "signal_flip"


def test_cta_strategy_can_replay_research_signals(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    tbm_path = tmp_path / "tbm.parquet"
    features_path = tmp_path / "events_features.parquet"
    joblib.dump(DummyModel(), model_path)

    index = pd.DatetimeIndex(["2026-01-05 09:30:00", "2026-01-05 09:32:00"], name="timestamp")
    pd.DataFrame(
        {
            "label": [0, 0],
            "touch_idx": [1, 3],
            "ret": [0.0, 0.0],
            "max_rb_ratio": [1.0, 1.0],
            "touch_type": ["upper", "vertical"],
            "side": [1, -1],
            "entry_price": [100.0, 98.0],
            "target": [0.01, 0.01],
            "exit_price": [101.0, 97.0],
            "exit_ts": [pd.Timestamp("2026-01-05 09:31:00"), pd.Timestamp("2026-01-05 09:33:00")],
            "pnl": [1.0, 1.0],
        },
        index=index,
    ).to_parquet(tbm_path)
    pd.DataFrame({"feat_demo": [0.25, 0.75]}, index=index).to_parquet(features_path)

    strategy = Al9999CtaStrategy(
        cta_engine=None,
        strategy_name="al_replay",
        vt_symbol="AL2506.SHFE",
        setting={
            "fixed_size": 1,
            "model_path": str(model_path),
            "research_symbol": "AL9999",
            "symbol_mapping": {"AL9999": "AL2506.SHFE"},
            "emit_orders": False,
            "input_bar_mode": "dollar",
            "replay_tbm_path": str(tbm_path),
            "replay_features_path": str(features_path),
        },
    )

    strategy.on_bar(
        BarData(
            datetime=pd.Timestamp("2026-01-05 09:30:00"),
            open_price=100.0,
            high_price=100.5,
            low_price=99.5,
            close_price=100.0,
            volume=10,
            open_interest=1000.0,
        )
    )
    assert strategy.position_state is not None
    assert strategy.position_state.direction == 1

    strategy.on_bar(
        BarData(
            datetime=pd.Timestamp("2026-01-05 09:31:00"),
            open_price=100.0,
            high_price=101.5,
            low_price=99.5,
            close_price=101.0,
            volume=10,
            open_interest=1000.0,
        )
    )

    assert strategy.position_state is not None
    assert strategy.position_state.direction == 1

    strategy.on_bar(
        BarData(
            datetime=pd.Timestamp("2026-01-05 09:32:00"),
            open_price=101.0,
            high_price=101.5,
            low_price=97.5,
            close_price=98.0,
            volume=10,
            open_interest=1000.0,
        )
    )

    assert strategy.position_state is not None
    assert strategy.position_state.direction == -1
    assert strategy.runtime_state["last_exit_reason"] == "reverse_signal"


def test_replay_mode_waits_for_trade_callback_before_marking_position(tmp_path):
    model_path = tmp_path / "meta_model.pkl"
    tbm_path = tmp_path / "tbm.parquet"
    features_path = tmp_path / "events_features.parquet"
    joblib.dump(DummyModel(), model_path)

    index = pd.DatetimeIndex(["2026-01-05 09:30:00"], name="timestamp")
    pd.DataFrame(
        {
            "label": [0],
            "touch_idx": [2],
            "ret": [0.0],
            "max_rb_ratio": [1.0],
            "touch_type": ["vertical"],
            "side": [1],
            "entry_price": [100.0],
            "target": [0.01],
            "exit_price": [100.0],
            "exit_ts": [pd.Timestamp("2026-01-05 09:32:00")],
            "pnl": [0.0],
        },
        index=index,
    ).to_parquet(tbm_path)
    pd.DataFrame({"feat_demo": [0.25]}, index=index).to_parquet(features_path)

    strategy = Al9999CtaStrategy(
        cta_engine=None,
        strategy_name="al_replay_trade",
        vt_symbol="AL2506.SHFE",
        setting={
            "fixed_size": 1,
            "model_path": str(model_path),
            "research_symbol": "AL9999",
            "symbol_mapping": {"AL9999": "AL2506.SHFE"},
            "emit_orders": True,
            "input_bar_mode": "dollar",
            "replay_tbm_path": str(tbm_path),
            "replay_features_path": str(features_path),
        },
    )

    strategy.on_bar(
        BarData(
            datetime=pd.Timestamp("2026-01-05 09:30:00"),
            open_price=100.0,
            high_price=100.5,
            low_price=99.5,
            close_price=100.0,
            volume=10,
            open_interest=1000.0,
        )
    )

    assert strategy.position_state is None
    assert getattr(strategy, "pending_entry_snapshot", None) is not None

    strategy.on_trade(
        SimpleNamespace(
            datetime=pd.Timestamp("2026-01-05 09:31:00"),
            price=100.0,
            volume=1.0,
            direction=SimpleNamespace(name="LONG"),
            offset=SimpleNamespace(name="OPEN"),
        )
    )

    assert strategy.position_state is not None
    assert strategy.position_state.direction == 1


def test_export_comparison_html_writes_interactive_report(tmp_path):
    module = importlib.import_module("strategies.AL9999.12_run_vnpy_cta_backtest")

    vnpy_trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(["2026-01-05 09:30:00", "2026-01-05 10:00:00"]),
            "net_pnl": [10.0, -3.0],
        }
    )
    research_trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(["2026-01-05 09:30:00", "2026-01-05 10:00:00"]),
            "net_pnl": [10.0, -3.0],
        }
    )
    comparison = {
        "research_trade_count": 2,
        "vnpy_trade_count": 2,
        "research_total_net_pnl": 7.0,
        "vnpy_total_net_pnl": 7.0,
        "net_pnl_delta": 0.0,
        "matched_entry_timestamps": 2,
        "matched_exit_timestamps": 2,
        "matched_sides": 2,
        "vnpy_statistics": {"sharpe_ratio": 1.23, "max_drawdown": -2.0},
    }
    output_path = tmp_path / "comparison.html"

    module.export_comparison_html(vnpy_trades, research_trades, comparison, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert "AL9999 vn.py vs Research Backtest" in html
    assert "vn.py Equity Curve" in html
    assert "Research Equity Curve" in html
    assert "Curves overlap exactly on matched exits." in html
    assert "Daily Pnl" in html
    assert "Pnl Distribution" in html
    assert "Difference vs Research" in html
