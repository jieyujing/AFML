"""
vn.py compatible AL9999 CTA strategy wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.AL9999.live_config import Al9999LiveConfig
from strategies.AL9999.live_runtime import Al9999LiveRuntime, PositionState, SignalSnapshot
from strategies.AL9999 import config as research_config


try:  # pragma: no cover - exercised only when vn.py is installed
    from vnpy_ctastrategy import CtaTemplate, StopOrder, TickData, BarData, TradeData, OrderData
except Exception:  # pragma: no cover - fallback for local tests
    @dataclass
    class StopOrder:
        vt_orderid: str = ""

    @dataclass
    class TickData:
        datetime: pd.Timestamp
        last_price: float = 0.0

    @dataclass
    class BarData:
        datetime: pd.Timestamp
        open_price: float
        high_price: float
        low_price: float
        close_price: float
        volume: float
        open_interest: float = 0.0

    @dataclass
    class TradeData:
        datetime: pd.Timestamp
        price: float = 0.0
        volume: float = 0.0
        direction: Any = None
        offset: Any = None

    @dataclass
    class OrderData:
        datetime: pd.Timestamp
        status: str = ""

    class CtaTemplate:
        author = "Codex"
        parameters: list[str] = []
        variables: list[str] = []

        def __init__(self, cta_engine: Any, strategy_name: str, vt_symbol: str, setting: dict[str, Any]):
            self.cta_engine = cta_engine
            self.strategy_name = strategy_name
            self.vt_symbol = vt_symbol
            self.pos = 0

        def write_log(self, msg: str) -> None:
            self._last_log = msg

        def put_event(self) -> None:
            return None

        def buy(self, price: float, volume: float, stop: bool = False) -> list[str]:
            return [f"buy:{price}:{volume}:{stop}"]

        def sell(self, price: float, volume: float, stop: bool = False) -> list[str]:
            return [f"sell:{price}:{volume}:{stop}"]

        def short(self, price: float, volume: float, stop: bool = False) -> list[str]:
            return [f"short:{price}:{volume}:{stop}"]

        def cover(self, price: float, volume: float, stop: bool = False) -> list[str]:
            return [f"cover:{price}:{volume}:{stop}"]

        def cancel_all(self) -> None:
            return None

        def load_bar(self, days: int) -> None:
            return None


class Al9999CtaStrategy(CtaTemplate):
    """
    CTA shell around the AL9999 live runtime.
    """

    author = "Codex"
    parameters = [
        "research_symbol",
        "model_path",
        "fixed_size",
        "emit_orders",
        "input_bar_mode",
    ]
    variables = [
        "safe_mode",
        "last_signal_time",
        "last_exit_reason",
        "live_symbol",
        "selected_threshold",
        "side_mode",
        "guard_enabled",
        "min_hold_bars",
        "cooldown_bars",
        "reverse_confirmation_delta",
    ]

    def __init__(self, cta_engine: Any, strategy_name: str, vt_symbol: str, setting: dict[str, Any]) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.config = Al9999LiveConfig.from_setting(setting)
        self.research_symbol = self.config.research_symbol
        self.model_path = str(self.config.resolve_model_path())
        self.fixed_size = int(self.config.fixed_size)
        self.emit_orders = bool(self.config.emit_orders)
        self.input_bar_mode = str(self.config.input_bar_mode)
        self.replay_tbm_path = str(self.config.replay_tbm_path or "")
        self.replay_features_path = str(self.config.replay_features_path or "")
        self.use_replay_signals = bool(self.replay_tbm_path and self.replay_features_path)
        self.parity_mode = str(self.config.parity_mode)
        self.strict_filter_first_artifacts = bool(self.config.strict_filter_first_artifacts)

        self.selected_threshold = float(self.config.meta_probability_threshold)
        self.side_mode = "both"
        self.scheme_used = "default"
        self.trade_shrinkage = np.nan
        self.guard_enabled = False
        self.min_hold_bars = 0
        self.cooldown_bars = 0
        self.reverse_confirmation_delta = 0.0

        self.runtime = Al9999LiveRuntime.from_config(self.config)
        self.position_state: PositionState | None = None
        self.pending_entry_snapshot: SignalSnapshot | None = None
        self.pending_exit_reason: str = ""
        self.last_exit_bar_idx: int | None = None
        self.action_log: list[dict[str, Any]] = []
        self.replay_signal_table = pd.DataFrame()
        self.runtime_state: dict[str, Any] = {
            "safe_mode": True,
            "last_signal_time": "",
            "last_exit_reason": "",
            "live_symbol": "",
            "dollar_bar_count": 0,
        }
        if self.use_replay_signals:
            self._load_filter_first_controls()
            self._align_runtime_filter_first_diagnostics()
            self.replay_signal_table = self._load_replay_signal_table()

    def on_init(self) -> None:
        self.write_log("Initializing AL9999 live CTA strategy.")
        if not self.use_replay_signals:
            self.runtime.load_state(self.config.resolve_state_path())
        self.load_bar(3)

    def on_start(self) -> None:
        self.write_log("Starting AL9999 live CTA strategy.")
        self.put_event()

    def on_stop(self) -> None:
        if not self.use_replay_signals:
            self.runtime.save_state(self.config.resolve_state_path())
        self.write_log("Stopping AL9999 live CTA strategy.")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        return None

    def on_bar(self, bar: BarData) -> None:
        if self.input_bar_mode == "dollar":
            new_dollar_bars = [self.runtime.append_dollar_bar(bar)]
        else:
            new_dollar_bars = self.runtime.update_minute_bar(bar)
        for dollar_bar in new_dollar_bars:
            self.runtime_state["dollar_bar_count"] = int(dollar_bar.bar_index) + 1
            if not self.use_replay_signals:
                self.evaluate_exit(dollar_bar)
            snapshot = self._build_replay_snapshot(dollar_bar) if self.use_replay_signals else self.runtime.on_dollar_bar(dollar_bar)
            if snapshot is not None:
                self.process_signal(snapshot)
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        self.action_log.append({"type": "order", "datetime": str(order.datetime), "status": getattr(order, "status", "")})

    def on_trade(self, trade: TradeData) -> None:
        direction_name = self._enum_name(getattr(trade, "direction", ""))
        offset_name = self._enum_name(getattr(trade, "offset", ""))
        trade_price = float(getattr(trade, "price", 0.0))
        trade_time = self._normalize_timestamp(getattr(trade, "datetime", pd.Timestamp.utcnow()))
        self.action_log.append(
            {
                "type": "trade",
                "datetime": str(trade.datetime),
                "price": trade_price,
                "direction": direction_name,
                "offset": offset_name,
            }
        )

        if offset_name == "OPEN":
            snapshot = self.pending_entry_snapshot
            if snapshot is None:
                direction = 1 if direction_name == "LONG" else -1
                self.position_state = PositionState(
                    direction=direction,
                    entry_price=trade_price,
                    entry_bar_idx=max(int(self.runtime_state.get("dollar_bar_count", 1)) - 1, 0),
                    tp_price=float("inf"),
                    sl_price=float("-inf"),
                    expire_bar_idx=max(int(self.runtime_state.get("dollar_bar_count", 1)) - 1, 0),
                    entry_signal_ts=trade_time,
                )
            else:
                self.position_state = PositionState(
                    direction=int(snapshot.primary_side),
                    entry_price=trade_price,
                    entry_bar_idx=int(snapshot.bar_index),
                    tp_price=float(snapshot.target_tp_price),
                    sl_price=float(snapshot.target_sl_price),
                    expire_bar_idx=int(snapshot.expire_bar_idx),
                    entry_signal_ts=pd.Timestamp(snapshot.timestamp),
                )
            self.pending_entry_snapshot = None
            return

        if offset_name in {"CLOSE", "CLOSETODAY", "CLOSEYESTERDAY"}:
            self.position_state = None
            self.pending_exit_reason = ""

    def on_stop_order(self, stop_order: StopOrder) -> None:
        self.action_log.append({"type": "stop_order", "vt_orderid": getattr(stop_order, "vt_orderid", "")})

    def _record_action(self, action: str, price: float, reason: str = "") -> None:
        self.action_log.append(
            {
                "action": action,
                "price": float(price),
                "reason": reason,
                "position": self.position_state.direction if self.position_state is not None else 0,
            }
        )

    def _send_entry_order(self, direction: int, price: float) -> None:
        if not self.emit_orders:
            return
        if direction > 0:
            self.buy(price, self.fixed_size)
        else:
            self.short(price, self.fixed_size)

    def _send_exit_order(self, direction: int, price: float) -> None:
        if not self.emit_orders:
            return
        if direction > 0:
            self.sell(price, self.fixed_size)
        else:
            self.cover(price, self.fixed_size)

    def enter_position(self, snapshot: SignalSnapshot) -> None:
        self.runtime_state["live_symbol"] = snapshot.live_symbol or ""
        self.runtime_state["safe_mode"] = snapshot.safe_mode
        self.runtime_state["last_signal_time"] = snapshot.timestamp.isoformat()
        self._record_action("enter", snapshot.bar_close)
        if self.emit_orders:
            self.pending_entry_snapshot = snapshot
            self._send_entry_order(snapshot.primary_side, snapshot.bar_close)
            return
        self.position_state = PositionState(
            direction=int(snapshot.primary_side),
            entry_price=float(snapshot.bar_close),
            entry_bar_idx=int(snapshot.bar_index),
            tp_price=float(snapshot.target_tp_price),
            sl_price=float(snapshot.target_sl_price),
            expire_bar_idx=int(snapshot.expire_bar_idx),
            entry_signal_ts=pd.Timestamp(snapshot.timestamp),
        )
        self._send_entry_order(snapshot.primary_side, snapshot.bar_close)

    def exit_position(self, price: float, reason: str) -> None:
        if self.position_state is None:
            return
        direction = self.position_state.direction
        self._send_exit_order(direction, price)
        self.runtime_state["last_exit_reason"] = reason
        self._record_action("exit", price, reason=reason)
        if self.emit_orders:
            self.pending_exit_reason = reason
            return
        self.position_state = None

    def evaluate_exit(self, bar: Any) -> str | None:
        if self.position_state is None:
            return None

        bar_index = int(getattr(bar, "bar_index", self.runtime_state.get("dollar_bar_count", 0)))
        high_price = float(self._read_bar_value(bar, "high", "high_price"))
        low_price = float(self._read_bar_value(bar, "low", "low_price"))
        close_price = float(self._read_bar_value(bar, "close", "close_price"))

        direction = self.position_state.direction
        if direction > 0:
            if low_price <= self.position_state.sl_price:
                self.exit_position(self.position_state.sl_price, "stop_loss")
                return "stop_loss"
            if high_price >= self.position_state.tp_price:
                self.exit_position(self.position_state.tp_price, "take_profit")
                return "take_profit"
        else:
            if high_price >= self.position_state.sl_price:
                self.exit_position(self.position_state.sl_price, "stop_loss")
                return "stop_loss"
            if low_price <= self.position_state.tp_price:
                self.exit_position(self.position_state.tp_price, "take_profit")
                return "take_profit"

        if bar_index >= self.position_state.expire_bar_idx:
            self.exit_position(close_price, "time_barrier")
            return "time_barrier"

        return None

    @staticmethod
    def _read_bar_value(bar: Any, *names: str) -> Any:
        """
        Read the first available bar field from a dict or object.
        """
        for name in names:
            if isinstance(bar, dict) and name in bar:
                return bar[name]
            if hasattr(bar, name):
                return getattr(bar, name)
        raise AttributeError(f"Unable to read any of {names!r} from {type(bar)!r}.")

    @staticmethod
    def _normalize_timestamp(value: Any) -> pd.Timestamp:
        """
        Normalize timestamps to naive pandas timestamps for research index alignment.
        """
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp

    def process_signal(self, snapshot: SignalSnapshot) -> None:
        self.runtime_state["safe_mode"] = snapshot.safe_mode
        self.runtime_state["last_signal_time"] = snapshot.timestamp.isoformat()
        self.runtime_state["live_symbol"] = snapshot.live_symbol or ""

        if not snapshot.entry_allowed:
            return

        if self.use_replay_signals:
            self.process_replay_signal(snapshot)
            return

        if self.position_state is None:
            self.enter_position(snapshot)
            return

        if self.position_state.direction == snapshot.primary_side:
            return

        self.exit_position(snapshot.bar_close, "signal_flip")
        self.enter_position(snapshot)

    def process_replay_signal(self, snapshot: SignalSnapshot) -> None:
        """
        Process research replay signals using the rolling_backtest single-position logic.
        """
        if self.side_mode == "long_only" and int(snapshot.primary_side) < 0:
            return
        if self.side_mode == "short_only" and int(snapshot.primary_side) > 0:
            return

        current_direction = self._current_direction()
        current_bar_idx = int(snapshot.bar_index)

        if current_direction == 0:
            if self.guard_enabled and self.cooldown_bars > 0 and self.last_exit_bar_idx is not None:
                bars_since_exit = current_bar_idx - int(self.last_exit_bar_idx)
                if bars_since_exit <= self.cooldown_bars:
                    return
            self.enter_position(snapshot)
            return

        if current_direction == snapshot.primary_side:
            return

        if self.guard_enabled and self.min_hold_bars > 0 and self.position_state is not None:
            held_bars = current_bar_idx - int(self.position_state.entry_bar_idx)
            if held_bars < self.min_hold_bars:
                return
        if self.guard_enabled and self.reverse_confirmation_delta > 0:
            confirm_threshold = float(self.selected_threshold) + float(self.reverse_confirmation_delta)
            if float(snapshot.meta_prob) < confirm_threshold:
                return

        self.last_exit_bar_idx = current_bar_idx
        self.exit_position(snapshot.bar_close, "reverse_signal")
        self.enter_position(snapshot)

    def _load_filter_first_controls(self) -> None:
        """
        Load filter-first selection metadata and execution guard config.
        """
        guard_cfg = research_config.FILTER_FIRST_CONFIG.get("execution_guard", {})
        self.guard_enabled = bool(guard_cfg.get("enabled", False))
        self.min_hold_bars = int(guard_cfg.get("min_hold_bars", 0))
        self.cooldown_bars = int(guard_cfg.get("cooldown_bars", 0))
        self.reverse_confirmation_delta = float(guard_cfg.get("reverse_confirmation_delta", 0.0))

        if self.parity_mode != "filter_first":
            return

        selection_path = self.config.resolve_filter_first_selection_path()
        threshold_report_path = self.config.resolve_filter_first_threshold_report_path()
        required_paths = [selection_path, threshold_report_path]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing and self.strict_filter_first_artifacts:
            raise FileNotFoundError(
                "Filter-First 对接缺少必需产物，请先运行 10_combined_backtest.py: " + ", ".join(missing)
            )
        if not selection_path.exists():
            return

        selection = pd.read_parquet(selection_path)
        if selection.empty:
            if self.strict_filter_first_artifacts:
                raise ValueError(f"Filter-First selection 文件为空: {selection_path}")
            return

        row = selection.iloc[0]
        self.selected_threshold = float(row.get("selected_threshold", self.selected_threshold))
        self.side_mode = str(row.get("side_mode", self.side_mode))
        self.scheme_used = str(row.get("scheme_used", self.scheme_used))
        self.trade_shrinkage = float(row.get("trade_shrinkage", np.nan))

        scheme = self.scheme_used
        if scheme not in {"", "default", "full"}:
            current_model = Path(self.model_path)
            if current_model.name == "meta_model.pkl":
                candidate = current_model.with_name(f"meta_model_{scheme}.pkl")
                if candidate.exists():
                    self.model_path = str(candidate)
                    self.runtime.model.model_path = candidate
                elif self.strict_filter_first_artifacts:
                    raise FileNotFoundError(
                        f"Filter-First scheme={scheme} 需要模型文件 {candidate}，当前不存在。"
                    )

    def _align_runtime_filter_first_diagnostics(self) -> None:
        """
        Keep runtime diagnostics aligned with selected filter-first controls.
        """
        self.runtime.selected_threshold = float(self.selected_threshold)
        self.runtime.side_mode = str(self.side_mode)
        self.runtime.scheme_used = str(self.scheme_used)
        self.runtime.guard_flags = {
            "enabled": bool(self.guard_enabled),
            "min_hold_bars": int(self.min_hold_bars),
            "cooldown_bars": int(self.cooldown_bars),
            "reverse_confirmation_delta": float(self.reverse_confirmation_delta),
        }

    def _load_replay_signal_table(self) -> pd.DataFrame:
        """
        Load research TBM and feature rows for fast backtesting replay.
        """
        tbm = pd.read_parquet(self.replay_tbm_path).sort_index()
        features = pd.read_parquet(self.replay_features_path).sort_index()
        common_index = tbm.index.intersection(features.index).sort_values()
        if common_index.empty:
            return pd.DataFrame()

        feature_cols = [column for column in features.columns if column.startswith("feat_")]
        feature_frame = features.loc[common_index, feature_cols].fillna(0.0)

        self.runtime.model.load()
        model = self.runtime.model._model
        if model is None:
            meta_pred = np.zeros(len(common_index), dtype=int)
            meta_prob = np.zeros(len(common_index), dtype=float)
            safe_mode = True
        else:
            if hasattr(model, "predict_proba"):
                meta_prob = model.predict_proba(feature_frame)[:, 1]
            else:
                meta_prob = model.predict(feature_frame).astype(float)
            meta_pred = self._apply_filter_first_threshold_to_pred(
                meta_prob=meta_prob,
                sides=tbm.loc[common_index, "side"].values,
            )
            safe_mode = False

        replay_table = tbm.loc[common_index].copy()
        replay_table = replay_table.join(features.loc[common_index, feature_cols], how="left")
        replay_table["meta_pred"] = meta_pred
        replay_table["meta_prob"] = meta_prob
        replay_table["safe_mode"] = safe_mode
        replay_table["selected_threshold"] = float(self.selected_threshold)
        replay_table["side_mode"] = str(self.side_mode)
        replay_table["scheme_used"] = str(self.scheme_used)
        replay_table["trade_shrinkage"] = float(self.trade_shrinkage) if np.isfinite(self.trade_shrinkage) else np.nan
        return replay_table

    def _apply_filter_first_threshold_to_pred(self, meta_prob: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """
        Recompute meta_pred using the same threshold policy as filter-first backtest.
        """
        threshold = float(self.selected_threshold)
        side_mode = str(self.side_mode)
        short_delta = float(research_config.FILTER_FIRST_CONFIG.get("short_penalty_delta", 0.0))
        probs = np.asarray(meta_prob, dtype=np.float64)
        side_values = np.asarray(sides, dtype=np.int64)

        if side_mode == "both_with_short_penalty":
            short_threshold = threshold + short_delta
            return np.where(
                side_values == -1,
                (probs >= short_threshold).astype(np.int64),
                (probs >= threshold).astype(np.int64),
            )
        if side_mode == "long_only":
            return np.where((side_values == 1) & (probs >= threshold), 1, 0).astype(np.int64)
        if side_mode == "short_only":
            return np.where((side_values == -1) & (probs >= threshold), 1, 0).astype(np.int64)
        return (probs >= threshold).astype(np.int64)

    def _build_replay_snapshot(self, dollar_bar: Any) -> SignalSnapshot | None:
        """
        Build a signal snapshot from precomputed research artifacts.
        """
        timestamp = self._normalize_timestamp(getattr(dollar_bar, "timestamp", self._read_bar_value(dollar_bar, "datetime", "timestamp")))
        if self.replay_signal_table.empty or timestamp not in self.replay_signal_table.index:
            return None

        row = self.replay_signal_table.loc[timestamp]
        side = int(row["side"])
        exit_price = float(row["exit_price"])
        touch_type = str(row.get("touch_type", "vertical"))
        expire_bar_idx = int(row["touch_idx"])

        if side > 0:
            tp_price = exit_price if touch_type == "upper" else float("inf")
            sl_price = exit_price if touch_type == "lower" else float("-inf")
        else:
            tp_price = exit_price if touch_type == "lower" else float("-inf")
            sl_price = exit_price if touch_type == "upper" else float("inf")

        feature_row = {key: float(value) for key, value in row.items() if str(key).startswith("feat_") and pd.notna(value)}
        try:
            live_symbol = self.runtime.contract_resolver.resolve_research_symbol(self.research_symbol)
        except Exception:
            live_symbol = self.vt_symbol

        return SignalSnapshot(
            timestamp=timestamp,
            bar_index=int(getattr(dollar_bar, "bar_index", self.runtime_state.get("dollar_bar_count", 1) - 1)),
            primary_side=side,
            meta_prob=float(row["meta_prob"]),
            meta_pred=int(row["meta_pred"]),
            entry_allowed=bool(int(row["meta_pred"]) == 1),
            target_tp_price=tp_price,
            target_sl_price=sl_price,
            expire_bar_idx=expire_bar_idx,
            bar_close=float(self._read_bar_value(dollar_bar, "close", "close_price")),
            feature_row=feature_row,
            live_symbol=live_symbol,
            safe_mode=bool(row["safe_mode"]),
            diagnostics={
                "mode": "research_replay",
                "touch_type": touch_type,
                "touch_idx": expire_bar_idx,
                "selected_threshold": float(self.selected_threshold),
                "side_mode": str(self.side_mode),
                "guard_flags": {
                    "enabled": bool(self.guard_enabled),
                    "min_hold_bars": int(self.min_hold_bars),
                    "cooldown_bars": int(self.cooldown_bars),
                    "reverse_confirmation_delta": float(self.reverse_confirmation_delta),
                },
                "scheme_used": str(self.scheme_used),
            },
        )

    @staticmethod
    def _enum_name(value: Any) -> str:
        """
        Read enum-like name values from vn.py objects or lightweight stubs.
        """
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)

    def _current_direction(self) -> int:
        """
        Return the current effective direction including pending opens.
        """
        if self.position_state is not None:
            return int(self.position_state.direction)
        if self.pending_entry_snapshot is not None:
            return int(self.pending_entry_snapshot.primary_side)
        return 0
