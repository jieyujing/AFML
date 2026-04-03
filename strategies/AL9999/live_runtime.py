"""
AL9999 live runtime utilities for vn.py integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from afmlkit.feature.core.frac_diff import frac_diff_ffd
from afmlkit.feature.core.ma import ewma
from afmlkit.sampling.filters import cusum_filter_with_state
from strategies.AL9999.config import CONTRACT_MULTIPLIER, map_to_trading_date
from strategies.AL9999.feature_compute import compute_all_features, compute_event_features
from strategies.AL9999.live_config import Al9999LiveConfig
from strategies.AL9999 import config as research_config


@dataclass
class DollarBar:
    """
    Lightweight runtime dollar bar.
    """

    timestamp: pd.Timestamp
    trading_date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    open_interest: float
    n_ticks: int
    bar_index: int


@dataclass
class ModelPrediction:
    """
    Meta model prediction output.
    """

    meta_pred: int
    meta_prob: float
    safe_mode: bool
    reason: str = ""


@dataclass
class SignalSnapshot:
    """
    Stable signal snapshot for the strategy state machine.
    """

    timestamp: pd.Timestamp
    bar_index: int
    primary_side: int
    meta_prob: float
    meta_pred: int
    entry_allowed: bool
    target_tp_price: float
    target_sl_price: float
    expire_bar_idx: int
    bar_close: float
    feature_row: dict[str, float]
    live_symbol: str | None
    safe_mode: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalSnapshot":
        payload = dict(data)
        payload["timestamp"] = pd.Timestamp(payload["timestamp"])
        return cls(**payload)


@dataclass
class PositionState:
    """
    Runtime position snapshot.
    """

    direction: int
    entry_price: float
    entry_bar_idx: int
    tp_price: float
    sl_price: float
    expire_bar_idx: int
    entry_signal_ts: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry_signal_ts"] = self.entry_signal_ts.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionState":
        payload = dict(data)
        payload["entry_signal_ts"] = pd.Timestamp(payload["entry_signal_ts"])
        return cls(**payload)


class ContractResolver:
    """
    Resolve the research symbol to the live tradable symbol.
    """

    def __init__(
        self,
        symbol_mapping: dict[str, str] | None = None,
        manual_overrides: dict[str, str] | None = None,
    ) -> None:
        self.symbol_mapping = dict(symbol_mapping or {})
        self.manual_overrides = dict(manual_overrides or {})

    def resolve_research_symbol(self, symbol: str) -> str:
        """
        Resolve research symbol to the live tradable symbol.
        """
        if symbol in self.manual_overrides:
            return self.manual_overrides[symbol]
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol]
        raise KeyError(f"Missing live contract mapping for {symbol}.")


class LiveSignalModel:
    """
    Thin adapter around the offline-trained meta model.
    """

    def __init__(self, model_path: str | Path | None = None, probability_threshold: float = 0.5) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.probability_threshold = float(probability_threshold)
        self._model = None
        self._load_error: str | None = None

    @property
    def safe_mode(self) -> bool:
        return self._model is None

    def load(self) -> None:
        """
        Load the persisted model if available.
        """
        if self._model is not None:
            return
        if self.model_path is None or not self.model_path.exists():
            self._load_error = f"Model file not found: {self.model_path}"
            return
        try:
            self._model = joblib.load(self.model_path)
            self._load_error = None
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = str(exc)
            self._model = None

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """
        Predict whether the signal should be traded.
        """
        self.load()
        if self._model is None:
            return ModelPrediction(meta_pred=0, meta_prob=0.0, safe_mode=True, reason=self._load_error or "model_unavailable")

        if X.empty:
            return ModelPrediction(meta_pred=0, meta_prob=0.0, safe_mode=True, reason="empty_features")

        try:
            frame = X.fillna(0.0)
            if hasattr(self._model, "predict_proba"):
                prob = float(self._model.predict_proba(frame)[0, 1])
            else:  # pragma: no cover - fallback path
                prob = float(self._model.predict(frame)[0])
            pred = int(prob >= self.probability_threshold)
            if hasattr(self._model, "predict"):
                pred = int(self._model.predict(frame)[0])
            return ModelPrediction(meta_pred=pred, meta_prob=prob, safe_mode=False)
        except Exception as exc:  # pragma: no cover - defensive
            return ModelPrediction(meta_pred=0, meta_prob=0.0, safe_mode=True, reason=str(exc))


class Al9999LiveRuntime:
    """
    Recomputing runtime that keeps the live adapter aligned with offline research code.
    """

    def __init__(
        self,
        target_daily_bars: int | None = None,
        ewma_span: int | None = None,
        fracdiff_threshold: float | None = None,
        fracdiff_d: float = 0.0,
        cusum_window: int | None = None,
        cusum_multiplier: float | None = None,
        primary_span: int | None = None,
        feature_config: dict[str, Any] | None = None,
        tbm_config: dict[str, Any] | None = None,
        model: LiveSignalModel | None = None,
        contract_resolver: ContractResolver | None = None,
        research_symbol: str = "AL9999",
        feature_nan_ratio_limit: float = 0.5,
    ) -> None:
        self.target_daily_bars = int(target_daily_bars or research_config.TARGET_DAILY_BARS)
        self.ewma_span = int(ewma_span or research_config.EWMA_SPAN)
        self.fracdiff_threshold = float(fracdiff_threshold or research_config.FRACDIFF_THRES)
        self.fracdiff_d = float(fracdiff_d)
        self.cusum_window = int(cusum_window or research_config.CUSUM_WINDOW)
        self.cusum_multiplier = float(cusum_multiplier if cusum_multiplier is not None else research_config.CUSUM_MULTIPLIER)
        self.primary_span = int(primary_span or research_config.MA_PRIMARY_MODEL.get("span", 20))
        self.feature_config = feature_config or research_config.FEATURE_CONFIG
        self.tbm_config = tbm_config or research_config.TBM_CONFIG
        self.model = model or LiveSignalModel()
        self.contract_resolver = contract_resolver or ContractResolver()
        self.research_symbol = research_symbol
        self.feature_nan_ratio_limit = float(feature_nan_ratio_limit)

        self.minute_bars = pd.DataFrame()
        self.dollar_bars = pd.DataFrame()
        self.last_signal: SignalSnapshot | None = None
        self.last_event_timestamp: pd.Timestamp | None = None

    def _coerce_minute_bar(self, bar: Any) -> dict[str, Any]:
        """
        Normalize incoming minute bar data.
        """
        if isinstance(bar, dict):
            dt = pd.Timestamp(bar["datetime"])
            open_ = float(bar["open"])
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            volume = float(bar["volume"])
            open_interest = float(bar.get("open_interest", np.nan))
        else:
            dt = pd.Timestamp(getattr(bar, "datetime"))
            open_ = float(self._read_value(bar, "open_price", "open"))
            high = float(self._read_value(bar, "high_price", "high"))
            low = float(self._read_value(bar, "low_price", "low"))
            close = float(self._read_value(bar, "close_price", "close"))
            volume = float(getattr(bar, "volume"))
            open_interest = float(getattr(bar, "open_interest", np.nan))

        return {
            "datetime": dt,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": open_interest,
        }

    @staticmethod
    def _read_value(obj: Any, *names: str) -> Any:
        """
        Read the first available attribute or key.
        """
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"Unable to read any of {names!r} from {type(obj)!r}.")

    def _compute_dynamic_thresholds(self, frame: pd.DataFrame) -> pd.Series:
        daily_dollar = frame["dollar_volume"].astype(float).groupby(frame["trading_date"]).sum()
        daily_dollar = daily_dollar.replace(0, np.nan).ffill()
        daily_ewma = daily_dollar.ewm(span=self.ewma_span, min_periods=1).mean()
        thresholds = daily_ewma / float(self.target_daily_bars)
        return thresholds.rename("threshold")

    def _build_dollar_bars(self) -> pd.DataFrame:
        if self.minute_bars.empty:
            return pd.DataFrame()

        frame = self.minute_bars.copy()
        thresholds = self._compute_dynamic_thresholds(frame)
        threshold_map = thresholds.to_dict()
        frame["threshold"] = frame["trading_date"].map(lambda x: threshold_map.get(pd.Timestamp(x), thresholds.mean()))
        frame["threshold"] = frame["threshold"].fillna(float(thresholds.mean()))

        close_indices = [0]
        cum_dollar = frame["dollar_volume"].iloc[0]
        for i in range(1, len(frame)):
            cum_dollar += frame["dollar_volume"].iloc[i]
            if cum_dollar >= frame["threshold"].iloc[i]:
                close_indices.append(i)
                cum_dollar -= frame["threshold"].iloc[i]

        if len(close_indices) < 2:
            return pd.DataFrame()

        bars = []
        for bar_index, (start_idx, end_idx) in enumerate(zip(close_indices[:-1], close_indices[1:])):
            segment = frame.iloc[start_idx : end_idx + 1]
            bars.append(
                {
                    "timestamp": segment.index[-1],
                    "trading_date": segment["trading_date"].iloc[-1],
                    "open": float(segment["open"].iloc[0]),
                    "high": float(segment["high"].max()),
                    "low": float(segment["low"].min()),
                    "close": float(segment["close"].iloc[-1]),
                    "volume": float(segment["volume"].sum()),
                    "dollar_volume": float(segment["dollar_volume"].sum()),
                    "open_interest": float(segment["open_interest"].iloc[-1]),
                    "n_ticks": int(len(segment)),
                    "bar_index": int(bar_index),
                }
            )

        bars_df = pd.DataFrame(bars).set_index("timestamp")
        return bars_df

    def update_minute_bar(self, bar: Any) -> list[DollarBar]:
        """
        Ingest a minute bar and return any newly completed dollar bars.
        """
        row = self._coerce_minute_bar(bar)
        timestamp = row.pop("datetime")
        row["dollar_volume"] = row["close"] * row["volume"] * CONTRACT_MULTIPLIER
        row["trading_date"] = map_to_trading_date(timestamp)

        self.minute_bars.loc[timestamp, list(row.keys())] = list(row.values())
        self.minute_bars = self.minute_bars.sort_index()

        previous_index = set(self.dollar_bars.index) if not self.dollar_bars.empty else set()
        rebuilt = self._build_dollar_bars()
        self.dollar_bars = rebuilt

        if rebuilt.empty:
            return []

        new_rows = rebuilt.loc[~rebuilt.index.isin(previous_index)]
        return [
            DollarBar(
                timestamp=index,
                trading_date=pd.Timestamp(item["trading_date"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
                dollar_volume=float(item["dollar_volume"]),
                open_interest=float(item["open_interest"]),
                n_ticks=int(item["n_ticks"]),
                bar_index=int(item["bar_index"]),
            )
            for index, item in new_rows.iterrows()
        ]

    def append_dollar_bar(self, bar: Any) -> DollarBar:
        """
        Append a prebuilt dollar bar directly into the runtime state.
        """
        if isinstance(bar, DollarBar):
            dollar_bar = bar
        else:
            timestamp = pd.Timestamp(self._read_value(bar, "timestamp", "datetime", "name"))
            bar_index = len(self.dollar_bars)
            has_trading_date = (isinstance(bar, dict) and "trading_date" in bar) or hasattr(bar, "trading_date")
            has_dollar_volume = (isinstance(bar, dict) and "dollar_volume" in bar) or hasattr(bar, "dollar_volume")
            has_open_interest = (isinstance(bar, dict) and "open_interest" in bar) or hasattr(bar, "open_interest")
            has_n_ticks = (isinstance(bar, dict) and "n_ticks" in bar) or hasattr(bar, "n_ticks")
            has_bar_index = (isinstance(bar, dict) and "bar_index" in bar) or hasattr(bar, "bar_index")
            dollar_bar = DollarBar(
                timestamp=timestamp,
                trading_date=pd.Timestamp(self._read_value(bar, "trading_date")) if has_trading_date else map_to_trading_date(timestamp),
                open=float(self._read_value(bar, "open", "open_price")),
                high=float(self._read_value(bar, "high", "high_price")),
                low=float(self._read_value(bar, "low", "low_price")),
                close=float(self._read_value(bar, "close", "close_price")),
                volume=float(self._read_value(bar, "volume")),
                dollar_volume=float(self._read_value(bar, "dollar_volume")) if has_dollar_volume else float(self._read_value(bar, "close", "close_price")) * float(self._read_value(bar, "volume")),
                open_interest=float(self._read_value(bar, "open_interest")) if has_open_interest else np.nan,
                n_ticks=int(self._read_value(bar, "n_ticks")) if has_n_ticks else 1,
                bar_index=int(self._read_value(bar, "bar_index")) if has_bar_index else bar_index,
            )

        payload = {
            "trading_date": dollar_bar.trading_date,
            "open": float(dollar_bar.open),
            "high": float(dollar_bar.high),
            "low": float(dollar_bar.low),
            "close": float(dollar_bar.close),
            "volume": float(dollar_bar.volume),
            "dollar_volume": float(dollar_bar.dollar_volume),
            "open_interest": float(dollar_bar.open_interest),
            "n_ticks": int(dollar_bar.n_ticks),
            "bar_index": int(dollar_bar.bar_index),
        }
        self.dollar_bars.loc[dollar_bar.timestamp, list(payload.keys())] = list(payload.values())
        self.dollar_bars = self.dollar_bars.sort_index()
        return dollar_bar

    def _compute_fracdiff_series(self, bars: pd.DataFrame) -> pd.Series:
        prices = pd.Series(bars["close"].values, index=bars.index, name="close")
        series = frac_diff_ffd(prices, d=self.fracdiff_d, thres=self.fracdiff_threshold)
        return series.reindex(bars.index).ffill()

    def _compute_cusum_events(self, fracdiff_series: pd.Series) -> pd.DatetimeIndex:
        diff = fracdiff_series.diff().dropna()
        min_points = 1 if self.cusum_multiplier <= 0 else 2
        if len(diff) < min_points:
            return pd.DatetimeIndex([])
        if self.cusum_multiplier <= 0:
            return pd.DatetimeIndex(diff.index)

        threshold = diff.rolling(self.cusum_window).std() * self.cusum_multiplier
        fallback = diff.std() * self.cusum_multiplier
        if not np.isfinite(fallback):
            fallback = 0.0
        threshold = threshold.fillna(fallback)
        threshold = threshold.abs()

        threshold_values = threshold.values.astype(np.float64)
        if np.allclose(threshold_values, 0.0):
            threshold_values = np.full(len(diff), 1e-12, dtype=np.float64)

        event_indices, _, _, _ = cusum_filter_with_state(
            diff.values.astype(np.float64),
            threshold_values,
        )
        return pd.DatetimeIndex(diff.index[event_indices])

    def _compute_primary_side(self, close_values: np.ndarray) -> int:
        ma_values = ewma(close_values.astype(np.float64), self.primary_span)
        side = np.sign(close_values[-1] - ma_values[-1])
        return int(side if side != 0 else 1)

    def _resolve_live_symbol(self) -> tuple[str | None, bool, str]:
        try:
            return self.contract_resolver.resolve_research_symbol(self.research_symbol), False, ""
        except Exception as exc:
            return None, True, str(exc)

    def on_dollar_bar(self, bar: DollarBar | pd.Series | dict[str, Any]) -> SignalSnapshot | None:
        """
        Process a completed dollar bar and return a live signal when the bar is a CUSUM event.
        """
        if self.dollar_bars.empty:
            return None

        current_timestamp = pd.Timestamp(getattr(bar, "timestamp", None) or getattr(bar, "name", None) or bar.get("timestamp"))
        if current_timestamp not in self.dollar_bars.index:
            return None

        bars = self.dollar_bars.copy()
        if "bar_index" not in bars.columns:
            bars["bar_index"] = np.arange(len(bars), dtype=np.int64)
        if len(bars) < max(self.primary_span, 2):
            return None

        fracdiff_series = self._compute_fracdiff_series(bars)
        event_timestamps = self._compute_cusum_events(fracdiff_series)
        if current_timestamp not in event_timestamps:
            return None

        event_positions = bars.index.get_indexer(event_timestamps)
        event_positions = event_positions[event_positions >= 0]
        if len(event_positions) == 0:
            return None

        features = compute_all_features(bars, self.feature_config)
        event_features = compute_event_features(bars, features, event_positions)
        if current_timestamp not in event_features.index:
            return None

        latest_features = event_features.loc[[current_timestamp]].copy()
        feature_cols = [col for col in latest_features.columns if col.startswith("feat_")]
        feature_frame = latest_features[feature_cols]

        nan_ratio = float(feature_frame.isna().mean().mean()) if not feature_frame.empty else 1.0
        live_symbol, symbol_safe_mode, symbol_reason = self._resolve_live_symbol()

        model_prediction = self.model.predict(feature_frame)
        primary_side = self._compute_primary_side(bars["close"].values.astype(np.float64))
        close_price = float(bars.loc[current_timestamp, "close"])

        target_col = self.tbm_config.get("target_ret_col", "feat_ewm_vol_20")
        target_ret = float(latest_features[target_col].iloc[0]) if target_col in latest_features.columns else np.nan
        min_ret = float(self.tbm_config.get("min_ret", 0.0))
        if not np.isfinite(target_ret):
            current_high = float(bars.loc[current_timestamp, "high"])
            current_low = float(bars.loc[current_timestamp, "low"])
            target_ret = abs(current_high - current_low) / max(close_price, 1e-12)
        if target_ret <= 0:
            target_ret = max(min_ret, 1e-6)

        tp_mult, sl_mult = self.tbm_config.get("profit_loss_barriers", (2.0, 2.0))
        target_value = max(abs(target_ret), min_ret) if np.isfinite(target_ret) else np.nan

        if primary_side > 0:
            tp_price = close_price * (1.0 + target_value * tp_mult) if np.isfinite(target_value) else np.nan
            sl_price = close_price * (1.0 - target_value * sl_mult) if np.isfinite(target_value) else np.nan
        else:
            tp_price = close_price * (1.0 - target_value * tp_mult) if np.isfinite(target_value) else np.nan
            sl_price = close_price * (1.0 + target_value * sl_mult) if np.isfinite(target_value) else np.nan

        safe_mode = bool(
            model_prediction.safe_mode
            or symbol_safe_mode
            or feature_frame.empty
            or nan_ratio > self.feature_nan_ratio_limit
            or not np.isfinite(target_value)
        )
        entry_allowed = bool(model_prediction.meta_pred == 1 and not safe_mode)

        diagnostics = {
            "event_detected": True,
            "target_ret": target_ret,
            "nan_ratio": nan_ratio,
            "model_reason": model_prediction.reason,
            "symbol_reason": symbol_reason,
        }

        snapshot = SignalSnapshot(
            timestamp=current_timestamp,
            bar_index=int(bars.loc[current_timestamp, "bar_index"]),
            primary_side=primary_side,
            meta_prob=float(model_prediction.meta_prob),
            meta_pred=int(model_prediction.meta_pred),
            entry_allowed=entry_allowed,
            target_tp_price=float(tp_price) if np.isfinite(tp_price) else np.nan,
            target_sl_price=float(sl_price) if np.isfinite(sl_price) else np.nan,
            expire_bar_idx=int(bars.loc[current_timestamp, "bar_index"]) + int(self.tbm_config.get("vertical_barrier_bars", 1)),
            bar_close=close_price,
            feature_row={col: float(feature_frame.iloc[0][col]) if np.isfinite(feature_frame.iloc[0][col]) else 0.0 for col in feature_cols},
            live_symbol=live_symbol,
            safe_mode=safe_mode,
            diagnostics=diagnostics,
        )
        self.last_signal = snapshot
        self.last_event_timestamp = current_timestamp
        return snapshot

    def save_state(self, path: str | Path) -> None:
        """
        Persist lightweight runtime state for restart recovery.
        """
        payload = {
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
            "last_event_timestamp": self.last_event_timestamp.isoformat() if self.last_event_timestamp is not None else None,
            "dollar_bar_count": int(len(self.dollar_bars)),
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_state(self, path: str | Path) -> None:
        """
        Restore lightweight runtime state.
        """
        input_path = Path(path)
        if not input_path.exists():
            return
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if payload.get("last_signal"):
            self.last_signal = SignalSnapshot.from_dict(payload["last_signal"])
        if payload.get("last_event_timestamp"):
            self.last_event_timestamp = pd.Timestamp(payload["last_event_timestamp"])

    @classmethod
    def from_config(cls, config: Al9999LiveConfig) -> "Al9999LiveRuntime":
        """
        Construct a runtime from the shared live config.
        """
        model = LiveSignalModel(
            model_path=config.resolve_model_path(),
            probability_threshold=config.meta_probability_threshold,
        )
        resolver = ContractResolver(
            symbol_mapping=config.symbol_mapping,
            manual_overrides=config.manual_symbol_overrides,
        )
        return cls(
            target_daily_bars=config.target_daily_bars,
            ewma_span=config.ewma_span,
            fracdiff_threshold=config.fracdiff_threshold,
            fracdiff_d=config.fracdiff_d,
            cusum_window=config.cusum_window,
            cusum_multiplier=config.cusum_multiplier,
            primary_span=config.primary_span,
            feature_config=config.feature_config,
            tbm_config=config.tbm_config,
            model=model,
            contract_resolver=resolver,
            research_symbol=config.research_symbol,
            feature_nan_ratio_limit=config.feature_nan_ratio_limit,
        )
