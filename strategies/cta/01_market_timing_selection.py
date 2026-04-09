"""
CTA tree step 0/1: build dollar bars then select market timing points with CUSUM.

This module implements:
1. Build dynamic-threshold dollar bars from 1-minute OHLCV data.
2. Calibrate CUSUM threshold k for target event rates.
3. Validate event quality against random timestamps.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from afmlkit.sampling import cusum_filter


def map_to_trading_date(dt: pd.Timestamp) -> pd.Timestamp:
    """
    Map timestamp to trading date for CN futures.

    Night session (21:00+) is mapped to next trade date.

    :param dt: Intraday timestamp.
    :returns: Trading date as normalized timestamp.
    """
    ts = pd.Timestamp(dt)
    if ts.hour >= 21:
        return ts.normalize() + pd.Timedelta(days=1)
    return ts.normalize()


def load_minute_csv(data_path: str) -> pd.DataFrame:
    """
    Load 1-minute OHLCV csv into a normalized DataFrame.

    :param data_path: CSV path with columns including datetime/open/high/low/close/volume.
    :returns: DataFrame indexed by datetime.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    df = pd.read_csv(data_path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).set_index("datetime")

    if "open_interest" not in df.columns:
        df["open_interest"] = np.nan

    return df


def _compute_dynamic_daily_thresholds(
    minute_df: pd.DataFrame,
    target_daily_bars: int,
    ewma_span: int,
    contract_multiplier: float,
) -> pd.Series:
    """
    Compute dynamic dollar-bar threshold per trading date.

    threshold = EWMA(daily_dollar_value) / target_daily_bars
    """
    if target_daily_bars <= 0:
        raise ValueError("target_daily_bars must be > 0.")

    df = minute_df.copy()
    df["trading_date"] = df.index.map(map_to_trading_date)
    df["dollar_value"] = df["close"].astype(float) * df["volume"].astype(float) * float(contract_multiplier)

    daily_dollar = df.groupby("trading_date")["dollar_value"].sum().replace(0.0, np.nan).ffill()
    thresholds = daily_dollar.ewm(span=ewma_span, min_periods=1).mean() / float(target_daily_bars)
    return thresholds.rename("threshold")


def _dynamic_dollar_bar_indices(
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
    contract_multiplier: float,
) -> np.ndarray:
    """
    Compute close indices for dynamic dollar bars with threshold carry-over.
    """
    n = len(prices)
    if n < 2:
        return np.array([], dtype=np.int64)

    indices: list[int] = [0]
    cum_dollar = float(prices[0]) * float(volumes[0]) * float(contract_multiplier)
    for i in range(1, n):
        cum_dollar += float(prices[i]) * float(volumes[i]) * float(contract_multiplier)
        if cum_dollar >= float(thresholds[i]):
            indices.append(i)
            cum_dollar -= float(thresholds[i])
    return np.asarray(indices, dtype=np.int64)


def _build_ohlcv_from_close_indices(minute_df: pd.DataFrame, close_indices: np.ndarray) -> pd.DataFrame:
    """
    Aggregate minute bars into dollar bars using close indices.
    """
    if len(close_indices) < 2:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_volume", "n_ticks"])

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for i in range(len(close_indices) - 1):
        start_idx = int(close_indices[i])
        end_idx = int(close_indices[i + 1])
        segment = minute_df.iloc[start_idx : end_idx + 1]
        rows.append(
            {
                "timestamp": minute_df.index[end_idx],
                "trading_date": map_to_trading_date(minute_df.index[end_idx]),
                "open": float(segment["open"].iloc[0]),
                "high": float(segment["high"].max()),
                "low": float(segment["low"].min()),
                "close": float(segment["close"].iloc[-1]),
                "volume": float(segment["volume"].sum()),
                "dollar_volume": float((segment["close"] * segment["volume"]).sum()),
                "open_interest": float(segment["open_interest"].iloc[-1]),
                "n_ticks": int(len(segment)),
            }
        )

    bars = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return bars


def build_dollar_bars_from_minute(
    minute_df: pd.DataFrame,
    target_daily_bars: int = 15,
    ewma_span: int = 20,
    contract_multiplier: float = 5.0,
) -> pd.DataFrame:
    """
    Build dynamic-threshold dollar bars from minute OHLCV.

    :param minute_df: Minute OHLCV DataFrame with datetime index.
    :param target_daily_bars: Target bars per trading date.
    :param ewma_span: EWMA span for daily threshold smoothing.
    :param contract_multiplier: Futures contract multiplier.
    :returns: Dollar bars DataFrame.
    """
    if minute_df.empty:
        raise ValueError("minute_df is empty.")

    df = minute_df.copy()
    df["trading_date"] = df.index.map(map_to_trading_date)
    thresholds = _compute_dynamic_daily_thresholds(
        minute_df=df,
        target_daily_bars=target_daily_bars,
        ewma_span=ewma_span,
        contract_multiplier=contract_multiplier,
    )

    threshold_map = thresholds.to_dict()
    fallback = float(thresholds.mean())
    df["threshold"] = df["trading_date"].map(lambda x: float(threshold_map.get(x, fallback)))

    close_indices = _dynamic_dollar_bar_indices(
        prices=df["close"].to_numpy(dtype=np.float64),
        volumes=df["volume"].to_numpy(dtype=np.float64),
        thresholds=df["threshold"].to_numpy(dtype=np.float64),
        contract_multiplier=float(contract_multiplier),
    )
    return _build_ohlcv_from_close_indices(df, close_indices)


def compute_event_indices_from_bars(bars: pd.DataFrame, k: float) -> np.ndarray:
    """
    Apply CUSUM on log returns from dollar bars.
    """
    if k <= 0:
        raise ValueError("k must be > 0.")
    if len(bars) < 3:
        return np.array([], dtype=np.int64)

    close = bars["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close))
    return cusum_filter(log_returns, np.array([float(k)], dtype=np.float64))


def calibrate_cusum_rates_on_bars(
    bars: pd.DataFrame,
    target_rates: Sequence[float] = (0.05, 0.10, 0.15),
    k_min: float = 1e-5,
    k_max: float = 0.05,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> pd.DataFrame:
    """
    Calibrate CUSUM threshold for target event rates via binary search.
    """
    if len(bars) < 3:
        raise ValueError("bars length must be >= 3.")

    close = bars["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close))
    n = len(log_returns)
    if n <= 1:
        raise ValueError("not enough returns for CUSUM calibration.")

    records = []
    for target_rate in target_rates:
        if not (0 < float(target_rate) < 1):
            raise ValueError(f"target rate must be in (0,1), got {target_rate}.")

        low = float(k_min)
        high = float(k_max)
        guess = (low + high) / 2.0
        best_guess = guess
        best_error = np.inf

        for _ in range(max_iter):
            events = cusum_filter(log_returns, np.array([guess], dtype=np.float64))
            actual_rate = len(events) / float(n)
            err = abs(actual_rate - float(target_rate))
            if err < best_error:
                best_error = err
                best_guess = guess
            if abs(actual_rate - float(target_rate)) <= tol:
                break
            if actual_rate > float(target_rate):
                low = guess
            else:
                high = guess
            guess = (low + high) / 2.0

        events = cusum_filter(log_returns, np.array([best_guess], dtype=np.float64))
        actual_rate = len(events) / float(n)
        records.append(
            {
                "rate": float(target_rate),
                "k": float(best_guess),
                "actual_rate": float(actual_rate),
                "n_events": int(len(events)),
            }
        )

    return pd.DataFrame(records).sort_values("rate").reset_index(drop=True)


def _forward_return_windows(log_returns: np.ndarray, event_indices: np.ndarray, horizon: int) -> np.ndarray:
    """
    Compute forward cumulative log returns for event windows.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if len(event_indices) == 0:
        return np.array([], dtype=np.float64)

    valid = event_indices[event_indices + horizon < len(log_returns)]
    if len(valid) == 0:
        return np.array([], dtype=np.float64)

    out = np.empty(len(valid), dtype=np.float64)
    for i, idx in enumerate(valid):
        out[i] = float(np.sum(log_returns[idx + 1 : idx + horizon + 1]))
    return out


def compute_market_timing_metrics(
    bars: pd.DataFrame,
    event_indices: np.ndarray,
    overlap_window_bars: int = 20,
    eval_horizons: Sequence[int] = (5, 10, 20),
    random_seed: int = 7,
) -> dict[str, float]:
    """
    Compute cta_tree step-1 validation metrics.

    :param bars: Dollar bars DataFrame.
    :param event_indices: Event indices from CUSUM on bar returns.
    :param overlap_window_bars: Overlap threshold in bars.
    :param eval_horizons: Forward return horizons.
    :param random_seed: Random seed for baseline comparison.
    :returns: Dict of metrics.
    """
    n_bars = int(len(bars))
    n_events = int(len(event_indices))
    metrics: dict[str, float] = {
        "n_bars": float(n_bars),
        "n_events": float(n_events),
        "events_per_1000_bars": float((n_events / max(n_bars, 1)) * 1000.0),
    }

    if n_events <= 1:
        metrics["mean_bars_between_events"] = np.nan
        metrics["median_bars_between_events"] = np.nan
        metrics["event_overlap_ratio"] = 0.0
    else:
        gaps = np.diff(event_indices)
        metrics["mean_bars_between_events"] = float(np.mean(gaps))
        metrics["median_bars_between_events"] = float(np.median(gaps))
        metrics["event_overlap_ratio"] = float(np.mean(gaps <= int(overlap_window_bars)))

    close = bars["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close))

    rng = np.random.default_rng(random_seed)
    for horizon in eval_horizons:
        event_fw = _forward_return_windows(log_returns, event_indices, int(horizon))
        if len(event_fw) == 0:
            metrics[f"event_absret_adv_h{horizon}"] = np.nan
            metrics[f"event_dir_adv_h{horizon}"] = np.nan
            continue

        max_start = len(log_returns) - int(horizon) - 1
        if max_start <= 1:
            metrics[f"event_absret_adv_h{horizon}"] = np.nan
            metrics[f"event_dir_adv_h{horizon}"] = np.nan
            continue

        random_candidates = np.arange(1, max_start + 1, dtype=np.int64)
        random_size = min(len(event_fw), len(random_candidates))
        random_idx = rng.choice(random_candidates, size=random_size, replace=False)
        rand_fw = _forward_return_windows(log_returns, random_idx, int(horizon))

        if len(rand_fw) == 0:
            metrics[f"event_absret_adv_h{horizon}"] = np.nan
            metrics[f"event_dir_adv_h{horizon}"] = np.nan
            continue

        m = min(len(event_fw), len(rand_fw))
        event_fw = event_fw[:m]
        rand_fw = rand_fw[:m]
        metrics[f"event_absret_adv_h{horizon}"] = float(np.mean(np.abs(event_fw)) - np.mean(np.abs(rand_fw)))
        metrics[f"event_dir_adv_h{horizon}"] = float(np.mean(event_fw) - np.mean(rand_fw))

    return metrics


def run_market_timing_selection(
    data_path: str,
    output_dir: str,
    target_daily_bars: int = 15,
    ewma_span: int = 20,
    contract_multiplier: float = 5.0,
    target_rates: Sequence[float] = (0.05, 0.10, 0.15),
    overlap_window_bars: int = 20,
    eval_horizons: Sequence[int] = (5, 10, 20),
) -> dict[str, object]:
    """
    Run step 0+1 pipeline and persist artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    minute_df = load_minute_csv(data_path)
    bars = build_dollar_bars_from_minute(
        minute_df=minute_df,
        target_daily_bars=target_daily_bars,
        ewma_span=ewma_span,
        contract_multiplier=contract_multiplier,
    )

    bars_path = os.path.join(output_dir, "dollar_bars.parquet")
    bars.to_parquet(bars_path)

    calibration = calibrate_cusum_rates_on_bars(
        bars=bars,
        target_rates=target_rates,
    )
    calibration_path = os.path.join(output_dir, "cusum_calibration.parquet")
    calibration.to_parquet(calibration_path, index=False)

    validation_rows: list[dict[str, float]] = []
    events_paths: dict[float, str] = {}
    for row in calibration.itertuples(index=False):
        rate = float(row.rate)
        k = float(row.k)
        event_indices = compute_event_indices_from_bars(bars=bars, k=k)
        event_ts = bars.index[event_indices]

        events_df = pd.DataFrame({"timestamp": event_ts, "bar_index": event_indices, "cusum_rate": rate})
        events_path = os.path.join(output_dir, f"cusum_events_rate_{rate:.2f}.parquet")
        events_df.to_parquet(events_path, index=False)
        events_paths[rate] = events_path

        metrics = compute_market_timing_metrics(
            bars=bars,
            event_indices=event_indices,
            overlap_window_bars=overlap_window_bars,
            eval_horizons=eval_horizons,
        )
        metrics["target_rate"] = rate
        metrics["calibrated_k"] = k
        metrics["actual_rate"] = float(row.actual_rate)
        validation_rows.append(metrics)

    validation = pd.DataFrame(validation_rows).sort_values("target_rate").reset_index(drop=True)
    validation_path = os.path.join(output_dir, "market_timing_validation.parquet")
    validation.to_parquet(validation_path, index=False)
    validation.to_csv(os.path.join(output_dir, "market_timing_validation.csv"), index=False)

    return {
        "bars_path": bars_path,
        "calibration_path": calibration_path,
        "validation_path": validation_path,
        "events_paths": events_paths,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTA step1 market timing selection.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/link/Documents/AFMLKIT/data/csv/AL9999.XSGE-2020-1-1-To-2026-04-02-1m.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/link/Documents/AFMLKIT/strategies/cta/output/market_timing",
    )
    parser.add_argument("--target-daily-bars", type=int, default=15)
    parser.add_argument("--ewma-span", type=int, default=20)
    parser.add_argument("--contract-multiplier", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = run_market_timing_selection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_daily_bars=args.target_daily_bars,
        ewma_span=args.ewma_span,
        contract_multiplier=args.contract_multiplier,
    )
    print("Market timing selection completed.")
    print(f"  Dollar bars: {artifacts['bars_path']}")
    print(f"  CUSUM calibration: {artifacts['calibration_path']}")
    print(f"  Validation report: {artifacts['validation_path']}")
    for rate, path in sorted(artifacts["events_paths"].items()):
        print(f"  Events ({rate:.2f}): {path}")


if __name__ == "__main__":
    main()
