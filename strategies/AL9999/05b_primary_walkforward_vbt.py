"""
05b_primary_walkforward_vbt.py - Primary rule mini walk-forward via vectorbt.

目标：
1. 对 3 个 primary 候选做小型 walk-forward 对比
2. 做 commission/slippage 成本压力测试
3. 输出最终推荐的 primary rule
"""

import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from afmlkit.sampling import cusum_filter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import (
    BARS_DIR,
    COMMISSION_RATE,
    COST_PRESSURE_CONFIG,
    OUTPUT_DIR,
    SLIPPAGE_POINTS,
)


DEFAULT_CANDIDATES = [
    "rate=0.15_fast=5_slow=20",
    "rate=0.05_fast=5_slow=20",
    "rate=0.1_fast=5_slow=20",
]


def parse_combo_id(combo_id: str) -> dict:
    """Parse combo_id into rate/fast/slow fields."""
    parts = combo_id.split("_")
    parsed = {}
    for item in parts:
        key, value = item.split("=")
        if key == "rate":
            parsed[key] = float(value)
        else:
            parsed[key] = int(value)
    return parsed


def generate_walk_forward_slices(
    index: pd.Index,
    n_splits: int = 3,
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
) -> list[dict]:
    """
    Generate sequential walk-forward slices with non-overlapping OOS windows.
    """
    n = len(index)
    if n == 0:
        return []

    train_size = max(int(n * train_ratio), 1)
    available = max(n - train_size, 0)
    if available == 0:
        return []

    requested_test_size = max(int(n * test_ratio), 1)
    test_size = max(1, min(requested_test_size, available // max(n_splits, 1)))
    if test_size == 0:
        return []

    slices = []
    train_start = 0
    cursor = train_size
    for fold_id in range(1, n_splits + 1):
        train_end = min(train_start + train_size, n)
        test_start = max(cursor, train_end)
        test_end = min(test_start + test_size, n)
        if test_start >= test_end:
            break
        slices.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = test_end

    return slices


def generate_dma_side(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """
    Generate full-bar DMA side series.
    """
    fast_ma = bars["close"].ewm(span=fast, adjust=False).mean()
    slow_ma = bars["close"].ewm(span=slow, adjust=False).mean()
    side = pd.Series(0, index=bars.index, dtype=np.int8)
    side[fast_ma > slow_ma] = 1
    side[fast_ma < slow_ma] = -1
    return side


def load_cusum_k_lookup() -> dict[float, float]:
    """
    Load calibrated CUSUM rate -> k lookup.
    """
    path = os.path.join(OUTPUT_DIR, "primary_search", "cusum_calibration.parquet")
    calibration = pd.read_parquet(path)
    return {float(row["rate"]): float(row["k"]) for _, row in calibration.iterrows()}


def build_event_driven_side_from_events(
    bars: pd.DataFrame,
    event_indices: Iterable[int],
    fast: int,
    slow: int,
) -> pd.Series:
    """
    Build position side from sparse event timestamps and forward-hold to next event.
    """
    dma_side = generate_dma_side(bars=bars, fast=fast, slow=slow)
    event_side = pd.Series(0, index=bars.index, dtype=np.int8)

    valid = []
    for idx in event_indices:
        idx = int(idx)
        if 0 <= idx < len(bars):
            valid.append(idx)

    if not valid:
        return event_side

    event_side.iloc[valid] = dma_side.iloc[valid]
    event_side = event_side.replace(0, np.nan).ffill().fillna(0).astype(np.int8)
    return event_side


def generate_event_driven_side(
    bars: pd.DataFrame,
    combo_id: str,
    k_lookup: dict[float, float],
) -> pd.Series:
    """
    Generate event-driven side path for a primary candidate.
    """
    params = parse_combo_id(combo_id)
    k = k_lookup[params["rate"]]
    close = bars["close"].values.astype(np.float64)
    log_returns = np.diff(np.log(close))
    event_indices = cusum_filter(log_returns, np.array([k]))
    return build_event_driven_side_from_events(
        bars=bars,
        event_indices=event_indices,
        fast=params["fast"],
        slow=params["slow"],
    )


def run_vbt_backtest(
    bars: pd.DataFrame,
    side: pd.Series,
    commission: float,
    slippage_points: float,
    init_cash: float = 1_000_000.0,
) -> dict:
    """
    Run single-position long/short backtest via vectorbt.
    """
    try:
        import vectorbt as vbt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vectorbt is not installed. Please install vectorbt>=0.28.5 with Python 3.11."
        ) from exc

    price = bars["open"] if "open" in bars.columns else bars["close"]
    price = price.astype(np.float64)
    aligned_side = side.reindex(price.index).fillna(0).astype(np.int8)
    prev_side = aligned_side.shift(1).fillna(0).astype(np.int8)

    entries = (aligned_side == 1) & (prev_side != 1)
    exits = (prev_side == 1) & (aligned_side != 1)
    short_entries = (aligned_side == -1) & (prev_side != -1)
    short_exits = (prev_side == -1) & (aligned_side != -1)

    slippage_pct = (slippage_points / price.replace(0, np.nan)).fillna(0.0)

    portfolio = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        fees=commission,
        slippage=slippage_pct.values,
        freq="1D",
    )

    returns = portfolio.returns()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))

    equity = portfolio.value()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "total_pnl": float(portfolio.total_profit()),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trade_count": int(portfolio.trades.count()),
    }


def run_candidate_walk_forward(
    bars: pd.DataFrame,
    combo_id: str,
    k_lookup: dict[float, float],
    n_splits: int = 3,
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    commission: float = COMMISSION_RATE,
    slippage_points: float = SLIPPAGE_POINTS,
) -> pd.DataFrame:
    """
    Run walk-forward evaluation for one combo.
    """
    side = generate_event_driven_side(bars=bars, combo_id=combo_id, k_lookup=k_lookup)
    slices = generate_walk_forward_slices(
        index=bars.index,
        n_splits=n_splits,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )

    rows = []
    for slc in slices:
        test_bars = bars.iloc[slc["test_start"]:slc["test_end"]]
        test_side = side.iloc[slc["test_start"]:slc["test_end"]]
        metrics = run_vbt_backtest(
            bars=test_bars,
            side=test_side,
            commission=commission,
            slippage_points=slippage_points,
        )
        rows.append(
            {
                "combo_id": combo_id,
                "fold_id": slc["fold_id"],
                "train_start": bars.index[slc["train_start"]],
                "train_end": bars.index[slc["train_end"] - 1],
                "test_start": bars.index[slc["test_start"]],
                "test_end": bars.index[slc["test_end"] - 1],
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def run_candidate_cost_pressure(
    bars: pd.DataFrame,
    combo_id: str,
    k_lookup: dict[float, float],
    commission_multipliers: Iterable[float],
    slippage_multipliers: Iterable[float],
) -> pd.DataFrame:
    """
    Run cost pressure test on the final OOS segment.
    """
    side = generate_event_driven_side(bars=bars, combo_id=combo_id, k_lookup=k_lookup)
    test_start = int(len(bars) * 0.8)
    test_bars = bars.iloc[test_start:]
    test_side = side.iloc[test_start:]

    rows = []
    for comm_mult in commission_multipliers:
        for slip_mult in slippage_multipliers:
            metrics = run_vbt_backtest(
                bars=test_bars,
                side=test_side,
                commission=COMMISSION_RATE * comm_mult,
                slippage_points=SLIPPAGE_POINTS * slip_mult,
            )
            rows.append(
                {
                    "combo_id": combo_id,
                    "commission_mult": float(comm_mult),
                    "slippage_mult": float(slip_mult),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def build_candidate_comparison_table(
    wf_results: pd.DataFrame,
    cost_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build final ranking table from walk-forward and cost pressure results.
    """
    if "trade_count" not in wf_results.columns:
        wf_results = wf_results.copy()
        wf_results["trade_count"] = 0
    if "max_dd" not in cost_results.columns:
        cost_results = cost_results.copy()
        cost_results["max_dd"] = 0.0

    wf_summary = wf_results.groupby("combo_id").agg(
        wf_median_sharpe=("sharpe", "median"),
        wf_mean_sharpe=("sharpe", "mean"),
        wf_total_pnl=("total_pnl", "sum"),
        wf_worst_dd=("max_dd", "min"),
        wf_trade_count=("trade_count", "sum"),
    )
    cost_summary = cost_results.groupby("combo_id").agg(
        cost_worst_sharpe=("sharpe", "min"),
        cost_worst_pnl=("total_pnl", "min"),
        cost_worst_dd=("max_dd", "min"),
    )

    summary = wf_summary.join(cost_summary, how="outer").reset_index()
    summary = summary.sort_values(
        ["wf_median_sharpe", "cost_worst_sharpe", "wf_total_pnl", "wf_worst_dd"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary["final_rank"] = np.arange(1, len(summary) + 1)
    return summary


def load_bars() -> pd.DataFrame:
    """
    Load primary research dollar bars.
    """
    path = os.path.join(BARS_DIR, "dollar_bars_target15.parquet")
    return pd.read_parquet(path)


def main():
    """
    Run mini vectorbt walk-forward analysis for three primary candidates.
    """
    bars = load_bars()
    k_lookup = load_cusum_k_lookup()
    wf_frames = []
    cost_frames = []

    for combo_id in DEFAULT_CANDIDATES:
        wf_frames.append(
            run_candidate_walk_forward(
                bars=bars,
                combo_id=combo_id,
                k_lookup=k_lookup,
            )
        )
        cost_frames.append(
            run_candidate_cost_pressure(
                bars=bars,
                combo_id=combo_id,
                k_lookup=k_lookup,
                commission_multipliers=COST_PRESSURE_CONFIG["commission_multipliers"],
                slippage_multipliers=COST_PRESSURE_CONFIG["slippage_multipliers"],
            )
        )

    wf_results = pd.concat(wf_frames, ignore_index=True)
    cost_results = pd.concat(cost_frames, ignore_index=True)
    summary = build_candidate_comparison_table(wf_results=wf_results, cost_results=cost_results)

    output_dir = os.path.join(OUTPUT_DIR, "results", "primary_vbt_walkforward")
    os.makedirs(output_dir, exist_ok=True)
    wf_results.to_parquet(os.path.join(output_dir, "wf_results.parquet"), index=False)
    cost_results.to_parquet(os.path.join(output_dir, "cost_results.parquet"), index=False)
    summary.to_parquet(os.path.join(output_dir, "summary.parquet"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    print("[Primary VBT WF] Summary:")
    print(summary.to_string(index=False))
    print(f"[Primary VBT WF] Saved to: {output_dir}")


if __name__ == "__main__":
    main()
