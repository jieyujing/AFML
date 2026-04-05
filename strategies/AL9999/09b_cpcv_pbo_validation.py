"""
09b_cpcv_pbo_validation.py - AL9999 CPCV rank-PBO validation.

This script provides a minimal and testable CPCV workflow:
1) load rolling_combined_trades.parquet
2) build DatetimeIndex (entry_time) and t1 (exit_time)
3) run CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
4) generate 2D Sharpe matrices (IS/OOS) with deterministic rules
5) compute rank-based PBO
6) save parquet outputs and figures
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from afmlkit.validation import CombinatorialPurgedKFold, calculate_pbo
from strategies.AL9999.config import FEATURES_DIR, FIGURES_DIR


ANNUALIZATION_FACTOR = 252.0


def build_param_grid() -> list[dict]:
    """
    Build deterministic TBM parameter candidates.

    :returns: List of parameter dictionaries.
    """
    tbm_barriers = [0.8, 1.0, 1.2, 1.4, 1.6]
    min_rets = [0.0, 0.0005, 0.0010, 0.0015, 0.0020]

    params: list[dict] = []
    for barrier in tbm_barriers:
        for min_ret in min_rets:
            params.append(
                {
                    "tbm_barriers": (float(barrier), float(barrier)),
                    "min_ret": float(min_ret),
                }
            )
    return params


def compute_rank_pbo(sharpe_is: np.ndarray, sharpe_oos: np.ndarray) -> Tuple[float, Dict]:
    """
    Compute rank-based PBO from IS/OOS Sharpe matrices.

    :param sharpe_is: IS Sharpe matrix with shape (n_strategies, n_paths).
    :param sharpe_oos: OOS Sharpe matrix with shape (n_strategies, n_paths).
    :returns: (pbo, stats)
    """
    return calculate_pbo(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos, method="rank")


def _safe_sharpe(returns: np.ndarray, annualization_factor: float = ANNUALIZATION_FACTOR) -> float:
    """
    Compute annualized Sharpe safely for short or zero-volatility samples.
    """
    if returns.size < 2:
        return 0.0

    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0

    mean = float(np.mean(returns))
    return mean / std * np.sqrt(annualization_factor)


def _build_strategy_returns(base_returns: np.ndarray, param_grid: List[dict]) -> List[np.ndarray]:
    """
    Build deterministic strategy return series from base returns.
    """
    strategy_returns: list[np.ndarray] = []
    for params in param_grid:
        tp_mult, sl_mult = params["tbm_barriers"]
        min_ret = float(params["min_ret"])
        scale = 0.5 * (float(tp_mult) + float(sl_mult))
        adjusted = base_returns * scale - min_ret
        strategy_returns.append(adjusted.astype(np.float64, copy=False))
    return strategy_returns


def _aggregate_cpcv_sharpes(
    cpcv: CombinatorialPurgedKFold,
    X: pd.DataFrame,
    strategy_returns: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate IS/OOS Sharpe matrices aligned to CPCV paths.
    """
    n_strategies = len(strategy_returns)
    n_paths = cpcv.get_n_paths()
    path_lookup = cpcv.get_path_assignments()

    is_buckets = [[[] for _ in range(n_paths)] for _ in range(n_strategies)]
    oos_buckets = [[[] for _ in range(n_paths)] for _ in range(n_strategies)]

    for split_idx, (train_idx, _test_idx, test_folds) in enumerate(cpcv.split(X)):
        for strat_idx, ret_array in enumerate(strategy_returns):
            is_sharpe = _safe_sharpe(ret_array[train_idx])

            for fold_idx, fold_indices in test_folds:
                path_idx = path_lookup[(split_idx, fold_idx)]
                oos_sharpe = _safe_sharpe(ret_array[fold_indices])
                is_buckets[strat_idx][path_idx].append(is_sharpe)
                oos_buckets[strat_idx][path_idx].append(oos_sharpe)

    sharpe_is = np.zeros((n_strategies, n_paths), dtype=np.float64)
    sharpe_oos = np.zeros((n_strategies, n_paths), dtype=np.float64)

    for strat_idx in range(n_strategies):
        for path_idx in range(n_paths):
            is_vals = is_buckets[strat_idx][path_idx]
            oos_vals = oos_buckets[strat_idx][path_idx]
            sharpe_is[strat_idx, path_idx] = float(np.mean(is_vals)) if is_vals else 0.0
            sharpe_oos[strat_idx, path_idx] = float(np.mean(oos_vals)) if oos_vals else 0.0

    return sharpe_is, sharpe_oos


def _sharpe_matrix_to_frame(sharpe_matrix: np.ndarray, param_grid: List[dict]) -> pd.DataFrame:
    """
    Convert Sharpe matrix to a tabular parquet-friendly DataFrame.
    """
    n_paths = sharpe_matrix.shape[1]
    path_cols = [f"path_{i + 1}" for i in range(n_paths)]

    df = pd.DataFrame(sharpe_matrix, columns=path_cols)
    df.insert(0, "strategy_id", np.arange(len(param_grid), dtype=np.int64))
    df.insert(
        1,
        "tbm_tp",
        [float(params["tbm_barriers"][0]) for params in param_grid],
    )
    df.insert(
        2,
        "tbm_sl",
        [float(params["tbm_barriers"][1]) for params in param_grid],
    )
    df.insert(3, "min_ret", [float(params["min_ret"]) for params in param_grid])
    return df


def _plot_sharpe_distribution(sharpe_is: np.ndarray, sharpe_oos: np.ndarray, output_path: str) -> None:
    """
    Plot IS/OOS Sharpe distribution histogram.
    """
    is_values = sharpe_is.ravel()
    oos_values = sharpe_oos.ravel()

    plt.figure(figsize=(10, 6))
    plt.hist(is_values, bins=20, alpha=0.55, label="IS Sharpe", color="steelblue", edgecolor="black")
    plt.hist(oos_values, bins=20, alpha=0.55, label="OOS Sharpe", color="darkorange", edgecolor="black")
    plt.axvline(x=0.0, color="red", linestyle="--", linewidth=1.5, label="Sharpe = 0")
    plt.title("CPCV Sharpe Distribution (IS vs OOS)")
    plt.xlabel("Annualized Sharpe")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_rank_comparison(stats: Dict, output_path: str) -> None:
    """
    Plot selected OOS ranks per CPCV path.
    """
    selected_ranks = np.asarray(stats.get("selected_ranks", []), dtype=np.float64)
    if selected_ranks.size == 0:
        selected_ranks = np.array([0.0], dtype=np.float64)

    x = np.arange(1, selected_ranks.size + 1)
    median_rank = float(stats.get("median_rank", 0.0))

    plt.figure(figsize=(10, 5))
    plt.bar(x, selected_ranks, color="teal", alpha=0.75, edgecolor="black")
    plt.axhline(y=median_rank, color="crimson", linestyle="--", linewidth=1.5, label=f"Median Rank = {median_rank:.2f}")
    plt.title("Rank PBO: Selected OOS Rank by Path")
    plt.xlabel("Path")
    plt.ylabel("Selected OOS Rank")
    plt.xticks(x)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> Tuple[float, Dict]:
    """
    Run minimal CPCV rank-PBO validation workflow.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    combined_path = os.path.join(FEATURES_DIR, "rolling_combined_trades.parquet")
    combined_trades = pd.read_parquet(combined_path)

    required_cols = {"entry_time", "exit_time", "net_ret"}
    missing_cols = required_cols.difference(combined_trades.columns)
    if missing_cols:
        raise ValueError(f"rolling_combined_trades.parquet 缺少列: {sorted(missing_cols)}")

    combined_trades = combined_trades.sort_values("entry_time").reset_index(drop=True).copy()
    combined_trades["entry_time"] = pd.to_datetime(combined_trades["entry_time"])
    combined_trades["exit_time"] = pd.to_datetime(combined_trades["exit_time"])

    trade_index = pd.DatetimeIndex(combined_trades["entry_time"], name="entry_time")
    t1 = pd.Series(combined_trades["exit_time"].to_numpy(), index=trade_index, name="t1")
    X = pd.DataFrame(index=trade_index)

    cpcv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_splits=2,
        t1=t1,
        embargo_pct=0.01,
    )

    param_grid = build_param_grid()
    base_returns = combined_trades["net_ret"].to_numpy(dtype=np.float64)
    strategy_returns = _build_strategy_returns(base_returns=base_returns, param_grid=param_grid)
    sharpe_is, sharpe_oos = _aggregate_cpcv_sharpes(cpcv=cpcv, X=X, strategy_returns=strategy_returns)

    pbo, stats = compute_rank_pbo(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos)

    sharpe_is_df = _sharpe_matrix_to_frame(sharpe_is, param_grid)
    sharpe_oos_df = _sharpe_matrix_to_frame(sharpe_oos, param_grid)
    pbo_df = pd.DataFrame(
        [
            {
                "pbo": float(pbo),
                "n_strategies": int(stats.get("n_strategies", sharpe_is.shape[0])),
                "n_paths": int(stats.get("n_paths", sharpe_is.shape[1])),
                "median_rank": float(stats.get("median_rank", np.nan)),
                "overfit_count": int(stats.get("overfit_count", 0)),
                "selected_rank_mean": float(np.mean(stats.get("selected_ranks", np.array([np.nan])))),
                "selected_rank_min": float(np.min(stats.get("selected_ranks", np.array([np.nan])))),
                "selected_rank_max": float(np.max(stats.get("selected_ranks", np.array([np.nan])))),
            }
        ]
    )

    sharpe_is_path = os.path.join(FEATURES_DIR, "cpcv_sharpe_is.parquet")
    sharpe_oos_path = os.path.join(FEATURES_DIR, "cpcv_sharpe_oos.parquet")
    pbo_results_path = os.path.join(FEATURES_DIR, "cpcv_pbo_results.parquet")
    sharpe_is_df.to_parquet(sharpe_is_path)
    sharpe_oos_df.to_parquet(sharpe_oos_path)
    pbo_df.to_parquet(pbo_results_path)

    sharpe_fig_path = os.path.join(FIGURES_DIR, "09b_cpcv_sharpe_distribution.png")
    rank_fig_path = os.path.join(FIGURES_DIR, "09b_pbo_rank_comparison.png")
    _plot_sharpe_distribution(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos, output_path=sharpe_fig_path)
    _plot_rank_comparison(stats=stats, output_path=rank_fig_path)

    print(f"Saved: {sharpe_is_path}")
    print(f"Saved: {sharpe_oos_path}")
    print(f"Saved: {pbo_results_path}")
    print(f"Saved: {sharpe_fig_path}")
    print(f"Saved: {rank_fig_path}")
    print(f"rank PBO: {pbo:.6f}")

    return pbo, stats


if __name__ == "__main__":
    main()
