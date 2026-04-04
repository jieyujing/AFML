"""
threshold_optimizer.py - AL9999 Filter-First 阈值筛选工具
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def calculate_trade_shrinkage(
    trade_count: int,
    baseline_trade_count: int,
) -> float:
    """
    Calculate trade shrinkage ratio versus baseline.

    :param trade_count: Trade count under a threshold.
    :param baseline_trade_count: Baseline trade count.
    :returns: Shrinkage ratio in [0, +inf). If baseline <= 0, returns 0.
    """
    if baseline_trade_count <= 0:
        return 0.0
    return 1.0 - (float(trade_count) / float(baseline_trade_count))


def build_threshold_report(
    rows: list[dict],
    baseline_trade_count: int,
) -> pd.DataFrame:
    """
    Build threshold report table and derive trade_shrinkage.

    :param rows: Raw rows with at least threshold and oos_n fields.
    :param baseline_trade_count: Baseline trade count.
    :returns: Normalized report DataFrame.
    """
    report = pd.DataFrame(rows).copy()
    if report.empty:
        return report

    if "oos_n" in report.columns and "trade_shrinkage" not in report.columns:
        report["trade_shrinkage"] = report["oos_n"].apply(
            lambda n: calculate_trade_shrinkage(trade_count=int(n), baseline_trade_count=baseline_trade_count)
        )

    return report.sort_values("threshold").reset_index(drop=True)


def select_best_threshold(
    result_df: pd.DataFrame,
    shrinkage_min: float,
    shrinkage_max: float,
) -> Optional[dict]:
    """
    Select best threshold from feasible candidates.

    Selection order:
    1) oos_dsr desc
    2) oos_sharpe desc

    :param result_df: Threshold-level metrics with trade_shrinkage/oos_dsr/oos_sharpe.
    :param shrinkage_min: Lower bound of acceptable shrinkage.
    :param shrinkage_max: Upper bound of acceptable shrinkage.
    :returns: Best row as dict; None if no feasible candidate exists.
    """
    if result_df.empty:
        return None

    required_cols = {"trade_shrinkage", "oos_dsr", "oos_sharpe"}
    missing = required_cols - set(result_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feasible = result_df[
        (result_df["trade_shrinkage"] >= shrinkage_min)
        & (result_df["trade_shrinkage"] <= shrinkage_max)
    ].copy()
    if feasible.empty:
        return None

    feasible = feasible.sort_values(
        ["oos_dsr", "oos_sharpe"],
        ascending=[False, False],
    )
    return feasible.iloc[0].to_dict()
