"""
11_verify_vnpy_factor_consistency.py - 验证 vn.py/live runtime 因子一致性

验证分两层进行：
1. 批量重算验证：使用保存好的 Dollar Bars 重算 Phase 2 事件与因子，
   与离线产物 `events_features.parquet` 做逐列比对。
2. 增量回放验证：模拟 vn.py 每根 Dollar Bar 逐步到达，抽样比较
   live runtime 生成的事件因子与参考因子。

说明：
- 批量验证应当逐列完全一致。
- 增量验证允许在冷启动阶段出现少量“早产事件”，
  原因是离线脚本对 CUSUM warmup 使用了全样本 std，
  而实时系统只能使用当下可见历史。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from afmlkit.feature.core.frac_diff import frac_diff_ffd
from afmlkit.sampling.filters import cusum_filter_with_state
from strategies.AL9999.config import (
    BARS_DIR,
    CUSUM_MULTIPLIER,
    CUSUM_WINDOW,
    FEATURE_CONFIG,
    FEATURES_DIR,
    FRACDIFF_THRES,
    MA_PRIMARY_MODEL,
    TBM_CONFIG,
    TARGET_DAILY_BARS,
    EWMA_SPAN,
)
from strategies.AL9999.feature_compute import compute_all_features, compute_event_features
from strategies.AL9999.live_runtime import (
    Al9999LiveRuntime,
    ContractResolver,
    DollarBar,
    LiveSignalModel,
)


def load_reference_data() -> tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    """
    Load saved offline research outputs.
    """
    bars = pd.read_parquet(os.path.join(BARS_DIR, f"dollar_bars_target{TARGET_DAILY_BARS}.parquet"))
    frac_params = pd.read_parquet(os.path.join(FEATURES_DIR, "fracdiff_params.parquet"))
    ref_events = pd.read_parquet(os.path.join(FEATURES_DIR, "cusum_events.parquet"))
    ref_features = pd.read_parquet(os.path.join(FEATURES_DIR, "events_features.parquet"))
    optimal_d = float(frac_params["optimal_d"].iloc[0])
    return bars, optimal_d, ref_events, ref_features


def recompute_batch_outputs(
    bars: pd.DataFrame,
    optimal_d: float,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    Recompute events and event features from the saved Dollar Bars.
    """
    prices = bars["close"]
    fracdiff_series = frac_diff_ffd(prices, d=optimal_d, thres=FRACDIFF_THRES).reindex(bars.index)
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index

    threshold_series = fracdiff_diff.rolling(CUSUM_WINDOW).std() * CUSUM_MULTIPLIER
    threshold_series = threshold_series.fillna(fracdiff_diff.std() * CUSUM_MULTIPLIER)

    event_indices, *_ = cusum_filter_with_state(
        fracdiff_diff.dropna().values.astype(np.float64),
        threshold_series.loc[valid_idx].values.astype(np.float64),
    )
    event_timestamps = pd.DatetimeIndex(valid_idx[event_indices], name="timestamp")
    bars_event_indices = np.array([bars.index.get_loc(ts) for ts in event_timestamps], dtype=np.int64)

    features_df = compute_all_features(bars, FEATURE_CONFIG)
    event_features_df = compute_event_features(bars, features_df, bars_event_indices)
    return event_timestamps, event_features_df


def run_incremental_sample_check(
    bars: pd.DataFrame,
    optimal_d: float,
    ref_features: pd.DataFrame,
    sample_limit: int = 25,
    max_bars: int = 1200,
) -> dict[str, Any]:
    """
    Simulate vn.py/live runtime bar-by-bar replay for a sample window.
    """
    replay_bars = bars.copy()
    replay_bars["bar_index"] = np.arange(len(replay_bars), dtype=np.int64)

    runtime = Al9999LiveRuntime(
        target_daily_bars=TARGET_DAILY_BARS,
        ewma_span=EWMA_SPAN,
        fracdiff_threshold=FRACDIFF_THRES,
        fracdiff_d=optimal_d,
        cusum_window=CUSUM_WINDOW,
        cusum_multiplier=CUSUM_MULTIPLIER,
        primary_span=int(MA_PRIMARY_MODEL.get("span", 20)),
        feature_config=FEATURE_CONFIG,
        tbm_config=TBM_CONFIG,
        model=LiveSignalModel(model_path=os.path.join(os.path.dirname(__file__), "output", "models", "meta_model.pkl")),
        contract_resolver=ContractResolver({"AL9999": "AL9999"}),
    )

    snapshots = []
    for i, (ts, row) in enumerate(replay_bars.iloc[:max_bars].iterrows()):
        runtime.dollar_bars = replay_bars.iloc[: i + 1].copy()
        snapshot = runtime.on_dollar_bar(
            DollarBar(
                timestamp=ts,
                trading_date=pd.Timestamp(row["trading_date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                dollar_volume=float(row["dollar_volume"]),
                open_interest=float(row["open_interest"]),
                n_ticks=int(row["n_ticks"]),
                bar_index=int(row["bar_index"]),
            )
        )
        if snapshot is not None:
            snapshots.append(snapshot)
        if len(snapshots) >= sample_limit:
            break

    sample = pd.DataFrame(
        [snapshot.feature_row for snapshot in snapshots],
        index=pd.DatetimeIndex([snapshot.timestamp for snapshot in snapshots], name="timestamp"),
    ).sort_index()
    common_idx = sample.index.intersection(ref_features.index)
    feat_cols = [col for col in sample.columns if col in ref_features.columns]
    abs_diff = (sample.loc[common_idx, feat_cols] - ref_features.loc[common_idx, feat_cols]).abs()
    max_diff = abs_diff.max().sort_values(ascending=False)

    sample_only = sample.index.difference(ref_features.index)
    return {
        "sampled_runtime_events": len(sample),
        "matched_reference_events": len(common_idx),
        "sample_only_events": [ts.isoformat() for ts in sample_only],
        "all_matched_feature_columns_within_1e_12": bool((max_diff <= 1e-12).all()),
        "top_feature_max_diff": {key: float(value) for key, value in max_diff.head(10).items()},
    }


def main() -> None:
    """
    Run the factor consistency verification and persist the report.
    """
    print("=" * 72)
    print("  AL9999 vn.py Factor Consistency Verification")
    print("=" * 72)

    bars, optimal_d, ref_events, ref_features = load_reference_data()
    print(f"\n[Load] bars={len(bars)}, reference_events={len(ref_events)}, reference_features={ref_features.shape}")
    print(f"[Load] optimal_d={optimal_d:.4f}")

    recomputed_events, recomputed_features = recompute_batch_outputs(bars, optimal_d)
    event_match = pd.DatetimeIndex(ref_events["timestamp"], name="timestamp").equals(recomputed_events)
    feature_cols = [col for col in ref_features.columns if col.startswith("feat_") and col in recomputed_features.columns]
    batch_abs_diff = (ref_features[feature_cols].sort_index() - recomputed_features[feature_cols].sort_index()).abs()
    batch_max_diff = batch_abs_diff.max().sort_values(ascending=False)

    print(f"\n[Batch] event_index_equal={event_match}")
    print(f"[Batch] feature_columns={len(feature_cols)}")
    print("[Batch] top max diffs:")
    print(batch_max_diff.head(10).to_string())

    incremental_report = run_incremental_sample_check(bars, optimal_d, ref_features)
    print(f"\n[Incremental] sampled_runtime_events={incremental_report['sampled_runtime_events']}")
    print(f"[Incremental] matched_reference_events={incremental_report['matched_reference_events']}")
    print(f"[Incremental] sample_only_events={incremental_report['sample_only_events']}")
    print(
        "[Incremental] all_matched_feature_columns_within_1e_12="
        f"{incremental_report['all_matched_feature_columns_within_1e_12']}"
    )

    report = {
        "batch": {
            "event_index_equal": bool(event_match),
            "feature_columns": len(feature_cols),
            "all_feature_columns_within_1e_12": bool((batch_max_diff <= 1e-12).all()),
            "top_feature_max_diff": {key: float(value) for key, value in batch_max_diff.head(10).items()},
        },
        "incremental_sample": incremental_report,
        "notes": [
            "批量重算验证使用与 Phase 2 相同的公式，应当完全一致。",
            "增量回放样本若出现少量 sample_only_events，通常来自 CUSUM warmup 阈值初始化口径差异。",
            "离线脚本使用全样本 std 填充早期 rolling std 缺口，而实时系统只能使用当下已知历史。",
        ],
    }

    output_path = os.path.join(FEATURES_DIR, "vnpy_factor_consistency_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] report={output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
