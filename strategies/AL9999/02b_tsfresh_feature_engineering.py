"""
02b_tsfresh_feature_engineering.py - tsfresh 风格统计特征提取

基于 CUSUM 事件切片，提取 16 个统计特征。
不使用 tsfresh.extract_features（开销过大），直接计算特征。

架构：
    输入层: close, log_close, volume, open_interest
         │
         ├── raw ──────────────→ 16 统计特征 ──→ feat_{col}_raw_{func}
         ├── pct_change ───────→ 16 统计特征 ──→ feat_{col}_pct_{func}
         ├── fracdiff ─────────→ 16 统计特征 ──→ feat_{col}_fd_{func}  (仅 close, log_close)
         │
         └── zscore_raw ───────→ 16 统计特征 ──→ feat_{col}_z{window}_{func}
                                                               │
                                                         windows: [10, 20, 40]

事件切片: 每个 CUSUM 事件点，回看 TSFRESH_LOOKBACK bars 提取特征
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from strategies.AL9999.config import (
    BARS_DIR,
    FEATURES_DIR,
    FRACDIFF_THRES,
    TARGET_DAILY_BARS,
    TSFRESH_CONFIG,
)
from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd


# =============================================================================
# 16 个统计特征计算函数（对应 tsfresh 的 MinimalFCParameters）
# =============================================================================


def _f_mean(x):
    return np.mean(x)


def _f_median(x):
    return np.median(x)


def _f_standard_deviation(x):
    return np.std(x, ddof=1)


def _f_skewness(x):
    if len(x) < 3:
        return np.nan
    return skew(x, bias=False)


def _f_kurtosis(x):
    if len(x) < 4:
        return np.nan
    return kurtosis(x, bias=False)


def _f_minimum(x):
    return np.min(x)


def _f_maximum(x):
    return np.max(x)


def _f_abs_energy(x):
    return np.sum(x**2)


def _f_mean_change(x):
    if len(x) < 2:
        return np.nan
    return np.mean(np.diff(x))


def _f_mean_abs_change(x):
    if len(x) < 2:
        return np.nan
    return np.mean(np.abs(np.diff(x)))


def _f_count_above_mean(x):
    m = np.mean(x)
    return np.sum(x > m)


def _f_count_below_mean(x):
    m = np.mean(x)
    return np.sum(x < m)


def _f_first_location_of_maximum(x):
    """归一化位置: (first_idx_of_max) / len(x)"""
    if len(x) == 0:
        return np.nan
    idx = np.argmax(x)
    return idx / len(x)


def _f_first_location_of_minimum(x):
    """归一化位置: (first_idx_of_min) / len(x)"""
    if len(x) == 0:
        return np.nan
    idx = np.argmin(x)
    return idx / len(x)


def _f_last_location_of_maximum(x):
    """归一化位置: (last_idx_of_max) / len(x)"""
    if len(x) == 0:
        return np.nan
    idx = len(x) - 1 - np.argmax(x[::-1])
    return idx / len(x)


def _f_last_location_of_minimum(x):
    """归一化位置: (last_idx_of_min) / len(x)"""
    if len(x) == 0:
        return np.nan
    idx = len(x) - 1 - np.argmin(x[::-1])
    return idx / len(x)


# 特征函数映射
TSFRESH_FEATURE_FUNCS = {
    "mean": _f_mean,
    "median": _f_median,
    "standard_deviation": _f_standard_deviation,
    "skewness": _f_skewness,
    "kurtosis": _f_kurtosis,
    "minimum": _f_minimum,
    "maximum": _f_maximum,
    "abs_energy": _f_abs_energy,
    "mean_change": _f_mean_change,
    "mean_abs_change": _f_mean_abs_change,
    "count_above_mean": _f_count_above_mean,
    "count_below_mean": _f_count_below_mean,
    "first_location_of_maximum": _f_first_location_of_maximum,
    "first_location_of_minimum": _f_first_location_of_minimum,
    "last_location_of_maximum": _f_last_location_of_maximum,
    "last_location_of_minimum": _f_last_location_of_minimum,
}


# =============================================================================
# 变换函数
# =============================================================================


def apply_raw(series: pd.Series) -> pd.Series:
    """返回原始序列"""
    return series


def apply_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """百分比变化"""
    return series.pct_change(periods)


def apply_fracdiff(
    series: pd.Series, d: float = None, thres: float = FRACDIFF_THRES
) -> pd.Series:
    """
    使用 FFD 分数差分。
    """
    if d is None:
        opt_d = optimize_d(series, thres=thres, d_step=0.05, max_d=1.0, min_corr=0.9)
        d = opt_d

    if d == 0.0:
        return series.copy()

    return frac_diff_ffd(series, d=d, thres=thres)


def apply_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动标准化 (x - rolling_mean) / rolling_std"""
    rolling_mean = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std


# =============================================================================
# 特征名构建
# =============================================================================


def build_feature_name(
    col: str, transform: str, func_name: str, window: int = None
) -> str:
    """
    构建特征名。
    """
    if transform == "zscore":
        return f"feat_{col}_z{window}_{func_name}"
    elif transform == "pct_change":
        return f"feat_{col}_pct_{func_name}"
    elif transform == "fracdiff":
        return f"feat_{col}_fd_{func_name}"
    else:  # raw
        return f"feat_{col}_raw_{func_name}"


# =============================================================================
# 特征提取（针对单个时间片，直接计算，不使用 tsfresh）
# =============================================================================


def extract_features_from_slice(
    slice_values: np.ndarray,
    func_names: list,
) -> dict:
    """
    从单个时间片的值数组中提取统计特征（直接计算，不走 tsfresh）。

    :param slice_values: 时间片的值数组（已变换，已去除 NaN）
    :param func_names: 特征函数名列表
    :returns: {特征名: 值} 字典
    """
    if len(slice_values) < 3:
        return {fn: np.nan for fn in func_names}

    result = {}
    for fn in func_names:
        func = TSFRESH_FEATURE_FUNCS.get(fn)
        if func is None:
            result[fn] = np.nan
            continue
        try:
            val = func(slice_values)
            # 处理数值问题
            if not np.isfinite(val):
                val = np.nan
            result[fn] = val
        except Exception:
            result[fn] = np.nan

    return result


# =============================================================================
# 主提取循环
# =============================================================================


def extract_tsfresh_features(
    bars: pd.DataFrame,
    cusum_events: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    在所有 CUSUM 事件点上提取统计特征。

    :param bars: Dollar Bars DataFrame（已计算 log_close）
    :param cusum_events: CUSUM 事件 DataFrame
    :param config: TSFRESH_CONFIG 配置字典
    :returns: 特征 DataFrame，index 为事件时间戳，columns 为特征名
    """
    if config is None:
        config = TSFRESH_CONFIG

    lookback = config["lookback"]
    fracdiff_cols = config["fracdiff_cols"]
    zscore_windows = config["zscore_windows"]
    func_names = config["features"]

    # 输入列
    input_cols = ["close", "log_close", "volume", "open_interest"]
    available_cols = [c for c in input_cols if c in bars.columns]

    # 事件索引
    if "event_idx" in cusum_events.columns:
        event_indices = cusum_events["event_idx"].values.astype(np.int64)
    elif "idx" in cusum_events.columns:
        event_indices = cusum_events["idx"].values.astype(np.int64)
    elif "timestamp" in cusum_events.columns:
        event_timestamps = cusum_events["timestamp"].values
        bars_timestamps = bars.index.values
        event_indices = np.searchsorted(bars_timestamps, event_timestamps)
        print(f"  从 timestamp 映射事件索引: {len(event_indices)} 个事件")
    else:
        raise ValueError(
            "cusum_events must have 'event_idx', 'idx', or 'timestamp' column"
        )

    # 获取 fracdiff 的 d 值（使用前 80% 数据优化）
    fracdiff_d_values = {}
    split_idx = int(len(bars) * 0.8)
    for col in fracdiff_cols:
        if col in available_cols:
            series = bars[col].iloc[:split_idx]
            d = optimize_d(
                series, thres=FRACDIFF_THRES, d_step=0.05, max_d=1.0, min_corr=0.9
            )
            fracdiff_d_values[col] = d
            print(f"  FracDiff d for {col}: {d:.4f}")

    # 计算特征数量
    n_features = (
        len(available_cols) * len(func_names)  # raw
        + len(available_cols) * len(func_names)  # pct_change
        + len(fracdiff_cols) * len(func_names)  # fracdiff
        + len(available_cols) * len(zscore_windows) * len(func_names)  # zscore
    )

    total_events = len(event_indices)
    print(
        f"提取统计特征: {total_events} 个事件, lookback={lookback}, "
        f"特征数/事件={n_features}"
    )

    # 存储所有特征（使用字典列表，最后转 DataFrame）
    all_features = []

    for i, event_idx in enumerate(event_indices):
        if event_idx < lookback:
            continue

        start_idx = event_idx - lookback
        end_idx = event_idx + 1
        slice_df = bars.iloc[start_idx:end_idx]

        if len(slice_df) < lookback:
            continue

        event_timestamp = bars.index[event_idx]
        features = {"event_idx": event_idx, "timestamp": event_timestamp}

        # 对每列每种变换提取特征
        for col in available_cols:
            # --- raw ---
            transformed = slice_df[col].values
            feat_dict = extract_features_from_slice(transformed, func_names)
            for fn, val in feat_dict.items():
                features[build_feature_name(col, "raw", fn)] = val

            # --- pct_change ---
            transformed = slice_df[col].pct_change().dropna().values
            if len(transformed) >= 3:
                feat_dict = extract_features_from_slice(transformed, func_names)
            else:
                feat_dict = {fn: np.nan for fn in func_names}
            for fn, val in feat_dict.items():
                features[build_feature_name(col, "pct_change", fn)] = val

            # --- zscore (多个窗口) ---
            for window in zscore_windows:
                rolling_mean = (
                    slice_df[col].rolling(window, min_periods=window).mean().values
                )
                rolling_std = (
                    slice_df[col].rolling(window, min_periods=window).std().values
                )
                transformed = (slice_df[col].values - rolling_mean) / np.where(
                    rolling_std > 0, rolling_std, np.nan
                )
                # 去除因窗口不足产生的 NaN
                valid_mask = ~np.isnan(transformed)
                if valid_mask.sum() >= 3:
                    feat_dict = extract_features_from_slice(
                        transformed[valid_mask], func_names
                    )
                else:
                    feat_dict = {fn: np.nan for fn in func_names}
                for fn, val in feat_dict.items():
                    features[build_feature_name(col, "zscore", fn, window)] = val

        # fracdiff（仅针对指定列）
        for col in fracdiff_cols:
            if col not in available_cols:
                continue
            d = fracdiff_d_values.get(col, 0.0)
            if d == 0.0:
                transformed = slice_df[col].values
            else:
                fd_series = frac_diff_ffd(
                    slice_df[col], d=d, thres=FRACDIFF_THRES
                )
                # 对齐到原始索引
                aligned = slice_df[col].copy()
                aligned.iloc[:] = np.nan
                common_idx = aligned.index.intersection(fd_series.index)
                aligned.loc[common_idx] = fd_series.loc[common_idx].values
                transformed = aligned.values
            feat_dict = extract_features_from_slice(transformed, func_names)
            for fn, val in feat_dict.items():
                features[build_feature_name(col, "fracdiff", fn)] = val

        all_features.append(features)

        if (i + 1) % 200 == 0 or (i + 1) == total_events:
            print(f"  进度: {i + 1}/{total_events} 事件已处理")

    # 构建结果 DataFrame
    result_df = pd.DataFrame(all_features)
    result_df.set_index("timestamp", inplace=True)

    return result_df


# =============================================================================
# main
# =============================================================================


def main():
    """主函数流程"""
    print("=" * 60)
    print("tsfresh 风格统计特征提取")
    print("=" * 60)

    # 1. 加载 Dollar Bars
    bars_path = os.path.join(
        BARS_DIR, f"dollar_bars_target{TARGET_DAILY_BARS}.parquet"
    )
    print(f"\n1. 加载 Dollar Bars: {bars_path}")
    bars = pd.read_parquet(bars_path)
    print(f"   Shape: {bars.shape}")
    print(f"   Columns: {bars.columns.tolist()}")

    # 2. 添加 log_close 列
    if "log_close" not in bars.columns:
        bars["log_close"] = np.log(bars["close"])
        print("   添加 log_close 列")

    # 3. 加载 CUSUM 事件点
    cusum_path = os.path.join(FEATURES_DIR, "cusum_events.parquet")
    print(f"\n2. 加载 CUSUM 事件: {cusum_path}")
    cusum_events = pd.read_parquet(cusum_path)
    print(f"   Shape: {cusum_events.shape}")
    print(f"   Columns: {cusum_events.columns.tolist()}")

    # 4. 提取特征
    print(f"\n3. 提取统计特征...")
    tsfresh_features = extract_tsfresh_features(bars, cusum_events, TSFRESH_CONFIG)

    print(f"\n   提取完成!")
    print(f"   Shape: {tsfresh_features.shape}")
    print(f"   特征数: {len(tsfresh_features.columns)}")

    # 5. 保存结果
    output_path = os.path.join(FEATURES_DIR, "tsfresh_features.parquet")
    tsfresh_features.to_parquet(output_path)
    print(f"\n4. 保存到: {output_path}")

    # 统计信息
    nan_pct = (tsfresh_features.isna().sum() / len(tsfresh_features) * 100).mean()
    print(f"\n   平均 NaN 比例: {nan_pct:.1f}%")
    print(f"   非 NaN 特征数: {(~tsfresh_features.isna()).sum().sum()}")
    print(f"\n   特征表示例:")
    print(f"   {list(tsfresh_features.columns[:10])}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

    return tsfresh_features


if __name__ == "__main__":
    features = main()
