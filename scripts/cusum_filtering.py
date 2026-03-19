import os
import time
from typing import Union, Tuple

import pandas as pd
import numpy as np

from afmlkit.feature.core.volatility import ewms
from afmlkit.feature.core.trend_scan import trend_scan_labels
from afmlkit.sampling.filters import cusum_filter, cusum_filter_with_state
from afmlkit.bar.data_model import TradesData
from afmlkit.label.kit import TBMLabel

def compute_dynamic_cusum_filter(
    df: pd.DataFrame,
    price_col: str = 'close',
    vol_span: int = 50,
    threshold_multiplier: float = 2.0,
    return_state: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DatetimeIndex],
    Tuple[pd.DataFrame, pd.DatetimeIndex, dict]
]:
    """
    通过 CUSUM 对时间序列的微观波动去噪。

    :param df: 输入数据框，包含价格和成交量列。
    :param price_col: 价格列名。
    :param vol_span: 波动率计算的指数加权窗口。
    :param threshold_multiplier: 动态阈值乘数。
    :param return_state: 如果为 True，返回 CUSUM 状态历史用于可视化。
    :returns:
        - return_state=False: (filtered_df, t_events)
        - return_state=True: (filtered_df, t_events, cusum_state)
    """
    print("Starting CUSUM filtering...")
    print(f"Parameters: price_col={price_col}, vol_span={vol_span}, threshold_multiplier={threshold_multiplier}")

    # 1. & 3. 提取 float64 格式的价格数组
    prices = df[price_col].values.astype(np.float64)
    n = len(prices)
    if n < 2:
        print("Data length is too short.")
        return df.copy()

    # 2. 计算 Log Returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(prices[1:] / prices[:-1])

    # 3. 计算指数加权标准差 (EWMS)，并处理 NaN
    print(f"Calculating rolling volatility (span={vol_span})...")
    volatility = ewms(log_ret, span=vol_span)

    # 前几项可能是 NaN，严格按 AFML 方法论：不做前向/后向填充，直接丢弃 (dropna) 无效数据！
    valid_mask = ~np.isnan(volatility)
    if not np.any(valid_mask):
        raise ValueError("Calculated volatility is all NaNs.")

    # 直接舍弃掉没有足够历史数据来计算波动率的那些观测行
    df = df.iloc[valid_mask].copy()
    prices = prices[valid_mask]
    clean_volatility = volatility[valid_mask]

    # 防止波动率为 0 导致后续无事件
    clean_volatility = np.maximum(clean_volatility, 1e-8)

    # 4. 生成动态阈值
    dynamic_threshold = clean_volatility * threshold_multiplier

    # 计算对数收益率传给 CUSUM
    diff_series = np.zeros_like(prices)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_series[1:] = np.log(prices[1:] / prices[:-1])

    diff_series = np.nan_to_num(diff_series, nan=0.0)

    # 5. CUSUM 过滤器
    print("Applying CUSUM filter...")
    start_time = time.time()

    if return_state:
        event_indices, s_pos_history, s_neg_history, threshold_arr = cusum_filter_with_state(
            diff_series, dynamic_threshold
        )
        # 初始化 s_pos/s_neg[0]
        s_pos_history[0] = 0.0
        s_neg_history[0] = 0.0
        cusum_state = {
            's_pos': s_pos_history,
            's_neg': s_neg_history,
            'threshold': float(np.mean(threshold_arr)),
            'time_index': df.index
        }
    else:
        event_indices = cusum_filter(diff_series, dynamic_threshold)
        cusum_state = None

    elapsed = time.time() - start_time
    print(f"CUSUM filter completed in {elapsed:.4f} seconds.")

    # 6. 数据提取与保存
    filtered_df = df.iloc[event_indices].copy()
    
    original_size = len(df)
    filtered_size = len(filtered_df)
    retention_ratio = filtered_size / original_size if original_size > 0 else 0
    
    print(f"Original records: {original_size}")
    print(f"Sampled events  : {filtered_size}")
    print(f"Compression ratio (Sampled/Original): {retention_ratio:.4f}")
    
    # 7. 导出 t_events — 干净的 DatetimeIndex 供下游 Trend Scan 使用
    if 'timestamp' in filtered_df.columns:
        t_events = pd.DatetimeIndex(pd.to_datetime(filtered_df['timestamp']))
    elif isinstance(filtered_df.index, pd.DatetimeIndex):
        t_events = filtered_df.index
    else:
        raise ValueError(
            "Cannot extract t_events: filtered_df has no 'timestamp' column "
            "and its index is not a DatetimeIndex."
        )
    
    # 时间因果性校验：确保 t_events 单调递增（CUSUM 保证了每个事件严格
    # 来源于截至该时刻的历史累积和，不包含未来信息）
    if not t_events.is_monotonic_increasing:
        print("WARNING: t_events is not monotonically increasing. Sorting...")
        t_events = t_events.sort_values()
    
    print(f"Exported t_events: {len(t_events)} discrete timestamps")

    if return_state:
        return filtered_df, t_events, cusum_state
    else:
        return filtered_df, t_events

def compute_labels_and_weights(
    full_df: pd.DataFrame, 
    sampled_df: pd.DataFrame,
    vol_col: str = 'volatility',
    trend_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    为采样后的事件计算三重屏障标签和样本权重。
    
    当提供 trend_df（来自 Trend Scan）时，启用 Meta-Labeling：
    - side 列由 Trend Scan 主模型提供
    - |t_value| 作为 sample weight 乘数
    
    Parameters
    ----------
    full_df : pd.DataFrame
        完整的 Bar 数据，含 'timestamp', 'close', 'volume' 列。
    sampled_df : pd.DataFrame
        CUSUM 采样后的事件子集。
    vol_col : str
        波动率列名，用于三重屏障的 target return。
    trend_df : pd.DataFrame, optional
        来自 trend_scan_labels() 的输出 DataFrame，索引为事件时间戳，
        含 'side' 和 't_value' 列。若提供则启用 Meta-Labeling。
    """
    is_meta = trend_df is not None
    mode_str = "Meta-Labeling" if is_meta else "Standard"
    print(f"\nStarting {mode_str} Labeling and Weighting...")

    # 1. 包装为 TradesData (TBM 需要该对象来评估价格路径)
    # 将 timestamp 转换为 Unix 纳秒整数
    if 'timestamp' in full_df.columns:
        ts_ns = pd.to_datetime(full_df['timestamp']).values.astype(np.int64)
    elif isinstance(full_df.index, pd.DatetimeIndex):
        ts_ns = full_df.index.values.astype(np.int64)
    else:
        raise ValueError("full_df must have 'timestamp' column or DatetimeIndex")

    prices = full_df['close'].values.astype(np.float64)
    volumes = full_df['volume'].values.astype(np.float64)

    trades_data = TradesData(
        ts=ts_ns,
        px=prices,
        qty=volumes,
        timestamp_unit='ns',
        preprocess=False
    )

    # 2. 准备特征数据 (sampled_df)，确保 DatetimeIndex
    features = sampled_df.copy()
    if not isinstance(features.index, pd.DatetimeIndex):
        if 'timestamp' in features.columns:
            features.index = pd.to_datetime(features['timestamp'])
        else:
            features.index = pd.to_datetime(features.index)
    
    # 3. 如果有 Trend Scan 输出，注入 side 列启用 Meta-Labeling
    if is_meta:
        # 对齐 trend_df 和 features 的索引
        common_idx = features.index.intersection(trend_df.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No overlapping indices between features and trend_df. "
                "Check timestamp alignment."
            )
        features = features.loc[common_idx]
        features['side'] = trend_df.loc[common_idx, 'side'].astype(int)
        
        # 过滤掉 side == 0 的事件（零方差横盘，无方向信号）
        zero_side_count = (features['side'] == 0).sum()
        if zero_side_count > 0:
            print(f"Dropping {zero_side_count} events with side=0 (flat/zero-variance)")
            features = features[features['side'] != 0]
        
        print(f"Meta-Labeling enabled: {len(features)} events with primary model side")
    
    # 4. 三重屏障标签
    # 参数设置参考 AFML 最佳实践：
    # target_ret_col: 波动率, min_ret: 0, horizontal_barriers: (1, 1), vertical_barrier: 1 day
    tbm = TBMLabel(
        features=features,
        target_ret_col=vol_col,
        min_ret=0.0,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=pd.Timedelta(days=2),
        is_meta=is_meta,
    )
    
    print("Computing TBM labels...")
    _, labels_output = tbm.compute_labels(trades_data)
    
    # 5. 样本权重
    print("Computing sample weights...")
    weights_df = tbm.compute_weights(trades_data)
    
    # 6. 合并结果
    # labels_output 包含 'bin' (label), 't1' (touch_time), 'ret' (log_return)
    # weights_df 包含 'avg_uniqueness', 'return_attribution'
    result_df = pd.concat([labels_output, weights_df], axis=1)
    
    # 7. 集成 Trend Scan 的 t_value 作为样本权重调制
    if is_meta and trend_df is not None:
        # 在 result_df 对齐后的索引上提取 |t_value|
        aligned_t_values = trend_df.loc[
            result_df.index.intersection(trend_df.index), 't_value'
        ].abs()
        result_df['trend_confidence'] = aligned_t_values
        
        # 标准化 t_value → [0, 1] 并乘入样本权重
        if 'avg_uniqueness' in result_df.columns:
            t_abs = result_df['trend_confidence'].fillna(0.0)
            t_max = t_abs.max()
            if t_max > 0:
                t_normalized = t_abs / t_max
            else:
                t_normalized = pd.Series(1.0, index=result_df.index)
            result_df['trend_weighted_uniqueness'] = (
                result_df['avg_uniqueness'] * t_normalized
            )
            print(
                f"Trend confidence integrated: "
                f"mean |t_value|={t_abs.mean():.4f}, "
                f"max |t_value|={t_max:.4f}"
            )
    
    # 将 weight 合并回 sampled_df
    # 确保 sampled_df 也使用相同的 DatetimeIndex 来进行对齐
    sampled_df_aligned = sampled_df.copy()
    if not isinstance(sampled_df_aligned.index, pd.DatetimeIndex):
        if 'timestamp' in sampled_df_aligned.columns:
            sampled_df_aligned.index = pd.to_datetime(sampled_df_aligned['timestamp'])
        else:
            sampled_df_aligned.index = pd.to_datetime(sampled_df_aligned.index)

    # 排除掉因为 TBM evaluation window 被 drop 掉的尾部事件
    final_df = sampled_df_aligned.loc[result_df.index].copy()
    for col in result_df.columns:
        final_df[col] = result_df[col]

    print(f"Final dataset with weights: {len(final_df)} rows")
    return final_df

def main():
    input_file = "outputs/dollar_bars/dollar_bars_freq20.csv"
    output_dir = "outputs/dollar_bars"
    output_file = f"{output_dir}/cusum_sampled_bars.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
        
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            print("Error: Cannot determine timestamp index from input file.")
            return

    if 'close' not in df.columns:
        print(f"Error: 'close' column not found in input data! Available columns: {df.columns.tolist()}")
        return
        
    # 计算波动率（哪怕不采样也需要它作为 TBM 的输入）
    prices = df['close'].values.astype(np.float64)
    log_ret = np.log(prices[1:] / prices[:-1])
    vol = ewms(np.insert(log_ret, 0, np.nan), span=50)
    df['volatility'] = vol
    
    # 假设均值是 1.2 hrs/bar，一天大约 20 个 bars。将 1-bar 波动率缩放到日度波动率
    df['daily_vol_est'] = vol * np.sqrt(20)
    
    # ======== CUSUM → Trend Scan → Meta-Labeling Pipeline ========
    
    # Step 1: CUSUM 去噪采样 — 返回 (sampled_df, t_events)
    sampled_df, t_events = compute_dynamic_cusum_filter(
        df, price_col='close', vol_span=50, threshold_multiplier=2.0
    )
    
    # Step 2: Trend Scan Primary Model — 动态趋势方向与置信度
    # 构建 DatetimeIndex 的价格序列（Trend Scan 需要完整连续价格）
    if 'timestamp' in df.columns:
        price_index = pd.to_datetime(df['timestamp'])
    elif isinstance(df.index, pd.DatetimeIndex):
        price_index = df.index
    else:
        print("Error: Cannot determine price index for Trend Scan.")
        return

    price_series = pd.Series(
        df['close'].values,
        index=price_index,
        name='close',
    )
    
    print("\n" + "=" * 60)
    print("Running Trend Scan Primary Model...")
    print("=" * 60)
    # Align L_windows to approximate 1-2 days (max 40 bars = ~48 hours)
    trend_df = trend_scan_labels(
        price_series=price_series,
        t_events=t_events,
        L_windows=[5, 10, 15, 20, 25, 30, 40],
    )
    
    # 打印 Trend Scan 概要
    print(f"\nTrend Scan Results:")
    print(f"  Events analyzed : {len(trend_df)}")
    print(f"  Mean |t_value|  : {trend_df['t_value'].abs().mean():.4f}")
    if 'side' in trend_df.columns:
        print(f"  Side distribution: {dict(trend_df['side'].value_counts())}")
        print(f"  Most common L   : {trend_df['t1'].mode().iloc[0] if len(trend_df) > 0 else 'N/A'}")
    
    # Step 3: Meta-Labeling — Trend Scan Side + t_value 驱动三重屏障
    # Using the daily volatility equivalent for target return so price doesn't hit TP/SL in 1 bar
    final_df = compute_labels_and_weights(
        df, sampled_df, vol_col='daily_vol_est', trend_df=trend_df
    )
    
    print(f"\nSaving enriched data to {output_file}...")
    final_df.to_csv(output_file, index=True)
    print("Done.")

if __name__ == "__main__":
    main()
