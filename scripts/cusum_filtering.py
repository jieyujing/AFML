import os
import time
import pandas as pd
import numpy as np

from afmlkit.feature.core.volatility import ewms
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.sampling.filters import cusum_filter
from afmlkit.bar.data_model import TradesData
from afmlkit.label.kit import TBMLabel

def compute_dynamic_cusum_filter(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    vol_span: int = 50, 
    threshold_multiplier: float = 2.0,
    use_frac_diff: bool = True
) -> pd.DataFrame:
    """
    通过 CUSUM 对时间序列的微观波动去噪
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
    
    # 如果需要做 FracDiff，我们直接对价格（或者对数价格）做差分。
    # 最稳健的做法是：
    # 1. 波动率（Log Returns 的波动率）反映的是百分比变化率。
    # 2. 如果我们用 CUSUM 走自带的 log(p_i / p_i-1)，相当于在算百分比空间跑。
    # 3. 如果序列做了分数差分，序列可能含有负数，此时 cusum_filter 内部的 np.log 会崩溃或产生 NaN。
    # 所以，对于使用 Fractional Differentiation 的情况，必须单独定制 CUSUM。

    if use_frac_diff:
        print("Applying Fractional Differentiation on log prices...")
        price_series = np.log(df[price_col])
        opt_d = optimize_d(price_series)
        print(f"Optimal d for stationarity: {opt_d}")
        
        # Apply frac_diff and keep only valid indices
        stationary_prices = frac_diff_ffd(price_series, d=opt_d)
        
        # We need to align stationary_prices with our current DataFrame `df`
        # which has already been filtered by `valid_mask` (from volatility calc)
        common_indices = df.index.intersection(stationary_prices.index)
        
        # Re-filter df, stationary_prices, and volatility using common indices
        df = df.loc[common_indices].copy()
        
        # We need dynamic threshold to match `df`. `idx` maps to original raw log_ret/volatility arrays.
        # But we already filtered by `valid_mask` earlier when we created the subset `clean_volatility`.
        # It's cleaner to align from the `df` directly using its index positions in the original df.
        
        prices_for_cusum = stationary_prices.loc[common_indices].values.astype(np.float64)
        
        # Align volatility: find integer positions of common_indices in the currently filtered df
        valid_positions = df.index.get_indexer(common_indices)
        aligned_volatility = clean_volatility[valid_positions]
        dynamic_threshold = aligned_volatility * threshold_multiplier
        print(f"Aligned data points after FracDiff: {len(prices_for_cusum)}")
        
        # 定制的 FracDiff CUSUM，不再依赖底层的 log_returns
        # 因为此时的价格是经过差分的，本身就已经是“差分量”，我们可以直接对其累加。
        print("Applying FracDiff Custom CUSUM filter...")
        start_time = time.time()
        
        # 因为 frac_diff 已经是基于 log price 的差分序列了，直接取它的增量以判断累积和的变动
        diff_series = np.zeros_like(prices_for_cusum)
        diff_series[1:] = prices_for_cusum[1:] - prices_for_cusum[:-1]
        
        event_indices = cusum_filter(diff_series, dynamic_threshold)
        
        elapsed = time.time() - start_time
        print(f"FracDiff CUSUM filter completed in {elapsed:.4f} seconds.")
    else:
        prices_for_cusum = prices
        dynamic_threshold = clean_volatility * threshold_multiplier
        
        # 对于不使用 FracDiff 的情况，直接计算对数收益率传给外部的 CUSUM！
        diff_series = np.zeros_like(prices_for_cusum)
        with np.errstate(divide='ignore', invalid='ignore'):
            diff_series[1:] = np.log(prices_for_cusum[1:] / prices_for_cusum[:-1])
            
        diff_series = np.nan_to_num(diff_series, nan=0.0)
        
        # 5. CUSUM 过滤器
        print("Applying Standard CUSUM filter...")
        start_time = time.time()
        event_indices = cusum_filter(diff_series, dynamic_threshold)
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
    
    return filtered_df

def compute_labels_and_weights(
    full_df: pd.DataFrame, 
    sampled_df: pd.DataFrame,
    vol_col: str = 'volatility'
) -> pd.DataFrame:
    """
    为采样后的事件计算三重屏障标签和样本权重
    """
    print("\nStarting Labeling and Weighting...")
    
    # 1. 包装为 TradesData (TBM 需要该对象来评估价格路径)
    # 将 timestamp 转换为 Unix 纳秒整数
    ts_ns = pd.to_datetime(full_df['timestamp']).values.astype(np.int64)
    prices = full_df['close'].values.astype(np.float64)
    volumes = full_df['volume'].values.astype(np.float64)
    
    trades_data = TradesData(
        ts=ts_ns,
        px=prices,
        qty=volumes,
        timestamp_unit='ns',
        preprocess=False  # 数据已经是 Bars，无需再次 preprocessing
    )
    
    # 2. 准备特征数据 (sampled_df)，确保 DatetimeIndex
    features = sampled_df.copy()
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features['timestamp'])
        
    # 3. 三重屏障标签
    # 参数设置参考 AFML 最佳实践：
    # target_ret_col: 波动率, min_ret: 0, horizontal_barriers: (1, 1), vertical_barrier: 1 day
    tbm = TBMLabel(
        features=features,
        target_ret_col=vol_col,
        min_ret=0.0,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=pd.Timedelta(days=1)
    )
    
    print("Computing TBM labels...")
    _, labels_output = tbm.compute_labels(trades_data)
    
    # 4. 样本权重
    print("Computing sample weights...")
    weights_df = tbm.compute_weights(trades_data)
    
    # 5. 合并结果
    # labels_output 包含 'bin' (label), 't1' (touch_time), 'ret' (log_return)
    # weights_df 包含 'avg_uniqueness', 'return_attribution'
    result_df = pd.concat([labels_output, weights_df], axis=1)
    
    # 将 weight 合并回 sampled_df
    # 确保 sampled_df 也使用相同的 DatetimeIndex 来进行对齐
    sampled_df_aligned = sampled_df.copy()
    sampled_df_aligned.index = pd.to_datetime(sampled_df_aligned['timestamp'])
    
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
    df = pd.read_csv(input_file)
    
    if 'close' not in df.columns:
        print(f"Error: 'close' column not found in input data! Available columns: {df.columns.tolist()}")
        return
        
    # 计算波动率（哪怕不采样也需要它作为 TBM 的输入）
    prices = df['close'].values.astype(np.float64)
    log_ret = np.log(prices[1:] / prices[:-1])
    vol = ewms(np.insert(log_ret, 0, np.nan), span=50)
    df['volatility'] = vol
    
    # 执行 CUSUM 去噪采样
    sampled_df = compute_dynamic_cusum_filter(df, price_col='close', vol_span=50, threshold_multiplier=2.0)
    
    # 执行标签和权重计算
    final_df = compute_labels_and_weights(df, sampled_df, vol_col='volatility')
    
    print(f"Saving enriched data to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
