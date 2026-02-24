import os
import time
import pandas as pd
import numpy as np

from afmlkit.feature.core.volatility import ewms
from afmlkit.sampling.filters import cusum_filter

def compute_dynamic_cusum_filter(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    vol_span: int = 50, 
    threshold_multiplier: float = 2.0
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
    
    # 5. CUSUM 过滤器
    print("Applying CUSUM filter...")
    start_time = time.time()
    event_indices = cusum_filter(prices, dynamic_threshold)
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
        
    # 执行 CUSUM 去噪采样
    filtered_df = compute_dynamic_cusum_filter(df, price_col='close', vol_span=50, threshold_multiplier=2.0)
    
    print(f"Saving filtered data to {output_file}...")
    filtered_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
