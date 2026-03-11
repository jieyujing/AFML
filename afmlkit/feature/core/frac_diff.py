import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_weights_ffd(d: float, thres: float, max_len: int = int(1e5)) -> np.ndarray:
    """
    Generate weights for fractional differentiation using Fixed-Width Window (FFD).
    
    :param d: fractional differencing order. 
    :param thres: threshold for weight cutoff.
    :param max_len: maximum length of the weight array to prevent infinite loop
    :return: array of weights for FFD
    """
    w = [1.0]
    k = 1
    
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres or k >= max_len:
            break
        w.append(w_)
        k += 1
        
    # Returns 1D array: [w_0, w_1, ..., w_k]
    # Which aligns perfectly with np.convolve calculating sum_{i} x[n-i] * w[i]
    return np.array(w)

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Compute Fractional Differentiation (FFD method).
    
    :param series: input pandas Series (e.g. prices)
    :param d: differencing order, typically 0 < d < 1
    :param thres: threshold for determining the window size
    :return: fractionally differentiated pandas Series
    """
    if d == 0.0:
        return series.copy()
        
    # 1. Compute weights
    w = get_weights_ffd(d, thres)
    width = len(w)
    
    if len(series) < width:
        return pd.Series(dtype=np.float64, index=series.index, name=series.name)
        
    # 2. Apply weights via highly optimized 1D convolution
    # mode='valid' computes the convolution only where the signals completely overlap.
    # This precisely avoids padding and skips the first `width - 1` elements.
    result_values = np.convolve(series.values, w, mode='valid')
    
    res = pd.Series(
        data=result_values, 
        index=series.index[width - 1:], 
        name=series.name
    )
    return res

def optimize_d(
    series: pd.Series,
    thres: float = 1e-4,
    d_step: float = 0.05,
    max_d: float = 1.0,
) -> float:
    """
    寻找使序列平稳的最小 d 值。

    :param series: 输入价格序列
    :param thres: FFD 权重截断阈值
    :param d_step: d 的搜索步长，默认 0.05
    :param max_d: d 的最大搜索值，默认 1.0（极端情况下可调至 2.0）
    :return: 最优 d 值
    """
    # 检查原始序列是否已平稳
    valid_series = series.dropna()
    if len(valid_series) > 10:
        p_val_orig = adfuller(valid_series)[1]
        if p_val_orig < 0.05:
            return 0.0

    # 网格搜索：d 从 0.0 开始，每次增加 d_step
    for d in np.arange(0.0, max_d + d_step, d_step):
        diff_series = frac_diff_ffd(series, d=float(d), thres=thres)
        diff_series = diff_series.dropna()
        if len(diff_series) < 10:
            continue

        # ADF 检验
        p_val = adfuller(diff_series)[1]

        # 熔断机制：p < 0.05 立即停止
        if p_val < 0.05:
            # 计算与原始序列的相关性（仅日志观察，不干预选择）
            common_idx = diff_series.index.intersection(series.index)
            if len(common_idx) >= 10:
                diff_aligned = diff_series.loc[common_idx]
                orig_aligned = series.loc[common_idx].dropna()
                common_idx = diff_aligned.index.intersection(orig_aligned.index)
                if len(common_idx) >= 10:
                    corr = np.corrcoef(
                        diff_aligned.loc[common_idx].values,
                        orig_aligned.loc[common_idx].values
                    )[0, 1]
                    # 日志输出相关性供观察
                    # print(f"[optimize_d] d*={d:.4f}, corr={corr:.4f}")
            return float(round(d, 4))

    # 若所有 d 都无法使序列平稳，返回上限值
    return max_d
