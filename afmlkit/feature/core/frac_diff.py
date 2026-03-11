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

def optimize_d(series: pd.Series, thres: float = 1e-4, d_step: float = 0.05, max_d: float = 1.0) -> float:
    """
    Find the minimum d that makes the fractionally differentiated series stationary.

    :param series: The price series to apply FracDiff to
    :param thres: Threshold for weight cutoff in FFD
    :param d_step: Step size for exploring d values
    :param max_d: Maximum value for d to check
    :return: optimal d value
    """
    # Check if the original series is already stationary
    valid_series = series.dropna()
    if len(valid_series) > 10:
        p_val_orig = adfuller(valid_series)[1]
        if p_val_orig < 0.05:
            return 0.0

    for d in np.arange(0.01, max_d + d_step, d_step):
        diff_series = frac_diff_ffd(series, d=d, thres=thres)
        diff_series = diff_series.dropna()
        if len(diff_series) < 10:
            continue

        # ADF Test
        p_val = adfuller(diff_series)[1]

        if p_val < 0.05:
            return round(d, 4)

    # Default to 1.0 if no d < 1 creates stationarity
    return 1.0
