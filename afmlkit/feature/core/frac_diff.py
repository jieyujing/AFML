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
    min_corr: float = 0.0
) -> float:
    """
    Find the minimum d that makes the fractionally differentiated series stationary.

    :param series: The price series to apply FracDiff to
    :param thres: Threshold for weight cutoff in FFD
    :param d_step: Step size for exploring d values
    :param max_d: Maximum value for d to check
    :param min_corr: Minimum correlation coefficient with original series (0.0-1.0).
                     Set to 0.0 to disable correlation check.
                     Recommended: 0.7-0.8 for balanced memory retention.
    :return: optimal d value
    """
    # Track the best d that satisfies correlation constraint (for fallback)
    best_d_with_corr: float | None = None

    # Check if the original series is already stationary
    valid_series = series.dropna()
    if len(valid_series) > 10:
        p_val_orig = adfuller(valid_series)[1]
        if p_val_orig < 0.05:
            return 0.0

    for d in np.arange(0.01, max_d + d_step, d_step):
        diff_series = frac_diff_ffd(series, d=float(d), thres=thres)
        diff_series = diff_series.dropna()
        if len(diff_series) < 10:
            continue

        # ADF Test
        p_val = adfuller(diff_series)[1]

        if p_val < 0.05:
            # Check correlation with original series if min_corr is specified
            if min_corr > 0:
                # Align indices for correlation calculation
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

                        # Return immediately if correlation is sufficient
                        if corr >= min_corr:
                            return float(round(d, 4))
                        # Track this d as fallback (it's stationary, just not enough corr)
                        if best_d_with_corr is None:
                            best_d_with_corr = float(round(d, 4))
                        continue  # Try next d for better correlation
            else:
                # No correlation constraint, return immediately
                return float(round(d, 4))

    # If min_corr > 0 and no d satisfied both conditions:
    # Return the first stationary d found (best effort) or default to 1.0
    if min_corr > 0 and best_d_with_corr is not None:
        return best_d_with_corr

    # Default to 1.0 if no d < 1 creates stationarity
    return 1.0
