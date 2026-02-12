"""
Stationarity Enforcement Module (Polars Optimized).

This module implements tools for ensuring time series stationarity,
a critical prerequisite for financial machine learning (AFML Chapter 2).

Key components:
- Fractional Differentiation (FFD) (AFML Chapter 5)
- Automated Min-d Search
- ADF Stationarity Testing
"""

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller
from typing import Union, Tuple, Optional


def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
    """
    Calculate weights for the underlying fractional differentiation series.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres or len(w) >= lim:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(
    series: Union[pl.Series, np.ndarray],
    d: float,
    thres: float = 1e-4
) -> np.ndarray:
    """
    Apply Fractional Differentiation (FFD) to a series.
    """
    if isinstance(series, pl.Series):
        x = series.to_numpy()
    else:
        x = series
        
    if d == 0:
        return x
            
    w = get_weights_ffd(d, thres, len(x))
    width = len(w)
    
    if width < 2:
        return x
        
    output = np.full_like(x, np.nan)
    
    for i in range(width - 1, len(x)):
        window = x[i - width + 1 : i + 1]
        output[i] = np.dot(window, w)
        
    return output


def check_stationarity(
    series: Union[pl.Series, np.ndarray],
    p_val_thres: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Check for stationarity using Augmented Dickey-Fuller (ADF) test.
    """
    if isinstance(series, pl.Series):
        x = series.drop_nulls().to_numpy()
    else:
        x = series[~np.isnan(series)]
        
    if len(x) < 20: 
        return False, 1.0, {}
        
    try:
        result = adfuller(x, maxlag=None, regression='c', autolag='AIC')
        p_value = result[1]
        stats = {
            'adf_stat': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_obs': result[3],
            'critical_values': result[4]
        }
        return p_value < p_val_thres, p_value, stats
    except Exception as e:
        print(f"ADF test failed: {e}")
        return False, 1.0, {}


def get_min_d(
    series: Union[pl.Series, np.ndarray],
    max_d: float = 1.0,
    step_size: float = 0.1,
    p_val_thres: float = 0.05,
    min_len: int = 100
) -> Tuple[float, float]:
    """
    Find minimum fractional differentiation d that satisfies stationarity.
    """
    if isinstance(series, pl.Series):
        x = series.to_numpy()
    else:
        x = series

    x = x[~np.isnan(x)]
    if len(x) < min_len:
        return 0.0, 1.0
        
    is_stat, p_val, _ = check_stationarity(x, p_val_thres)
    if is_stat:
        return 0.0, p_val
        
    possible_ds = np.arange(step_size, max_d + step_size/100, step_size)
    
    for d in possible_ds:
        d = round(d, 2)
        diff_x = frac_diff_ffd(x, d)
        diff_x_clean = diff_x[~np.isnan(diff_x)]
        
        is_stat, p_val, _ = check_stationarity(diff_x_clean, p_val_thres)
        if is_stat:
            return d, p_val
            
    return max_d, p_val


def get_stationarity_search_history(
    series: Union[pl.Series, np.ndarray],
    max_d: float = 1.0,
    step_size: float = 0.05,
    p_val_thres: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the history of ADF p-values for different d values.
    Used for visualization of the stationarity search process.
    """
    if isinstance(series, pl.Series):
        x = series.to_numpy()
    else:
        x = series

    x = x[~np.isnan(x)]
    
    d_vals = np.arange(0.0, max_d + step_size/100, step_size)
    p_vals = []
    
    for d in d_vals:
        if d == 0:
            _, p_val, _ = check_stationarity(x, p_val_thres)
        else:
            diff_x = frac_diff_ffd(x, d)
            diff_x_clean = diff_x[~np.isnan(diff_x)]
            _, p_val, _ = check_stationarity(diff_x_clean, p_val_thres)
        p_vals.append(p_val)
        
    return np.array(d_vals), np.array(p_vals)
