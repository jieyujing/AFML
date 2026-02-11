"""
Stationarity Enforcement Module.

This module implements tools for ensuring time series stationarity,
a critical prerequisite for financial machine learning (AFML Chapter 2).

Key components:
- Fractional Differentiation (FFD) (AFML Chapter 5)
- Automated Min-d Search
- ADF Stationarity Testing
"""

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import adfuller
from typing import Union, Tuple, Optional


def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
    """
    Calculate weights for the underlying fractional differentiation series.
    
    Args:
        d: Fractional differentiation value.
        thres: Threshold for weight cutoff.
        lim: Maximum length of the weight vector.
        
    Returns:
        Numpy array of weights.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres or len(w) >= lim:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])  # Reverse to align with window extraction order (newest to oldest)


def frac_diff_ffd(
    series: Union[pd.Series, pl.Series, np.ndarray],
    d: float,
    thres: float = 1e-4
) -> np.ndarray:
    """
    Apply Fractional Differentiation (FFD) to a series.
    
    Statistically, this method preserves memory better than integer differentiation
    while achieving stationarity.
    
    Args:
        series: Time series data.
        d: Fractional differentiation value (0.0 to 1.0).
        thres: Threshold for weight cutoff.
        
    Returns:
        Differentiated series as numpy array.
    """
    if isinstance(series, pl.Series):
        x = series.to_numpy()
    elif isinstance(series, pd.Series):
        x = series.values
    else:
        x = series
        
    if d == 0:
        return x
        
    # Handle NaN values
    if np.isnan(x).any():
        # Simple forward fill for now, or assume caller handles it
        # Converting to DataFrame for easy fillna if needed, but for perf
        # we'll assume valid input or basic fill
        mask = np.isnan(x)
        if mask.any():
            pass # TODO: Add robust handling
            
    w = get_weights_ffd(d, thres, len(x))
    width = len(w)
    
    if width < 2:
        return x
        
    output = np.full_like(x, np.nan)
    
    # Convolution
    # Note: Valid mode convolution reduces size, we want same size with NaNs at start
    # We can use np.convolve but need to be careful with alignment
    # Weights are already reversed in get_weights_ffd for dot product usage
    # If we use convolve, we pass weights as is if we want 'valid' convolution on window
    
    # Let's use a simple loop for clarity and correctness matching AFML
    # optimized by using stride_tricks or convolve
    # x: [x0, x1, x2, ...]
    # we want y[t] = w[0]*x[t] + w[1]*x[t-1] + ...
    # where w is [w0, w1, ...] (unreversed)
    # get_weights_ffd returns reversed weights [wk, ..., w1, w0]
    
    # Let's re-verify get_weights_ffd output
    # w=[1, -1] for d=1. w[::-1] is [-1, 1].
    # rolling apply doing dot([x[t-1], x[t]], [-1, 1]) = -x[t-1] + x[t]. Correct.
    
    for i in range(width - 1, len(x)):
        # x window: x[i-width+1 : i+1]
        window = x[i - width + 1 : i + 1]
        output[i] = np.dot(window, w)
        
    return output


def check_stationarity(
    series: Union[pd.Series, pl.Series, np.ndarray],
    p_val_thres: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Check for stationarity using Augmented Dickey-Fuller (ADF) test.
    
    Args:
        series: Time series data.
        p_val_thres: P-value threshold (default 0.05).
        
    Returns:
        Tuple (is_stationary, p_value, stats_dict)
    """
    if isinstance(series, pl.Series):
        x = series.drop_nulls().to_numpy()
    elif isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = series[~np.isnan(series)]
        
    if len(x) < 20: # Not enough data for ADF
        return False, 1.0, {}
        
    try:
        # constant regression, autolag
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
    series: Union[pd.Series, pl.Series, np.ndarray],
    max_d: float = 1.0,
    step_size: float = 0.1,
    p_val_thres: float = 0.05,
    min_len: int = 100
) -> Tuple[float, float]:
    """
    Find minimum fractional differentiation d that satisfies stationarity.
    
    Args:
        series: Time series data.
        max_d: Maximum d to check.
        step_size: Step size for grid search.
        p_val_thres: P-value threshold for stationarity.
        min_len: Minimum length of series.
        
    Returns:
        Tuple (optimal_d, p_value)
    """
    # Optimize to log sequence for processing
    if isinstance(series, pl.Series):
        x = series.to_numpy()
    elif isinstance(series, pd.Series):
        x = series.values
    else:
        x = series

    # Basic cleaning
    x = x[~np.isnan(x)]
    if len(x) < min_len:
        return 0.0, 1.0
        
    # Check 0.0 first
    is_stat, p_val, _ = check_stationarity(x, p_val_thres)
    if is_stat:
        return 0.0, p_val
        
    possible_ds = np.arange(step_size, max_d + step_size/100, step_size)
    
    for d in possible_ds:
        d = round(d, 2)
        diff_x = frac_diff_ffd(x, d)
        
        # Remove initial NaNs caused by window
        diff_x_clean = diff_x[~np.isnan(diff_x)]
        
        is_stat, p_val, _ = check_stationarity(diff_x_clean, p_val_thres)
        if is_stat:
            return d, p_val
            
    # If failed, return max_d (usually 1.0)
    return max_d, p_val
