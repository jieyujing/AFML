"""
Bet Sizing

This module provides functions for translating Meta-Model probability outputs into
actionable, discrete position sizes, handling concurrent signals through averaging,
and discretizing allocations using a step-size.

Based on Advances in Financial Machine Learning, Chapter 10.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def get_signal_size(prob: float | pd.Series | np.ndarray, side: int | pd.Series | np.ndarray = 1) -> float | pd.Series | np.ndarray:
    """
    Convert raw Meta-Model probabilities to a base target size (m) using the standard normal CDF mapping.
    Includes a safety bound clip to prevent ZeroDivisionError during z-score calculation.
    
    Formula:
        p_clipped = np.clip(prob, 1e-5, 1 - 1e-5)
        z = (p_clipped - 0.5) / np.sqrt(p_clipped * (1.0 - p_clipped))
        m = 2 * norm.cdf(z) - 1
        s = m * side
    
    Parameters
    ----------
    prob : float or array-like
        The Meta-Model probability forecast P(bin=1).
    side : int or array-like, default 1
        The primary model's predicted side (e.g., 1 for long, -1 for short).
        
    Returns
    -------
    float or array-like
        The continuous base bet size s_i.
    """
    # Safety bound clip
    p_clipped = np.clip(prob, 1e-5, 1.0 - 1e-5)
    
    # Standardize to Z-score
    z = (p_clipped - 0.5) / np.sqrt(p_clipped * (1.0 - p_clipped))
    
    # Map to [-1, 1] range using CDF
    m = 2.0 * norm.cdf(z) - 1.0
    
    return m * side

def discretize_size(size: float | pd.Series | np.ndarray, step_size: float) -> float | pd.Series | np.ndarray:
    """
    Discretize the continuous aggregated target position using a configurable 
    step size to minimize excessive small trades.
    
    Formula:
        S_final = round(S / step_size) * step_size
        
    Parameters
    ----------
    size : float or array-like
        The continuous target position.
    step_size : float
        The minimum trade step size (e.g., 0.1, 0.05).
        
    Returns
    -------
    float or array-like
        The discretized position size.
    """
    if step_size <= 0:
        raise ValueError("step_size must be > 0.")
    return np.round(size / step_size) * step_size

def get_size_change_signals(sizes: pd.Series) -> pd.Series:
    """
    Identify points where the discretized size changes compared to the previous state.
    
    Parameters
    ----------
    sizes : pd.Series
        Series of discretized sizes over time.
        
    Returns
    -------
    pd.Series
        Boolean series indicating where a size change trigger occurs.
    """
    return sizes != sizes.shift(1).fillna(0)

def get_concurrent_sizes(
    sizes: pd.Series | np.ndarray, 
    t_events: pd.DatetimeIndex, 
    t_exits: pd.Series | pd.DatetimeIndex, 
    time_grid: pd.DatetimeIndex = None
) -> pd.Series:
    """
    Average continuous position sizes across concurrent overlapping signals.
    
    For a given time t, the active set of signals Active(t) consists of all signals k 
    where t_events[k] <= t < t_exits[k]. The aggregated size at t is the average 
    of the base sizes of all active signals at t.
    
    Parameters
    ----------
    sizes : pd.Series or np.ndarray
        Base sizes for each signal, aligned with t_events.
    t_events : pd.DatetimeIndex
        The start times of each signal.
    t_exits : pd.Series or pd.DatetimeIndex
        The exit times of each signal.
    time_grid : pd.DatetimeIndex, optional
        The time grid to evaluate the signal sizes on. 
        If None, evaluates on the union of t_events and t_exits.
        
    Returns
    -------
    pd.Series
        The aggregated position sizes evaluated on the time_grid.
    """
    if len(sizes) != len(t_events) or len(sizes) != len(t_exits):
        raise ValueError("sizes, t_events, and t_exits must have the same length.")
        
    t_exits_series = pd.Series(t_exits)
    
    if time_grid is None:
        grid_times = np.concatenate([t_events.values, t_exits_series.dropna().values])
        time_grid = pd.DatetimeIndex(np.unique(grid_times)).sort_values()
        
    if isinstance(sizes, pd.Series):
        s_vals = sizes.values
    else:
        s_vals = np.asarray(sizes)
        
    # Drop signals where exit is NaT
    valid_mask = pd.notna(t_exits_series.values)
    
    valid_events = t_events[valid_mask]
    valid_exits = t_exits_series.values[valid_mask]
    valid_sizes = s_vals[valid_mask]
    
    # 1. Start events
    starts_df = pd.DataFrame({'size': valid_sizes, 'count': 1}, index=valid_events)
    starts_df = starts_df.groupby(level=0).sum()
    
    # 2. Exit events
    exits_df = pd.DataFrame({'size': -valid_sizes, 'count': -1}, index=valid_exits)
    exits_df = exits_df.groupby(level=0).sum()
    
    # 3. Combine and align to grid
    net_df = starts_df.add(exits_df, fill_value=0)
    net_aligned = net_df.reindex(time_grid, fill_value=0)
    
    # 4. Cumulative sum to get active state at each time point
    active_sum = net_aligned['size'].cumsum()
    active_count = net_aligned['count'].cumsum()
    
    # 5. Calculate average size
    avg_size = active_sum / active_count
    # Handle division by zero where active_count == 0
    avg_size = avg_size.fillna(0.0)
    
    return avg_size
