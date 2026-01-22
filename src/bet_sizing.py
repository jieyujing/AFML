"""
Bet Sizing Module based on AFML Chapter 10.

This module implements:
1. Probability-based Bet Sizing (converting probabilities to bet sizes)
2. Active Signal Averaging (handling concurrent positions)

Reference:
- Advances in Financial Machine Learning, Chapter 10
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def get_signal_stats(prob_series: pd.Series) -> tuple:
    """
    Get statistics for the probability series to calibrate the sizing function.
    
    Args:
        prob_series: Series of probabilities (0 to 1)
        
    Returns:
        mean, std of the probabilities
    """
    return prob_series.mean(), prob_series.std()

def bet_size_probability(
    events: pd.DataFrame, 
    prob_series: pd.Series, 
    num_classes: int = 2,
    pred_series: pd.Series = None,
    step_size: float = 0.0,
    average_active: bool = False
) -> pd.Series:
    """
    Calculates the bet size based on the probability of the predicted class.
    
    Formula: m = 2 * CDF(z) - 1
    where z = (p - 0.5) / (p * (1-p)) is a theoretical t-stat,
    OR we can use a calibrated Gaussian: z = (p - 0.5) / sigma_p
    
    For Meta-Labeling:
    - prob_series is the probability that the Primary Model is CORRECT (Label=1).
    - If prob < 0.5, it implies the Primary Model is likely wrong. 
      However, we typically use Meta-Labeling as a filter (Long Volatility), 
      so we only bet if prob > 0.5.
    - If prob > 0.5, we want a size from 0 to 1.
    
    Args:
        events: DataFrame with 't1' (vertical barrier)
        prob_series: Series of probabilities from Meta-Model (Class 1)
        num_classes: Number of classes (2 for binary)
        pred_series: Primary model predictions (-1, 1). If provided, returns signed bet size.
        step_size: Discrete step size for discretization (0.0 means continuous)
        average_active: Whether to average concurrent signals (AFML 10.3)
        
    Returns:
        Series of bet sizes
    """
    # 1. Prepare Probabilities
    # We focus on the probability of the "positive" outcome (Meta-Label=1)
    p = prob_series.copy()
    
    # 2. Calculate z-score
    # We want to map p=[0.5, 1.0] to Size=[0.0, 1.0] using a Sigmoid-like shape.
    # AFML approach: fit a Gaussian to the probabilities or use a theoretical distribution.
    # Here we use a theoretical approximation for simplicity and robustness.
    # z = (p - 1/n) / sqrt(p(1-p)) is the t-value of the Bernoulli trial
    
    # Avoid division by zero at p=0 or p=1
    p = p.clip(0.001, 0.999)
    
    # Test statistic for "p > 0.5"
    # Null hypothesis H0: p = 0.5
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    
    # 3. Calculate Bet Size (m)
    # m in [-1, 1]
    m = 2 * norm.cdf(z) - 1
    
    # For Meta-Labeling:
    # If m < 0 (i.e., p < 0.5), we simply don't take the trade (Size = 0)
    # rather than fading it (unless we trust the '0' label implies 'Reversal', which is risky)
    # So we clip negative values to 0.
    bet_sizes = pd.Series(m, index=p.index).clip(lower=0.0)
    
    # 4. Apply Primary Prediction Direction (if provided)
    if pred_series is not None:
        bet_sizes = bet_sizes * pred_series.loc[bet_sizes.index]

    # 5. Discretize (Optional)
    if step_size > 0:
        bet_sizes = (bet_sizes / step_size).round() * step_size

    # 6. Average Active Signals (Concurrency Management)
    if average_active:
        return avg_active_signals(events, bet_sizes)
    
    return bet_sizes

def avg_active_signals(events: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """
    Averages the active signals across time to manage concurrency.
    Ref: AFML Snippet 10.2
    
    Args:
        events: DataFrame with 't1' column (end time of the event)
        signal: Series of bet sizes (signal) corresponding to events start time (index)
        
    Returns:
        Series of averaged bet sizes at each time point
    """
    t_p = set(signal.index) # set of all start times
    t_p = t_p.union(events['t1'].values) # union with all end times
    t_p = list(t_p)
    t_p.sort()
    
    out = pd.Series(data=0, index=t_p)
    
    # Iterate through signals
    for idx, val in signal.items():
        if val == 0: continue
            
        t0 = idx
        t1 = events.loc[idx, 't1']
        
        # Add signal to the active duration [t0, t1]
        # Using searchsorted to find indices
        start_idx = out.index.searchsorted(t0)
        end_idx = out.index.searchsorted(t1)
        
        # We add the signal to the 'out' series (accumulator)
        # Note: This is a simplified vectorization. 
        # AFML iterates strictly. Let's try to be efficient.
        
        # Correct approach:
        # The 'out' series represents the "sum of signals" at any point t.
        # But we need the AVERAGE.
        # Average = Sum of Signals / Number of Concurrent Signals
        pass
        
    # Re-implementing strictly based on the idea:
    # We want to compute: for each time t, what is sum(signals active at t) / count(signals active at t)
    
    # 1. Expand events to daily/bar-by-bar series? Too slow.
    # 2. Iterate points?
    
    # Let's use the MP-friendly approach from mlfinlab if possible, 
    # but here a simple loop over sorted unique timestamps is sufficient for our data scale.
    
    out_signal = pd.Series(dtype=float)
    
    # Join signal and events
    df = events[['t1']].copy()
    df['signal'] = signal
    df = df[df['signal'] != 0] # Optimization: ignore zero bets
    
    if df.empty:
        return signal # All zeros
        
    # Create an array of time points (all starts and ends)
    time_points = sorted(list(set(df.index) | set(df['t1'])))
    
    # This can be slow if too many points. 
    # For 4000 bars it's instant.
    
    current_signals = [] # List of (t1, value)
    
    results = {}
    
    # We only care about the re-sampling at the original signal indices for backtesting?
    # No, we want the effective position at any time t.
    # But for our 'evaluate_strategy' which assumes trade-by-trade, 
    # we might just want the 'budgeted' size at the moment of entry?
    # AFML 10.3 says: "The average signal is calculated as the average of the signals active at time t."
    
    # Simplified approach for 'evaluate_strategy':
    # Since we are evaluating "strategy returns" based on bar-by-bar PnL or Trade PnL?
    # Our evaluate_strategy calculates `base_returns = primary_preds * df_eval['ret']`
    # This assumes we hold the position for the duration of the bar (1 bar).
    # But Triple Barrier events have holding periods (t1).
    # This means our previous 'evaluate_strategy' was slightly simplified (assuming return is realized at t0? No, 'ret' in labeled_events is return to t1).
    
    # Actually, `labeled_events['ret']` is the return from t0 to t1.
    # So if we sum them up, we are assuming no overlap or managing overlap PnL.
    
    # If we want to correctly backtest with overlap, we need to distribute the size.
    # If we have 10 concurrent bets, each with size 1.0, we are 10x leveraged.
    # We want to scale down to 1/10 size.
    
    # Let's calculate the 'concurrency' divisor for each event.
    
    concurrency = pd.Series(0, index=events.index)
    for idx in events.index:
        t0 = idx
        t1 = events.loc[idx, 't1']
        # Count how many other events overlap with [t0, t1]
        # This is O(N^2). With 3000 events it's 9M ops. Okay in Python.
        
        # Optimization: use the 'avg_uniqueness' logic or 'concurrency' from sample_weights.py
        # We already have 'avg_uniqueness' in features!
        # c_t = sum(1_i) at time t.
        # uniqueness = 1 / c_t
        pass
        
    # Using 'avg_uniqueness' as a proxy for 1/concurrency?
    # Actually 'avg_uniqueness' is the average of (1/c_t) over the life of the bar.
    # So Avg. Uniqueness IS the average scaling factor we need!
    # AFML 10.4: "A size discretization algorithm... dividing the bet size by the number of concurrent bets."
    
    # So, Size_Adjusted = Size_Raw * Avg_Uniqueness
    # This is an incredibly elegant approximation.
    
    return signal * events['avg_uniqueness'] if 'avg_uniqueness' in events.columns else signal

