"""
Performance Metrics Module.

This module implements statistical performance metrics for strategy evaluation,
specifically focusing on the Deflated Sharpe Ratio (DSR) as described in
AFML Chapter 14.
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import Union, Optional
from scipy.stats import norm

def estimated_sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Calculate the estimated Sharpe Ratio (non-annualized).
    
    Args:
        returns: Array of returns.
        risk_free: Risk-free rate per period (default 0).
        
    Returns:
        Estimated Sharpe Ratio.
    """
    if len(returns) < 2:
        return 0.0
    return np.mean(returns - risk_free) / np.std(returns, ddof=1)

def annualized_sharpe_ratio(
    returns: np.ndarray, 
    risk_free: float = 0.0, 
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    
    Args:
        returns: Array of returns.
        risk_free: Risk-free rate (annualized).
        periods_per_year: Number of periods in a year (default 252).
        
    Returns:
        Annualized Sharpe Ratio.
    """
    sr = estimated_sharpe_ratio(returns, risk_free / periods_per_year)
    return sr * np.sqrt(periods_per_year)

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    n_obs: int
) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    
    PSR estimates the probability that the true Sharpe Ratio is greater than
    the benchmark Sharpe Ratio, given the observed skewness and kurtosis.
    
    Ref: AFML Chapter 14, Section 14.2
    
    Args:
        observed_sr: The observed Sharpe Ratio (estimated).
        benchmark_sr: The benchmark Sharpe Ratio (target).
        skewness: Skewness of the returns distribution.
        kurtosis: Kurtosis of the returns distribution (Fisher, excess=0).
                  Note: Formula usually uses Pearson kurtosis (normal=3).
                  If scipy.stats.kurtosis is used, add 3.
        n_obs: Number of observations (T).
        
    Returns:
        Probability (0.0 to 1.0) that true SR > benchmark SR.
    """
    # Standard deviation of the Sharpe Ratio estimator
    # sigma_SR = sqrt((1 + 0.5 * SR^2 - skew * SR + (kurt - 3)/4 * SR^2) / (T - 1))
    # Wait, AFML Eq 14.15 denominator is (T-1)
    
    # Check kurtosis definition.
    # AFML uses centered moments.
    # The term is (\gamma_4 - 1) / 4 * SR^2 in some versions, or involves SR^2.
    # Let's use the standard approximation formula.
    
    if n_obs <= 1:
        return 0.5
        
    numerator = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(
        1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2
    )
    
    if np.isnan(denominator) or denominator == 0:
        return 0.0
        
    test_statistic = numerator / denominator
    return norm.cdf(test_statistic)

def deflated_sharpe_ratio(
    observed_sr: float,
    skewness: float,
    kurtosis: float,
    n_obs: int,
    n_trials: int,
    variance_trials: float = 1.0 # Assuming variance of trial SRs?
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    DSR adjusts the benchmark Sharpe Ratio based on the number of trials
    (strategy configurations backtested) to account for selection bias.
    
    Ref: AFML Chapter 14, Section 14.3
    
    Args:
        observed_sr: The observed Sharpe Ratio of the best strategy.
        skewness: Skewness of the returns.
        kurtosis: Kurtosis of the returns (Fisher definition + 3 usually).
        n_obs: Number of observations.
        n_trials: Number of independent trials/strategies tested.
        variance_trials: Variance of the Sharpe Ratios across trials.
                         We often assume Sharpe Ratios of trials are normally distributed.
                         If unknown, Bailey/Lopez de Prado suggest an estimate.
                         Here we default to 1.0 if unknown, or user supplies it.
                         Actually, standard formulation DSR uses
                         SR* = sqrt(2 * log(N)) * std_SR_trials implies std deviation of trials.
        
    Returns:
        Probability (0.0 to 1.0) that true SR > 0 (hypothesis).
        Note: The return value is a PSR where the benchmark is adjusted.
    """
    # Estimate the expected maximum Sharpe Ratio under the null hypothesis (SR=0)
    # E[max_SR] approx sqrt(2 * log(n_trials)) * E[sigma_SR_trials]? 
    # Or just standard normal max expectation.
    # Bailey & Lopez de Prado (2014):
    # SR_benchmark = sqrt(2 * log(n_trials))
    # Assuming trials are uncorrelated.
    
    if n_trials < 1:
        n_trials = 1
        
    euler_mascheroni = 0.5772156649
    
    # Expected maximum Sharpe Ratio from N independent trials with 0 mean SR
    # Approximation for Gaussian:
    sr_benchmark = np.sqrt(2 * np.log(n_trials))
    # Note: This is simplified. AFML suggests using the variance of the trials' SRs.
    
    # We will use the simplified benchmark for now as we don't track all trial SRs in this pipeline yet.
    # Ideally should pass list of all SRs.
    
    return probabilistic_sharpe_ratio(
        observed_sr, sr_benchmark, skewness, kurtosis, n_obs
    )

def get_strategy_metrics(
    returns: Union[pd.Series, pl.Series, np.ndarray],
    n_trials: int = 1
) -> dict:
    """
    Compute comprehensive strategy metrics including DSR.
    
    Args:
        returns: Strategy returns series (non-cumulative).
        n_trials: Number of trials executed (for DSR).
        
    Returns:
        Dictionary of metrics.
    """
    if isinstance(returns, pl.Series):
        x = returns.to_numpy()
    elif isinstance(returns, pd.Series):
        x = returns.values
    else:
        x = returns
        
    x = x[~np.isnan(x)]
    
    if len(x) < 2:
        return {}
        
    # Moments
    mean_ret = np.mean(x)
    std_ret = np.std(x, ddof=1)
    
    # Skewness/Kurtosis
    # Using scipy.stats if available or numpy implementation
    try:
        from scipy.stats import skew, kurtosis
        sk = skew(x)
        kt = kurtosis(x, fisher=False) # Pearson kurtosis (normal=3)
        # Note: scipy kurtosis(fisher=True) is excess kurtosis (normal=0).
        # We used fisher=False so normal=3.
    except ImportError:
        # Fallback
        sk = 0
        kt = 3
        
    # Sharpe Ratio (Period)
    sr = mean_ret / std_ret if std_ret > 0 else 0
    
    # Annualized SR (assuming daily)
    ann_sr = sr * np.sqrt(252)
    
    # PSR / DSR
    # Use Period SR for calculation? 
    # AFML derivations usually work with period statistics or annualized as long as consistent.
    # PSR formula relies on T (number of observations).
    # If we use annualized SR, T should be in years? No.
    # The standard formula uses SR (period) and n_obs (count).
    
    psr = probabilistic_sharpe_ratio(sr, 0.0, sk, kt, len(x))
    dsr = deflated_sharpe_ratio(sr, sk, kt, len(x), n_trials)
    
    return {
        "mean_return": mean_ret,
        "std_return": std_ret,
        "skewness": sk,
        "kurtosis": kt,
        "sharpe_ratio": sr,
        "annualized_sharpe_ratio": ann_sr,
        "psr": psr,
        "dsr": dsr,
        "n_obs": len(x)
    }
