"""
Backtest Statistics Module based on AFML Chapter 14.

This module implements:
1. Probabilistic Sharpe Ratio (PSR) - Statistical significance of Sharpe ratio
2. Deflated Sharpe Ratio (DSR) - Adjusted for multiple testing
3. Minimum Track Record Length (MinTRL)

Reference:
- Advances in Financial Machine Learning, Chapter 14
- mlfinlab.backtest_statistics
"""

import pandas as pd
import numpy as np
from scipy import stats


def sharpe_ratio(
    returns: pd.Series,
    entries_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Reference: AFML Chapter 14.
    
    Args:
        returns: Series of returns (log or simple)
        entries_per_year: Number of observations per year (252 for daily)
        risk_free_rate: Risk-free rate for the same period
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Annualize
    sharpe_annualized = sharpe * np.sqrt(entries_per_year)
    
    return sharpe_annualized


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    num_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    PSR provides an adjusted estimate of SR by removing the inflationary effect
    caused by short series with skewed and/or fat-tailed returns.
    
    PSR answers: "What is the probability that the observed SR exceeds the benchmark?"
    A PSR > 0.95 indicates statistical significance at the 5% level.
    
    Reference: AFML Chapter 14, Snippet 14.1.
    
    Args:
        observed_sr: The observed (annualized) Sharpe ratio
        benchmark_sr: The benchmark Sharpe ratio to test against (e.g., 0)
        num_returns: Number of return observations
        skewness: Skewness of returns (0 for normal)
        kurtosis: Kurtosis of returns (3 for normal)
        
    Returns:
        Probability (0-1) that true SR exceeds benchmark
    """
    # Avoid division by zero
    if num_returns <= 1:
        return 0.5
        
    # Standard error of Sharpe ratio estimate under non-normality
    # From Lo (2002) and Bailey & López de Prado
    std_sr = np.sqrt(
        (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2)
        / (num_returns - 1)
    )
    
    if std_sr == 0 or np.isnan(std_sr):
        return 0.5
    
    # Z-score
    z = (observed_sr - benchmark_sr) / std_sr
    
    # Probability from standard normal CDF
    psr = stats.norm.cdf(z)
    
    return psr


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: list,
    num_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    return_benchmark: bool = False
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR).
    
    DSR is a PSR where the benchmark is adjusted to reflect the multiplicity of trials.
    It corrects SR for:
    - Non-normal returns
    - Track record length
    - Multiple testing / selection bias
    
    A DSR > 0.95 indicates statistical significance at the 5% level,
    accounting for the number of strategies tried.
    
    Reference: AFML Chapter 14.
    
    Args:
        observed_sr: The observed Sharpe ratio
        sr_estimates: List of Sharpe ratios from all trials (or [std, n_trials])
        num_returns: Number of return observations
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns
        return_benchmark: If True, return the calculated benchmark SR instead
        
    Returns:
        Deflated Sharpe Ratio (probability) or Benchmark SR
    """
    if isinstance(sr_estimates, list) and len(sr_estimates) >= 2:
        # Calculate Benchmark SR from the distribution of trials
        # Using the formula from Bailey & López de Prado (2014)
        sr_std = np.std(sr_estimates)
        n_trials = len(sr_estimates)
    else:
        # Assume sr_estimates = [std_of_estimates, n_trials]
        sr_std = sr_estimates[0]
        n_trials = sr_estimates[1]
    
    # Expected maximum Sharpe ratio from N trials of a null strategy (SR=0)
    # E[max(SR)] ≈ (1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    
    euler_gamma = 0.5772156649
    
    if n_trials <= 1:
        benchmark_sr = 0.0
    else:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        benchmark_sr = sr_std * ((1 - euler_gamma) * z1 + euler_gamma * z2)
    
    if return_benchmark:
        return benchmark_sr
    
    # Return PSR with the deflated benchmark
    return probabilistic_sharpe_ratio(
        observed_sr, benchmark_sr, num_returns, skewness, kurtosis
    )


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05
) -> float:
    """
    Calculate Minimum Track Record Length (MinTRL).
    
    MinTRL answers: "How long should the track record be to have statistical
    confidence that the observed Sharpe ratio exceeds the benchmark?"
    
    Reference: AFML Chapter 14.
    
    Args:
        observed_sr: Observed (annualized) Sharpe ratio
        benchmark_sr: Benchmark Sharpe ratio
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns
        alpha: Significance level (default 5%)
        
    Returns:
        Minimum number of observations required
    """
    if observed_sr <= benchmark_sr:
        return float('inf')
        
    z_alpha = stats.norm.ppf(1 - alpha)
    
    min_trl = 1 + (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) * \
              (z_alpha / (observed_sr - benchmark_sr))**2
              
    return min_trl


def drawdown_and_time_under_water(returns: pd.Series) -> tuple:
    """
    Calculate drawdowns and time under water.
    
    Reference: AFML Chapter 14.
    
    Args:
        returns: Series of cumulative returns or prices
        
    Returns:
        Tuple of (max_drawdown, time_under_water_in_periods)
    """
    # Calculate running maximum (high water mark)
    cum_returns = (1 + returns).cumprod()
    hwm = cum_returns.expanding().max()
    
    # Drawdown series
    drawdowns = cum_returns / hwm - 1
    
    # Max drawdown
    max_dd = drawdowns.min()
    
    # Time under water (periods below HWM)
    underwater = drawdowns < 0
    time_under_water = underwater.sum()
    
    return max_dd, time_under_water


def bets_concentration(returns: pd.Series) -> float:
    """
    Calculate concentration of returns using Herfindahl-Hirschman Index.
    
    Reference: AFML Snippet 14.3.
    
    Args:
        returns: Series of returns from bets
        
    Returns:
        Concentration metric (0 = perfectly diversified, 1 = single bet)
    """
    if len(returns) <= 2 or returns.sum() == 0:
        return float('nan')
        
    weights = returns.abs() / returns.abs().sum()
    hhi = (weights ** 2).sum()
    
    # Normalize: 0 = uniform, 1 = concentrated
    n = len(returns)
    hhi_normalized = (hhi - 1/n) / (1 - 1/n)
    
    return hhi_normalized


def compute_all_statistics(
    returns: pd.Series,
    n_trials: int = 1,
    entries_per_year: int = 252 * 4,  # 4 bars/day for dollar bars
    benchmark_sr: float = 0.0
) -> dict:
    """
    Compute comprehensive backtest statistics.
    
    Reference: AFML Chapter 14.
    
    Args:
        returns: Series of strategy returns (per-trade or per-bar)
        n_trials: Number of strategies/parameter sets tried
        entries_per_year: Annualization factor
        benchmark_sr: Benchmark Sharpe ratio
        
    Returns:
        Dictionary of statistics
    """
    # Basic stats
    n = len(returns)
    total_return = returns.sum()
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Sharpe Ratio
    sr = sharpe_ratio(returns, entries_per_year)
    
    # Higher moments
    skew = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna()) + 3  # Fisher kurtosis + 3 for excess
    
    # PSR
    psr = probabilistic_sharpe_ratio(sr, benchmark_sr, n, skew, kurt)
    
    # DSR (if multiple trials)
    if n_trials > 1:
        # Assume all trials had similar variance, estimate SR std
        sr_std = sr / np.sqrt(n) * 2  # Rough estimate
        dsr = deflated_sharpe_ratio(sr, [sr_std, n_trials], n, skew, kurt)
        benchmark_sr_deflated = deflated_sharpe_ratio(
            sr, [sr_std, n_trials], n, skew, kurt, return_benchmark=True
        )
    else:
        dsr = psr
        benchmark_sr_deflated = benchmark_sr
    
    # MinTRL
    min_trl = minimum_track_record_length(sr, benchmark_sr, skew, kurt)
    
    # Drawdown
    max_dd, tuw = drawdown_and_time_under_water(returns)
    
    # Concentration
    concentration = bets_concentration(returns)
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor (gross profit / gross loss)
    gross_profit = returns[returns > 0].sum()
    gross_loss = returns[returns < 0].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'n_trades': n,
        'total_return': total_return,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sr,
        'skewness': skew,
        'kurtosis': kurt,
        'psr': psr,
        'psr_significant': psr > 0.95,
        'dsr': dsr,
        'benchmark_sr_deflated': benchmark_sr_deflated,
        'dsr_significant': dsr > 0.95,
        'min_trl': min_trl,
        'track_record_sufficient': n >= min_trl,
        'max_drawdown': max_dd,
        'time_under_water': tuw,
        'concentration': concentration,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def print_statistics_report(stats: dict) -> None:
    """Print a formatted statistics report."""
    print("\n" + "=" * 60)
    print(" AFML BACKTEST STATISTICS REPORT (Chapter 14)")
    print("=" * 60)
    
    print("\n📊 BASIC METRICS:")
    print(f"  Trades:        {stats['n_trades']}")
    print(f"  Total Return:  {stats['total_return']:.4f}")
    print(f"  Win Rate:      {stats['win_rate']:.2%}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    print("\n📈 RISK-ADJUSTED RETURNS:")
    print(f"  Sharpe Ratio:  {stats['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:  {stats['max_drawdown']:.2%}")
    print(f"  Skewness:      {stats['skewness']:.3f}")
    print(f"  Kurtosis:      {stats['kurtosis']:.3f}")
    
    print("\n🔬 STATISTICAL SIGNIFICANCE (AFML Ch.14):")
    print(f"  Probabilistic Sharpe Ratio (PSR): {stats['psr']:.3f}")
    print(f"    → {'✅ SIGNIFICANT' if stats['psr_significant'] else '❌ NOT significant'} at 95% confidence")
    
    print(f"  Deflated Sharpe Ratio (DSR):      {stats['dsr']:.3f}")
    print(f"    → Benchmark SR (after deflation): {stats['benchmark_sr_deflated']:.3f}")
    print(f"    → {'✅ SIGNIFICANT' if stats['dsr_significant'] else '❌ NOT significant'} at 95% confidence")
    
    print(f"\n  Minimum Track Record Length:      {stats['min_trl']:.0f} observations")
    print(f"    → {'✅ SUFFICIENT' if stats['track_record_sufficient'] else '❌ INSUFFICIENT'} ({stats['n_trades']} vs {stats['min_trl']:.0f} needed)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo: Run on Walk-Forward results
    import os
    
    print("Loading Walk-Forward backtest results...")
    
    if os.path.exists(os.path.join("data", "output", "backtest_wf_results.csv")):
        df = pd.read_csv(os.path.join("data", "output", "backtest_wf_results.csv"), index_col=0, parse_dates=True)
        
        if 'pnl' in df.columns:
            returns = df['pnl']
        elif 'pnl_strategy' in df.columns:
            returns = df['pnl_strategy']
        else:
            print("Cannot find returns column.")
            exit(1)
            
        # Compute stats (assuming ~5 trials for hyperparameter search + PCA experiments)
        stats = compute_all_statistics(
            returns,
            n_trials=5,  # Approximate number of major experiments
            entries_per_year=252 * 4,  # ~4 bars/day
            benchmark_sr=0.0
        )
        
        print_statistics_report(stats)
        
        # Save to CSV
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join("data", "output", "backtest_statistics.csv"), index=False)
        print("\nSaved statistics to backtest_statistics.csv")
    else:
        print("No backtest results found. Run backtest_walk_forward.py first.")
