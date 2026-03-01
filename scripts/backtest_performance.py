import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_psr(returns, benchmark_sr=0):
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    Formula from AFML Chapter 14.
    """
    if len(returns) < 2:
        return 0.0
    
    n = len(returns)
    sr = returns.mean() / returns.std(ddof=1)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 # Excess kurtosis to total kurtosis
    
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr)
    return psr

def calculate_dsr(returns, n_trials, sr_std):
    """
    Calculate Deflated Sharpe Ratio (DSR).
    Formula from AFML Chapter 14.
    """
    # Expected maximum Sharpe Ratio among N trials
    expected_max_sr = sr_std * ( (1 - 0.5772156649) * norm.ppf(1 - 1/n_trials) + 0.5772156649 * norm.ppf(1 - 1/n_trials * np.exp(-1)) )
    
    # Use expected_max_sr as the benchmark for PSR
    dsr = calculate_psr(returns, benchmark_sr=expected_max_sr)
    return dsr, expected_max_sr

def run_backtest():
    bars_file = "outputs/dollar_bars/cusum_sampled_bars.csv"
    sizes_file = "outputs/bet_sizing/discretized_sizes.csv"
    output_dir = "outputs/backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(bars_file) or not os.path.exists(sizes_file):
        print("Required input files missing.")
        return

    print("Loading data...")
    bars = pd.read_csv(bars_file, index_col='timestamp', parse_dates=True)
    sizes = pd.read_csv(sizes_file, index_col=0, parse_dates=True)
    
    # Align data
    df = bars[['close']].copy()
    df['pos'] = sizes['discretized']
    df['pos'] = df['pos'].ffill().fillna(0) # In case grid differs
    
    print("Calculating returns...")
    # Return at t is (Price_t / Price_{t-1}) - 1
    df['ret'] = df['close'].pct_change()
    
    # Strategy return at t is Pos_{t-1} * Return_t
    # shift(1) ensures no look-ahead bias
    df['strat_ret'] = df['pos'].shift(1) * df['ret']
    df['strat_ret'] = df['strat_ret'].fillna(0)
    
    # Cumulative returns
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    
    # Performance metrics
    total_ret = df['cum_ret'].iloc[-1] - 1
    
    # Annualization factor (assuming Dollar Bars frequency, let's estimate daily count)
    # Average time between bars
    avg_delta = df.index.to_series().diff().mean()
    annualization_factor = (pd.Timedelta(days=365) / avg_delta)
    
    sr = df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(annualization_factor)
    
    # Drawdown
    rolling_max = df['cum_ret'].cummax()
    drawdown = (df['cum_ret'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # PSR & DSR
    # We estimate n_trials based on typical development iterations
    n_trials = 50 
    # Use the cross-validation standard deviation of SR if available, 
    # but here we approximate based on the current sample.
    sr_std = 0.5 # A conservative estimate of the standard deviation of SR across trials
    
    psr = calculate_psr(df['strat_ret'])
    dsr, exp_max_sr = calculate_dsr(df['strat_ret'], n_trials, sr_std / np.sqrt(annualization_factor))
    
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"Total Return:       {total_ret:.2%}")
    print(f"Annualized Sharpe: {sr:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")
    print(f"PSR (SR > 0):      {psr:.4f}")
    print(f"Number of Trials:  {n_trials}")
    print(f"DSR Probability:    {dsr:.4f}")
    print(f"Verdict:           {'ACCEPT' if dsr > 0.95 else 'REJECT'}")
    print("="*30)
    
    # Save results
    df.to_csv(f"{output_dir}/backtest_results.csv")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['cum_ret'], label='Strategy Equity')
    plt.title('Strategy Performance')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.fill_between(df.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['pos'], label='Position Size', color='green', alpha=0.7)
    plt.ylabel('Position Size')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_plot.png")
    print(f"\nPlot saved to {output_dir}/performance_plot.png")

if __name__ == "__main__":
    run_backtest()
