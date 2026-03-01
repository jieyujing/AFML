import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def adjusted_sharpe(r, lag=5):
    """
    Compute Lo (2002) autocorrelation-adjusted Sharpe Ratio.
    """
    if len(r) < 2 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    acf = [r.autocorr(i) for i in range(1, lag+1)]
    acf = [0 if np.isnan(a) else a for a in acf]
    adj_sum = sum((1 - i/(lag+1)) * acf[i-1] for i in range(1, lag+1))
    adj = np.sqrt(max(1 + 2 * adj_sum, 0.001))
    return sr / adj

def calculate_psr(returns, benchmark_sr=0.0, lag=5):
    """
    Calculate Probabilistic Sharpe Ratio (PSR) with Lo (2002) adjustment.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0, 0.0
    
    n = len(returns)
    sr = adjusted_sharpe(returns, lag=lag)
    
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 # Excess kurtosis to total kurtosis
    
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    if sigma_sr == 0:
        return 0.0, sr
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr)
    return psr, sr

def calculate_dsr(returns, n_trials, sr_std, lag=5):
    """
    Calculate Deflated Sharpe Ratio (DSR) using CPCV/Purged CV standard deviation.
    """
    if n_trials < 2 or len(returns) < 2 or sr_std == 0:
        return 0.0, 0.0
        
    expected_max_sr = sr_std * ( (1 - 0.5772156649) * norm.ppf(1 - 1/n_trials) + 0.5772156649 * norm.ppf(1 - 1/n_trials * np.exp(-1)) )
    
    dsr, _ = calculate_psr(returns, benchmark_sr=expected_max_sr, lag=lag)
    return dsr, expected_max_sr

def run_backtest():
    # Use the continuous dollar bars to evaluate path-dependent returns
    bars_file = "outputs/dollar_bars/dollar_bars_freq20.csv"
    sizes_file = "outputs/bet_sizing/discretized_sizes.csv"
    output_dir = "outputs/backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(bars_file) or not os.path.exists(sizes_file):
        print("Required input files missing.")
        return

    print(f"Loading data from {bars_file}...")
    bars = pd.read_csv(bars_file, index_col='timestamp', parse_dates=True)
    sizes = pd.read_csv(sizes_file, index_col=0, parse_dates=True)
    
    # Align data on the dense continuous index
    df = bars[['close']].copy()
    
    sizes = sizes[~sizes.index.duplicated(keep='last')]
    df = df.join(sizes, how='left')
    
    path_cols = [c for c in sizes.columns if c.startswith('path_')]
    if not path_cols and 'discretized' in sizes.columns:
        path_cols = ['discretized']
    
    print(f"Calculating path-dependent returns for {len(path_cols)} CPCV paths with transaction costs...")
    df['ret'] = df['close'].pct_change()
    
    txn_cost_rate = 0.0006 
    
    path_sharpes = []
    path_returns = []
    
    # Evaluate each CPCV path independently
    for p_col in path_cols:
        # Fix Look-ahead bias: Position established at 't-1' applies to 't' return
        pos = df[p_col].shift(1).ffill().fillna(0.0)
        p_ret = pos * df['ret']
        
        # Calculate Transaction Costs (TC) - REAL Turnover is |Δpos| / 2
        turnover = pos.diff().abs() / 2.0
        tc = turnover * txn_cost_rate
        
        # Net Returns for this path
        net_ret = p_ret.fillna(0) - tc.fillna(0)
        path_returns.append(net_ret)
        
        path_sr = adjusted_sharpe(net_ret)
        path_sharpes.append(path_sr)
        
    print("Loading CPCV logic for DSR...")
    sr_std = np.std(path_sharpes, ddof=1) if len(path_sharpes) > 1 else (np.std(path_sharpes) if len(path_sharpes) == 1 else 0.5)
    print(f"   Expected DSR StdDev based on {len(path_sharpes)} CPCV Paths: {sr_std:.4f}")
    
    # The ensemble strategy return is the average of the evaluated out-of-sample paths
    if path_returns:
        df['strat_ret'] = pd.concat(path_returns, axis=1).mean(axis=1)
    else:
        df['strat_ret'] = 0.0
        
    # Cumulative returns
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    
    # Ensemble position size (purely for visualization)
    if path_cols:
        df['pos'] = df[path_cols].mean(axis=1).shift(1).ffill().fillna(0.0)
    else:
        df['pos'] = 0.0
    
    # Performance metrics
    total_ret = df['cum_ret'].iloc[-1] - 1
    
    # DO NOT annualize the Sharpe Ratio. Information time does not match clock time.
    lo_sr = adjusted_sharpe(df['strat_ret'])
    
    # Drawdown
    rolling_max = df['cum_ret'].cummax()
    drawdown = (df['cum_ret'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # PSR & DSR metrics
    n_trials = 50 
    psr, _ = calculate_psr(df['strat_ret'])
    dsr, exp_max_sr = calculate_dsr(df['strat_ret'], n_trials, sr_std)
    
    print("\n" + "="*30)
    print("BACKTEST RESULTS (Institutional Grade)")
    print("="*30)
    print(f"Total Return:       {total_ret:.2%}")
    print(f"Lo-Adjusted Sharpe: {lo_sr:.4f}")
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
    plt.plot(df.index, df['cum_ret'], label='Strategy Equity (CPCV Ensemble)')
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
    plt.plot(df.index, df['pos'], label='Average Position Size (Across Paths)', color='green', alpha=0.7)
    plt.ylabel('Position Size')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_plot.png")
    print(f"\nPlot saved to {output_dir}/performance_plot.png")

if __name__ == "__main__":
    run_backtest()
