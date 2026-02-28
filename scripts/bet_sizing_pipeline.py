import os
import pandas as pd
import numpy as np

from afmlkit.label.bet_size import (
    get_signal_size,
    get_concurrent_sizes,
    discretize_size,
    get_size_change_signals
)

def simulate_bet_sizing():
    input_file = "outputs/dollar_bars/cusum_sampled_bars.csv"
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}. Creating synthetic data for simulation.")
        # Create synthetic data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            'timestamp': dates,
            'side': np.random.choice([-1, 1], size=10),
            't1': dates + pd.Timedelta(days=2), # Exit 2 days later
        })
    else:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
    print("Simulating Meta-Model probabilities...")
    # Add random probabilities between 0.3 and 0.99
    np.random.seed(42)
    df['prob'] = np.random.uniform(0.3, 0.99, size=len(df))
    
    # Needs side, t1, and timestamp
    if 'side' not in df.columns:
        df['side'] = np.random.choice([-1, 1], size=len(df))
        
    if 't1' not in df.columns:
        df['t1'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=48)
        
    print("Calculating base signal sizes...")
    df['base_size'] = get_signal_size(df['prob'], df['side'])
    
    # Create required inputs for concurrent sizes
    t_events = pd.DatetimeIndex(pd.to_datetime(df['timestamp']))
    # Handle missing t1
    has_t1 = df['t1'].notna()
    if not has_t1.all():
        print(f"Warning: {has_t1.sum()} events do not have t1 (exit time). Forward filling them or dropping them.")
    t_exits = pd.Series(pd.to_datetime(df['t1']).values, index=t_events)
    
    # We want to evaluate the sizes over all unique times
    print("Calculating concurrent active sizes...")
    active_sizes = get_concurrent_sizes(df['base_size'], t_events, t_exits)
    
    print("Discretizing sizes (step=0.1)...")
    disc_sizes = discretize_size(active_sizes, step_size=0.1)
    
    change_signals = get_size_change_signals(disc_sizes)
    
    result = pd.DataFrame({
        'avg_size': active_sizes,
        'discretized': disc_sizes,
        'change_signal': change_signals
    })
    
    print("\nResult Sample:")
    print(result.head(15))
    
    output_dir = "outputs/bet_sizing"
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/discretized_sizes.csv"
    result.to_csv(out_file)
    print(f"\nSaved bet sizing results to {out_file}")
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Plot only the first 100 rows for clarity if dataset is large, or all if small
        plot_df = result.head(200)
        
        plt.plot(plot_df.index, plot_df['avg_size'], label='Continuous Average Size', alpha=0.6, marker='.')
        plt.step(plot_df.index, plot_df['discretized'], label='Discretized Size (step=0.1)', where='post', color='r')
        plt.scatter(
            plot_df.index[plot_df['change_signal']], 
            plot_df['discretized'][plot_df['change_signal']], 
            color='black', zorder=5, label='Change Trigger', s=30
        )
        
        plt.title('Concurrent Bet Sizing Pipeline Output')
        plt.xlabel('Time')
        plt.ylabel('Target Position Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = f"{output_dir}/bet_sizes_plot.png"
        plt.savefig(plot_file)
        print(f"Saved plot to {plot_file}")
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    simulate_bet_sizing()
