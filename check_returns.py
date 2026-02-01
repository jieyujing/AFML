import pandas as pd
import os

# Load results
results_path = os.path.join("data", "output", "backtest_wf_results.csv")
df = pd.read_csv(results_path)

print(f"Strategy Total Return: {df['pnl'].sum():.4f}")
print(f"Baseline (Primary) Total Return: {df['pnl_bh'].sum():.4f}")

# Load raw bars for full B&H
bars_path = os.path.join("data", "output", "dynamic_dollar_bars.csv")
if os.path.exists(bars_path):
    bars = pd.read_csv(bars_path)
    # Match the backtest period
    df['date'] = pd.to_datetime(df['date'])
    bars['datetime'] = pd.to_datetime(bars['datetime'])
    start_dt = df['date'].min()
    end_dt = df['date'].max()
    bars = bars[(bars['datetime'] >= start_dt) & (bars['datetime'] <= end_dt)]
    
    total_bh_ret = (bars['close'].iloc[-1] / bars['close'].iloc[0]) - 1
    print(f"Underlying Asset (True Buy & Hold) Total Return: {total_bh_ret:.4f}")

# Check average bet size
print(f"Average Bet Size Magnitude: {df['size_magnitude'].mean():.4f}")
print(f"Avg Probability: {df['prob'].mean():.4f}")
