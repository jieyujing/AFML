import pandas as pd
import numpy as np
import os

# Load results
results_path = os.path.join("data", "output", "backtest_wf_results.csv")
df = pd.read_csv(results_path)

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
    
    # Yearly alignment (bars usually 4 per day, 252 days)
    # But wait, we can just look at daily-like returns
    rets = bars['close'].pct_change().dropna()
    
    # Annualized Sharpe (assuming 4 bars per day * 252 days = 1008 bars/year)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(1008)
    
    print(f"B&H Annualized Sharpe: {sharpe:.4f}")
    print(f"B&H Total Return: {(bars['close'].iloc[-1]/bars['close'].iloc[0]-1):.4f}")
    
    # Strategy
    strat_rets = df['pnl']
    # Strat trades are less frequent. But we should annualize similarly.
    # trades = 780. Period: 2020-2026 (~6 years). 130 trades/year.
    strat_sharpe = (strat_rets.mean() / strat_rets.std()) * np.sqrt(130)
    print(f"Strategy Annualized Sharpe (approx): {strat_sharpe:.4f}")
