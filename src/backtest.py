import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data():
    """Load PCA features and labeled events."""
    print("Loading data...")
    # 1. Features (PCA)
    try:
        df = pd.read_csv('features_pca.csv')
    except Exception as e:
        print(f"Error reading features_pca.csv: {e}")
        return None, None
        
    # Handle date index
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    # 2. Events (for ret and t1)
    events = pd.read_csv('labeled_events.csv', index_col=0, parse_dates=True)
    
    # Align
    df = df.join(events[['t1', 'ret', 'trgt', 'side']], rsuffix='_events')
    
    if 't1' not in df.columns and 't1_events' in df.columns:
        df['t1'] = df['t1_events']
        
    df['t1'] = pd.to_datetime(df['t1'])
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c.startswith('PC_')]
    
    print(f"Loaded {len(df)} events with {len(feature_cols)} PCA features.")
    return df, feature_cols

def run_temporal_split_backtest(df, feature_cols, params, split_pct=0.8):
    """
    Run backtest with a simple temporal split (IS/OOS).
    """
    # 1. Split Data
    split_idx = int(len(df) * split_pct)
    split_date = df.index[split_idx]
    
    print(f"\nTemporal Split: 80% Train / 20% Test")
    print(f"Split Date: {split_date}")
    
    X = df[feature_cols]
    y = df['label'].astype(int)
    weights = df['sample_weight'] if 'sample_weight' in df.columns else None
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    w_train = weights.iloc[:split_idx] if weights is not None else None
    
    # 2. Train Model (IS)
    print("Training LightGBM on In-Sample data...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # 3. Predict All
    probs = model.predict_proba(X)[:, 1]
    df['prob'] = probs
    
    # 4. Generate Signals
    # Bet Sizing: 2*P - 1
    df['bet_size'] = 2 * df['prob'] - 1
    df['signal_binary'] = np.where(df['prob'] > 0.5, 1, -1)
    
    # 5. Calculate PnL (Net of Costs)
    cost_bps = 2
    cost = cost_bps / 10000.0
    
    # Strategy PnL
    df['pnl_strategy'] = df['bet_size'] * df['ret'] - (np.abs(df['bet_size']) * cost)
    
    # Buy & Hold PnL (Long Only benchmark)
    # Assume we buy 1 unit at start of every event and close at end
    df['pnl_bh'] = df['ret'] - cost # Still pay cost for B&H turnover if we assume discrete trades
    # Or strict B&H (just holding asset)?
    # Since 'ret' is discrete event return, "B&H" here implies "Always Long Strategy"
    
    df['cum_pnl_strategy'] = df['pnl_strategy'].cumsum()
    df['cum_pnl_bh'] = df['pnl_bh'].cumsum()
    
    return df, split_date

def analyze_and_plot(df, split_date):
    """
    Analyze performance IS vs OOS and plot.
    """
    # Separate Data
    df_is = df[df.index < split_date]
    df_oos = df[df.index >= split_date]
    
    def get_stats(data, name):
        sharpe = data['pnl_strategy'].mean() / data['pnl_strategy'].std() * np.sqrt(252 * 4)
        total = data['pnl_strategy'].sum()
        return f"{name} Sharpe: {sharpe:.2f} | Total: {total:.2f}"

    print("\nPerformance Stats:")
    print(get_stats(df_is, "In-Sample "))
    print(get_stats(df_oos, "Out-of-Sample"))
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot Curves
    plt.plot(df.index, df['cum_pnl_strategy'], label='Active Strategy (Sized)', color='#1f77b4', linewidth=2)
    plt.plot(df.index, df['cum_pnl_bh'], label='Buy & Hold (Always Long)', color='gray', linestyle='--', alpha=0.7)
    
    # Add Split Line
    plt.axvline(x=split_date, color='black', linestyle='-', linewidth=1.5, label='IS / OOS Split')
    
    # Shade In-Sample Region
    plt.axvspan(df.index[0], split_date, color='gray', alpha=0.1, label='In-Sample (Train)')
    
    # Styling
    plt.title(f'Strategy Backtest: In-Sample vs Out-of-Sample (Split: {split_date.date()})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Units)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = 'visual_analysis/backtest_is_oos.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

def main():
    # 1. Load Data
    df, feature_cols = load_data()
    if df is None: return
    
    # 2. Load Params
    try:
        params_df = pd.read_csv('best_hyperparameters_lgbm_pca.csv')
        params = params_df.iloc[0].to_dict()
        if 'best_auc' in params: del params['best_auc']
        
        int_params = ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']
        for p in int_params:
            if p in params: params[p] = int(params[p])
            
        params['random_state'] = 42
        params['n_jobs'] = -1
        params['verbosity'] = -1
        
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    # 3. Run Split Backtest
    df_res, split_date = run_temporal_split_backtest(df, feature_cols, params)
    
    # 4. Analyze
    analyze_and_plot(df_res, split_date)
    
    # Save CSV
    df_res.to_csv('backtest_split_results.csv')
    print("Saved results to backtest_split_results.csv")

if __name__ == "__main__":
    main()