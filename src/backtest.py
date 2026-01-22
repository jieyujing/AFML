import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cv_setup import PurgedKFold

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
    # Join events to df to get ret, t1, trgt
    df = df.join(events[['t1', 'ret', 'trgt', 'side']], rsuffix='_events')
    
    # Fill t1 if missing (from join logic in previous scripts)
    if 't1' not in df.columns and 't1_events' in df.columns:
        df['t1'] = df['t1_events']
        
    df['t1'] = pd.to_datetime(df['t1'])
    
    # Drop NaNs
    df = df.dropna()
    
    # Feature columns (PCA)
    feature_cols = [c for c in df.columns if c.startswith('PC_')]
    
    print(f"Loaded {len(df)} events with {len(feature_cols)} PCA features.")
    return df, feature_cols

def get_oos_predictions(df, feature_cols, params):
    """
    Generate Out-Of-Sample predictions using Purged K-Fold CV.
    """
    X = df[feature_cols]
    y = df['label'].astype(int)
    t1 = df['t1']
    sample_weights = df['sample_weight'] if 'sample_weight' in df.columns else None
    
    # Setup CV
    n_splits = 5
    embargo_pct = 0.01
    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, embargo=embargo_pct)
    
    # Store predictions
    # Initialize with 0.5 (neutral)
    probs = pd.Series(0.5, index=df.index, name='prob_1')
    
    print(f"Generating OOS predictions with {n_splits}-Fold Purged CV...")
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx] if sample_weights is not None else None
        
        # Train Model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)
        
        # Store (assuming binary classification, col 1 is positive class)
        probs.iloc[test_idx] = y_pred_proba[:, 1]
        
    return probs

def run_backtest(df, probs):
    """
    Simulate trading strategy based on predictions.
    """
    print("\nRunning Backtest...")
    
    # 1. Signal Generation
    # Strategy: Long if P > 0.5, Short if P < 0.5
    # Bet Sizing: 2*P - 1 (Linear scaling from -1 to 1)
    
    df['prob'] = probs
    df['signal_binary'] = np.where(df['prob'] > 0.5, 1, -1)
    df['bet_size'] = 2 * df['prob'] - 1  # Range [-1, 1]
    
    # 2. PnL Calculation
    # Trade Return = Bet Size * Event Return
    # We assume we can short (Bet Size < 0)
    
    # Note: 'ret' in labeled_events is return over holding period.
    # If side was used in labeling, we must align.
    # label=1 means return > target?
    # In src/labeling.py, 1 is profit, -1 is loss.
    # So if label=1, ret should be positive?
    # Let's check correlation between label and ret.
    
    corr_label_ret = df[['label', 'ret']].corr().iloc[0,1]
    print(f"Correlation (Label vs Ret): {corr_label_ret:.4f}")
    
    if corr_label_ret < 0:
        print("WARNING: Label and Return are negatively correlated. Label -1 might mean 'down' but ret is raw return.")
    
    # Raw Strategy (Binary)
    df['pnl_binary'] = df['signal_binary'] * df['ret']
    
    # Sized Strategy
    df['pnl_sized'] = df['bet_size'] * df['ret']
    
    # 3. Transaction Costs
    # Simple assumption: Cost per trade (entry + exit)
    # Since these are discrete events (enter t0, exit t1), each row is a trade.
    cost_bps = 2  # 2 bps per trade (round trip)
    cost = cost_bps / 10000.0
    
    df['pnl_binary_net'] = df['pnl_binary'] - cost
    df['pnl_sized_net'] = df['pnl_sized'] - (np.abs(df['bet_size']) * cost)
    
    return df

def analyze_performance(df):
    """Calculate and plot performance metrics."""
    print("\nPerformance Metrics (Net of Costs):")
    
    # Metrics for Binary
    win_rate = (df['pnl_binary_net'] > 0).mean()
    avg_ret = df['pnl_binary_net'].mean()
    sharpe = df['pnl_binary_net'].mean() / df['pnl_binary_net'].std() * np.sqrt(252 * 4) # Approx 4 trades/day
    total_ret = df['pnl_binary_net'].sum()
    
    print("-" * 40)
    print("Strategy: Binary (Long/Short)")
    print(f"Win Rate:       {win_rate:.2%}")
    print(f"Avg Trade Ret:  {avg_ret:.4%}")
    print(f"Total Return:   {total_ret:.4f} (Units)")
    print(f"Approx Sharpe:  {sharpe:.2f}")
    
    # Metrics for Sized
    win_rate_sized = (df['pnl_sized_net'] > 0).mean()
    avg_ret_sized = df['pnl_sized_net'].mean()
    sharpe_sized = df['pnl_sized_net'].mean() / df['pnl_sized_net'].std() * np.sqrt(252 * 4)
    total_ret_sized = df['pnl_sized_net'].sum()
    
    print("-" * 40)
    print("Strategy: Kelly-like Bet Sizing")
    print(f"Win Rate:       {win_rate_sized:.2%}")
    print(f"Avg Trade Ret:  {avg_ret_sized:.4%}")
    print(f"Total Return:   {total_ret_sized:.4f} (Units)")
    print(f"Approx Sharpe:  {sharpe_sized:.2f}")
    
    # Plot Cumulative PnL
    df['cum_pnl_binary'] = df['pnl_binary_net'].cumsum()
    df['cum_pnl_sized'] = df['pnl_sized_net'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cum_pnl_binary'], label='Binary Strategy')
    plt.plot(df.index, df['cum_pnl_sized'], label='Sized Strategy', alpha=0.8)
    plt.title('Backtest Cumulative PnL (Net of 2bps Cost)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('visual_analysis', exist_ok=True)
    plt.savefig('visual_analysis/backtest_performance.png')
    print("Saved plot to visual_analysis/backtest_performance.png")
    
    return df

def main():
    # 1. Load Data
    df, feature_cols = load_data()
    if df is None: return
    
    # 2. Load Params
    try:
        params_df = pd.read_csv('best_hyperparameters_lgbm_pca.csv')
        params = params_df.iloc[0].to_dict()
        # Clean params (remove 'best_auc' if exists)
        if 'best_auc' in params: del params['best_auc']
        
        # Ensure correct types for LGBM
        int_params = ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']
        for p in int_params:
            if p in params:
                params[p] = int(params[p])
            
        params['random_state'] = 42
        params['n_jobs'] = -1
        params['verbosity'] = -1
        
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    # 3. Get OOS Predictions
    probs = get_oos_predictions(df, feature_cols, params)
    
    # 4. Run Backtest
    df_res = run_backtest(df, probs)
    
    # 5. Analyze
    analyze_performance(df_res)
    
    # Save predictions
    df_res[['label', 'prob', 'signal_binary', 'bet_size', 'ret', 'pnl_sized_net']].to_csv('backtest_results.csv')
    print("Saved detailed results to backtest_results.csv")

if __name__ == "__main__":
    main()
