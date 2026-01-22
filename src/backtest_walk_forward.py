import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data():
    """Load PCA features and labeled events."""
    print("Loading data...")
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
        
    events = pd.read_csv('labeled_events.csv', index_col=0, parse_dates=True)
    df = df.join(events[['t1', 'ret', 'trgt', 'side']], rsuffix='_events')
    
    if 't1' not in df.columns and 't1_events' in df.columns:
        df['t1'] = df['t1_events']
        
    df['t1'] = pd.to_datetime(df['t1'])
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c.startswith('PC_')]
    
    # Sort by index to ensure time order
    df = df.sort_index()
    
    print(f"Loaded {len(df)} events.")
    return df, feature_cols

def get_train_indices(df, test_start_time, embargo_pct=0.01):
    """
    Get training indices respecting purging and embargo.
    Train set: events that END before test_start_time.
    Embargo: Enforce gap after any previous test set (if we were chaining, 
             but here we just strictly define train as ended before test start).
    
    Actually, for Walk-Forward:
    Train ends at 'test_start_time'.
    We must purge training labels that overlap with 'test_start_time'.
    
    Since we don't know the overlap structure perfectly without iterating, 
    the safest rule is:
    Train on all samples where t1 < test_start_time.
    """
    # Purging: Exclude any train sample whose label (t1) overlaps with test period start
    # Train samples must be fully realized before we start trading the test set.
    train_mask = df['t1'] < test_start_time
    
    # Embargo? 
    # Usually embargo removes samples immediately *after* a test set from the *subsequent* train set.
    # In Walk-Forward (Expanding), we just add data.
    # The constraint `t1 < test_start` implicitly handles look-ahead.
    # We just need to ensure we don't use data that wasn't available.
    
    return df[train_mask].index

def run_walk_forward(df, feature_cols, params, initial_train_size=400, step_size=20, top_n_features=20):
    """
    Expanding Window Walk-Forward Validation with Dynamic Feature Selection.
    """
    print(f"\nStarting Walk-Forward Validation with Dynamic Feature Selection")
    print(f"Initial Train Size: {initial_train_size} events | Step Size: {step_size} events")
    print(f"Dynamic Feature Selection: Top {top_n_features} by MDI")
    
    oos_probs = pd.Series(index=df.index, dtype=float)
    oos_probs[:] = np.nan
    
    n_samples = len(df)
    current_idx = initial_train_size
    pbar = tqdm(total=n_samples - initial_train_size)
    
    while current_idx < n_samples:
        test_end_idx = min(current_idx + step_size, n_samples)
        test_indices = df.index[current_idx : test_end_idx]
        test_start_time = test_indices[0]
        
        train_indices = df[df['t1'] < test_start_time].index
        
        X_train_full = df.loc[train_indices, feature_cols]
        y_train = df.loc[train_indices, 'label'].astype(int)
        w_train = df.loc[train_indices, 'sample_weight'] if 'sample_weight' in df.columns else None
        
        # --- Dynamic Feature Selection (MDI) ---
        temp_model = RandomForestClassifier(**params)
        temp_model.fit(X_train_full, y_train, sample_weight=w_train)
        
        importances = pd.Series(temp_model.feature_importances_, index=feature_cols)
        selected_features = importances.sort_values(ascending=False).head(top_n_features).index.tolist()
        # ---------------------------------------
        
        X_train_sel = X_train_full[selected_features]
        X_test_sel = df.loc[test_indices, selected_features]
        
        # Final Train on selected features
        model = RandomForestClassifier(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        
        preds = model.predict_proba(X_test_sel)[:, 1]
        oos_probs.loc[test_indices] = preds
        
        step_actual = test_end_idx - current_idx
        pbar.update(step_actual)
        current_idx = test_end_idx
        
    pbar.close()
    return oos_probs.dropna()

def analyze_wf_performance(df, probs):
    """Calculate Walk-Forward performance."""
    # Align
    df_res = df.loc[probs.index].copy()
    df_res['prob'] = probs
    
    # Strategy
    df_res['bet_size'] = 2 * df_res['prob'] - 1
    df_res['signal'] = np.where(df_res['prob'] > 0.5, 1, -1)
    
    # Costs
    cost = 2.0 / 10000 # 2bps
    
    # PnL
    df_res['pnl'] = df_res['bet_size'] * df_res['ret'] - (np.abs(df_res['bet_size']) * cost)
    df_res['pnl_bh'] = df_res['ret'] - cost
    
    # Cum PnL
    df_res['cum_pnl'] = df_res['pnl'].cumsum()
    df_res['cum_bh'] = df_res['pnl_bh'].cumsum()
    
    # Stats
    sharpe = df_res['pnl'].mean() / df_res['pnl'].std() * np.sqrt(252*4)
    total = df_res['pnl'].sum()
    win_rate = (df_res['pnl'] > 0).mean()
    
    print("\nWalk-Forward Results (OOS):")
    print("-" * 30)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Return: {total:.4f}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Trades:       {len(df_res)}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['cum_pnl'], label=f'WF Strategy (Sharpe {sharpe:.2f})', linewidth=2)
    plt.plot(df_res.index, df_res['cum_bh'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    plt.title('Walk-Forward Backtest (Expanding Window, Purged)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('visual_analysis', exist_ok=True)
    plt.savefig('visual_analysis/backtest_walk_forward.png')
    print("Saved plot to visual_analysis/backtest_walk_forward.png")
    
    return df_res

def main():
    # 1. Load Data
    df, feature_cols = load_data()
    if df is None: return
    
    # 2. Load Params
    try:
        # Try to load Random Forest best params
        if os.path.exists('best_hyperparameters.csv'):
            params_df = pd.read_csv('best_hyperparameters.csv')
            params = params_df.iloc[0].to_dict()
            if 'best_auc' in params: del params['best_auc']
            
            # Ensure proper types for RF
            int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            for p in int_params:
                if p in params: params[p] = int(params[p])
            
            # Add fixed AFML recommended defaults if missing
            params['class_weight'] = 'balanced_subsample'
            params['bootstrap'] = True
            params['random_state'] = 42
            params['n_jobs'] = -1
        else:
            print("best_hyperparameters.csv not found. Using robust RF defaults.")
            params = {
                'n_estimators': 1000,
                'max_depth': 5,
                'class_weight': 'balanced_subsample',
                'criterion': 'entropy',
                'random_state': 42,
                'n_jobs': -1
            }
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    # 3. Run Walk Forward
    # Use ~60% as initial train (400 events), roll for rest (~300 events)
    oos_probs = run_walk_forward(df, feature_cols, params, initial_train_size=400, step_size=20)
    
    # 4. Analyze
    df_res = analyze_wf_performance(df, oos_probs)
    
    # Save
    df_res.to_csv('backtest_wf_results.csv')

if __name__ == "__main__":
    main()
