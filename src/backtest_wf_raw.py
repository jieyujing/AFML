import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data():
    """Load Raw Feature 2.0 data."""
    print("Loading features_v2_labeled.csv...")
    try:
        df = pd.read_csv('features_v2_labeled.csv', index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return None, None

    # Check for t1 and other metadata
    required_meta = ['t1', 'ret', 'side', 'trgt']
    missing_meta = [c for c in required_meta if c not in df.columns]
    
    if missing_meta:
        print(f"   Missing metadata {missing_meta}. Joining with labeled_events.csv...")
        try:
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            # Join only missing columns
            cols_to_join = [c for c in required_meta if c in events.columns and c not in df.columns]
            df = df.join(events[cols_to_join], rsuffix='_events')
            
            # Handle t1 specifically if it resulted in t1_events duplication logic (though join usually handles distinct names)
            if 't1' not in df.columns and 't1_events' in df.columns:
                 df['t1'] = df['t1_events']
                 
        except Exception as e:
            print(f"   Could not load metadata from events: {e}")
            return None, None

    df['t1'] = pd.to_datetime(df['t1'])
    df = df.dropna()
    
    # Identify feature columns (exclude labels, weights, metadata)
    exclude_cols = ['label', 'ret', 'return', 'side', 't1', 'trgt', 'pt_sl', 'sample_weight', 'avg_uniqueness', 'bin', 'w_raw', 'holding_period']
    # Also exclude any 'Unnamed' cols
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    # Select only numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Loaded {len(df)} events.")
    print(f"Total Features available: {len(feature_cols)}")
    return df.sort_index(), feature_cols

def run_walk_forward_dynamic(df, feature_cols, params, initial_train_size=400, step_size=20, top_n=20):
    """
    Rolling Walk-Forward with Dynamic Feature Selection on RAW features.
    Saves selected features map.
    """
    print(f"\nStarting Dynamic Walk-Forward (Raw Features)...")
    print(f"Initial Train: {initial_train_size}, Step: {step_size}, Top N: {top_n}")
    
    oos_probs = pd.Series(index=df.index, dtype=float)
    oos_probs[:] = np.nan
    
    feature_selection_log = [] # List of dicts: {'date': test_start, 'features': [f1, f2...]}
    
    n_samples = len(df)
    current_idx = initial_train_size
    pbar = tqdm(total=n_samples - initial_train_size)
    
    while current_idx < n_samples:
        test_end_idx = min(current_idx + step_size, n_samples)
        test_indices = df.index[current_idx : test_end_idx]
        test_start_time = test_indices[0]
        
        # Purged Train Set
        train_indices = df[df['t1'] < test_start_time].index
        
        X_train_full = df.loc[train_indices, feature_cols]
        y_train = df.loc[train_indices, 'label'].astype(int)
        w_train = df.loc[train_indices, 'sample_weight'] if 'sample_weight' in df.columns else None
        
        # --- Feature Selection ---
        # Using a restricted RF for selection to speed up
        fs_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'random_state': 42,
            'n_jobs': -1,
        }
        selector = RandomForestClassifier(**fs_params)
        selector.fit(X_train_full, y_train, sample_weight=w_train)
        
        imps = pd.Series(selector.feature_importances_, index=feature_cols)
        selected_feats = imps.sort_values(ascending=False).head(top_n).index.tolist()
        
        # Log selected features
        feature_selection_log.append({
            'date': test_start_time,
            'features': selected_feats
        })
        # -------------------------
        
        # Final Train & Predict
        X_train_sel = X_train_full[selected_feats]
        X_test_sel = df.loc[test_indices, selected_feats]
        
        model = RandomForestClassifier(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        
        preds = model.predict_proba(X_test_sel)[:, 1]
        oos_probs.loc[test_indices] = preds
        
        current_idx = test_end_idx
        pbar.update(len(test_indices))
        
    pbar.close()
    return oos_probs.dropna(), feature_selection_log

def analyze_feature_stability(log, all_feature_cols):
    """
    Visualize feature selection stability.
    """
    dates = [x['date'] for x in log]
    
    # Create a boolean DataFrame: Rows=Dates, Cols=All Features
    selection_matrix = pd.DataFrame(0, index=dates, columns=all_feature_cols)
    
    for entry in log:
        dt = entry['date']
        feats = entry['features']
        selection_matrix.loc[dt, feats] = 1
        
    # Filter to show only features that were selected at least once
    # Or better: selected at least 10% of the time, to reduce noise
    freq = selection_matrix.mean()
    active_features = freq[freq > 0.05].index.tolist() # >5% frequency
    
    # Sort features by frequency
    sorted_features = freq[active_features].sort_values(ascending=False).index.tolist()
    
    plot_matrix = selection_matrix[sorted_features]
    
    plt.figure(figsize=(15, 10))
    # Transpose for Time on X-axis
    sns.heatmap(plot_matrix.T, cmap="Blues", cbar=False, xticklabels=10)
    plt.title('Dynamic Feature Selection Stability (Top 20 per Window)')
    plt.xlabel('Rebalancing Date')
    plt.ylabel('Features (Sorted by Frequency)')
    plt.tight_layout()
    plt.savefig('visual_analysis/feature_stability_heatmap.png')
    print("Saved feature stability heatmap.")
    
    return selection_matrix, freq

def analyze_performance(df, probs):
    # Align
    df_res = df.loc[probs.index].copy()
    df_res['prob'] = probs
    
    # Simple Strategy
    df_res['bet'] = 2 * df_res['prob'] - 1
    
    # Cost 2bps
    cost = 0.0002
    df_res['pnl'] = df_res['bet'] * df_res['ret'] - np.abs(df_res['bet']) * cost
    df_res['cum_pnl'] = df_res['pnl'].cumsum()
    
    sharpe = df_res['pnl'].mean() / df_res['pnl'].std() * np.sqrt(252*4)
    print(f"Sharpe: {sharpe:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['cum_pnl'])
    plt.title(f'Walk-Forward (Raw Features) Sharpe: {sharpe:.2f}')
    plt.savefig('visual_analysis/backtest_wf_raw.png')
    return df_res

def main():
    # 1. Load
    df, feature_cols = load_data()
    if df is None: return
    
    # 2. Params
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
            print("Using default params")
            params = {'n_estimators': 500, 'max_depth': 5, 'criterion': 'entropy', 'random_state': 42, 'n_jobs': -1}
    except:
        print("Using default params")
        params = {'n_estimators': 500, 'max_depth': 5, 'criterion': 'entropy', 'random_state': 42, 'n_jobs': -1}

    # 3. Run
    probs, log = run_walk_forward_dynamic(df, feature_cols, params)
    
    # 4. Analyze Performance
    df_res = analyze_performance(df, probs)
    df_res.to_csv('backtest_wf_raw_results.csv')
    
    # 5. Analyze Features
    sel_matrix, freq = analyze_feature_stability(log, feature_cols)
    sel_matrix.to_csv('feature_selection_log.csv')
    freq.to_csv('feature_selection_frequency.csv')

if __name__ == "__main__":
    main()
