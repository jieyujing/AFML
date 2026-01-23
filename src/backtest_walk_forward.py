import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.backtest_statistics as bs

def load_data(input_path: str = None, suffix: str = ""):
    """Load features and labeled events."""
    print("Loading data...")
    if input_path:
        feat_path = input_path
        print(f"Loading features from: {feat_path}")
    else:
        # Defaults
        # Check for V2 first, then PCA, then legacy
        v2_path = os.path.join("data", "output", f"features_v2_labeled{suffix}.csv")
        pca_path = os.path.join("data", "output", f"features_pca{suffix}.csv")
        
        if os.path.exists(v2_path):
            feat_path = v2_path
            print(f"Loading V2 features from: {feat_path}")
        elif os.path.exists(pca_path):
            feat_path = pca_path
            print(f"Loading PCA features from: {feat_path}")
        else:
            feat_path = os.path.join("data", "output", f"features_labeled{suffix}.csv")
            print(f"Loading Legacy features from: {feat_path}")

    if not os.path.exists(feat_path):
        print(f"Error reading feature file: {feat_path}")
        return None, None

    df = pd.read_csv(feat_path)
    
    # Handle index
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Load labeled events for t1 and avg_uniqueness
    events_path = os.path.join("data", "output", f"labeled_events{suffix}.csv")
    if not os.path.exists(events_path):
        events_path = os.path.join("data", "output", "labeled_events.csv")
    
    if os.path.exists(events_path): 
        print(f"Loading labeled events from: {events_path}")
        events = pd.read_csv(events_path, index_col=0, parse_dates=True)
        # Merge required columns into df if missing
        for col in ['t1', 'ret', 'trgt', 'side', 'avg_uniqueness']:
            if col not in df.columns and col in events.columns:
                df[col] = events[col]
    
    if 't1' not in df.columns and 't1_events' in df.columns:
        df['t1'] = df['t1_events']
        
    if 't1' in df.columns:
        df['t1'] = pd.to_datetime(df['t1'])
        
    df = df.dropna(subset=['label'])
    
    # Identify feature columns (PCA or All Numerical)
    feature_cols = [c for c in df.columns if c.startswith('PC_')]
    if not feature_cols:
        exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events', 'holding_period', 'return']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Sort by index to ensure time order (Crucial for Walk Forward)
    df = df.sort_index()
    
    print(f"Loaded {len(df)} events.")
    return df, feature_cols

def get_train_indices(df, test_start_time, embargo_pct=0.01):
    """
    Get training indices respecting purging and embargo.
    Train set: events that END before test_start_time.
    We must purge training labels that overlap with 'test_start_time'.
    """
    # Purging: train samples must be fully realized before we start trading the test set.
    train_mask = df['t1'] < test_start_time
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
        
        train_indices = get_train_indices(df, test_start_time)
        
        if len(train_indices) < 50:
            print(f"Warning: Insufficient train samples ({len(train_indices)}). Skipping fold.")
            current_idx = test_end_idx
            continue
            
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

def analyze_wf_performance(df, probs, output_dir="data/output", suffix=""):
    """Calculate Walk-Forward performance using Meta-Labeling logic."""
    # Align
    df_res = df.loc[probs.index].copy()
    df_res['prob'] = probs
    
    # Ensure 'side' exists (from labeled_events)
    if 'side' not in df_res.columns:
        print("Warning: 'side' column missing. Assuming Long-only (side=1).")
        df_res['side'] = 1
    
    # --- Meta-Labeling Bet Sizing ---
    df_res['size_magnitude'] = (2 * df_res['prob'] - 1).clip(lower=0.0)
    
    # Final Bet Size = Magnitude * Direction
    df_res['bet_size'] = df_res['size_magnitude'] * df_res['side']
    
    # Signal (for statistics, simple Binary)
    df_res['signal'] = np.where(df_res['prob'] > 0.5, df_res['side'], 0)
    
    # Costs
    cost = 2.0 / 10000 # 2bps
    
    # PnL = |bet_size| * Strategy_Ret - Cost
    df_res['pnl'] = df_res['size_magnitude'] * df_res['ret'] - (np.abs(df_res['size_magnitude']) * cost)
    
    # Buy & Hold (Benchmark) - Taking every Primary Signal
    df_res['pnl_bh'] = df_res['ret'] - cost
    
    # Cum PnL
    df_res['cum_pnl'] = df_res['pnl'].cumsum()
    df_res['cum_bh'] = df_res['pnl_bh'].cumsum()
    
    # --- AFML STATISTICS Integration ---
    stats = bs.compute_all_statistics(
        df_res['pnl'], 
        n_trials=5, 
        entries_per_year=252*4, 
        benchmark_sr=0.0
    )
    
    bs.print_statistics_report(stats)
    
    # Save Stats
    os.makedirs(output_dir, exist_ok=True)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, f"backtest_statistics{suffix}.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # --- Plot Underlying Buy & Hold ---
    bars_path = os.path.join("data", "output", "dynamic_dollar_bars.csv")
    if os.path.exists(bars_path) and not df_res.empty:
        try:
            bars = pd.read_csv(bars_path)
            bars['datetime'] = pd.to_datetime(bars['datetime'])
            bars.set_index('datetime', inplace=True)
            
            # Align time range
            start_dt, end_dt = df_res.index.min(), df_res.index.max()
            mask = (bars.index >= start_dt) & (bars.index <= end_dt)
            bars = bars[mask].copy()
            
            if not bars.empty:
                # Calculate Cumulative Returns (Simple Summation to match PnL logic)
                bars['ret'] = bars['close'].pct_change().fillna(0)
                bars['cum_bnh'] = bars['ret'].cumsum()
                
                plt.plot(bars.index, bars['cum_bnh'], label='Underlying Asset (Buy & Hold)', color='gray', alpha=0.6, linestyle=':', linewidth=1.5)
                print(f"Added Underlying Buy & Hold comparison ({len(bars)} bars)")
        except Exception as e:
            print(f"Warning: Failed to plot underlying asset: {e}")

    sr_str = f"{stats['sharpe_ratio']:.2f}"
    psr_str = f"{stats['psr']:.2f}"
    
    plt.plot(df_res.index, df_res['cum_pnl'], label=f'WF Strategy (SR {sr_str} | PSR {psr_str})', color='#1f77b4', linewidth=2)
    plt.plot(df_res.index, df_res['cum_bh'], label='Primary Model (Baseline)', color='#ff7f0e', linestyle='--', alpha=0.8)
    plt.title(f'Walk-Forward Backtest (Expanding Window) {suffix}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    vis_dir = 'visual_analysis'
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'backtest_walk_forward{suffix}.png'))
    print(f"Saved plot to {vis_dir}/backtest_walk_forward{suffix}.png")
    
    return df_res

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Input features file")
    parser.add_argument("--output", type=str, default=None, help="Output results file")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for picking default files (e.g. '_meta')")
    parser.add_argument("--meta", action="store_true", help="Enable Meta-Labeling logic")
    args = parser.parse_args()

    # 1. Load Data
    df, feature_cols = load_data(input_path=args.input, suffix=args.suffix)
    if df is None: return
    
    # META LABELING TRANSFORMATION
    if args.meta:
        print("   Applying Meta-Labeling transformation for WF training: {-1, 0} -> 0, {1} -> 1")
        df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

    # 2. Load Params
    try:
        # Try to load Random Forest best params
        if os.path.exists(os.path.join("data", "output", "best_hyperparameters.csv")):
            params_df = pd.read_csv(os.path.join("data", "output", "best_hyperparameters.csv"))
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
    # Use ~60% as initial train, or fixed number
    # For IF9999 (1000 samples), 400 is reasonable (~40%)
    initial_train_size = int(len(df) * 0.4) 
    step_size = int(len(df) * 0.05) # 5% steps
    print(f"Walk Forward Config: Initial={initial_train_size}, Step={step_size}")
    
    oos_probs = run_walk_forward(df, feature_cols, params, initial_train_size=initial_train_size, step_size=step_size)
    
    # 4. Analyze
    output_dir = "data/output"
    df_res = analyze_wf_performance(df, oos_probs, output_dir=output_dir, suffix=args.suffix)
    
    # Save
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(output_dir, f"backtest_wf_results{args.suffix}.csv")
        
    df_res.to_csv(out_path)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()