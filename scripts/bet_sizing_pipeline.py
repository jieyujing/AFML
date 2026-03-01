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
    import joblib
    
    features_file = "outputs/dollar_bars/feature_matrix.csv"
    labels_file = "outputs/dollar_bars/tbm_labels.csv"
    model_file = "outputs/models/meta_model/meta_model.pkl"
    
    if not os.path.exists(features_file) or not os.path.exists(model_file) or not os.path.exists(labels_file):
        print("Required files for actual bet sizing are missing. Running synthetic simulation.")
        # Fallback to simple random df if no files
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            'timestamp': dates,
            'side': np.random.choice([-1, 1], size=10),
            't1': dates + pd.Timedelta(days=2),
            'prob': np.random.uniform(0.3, 0.99, size=10)
        })
    else:
        print("Loading feature matrix and actual Meta-Model...")
        df_feat = pd.read_csv(features_file, parse_dates=['timestamp'])
        df_lab = pd.read_csv(labels_file, parse_dates=['timestamp'])
        
        # tbm_labels contains the original side as 'bin' (1 or -1)
        df_side = df_lab[['timestamp', 'bin']].rename(columns={'bin': 'side'})
        
        df = pd.merge(df_feat, df_side, on='timestamp', how='left')
        df = df.set_index('timestamp')
        df = df.dropna(subset=['side', 't1'])
        
        # Prepare features for prediction exactly as training
        # Drop excluded cols
        META_COLS = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades', 'median_trade_size', 'log_return', 'bin', 't1', 'avg_uniqueness', 'return_attribution', 'timestamp']
        exclude_cols = META_COLS + ['trend_weighted_uniqueness', 'trend_confidence', 'vertical_touch_weights', 'event_idx', 'touch_idx', 'ret', 'side']
        
        # Drop noisy features 
        noisy_features = [
            'vol_parkinson', 'liq_amihud',  # Cluster 4
            'trend_variance_ratio_20',      # Cluster 6
            'vol_atr_14', 'ema_short', 'ema_long', 'ffd_log_price'  # Cluster 2
        ]
        exclude_cols.extend(noisy_features)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy().ffill().bfill()
        
        # Load model and predict
        model = joblib.load(model_file)
        
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = model.predict_proba(X)
        
        # We want the probability of class 1 (meta-model says TRUE)
        df['prob'] = probs[:, 1]
        df = df.reset_index()
        
    print("Calculating base signal sizes...")
    df['base_size'] = get_signal_size(df['prob'], df['side'])
    
    # Create required inputs for concurrent sizes
    t_events = pd.DatetimeIndex(pd.to_datetime(df['timestamp']))
    df['t1'] = pd.to_datetime(df['t1'])
    t_exits = pd.Series(df['t1'].values, index=t_events)
    
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
