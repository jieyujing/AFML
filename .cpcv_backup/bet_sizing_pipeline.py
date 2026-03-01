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
    oof_predictions_file = "outputs/models/meta_model/oof_predictions.csv"
    
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
        print("Loading actual Out-Of-Fold Meta-Model probabilities...")
        df_feat = pd.read_csv(features_file, parse_dates=['timestamp'])
        df_lab = pd.read_csv(labels_file, parse_dates=['timestamp'])
        oof_df = pd.read_csv(oof_predictions_file, parse_dates=['timestamp'])
        
        # tbm_labels contains the original side as 'bin' (1 or -1)
        df_side = df_lab[['timestamp', 'bin']].rename(columns={'bin': 'side'})
        
        df = pd.merge(df_feat, df_side, on='timestamp', how='left')
        
        # 'df' has base features and 'side'
        # CPCV oof_predictions.csv provides multiple paths
        path_cols = [c for c in oof_df.columns if c.startswith('path_')]
        if not path_cols and 'prob' in oof_df.columns:
            oof_df['path_0'] = oof_df['prob']
            path_cols = ['path_0']
            
        print(f"Calculating concurrent active sizes for {len(path_cols)} independent paths...")
        
        all_disc_sizes = {}
        for p_col in path_cols:
            df_p = pd.merge(df.copy(), oof_df[['timestamp', p_col]], on='timestamp', how='inner')
            df_p = df_p.dropna(subset=['side', 't1', p_col]).reset_index(drop=True)
            
            df_p['base_size'] = get_signal_size(df_p[p_col], df_p['side'])
            
            t_events = pd.DatetimeIndex(pd.to_datetime(df_p['timestamp']))
            df_p['t1'] = pd.to_datetime(df_p['t1'])
            t_exits = pd.Series(df_p['t1'].values, index=t_events)
            
            active_sizes = get_concurrent_sizes(df_p['base_size'], t_events, t_exits)
            disc_sizes = discretize_size(active_sizes, step_size=0.1)
            
            all_disc_sizes[p_col] = disc_sizes
            
        result = pd.DataFrame(all_disc_sizes)
        result = result.ffill().fillna(0.0)
        
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
        
        # Plot only the first 200 rows for the first path for clarity
        plot_df = result.head(200)
        first_path = path_cols[0]
        
        plt.step(plot_df.index, plot_df[first_path], label=f'Discretized Size (step=0.1) - {first_path}', where='post', color='r')
        
        plt.title('Concurrent Bet Sizing Pipeline Output (CPCV First Path)')
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
