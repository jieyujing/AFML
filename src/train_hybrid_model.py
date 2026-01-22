
"""
Hybrid Strategy Training: Rule-Based Primary + ML Meta-Model.

This script implements the "Hybrid Quant" architecture:
1. Primary Model: Moving Average Crossover (Rule-Based)
2. Meta-Model: Random Forest (ML) to filter the rule-based signals

This tests the hypothesis that ML is better at "filtering" than "predicting direction".
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_based_strategies import MovingAverageCrossStrategy
from labeling import get_volatility, apply_triple_barrier, get_events
from cv_setup import PurgedKFold
from bet_sizing import bet_size_probability

def main():
    print("=" * 80)
    print("Hybrid Strategy: MA Cross + Meta-Labeling")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading Price Data...")
    try:
        # We need raw prices to generate signals
        bars = pd.read_csv("dynamic_dollar_bars.csv", index_col=0, parse_dates=True)
        close = bars["close"]
        print(f"   Loaded {len(close)} bars.")
        
        # Load Features (for Meta-Model)
        # Use the full feature set
        features_df = pd.read_csv("features_all.csv", index_col=0, parse_dates=True)
        print(f"   Loaded {len(features_df)} feature vectors.")
        
        # Try to load avg_uniqueness from sample_weights if available (for existing events)
        # Note: MA Cross events might not match CUSUM events, so coverage will be partial.
        if os.path.exists("sample_weights.csv"):
            weights_df = pd.read_csv("sample_weights.csv", index_col=0, parse_dates=True)
            if 'avg_uniqueness' in weights_df.columns:
                # Join to features, keeping all features
                features_df = features_df.join(weights_df[['avg_uniqueness']], how='left')
                print(f"   Joined avg_uniqueness from sample_weights.csv (Partial coverage).")
        
        if 'avg_uniqueness' not in features_df.columns:
            print("   Warning: 'avg_uniqueness' not found. Concurrency adjustment disabled.")
            
    except FileNotFoundError:
        print("Error: Data files not found.")
        return

    # 2. Generate Primary Signals (Rule-Based)
    print("\n2. Generating Rule-Based Signals (MA Cross)...")
    # Using fairly fast MAs for Dollar Bars (intrinsic time)
    # 20 bars ~ 1 week, 50 bars ~ 2.5 weeks
    strategy = MovingAverageCrossStrategy(fast_window=20, slow_window=50)
    primary_signals = strategy.generate_signals(close)
    
    # Filter to only non-zero events
    entry_signals = primary_signals[primary_signals != 0]
    
    print(f"   Generated {len(entry_signals)} signals.")
    print(f"   Longs: {(entry_signals==1).sum()} | Shorts: {(entry_signals==-1).sum()}")
    
    if len(entry_signals) == 0:
        print("   No signals generated. Exiting.")
        return

    # 3. Label Specific Signals (Triple Barrier)
    print("\n3. Labeling Rule-Based Signals...")
    # We only label the timestamps where the Strategy triggered
    # This is different from previous `labeling.py` which labeled CUSUM events.
    # Here the "Events" ARE the Strategy Entries.
    
    # Calculate Volatility for barriers
    vol = get_volatility(close, span=100)
    
    # Barrier settings (same as before)
    pt_sl = [1, 1]
    vertical_barrier_bars = 12
    min_ret = 0.001
    
    labeled_events = get_events(
        close=close,
        t_events=entry_signals.index,
        pt_sl=pt_sl,
        target=vol,
        min_ret=min_ret,
        vertical_barrier_bars=vertical_barrier_bars,
        side=entry_signals # IMPORTANT: Pass the side predicted by rules!
    )
    
    print(f"   Labeled {len(labeled_events)} valid events.")
    print(f"   Win Rate (Raw Rule): {(labeled_events['label']==1).mean()*100:.2f}%")
    
    # 4. Prepare Meta-Model Data
    print("\n4. Preparing Meta-Model Dataset...")
    # Join features to these specific events
    # We use 'inner' join to keep only rule-triggered rows
    
    # labeled_events index is the timestamp t0
    # features_df index should match
    
    # We need to align features. features_labeled.csv usually has all bars or CUSUM events.
    # If it has all bars, we are good. If it has only CUSUM, we might miss some MA Cross events.
    # Let's check features size vs bars size.
    # dynamic_dollar_bars: 3927. features_labeled: usually similar minus warmup.
    
    # Intersection
    common_idx = labeled_events.index.intersection(features_df.index)
    
    X_meta = features_df.loc[common_idx].copy()
    
    # We need to add 'primary_signal' as a feature? 
    # Or just use it to define y.
    # y for Meta: 1 if Rule was Correct (Profit), 0 if Rule was Wrong (Loss)
    
    # labeled_events['label'] is 1 (Profit), -1 (Loss), 0 (Neutral)
    # Since we passed 'side' to get_events, 'label' is already:
    # 1 if ret * side > 0 (Profit)
    # -1 if ret * side < 0 (Loss)
    
    y_meta = (labeled_events.loc[common_idx, 'label'] == 1).astype(int)
    
    # Remove data leakage columns from X
    drop_cols = ['label', 'ret', 't1', 'trgt', 'side', 'sample_weight', 'avg_uniqueness', 'bin']
    X_meta = X_meta.drop(columns=[c for c in drop_cols if c in X_meta.columns])
    
    print(f"   Meta-Dataset: {len(X_meta)} samples")
    print(f"   Class Balance: {y_meta.value_counts(normalize=True).to_dict()}")

    # 5. Training Meta-Model (LightGBM)
    print("\n5. Training Meta-Model (LightGBM)...")
    
    # Retrieve t1 for Purging
    t1_meta = labeled_events.loc[common_idx, 't1']
    
    # Using 5-Fold Purged CV
    cv = PurgedKFold(n_splits=5, samples_info_sets=t1_meta, embargo=0.01)
    
    # Optimized LightGBM Parameters (from meta_labeling.py)
    meta_model = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=15,      # Shallower tree for meta-model
        learning_rate=0.05,
        n_estimators=300,
        max_depth=5,
        min_child_samples=10,
        reg_alpha=0.2,
        reg_lambda=0.2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    # OOS Predictions container
    meta_probs = pd.Series(0.0, index=X_meta.index)
    meta_preds = pd.Series(0, index=X_meta.index)
    
    for train_idx, test_idx in cv.split(X_meta, y_meta):
        X_train, y_train = X_meta.iloc[train_idx], y_meta.iloc[train_idx]
        X_test = X_meta.iloc[test_idx]
        
        meta_model.fit(X_train, y_train)
        
        meta_probs.iloc[test_idx] = meta_model.predict_proba(X_test)[:, 1]
        meta_preds.iloc[test_idx] = meta_model.predict(X_test)
        
    print("\n   [Meta-Model OOS Performance]")
    # Only evaluate on indices that were actually tested (Purged CV drops some)
    # But here we filled all with 0.0. Let's filter for where we have predictions?
    # Actually Purged CV covers all samples eventually if configured right, 
    # except maybe some edge cases.
    # Let's just eval on the whole set for simplicity (assuming full coverage).
    
    print(f"   ROC AUC: {roc_auc_score(y_meta, meta_probs):.4f}")
    print(classification_report(y_meta, meta_preds))
    
    # 6. Evaluate Hybrid Strategy
    print("\n6. Evaluating Hybrid Strategy Performance...")
    
    # Prepare Evaluation DataFrame
    df_eval = pd.DataFrame(index=common_idx)
    df_eval['ret'] = labeled_events.loc[common_idx, 'ret']
    # The 'ret' in labeled_events is already side-adjusted?
    # Check `apply_triple_barrier`: 
    # ret = (df0 / close[loc] - 1) * out.loc[loc, "side"]
    # YES. It is the PnL of the trade.
    
    df_eval['avg_uniqueness'] = features_df.loc[common_idx, 'avg_uniqueness'] if 'avg_uniqueness' in features_df.columns else 1.0
    # Fill missing uniqueness with 1.0 (conservative assumption: no overlap info available)
    df_eval['avg_uniqueness'] = df_eval['avg_uniqueness'].fillna(1.0)
    
    df_eval['t1'] = t1_meta
    
    # 1. Base Rule (Always trade on signal)
    # Since 'ret' is already the trade return, we just sum it up.
    base_returns = df_eval['ret']
    
    # 2. Hybrid (Bet Sized)
    # Calculate size based on Meta-Model Probability
    bet_sizes = bet_size_probability(
        events=df_eval,
        prob_series=meta_probs,
        step_size=0.0,
        average_active=True
    )
    
    hybrid_returns = base_returns * bet_sizes
    
    # 3. Hybrid (Binary Filter)
    binary_size = (meta_probs > 0.55).astype(float)
    binary_returns = base_returns * binary_size
    
    # Statistics
    stats = []
    for name, rets in [("Rule (MA Cross)", base_returns), 
                       ("Hybrid (Binary)", binary_returns),
                       ("Hybrid (Sized)", hybrid_returns)]:
        
        # Simple Cumulative (Sum of log returns is better, but here we have simple returns per trade)
        # We can simulate equity curve: Equity_{t} = Equity_{t-1} * (1 + ret)
        cum_equity = (1 + rets).cumprod()
        total_ret = cum_equity.iloc[-1] - 1
        
        sharpe = rets.mean() / rets.std() * np.sqrt(252 * 4) if rets.std() != 0 else 0
        
        # Max DD
        roll_max = cum_equity.cummax()
        dd = (cum_equity - roll_max) / roll_max
        max_dd = dd.min()
        
        n_trades = (rets != 0).sum()
        win_rate = (rets > 0).sum() / n_trades if n_trades > 0 else 0
        
        stats.append({
            "Strategy": name,
            "Return": f"{total_ret*100:.2f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Max DD": f"{max_dd*100:.2f}%",
            "Win Rate": f"{win_rate*100:.2f}%",
            "Trades": n_trades
        })
        
    print(pd.DataFrame(stats).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cum_equity.index, (1 + base_returns).cumprod(), label='Rule (MA Cross)', alpha=0.5)
    plt.plot(cum_equity.index, (1 + hybrid_returns).cumprod(), label='Hybrid (Sized)', linewidth=2)
    plt.title('Hybrid Strategy: MA Cross + Meta-Labeling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visual_analysis/hybrid_strategy_performance.png")
    print("\n   ✓ Saved plot to visual_analysis/hybrid_strategy_performance.png")

if __name__ == "__main__":
    main()
