"""
Feature Engineering 2.0 - Integrated Pipeline

Combines all feature generators:
1. Alpha158 (Baseline)
2. Fractional Differentiation (Memory Preservation)
3. Market Regime Features
4. Microstructure Features (NEW - AFML Chapter 19)
5. Signal-Based Features (NEW)

This creates a comprehensive feature set to breakthrough the 0.524 AUC ceiling.

Reference:
- AFML Chapter 19: Feature Engineering
- MLFinLab: Feature Engineering Best Practices
"""

import pandas as pd
import numpy as np
import os
from features import Alpha158FeatureGenerator, FracDiffFeatureGenerator, MarketRegimeFeatureGenerator
from microstructure_features import MicrostructureFeatureGenerator
from signal_features import SignalFeatureGenerator


def main():
    """Generate comprehensive features combining all generators."""
    print("=" * 80)
    print("FEATURE ENGINEERING 2.0 - INTEGRATED PIPELINE")
    print("=" * 80)

    # 1. Load dollar bars
    print("\n1. Loading dollar bars...")
    try:
        input_file = os.path.join("data", "output", "dynamic_dollar_bars.csv")
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'dynamic_dollar_bars.csv' not found.")
        return

    print(f"   Loaded {len(df)} bars")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # 2. Generate Alpha158 features (Baseline)
    print("\n2. Generating Alpha158 features...")
    alpha_gen = Alpha158FeatureGenerator()
    features_alpha = alpha_gen.generate(df)
    print(f"   ✓ Alpha158: {len(features_alpha.columns)} features")

    # 3. Generate FFD-based Momentum features
    print("\n3. Generating FFD-based Momentum features...")
    ffd_gen = FracDiffFeatureGenerator(
        check_stationarity=True,
        windows=[5, 10, 20, 30, 50],
        normalize=True
    )
    features_ffd = ffd_gen.generate(df, content_cols=["close", "volume"])
    print(f"   ✓ FFD: {len(features_ffd.columns)} features")

    # 4. Generate Market Regime features
    print("\n4. Generating Market Regime features...")
    regime_gen = MarketRegimeFeatureGenerator(windows=[20, 50, 100])
    features_regime = regime_gen.generate(df)
    print(f"   ✓ Regime: {len(features_regime.columns)} features")

    # 5. Generate Microstructure features (NEW)
    print("\n5. Generating Microstructure features...")
    micro_gen = MicrostructureFeatureGenerator(vpin_buckets=50, windows=[20, 50, 100])
    features_micro = micro_gen.generate(df)
    print(f"   ✓ Microstructure: {len(features_micro.columns)} features")

    # 6. Generate Signal-Based features (NEW)
    print("\n6. Generating Signal-Based features...")
    signal_gen = SignalFeatureGenerator()
    features_signal = signal_gen.generate(df)
    print(f"   ✓ Signal: {len(features_signal.columns)} features")

    # 7. Combine all features
    print("\n7. Combining all features...")
    features = pd.concat(
        [features_alpha, features_ffd, features_regime, features_micro, features_signal],
        axis=1
    )

    print(f"   Total features before cleanup: {len(features.columns)}")

    # 8. Clean up (drop NaN rows from warmup)
    original_len = len(features)
    features = features.dropna()
    print(f"   Dropped {original_len - len(features)} rows (warmup period)")
    print(f"   ✓ Final feature set: {len(features)} samples x {len(features.columns)} features")

    # 9. Save full feature set
    print("\n8. Saving feature sets...")
    
    # Save comprehensive feature set
    # Save comprehensive feature set
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    features.to_csv(os.path.join(output_dir, "features_v2.csv"))
    print(f"   ✓ Saved comprehensive features to: features_v2.csv")
    
    # Save individual feature sets for analysis
    features_micro.to_csv(os.path.join(output_dir, "features_microstructure.csv"))
    features_signal.to_csv(os.path.join(output_dir, "features_signal.csv"))
    print(f"   ✓ Saved microstructure features to: features_microstructure.csv")
    print(f"   ✓ Saved signal features to: features_signal.csv")

    # 10. Merge with labels (if available)
    print("\n9. Merging with labels...")
    try:
        labels = pd.read_csv(os.path.join("data", "output", "labeled_events.csv"), index_col=0, parse_dates=True)
        # Standardize column names
        if "ret" in labels.columns:
            labels = labels.rename(columns={"ret": "return"})
            
        # Calculate holding period if missing (t1 - index in hours)
        if "holding_period" not in labels.columns and "t1" in labels.columns:
            labels["t1"] = pd.to_datetime(labels["t1"])
            labels["holding_period"] = (labels["t1"] - labels.index).dt.total_seconds() / 3600

        # Merge features with labels
        features_labeled = features.join(labels[["label", "return", "holding_period"]], how="inner")
        
        # Load sample weights if available
        try:
            weights = pd.read_csv(os.path.join("data", "output", "sample_weights.csv"), index_col=0, parse_dates=True)
            features_labeled = features_labeled.join(weights[["sample_weight", "avg_uniqueness"]], how="left")
            print(f"   ✓ Merged with sample weights")
        except FileNotFoundError:
            print(f"   ! Sample weights not found, skipping")
        
        features_labeled.to_csv(os.path.join("data", "output", "features_v2_labeled.csv"))
        print(f"   ✓ Saved labeled features to: features_v2_labeled.csv")
        print(f"   ✓ Labeled samples: {len(features_labeled)}")
        
        # Label distribution
        print(f"\n   Label distribution:")
        print(f"   Loss (-1):  {(features_labeled['label'] == -1).sum()} ({(features_labeled['label'] == -1).mean()*100:.2f}%)")
        print(f"   Profit (1): {(features_labeled['label'] == 1).sum()} ({(features_labeled['label'] == 1).mean()*100:.2f}%)")
        
    except FileNotFoundError:
        print("   ! labeled_events.csv not found, skipping label merge")

    # 11. Feature Summary
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING 2.0 SUMMARY")
    print("-" * 80)
    print(f"Total Features:       {len(features.columns)}")
    print(f"  - Alpha158:         {len(features_alpha.columns)}")
    print(f"  - FFD (Stationary): {len(features_ffd.columns)}")
    print(f"  - Regime (State):   {len(features_regime.columns)}")
    print(f"  - Microstructure:   {len(features_micro.columns)} 🆕")
    print(f"  - Signal-Based:     {len(features_signal.columns)} 🆕")
    print()
    print(f"Sample Count:         {len(features)}")
    print(f"Memory Usage:         {features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # FFD Parameters
    print("\nOptimized FFD Parameters (Memory Preservation):")
    for col, d in ffd_gen.d_values.items():
        w = ffd_gen.get_weights_ffd(d, 1e-4, len(df))
        print(f"  {col.upper():<10} | d = {d:.2f} | Memory Window: {len(w):>4} bars")

    print("\n" + "=" * 80)
    print("✓ Feature Engineering 2.0 Complete!")
    print("\nNext Steps:")
    print("  1. Run feature importance analysis with new features")
    print("  2. Retrain model with expanded feature set")
    print("  3. Compare AUC against baseline (0.524)")


if __name__ == "__main__":
    main()
