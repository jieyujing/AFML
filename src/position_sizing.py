"""
Position Sizing Pipeline based on AFML Chapter 10.

This script serves as the bridge between Model Training (Step 7) and Backtesting (Step 8).
It generates position sizes based on model probabilities using:
1. Gaussian-based Bet Sizing (prob -> z-score -> CDF -> bet size)
2. Concurrency Averaging (using avg_uniqueness as scaling factor)

Reference:
- Advances in Financial Machine Learning, Chapter 10
- mlfinlab skill: Portfolio Optimization & Bet Sizing
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Note: bet_sizing module provides bet_size_probability and avg_active_signals
# Here we use ProbabilisticBetSizer class which encapsulates these concepts


class ProbabilisticBetSizer:
    """
    Probabilistic Bet Sizing based on AFML Chapter 10.
    
    Converts model probabilities to bet sizes using:
    - Gaussian transformation (Sigmoid-like mapping)
    - Concurrency adjustment (using avg_uniqueness)
    
    Reference:
    - AFML Chapter 10: Bet Sizing
    """
    
    def __init__(
        self,
        step_size: float = 0.0,
        average_active: bool = True,
        max_leverage: float = 1.0,
        prob_threshold: float = 0.5
    ):
        """
        Initialize the bet sizer.
        
        Args:
            step_size: Discretization step (0.0 = continuous)
            average_active: Whether to scale by concurrency (avg_uniqueness)
            max_leverage: Maximum bet size (1.0 = fully invested)
            prob_threshold: Minimum probability to take a position
        """
        self.step_size = step_size
        self.average_active = average_active
        self.max_leverage = max_leverage
        self.prob_threshold = prob_threshold
        
    def fit(self, prob_train: pd.Series) -> 'ProbabilisticBetSizer':
        """
        Calibrate the bet sizer using training probabilities.
        
        Currently stores statistics for potential future use
        (e.g., calibrated Gaussian parameters).
        
        Args:
            prob_train: Training set probabilities
            
        Returns:
            self
        """
        self.prob_mean_ = prob_train.mean()
        self.prob_std_ = prob_train.std()
        return self
    
    def transform(
        self,
        events: pd.DataFrame,
        prob_series: pd.Series,
        pred_series: pd.Series = None
    ) -> pd.Series:
        """
        Transform probabilities to bet sizes.
        
        Args:
            events: DataFrame with 't1' (vertical barrier) and optionally 'avg_uniqueness'
            prob_series: Series of model probabilities (P(Class=1))
            pred_series: Primary model predictions (-1, 1). If None, returns unsigned size.
            
        Returns:
            Series of bet sizes in [-max_leverage, max_leverage]
        """
        # 1. Prepare probabilities
        p = prob_series.copy().clip(0.001, 0.999)
        
        # 2. Calculate z-score (test statistic for H0: p = 0.5)
        # z = (p - 0.5) / sqrt(p * (1 - p))
        z = (p - 0.5) / np.sqrt(p * (1 - p))
        
        # 3. Calculate bet size using CDF
        # m = 2 * Phi(z) - 1, where Phi is standard normal CDF
        # This maps z to [-1, 1] via a sigmoid-like function
        m = 2 * norm.cdf(z) - 1
        
        # 4. Apply threshold (only bet if confident enough)
        # If p < prob_threshold, bet size = 0
        bet_sizes = pd.Series(m, index=p.index)
        bet_sizes[p < self.prob_threshold] = 0.0
        
        # 5. Apply direction from primary model (if provided)
        if pred_series is not None:
            # Align indices
            aligned_preds = pred_series.reindex(bet_sizes.index).fillna(0)
            bet_sizes = bet_sizes * aligned_preds
        
        # 6. Discretize (optional)
        if self.step_size > 0:
            bet_sizes = (bet_sizes / self.step_size).round() * self.step_size
        
        # 7. Apply concurrency scaling (using avg_uniqueness)
        if self.average_active and 'avg_uniqueness' in events.columns:
            # avg_uniqueness is in [0, 1], so this scales down overlapping bets
            aligned_uniqueness = events['avg_uniqueness'].reindex(bet_sizes.index).fillna(1.0)
            bet_sizes = bet_sizes * aligned_uniqueness
        
        # 8. Apply max leverage constraint
        bet_sizes = bet_sizes.clip(-self.max_leverage, self.max_leverage)
        
        return bet_sizes
    
    def fit_transform(
        self,
        events: pd.DataFrame,
        prob_train: pd.Series,
        prob_series: pd.Series,
        pred_series: pd.Series = None
    ) -> pd.Series:
        """Fit on training probs and transform the target probs."""
        self.fit(prob_train)
        return self.transform(events, prob_series, pred_series)


def load_data():
    """Load the required datasets."""
    print("=" * 60)
    print("POSITION SIZING PIPELINE (AFML Chapter 10)")
    print("=" * 60)
    
    # Load PCA features (includes labels, weights, and metadata)
    pca_path = os.path.join("data", "output", "features_pca.csv")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA features not found: {pca_path}")
    
    df_pca = pd.read_csv(pca_path)
    
    # Handle index
    if 'Unnamed: 0' in df_pca.columns:
        df_pca.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    if 'date' in df_pca.columns:
        df_pca['date'] = pd.to_datetime(df_pca['date'])
        df_pca.set_index('date', inplace=True)
    
    # Load labeled events for t1 and avg_uniqueness
    events_path = os.path.join("data", "output", "labeled_events.csv")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Labeled events not found: {events_path}")
    
    events = pd.read_csv(events_path, index_col=0, parse_dates=True)
    
    # Load sample weights for avg_uniqueness
    weights_path = os.path.join("data", "output", "sample_weights.csv")
    if os.path.exists(weights_path):
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        if 'avg_uniqueness' in weights.columns:
            events = events.join(weights[['avg_uniqueness']])
    
    # Merge t1 into df_pca
    df_pca = df_pca.join(events[['t1', 'ret', 'side', 'avg_uniqueness']], rsuffix='_events')
    
    if 't1' not in df_pca.columns and 't1_events' in df_pca.columns:
        df_pca['t1'] = df_pca['t1_events']
    
    if 't1' in df_pca.columns:
        df_pca['t1'] = pd.to_datetime(df_pca['t1'])
    
    df_pca = df_pca.dropna(subset=['label'])
    df_pca = df_pca.sort_index()
    
    print(f"Loaded {len(df_pca)} events with PCA features")
    print(f"Date Range: {df_pca.index.min()} to {df_pca.index.max()}")
    
    return df_pca, events


def generate_cv_probabilities(df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """
    Generate out-of-sample probabilities using Purged Cross-Validation.
    
    This simulates what the model would predict for each sample
    using only information available up to that point.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        
    Returns:
        Series of OOS probabilities
    """
    from src.cv_setup import PurgedKFold
    
    print("\nGenerating OOS probabilities via Purged 5-Fold CV...")
    
    X = df[feature_cols]
    y = df['label'].astype(int)
    w = df['sample_weight'] if 'sample_weight' in df.columns else None
    
    # Load best hyperparameters
    params_path = os.path.join("data", "output", "best_hyperparameters.csv")
    if os.path.exists(params_path):
        params_df = pd.read_csv(params_path)
        params = params_df.iloc[0].to_dict()
        if 'best_auc' in params:
            del params['best_auc']
        for p in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if p in params:
                params[p] = int(params[p])
        params['class_weight'] = 'balanced_subsample'
        params['random_state'] = 42
        params['n_jobs'] = -1
    else:
        params = {
            'n_estimators': 1000,
            'max_depth': 5,
            'class_weight': 'balanced_subsample',
            'criterion': 'entropy',
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Prepare samples_info_sets (t1 for each sample) for purging
    samples_info_sets = df['t1']
    
    # Initialize Purged K-Fold (embargo = 1% of dataset)
    cv = PurgedKFold(n_splits=5, samples_info_sets=samples_info_sets, embargo=0.01)
    
    # Generate OOS predictions
    oos_probs = pd.Series(index=df.index, dtype=float)
    oos_probs[:] = np.nan
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        w_train = w.iloc[train_idx] if w is not None else None
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        oos_probs.iloc[test_idx] = probs
        
        print(f"  Fold {fold_idx + 1}/5: {len(test_idx)} samples processed")
    
    return oos_probs.dropna()


def visualize_bet_sizing(df: pd.DataFrame, bet_sizes: pd.Series, probs: pd.Series):
    """
    Visualize the relationship between probabilities and bet sizes.
    
    Args:
        df: DataFrame with labels and returns
        bet_sizes: Series of computed bet sizes
        probs: Series of model probabilities
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability Distribution
    ax1 = axes[0, 0]
    ax1.hist(probs, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
    ax1.axvline(0.5, color='red', linestyle='--', label='Decision Boundary')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Model Probability Distribution')
    ax1.legend()
    
    # 2. Probability to Bet Size Mapping
    ax2 = axes[0, 1]
    ax2.scatter(probs, bet_sizes, alpha=0.3, s=10, c='steelblue')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Bet Size')
    ax2.set_title('Probability → Bet Size Transformation')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # 3. Bet Size Distribution
    ax3 = axes[1, 0]
    ax3.hist(bet_sizes, bins=50, edgecolor='white', alpha=0.7, color='forestgreen')
    ax3.axvline(0, color='red', linestyle='--', label='No Position')
    ax3.set_xlabel('Bet Size')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bet Size Distribution')
    ax3.legend()
    
    # 4. Expected PnL by Bet Size Bucket
    ax4 = axes[1, 1]
    df_viz = pd.DataFrame({'bet_size': bet_sizes, 'ret': df.loc[bet_sizes.index, 'ret']})
    df_viz['pnl'] = df_viz['bet_size'] * df_viz['ret']
    df_viz['bucket'] = pd.cut(df_viz['bet_size'].abs(), bins=5)
    bucket_pnl = df_viz.groupby('bucket')['pnl'].mean()
    bucket_pnl.plot(kind='bar', ax=ax4, color='darkorange', edgecolor='white')
    ax4.set_xlabel('|Bet Size| Bucket')
    ax4.set_ylabel('Average PnL')
    ax4.set_title('Expected PnL by Bet Size')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    os.makedirs('visual_analysis', exist_ok=True)
    plt.savefig('visual_analysis/bet_sizing_analysis.png', dpi=150)
    print("Saved visualization to visual_analysis/bet_sizing_analysis.png")
    plt.close()


def main():
    """Main execution pipeline for position sizing."""
    # 1. Load Data
    df, events = load_data()
    feature_cols = [c for c in df.columns if c.startswith('PC_')]
    
    print(f"\nFeatures: {len(feature_cols)} PCA components")
    
    # 2. Generate OOS Probabilities
    probs = generate_cv_probabilities(df, feature_cols)
    
    print("\nProbability Statistics:")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Std:  {probs.std():.4f}")
    print(f"  Min:  {probs.min():.4f}")
    print(f"  Max:  {probs.max():.4f}")
    
    # 3. Initialize Bet Sizer
    bet_sizer = ProbabilisticBetSizer(
        step_size=0.0,        # Continuous bet sizes
        average_active=True,  # Scale by avg_uniqueness
        max_leverage=1.0,     # Maximum bet = 100% capital
        prob_threshold=0.5    # Only bet if prob > 0.5
    )
    
    # 4. Generate Bet Sizes
    print("\nCalculating bet sizes...")
    
    # Create prediction series (direction: +1 for prob > 0.5, -1 otherwise)
    # Note: In Triple Barrier, label=1 means profit, label=0 means loss.
    # The 'side' column in labeled_events tells us Long (+1) or Short (-1).
    # For simplicity, we treat prob > 0.5 as "take the trade" with original side.
    
    if 'side' in df.columns:
        pred_series = df.loc[probs.index, 'side']
    else:
        # Default to Long if side not available
        pred_series = pd.Series(1, index=probs.index)
    
    # Prepare events DataFrame for bet sizing
    events_for_sizing = df.loc[probs.index, ['t1', 'avg_uniqueness']].copy()
    
    bet_sizes = bet_sizer.fit_transform(
        events=events_for_sizing,
        prob_train=probs,
        prob_series=probs,
        pred_series=pred_series
    )
    
    # 5. Report Statistics
    print("\nBet Size Statistics:")
    print(f"  Mean:          {bet_sizes.mean():.4f}")
    print(f"  Std:           {bet_sizes.std():.4f}")
    print(f"  No Position:   {(bet_sizes == 0).sum()} ({(bet_sizes == 0).mean():.1%})")
    print(f"  Long Bets:     {(bet_sizes > 0).sum()} ({(bet_sizes > 0).mean():.1%})")
    print(f"  Short Bets:    {(bet_sizes < 0).sum()} ({(bet_sizes < 0).mean():.1%})")
    
    # 6. Calculate theoretical PnL (no costs)
    rets = df.loc[probs.index, 'ret']
    pnl = bet_sizes * rets
    
    sharpe_raw = pnl.mean() / pnl.std() * np.sqrt(252 * 4) if pnl.std() > 0 else 0
    
    print("\nTheoretical Performance (Before Costs):")
    print(f"  Total Return:  {pnl.sum():.4f}")
    print(f"  Sharpe Ratio:  {sharpe_raw:.2f}")
    
    # 7. Save Results
    output = pd.DataFrame({
        'probability': probs,
        'bet_size': bet_sizes,
        'side': pred_series,
        'ret': rets
    })
    output['pnl'] = output['bet_size'] * output['ret']
    
    output_path = os.path.join("data", "output", "position_sizes.csv")
    output.to_csv(output_path)
    print(f"\nSaved position sizes to {output_path}")
    
    # 8. Visualize
    visualize_bet_sizing(df, bet_sizes, probs)
    
    print("\n" + "=" * 60)
    print("POSITION SIZING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
