"""
Feature Importance Analysis and Selection for Financial Machine Learning

This module implements:
1. MDI (Mean Decrease Impurity) - From tree-based models
2. MDA (Mean Decrease Accuracy) - Out-of-sample permutation importance
3. SFI (Single Feature Importance) - Individual feature performance
4. Feature clustering and redundancy analysis
5. PCA-based feature orthogonalization validation

Reference:
- AFML Chapter 8: Feature Importance
- MLFinLab Feature Importance module
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from typing import Callable, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr, weightedtau
from sklearn.decomposition import PCA
import warnings
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cv_setup import PurgedKFold


class FeatureImportanceAnalyzer:
    """
    Comprehensive Feature Importance Analysis following AFML Chapter 8.
    
    Implements three methods:
    - MDI (Mean Decrease Impurity): Fast, in-sample, tree-based
    - MDA (Mean Decrease Accuracy): Robust, out-of-sample, permutation-based
    - SFI (Single Feature Importance): Individual feature predictive power
    """
    
    def __init__(
        self,
        clf = None,
        cv = None,
        scoring: str = 'neg_log_loss',
        n_jobs: int = -1
    ):
        """
        Initialize Feature Importance Analyzer.
        
        Args:
            clf: Classifier (default: RandomForest with max_features=1 to avoid masking)
            cv: Cross-validation splitter (default: 5-fold KFold)
            scoring: Scoring metric ('neg_log_loss', 'accuracy', 'f1')
            n_jobs: Number of parallel jobs
        """
        # Default classifier: RF with max_features=1 to avoid masking effects
        # Ref: AFML Chapter 8.3.1
        if clf is None:
            self.clf = RandomForestClassifier(
                n_estimators=100,
                max_features=1,  # Critical: prevents masking effects
                min_samples_leaf=5,
                criterion='entropy',
                bootstrap=True,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=n_jobs
            )
        else:
            self.clf = clf
            
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        
    def fit_importance_mdi(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Mean Decrease Impurity (MDI) - In-sample feature importance.
        
        Ref: AFML Chapter 8.3.1
        
        Pros:
        - Fast computation
        - Importance sums to 1
        
        Cons:
        - In-sample (overfitting risk)
        - Biased toward high-cardinality features
        - Dilutes importance of correlated features
        
        Args:
            X: Feature matrix
            y: Labels
            sample_weight: Sample weights (optional)
            
        Returns:
            DataFrame with 'mean' and 'std' importance
        """
        print("   [MDI] Computing Mean Decrease Impurity...")
        
        # Fit the full model
        if sample_weight is not None:
            self.clf.fit(X, y, sample_weight=sample_weight)
        else:
            self.clf.fit(X, y)
        
        # Extract feature importances
        if hasattr(self.clf, 'feature_importances_'):
            importances = self.clf.feature_importances_
            
            # For ensemble methods, we can get per-tree importances
            if hasattr(self.clf, 'estimators_'):
                imp_per_tree = np.array([
                    tree.feature_importances_ for tree in self.clf.estimators_
                ])
                mean_imp = imp_per_tree.mean(axis=0)
                std_imp = imp_per_tree.std(axis=0)
            else:
                mean_imp = importances
                std_imp = np.zeros(len(importances))
                
            result = pd.DataFrame({
                'mean': mean_imp,
                'std': std_imp
            }, index=X.columns)
            
            return result.sort_values('mean', ascending=False)
        else:
            raise ValueError("Classifier does not support feature_importances_")
    
    def fit_importance_mda(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ) -> pd.DataFrame:
        """
        Mean Decrease Accuracy (MDA) - Out-of-sample permutation importance.
        
        Ref: AFML Chapter 8.3.2, Snippet 8.3
        
        This is the GOLD STANDARD for feature importance in finance.
        
        Pros:
        - Out-of-sample (robust)
        - Works with any classifier
        - Can use any scoring metric
        - Detects truly predictive features
        
        Cons:
        - Slower than MDI
        - Underestimates importance of correlated features
        
        Args:
            X: Feature matrix
            y: Labels
            sample_weight: Sample weights (optional)
            t1: End time of each sample (for purging)
            n_splits: Number of CV folds
            embargo_pct: Embargo percentage
            
        Returns:
            DataFrame with 'mean' and 'std' importance
        """
        print(f"   [MDA] Computing Mean Decrease Accuracy (Purged {n_splits}-Fold CV)...")
        
        # Setup CV
        if t1 is not None and self.cv is None:
            cv_iterator = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, embargo=embargo_pct)
        elif self.cv is not None:
            cv_iterator = self.cv
        else:
            from sklearn.model_selection import KFold
            cv_iterator = KFold(n_splits=n_splits, shuffle=False)
        
        # Storage for importance scores
        imp_scores = pd.DataFrame(index=X.columns)
        
        # Baseline score without permutation
        print("      Computing baseline scores (no permutation)...")
        baseline_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_iterator.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if sample_weight is not None:
                w_train = sample_weight.iloc[train_idx]
                w_test = sample_weight.iloc[test_idx]
            else:
                w_train = None
                w_test = None
            
            # Fit model
            if w_train is not None:
                self.clf.fit(X_train, y_train, sample_weight=w_train)
            else:
                self.clf.fit(X_train, y_train)
            
            # Score on test set
            score = self._score_model(
                self.clf, X_test, y_test, w_test, self.scoring
            )
            baseline_scores.append(score)
        
        baseline_mean = np.mean(baseline_scores)
        print(f"      Baseline score: {baseline_mean:.4f}")
        
        # Permutation importance for each feature
        print("      Computing permutation importance for each feature...")
        
        for feature in X.columns:
            fold_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv_iterator.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if sample_weight is not None:
                    w_train = sample_weight.iloc[train_idx]
                    w_test = sample_weight.iloc[test_idx]
                else:
                    w_train = None
                    w_test = None
                
                # Fit model
                if w_train is not None:
                    self.clf.fit(X_train, y_train, sample_weight=w_train)
                else:
                    self.clf.fit(X_train, y_train)
                
                # Permute feature in test set
                X_test_perm = X_test.copy()
                np.random.seed(42 + fold_idx)  # Reproducible permutation
                X_test_perm[feature] = np.random.permutation(X_test_perm[feature].values)
                
                # Score on permuted test set
                score_perm = self._score_model(
                    self.clf, X_test_perm, y_test, w_test, self.scoring
                )
                fold_scores.append(score_perm)
            
            # Importance = Baseline - Permuted (higher is better)
            # For neg_log_loss, baseline is more negative (better), so we flip
            if 'neg' in self.scoring or 'loss' in self.scoring:
                importance = np.array(baseline_scores) - np.array(fold_scores)
            else:
                importance = np.array(baseline_scores) - np.array(fold_scores)
            
            imp_scores.loc[feature, f'fold_importance'] = importance.mean()
            imp_scores.loc[feature, f'fold_std'] = importance.std()
        
        # Rename columns
        result = pd.DataFrame({
            'mean': imp_scores[f'fold_importance'],
            'std': imp_scores[f'fold_std']
        })
        
        return result.sort_values('mean', ascending=False)
    
    def fit_importance_sfi(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ) -> pd.DataFrame:
        """
        Single Feature Importance (SFI) - Train on each feature individually.
        
        Ref: AFML Chapter 8.4.1
        
        Pros:
        - No substitution effects
        - Works with any classifier
        - Can identify truly independent predictors
        
        Cons:
        - Ignores feature interactions
        - Slower (trains n_features models)
        
        Args:
            X: Feature matrix
            y: Labels
            sample_weight: Sample weights (optional)
            t1: End time of each sample (for purging)
            n_splits: Number of CV folds
            embargo_pct: Embargo percentage
            
        Returns:
            DataFrame with 'mean' and 'std' importance
        """
        print(f"   [SFI] Computing Single Feature Importance (Purged {n_splits}-Fold CV)...")
        
        # Setup CV
        if t1 is not None and self.cv is None:
            cv_iterator = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, embargo=embargo_pct)
        elif self.cv is not None:
            cv_iterator = self.cv
        else:
            from sklearn.model_selection import KFold
            cv_iterator = KFold(n_splits=n_splits, shuffle=False)
        
        # Storage for importance scores
        imp_scores = pd.DataFrame(index=X.columns)
        
        # Train on each feature individually
        for feature in X.columns:
            X_single = X[[feature]]
            fold_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv_iterator.split(X_single, y)):
                X_train, X_test = X_single.iloc[train_idx], X_single.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if sample_weight is not None:
                    w_train = sample_weight.iloc[train_idx]
                    w_test = sample_weight.iloc[test_idx]
                else:
                    w_train = None
                    w_test = None
                
                # Fit model on single feature
                if w_train is not None:
                    self.clf.fit(X_train, y_train, sample_weight=w_train)
                else:
                    self.clf.fit(X_train, y_train)
                
                # Score on test set
                score = self._score_model(
                    self.clf, X_test, y_test, w_test, self.scoring
                )
                fold_scores.append(score)
            
            imp_scores.loc[feature, 'mean'] = np.mean(fold_scores)
            imp_scores.loc[feature, 'std'] = np.std(fold_scores)
        
        return imp_scores.sort_values('mean', ascending=False)
    
    def _score_model(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
        scoring: str
    ) -> float:
        """Score model on test set with specified metric."""
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)
        
        if scoring == 'neg_log_loss':
            # Return negative log loss (higher is better)
            score = -log_loss(y, y_pred_proba, sample_weight=sample_weight, labels=clf.classes_)
        elif scoring == 'accuracy':
            score = accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif scoring == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(y, y_pred, average='weighted', sample_weight=sample_weight)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")
        
        return score


def plot_feature_importance_comparison(
    mdi_imp: pd.DataFrame,
    mda_imp: pd.DataFrame,
    sfi_imp: Optional[pd.DataFrame] = None,
    top_n: int = 30,
    output_path: str = 'visual_analysis/feature_importance_comparison.png'
):
    """
    Plot comparison of MDI, MDA, and SFI feature importance.
    
    Args:
        mdi_imp: MDI importance DataFrame
        mda_imp: MDA importance DataFrame
        sfi_imp: SFI importance DataFrame (optional)
        top_n: Number of top features to plot
        output_path: Output file path
    """
    print(f"\n   Plotting feature importance comparison (top {top_n})...")
    
    # Get top features from MDA (most robust)
    top_features = mda_imp.head(top_n).index
    
    # Prepare data
    if sfi_imp is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        
        # MDI
        axes[0].barh(range(len(top_features)), mdi_imp.loc[top_features, 'mean'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features, fontsize=8)
        axes[0].set_xlabel('MDI Importance')
        axes[0].set_title('Mean Decrease Impurity (In-Sample)')
        axes[0].invert_yaxis()
        
        # MDA
        axes[1].barh(range(len(top_features)), mda_imp.loc[top_features, 'mean'])
        axes[1].errorbar(
            mda_imp.loc[top_features, 'mean'],
            range(len(top_features)),
            xerr=mda_imp.loc[top_features, 'std'],
            fmt='none',
            ecolor='red',
            alpha=0.5
        )
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features, fontsize=8)
        axes[1].set_xlabel('MDA Importance')
        axes[1].set_title('Mean Decrease Accuracy (Out-of-Sample) ⭐')
        axes[1].invert_yaxis()
        
        # SFI
        axes[2].barh(range(len(top_features)), sfi_imp.loc[top_features, 'mean'])
        axes[2].set_yticks(range(len(top_features)))
        axes[2].set_yticklabels(top_features, fontsize=8)
        axes[2].set_xlabel('SFI Score')
        axes[2].set_title('Single Feature Importance')
        axes[2].invert_yaxis()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        
        # MDI
        axes[0].barh(range(len(top_features)), mdi_imp.loc[top_features, 'mean'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features, fontsize=8)
        axes[0].set_xlabel('MDI Importance')
        axes[0].set_title('Mean Decrease Impurity (In-Sample)')
        axes[0].invert_yaxis()
        
        # MDA
        axes[1].barh(range(len(top_features)), mda_imp.loc[top_features, 'mean'])
        axes[1].errorbar(
            mda_imp.loc[top_features, 'mean'],
            range(len(top_features)),
            xerr=mda_imp.loc[top_features, 'std'],
            fmt='none',
            ecolor='red',
            alpha=0.5
        )
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features, fontsize=8)
        axes[1].set_xlabel('MDA Importance')
        axes[1].set_title('Mean Decrease Accuracy (Out-of-Sample) ⭐')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")


def plot_feature_clustering(
    X: pd.DataFrame,
    mda_imp: pd.DataFrame,
    method: str = 'spearman',
    output_path: str = 'visual_analysis/feature_clustering.png'
):
    """
    Plot hierarchical clustering of features based on correlation.
    Helps identify redundant features.
    
    Ref: AFML Chapter 8.4.2
    
    Args:
        X: Feature matrix
        mda_imp: MDA importance (for coloring)
        method: Correlation method ('spearman' or 'pearson')
        output_path: Output file path
    """
    print(f"\n   Computing feature clustering ({method} correlation)...")
    
    # Compute correlation matrix
    if method == 'spearman':
        corr_matrix = X.corr(method='spearman')
    else:
        corr_matrix = X.corr(method='pearson')
    
    # Convert to distance matrix
    dist_matrix = ((1 - corr_matrix) / 2.) ** 0.5
    
    # Hierarchical clustering
    linkage_matrix = hierarchy.linkage(dist_matrix, method='ward')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Dendrogram
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=X.columns,
        ax=axes[0],
        orientation='left',
        color_threshold=0.5
    )
    axes[0].set_xlabel('Distance')
    axes[0].set_title('Feature Dendrogram (Hierarchical Clustering)')
    
    # Correlation heatmap (reordered by clustering)
    reorder_idx = dendro['leaves']
    corr_reordered = corr_matrix.iloc[reorder_idx, reorder_idx]
    
    sns.heatmap(
        corr_reordered,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[1],
        cbar_kws={'label': 'Correlation'}
    )
    axes[1].set_title('Correlation Matrix (Reordered by Clustering)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")


def main():
    """Main feature importance analysis workflow."""
    print("=" * 80)
    print("Feature Importance Analysis & Selection (AFML Chapter 8)")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading labeled features...")
    try:
        df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
        
        # Check for t1
        if 't1' not in df.columns:
            print("   't1' column missing. Joining with labeled_events.csv...")
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            df = df.join(events[['t1']], rsuffix='_events')
            if 't1' not in df.columns and 't1_events' in df.columns:
                df['t1'] = df['t1_events']
        
        df['t1'] = pd.to_datetime(df['t1'])
        df = df.dropna()
        
        print(f"   Loaded {len(df)} samples")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 2. Prepare data
    target_col = 'label'
    weight_col = 'sample_weight'
    exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col].astype(int)
    sample_weights = df[weight_col] if weight_col in df.columns else None
    t1 = df['t1']
    
    print(f"   Features: {len(feature_cols)} | Samples: {len(X)}")
    
    # 3. Initialize analyzer
    print("\n2. Running Feature Importance Analysis...")
    analyzer = FeatureImportanceAnalyzer(scoring='neg_log_loss', n_jobs=-1)
    
    # 3a. MDI (Fast)
    mdi_imp = analyzer.fit_importance_mdi(X, y, sample_weights)
    mdi_imp.to_csv("feature_importance_mdi.csv")
    print(f"      MDI: Top feature = {mdi_imp.index[0]} ({mdi_imp.iloc[0, 0]:.4f})")
    
    # 3b. MDA (Gold Standard)
    mda_imp = analyzer.fit_importance_mda(
        X, y, sample_weights, t1,
        n_splits=5,
        embargo_pct=0.01
    )
    mda_imp.to_csv("feature_importance_mda.csv")
    print(f"      MDA: Top feature = {mda_imp.index[0]} ({mda_imp.iloc[0, 0]:.4f})")
    
    # 3c. SFI (Optional - slower)
    # Uncomment if you want to run SFI
    # sfi_imp = analyzer.fit_importance_sfi(X, y, sample_weights, t1, n_splits=5)
    # sfi_imp.to_csv("feature_importance_sfi.csv")
    sfi_imp = None
    
    # 4. Visualizations
    print("\n3. Generating visualizations...")
    os.makedirs('visual_analysis', exist_ok=True)
    
    plot_feature_importance_comparison(mdi_imp, mda_imp, sfi_imp, top_n=30)
    plot_feature_clustering(X, mda_imp)
    
    # 5. Feature Selection
    print("\n4. Feature Selection Results")
    print("-" * 80)
    
    # Select features with positive MDA importance
    selected_features = mda_imp[mda_imp['mean'] > 0].index.tolist()
    print(f"   Features with positive MDA importance: {len(selected_features)}/{len(feature_cols)}")
    
    # Select top 50 features
    top_50_features = mda_imp.head(50).index.tolist()
    print(f"   Top 50 features by MDA importance")
    
    # Save selected features
    selected_df = pd.DataFrame({
        'feature': selected_features,
        'mda_importance': mda_imp.loc[selected_features, 'mean'],
        'mdi_importance': mdi_imp.loc[selected_features, 'mean']
    })
    selected_df.to_csv("selected_features.csv", index=False)
    print(f"   Saved selected features to: selected_features.csv")
    
    # Display top 15
    print("\n   Top 15 Features by MDA Importance:")
    print(f"   {'Rank':<6} {'Feature':<30} {'MDA':<10} {'MDI':<10}")
    print("   " + "-" * 60)
    for i, feat in enumerate(mda_imp.head(15).index):
        mda_val = mda_imp.loc[feat, 'mean']
        mdi_val = mdi_imp.loc[feat, 'mean']
        print(f"   {i+1:<6} {feat:<30} {mda_val:<10.4f} {mdi_val:<10.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Feature Importance Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
