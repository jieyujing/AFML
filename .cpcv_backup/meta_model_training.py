"""
Meta-Model Training Script — AFML Pipeline
Trains a secondary model on the features to predict if the primary model's
signal is correct (meta-label prediction).
Implements uniqueness-constrained Bagging Classifier, Purged K-Fold CV,
and generates comprehensive valuation visualizations.
"""

from __future__ import annotations

import joblib
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    accuracy_score,
    log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve

from afmlkit.validation.purged_cv import PurgedKFold, CombinatorialPurgedKFold
from afmlkit.importance.clustering import cluster_features, get_feature_distance_matrix, hierarchical_clustering
from afmlkit.importance.mda import clustered_mda
from scipy.cluster.hierarchy import dendrogram


# Configuration
FEATURES_PATH = Path("outputs/dollar_bars/feature_matrix.csv")
LABELS_PATH = Path("outputs/dollar_bars/cusum_sampled_bars.csv")
OUTPUT_DIR = Path("outputs/models/meta_model")

META_COLS = [
    "open", "high", "low", "close", "volume", "vwap",
    "trades", "median_trade_size", "log_return",
    "bin", "t1", "avg_uniqueness", "return_attribution",
]

CV_N_SPLITS = 5
CV_EMBARGO_PCT = 0.01
N_ESTIMATORS = 1000
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load features and merge with full labeling info."""
    df_feat = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    df_lab = pd.read_csv(LABELS_PATH, parse_dates=["timestamp"])
    
    # Merge label columns that might be missing from features
    common_cols = ['timestamp']
    df = pd.merge(df_feat, df_lab[['timestamp', 'trend_weighted_uniqueness']], on='timestamp', how='left')
    
    # Use timestamp as index
    df = df.set_index('timestamp')
    df = df.dropna(subset=['bin', 't1'])
    df['t1'] = pd.to_datetime(df['t1'])
    
    # Determine sample weight
    if 'trend_weighted_uniqueness' in df.columns and df['trend_weighted_uniqueness'].notnull().any():
        print("[Data] Using 'trend_weighted_uniqueness' as sample weights.")
        sample_weight = df['trend_weighted_uniqueness'].fillna(df['avg_uniqueness']).clip(lower=0)
    else:
        print("[Data] Fallback: Using 'avg_uniqueness' as sample weights.")
        sample_weight = df['avg_uniqueness'].fillna(1.0).clip(lower=0)
        
    y = df['bin'].astype(int)
    t1 = df['t1']
    
    # Feature columns
    exclude_cols = META_COLS + ['trend_weighted_uniqueness', 'trend_confidence', 'vertical_touch_weights', 'event_idx', 'touch_idx', 'ret']
    
    # Drop noisy features identified from Clustered MDA (importance <= 0)
    noisy_features = [
        'vol_parkinson', 'liq_amihud',  # Cluster 4
        'trend_variance_ratio_20',      # Cluster 6
        'vol_atr_14', 'ema_short', 'ema_long', 'ffd_log_price'  # Cluster 2
    ]
    
    exclude_cols.extend(noisy_features)
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # handle NaN (NO bfill allowed to prevent future peeking!)
    nan_cols = X.columns[X.isna().all()]
    if len(nan_cols) > 0:
        X = X.drop(columns=nan_cols)
        
    X = X.ffill()
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    t1 = t1.loc[valid_idx]
    sample_weight = sample_weight.loc[valid_idx]
    
    avg_u = df['avg_uniqueness'].mean() if 'avg_uniqueness' in df.columns else 0.5
    
    print(f"[Data] Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"[Data] Label distribution: {y.value_counts().to_dict()}")
    print(f"[Data] Mean base uniqueness: {avg_u:.4f}")
    
    return X, y, t1, sample_weight, avg_u

class WeightedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    True AFML Bagging: Bootstrap sampling occurs with probabilities 
    proportional to sample_weight (Uniqueness), rather than just passing 
    weights to the base estimator splits.
    """
    def __init__(self, base_estimator, n_estimators=1000, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y, sample_weight=None):
        from sklearn.base import clone
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.classes_ = np.unique(y)
        
        N = len(X)
        n_samples = max(1, int(N * self.max_samples))
        
        if sample_weight is None:
            prob = np.ones(N) / N
        else:
            w = np.array(sample_weight)
            # safeguard against exact zeros or negatives
            w = np.clip(w, 1e-8, None)
            prob = w / np.sum(w)
            
        for _ in range(self.n_estimators):
            # Weighted Bootstrap
            idx = rng.choice(N, size=n_samples, replace=True, p=prob)
            X_b = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
            y_b = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx]
            
            clf = clone(self.base_estimator)
            # Fits on uniformly weighted bootstrap samples (since frequency already reflects weight)
            clf.fit(X_b, y_b)
            self.estimators_.append(clf)
        
        return self

    def predict_proba(self, X):
        probas = np.mean([clf.predict_proba(X) for clf in self.estimators_], axis=0)
        return probas
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def build_model(mean_uniqueness: float):
    """Build Uniqueness-constrained Bagging Classifier."""
    max_samples = np.clip(mean_uniqueness, 0.1, 1.0)
    print(f"[Model] Using max_samples = {max_samples:.4f} based on mean uniqueness")
    
    base_m = DecisionTreeClassifier(
        criterion='entropy',
        max_features='sqrt',  # Changed to sqrt to reduce variance of posterior probabilities
        class_weight='balanced',
        min_weight_fraction_leaf=0.05,
        random_state=RANDOM_STATE
    )
    
    model = WeightedBaggingClassifier(
        base_estimator=base_m,
        n_estimators=N_ESTIMATORS,
        max_samples=max_samples,
        random_state=RANDOM_STATE
    )
    
    return model


def plot_cv_variance(metrics: dict, output_dir: Path):
    """Plot Purged CV Fold Variance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = np.arange(1, len(metrics['roc_auc']) + 1)
    
    ax.plot(folds, metrics['roc_auc'], marker='o', label='ROC AUC', color='#2980b9')
    ax.plot(folds, metrics['accuracy'], marker='s', label='Accuracy', color='#2ecc71')
    ax.plot(folds, metrics['f1'], marker='^', label='F1 Score', color='#8e44ad')
    
    ax.axhline(np.mean(metrics['roc_auc']), color='#2980b9', linestyle='--', alpha=0.5)
    ax.axhline(np.mean(metrics['accuracy']), color='#2ecc71', linestyle='--', alpha=0.5)
    ax.axhline(np.mean(metrics['f1']), color='#8e44ad', linestyle='--', alpha=0.5)
    
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Purged K-Fold Cross Validation Variance")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    fig.savefig(output_dir / "purged_cv_scores.png", dpi=150)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, output_dir: Path):
    """Plot total confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Meta-Model Confusion Matrix (CV Aggregated)')
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['False Signal (0)', 'True Signal (1)'])
    ax.set_yticklabels(['False Signal (0)', 'True Signal (1)'])
    
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

def plot_roc_pr_curves(y_true, y_prob, output_dir: Path):
    """Plot ROC and P-R curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color='#e74c3c', label=f'AUC = {auc:.4f}')
    ax1.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(rec, prec, color='#2ecc71')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration_curve(y_true, y_prob, output_dir: Path):
    """Plot reliability curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="#3498db", label="Meta-Model")
    
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Probability Calibration Curve (Reliability)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    brier = brier_score_loss(y_true, y_prob)
    ax.text(0.05, 0.9, f"Brier Score: {brier:.4f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
    fig.tight_layout()
    fig.savefig(output_dir / "probability_calibration.png", dpi=150)
    plt.close(fig)


def plot_bet_size_dist(y_prob, output_dir: Path):
    """Plot bet size distribution (derived from probability)."""
    # Simple continuous Kelly size derived from probability
    # bet_size = 2 * p - 1  (bounded between 0 and 1)
    bet_sizes = np.clip(2 * y_prob - 1, 0, 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(bet_sizes, bins=20, kde=True, color="#f1c40f", ax=ax)
    
    ax.set_xlabel("Bet Size / Allocation")
    ax.set_ylabel("Frequency")
    ax.set_title("Bet Size Distribution")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(output_dir / "bet_size_distribution.png", dpi=150)
    plt.close(fig)

def plot_dendrogram(X: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save a dendrogram of the feature clustering."""
    dist = get_feature_distance_matrix(X)
    link = hierarchical_clustering(dist, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        link,
        labels=X.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title("Feature Clustering Dendrogram (Ward Linkage)", fontsize=14)
    ax.set_ylabel("Distance", fontsize=12)
    fig.tight_layout()

    fpath = output_dir / "feature_dendrogram.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

def plot_distance_heatmap(X: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save a correlation-distance heatmap."""
    dist = get_feature_distance_matrix(X)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(dist.columns)))
    ax.set_yticks(range(len(dist.index)))
    ax.set_xticklabels(dist.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(dist.index, fontsize=7)
    ax.set_title("Feature Distance Matrix  D = \u221a(0.5 \u00d7 (1 \u2212 \u03c1))", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    fpath = output_dir / "feature_distance_heatmap.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

def run_cpcv(X, y, t1, sample_weight, model, output_dir):
    """Run Combinatorial Purged K-Fold CV and return Out-Of-Fold predictions mapping to independent paths."""
    print(f"\n[CV] Running Combinatorial Purged CV...")
    
    # Standard settings: N=6 groups, k=2 test groups -> 15 splits, 5 independent paths
    N_GROUPS = 6
    N_TEST_GROUPS = 2
    cv = CombinatorialPurgedKFold(
        n_groups=N_GROUPS,
        n_test_groups=N_TEST_GROUPS,
        t1=t1,
        embargo_pct=CV_EMBARGO_PCT
    )
    
    paths = cv.get_paths()
    n_paths = len(paths)
    
    # Store test predictions for each split
    split_preds = {} 
    
    X_arr = X.values
    y_arr = y.values
    w_arr = sample_weight.values
    
    metrics = {'roc_auc': [], 'accuracy': [], 'f1': []}
    
    from sklearn.base import clone
    
    split_idx = 0
    for train_idx, test_idx in cv.split(X):
        print(f"  --> CPCV Split {split_idx+1}/{cv.get_n_splits()} Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        X_train, y_train, w_train = X_arr[train_idx], y_arr[train_idx], w_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]
        
        clf = clone(model)
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        
        split_preds[split_idx] = (test_idx, probs, preds)
        
        # Metrics over this split
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
            metrics['roc_auc'].append(auc)
        metrics['accuracy'].append(accuracy_score(y_test, preds))
        metrics['f1'].append(f1_score(y_test, preds, zero_division=0))
        
        split_idx += 1
        
    print("\n[CV] Metrics Summary Across Splits:")
    for k, v in metrics.items():
        arr = np.array(v)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            print(f"  {k}: {np.mean(arr):.4f} \u00b1 {np.std(arr):.4f}")
            
    # Reconstruct OOF DataFrame for each path
    # A path is a complete out-of-sample prediction series
    oof_df = pd.DataFrame(index=X.index)
    
    bounds = cv._get_fold_bounds(len(y))
    
    for p, path_dict in paths.items():
        probs_p = np.full(len(y), np.nan)
        
        for group_idx, s_idx in path_dict.items():
            s_test_idx, s_probs, _ = split_preds[s_idx]
            
            # Find elements in s_test_idx that belong to group_idx
            start, end = bounds[group_idx]
            g_indices = np.arange(start, end)
            
            mask = np.isin(s_test_idx, g_indices)
            actual_sub_idx = s_test_idx[mask]
            actual_probs = s_probs[mask]
            
            probs_p[actual_sub_idx] = actual_probs
            
        oof_df[f'path_{p}'] = probs_p
        
    oof_df['true'] = y_arr
    # Drop rows where all paths are NaN (can happen due to purging overlaps)
    oof_df = oof_df.dropna(subset=[f'path_{p}' for p in range(n_paths)], how='all')
    
    return oof_df, metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    X, y, t1, sample_weight, mean_u = load_and_prepare_data()
    
    # Build Model
    model = build_model(mean_u)
    
    # 1. CPCV & Generation of Paths
    oof_df, metrics = run_cpcv(X, y, t1, sample_weight, model, OUTPUT_DIR)
    
    oof_path = OUTPUT_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_path)
    print(f"\n[Model] Saved CPCV Multipath Out-Of-Fold predictions to {oof_path} (Safe for backtesting)")
    
    # Train the final model on all data for serialization (for live trading only!)
    print("\n[Model] Training final model on entire dataset (DO NOT use for historical backtesting)...")
    model.fit(X.values, y.values, sample_weight=sample_weight.values)
    model_path = OUTPUT_DIR / "meta_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved final model to {model_path}")
    
    # Flatten CPCV paths for total ROC/PR analysis
    path_cols = [c for c in oof_df.columns if c.startswith('path_')]
    y_true_flat = []
    y_prob_flat = []
    
    for p_col in path_cols:
        valid_mask = oof_df[p_col].notna()
        y_true_flat.extend(oof_df.loc[valid_mask, 'true'])
        y_prob_flat.extend(oof_df.loc[valid_mask, p_col])
        
    y_true_flat = np.array(y_true_flat)
    y_prob_flat = np.array(y_prob_flat)
    y_pred_flat = (y_prob_flat > 0.5).astype(int)
    
    # 2. Visualizations
    print("\n[Vis] Generating Visualizations...")
    plot_cv_variance(metrics, OUTPUT_DIR)
    plot_confusion_matrix(y_true_flat, y_pred_flat, OUTPUT_DIR)
    plot_roc_pr_curves(y_true_flat, y_prob_flat, OUTPUT_DIR)
    plot_calibration_curve(y_true_flat, y_prob_flat, OUTPUT_DIR)
    plot_bet_size_dist(y_prob_flat, OUTPUT_DIR)
    plot_dendrogram(X, OUTPUT_DIR)
    plot_distance_heatmap(X, OUTPUT_DIR)
    
    # 3. Clustered MDA
    print("\n[MDA] Running Clustered MDA feature importance...")
    
    clusters = cluster_features(X, method="ward")
    print(f"  Found {len(clusters)} feature clusters.")
    
    mda_results = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        classifier=model,
        sample_weight=sample_weight,
        n_splits=CV_N_SPLITS,
        embargo_pct=CV_EMBARGO_PCT,
        n_repeats=1,
        random_state=RANDOM_STATE
    )
    
    mda_path = OUTPUT_DIR / "clustered_mda_results.csv"
    mda_results.to_csv(mda_path, index=False)
    print(f"[MDA] Saved MDA results to {mda_path}")
    
    # Generate MDA plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mda_sorted = mda_results.sort_values("mean_importance", ascending=True)
    
    labels = [f"C{int(r.cluster_id)}: {', '.join(r.features[:2])}..." for _, r in mda_sorted.iterrows()]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in mda_sorted["mean_importance"]]
    
    ax.barh(np.arange(len(labels)), mda_sorted["mean_importance"], xerr=mda_sorted["std_importance"], color=colors)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Information Loss (Log-Loss Increase)")
    ax.set_title("Clustered MDA Feature Importance (Meta-Model)")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clustered_mda_importance.png", dpi=150)
    plt.close(fig)
    
    print("\nAll tasks completed successfully!")
    print(f"Outputs generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
