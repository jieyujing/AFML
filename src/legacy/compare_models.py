"""
Compare model performance: All features vs Selected features

This script compares the baseline model (181 features) with the refined model 
(125 MDA-selected features) to quantify the improvement from feature selection.

Reference:
- AFML Chapter 8: Feature Importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cv_setup import PurgedKFold


def train_and_evaluate(X, y, sample_weights, t1, model_name="Model"):
    """Train model and return CV metrics."""
    rf_params = {
        'n_estimators': 1000,
        'max_depth': 5,
        'criterion': 'entropy',
        'class_weight': 'balanced_subsample',
        'bootstrap': True,
        'max_features': 'sqrt',
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**rf_params)
    cv = PurgedKFold(n_splits=5, samples_info_sets=t1, embargo=0.01)
    
    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    log_loss_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]
        
        model.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        try:
            log_loss_scores.append(log_loss(y_test, y_pred_proba, labels=model.classes_))
        except ValueError:
            log_loss_scores.append(np.nan)
    
    return {
        'name': model_name,
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'logloss_mean': np.mean(log_loss_scores),
        'logloss_std': np.std(log_loss_scores),
    }


def main():
    print("=" * 80)
    print("Model Comparison: All Features vs MDA-Selected Features")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
    
    if 't1' not in df.columns:
        events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
        df = df.join(events[['t1']], rsuffix='_events')
        if 't1' not in df.columns and 't1_events' in df.columns:
            df['t1'] = df['t1_events']
    
    df['t1'] = pd.to_datetime(df['t1'])
    df = df.dropna()
    
    # Prepare common data
    target_col = 'label'
    weight_col = 'sample_weight'
    exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events']
    
    y = df[target_col].astype(int)
    sample_weights = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    t1 = df['t1']
    
    # 2. All features
    print("\n2. Training with ALL features (181)...")
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_all = df[all_feature_cols]
    print(f"   Features: {len(all_feature_cols)}")
    
    baseline_results = train_and_evaluate(X_all, y, sample_weights, t1, "Baseline (181 features)")
    
    # 3. Selected features
    print("\n3. Training with SELECTED features (MDA > 0)...")
    selected_df = pd.read_csv("selected_features.csv")
    selected_feature_cols = [c for c in selected_df['feature'].tolist() if c in df.columns]
    X_selected = df[selected_feature_cols]
    print(f"   Features: {len(selected_feature_cols)}")
    
    selected_results = train_and_evaluate(X_selected, y, sample_weights, t1, "Selected (125 features)")
    
    # 4. Display comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (Purged 5-Fold CV)")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'Baseline (181)':<25} {'Selected (125)':<25} {'Delta':<15}")
    print("-" * 85)
    
    metrics = [
        ('Accuracy', 'accuracy_mean', 'accuracy_std'),
        ('F1 Score', 'f1_mean', 'f1_std'),
        ('ROC AUC', 'auc_mean', 'auc_std'),
        ('Log Loss', 'logloss_mean', 'logloss_std'),
    ]
    
    for label, mean_key, std_key in metrics:
        baseline_val = baseline_results[mean_key]
        baseline_std = baseline_results[std_key]
        selected_val = selected_results[mean_key]
        selected_std = selected_results[std_key]
        
        delta = selected_val - baseline_val
        
        # For log loss, lower is better
        if 'loss' in mean_key.lower():
            delta_sign = '+' if delta > 0 else ''  # + means worse for log loss
            improvement = 'worse' if delta > 0 else 'better'
        else:
            delta_sign = '+' if delta > 0 else ''
            improvement = 'better' if delta > 0 else 'worse'
        
        print(f"{label:<20} {baseline_val:.4f} (+/-{baseline_std:.4f})     {selected_val:.4f} (+/-{selected_std:.4f})     {delta_sign}{delta:.4f} ({improvement})")
    
    # Summary
    print("\n" + "-" * 80)
    auc_improvement = (selected_results['auc_mean'] - baseline_results['auc_mean']) / baseline_results['auc_mean'] * 100
    feature_reduction = (181 - 125) / 181 * 100
    
    print(f"\nSUMMARY:")
    print(f"  - Feature Reduction: 181 -> 125 ({feature_reduction:.1f}% fewer features)")
    print(f"  - AUC Improvement:   {baseline_results['auc_mean']:.4f} -> {selected_results['auc_mean']:.4f} ({auc_improvement:+.2f}%)")
    
    if selected_results['auc_mean'] > baseline_results['auc_mean']:
        print(f"\n  SUCCESS! Feature selection IMPROVED model performance.")
    elif abs(selected_results['auc_mean'] - baseline_results['auc_mean']) < 0.01:
        print(f"\n  Feature selection maintained performance with {feature_reduction:.1f}% fewer features.")
    else:
        print(f"\n  NOTE: Performance decreased slightly. Consider adjusting feature selection threshold.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
