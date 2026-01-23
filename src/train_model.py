import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cv_setup import PurgedKFold

def get_feature_importance(model, features):
    """
    Extract feature importance from the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Return as DataFrame
    fi_df = pd.DataFrame({
        'Feature': [features[i] for i in indices],
        'Importance': importances[indices]
    })
    return fi_df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to input features file")
    parser.add_argument("--meta", action="store_true", help="Enable Meta-Labeling mode (map -1 to 0, keep 1)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Training Random Forest with Purged K-Fold CV (Meta Mode: {args.meta})")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading Data...")
    try:
        if args.input:
            input_path = args.input
            print(f"   Using provided input: {input_path}")
        else:
            # Prioritize PCA features -> V2 Features -> Legacy Features
            if os.path.exists(os.path.join("data", "output", "features_pca.csv")):
                 input_path = os.path.join("data", "output", "features_pca.csv")
                 print("   Using PCA Features (features_pca.csv)")
            elif os.path.exists(os.path.join("data", "output", "features_v2_labeled.csv")):
                 input_path = os.path.join("data", "output", "features_v2_labeled.csv")
                 print("   Using V2 Features (features_v2_labeled.csv)")
            else:
                 input_path = os.path.join("data", "output", "features_labeled.csv")
                 print("   Using Legacy Features (features_labeled.csv)")

        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"   Loaded data: {df.shape}")
        
        # Check for t1 (required for purging)
        if 't1' not in df.columns:
            print("   't1' column missing in features. Joining with labeled_events.csv...")
            # Try to find corresponding labeled_events
            # If input was features_v2_labeled_meta.csv, look for labeled_events_meta.csv
            if "_meta" in input_path:
                events_path = os.path.join("data", "output", "labeled_events_meta.csv")
            else:
                events_path = os.path.join("data", "output", "labeled_events.csv")
            
            if os.path.exists(events_path):
                events = pd.read_csv(events_path, index_col=0, parse_dates=True)
                df = df.join(events[['t1']], rsuffix='_events')
                if 't1' not in df.columns and 't1_events' in df.columns:
                    df['t1'] = df['t1_events']
            else:
                 print(f"   Warning: Could not find labeled events file at {events_path}")
        
        if 't1' in df.columns:
            df['t1'] = pd.to_datetime(df['t1'])
        else:
             print("   Error: 't1' column required for Purged K-Fold. Aborting.")
             return

        # Drop rows with NaN
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            print(f"   Dropped {original_len - len(df)} rows with NaNs")
            
        print(f"   Final Dataset: {len(df)} samples")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Prepare X, y, weights
    target_col = 'label'
    weight_col = 'sample_weight'
    exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events', 'holding_period', 'return']
    
    # Feature Selection Logic
    pc_cols = [c for c in df.columns if c.startswith('PC_')]
    if len(pc_cols) > 0:
        print(f"   Using {len(pc_cols)} PCA components")
        feature_cols = pc_cols
    # NOTE: Disabling selected_features.csv logic if explicit input is provided to avoid staleness
    elif args.input is None and os.path.exists(os.path.join("data", "output", "selected_features.csv")):
        print("\n   Found selected_features.csv - using MDA-filtered features")
        selected_df = pd.read_csv(os.path.join("data", "output", "selected_features.csv"))
        feature_cols = selected_df['feature'].tolist()
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"   Using {len(feature_cols)} selected features (MDA > 0)")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        print(f"   Using all {len(feature_cols)} features")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # META LABELING TRANSFORMATION
    if args.meta:
        print("   Applying Meta-Labeling transformation: {-1, 0} -> 0, {1} -> 1")
        y = y.apply(lambda x: 1 if x == 1 else 0)
    
    # Ensure labels are integers for sklearn
    y = y.astype(int)
    
    # Sample Weights (if available)
    if weight_col in df.columns:
        sample_weights = df[weight_col]
        print(f"   Using Sample Weights (Mean: {sample_weights.mean():.4f})")
    else:
        sample_weights = pd.Series(1.0, index=df.index)
        print("   No sample weights found. Using uniform weights.")

    t1 = df['t1']
    
    print(f"   Features: {len(feature_cols)} | Classes: {y.unique()}")
    
    # 3. Setup Model and CV
    # Load optimized hyperparameters if available
    params_path = os.path.join("data", "output", "best_hyperparameters.csv")
    if os.path.exists(params_path):
        print("\n   Loading optimized hyperparameters...")
        params_df = pd.read_csv(params_path)
        # Drop best_auc column if it exists
        if 'best_auc' in params_df.columns:
            params_df = params_df.drop(columns=['best_auc'])
        rf_params = params_df.iloc[0].to_dict()
        # Ensure correct types
        if 'n_estimators' in rf_params: rf_params['n_estimators'] = int(rf_params['n_estimators'])
        if 'max_depth' in rf_params: rf_params['max_depth'] = int(rf_params['max_depth'])
        if 'min_samples_split' in rf_params: rf_params['min_samples_split'] = int(rf_params['min_samples_split'])
        if 'min_samples_leaf' in rf_params: rf_params['min_samples_leaf'] = int(rf_params['min_samples_leaf'])
        
        # Add constant params
        rf_params['n_jobs'] = -1
        rf_params['random_state'] = 42
        rf_params['class_weight'] = 'balanced_subsample'
        rf_params['bootstrap'] = True
        
        print("   Parameters loaded:")
        for k, v in rf_params.items():
            print(f"     {k}: {v}")
    else:
        print("\n   Using default parameters (Optimization not found)")
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
    
    n_splits = args.n_splits
    embargo_pct = 0.01
    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, embargo=embargo_pct)
    
    print(f"\n2. Starting Cross-Validation ({n_splits} folds, Embargo: {embargo_pct*100}%)...")
    
    # Metrics storage
    accuracy_scores = []
    log_loss_scores = []
    f1_scores = []
    auc_scores = []
    feature_importances = pd.DataFrame(index=feature_cols)
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f"\n   Fold {i+1}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]
        
        # Train
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        
        # Log Loss (needs careful handling if classes are missing in test/train)
        try:
            ll = log_loss(y_test, y_pred_proba, labels=model.classes_)
        except ValueError:
            ll = np.nan
            
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC (Handle multiclass or binary)
        if len(np.unique(y)) == 2:
             # Binary
             auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
             # Multiclass
             if y_pred_proba.shape[1] == len(np.unique(y)):
                 auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
             else:
                 auc = np.nan
                 print("    Warning: skipping AUC calc due to class mismatch")

        accuracy_scores.append(acc)
        log_loss_scores.append(ll)
        f1_scores.append(f1)
        auc_scores.append(auc)
        
        # Feature Importance
        fi = model.feature_importances_
        feature_importances[f'Fold_{i+1}'] = fi
        
        print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")

    # 4. Results Summary
    print("\n3. CV Results Summary")
    print("-" * 40)
    print(f"   Mean Accuracy:  {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"   Mean F1 Score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"   Mean ROC AUC:   {np.nanmean(auc_scores):.4f} (+/- {np.nanstd(auc_scores):.4f})")
    print(f"   Mean Log Loss:  {np.mean(log_loss_scores):.4f} (+/- {np.std(log_loss_scores):.4f})")
    
    # 5. Save Feature Importance
    print("\n4. Saving Feature Importance...")
    feature_importances['Mean_Importance'] = feature_importances.mean(axis=1)
    feature_importances['Std_Importance'] = feature_importances.std(axis=1)
    
    fi_sorted = feature_importances.sort_values('Mean_Importance', ascending=False)
    fi_sorted.to_csv(os.path.join("data", "output", "feature_importance.csv"))
    print("   Saved to feature_importance.csv")
    
    # Plot top 20
    plt.figure(figsize=(10, 12))
    sns.barplot(x='Mean_Importance', y=fi_sorted.index[:20], data=fi_sorted.iloc[:20])
    plt.title('Top 20 Features (Mean Impurity Decrease)')
    plt.tight_layout()
    plt.savefig('visual_analysis/feature_importance.png')
    print("   Saved plot to visual_analysis/feature_importance.png")

    print("\n" + "=" * 80)
    print("✓ Model Training Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
