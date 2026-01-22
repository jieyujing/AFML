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
    print("=" * 80)
    print("Training Random Forest with Purged K-Fold CV")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading Data...")
    try:
        df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
        print(f"   Loaded features_labeled.csv: {df.shape}")
        
        # Check for t1 (required for purging)
        if 't1' not in df.columns:
            print("   't1' column missing in features. Joining with labeled_events.csv...")
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            df = df.join(events[['t1']], rsuffix='_events')
            if 't1' not in df.columns and 't1_events' in df.columns:
                df['t1'] = df['t1_events']
        
        df['t1'] = pd.to_datetime(df['t1'])
        
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
    exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events']
    
    # Check for selected features from MDA analysis
    use_selected_features = False
    if os.path.exists("selected_features.csv"):
        print("\n   Found selected_features.csv - using MDA-filtered features")
        selected_df = pd.read_csv("selected_features.csv")
        feature_cols = selected_df['feature'].tolist()
        # Filter to only available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        use_selected_features = True
        print(f"   Using {len(feature_cols)} selected features (MDA > 0)")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        print(f"   Using all {len(feature_cols)} features")
    
    X = df[feature_cols]
    y = df[target_col]
    
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
    # Random Forest parameters suitable for financial data
    rf_params = {
        'n_estimators': 1000,
        'max_depth': 5, # Prevent overfitting
        'criterion': 'entropy', # Information gain
        'class_weight': 'balanced_subsample', # Handle class imbalance per tree
        'bootstrap': True,
        'max_features': 'sqrt',
        'min_samples_leaf': 5, # Regularization
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**rf_params)
    
    n_splits = 5
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
             auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

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
    print(f"   Mean ROC AUC:   {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"   Mean Log Loss:  {np.mean(log_loss_scores):.4f} (+/- {np.std(log_loss_scores):.4f})")
    
    # 5. Save Feature Importance
    print("\n4. Saving Feature Importance...")
    feature_importances['Mean_Importance'] = feature_importances.mean(axis=1)
    feature_importances['Std_Importance'] = feature_importances.std(axis=1)
    
    fi_sorted = feature_importances.sort_values('Mean_Importance', ascending=False)
    fi_sorted.to_csv("feature_importance.csv")
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
