import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))
from cv_setup import PurgedKFold

def main():
    path = "data/output/features_pca.csv"
    df = pd.read_csv(path)
    
    # Check for t1
    if 't1' not in df.columns:
        events = pd.read_csv("data/output/labeled_events.csv")
        # Ensure indices match or join
        # In PCA path, 'date' column might be there or index
        # Let's just use the features_v2_labeled for simplicity
        path = "data/output/features_v2_labeled.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    df['t1'] = pd.to_datetime(df['t1'])
    df = df.dropna()
    
    y = df['label'].astype(int)
    # Map -1 to 0 for binary AUC if needed, or use ovr
    # Sklearn roc_auc_score handles multi-class if configured
    
    exclude = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events', 'return', 'holding_period']
    features = [c for c in df.columns if c not in exclude and 'Unnamed' not in c]
    X = df[features]
    weights = df['sample_weight'] if 'sample_weight' in df.columns else None
    
    cv = PurgedKFold(n_splits=5, samples_info_sets=df['t1'], embargo=0.01)
    
    auc_scores = []
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced_subsample', n_jobs=-1)
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = weights.iloc[train_idx] if weights is not None else None
        
        rf.fit(X_train, y_train, sample_weight=w_train)
        probs = rf.predict_proba(X_test)
        
        # Binary case (-1, 1) -> 0, 1
        # classes_ might be [-1, 1]
        auc = roc_auc_score(y_test, probs[:, 1])
        auc_scores.append(auc)
        
    print(f"Mean Purged CV AUC: {np.mean(auc_scores):.4f}")
    print(f"Standard Deviation: {np.std(auc_scores):.4f}")

if __name__ == '__main__':
    main()
