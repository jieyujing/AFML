import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from strategies.AL9999.config import FEATURES_DIR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from afmlkit.validation.purged_cv import PurgedKFold

features = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))
tsfresh_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'tsfresh_features.parquet'))
tsfresh_df = tsfresh_df.drop(columns=['event_idx', 'timestamp'], errors='ignore')
common_idx = features.index.intersection(tsfresh_df.index)
features = features.loc[common_idx].join(tsfresh_df.loc[common_idx], rsuffix='_tsfresh')

labels = pd.read_parquet(os.path.join(FEATURES_DIR, 'meta_labels.parquet'))
tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))
rf = pd.read_parquet(os.path.join(FEATURES_DIR, 'rf_primary_signals.parquet'))
common_idx = features.index.intersection(labels.index).intersection(tbm.index)
X = features.loc[common_idx]
y = labels.loc[common_idx, 'bin']
sw = labels.loc[common_idx, 'sample_weight'].fillna(labels.loc[common_idx, 'sample_weight'].mean())

feat_cols = [c for c in X.columns if c.startswith('feat_')]
X = X[feat_cols].fillna(0)
merged = X.copy()
merged['rf_prob'] = rf.reindex(merged.index)['y_prob'].astype(float).fillna(0.5)
X_all = merged

holdout_start = X_all.index.max() - pd.DateOffset(months=6)
train_mask = X_all.index < holdout_start
X_tr = X_all[train_mask]
y_tr = y[train_mask]
sw_tr = sw[train_mask]
X_ho = X_all[~train_mask]
y_ho = y[~train_mask]

t1 = pd.to_datetime(tbm.loc[X_tr.index, 'exit_ts'], errors='coerce')
t1_tr = pd.Series(t1.values, index=X_tr.index)
cv = PurgedKFold(n_splits=5, t1=t1_tr, embargo_pct=0.05)

time_cols = [c for c in X_all.columns if any(x in c for x in ['sin_', 'cos_', 'dow', 'sess', 'night'])]
time_feats = [c for c in time_cols if c in X_all.columns]

print('=== balanced vs non-balanced class weights ===')
configs = [
    ('Full_bal',   list(X_all.columns),      True),
    ('Full_nobal', list(X_all.columns),      False),
    ('C12_bal',    time_feats + ['rf_prob'], True),
    ('C12_nobal',  time_feats + ['rf_prob'], False),
]
for name, feat_set, use_bal in configs:
    Xs_tr = X_tr[feat_set]
    Xs_ho = X_ho[feat_set]
    oof = np.full(len(y_tr), np.nan)
    for tr, te in cv.split(Xs_tr):
        base = DecisionTreeClassifier(
            criterion='entropy', max_features=1, max_depth=5,
            class_weight='balanced' if use_bal else None, random_state=42
        )
        model = BaggingClassifier(
            estimator=base, n_estimators=500, max_samples=0.5,
            max_features=1.0, n_jobs=-1, random_state=42
        )
        model.fit(Xs_tr.iloc[tr], y_tr.iloc[tr])
        oof[te] = model.predict_proba(Xs_tr.iloc[te])[:, 1]

    base = DecisionTreeClassifier(
        criterion='entropy', max_features=1, max_depth=5,
        class_weight='balanced' if use_bal else None, random_state=42
    )
    model = BaggingClassifier(
        estimator=base, n_estimators=500, max_samples=0.5,
        max_features=1.0, n_jobs=-1, random_state=42
    )
    model.fit(Xs_tr, y_tr)
    ho_prob = model.predict_proba(Xs_ho)[:, 1]

    auc = roc_auc_score(y_ho, ho_prob)
    auc_f = roc_auc_score(y_ho, 1 - ho_prob)
    print(f'{name}: AUC={auc:.4f} AUC(flip)={auc_f:.4f} mean={ho_prob.mean():.4f} std={ho_prob.std():.4f} n={len(feat_set)}')

print()
print('=== C12 单变量 AUC (OOF) ===')
for col in time_feats:
    if col == 'rf_prob':
        continue
    try:
        auc = roc_auc_score(y_tr, X_tr[col])
        print(f'  {col}: AUC_tr={auc:.4f}')
    except Exception as e:
        print(f'  {col}: ERROR {e}')

print()
print('=== HO prob distribution per class ===')
for name, feat_set, use_bal in [('C12_bal', time_feats + ['rf_prob'], True), ('C12_nobal', time_feats + ['rf_prob'], False)]:
    Xs_tr = X_tr[feat_set]
    Xs_ho = X_ho[feat_set]
    base = DecisionTreeClassifier(criterion='entropy', max_features=1, max_depth=5, class_weight='balanced' if use_bal else None, random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=500, max_samples=0.5, max_features=1.0, n_jobs=-1, random_state=42)
    model.fit(Xs_tr, y_tr)
    ho_prob = model.predict_proba(Xs_ho)[:, 1]
    for cls in [0, 1]:
        mask = y_ho == cls
        if mask.sum() > 0:
            print(f'  {name} y={cls}: n={mask.sum()} mean_prob={ho_prob[mask].mean():.4f} std={ho_prob[mask].std():.4f}')
