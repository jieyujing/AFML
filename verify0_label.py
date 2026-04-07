import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from strategies.AL9999.config import FEATURES_DIR

# ============================================================
# 验证 0: 方向与标签语义核查
# ============================================================

# 1. label 语义
labels = pd.read_parquet(FEATURES_DIR + '/meta_labels.parquet')
y_col = 'bin'  # label 列名
print('=== label 语义 ===')
print('bin=1 = DMA与Trend一致 = 信号"正确"')
print('bin=0 = DMA与Trend不一致 = 信号"错误"')
print(f'正类(bin=1)比例: {labels[y_col].mean():.3f}')
print(f'样本数: {len(labels)}')
print()

# 2. 加载特征
features = pd.read_parquet(FEATURES_DIR + '/events_features.parquet')
tsfresh = pd.read_parquet(FEATURES_DIR + '/tsfresh_features.parquet')
tsfresh = tsfresh.drop(columns=['event_idx', 'timestamp'], errors='ignore')
common_idx = features.index.intersection(tsfresh.index)
features = features.loc[common_idx].join(tsfresh.loc[common_idx], rsuffix='_tsfresh')

rf = pd.read_parquet(FEATURES_DIR + '/rf_primary_signals.parquet')
tbm = pd.read_parquet(FEATURES_DIR + '/tbm_results.parquet')
common = features.index.intersection(labels.index).intersection(tbm.index)
X = features.loc[common]
y = labels.loc[common, y_col]
sw = labels.loc[common, 'sample_weight'].fillna(labels.loc[common, 'sample_weight'].mean())

feat_cols = [c for c in X.columns if c.startswith('feat_')]
X = X[feat_cols].fillna(0)
merged = X.copy()
merged['rf_prob'] = rf.reindex(merged.index)['y_prob'].fillna(0.5)
X_all = merged

holdout_start = X_all.index.max() - pd.DateOffset(months=6)
train_mask = X_all.index < holdout_start
X_tr = X_all[train_mask]
y_tr = y[train_mask]
sw_tr = sw[train_mask]
X_ho = X_all[~train_mask]
y_ho = y[~train_mask]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from afmlkit.validation.purged_cv import PurgedKFold

t1 = pd.to_datetime(tbm.loc[X_tr.index, 'exit_ts'], errors='coerce')
t1_tr = pd.Series(t1.values, index=X_tr.index)
cv = PurgedKFold(n_splits=5, t1=t1_tr, embargo_pct=0.05)

print('=== OOF 预测 ===')
oof_p1 = np.full(len(y_tr), np.nan)
oof_p0 = np.full(len(y_tr), np.nan)
for tr, te in cv.split(X_tr):
    base = DecisionTreeClassifier(criterion='entropy', max_features=1, max_depth=5, class_weight='balanced', random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=500, max_samples=0.5, max_features=1.0, n_jobs=-1, random_state=42)
    model.fit(X_tr.iloc[tr], y_tr.iloc[tr], sample_weight=sw_tr.iloc[tr])
    oof_p1[te] = model.predict_proba(X_tr.iloc[te])[:, 1]
    oof_p0[te] = model.predict_proba(X_tr.iloc[te])[:, 0]

# AUC 检查
auc_p1 = roc_auc_score(y_tr, oof_p1)
auc_p0 = roc_auc_score(y_tr, oof_p0)
auc_flip = roc_auc_score(y_tr, 1 - oof_p1)
print(f'OOF AUC (y=1, prob_col=1):    {auc_p1:.4f}')
print(f'OOF AUC (y=1, prob_col=0):    {auc_p0:.4f}')
print(f'OOF AUC (y=1, flip=1-prob_col=1): {auc_flip:.4f}')
print()

# ============================================================
# 核心分桶分析：higher prob(y=1) 对应 higher 正类率？
# ============================================================
print('=== 分桶分析 ===')
n_buckets = 10
oof_df = pd.DataFrame({'prob': oof_p1, 'y_true': y_tr.values})
oof_df['bucket'] = pd.qcut(oof_df['prob'], q=n_buckets, labels=False, duplicates='drop')
bucket_stats = oof_df.groupby('bucket').agg(
    mean_prob=('prob', 'mean'),
    positive_rate=('y_true', 'mean'),
    count=('y_true', 'count')
)
print(bucket_stats.to_string())
print()

# prob_flip 分桶
print('=== prob_flip 分桶 ===')
oof_df['prob_flip'] = 1 - oof_df['prob']
oof_df['bucket_flip'] = pd.qcut(oof_df['prob_flip'], q=n_buckets, labels=False, duplicates='drop')
bucket_stats_f = oof_df.groupby('bucket_flip').agg(
    mean_prob_flip=('prob_flip', 'mean'),
    positive_rate=('y_true', 'mean'),
    count=('y_true', 'count')
)
print(bucket_stats_f.to_string())
print()

# 逐 fold AUC
print('=== per-fold AUC ===')
for fold_i, (tr, te) in enumerate(cv.split(X_tr)):
    y_te = y_tr.iloc[te].values
    p_te = oof_p1[te]
    auc_f = roc_auc_score(y_te, p_te)
    auc_ff = roc_auc_score(y_te, 1 - p_te)
    pos_rate = y_te.mean()
    print(f'  Fold {fold_i}: AUC={auc_f:.4f} AUC(flip)={auc_ff:.4f} pos_rate={pos_rate:.3f} n={len(y_te)}')

print()
print('=== 核心问题诊断 ===')
# prob >= 0.5 时正类率
for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
    mask = oof_p1 >= th
    if mask.sum() > 0:
        pos_rate = y_tr[mask].mean()
        print(f'  prob >= {th}: n={mask.sum()}, 正类率={pos_rate:.3f}')

print()
print('=== 结论 ===')
if auc_p1 > 0.5:
    print('prob_col=1 是正确的正类概率')
    print('AUC > 0.5，模型有正向排序能力')
elif auc_flip > auc_p1:
    print('prob_col=1 方向可能反了')
    print('AUC(flip) > AUC，说明 prob(y=1) 越高真实正类率越低')
else:
    print(f'AUC={auc_p1:.4f}，排序能力弱')
