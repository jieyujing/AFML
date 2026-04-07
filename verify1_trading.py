"""
验证 1: Score 分桶与交易层指标

分层验证：
A. 分桶测试: top/bottom bucket 的真实 outcome 差异
B. 作为 meta filter: 接到 primary 策略上
"""
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from strategies.AL9999.config import FEATURES_DIR

# ============================================================
# 数据加载
# ============================================================
labels = pd.read_parquet(FEATURES_DIR + '/meta_labels.parquet')
features = pd.read_parquet(FEATURES_DIR + '/events_features.parquet')
tsfresh = pd.read_parquet(FEATURES_DIR + '/tsfresh_features.parquet')
tsfresh = tsfresh.drop(columns=['event_idx', 'timestamp'], errors='ignore')
common_idx = features.index.intersection(tsfresh.index)
features = features.loc[common_idx].join(tsfresh.loc[common_idx], rsuffix='_tsfresh')

rf = pd.read_parquet(FEATURES_DIR + '/rf_primary_signals.parquet')
tbm = pd.read_parquet(FEATURES_DIR + '/tbm_results.parquet')
common = features.index.intersection(labels.index).intersection(tbm.index)
X = features.loc[common]
y = labels.loc[common, 'bin']
sw = labels.loc[common, 'sample_weight'].fillna(labels.loc[common, 'sample_weight'].mean())

# TBM outcome 列
tbm_aligned = tbm.loc[common]
# 找 return 列
ret_col = None
for col in ['ret', 'return', 'pnl', 'result']:
    if col in tbm_aligned.columns:
        ret_col = col
        break
if ret_col is None:
    ret_col = 'fracdiff'

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

# ============================================================
# 构建 OOF 预测
# ============================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from afmlkit.validation.purged_cv import PurgedKFold

t1 = pd.to_datetime(tbm.loc[X_tr.index, 'exit_ts'], errors='coerce')
t1_tr = pd.Series(t1.values, index=X_tr.index)
cv = PurgedKFold(n_splits=5, t1=t1_tr, embargo_pct=0.05)

oof = np.full(len(y_tr), np.nan)
for tr, te in cv.split(X_tr):
    base = DecisionTreeClassifier(criterion='entropy', max_features=1, max_depth=5, class_weight='balanced', random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=500, max_samples=0.5, max_features=1.0, n_jobs=-1, random_state=42)
    model.fit(X_tr.iloc[tr], y_tr.iloc[tr], sample_weight=sw_tr.iloc[tr])
    oof[te] = model.predict_proba(X_tr.iloc[te])[:, 1]

auc = roc_auc_score(y_tr, oof)
print(f'OOF AUC: {auc:.4f}')

# TBM return
tbm_ret_all = pd.Series(tbm_aligned[ret_col].values, index=tbm_aligned.index)
ret_oof = tbm_ret_all[train_mask].values

# ============================================================
# A: 分桶 outcome 差异
# ============================================================
print()
print('=== A: 5-分桶 outcome 差异 ===')
bucket_labels = pd.qcut(pd.Series(oof), q=5, labels=False, duplicates='drop').values

bucket_stats = []
for b in range(5):
    mask = bucket_labels == b
    n = int(np.sum(mask))
    if n > 0:
        pos_rate = float(np.nanmean(y_tr.values[mask]))
        mean_ret = float(np.nanmean(ret_oof[mask]))
        median_ret = float(np.nanmedian(ret_oof[mask]))
        bucket_stats.append({
            'bucket': b,
            'n': n,
            'mean_prob': float(np.nanmean(oof[mask])),
            'pos_rate': pos_rate,
            'mean_ret': mean_ret,
            'median_ret': median_ret,
        })

bucket_df = pd.DataFrame(bucket_stats)
print(bucket_df.to_string(index=False))

# ============================================================
# B: Hit rate per bucket
# ============================================================
print()
print('=== B: Hit rate per bucket ===')
bucket_stats2 = []
for b in range(5):
    mask = bucket_labels == b
    n = int(np.sum(mask))
    if n > 0:
        hit_rate = float(np.nanmean((ret_oof[mask] > 0).astype(float)))
        bucket_stats2.append({
            'bucket': b,
            'n': n,
            'mean_prob': float(np.nanmean(oof[mask])),
            'hit_rate': hit_rate,
            'mean_ret': float(np.nanmean(ret_oof[mask])),
        })

bucket_df2 = pd.DataFrame(bucket_stats2)
print(bucket_df2.to_string(index=False))

# ============================================================
# C: Top/Bottom decile spread
# ============================================================
print()
print('=== C: Top/Bottom decile 分析 ===')
p10 = float(np.percentile(oof, 10))
p20 = float(np.percentile(oof, 20))
p80 = float(np.percentile(oof, 80))
p90 = float(np.percentile(oof, 90))

for name, lo, hi in [
    ('Top10%',  p90, 1.0),
    ('Top20%',  p80, 1.0),
    ('Bottom20%', 0.0, p20),
    ('Bottom10%', 0.0, p10),
    ('Middle60%', p20, p80),
]:
    mask = (oof >= lo) & (oof <= hi)
    n = int(np.sum(mask))
    if n > 0:
        pos_rate = float(np.nanmean(y_tr.values[mask]))
        hit_rate = float(np.nanmean((ret_oof[mask] > 0).astype(float)))
        mean_ret = float(np.nanmean(ret_oof[mask]))
        print(f'  {name}: n={n}, pos_rate={pos_rate:.3f}, hit_rate={hit_rate:.3f}, mean_ret={mean_ret:.4f}')

# ============================================================
# D: Meta filter 效果对比
# ============================================================
print()
print('=== D: Meta Filter 效果对比 ===')
n_all = len(y_tr)
all_hit = float(np.nanmean((ret_oof > 0).astype(float)))
all_ret = float(np.nanmean(ret_oof))
print(f'  Baseline (no filter): n={n_all}, hit_rate={all_hit:.3f}, mean_ret={all_ret:.4f}')

for th in [0.50, 0.55]:
    mask = oof >= th
    n = int(np.sum(mask))
    if n > 0 and n < n_all:
        hit_rate = float(np.nanmean((ret_oof[mask] > 0).astype(float)))
        ret = float(np.nanmean(ret_oof[mask]))
        delta_hit = hit_rate - all_hit
        delta_ret = ret - all_ret
        pct = n / n_all * 100
        print(f'  thresh={th}: n={n}/{n_all} ({pct:.0f}%), hit_rate={hit_rate:.3f} (delta={delta_hit:+.3f}), mean_ret={ret:.4f} (delta={delta_ret:+.4f})')

# ============================================================
# E: Top vs Bottom spread 统计检验
# ============================================================
print()
print('=== E: Top20 vs Bottom20 spread ===')
top_mask = oof >= p80
bot_mask = oof <= p20
top_n = int(np.sum(top_mask))
bot_n = int(np.sum(bot_mask))
if top_n > 0 and bot_n > 0:
    top_hit = float(np.nanmean((ret_oof[top_mask] > 0).astype(float)))
    bot_hit = float(np.nanmean((ret_oof[bot_mask] > 0).astype(float)))
    top_ret = float(np.nanmean(ret_oof[top_mask]))
    bot_ret = float(np.nanmean(ret_oof[bot_mask]))
    print(f'  Top20:  n={top_n}, hit_rate={top_hit:.3f}, mean_ret={top_ret:.4f}')
    print(f'  Bot20:  n={bot_n}, hit_rate={bot_hit:.3f}, mean_ret={bot_ret:.4f}')
    print(f'  Spread: hit_rate={top_hit-bot_hit:+.3f}, mean_ret={top_ret-bot_ret:+.4f}')
