"""
验证 4: Meta Filter 容量与收益曲线

三条曲线:
1. Coverage vs Hit Rate
2. Coverage vs Mean Return
3. Coverage vs Total Return / Opportunity Count

+ 时间切片分析
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

tbm_aligned = tbm.loc[common]
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from afmlkit.validation.purged_cv import PurgedKFold

t1_series = pd.to_datetime(tbm.loc[X_tr.index, 'exit_ts'], errors='coerce')
t1_tr = pd.Series(t1_series.values, index=X_tr.index)
cv = PurgedKFold(n_splits=5, t1=t1_tr, embargo_pct=0.05)

oof = np.full(len(y_tr), np.nan)
for tr, te in cv.split(X_tr):
    base = DecisionTreeClassifier(criterion='entropy', max_features=1, max_depth=5, class_weight='balanced', random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=500, max_samples=0.5, max_features=1.0, n_jobs=-1, random_state=42)
    model.fit(X_tr.iloc[tr], y_tr.iloc[tr], sample_weight=sw_tr.iloc[tr])
    oof[te] = model.predict_proba(X_tr.iloc[te])[:, 1]

# TBM return
tbm_ret_all = pd.Series(tbm_aligned[ret_col].values, index=tbm_aligned.index)
ret_oof = tbm_ret_all[train_mask].values

# ============================================================
# 构建 OOF DataFrame
# ============================================================
df = pd.DataFrame({
    'timestamp': X_tr.index,
    'prob': oof,
    'y': y_tr.values,
    'ret': ret_oof,
    'hit': (ret_oof > 0).astype(float),
})
df['year_month'] = df['timestamp'].dt.to_period('M')

# ============================================================
# 曲线 1+2+3: Coverage vs Hit Rate / Mean Ret / Total Ret
# ============================================================
print('=== Coverage vs 指标曲线 ===')
print(f'{"保留比例":>10} {"n":>6} {"hit_rate":>10} {"delta_hit":>10} {"mean_ret":>12} {"delta_ret":>12} {"total_ret":>12}')
print('-' * 90)

all_hit = df['hit'].mean()
all_ret = df['ret'].mean()
total_ret_all = df['ret'].sum()

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0]
prev_hit = None
prev_ret = None
for th in thresholds:
    mask = df['prob'] >= th
    n = int(np.sum(mask))
    if n == 0:
        continue
    pct = n / len(df) * 100
    hit = float(df.loc[mask, 'hit'].mean())
    mean_ret = float(df.loc[mask, 'ret'].mean())
    total_ret = float(df.loc[mask, 'ret'].sum())
    delta_hit = hit - all_hit
    delta_ret = mean_ret - all_ret
    print(f'{pct:>9.1f}% {n:>6} {hit:>10.3f} {delta_hit:>+10.3f} {mean_ret:>12.6f} {delta_ret:>+12.6f} {total_ret:>12.6f}')

print()
print(f'Baseline: n={len(df)}, hit_rate={all_hit:.3f}, mean_ret={all_ret:.6f}, total_ret={total_ret_all:.4f}')
print()

# ============================================================
# 曲线 3: Total Return Contribution
# ============================================================
print('=== Total Return Contribution ===')
for th in thresholds:
    mask = df['prob'] >= th
    n = int(np.sum(mask))
    if n == 0:
        continue
    pct = n / len(df) * 100
    filtered_ret = float(df.loc[mask, 'ret'].sum())
    filtered_pct = filtered_ret / total_ret_all * 100 if total_ret_all != 0 else 0
    print(f'  Top {pct:4.1f}% (n={n:4d}): total_ret={filtered_ret:+.4f} ({filtered_pct:+.1f}% of baseline total return)')

# ============================================================
# 时间切片分析: 月度分组
# ============================================================
print()
print('=== 时间切片: 月度分组 ===')

# 按月统计各分组的 hit_rate
monthly = df.groupby('year_month').agg(
    n=('hit', 'count'),
    hit_rate=('hit', 'mean'),
    mean_ret=('ret', 'mean'),
    total_ret=('ret', 'sum'),
).reset_index()

# 高置信门控: prob >= 0.55
high_conf_mask = df['prob'] >= 0.55
df_high = df[high_conf_mask]
monthly_high = df_high.groupby('year_month').agg(
    n_high=('hit', 'count'),
    hit_rate_high=('hit', 'mean'),
    mean_ret_high=('ret', 'mean'),
).reset_index()

# 合并
monthly_merged = monthly.merge(monthly_high, on='year_month', how='left')

print(f'{"月份":>10} {"n_all":>6} {"hit_all":>8} {"n_high":>6} {"hit_high":>8} {"delta_hit":>8} {"备注":>20}')
print('-' * 80)
for _, row in monthly_merged.iterrows():
    ym = str(row['year_month'])
    n_all = int(row['n'])
    hit_all = row['hit_rate'] if not pd.isna(row['hit_rate']) else 0
    n_high = int(row['n_high']) if not pd.isna(row['n_high']) else 0
    hit_high = row['hit_rate_high'] if not pd.isna(row['hit_rate_high']) else 0
    delta = hit_high - hit_all if n_high > 0 else 0
    note = f'n={n_high}' if n_high > 0 else '—'
    print(f'{ym:>10} {n_all:>6} {hit_all:>8.3f} {n_high:>6} {hit_high:>8.3f} {delta:>+8.3f} {note:>20}')

# ============================================================
# 时间切片: 年/季度分组
# ============================================================
print()
print('=== 时间切片: 年度分组 ===')
df['year'] = df['timestamp'].dt.year
df['quarter'] = df['timestamp'].dt.to_period('Q')

yearly = df.groupby('year').agg(
    n=('hit', 'count'),
    hit_rate=('hit', 'mean'),
    mean_ret=('ret', 'mean'),
).reset_index()

high_conf_yearly = df[high_conf_mask].groupby('year').agg(
    n_high=('hit', 'count'),
    hit_rate_high=('hit', 'mean'),
    mean_ret_high=('ret', 'mean'),
).reset_index()

yearly_merged = yearly.merge(high_conf_yearly, on='year', how='left')

print(f'{"年份":>6} {"n_all":>6} {"hit_all":>8} {"n_high":>6} {"hit_high":>8} {"delta_hit":>8}')
print('-' * 55)
for _, row in yearly_merged.iterrows():
    yr = int(row['year'])
    n_all = int(row['n'])
    hit_all = row['hit_rate'] if not pd.isna(row['hit_rate']) else 0
    n_high = int(row['n_high']) if not pd.isna(row['n_high']) else 0
    hit_high = row['hit_rate_high'] if not pd.isna(row['hit_rate_high']) else 0
    delta = hit_high - hit_all if n_high > 0 else 0
    print(f'{yr:>6} {n_all:>6} {hit_all:>8.3f} {n_high:>6} {hit_high:>8.3f} {delta:>+8.3f}')

# ============================================================
# 时间切片: 波动 Regime 分析 (用 ret std 分组)
# ============================================================
print()
print('=== 时间切片: 波动 Regime ===')
# 用 rolling vol 近似波动 regime
df['ret_abs'] = np.abs(df['ret'])
df['vol_regime'] = pd.qcut(df['ret_abs'], q=3, labels=['LowVol', 'MedVol', 'HighVol'])

vol_stats = df.groupby('vol_regime').agg(
    n=('hit', 'count'),
    hit_rate=('hit', 'mean'),
    mean_ret=('ret', 'mean'),
).reset_index()

high_conf_vol = df[high_conf_mask].groupby('vol_regime').agg(
    n_high=('hit', 'count'),
    hit_rate_high=('hit', 'mean'),
    mean_ret_high=('ret', 'mean'),
).reset_index()

vol_merged = vol_stats.merge(high_conf_vol, on='vol_regime', how='left')

print(f'{"Regime":>10} {"n_all":>6} {"hit_all":>8} {"n_high":>6} {"hit_high":>8} {"delta_hit":>8}')
print('-' * 58)
for _, row in vol_merged.iterrows():
    regime = str(row['vol_regime'])
    n_all = int(row['n'])
    hit_all = row['hit_rate'] if not pd.isna(row['hit_rate']) else 0
    n_high = int(row['n_high']) if not pd.isna(row['n_high']) else 0
    hit_high = row['hit_rate_high'] if not pd.isna(row['hit_rate_high']) else 0
    delta = hit_high - hit_all if n_high > 0 else 0
    note = f'n={n_high}' if n_high > 0 else '0'
    print(f'{regime:>10} {n_all:>6} {hit_all:>8.3f} {n_high:>6} {hit_high:>8.3f} {delta:>+8.3f} {note:>10}')

# ============================================================
# 年化频率分析
# ============================================================
print()
print('=== 年化频率分析 ===')
# 训练集时间跨度
ts_all = df['timestamp']
date_range_days = (ts_all.max() - ts_all.min()).days
years = date_range_days / 365.25
annual_rate_all = len(df) / years
print(f'训练集时间跨度: {date_range_days:.0f} 天 ({years:.1f} 年)')
print(f'Baseline 年化交易次数: {annual_rate_all:.1f} 次/年')
for th in [0.50, 0.55, 0.60]:
    mask = df['prob'] >= th
    n = int(np.sum(mask))
    ann_rate = n / years
    print(f'thresh={th}: {n} 单/年 = {ann_rate:.1f} 次/年')

# ============================================================
# 总结: 最优容量-质量点
# ============================================================
print()
print('=== 最优容量-质量点分析 ===')
print('查找"质量提升明显但容量仍可接受"的阈值点:')
for th in thresholds:
    mask = df['prob'] >= th
    n = int(np.sum(mask))
    if n == 0:
        continue
    pct = n / len(df) * 100
    hit = float(df.loc[mask, 'hit'].mean())
    mean_ret = float(df.loc[mask, 'ret'].mean())
    delta_hit = hit - all_hit
    ann_n = n / years
    # 质量改善且年化次数 > 20
    if delta_hit > 0.02 and ann_n >= 20:
        print(f'  thresh={th}: n={n}({pct:.0f}%), hit_rate={hit:.3f}(+{delta_hit:.3f}), mean_ret={mean_ret:.6f}, ann_n={ann_n:.0f}/yr ★')
    elif delta_hit > 0:
        print(f'  thresh={th}: n={n}({pct:.0f}%), hit_rate={hit:.3f}(+{delta_hit:.3f}), mean_ret={mean_ret:.6f}, ann_n={ann_n:.0f}/yr')
