"""
05_feature_importance.py - IF9999 策略特征重要性分析

根据 AFML Chapter 8，使用 Clustered MDA 评估特征重要性：
1. 特征聚类（解决共线性导致的替代效应）
2. Clustered MDA（对整个 cluster 置换，而非单个特征）
3. 输出重要性排名和可视化

参考文献：
- AFML Chapter 8: Feature Importance
- López de Prado (2017) "Feature Importance for Financial ML"
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR
from afmlkit.importance.clustering import (
    cluster_features,
    get_feature_distance_matrix,
    hierarchical_clustering,
)
from afmlkit.importance.mda import clustered_mda
from scipy.cluster.hierarchy import dendrogram


# ============================================================
# 配置
# ============================================================

# 排除的元数据列
META_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'trades', 'median_trade_size', 'log_return',
    'bin', 't1', 'avg_uniqueness', 'return_attribution',
    'side', 'ret', 'touch', 't_value', 'agreement',
]

# CV 配置
CV_N_SPLITS = 5
CV_EMBARGO_PCT = 0.01
MDA_N_REPEATS = 1
RANDOM_STATE = 42


# ============================================================
# 辅助函数
# ============================================================

def load_features_and_labels():
    """加载特征矩阵和标签。"""
    # 加载特征
    features_path = os.path.join(FEATURES_DIR, 'features.parquet')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征文件不存在: {features_path}")

    X = pd.read_parquet(features_path)
    print(f"[Data] 加载特征: {X.shape}")

    # 加载 Meta Labels
    meta_labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    if os.path.exists(meta_labels_path):
        labels = pd.read_parquet(meta_labels_path)
        y = labels['agreement']
        t1 = labels['t1'] if 't1' in labels.columns else None
        sample_weight = labels.get('sample_weight')
        print(f"[Data] 加载 Meta Labels: {len(y)}")
    else:
        # 回退到 TBM 结果
        tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
        if not os.path.exists(tbm_path):
            raise FileNotFoundError(f"标签文件不存在: {tbm_path}")

        tbm = pd.read_parquet(tbm_path)
        y = (tbm['ret'] > 0).astype(int)  # 简单正负标签
        t1 = tbm.get('t1')
        sample_weight = None
        print(f"[Data] 加载 TBM 标签: {len(y)}")

    # 对齐索引
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    if t1 is not None:
        t1 = t1.loc[common_idx]
    if sample_weight is not None:
        sample_weight = sample_weight.loc[common_idx]

    # 排除元数据列
    feature_cols = [c for c in X.columns if c not in META_COLS]
    X = X[feature_cols].copy()

    # 清理数据
    X = X.ffill().bfill()

    # 删除常量特征
    const_cols = X.columns[X.nunique() <= 1]
    if len(const_cols) > 0:
        print(f"[Data] 删除常量特征: {const_cols.tolist()}")
        X = X.drop(columns=const_cols)

    # 处理无穷值
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 删除仍有 NaN 的列
    nan_cols = X.columns[X.isna().any()]
    if len(nan_cols) > 0:
        print(f"[Data] 删除含 NaN 特征: {nan_cols.tolist()}")
        X = X.drop(columns=nan_cols)

    print(f"[Data] 最终特征数: {X.shape[1]}")
    print(f"[Data] 标签分布: {y.value_counts().to_dict()}")

    return X, y, t1, sample_weight


def plot_dendrogram(X, output_dir):
    """绘制特征聚类树状图。"""
    dist = get_feature_distance_matrix(X)
    link = hierarchical_clustering(dist, method='ward')

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        link,
        labels=X.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title('Feature Clustering Dendrogram (Ward Linkage)', fontsize=14)
    ax.set_ylabel('Distance', fontsize=12)
    fig.tight_layout()

    fpath = os.path.join(output_dir, 'feature_dendrogram.png')
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[Plot] 保存树状图 → {fpath}")


def plot_importance(df_result, output_dir):
    """绘制特征重要性条形图。"""
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_result) * 0.6)))

    # 升序排列（从下到上显示）
    df_plot = df_result.sort_values('mean_importance', ascending=True)

    labels = [
        f"C{int(row.cluster_id)}: {', '.join(row.features[:3])}"
        + ('…' if len(row.features) > 3 else '')
        for _, row in df_plot.iterrows()
    ]
    y_pos = np.arange(len(labels))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in df_plot['mean_importance']]

    ax.barh(
        y_pos,
        df_plot['mean_importance'],
        xerr=df_plot['std_importance'],
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Importance (neg-log-loss drop)', fontsize=11)
    ax.set_title('Clustered MDA Feature Importance', fontsize=14)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    fig.tight_layout()

    fpath = os.path.join(output_dir, 'clustered_mda_importance.png')
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[Plot] 保存重要性图 → {fpath}")


# ============================================================
# 主函数
# ============================================================

def main():
    """IF9999 特征重要性分析主流程。"""
    print("=" * 70)
    print("  IF9999 Feature Importance Analysis")
    print("=" * 70)

    # 创建输出目录
    output_dir = os.path.join(FEATURES_DIR.replace('features', 'figures'))
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载特征和标签...")
    X, y, t1, sample_weight = load_features_and_labels()

    # Step 2: 特征聚类
    print("\n[Step 2] 特征聚类...")
    clusters = cluster_features(X, method='ward')
    print(f"   聚类数: {len(clusters)}")

    # 绘制树状图
    plot_dendrogram(X, output_dir)

    # Step 3: Clustered MDA
    print("\n[Step 3] Clustered MDA (PurgedKFold + Log-loss)...")

    if t1 is None:
        # 使用简单的时间偏移
        t1 = X.index + pd.Timedelta(days=1)

    df_importance = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        sample_weight=sample_weight,
        n_splits=CV_N_SPLITS,
        embargo_pct=CV_EMBARGO_PCT,
        n_repeats=MDA_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # 保存结果
    results_path = os.path.join(FEATURES_DIR, 'feature_importance.parquet')
    df_importance.to_parquet(results_path)
    print(f"\n[Output] 保存重要性结果 → {results_path}")

    # 绘制重要性图
    plot_importance(df_importance, output_dir)

    # Step 4: 输出摘要
    print("\n" + "=" * 70)
    print("  摘要")
    print("=" * 70)
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  聚类数: {len(clusters)}")
    print(f"  CV 折数: {CV_N_SPLITS} (embargo={CV_EMBARGO_PCT:.1%})")
    print(f"  评分方式: Weighted Log-loss")

    # Top 5 重要特征
    print("\n  Top 5 重要特征聚类:")
    for _, row in df_importance.nlargest(5, 'mean_importance').iterrows():
        features_str = ', '.join(row['features'][:3])
        print(f"    [{int(row['cluster_id'])}] {features_str}: {row['mean_importance']:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()