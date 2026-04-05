"""
06_meta_labels.py - AL9999 Meta-Labeling 裁决 (DMA × Trend Scanning)

AFML 元标签方法论:
- Primary Model (DMA) 给出预测方向 Side (-1/+1)
- Trend Scanning 定义"真实趋势" (基于 |t_value| 最大化)
- Meta Label: bin = 1 if DMA 与 Trend 方向一致, bin = 0 if 不一致

裁决公式:
    meta_label = 1  if  dma_side == trend_side  and  |t_value| > MIN_T_VALUE
    meta_label = 0  if  dma_side != trend_side  and  |t_value| > MIN_T_VALUE
    (|t_value| <= MIN_T_VALUE 的事件被过滤为纯噪音)

输出:
  - meta_labels.parquet: timestamp, dma_side, trend_side, t_value, meta_label
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    FEATURES_DIR, FIGURES_DIR, MIN_T_VALUE
)

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# Meta-Label 裁决
# ============================================================

def generate_meta_labels(
    dma_signals: pd.DataFrame,
    trend_labels: pd.DataFrame,
    min_t_value: float
) -> pd.DataFrame:
    """
    生成 Meta-Labels：对质 DMA 预测与 Trend Scanning 现实。

    :param dma_signals: DMA 信号 DataFrame (columns: side)
    :param trend_labels: Trend Scanning 标签 DataFrame (columns: side, t_value, is_valid)
    :param min_t_value: t-value 显著性门槛
    :returns: Meta-Label DataFrame
    """
    # 对齐时间戳
    common_idx = dma_signals.index.intersection(trend_labels.index)
    print(f"  对齐时间戳: DMA={len(dma_signals)}, Trend={len(trend_labels)}, 交集={len(common_idx)}")

    dma_aligned = dma_signals.loc[common_idx].copy()
    trend_aligned = trend_labels.loc[common_idx].copy()

    # 合并
    merged = pd.DataFrame({
        'dma_side': dma_aligned['side'].astype(int),
        'trend_side': trend_aligned['side'].astype(int),
        't_value': trend_aligned['t_value'].astype(float),
        't1': trend_aligned['t1'],
        'is_valid': trend_aligned['is_valid'].astype(bool) if 'is_valid' in trend_aligned.columns else trend_aligned['t_value'].abs() > min_t_value,
    })

    # Meta-Label 裁决: 一致 = 1, 不一致 = 0
    merged['meta_label'] = np.where(
        merged['dma_side'] == merged['trend_side'], 1, 0
    )

    # 过滤无效样本 (噪音事件)
    valid_mask = merged['is_valid'] & (merged['t_value'].abs() >= min_t_value)
    n_valid = int(valid_mask.sum())
    n_total = len(merged)

    merged = merged[valid_mask].copy()
    print(f"  过滤前样本: {n_total}")
    print(f"  过滤后样本: {n_valid} (|t| >= {min_t_value})")

    # Meta-Label 分布
    n_correct = int(merged['meta_label'].sum())
    n_wrong = int(len(merged) - n_correct)
    total = len(merged)
    print(f"  Meta-Label 分布: 正确({n_correct}/{total}, {n_correct/total*100:.1f}%), 错误({n_wrong}/{total}, {n_wrong/total*100:.1f}%)")
    print(f"  DMA 侧向分布: 多头={(merged['dma_side']==1).sum()}, 空头={(merged['dma_side']==-1).sum()}")
    print(f"  Trend 侧向分布: 多头={(merged['trend_side']==1).sum()}, 空头={(merged['trend_side']==-1).sum()}")

    return merged


# ============================================================
# 样本权重
# ============================================================

def compute_sample_weights(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算样本权重 (基于 |t_value| 的 z-score 归一化)。

    高于平均 |t| 的样本权重 > 1，低于平均的权重 < 1。
    相比 min-max 归一化，z-score 保留相对分布信息。
    """
    t_abs = labels_df['t_value'].abs()
    t_mean = t_abs.mean()
    t_std = t_abs.std()

    if t_std > 0:
        t_z = (t_abs - t_mean) / t_std
    else:
        t_z = t_abs - t_mean

    labels_df['sample_weight'] = np.clip(t_z + 1.0, 0.1, 1.0)
    print(f"  样本权重范围: [{labels_df['sample_weight'].min():.4f}, {labels_df['sample_weight'].max():.4f}]")
    print(f"  |t| 均值: {t_mean:.4f}, 标准差: {t_std:.4f}")
    return labels_df


# ============================================================
# 可视化
# ============================================================

def plot_meta_label_distribution(labels_df: pd.DataFrame, save_path: str):
    """绘制 Meta-Label 分布和 DMA vs Trend 一致性矩阵。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Meta-Label 分布
    label_counts = labels_df['meta_label'].value_counts().sort_index()
    colors = ['#ff6b6b' if x == 0 else '#4ecdc4' for x in label_counts.index]
    labels_text = ['Mismatch (0)', 'Match (1)']
    ax1.bar(range(len(label_counts)), label_counts.values, color=colors)
    ax1.set_xticks(range(len(label_counts)))
    ax1.set_xticklabels(labels_text)
    ax1.set_title('Meta-Label Distribution')
    ax1.set_ylabel('Count')

    # 2. DMA vs Trend 混淆矩阵
    confusion = pd.crosstab(labels_df['dma_side'], labels_df['trend_side'])
    ax2.imshow(confusion.values, cmap='Blues', aspect='auto')
    ax2.set_xlabel('Trend Scanning (Reality)')
    ax2.set_ylabel('DMA Primary (Prediction)')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels([-1, 1])
    ax2.set_yticklabels([-1, 1])
    ax2.set_title('DMA vs Trend: Consistency Matrix')
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax2.text(j, i, f'{confusion.values[i, j]}',
                     ha='center', va='center', color='black', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化图表已保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    """AL9999 Meta-Labeling 裁决主流程。"""
    print("=" * 70)
    print("  AL9999 Phase 6: Meta-Labeling (DMA × Trend Scanning)")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 DMA 信号
    print("\n[Step 1] 加载 DMA 信号...")
    dma_path = os.path.join(FEATURES_DIR, 'dma_signals.parquet')
    if not os.path.exists(dma_path):
        raise FileNotFoundError(f"找不到 DMA 信号文件: {dma_path}。请先运行 04_dma_primary_model.py。")
    dma_signals = pd.read_parquet(dma_path)
    print(f"  DMA 信号数: {len(dma_signals)}")

    # Step 2: 加载 Trend Scanning 标签
    print("\n[Step 2] 加载 Trend Scanning 标签...")
    trend_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    if not os.path.exists(trend_path):
        raise FileNotFoundError(f"找不到趋势标签文件: {trend_path}。请先运行 03_trend_scanning.py。")
    trend_labels = pd.read_parquet(trend_path)
    print(f"  Trend 标签数: {len(trend_labels)}")

    # Step 3: Meta-Label 裁决
    print("\n[Step 3] 裁决 Meta-Labels...")
    labels_df = generate_meta_labels(dma_signals, trend_labels, MIN_T_VALUE)

    # Step 4: 计算样本权重
    print("\n[Step 4] 计算样本权重...")
    labels_df = compute_sample_weights(labels_df)

    # Step 5: 保存输出
    print("\n[Step 5] 保存输出...")
    output_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels_df.to_parquet(output_path)
    print(f"  Meta-Labels 已保存: {output_path}")

    # Step 6: 可视化
    print("\n[Step 6] 生成可视化图表...")
    plot_meta_label_distribution(
        labels_df,
        os.path.join(FIGURES_DIR, '06_meta_label_distribution.png')
    )

    print("\n" + "=" * 70)
    print("  Meta-Labeling 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
