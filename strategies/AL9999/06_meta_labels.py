"""
06_meta_labels.py - AL9999 Phase 5 Meta Labels 生成 (修正版)

根据 AFML Meta-Labeling 方法论：
- Primary Model 输出 side (-1/+1)
- TBM 给出真实交易结果 (label: 1=盈利达标, 0=未达标)
- Meta Label: bin = 1 if 价格朝预测方向移动 (ret > 0)
- Meta Model 学习"什么条件下 MA 信号能够产生利润"

重要：TBM 在 meta labeling 模式下，ret 已经乘以了 side：
  ret = (log_close[j] - base_price) * side
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
    FEATURES_DIR, FIGURES_DIR
)

sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 70)
    print("  AL9999 Phase 5 Meta Labels Generation (Corrected)")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 TBM 结果
    print("\n[Step 1] 加载 TBM 结果...")
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    if not os.path.exists(tbm_path):
        raise FileNotFoundError(f"找不到 TBM 结果文件: {tbm_path}。请确保已运行之前的 TBM 脚本。")
    
    tbm = pd.read_parquet(tbm_path)
    print(f"  TBM 样本数: {len(tbm)}")
    print(f"  Side 分布: +1={(tbm['side']==1).sum()}, -1={(tbm['side']==-1).sum()}")

    # Step 2: 定义 Meta Labels
    print("\n[Step 2] 定义 Meta Labels...")
    
    # TBM meta mode 下，ret 已经乘以了 side
    # ret > 0 表示价格朝预测方向移动 = 正确预测 → bin=1
    # ret <= 0 表示价格未朝预测方向移动 = 错误预测 → bin=0
    correct = (tbm['ret'] > 0)
    
    labels_df = pd.DataFrame(index=tbm.index)
    labels_df['bin'] = correct.astype(int)
    labels_df['side'] = tbm['side']
    labels_df['ret'] = tbm['ret']
    labels_df['label_tbm'] = tbm['label'] # TBM 原始标签 (tp/sl/ts)

    n_pos = labels_df['bin'].sum()
    n_neg = len(labels_df) - n_pos
    print(f"  Meta Label 分布: 正确(1)={n_pos} ({n_pos/len(labels_df)*100:.1f}%), 错误(0)={n_neg}")

    # Step 3: 计算样本权重
    print("\n[Step 3] 计算样本权重 (基于盈利幅度)...")
    
    # 基于 |ret| 的样本权重，体现信号质量
    ret_abs = labels_df['ret'].abs()
    # 归一化并使用指数映射放大差异
    ret_norm = ret_abs / (ret_abs.max() + 1e-8)
    sample_weight = np.exp(ret_norm * 2) 
    sample_weight = np.clip(sample_weight / sample_weight.max(), 0.1, 1.0)
    
    labels_df['sample_weight'] = sample_weight
    
    print(f"  样本权重范围: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    print(f"  平均样本权重: {sample_weight.mean():.4f}")

    # Step 4: 可视化
    print("\n[Step 4] 生成可视化图表...")
    plot_path = os.path.join(FIGURES_DIR, '06_meta_label_distribution.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 标签分布
    sns.countplot(x='bin', data=labels_df, ax=axes[0], palette=['#ff6b6b', '#4ecdc4'])
    axes[0].set_title('Meta Label Distribution (0=Loss, 1=Gain)')
    
    # 2. 权重分布
    sns.histplot(labels_df['sample_weight'], bins=30, ax=axes[1], color='skyblue', kde=True)
    axes[1].set_title('Sample Weight Distribution')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✅ 可视化图表已保存: {plot_path}")

    # Step 5: 保存输出
    print("\n[Step 5] 保存输出...")
    output_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels_df.to_parquet(output_path)
    print(f"  ✅ Meta Labels 已保存: {output_path}")

    print("\n" + "=" * 70)
    print("  Phase 5 Meta Labels Generation 完成 (修正版)")
    print("=" * 70)


if __name__ == "__main__":
    main()