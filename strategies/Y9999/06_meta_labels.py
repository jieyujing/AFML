"""
06_meta_labels.py - Y9999 Phase 5 Meta Labels 生成

根据 AFML Meta-Labeling 方法论：
- Primary Model (MA) 输出 side (-1/+1)
- Trend Scanning 定义趋势方向
- Meta Label: bin = 1 if MA side == Trend Scan side else 0
- Meta Model 学习"什么条件下 MA 信号与统计趋势一致"

流程:
1. 加载 Dollar Bars、事件特征、MA Primary Signals
2. 在 CUSUM 事件点上计算 Trend Scanning
3. 定义 Meta Labels（一致性判断）
4. 计算样本权重
5. 保存输出

输出:
  - meta_labels.parquet: (bin, trend_t_value, trend_window, sample_weight)
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

from strategies.Y9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    PRIMARY_MODEL_TYPE
)

from afmlkit.feature.core.trend_scan import trend_scan_labels

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 配置参数
# ============================================================

# Trend Scanning 窗口范围
TREND_SCAN_WINDOWS = [5, 10, 20, 30, 50]

# 输出目录
MODELS_DIR = os.path.join(os.path.dirname(__file__), "output", "models")


# ============================================================
# 数据加载
# ============================================================

def load_data():
    """
    加载所有需要的数据。

    :returns: (bars, events_features, primary_signals)
    """
    print("\n[Step 1] 加载数据...")
    print(f"  Primary Model 类型: {PRIMARY_MODEL_TYPE}")

    # 加载 Dollar Bars
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target8.parquet')
    bars = pd.read_parquet(bars_path)
    print(f"  Dollar Bars: {len(bars)} bars")
    print(f"  时间范围: {bars.index.min()} ~ {bars.index.max()}")

    # 加载事件特征
    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    events_features = pd.read_parquet(features_path)
    print(f"  事件特征: {events_features.shape[0]} 样本 × {events_features.shape[1]} 特征")

    # 加载 Primary Model 输出
    if PRIMARY_MODEL_TYPE == 'ma':
        primary_path = os.path.join(FEATURES_DIR, 'ma_primary_signals.parquet')
        primary_signals = pd.read_parquet(primary_path)
        print(f"  MA Primary Signals: {len(primary_signals)} 样本")
        print(f"  Side 分布: +1={(primary_signals['side']==1).sum()}, -1={(primary_signals['side']==-1).sum()}")
    elif PRIMARY_MODEL_TYPE == 'supertrend':
        primary_path = os.path.join(FEATURES_DIR, 'supertrend_signals.parquet')
        primary_signals = pd.read_parquet(primary_path)
        print(f"  SuperTrend Signals: {len(primary_signals)} 样本")
        print(f"  Side 分布: +1={(primary_signals['side']==1).sum()}, -1={(primary_signals['side']==-1).sum()}")
    else:
        primary_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
        primary_signals = pd.read_parquet(primary_path)
        print(f"  Trend Labels: {len(primary_signals)} 样本")
        print(f"  Side 分布: +1={(primary_signals['side']==1).sum()}, -1={(primary_signals['side']==-1).sum()}")

    return bars, events_features, primary_signals


# ============================================================
# Trend Scanning Meta Labels 计算
# ============================================================

def compute_meta_labels(
    bars: pd.DataFrame,
    events_features: pd.DataFrame,
    primary_signals: pd.DataFrame
):
    """
    用 Trend Scanning 计算 Meta 标签。

    Meta Label 定义：
      - bin = 1: MA side 与 Trend Scan side 一致
      - bin = 0: MA side 与 Trend Scan side 不一致

    :param bars: Dollar Bars DataFrame
    :param events_features: 事件特征 DataFrame
    :param primary_signals: Primary Model 输出 (side)
    :returns: (features_valid, labels_df)
    """
    print("\n[Step 2] 准备 Features...")

    # 准备特征 DataFrame
    features = events_features.copy()

    # 注入 side 列 (Primary Model 输出)
    common_idx = features.index.intersection(primary_signals.index)
    if len(common_idx) == 0:
        raise ValueError("事件特征和 Primary Signals 没有交集！检查时间戳对齐。")

    features = features.loc[common_idx]
    features['primary_side'] = primary_signals.loc[common_idx, 'side'].astype(int)

    # 过滤 side=0（无信号）
    n_before = len(features)
    features = features[features['primary_side'] != 0].copy()
    print(f"  过滤 side=0: {n_before} → {len(features)} 个有效事件")

    print(f"  Primary Side 分布: +1={(features['primary_side']==1).sum()}, -1={(features['primary_side']==-1).sum()}")

    # 过滤距离数据末尾不足最大窗口的事件
    max_window = max(TREND_SCAN_WINDOWS)
    data_end_idx = len(bars) - 1
    min_event_idx = data_end_idx - max_window

    # 获取事件点的整数索引
    event_indices = [bars.index.get_loc(ts) for ts in features.index]
    valid_mask = np.array(event_indices) <= min_event_idx

    n_before = len(features)
    features = features[valid_mask].copy()
    print(f"  过滤末尾事件（保留足够评估窗口）: {n_before} → {len(features)} 个有效事件")

    print("\n[Step 3] 计算 Trend Scanning...")

    # 准备 Trend Scanning 输入
    price_series = pd.Series(bars['close'].values, index=bars.index)
    event_timestamps = pd.DatetimeIndex(features.index)

    # 调用 Trend Scanning
    trend_df = trend_scan_labels(price_series, event_timestamps, TREND_SCAN_WINDOWS)
    print(f"  Trend Scanning 完成: {len(trend_df)} 个标签")

    # Trend side 分布
    n_up = (trend_df['side'] == 1).sum()
    n_down = (trend_df['side'] == -1).sum()
    n_neutral = (trend_df['side'] == 0).sum()
    print(f"  Trend Side 分布: +1={n_up}, -1={n_down}, 0={n_neutral}")

    print("\n[Step 4] 定义 Meta Labels...")

    # 对齐索引
    trend_df = trend_df.loc[features.index]

    # Meta Label: bin = 1 if primary_side == trend_side else 0
    primary_side = features['primary_side']
    trend_side = trend_df['side']

    # t 值用于 sample weight（不过滤样本）
    t = trend_df['t_value'].abs()

    # 计算 bin
    agreement = (primary_side == trend_side).astype(int)
    bin_labels = agreement

    # 统计
    n_consistent = bin_labels.sum()
    n_inconsistent = len(bin_labels) - n_consistent
    print(f"  一致性分布: 一致={n_consistent} ({n_consistent/len(bin_labels)*100:.1f}%), 不一致={n_inconsistent}")

    # ========================================
    # 关键优化3: 添加冲突特征
    # ========================================
    features['agreement'] = agreement
    features['trend_strength'] = t.values
    features['primary_vs_trend'] = primary_side.values * trend_side.values  # +1=一致, -1=冲突

    print(f"  新增特征: agreement, trend_strength, primary_vs_trend")

    # 构建 labels DataFrame
    labels_df = pd.DataFrame({
        'bin': bin_labels,
        'trend_t_value': t.values,
        'trend_t1': trend_df['t1'].values,
        'trend_side': trend_side.values,
        'primary_side': primary_side.values,
        'agreement': agreement,
    }, index=features.index)

    print("\n[Step 5] 计算样本权重...")

    # ========================================
    # 关键优化2: 稳健权重设计
    # clip + normalize 避免 outlier 主导
    # ========================================
    t_clip = np.clip(t.values, 0, np.percentile(t.values, 95))
    t_norm = t_clip / (t_clip.max() + 1e-8)

    # 使用 exp 放大强趋势权重
    t_exp = np.exp(t_norm * 2)  # 2x 放大
    t_exp = t_exp / t_exp.max()  # 归一化到 [0, 1]

    labels_df['sample_weight'] = np.clip(t_exp, 0.1, 1.0)

    print(f"  权重设计: clip(0, p95) → normalize → exp(2x)")
    print(f"  样本权重范围: [{labels_df['sample_weight'].min():.3f}, {labels_df['sample_weight'].max():.3f}]")
    print(f"  平均样本权重: {labels_df['sample_weight'].mean():.3f}")

    return features, labels_df


# ============================================================
# 可视化
# ============================================================

def plot_meta_label_distribution(labels_df: pd.DataFrame, save_path: str):
    """绘制 Meta 标签分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. bin 分布
    ax1 = axes[0]
    bin_counts = labels_df['bin'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']  # 0=不一致, 1=一致
    ax1.bar(bin_counts.index, bin_counts.values, color=colors)
    ax1.set_xlabel('Label (0=Inconsistent, 1=Consistent)')
    ax1.set_ylabel('Count')
    ax1.set_title('Meta Label Distribution')
    for i, (idx, val) in enumerate(zip(bin_counts.index, bin_counts.values)):
        ax1.annotate(f'{val}\n({val/len(labels_df)*100:.1f}%)',
                     xy=(idx, val), ha='center', va='bottom')

    # 2. |t_value| 分布
    ax2 = axes[1]
    consistent = labels_df[labels_df['bin'] == 1]['trend_t_value']
    inconsistent = labels_df[labels_df['bin'] == 0]['trend_t_value']
    bins = np.linspace(labels_df['trend_t_value'].min(), labels_df['trend_t_value'].max(), 30)
    if len(inconsistent) > 0:
        ax2.hist(inconsistent, bins=bins, color='#ff6b6b', alpha=0.7, label=f'Inconsistent ({len(inconsistent)})')
    if len(consistent) > 0:
        ax2.hist(consistent, bins=bins, color='#4ecdc4', alpha=0.7, label=f'Consistent ({len(consistent)})')
    ax2.set_xlabel('|t-value|')
    ax2.set_ylabel('Count')
    ax2.set_title('Trend Strength by Label')
    ax2.legend()

    # 3. sample_weight 分布
    ax3 = axes[2]
    ax3.hist(labels_df['sample_weight'], bins=30, color='steelblue', alpha=0.7)
    ax3.axvline(labels_df['sample_weight'].mean(), color='red', linestyle='--',
                label=f'Mean: {labels_df["sample_weight"].mean():.3f}')
    ax3.set_xlabel('Sample Weight')
    ax3.set_ylabel('Count')
    ax3.set_title('Sample Weight Distribution')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Meta 标签分布图已保存: {save_path}")


def plot_tvalue_vs_label(labels_df: pd.DataFrame, save_path: str):
    """绘制 |t_value| vs Label 的关系。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按标签分组
    consistent = labels_df[labels_df['bin'] == 1]['trend_t_value']
    inconsistent = labels_df[labels_df['bin'] == 0]['trend_t_value']

    # 箱线图
    data = [consistent, inconsistent]
    bp = ax.boxplot(data, tick_labels=['Consistent (1)', 'Inconsistent (0)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ecdc4')
    bp['boxes'][1].set_facecolor('#ff6b6b')

    ax.set_xlabel('Meta Label')
    ax.set_ylabel('|t-value|')
    ax.set_title('Trend Strength vs Meta Label')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    con_mean = consistent.mean() if len(consistent) > 0 else 0
    inc_mean = inconsistent.mean() if len(inconsistent) > 0 else 0
    ax.annotate(f'Mean: {con_mean:.2f}', xy=(1, con_mean), xytext=(1.2, con_mean),
                fontsize=10, color='#4ecdc4')
    ax.annotate(f'Mean: {inc_mean:.2f}', xy=(2, inc_mean), xytext=(2.2, inc_mean),
                fontsize=10, color='#ff6b6b')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ t-value 分析图已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """Y9999 Meta Labels 生成主流程。"""
    print("=" * 70)
    print("  Y9999 Phase 5 Meta Labels Generation (Trend Scanning)")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    bars, events_features, primary_signals = load_data()

    # Step 2-5: 计算 Meta Labels
    features_valid, labels_df = compute_meta_labels(bars, events_features, primary_signals)

    # Step 6: 可视化
    print("\n[Step 6] 生成可视化图表...")
    plot_meta_label_distribution(
        labels_df,
        os.path.join(FIGURES_DIR, '06_meta_label_distribution.png')
    )
    plot_tvalue_vs_label(
        labels_df,
        os.path.join(FIGURES_DIR, '06_tvalue_vs_label.png')
    )

    # Step 7: 保存输出
    print("\n[Step 7] 保存输出...")

    # 保存 Meta Labels
    labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels_df.to_parquet(labels_path)
    print(f"✅ Meta Labels 已保存: {labels_path}")

    # 保存 Meta Features
    features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    features_valid.to_parquet(features_path)
    print(f"✅ Meta Features 已保存: {features_path}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("  Phase 5 Meta Labels Generation 完成")
    print("=" * 70)
    print(f"输出文件:")
    print(f"  - {FEATURES_DIR}/meta_labels.parquet")
    print(f"  - {FEATURES_DIR}/meta_features.parquet")
    print(f"\n标签统计:")
    bin_counts = labels_df['bin'].value_counts()
    print(f"  一致 (bin=1): {bin_counts.get(1, 0)} ({bin_counts.get(1, 0)/len(labels_df)*100:.1f}%)")
    print(f"  不一致 (bin=0): {bin_counts.get(0, 0)} ({bin_counts.get(0, 0)/len(labels_df)*100:.1f}%)")
    print(f"\n下一步: 运行 07_meta_model.py 训练 Meta Model")


if __name__ == "__main__":
    main()