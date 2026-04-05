"""
03_trend_scanning.py - AL9999 Phase 3 Trend Scanning 标签生成

流程:
1. 加载 Dollar Bars（原始价格）
2. 加载 CUSUM 事件点
3. 应用 Trend Scanning 生成标签
4. 保存输出
5. 生成诊断可视化

输出:
  - trend_labels.parquet: 包含 t1, t_value, side
  - 03_trend_distribution.png: side 和 t_value 分布
  - 03_trend_example.png: 趋势窗口示例
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
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    TREND_WINDOWS, TARGET_DAILY_BARS
)

from afmlkit.feature.core.trend_scan import trend_scan_labels

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 数据加载
# ============================================================

def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """
    加载 Dollar Bars 数据。

    :param bars_path: parquet 文件路径
    :returns: DataFrame with timestamp index
    """
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def load_cusum_events(events_path: str) -> pd.DatetimeIndex:
    """
    加载 CUSUM 事件点。

    :param events_path: parquet 文件路径
    :returns: 事件点 DatetimeIndex
    """
    events_df = pd.read_parquet(events_path)
    # 使用 timestamp 列构建 DatetimeIndex
    t_events = pd.DatetimeIndex(events_df['timestamp'])
    print(f"✅ 加载 CUSUM 事件点: {len(t_events)} 个")
    print(f"   时间范围: {t_events.min()} ~ {t_events.max()}")
    return t_events


# ============================================================
# Trend Scanning 处理
# ============================================================

def run_trend_scanning(
    prices: pd.Series,
    t_events: pd.DatetimeIndex,
    L_windows: list
) -> pd.DataFrame:
    """
    运行 Trend Scanning 标签生成。

    :param prices: 原始价格序列（close）
    :param t_events: CUSUM 事件点 DatetimeIndex
    :param L_windows: 窗口长度列表
    :returns: 标签 DataFrame (t1, t_value, side)
    """
    print("\n[Trend Scanning] 开始处理...")
    print(f"  窗口范围: {L_windows}")

    # 调用 afmlkit 的 trend_scan_labels
    trend_df = trend_scan_labels(
        price_series=prices,
        t_events=t_events,
        L_windows=L_windows
    )

    # 统计结果
    n_up = int((trend_df['side'] == 1).sum())
    n_down = int((trend_df['side'] == -1).sum())
    n_flat = int((trend_df['side'] == 0).sum())

    print(f"\n[Trend Scanning] 处理完成")
    print(f"  总事件数: {len(trend_df)}")
    print(f"  上涨趋势 (+1): {n_up} ({n_up/len(trend_df)*100:.1f}%)")
    print(f"  下跌趋势 (-1): {n_down} ({n_down/len(trend_df)*100:.1f}%)")
    print(f"  无趋势 (0): {n_flat} ({n_flat/len(trend_df)*100:.1f}%)")
    print(f"  平均 |t_value|: {trend_df['t_value'].abs().mean():.2f}")

    return trend_df


# ============================================================
# 输出保存
# ============================================================

def save_outputs(trend_df: pd.DataFrame):
    """
    保存 Trend Scanning 标签。

    :param trend_df: 标签 DataFrame
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 保存趋势标签
    labels_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    trend_df.to_parquet(labels_path)
    print(f"\n✅ Trend Labels 已保存: {labels_path}")
    print(f"   行数: {len(trend_df)}")
    print(f"   列: {list(trend_df.columns)}")


# ============================================================
# 可视化模块
# ============================================================

def plot_trend_distribution(trend_df: pd.DataFrame, save_path: str):
    """
    绘制 side 和 t_value 分布图。

    :param trend_df: 标签 DataFrame
    :param save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Side 分布
    side_counts = trend_df['side'].value_counts().sort_index()

    # 确保所有 side 值都有对应颜色
    present_sides = side_counts.index.tolist()
    bar_colors = []
    bar_labels = []
    for side in present_sides:
        if side == -1:
            bar_colors.append('#ff6b6b')
            bar_labels.append('Down (-1)')
        elif side == 0:
            bar_colors.append('#999999')
            bar_labels.append('Flat (0)')
        else:
            bar_colors.append('#4ecdc4')
            bar_labels.append('Up (+1)')

    ax1.bar(range(len(side_counts)), side_counts.values, color=bar_colors)
    ax1.set_xticks(range(len(side_counts)))
    ax1.set_xticklabels(bar_labels)
    ax1.set_title('Trend Direction Distribution', fontsize=12)
    ax1.set_ylabel('Count')

    # t_value 分布（绝对值）
    t_abs = trend_df['t_value'].abs()
    ax2.hist(t_abs, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(t_abs.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean |t|={t_abs.mean():.2f}')
    ax2.axvline(t_abs.median(), color='orange', linestyle='--', linewidth=1.5,
                label=f'Median |t|={t_abs.median():.2f}')
    ax2.set_title('|t-value| Distribution (Sample Weight)', fontsize=12)
    ax2.set_xlabel('|t-value|')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 趋势分布图已保存: {save_path}")


def plot_trend_example(
    prices: pd.Series,
    trend_df: pd.DataFrame,
    n_examples: int = 5,
    save_path: str = None
):
    """
    绘制趋势窗口示例图。

    :param prices: 价格序列
    :param trend_df: 标签 DataFrame
    :param n_examples: 示例数量
    :param save_path: 保存路径
    """
    # 选择 t_value 绝对值最大的几个事件作为示例
    top_events = trend_df['t_value'].abs().nlargest(n_examples)

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples), sharex=False)

    for i, (event_ts, t_abs) in enumerate(top_events.items()):
        ax = axes[i] if n_examples > 1 else axes

        row = trend_df.loc[event_ts]
        t1 = row['t1']
        side = row['side']
        t_value = row['t_value']

        # 获取趋势窗口内的价格
        if pd.notna(t1):
            window_prices = prices.loc[event_ts:t1]
            ax.plot(window_prices.index, window_prices.values,
                    color='steelblue', linewidth=1.5)

            # 标记起点和终点
            ax.scatter([event_ts], [prices.loc[event_ts]],
                       color='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter([t1], [prices.loc[t1]],
                       color='red', s=100, marker='x', label='End (t1)', zorder=5)

            # 标注 side 和 t_value
            direction = '↑ Up' if side == 1 else '↓ Down' if side == -1 else '— Flat'
            ax.set_title(f'{i+1}. {direction} (t={t_value:.2f}) | {event_ts.strftime("%Y-%m-%d %H:%M")} → {t1.strftime("%H:%M")}',
                        fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 趋势示例图已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    AL9999 Phase 3 Trend Scanning 主流程。
    """
    print("=" * 70)
    print("  AL9999 Phase 3 Trend Scanning Labels")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # Step 2: 加载 CUSUM 事件点
    print("\n[Step 2] 加载 CUSUM 事件点...")
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    t_events = load_cusum_events(events_path)

    # Step 3: Trend Scanning
    print("\n[Step 3] 应用 Trend Scanning...")
    trend_df = run_trend_scanning(prices, t_events, TREND_WINDOWS)

    # Step 4: 保存输出
    print("\n[Step 4] 保存输出文件...")
    save_outputs(trend_df)

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_trend_distribution(
        trend_df,
        os.path.join(FIGURES_DIR, '03_trend_distribution.png')
    )
    plot_trend_example(
        prices, trend_df, n_examples=5,
        save_path=os.path.join(FIGURES_DIR, '03_trend_example.png')
    )

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 3 Trend Scanning 完成")
    print("=" * 70)
    print(f"输出文件: {FEATURES_DIR}")
    print(f"  - trend_labels.parquet")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 03_trend_distribution.png")
    print(f"  - 03_trend_example.png")


if __name__ == "__main__":
    main()