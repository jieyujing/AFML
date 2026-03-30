"""
03_trend_scanning.py - IF9999 Phase 3 Trend Scanning 标签生成

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

from strategies.IF9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    TREND_WINDOWS
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
# 主函数
# ============================================================

def main():
    """
    IF9999 Phase 3 Trend Scanning 主流程。
    """
    print("=" * 70)
    print("  IF9999 Phase 3 Trend Scanning Labels")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
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

    # Step 5: 可视化（待实现）
    print("\n[Step 5] 生成可视化图表...")
    print("  (将在 Task 4 实现)")

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 3 Trend Scanning 完成")
    print("=" * 70)
    print(f"输出文件: {FEATURES_DIR}/trend_labels.parquet")


if __name__ == "__main__":
    main()