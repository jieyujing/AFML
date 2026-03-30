"""
04_primary_model_backtest.py - IF9999 Phase 3.5 Primary Model 回测

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 Trend Labels（side, t1, t_value）
3. 计算每个信号的点数收益
4. 统计分析（整体 + t_value 分位数）
5. 生成可视化图表

输出:
  - 统计报告（文本）
  - 04_pnl_distribution.png: 收益分布直方图
  - 04_cumulative_pnl.png: 累积收益曲线
  - 04_tvalue_vs_pnl.png: |t_value| vs pnl 散点图
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
    BARS_DIR, FIGURES_DIR, FEATURES_DIR
)

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


def load_trend_labels(labels_path: str) -> pd.DataFrame:
    """
    加载 Trend Labels 数据。

    :param labels_path: parquet 文件路径
    :returns: DataFrame with columns [t1, t_value, side]
    """
    labels = pd.read_parquet(labels_path)
    print(f"✅ 加载 Trend Labels: {len(labels)} 个信号")
    print(f"   时间范围: {labels.index.min()} ~ {labels.index.max()}")
    print(f"   Side 分布: +1={(labels['side']==1).sum()}, -1={(labels['side']==-1).sum()}")
    return labels


# ============================================================
# 收益计算
# ============================================================

def compute_pnl(
    prices: pd.Series,
    trend_df: pd.DataFrame
) -> pd.DataFrame:
    """
    计算每个信号的点数收益。

    :param prices: Dollar Bars close 价格序列
    :param trend_df: Trend Labels DataFrame (t1, t_value, side)
    :returns: DataFrame with columns [pnl, t_value, side, window_length, entry_price, exit_price]
    """
    results = []

    for event_ts, row in trend_df.iterrows():
        t1 = row['t1']
        side = row['side']
        t_value = row['t_value']

        # 入场价格（事件点 close）
        try:
            entry_price = prices.loc[event_ts]
        except KeyError:
            # 事件点可能在 bars 中不存在（边界情况）
            continue

        # 出场价格（t1 时间点 close）
        try:
            exit_price = prices.loc[t1]
        except KeyError:
            # t1 可能超出 bars 范围，取最近的
            if t1 > prices.index.max():
                exit_price = prices.iloc[-1]
            else:
                # 找 t1 之前最近的 bar
                valid_idx = prices.index[prices.index <= t1]
                if len(valid_idx) == 0:
                    continue
                exit_price = prices.loc[valid_idx[-1]]

        # 点数收益 = side * (exit - entry)
        pnl = side * (exit_price - entry_price)

        # 窗口长度（bars 数量）
        window_bars = len(prices.loc[event_ts:t1]) - 1  # 排除起点

        results.append({
            'timestamp': event_ts,
            'pnl': pnl,
            't_value': t_value,
            'side': side,
            'window_length': window_bars,
            'entry_price': entry_price,
            'exit_price': exit_price
        })

    pnl_df = pd.DataFrame(results)
    pnl_df = pnl_df.set_index('timestamp')

    print(f"\n✅ 收益计算完成: {len(pnl_df)} 个有效信号")
    print(f"   总收益: {pnl_df['pnl'].sum():.2f} 点")
    print(f"   平均收益: {pnl_df['pnl'].mean():.2f} 点")

    return pnl_df


# ============================================================
# Main
# ============================================================

def main():
    """IF9999 Primary Model Backtest 主流程."""
    print("=" * 70)
    print("  IF9999 Phase 3.5 Primary Model Backtest")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    labels_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    trend_df = load_trend_labels(labels_path)

    # Step 2: 计算收益
    print("\n[Step 2] 计算收益...")
    pnl_df = compute_pnl(prices, trend_df)

    # Step 3-5: 待实现
    print("\n[Step 3-5] 统计分析和可视化待实现...")

    print("\n完成！")


if __name__ == "__main__":
    main()
