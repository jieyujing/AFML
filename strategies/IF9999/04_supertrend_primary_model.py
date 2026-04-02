"""
04_supertrend_primary_model.py - IF9999 SuperTrend Primary Model 回测

SuperTrend 是一个趋势跟踪指标，使用 ATR 动态确定支撑/阻力位：
- direction = +1 → uptrend (做多)
- direction = -1 → downtrend (做空)

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 CUSUM 事件点
3. 计算 SuperTrend 指标
4. 在事件点生成信号（跟随 SuperTrend direction）
5. 计算每个信号的点数收益
6. 统计分析和可视化

输出:
  - 统计报告（文本）
  - 04_st_pnl_distribution.png: 收益分布直方图
  - 04_st_cumulative_pnl.png: 累积收益曲线
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
    SUPERTREND_CONFIG
)

from afmlkit.feature.core.trend import supertrend

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


def load_cusum_events(events_path: str) -> pd.DataFrame:
    """
    加载 CUSUM 事件点数据。

    :param events_path: parquet 文件路径
    :returns: DataFrame with columns [timestamp, price, fracdiff]
    """
    events = pd.read_parquet(events_path)
    print(f"✅ 加载 CUSUM 事件点: {len(events)} 个")
    print(f"   时间范围: {events['timestamp'].min()} ~ {events['timestamp'].max()}")
    return events


# ============================================================
# SuperTrend Primary Model 信号生成
# ============================================================

def compute_supertrend_signals(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    在 CUSUM 事件点生成 SuperTrend Primary Model 信号。

    SuperTrend direction:
      - +1: uptrend (做多)
      - -1: downtrend (做空)

    支持多参数组合融合：
      - 'or': 任一参数做多 → 做多（High Recall）
      - 'and': 所有参数做多 → 做多（High Precision）

    :param bars: Dollar Bars DataFrame
    :param events: CUSUM 事件点 DataFrame
    :param config: SUPERTREND_CONFIG
    :returns: DataFrame with columns [timestamp, price, side, confidence]
    """
    high = bars['high'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    close = bars['close'].values.astype(np.float64)

    params_combinations = config.get('params_combinations', [
        {'period': 10, 'multiplier': 3.0, 'name': 'default'}
    ])
    fusion_method = config.get('fusion_method', 'or')

    print(f"\n[SuperTrend] 计算多参数组合...")
    print(f"   参数组合数: {len(params_combinations)}")
    print(f"   融合方法: {fusion_method}")

    # 计算每个参数组合的 SuperTrend
    direction_maps = {}  # name -> direction array
    trend_line_maps = {}  # name -> trend_line array

    for params in params_combinations:
        period = params['period']
        multiplier = params['multiplier']
        name = params['name']

        trend_line, direction, upper, lower = supertrend(
            high, low, close,
            atr_period=period,
            multiplier=multiplier
        )

        direction_maps[name] = direction
        trend_line_maps[name] = trend_line

        # 统计趋势分布
        n_up = np.sum(direction == 1)
        n_down = np.sum(direction == -1)
        print(f"   {name} (period={period}, mult={multiplier}): uptrend={n_up}, downtrend={n_down}")

    # 获取事件点在 bars 中的索引位置
    event_indices = [bars.index.get_loc(ts) for ts in events['timestamp']]

    # 在事件点融合信号
    signals = events.copy()
    signals['idx'] = event_indices

    # 收集每个参数的 direction
    directions_at_events = []
    for params in params_combinations:
        name = params['name']
        direction = direction_maps[name]
        dir_at_events = direction[event_indices]
        signals[f'dir_{name}'] = dir_at_events
        directions_at_events.append(dir_at_events)

    # 融合决策
    directions_stack = np.vstack(directions_at_events)  # (n_params, n_events)

    if fusion_method == 'or':
        # 任一做多 → 做多，否则做空
        n_long_signals = np.sum(directions_stack == 1, axis=0)
        side = np.where(n_long_signals >= 1, 1, -1).astype(int)
        # confidence = 做多参数数 / 总参数数
        confidence = n_long_signals / len(params_combinations)
    elif fusion_method == 'and':
        # 所有做多 → 做多，否则做空
        n_long_signals = np.sum(directions_stack == 1, axis=0)
        side = np.where(n_long_signals == len(params_combinations), 1, -1).astype(int)
        confidence = np.where(side == 1, 1.0, n_long_signals / len(params_combinations))
    else:
        # 默认：使用第一个参数
        side = directions_stack[0].astype(int)
        confidence = np.ones(len(side))

    signals['side'] = side
    signals['confidence'] = confidence

    # 统计
    n_long = (signals['side'] == 1).sum()
    n_short = (signals['side'] == -1).sum()
    print(f"\n[信号] 生成完成")
    print(f"   融合方法: {fusion_method}")
    print(f"   总信号数: {len(signals)}")
    print(f"   多头信号: {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"   空头信号: {n_short} ({n_short/len(signals)*100:.1f}%)")
    print(f"   平均置信度: {signals['confidence'].mean():.2f}")

    return signals


# ============================================================
# 收益计算
# ============================================================

def compute_pnl_with_horizon(
    prices: pd.Series,
    signals: pd.DataFrame,
    horizon_bars: int = 10
) -> pd.DataFrame:
    """
    计算每个信号的点数收益（固定持仓周期）。

    :param prices: Dollar Bars close 价格序列
    :param signals: 信号 DataFrame (timestamp, price, side, confidence)
    :param horizon_bars: 持仓周期（bars 数量）
    :returns: DataFrame with columns [pnl, side, entry_price, exit_price, window_length]
    """
    results = []

    for row in signals.itertuples():
        event_ts = row.timestamp
        side = row.side
        entry_price = row.price

        # 找到退出时间点（固定 horizon_bars 后）
        event_idx = row.idx
        exit_idx = event_idx + horizon_bars

        # 检查是否超出数据范围
        if exit_idx >= len(prices):
            continue

        exit_ts = prices.index[exit_idx]
        exit_price = prices.iloc[exit_idx]

        # 点数收益 = side * (exit - entry)
        pnl = side * (exit_price - entry_price)

        results.append({
            'timestamp': event_ts,
            'pnl': pnl,
            'side': side,
            'confidence': row.confidence,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_ts': exit_ts,
            'window_length': horizon_bars
        })

    pnl_df = pd.DataFrame(results)
    pnl_df = pnl_df.set_index('timestamp')

    print(f"\n✅ 收益计算完成: {len(pnl_df)} 个有效信号")
    print(f"   持仓周期: {horizon_bars} bars")
    print(f"   总收益: {pnl_df['pnl'].sum():.2f} 点")
    print(f"   平均收益: {pnl_df['pnl'].mean():.2f} 点")

    return pnl_df


# ============================================================
# 统计分析
# ============================================================

def compute_overall_stats(pnl_df: pd.DataFrame) -> dict:
    """
    计算整体统计指标。

    :param pnl_df: 收益 DataFrame
    :returns: 统计指标字典
    """
    n_total = len(pnl_df)
    n_win = int((pnl_df['pnl'] > 0).sum())
    n_loss = int((pnl_df['pnl'] < 0).sum())
    n_flat = int((pnl_df['pnl'] == 0).sum())

    win_rate = n_win / n_total if n_total > 0 else 0
    avg_pnl = pnl_df['pnl'].mean()
    total_pnl = pnl_df['pnl'].sum()

    # 盈亏比
    wins = pnl_df[pnl_df['pnl'] > 0]['pnl']
    losses = pnl_df[pnl_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    max_win = pnl_df['pnl'].max()
    max_loss = pnl_df['pnl'].min()

    stats = {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'n_flat': n_flat,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss
    }

    return stats


def print_overall_stats(stats: dict, horizon: int):
    """
    打印整体统计报告。

    :param stats: 统计指标字典
    :param horizon: 持仓周期
    """
    print("\n" + "=" * 70)
    print(f"  SuperTrend Primary Model Backtest - 持仓周期: {horizon} bars")
    print("=" * 70)
    print(f"总信号数: {stats['n_total']}")
    print(f"盈利信号: {stats['n_win']} ({stats['win_rate']*100:.1f}%)")
    if stats['n_total'] > 0:
        print(f"亏损信号: {stats['n_loss']} ({stats['n_loss']/stats['n_total']*100:.1f}%)")
    print(f"持平信号: {stats['n_flat']}")
    print("-" * 70)
    print(f"胜率: {stats['win_rate']*100:.1f}%")
    print(f"平均收益: {stats['avg_pnl']:.2f} 点")
    print(f"总收益: {stats['total_pnl']:.2f} 点")
    print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    print(f"平均盈利: {stats['avg_win']:.2f} 点")
    print(f"平均亏损: {stats['avg_loss']:.2f} 点")
    print(f"最大盈利: {stats['max_win']:.2f} 点")
    print(f"最大亏损: {stats['max_loss']:.2f} 点")
    print("=" * 70)


# ============================================================
# 可视化模块
# ============================================================

def plot_pnl_distribution(pnl_df: pd.DataFrame, save_path: str, horizon: int):
    """绘制收益分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    wins = pnl_df[pnl_df['pnl'] > 0]['pnl']
    losses = pnl_df[pnl_df['pnl'] < 0]['pnl']

    bins = np.linspace(pnl_df['pnl'].min(), pnl_df['pnl'].max(), 50)

    if len(losses) > 0:
        ax.hist(losses, bins=bins, color='#ff6b6b', alpha=0.7, label=f'Loss ({len(losses)})')
    if len(wins) > 0:
        ax.hist(wins, bins=bins, color='#4ecdc4', alpha=0.7, label=f'Win ({len(wins)})')

    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(pnl_df['pnl'].mean(), color='blue', linestyle='-', linewidth=2,
                label=f'Mean: {pnl_df["pnl"].mean():.1f}')

    ax.set_title(f'SuperTrend Primary Model PnL Distribution (Horizon={horizon} bars)', fontsize=12)
    ax.set_xlabel('PnL (points)')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 收益分布图已保存: {save_path}")


def plot_cumulative_pnl(pnl_df: pd.DataFrame, save_path: str, horizon: int):
    """绘制累积收益曲线。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    cumulative = pnl_df['pnl'].cumsum()

    ax.plot(pnl_df.index, cumulative.values, color='steelblue', linewidth=1.5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    final_cum = cumulative.iloc[-1]

    ax.scatter([pnl_df.index[-1]], [final_cum], color='red', s=100, zorder=5)
    ax.annotate(f'Final: {final_cum:.1f}',
                xy=(pnl_df.index[-1], final_cum),
                xytext=(10, 10), textcoords='offset points')

    ax.set_title(f'SuperTrend Primary Model Cumulative PnL (Horizon={horizon} bars)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative PnL (points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 累积收益图已保存: {save_path}")


def plot_confidence_vs_pnl(pnl_df: pd.DataFrame, save_path: str):
    """绘制置信度 vs 收益散点图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按盈亏着色
    colors = np.where(pnl_df['pnl'] > 0, '#4ecdc4', '#ff6b6b')

    ax.scatter(pnl_df['confidence'], pnl_df['pnl'], c=colors, alpha=0.6, s=30)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title('Confidence vs PnL', fontsize=12)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('PnL (points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 置信度 vs 收益图已保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    """IF9999 SuperTrend Primary Model Backtest 主流程."""
    print("=" * 70)
    print("  IF9999 SuperTrend Primary Model Backtest")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = load_cusum_events(events_path)

    # Step 2: 生成信号
    print("\n[Step 2] 生成 SuperTrend Primary Model 信号...")
    signals = compute_supertrend_signals(bars, events, SUPERTREND_CONFIG)

    # Step 3: 计算收益（多个持仓周期）
    horizons = [5, 10, 20]

    for horizon in horizons:
        print(f"\n[Step 3.{horizon}] 计算收益 (horizon={horizon})...")
        pnl_df = compute_pnl_with_horizon(prices, signals, horizon_bars=horizon)

        # Step 4: 统计分析
        print("\n[Step 4] 整体统计分析...")
        stats = compute_overall_stats(pnl_df)
        print_overall_stats(stats, horizon)

        # Step 5: 可视化
        print("\n[Step 5] 生成可视化图表...")
        plot_pnl_distribution(
            pnl_df,
            os.path.join(FIGURES_DIR, f'04_st_pnl_dist_h{horizon}.png'),
            horizon
        )
        plot_cumulative_pnl(
            pnl_df,
            os.path.join(FIGURES_DIR, f'04_st_cum_pnl_h{horizon}.png'),
            horizon
        )

    # Step 6: 置信度分析
    print("\n[Step 6] 置信度分析...")
    pnl_df = compute_pnl_with_horizon(prices, signals, horizon_bars=20)
    plot_confidence_vs_pnl(pnl_df, os.path.join(FIGURES_DIR, '04_st_confidence_vs_pnl.png'))

    # 保存信号
    signals_to_save = signals.set_index('timestamp')
    signals_path = os.path.join(FEATURES_DIR, 'supertrend_signals.parquet')
    signals_to_save.to_parquet(signals_path)
    print(f"\n✅ 信号已保存: {signals_path}")

    print("\n" + "=" * 70)
    print("  SuperTrend Primary Model Backtest 完成")
    print("=" * 70)
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 04_st_pnl_dist_h5.png (持仓 5 bars)")
    print(f"  - 04_st_cum_pnl_h5.png")
    print(f"  - 04_st_pnl_dist_h10.png (持仓 10 bars)")
    print(f"  - 04_st_cum_pnl_h10.png")
    print(f"  - 04_st_pnl_dist_h20.png (持仓 20 bars)")
    print(f"  - 04_st_cum_pnl_h20.png")
    print(f"  - 04_st_confidence_vs_pnl.png")


if __name__ == "__main__":
    main()