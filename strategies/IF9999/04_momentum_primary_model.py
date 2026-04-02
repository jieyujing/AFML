"""
04_momentum_primary_model.py - IF9999 动量方向 Primary Model 回测

动量方向策略（趋势跟踪）：
- ROC > 0 → 做多 (+1)
- ROC < 0 → 做空 (-1)

与 Trend Scanning 理念一致，都是在捕捉趋势方向。

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 CUSUM 事件点
3. 计算 ROC（收益率）
4. 在事件点生成信号（跟随动量方向）
5. 计算每个信号的点数收益
6. 统计分析和可视化

输出:
  - 统计报告（文本）
  - 04_mom_pnl_distribution.png: 收益分布直方图
  - 04_mom_cumulative_pnl.png: 累积收益曲线
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
# 配置参数
# ============================================================

# 动量参数
MOMENTUM_CONFIG = {
    'lookback': 1,  # ROC 回看周期（1 = 使用前一个 bar 的收益率）
}


# ============================================================
# 数据加载
# ============================================================

def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """加载 Dollar Bars 数据。"""
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def load_cusum_events(events_path: str) -> pd.DataFrame:
    """加载 CUSUM 事件点数据。"""
    events = pd.read_parquet(events_path)
    print(f"✅ 加载 CUSUM 事件点: {len(events)} 个")
    print(f"   时间范围: {events['timestamp'].min()} ~ {events['timestamp'].max()}")
    return events


# ============================================================
# 动量方向信号生成
# ============================================================

def compute_momentum_signals(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    在 CUSUM 事件点生成动量方向信号。

    规则: ROC > 0 → long, ROC < 0 → short

    :param bars: Dollar Bars DataFrame
    :param events: CUSUM 事件点 DataFrame
    :param config: MOMENTUM_CONFIG
    :returns: DataFrame with columns [timestamp, price, side, roc]
    """
    close = bars['close'].values.astype(np.float64)
    lookback = config.get('lookback', 1)

    # 计算 ROC (简单收益率)
    roc = np.zeros(len(close))
    roc[:lookback] = np.nan
    for i in range(lookback, len(close)):
        roc[i] = (close[i] - close[i - lookback]) / close[i - lookback]

    print(f"\n[Momentum] 计算 ROC: lookback={lookback}")

    # 获取事件点在 bars 中的索引位置
    event_indices = [bars.index.get_loc(ts) for ts in events['timestamp']]

    # 在事件点生成信号
    signals = events.copy()
    signals['idx'] = event_indices
    signals['roc'] = roc[event_indices]

    # side = sign(roc)
    signals['side'] = np.sign(signals['roc'])
    signals['side'] = signals['side'].replace(0, 1).astype(int)  # roc=0 时默认做多

    # 统计
    n_long = (signals['side'] == 1).sum()
    n_short = (signals['side'] == -1).sum()
    print(f"\n[信号] 生成完成")
    print(f"   总信号数: {len(signals)}")
    print(f"   多头信号: {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"   空头信号: {n_short} ({n_short/len(signals)*100:.1f}%)")

    return signals


# ============================================================
# 收益计算
# ============================================================

def compute_pnl_with_horizon(
    prices: pd.Series,
    signals: pd.DataFrame,
    horizon_bars: int = 10
) -> pd.DataFrame:
    """计算每个信号的点数收益（固定持仓周期）。"""
    results = []

    for row in signals.itertuples():
        event_ts = row.timestamp
        side = row.side
        entry_price = row.price
        event_idx = row.idx
        exit_idx = event_idx + horizon_bars

        if exit_idx >= len(prices):
            continue

        exit_price = prices.iloc[exit_idx]
        pnl = side * (exit_price - entry_price)

        results.append({
            'timestamp': event_ts,
            'pnl': pnl,
            'side': side,
            'roc': row.roc,
            'entry_price': entry_price,
            'exit_price': exit_price,
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
    """计算整体统计指标。"""
    n_total = len(pnl_df)
    n_win = int((pnl_df['pnl'] > 0).sum())
    n_loss = int((pnl_df['pnl'] < 0).sum())
    win_rate = n_win / n_total if n_total > 0 else 0
    avg_pnl = pnl_df['pnl'].mean()
    total_pnl = pnl_df['pnl'].sum()

    wins = pnl_df[pnl_df['pnl'] > 0]['pnl']
    losses = pnl_df[pnl_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': pnl_df['pnl'].max(),
        'max_loss': pnl_df['pnl'].min()
    }


def print_overall_stats(stats: dict, horizon: int):
    """打印整体统计报告。"""
    print("\n" + "=" * 70)
    print(f"  Momentum Primary Model Backtest - 持仓周期: {horizon} bars")
    print("=" * 70)
    print(f"总信号数: {stats['n_total']}")
    print(f"盈利信号: {stats['n_win']} ({stats['win_rate']*100:.1f}%)")
    print(f"亏损信号: {stats['n_loss']} ({stats['n_loss']/stats['n_total']*100:.1f}%)")
    print("-" * 70)
    print(f"胜率: {stats['win_rate']*100:.1f}%")
    print(f"平均收益: {stats['avg_pnl']:.2f} 点")
    print(f"总收益: {stats['total_pnl']:.2f} 点")
    print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    print("=" * 70)


# ============================================================
# 可视化
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
    ax.set_title(f'Momentum Primary Model PnL Distribution (Horizon={horizon} bars)', fontsize=12)
    ax.set_xlabel('PnL (points)')
    ax.set_ylabel('Count')
    ax.legend()
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
    ax.set_title(f'Momentum Primary Model Cumulative PnL (Horizon={horizon} bars)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative PnL (points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 累积收益图已保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    """IF9999 Momentum Primary Model Backtest 主流程。"""
    print("=" * 70)
    print("  IF9999 Momentum Primary Model Backtest")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = load_cusum_events(events_path)

    # Step 2: 生成信号
    print("\n[Step 2] 生成 Momentum Primary Model 信号...")
    signals = compute_momentum_signals(bars, events, MOMENTUM_CONFIG)

    # Step 3: 计算收益
    horizons = [5, 10, 20]
    for horizon in horizons:
        print(f"\n[Step 3.{horizon}] 计算收益 (horizon={horizon})...")
        pnl_df = compute_pnl_with_horizon(prices, signals, horizon_bars=horizon)

        stats = compute_overall_stats(pnl_df)
        print_overall_stats(stats, horizon)

        plot_pnl_distribution(pnl_df, os.path.join(FIGURES_DIR, f'04_mom_pnl_dist_h{horizon}.png'), horizon)
        plot_cumulative_pnl(pnl_df, os.path.join(FIGURES_DIR, f'04_mom_cum_pnl_h{horizon}.png'), horizon)

    # 保存信号
    signals_to_save = signals.set_index('timestamp')
    signals_path = os.path.join(FEATURES_DIR, 'momentum_signals.parquet')
    signals_to_save.to_parquet(signals_path)
    print(f"\n✅ 信号已保存: {signals_path}")

    print("\n" + "=" * 70)
    print("  Momentum Primary Model Backtest 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()