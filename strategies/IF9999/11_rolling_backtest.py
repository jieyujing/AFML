"""
11_rolling_backtest.py - IF9999 滚动回测（修正版）

修正核心问题：
- 原回测直接累加 TBM 收益，忽略信号重叠导致的收益重复计算
- 滚动回测采用单仓位逻辑：同一时间只能持有一个仓位

规则：
1. 无仓位时：开新仓
2. 持多仓时遇空信号：平多，开空
3. 持空仓时遇多信号：平空，开多
4. 同方向信号：忽略（已有仓位继续持有）

交易成本：
- 手续费：双边 0.23 bp
- 滑点：双边 1.0 点

输出：
- Primary vs Combined 交易记录
- 修正后的绩效对比
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR, FIGURES_DIR, BARS_DIR, CONTRACT_MULTIPLIER

# ============================================================
# 配置
# ============================================================

COMMISSION_RATE = 0.000023  # 手续费率 (双边约 0.23 bp)
SLIPPAGE_POINTS = 0.5       # 单边滑点 (双边 1.0 点)
ANNUALIZATION_FACTOR = 1500
OOS_START = pd.Timestamp('2024-01-01')


# ============================================================
# 滚动回测核心逻辑
# ============================================================

def rolling_backtest(signals: pd.DataFrame, bars: pd.DataFrame, use_meta_filter: bool = True) -> pd.DataFrame:
    """
    单仓位滚动回测。

    :param signals: TBM 结果，包含 side, entry_price, exit_price, pnl 等
    :param bars: Dollar Bars 价格数据
    :param use_meta_filter: 是否使用 Meta 过滤
    :returns: 交易记录 DataFrame
    """
    trades = []
    position = 0  # 当前仓位: 0=无, 1=多, -1=空
    entry_time = None
    entry_price = None

    # 遍历信号（按时间排序）
    for idx, row in signals.sort_index().iterrows():
        signal_side = row['side']

        # Meta 过滤
        if use_meta_filter and row.get('meta_pred', 1) == 0:
            continue

        # 滚动回测逻辑
        if position == 0:
            # 无仓位：开新仓
            position = signal_side
            entry_time = idx
            entry_price = row['entry_price']
        elif position == 1 and signal_side == -1:
            # 持多仓遇空信号：平多，开空
            exit_price = row['entry_price']  # 用新信号的入场价作为平仓价
            pnl = (exit_price - entry_price) * position

            # 计算成本
            comm = (entry_price + exit_price) * COMMISSION_RATE
            slip = SLIPPAGE_POINTS * 2
            net_pnl = pnl - comm - slip
            net_ret = net_pnl / entry_price

            trades.append({
                'entry_time': entry_time,
                'exit_time': idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': position,
                'pnl': pnl,
                'ret': pnl / entry_price,
                'exit_reason': 'reverse_signal',
                'net_pnl': net_pnl,
                'net_ret': net_ret,
            })

            # 开新空仓
            position = signal_side
            entry_time = idx
            entry_price = row['entry_price']

        elif position == -1 and signal_side == 1:
            # 持空仓遇多信号：平空，开多
            exit_price = row['entry_price']
            pnl = (exit_price - entry_price) * position

            # 计算成本
            comm = (entry_price + exit_price) * COMMISSION_RATE
            slip = SLIPPAGE_POINTS * 2
            net_pnl = pnl - comm - slip
            net_ret = net_pnl / entry_price

            trades.append({
                'entry_time': entry_time,
                'exit_time': idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': position,
                'pnl': pnl,
                'ret': pnl / entry_price,
                'exit_reason': 'reverse_signal',
                'net_pnl': net_pnl,
                'net_ret': net_ret,
            })

            # 开新多仓
            position = signal_side
            entry_time = idx
            entry_price = row['entry_price']

        # 同方向信号：忽略，继续持有

    # 最后一个仓位：用最后一个 bar 的收盘价平仓
    if position != 0 and entry_time is not None:
        last_bar = bars.iloc[-1]
        exit_price = last_bar['close']
        pnl = (exit_price - entry_price) * position

        comm = (entry_price + exit_price) * COMMISSION_RATE
        slip = SLIPPAGE_POINTS * 2
        net_pnl = pnl - comm - slip
        net_ret = net_pnl / entry_price

        trades.append({
            'entry_time': entry_time,
            'exit_time': last_bar.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': position,
            'pnl': pnl,
            'ret': pnl / entry_price,
            'exit_reason': 'end_of_data',
            'net_pnl': net_pnl,
            'net_ret': net_ret,
        })

    return pd.DataFrame(trades)


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """计算绩效指标。"""
    if len(trades) == 0:
        return {}

    n = len(trades)
    total_pnl = trades['net_pnl'].sum()
    win_rate = (trades['net_pnl'] > 0).mean()

    # 年化 Sharpe
    sr = trades['net_ret'].mean() / trades['net_ret'].std() * np.sqrt(ANNUALIZATION_FACTOR) if trades['net_ret'].std() > 0 else 0

    # 最大回撤
    cum_pnl = trades['net_pnl'].cumsum()
    mdd = (cum_pnl - cum_pnl.cummax()).min()

    # 年化收益
    days = (trades['exit_time'].max() - trades['entry_time'].min()).days
    annual_pnl = total_pnl / max(days / 365, 0.1)

    return {
        'n_trades': n,
        'total_pnl': total_pnl,
        'annual_pnl': annual_pnl,
        'win_rate': win_rate,
        'sharpe': sr,
        'mdd': mdd,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  IF9999 滚动回测（修正版）")
    print("=" * 70)

    # 加载数据
    print("\n[Step 1] 加载数据...")

    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))
    meta_labels = pd.read_parquet(os.path.join(FEATURES_DIR, 'meta_labels.parquet'))
    bars = pd.read_parquet(os.path.join(BARS_DIR, 'dollar_bars_target6.parquet'))

    print(f"  TBM 事件: {len(tbm)}")
    print(f"  Meta Labels: {len(meta_labels)}")
    print(f"  Dollar Bars: {len(bars)}")

    # 合并 Meta Labels
    tbm['meta_pred'] = meta_labels['bin']

    # Step 2: 滚动回测
    print("\n[Step 2] 执行滚动回测...")

    # Primary Model (无 Meta 过滤)
    primary_trades = rolling_backtest(tbm, bars, use_meta_filter=False)
    print(f"  Primary 交易数: {len(primary_trades)}")

    # Combined Strategy (Meta 过滤)
    combined_trades = rolling_backtest(tbm, bars, use_meta_filter=True)
    print(f"  Combined 交易数: {len(combined_trades)}")

    # Step 3: 绩效对比
    print("\n[Step 3] 绩效统计...")

    primary_metrics = calculate_metrics(primary_trades)
    combined_metrics = calculate_metrics(combined_trades)

    print("\n--- Primary Model (无过滤) ---")
    for k, v in primary_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Combined Strategy (Meta 过滤) ---")
    for k, v in combined_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Step 4: IS/OOS 分解
    print("\n[Step 4] IS/OOS 分解...")

    is_mask = combined_trades['exit_time'] < OOS_START
    oos_mask = combined_trades['exit_time'] >= OOS_START

    is_metrics = calculate_metrics(combined_trades[is_mask])
    oos_metrics = calculate_metrics(combined_trades[oos_mask])

    print(f"  IS 交易数: {is_mask.sum()}")
    print(f"  OOS 交易数: {oos_mask.sum()}")

    if is_metrics:
        print(f"  IS Sharpe: {is_metrics['sharpe']:.2f}")
    if oos_metrics:
        print(f"  OOS Sharpe: {oos_metrics['sharpe']:.2f}")

    # Step 5: 对比原回测结果
    print("\n[Step 5] 对比原回测逻辑...")

    original_stats = pd.read_parquet(os.path.join(FEATURES_DIR, 'backtest_stats.parquet'))
    orig_primary_sharpe = original_stats.loc[original_stats['Metric'] == '年化夏普', 'Primary (Full)'].values[0]
    orig_combined_sharpe = original_stats.loc[original_stats['Metric'] == '年化夏普', 'Combined (Full)'].values[0]

    print(f"  原 Primary Sharpe: {orig_primary_sharpe:.2f}")
    print(f"  原 Combined Sharpe: {orig_combined_sharpe:.2f}")
    print(f"  修正 Primary Sharpe: {primary_metrics['sharpe']:.2f}")
    print(f"  修正 Combined Sharpe: {combined_metrics['sharpe']:.2f}")

    # Step 6: 保存结果
    print("\n[Step 6] 保存结果...")

    primary_trades.to_parquet(os.path.join(FEATURES_DIR, 'rolling_primary_trades.parquet'))
    combined_trades.to_parquet(os.path.join(FEATURES_DIR, 'rolling_combined_trades.parquet'))

    print(f"  Primary 交易记录: {FEATURES_DIR}/rolling_primary_trades.parquet")
    print(f"  Combined 交易记录: {FEATURES_DIR}/rolling_combined_trades.parquet")

    # Step 7: 可视化
    print("\n[Step 7] 生成图表...")

    plt.figure(figsize=(14, 8))

    # IS/OOS 背景
    plt.axvspan(bars.index.min(), OOS_START, color='gray', alpha=0.05, label='In-Sample')
    plt.axvspan(OOS_START, bars.index.max(), color='green', alpha=0.05, label='Out-of-Sample')
    plt.axvline(OOS_START, color='darkred', linestyle='--', alpha=0.5, lw=1.5)

    # 累积收益
    if len(primary_trades) > 0:
        plt.plot(primary_trades['exit_time'], primary_trades['net_pnl'].cumsum(),
                 label='Primary (Rolling)', alpha=0.7, color='#9b59b6', lw=1.5)

    if len(combined_trades) > 0:
        plt.plot(combined_trades['exit_time'], combined_trades['net_pnl'].cumsum(),
                 label='Combined (Rolling)', color='#e67e22', lw=2.5)

    plt.title("IF9999 Rolling Backtest: Corrected Single-Position Logic", fontsize=14, fontweight='bold')
    plt.ylabel("Net PnL (Points)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)

    # 标注修正后的 Sharpe
    plt.text(OOS_START + pd.Timedelta(days=30), combined_metrics['total_pnl'] * 0.1,
             f"OOS Sharpe: {oos_metrics.get('sharpe', 0):.2f}",
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'),
             fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '11_rolling_backtest.png'), dpi=200)
    plt.close()

    print(f"  图表: {FIGURES_DIR}/11_rolling_backtest.png")

    print("\n" + "=" * 70)
    print("  完成")
    print("=" * 70)


if __name__ == "__main__":
    main()