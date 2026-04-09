"""
08_pnl_chart.py - 生成策略 PnL 可视化图表

根据最终确定的策略配置 (DMA(3,15) + reverse_signal exit) 生成：
1. 累计 PnL 曲线（标注 IS/OOS 分界线）
2. 回撤曲线
3. 月度收益热力图
4. 滚动 Sharpe 曲线
5. 交易收益分布
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import (
    BARS_DIR, TARGET_DAILY_BARS, DMA_PRIMARY_CONFIG,
    COMMISSION_RATE, SLIPPAGE_POINTS, RESULTS_DIR, OUTPUT_DIR
)
from strategies.AL9999.backtest_utils import load_dollar_bars
from afmlkit.feature.core.ma import ewma, sma

# 输出目录
CHART_DIR = os.path.join(OUTPUT_DIR, 'figures', 'pnl')
os.makedirs(CHART_DIR, exist_ok=True)


def generate_signals(bars, config):
    """生成 DMA 信号 + reverse_signal exit 回测"""
    close = bars['close'].values.astype(np.float64)
    fast = config.get('fast_window', 3)
    slow = config.get('slow_window', 15)
    ma_type = config.get('ma_type', 'ewma')

    if ma_type == 'ewma':
        ma_fast = ewma(close, fast)
        ma_slow = ewma(close, slow)
    else:
        ma_fast = sma(close, fast)
        ma_slow = sma(close, slow)

    signal = np.where(ma_fast > ma_slow, 1, np.where(ma_fast < ma_slow, -1, 0))
    return signal, bars


def reverse_signal_backtest(signal_arr, bars, commission_rate, slippage_points):
    """
    reverse_signal 出场回测

    信号在 bar 收盘时产生，下一 bar 开盘成交。
    当信号方向翻转时平仓（反向信号出场）。
    """
    n = len(bars)
    trades = []
    position = 0
    entry_price = None
    entry_bar_idx = None

    for i in range(n):
        sig = signal_arr[i]
        close_price = bars.iloc[i]['close']
        open_price = bars.iloc[i]['open']

        # 检查出场：信号方向翻转
        if position != 0 and sig != 0 and sig * position < 0:
            # 下一根 bar 开盘出场
            exit_bar_idx = i + 1 if i + 1 < n else i
            exit_price = bars.iloc[exit_bar_idx]['open'] if exit_bar_idx < n else close_price
            gross_pnl = (exit_price - entry_price) * position
            cost = (entry_price + exit_price) * commission_rate + slippage_points * 2
            net_pnl = gross_pnl - cost

            trades.append({
                'entry_time': bars.index[entry_bar_idx],
                'exit_time': bars.index[exit_bar_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': position,
                'gross_pnl': gross_pnl,
                'cost': cost,
                'net_pnl': net_pnl,
            })
            position = 0
            entry_price = None

        # 检查入场：有信号且无仓位
        if position == 0 and sig != 0:
            entry_bar_idx = i + 1 if i + 1 < n else i
            fill_price = bars.iloc[entry_bar_idx]['open'] if entry_bar_idx < n else open_price
            position = sig
            entry_price = fill_price

    # 处理最后未平仓的仓位
    if position != 0:
        exit_bar_idx = n - 1
        exit_price = bars.iloc[exit_bar_idx]['close']
        gross_pnl = (exit_price - entry_price) * position
        cost = (entry_price + exit_price) * commission_rate + slippage_points * 2
        net_pnl = gross_pnl - cost
        trades.append({
            'entry_time': bars.index[entry_bar_idx],
            'exit_time': bars.index[exit_bar_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': position,
            'gross_pnl': gross_pnl,
            'cost': cost,
            'net_pnl': net_pnl,
        })

    return pd.DataFrame(trades)


def plot_cumulative_pnl(trades, full_bars, save_path):
    """图1：累计 PnL 曲线（标注 IS/OOS）"""
    fig, ax = plt.subplots(figsize=(14, 5))

    trades_sorted = trades.sort_values('exit_time')
    cum_pnl = trades_sorted['net_pnl'].cumsum()

    # IS/OOS 分割
    split_idx = int(len(full_bars) * 0.7)
    split_date = full_bars.index[split_idx]

    # 绘制 IS 和 OOS
    is_mask = trades_sorted['exit_time'] < split_date
    oos_mask = ~is_mask

    if is_mask.any():
        ax.plot(trades_sorted.loc[is_mask, 'exit_time'], cum_pnl[is_mask],
                color='#2196F3', linewidth=1.5, label='In-Sample (70%)')
    if oos_mask.any():
        # 延续 IS 最后一笔
        is_last_idx = is_mask[is_mask].index[-1] if is_mask.any() else None
        if is_last_idx is not None:
            ax.plot(trades_sorted.loc[is_mask, 'exit_time'], cum_pnl[is_mask],
                    color='#2196F3', linewidth=1.5, label='In-Sample (70%)')
            # OOS 从 IS 终点开始
            oos_cum = cum_pnl[oos_mask]
            oos_cum_adjusted = oos_cum
            ax.plot(trades_sorted.loc[oos_mask, 'exit_time'], oos_cum_adjusted,
                    color='#FF5722', linewidth=1.5, label='Out-of-Sample (30%)')
        else:
            ax.plot(trades_sorted.loc[oos_mask, 'exit_time'], cum_pnl[oos_mask],
                    color='#FF5722', linewidth=1.5, label='Out-of-Sample (30%)')

    ax.axvline(split_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'Cumulative PnL - AL9999 DMA(3,15)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative PnL (CNY)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    # 标注关键指标
    final_pnl = cum_pnl.iloc[-1]
    ax.text(0.02, 0.95, f'Final PnL: {final_pnl:,.0f} CNY',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 累计 PnL: {save_path}')


def plot_drawdown(trades, save_path):
    """图2：回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 4))

    cum_pnl = trades.sort_values('exit_time')['net_pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max

    ax.fill_between(drawdown.index, drawdown.values, 0, color='#FF5722', alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color='#FF5722', linewidth=1)

    max_dd = drawdown.min()
    ax.set_title(f'Max Drawdown: {max_dd:,.0f} CNY', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (CNY)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 回撤曲线: {save_path}')


def plot_monthly_heatmap(trades, save_path):
    """图3：月度收益热力图"""
    fig, ax = plt.subplots(figsize=(12, 5))

    df = trades.copy()
    df['year'] = df['exit_time'].dt.year
    df['month'] = df['exit_time'].dt.month
    monthly = df.groupby(['year', 'month'])['net_pnl'].sum().unstack()
    monthly.columns = [f'{m}月' for m in monthly.columns]

    im = ax.imshow(monthly.values, cmap='RdYlGn', aspect='auto', vmin=-5000, vmax=5000)
    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels(monthly.columns, fontsize=9)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index, fontsize=9)

    # 标注数值
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:,.0f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(val) > 3000 else 'black')

    ax.set_title('Monthly PnL Heatmap', fontsize=14)
    plt.colorbar(im, ax=ax, label='PnL (CNY)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 月度热力图: {save_path}')


def plot_rolling_sharpe(trades, save_path):
    """图4：滚动 Sharpe（60 笔交易窗口）"""
    fig, ax = plt.subplots(figsize=(14, 4))

    net_pnl = trades.sort_values('exit_time')['net_pnl']
    window = 60
    rolling_sharpe = net_pnl.rolling(window).mean() / net_pnl.rolling(window).std()

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='#4CAF50', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title(f'Rolling Sharpe (window={window})', fontsize=14)
    ax.set_xlabel('Trade Index')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 滚动 Sharpe: {save_path}')


def plot_trade_distribution(trades, save_path):
    """图5：交易收益分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    net_pnl = trades['net_pnl']

    # 直方图
    axes[0].hist(net_pnl[net_pnl > 0], bins=30, alpha=0.7, color='green', label='Wins')
    axes[0].hist(net_pnl[net_pnl <= 0], bins=30, alpha=0.7, color='red', label='Losses')
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].set_title('Trade PnL Distribution')
    axes[0].set_xlabel('PnL per Trade (CNY)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 箱线图（按月份）
    df = trades.copy()
    df['month'] = df['exit_time'].dt.to_period('M')
    monthly_pnl = df.groupby('month')['net_pnl']
    bp_data = [g.values for _, g in monthly_pnl if len(g) > 2]
    bp_labels = [str(k) for k, g in monthly_pnl if len(g) > 2]

    if bp_data:
        axes[1].boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False)
        axes[1].set_title('Monthly PnL Distribution')
        axes[1].set_xticklabels(bp_labels, rotation=45, fontsize=7)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)

    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 交易分布: {save_path}')


def main():
    print('[Step 8] PnL 图表生成')

    # 加载数据
    bars = load_dollar_bars()
    print(f'  数据: {len(bars)} bars, {bars.index[0].date()} ~ {bars.index[-1].date()}')

    # 生成信号
    signal_arr, _ = generate_signals(bars, DMA_PRIMARY_CONFIG)

    # 回测
    trades = reverse_signal_backtest(signal_arr, bars, COMMISSION_RATE, SLIPPAGE_POINTS)
    print(f'  交易数: {len(trades)}')

    if trades.empty:
        print('  ❌ 无交易数据')
        return 1

    # 保存交易记录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trades_path = os.path.join(RESULTS_DIR, 'trades.parquet')
    trades.to_parquet(trades_path, index=False)
    print(f'  交易记录: {trades_path}')

    # 生成图表
    plot_cumulative_pnl(trades, bars, os.path.join(CHART_DIR, 'cumulative_pnl.png'))
    plot_drawdown(trades, os.path.join(CHART_DIR, 'drawdown.png'))
    plot_monthly_heatmap(trades, os.path.join(CHART_DIR, 'monthly_heatmap.png'))
    plot_rolling_sharpe(trades, os.path.join(CHART_DIR, 'rolling_sharpe.png'))
    plot_trade_distribution(trades, os.path.join(CHART_DIR, 'trade_distribution.png'))

    # 打印关键指标
    net_pnl = trades['net_pnl']
    cum_pnl = net_pnl.cumsum()
    running_max = cum_pnl.cummax()
    max_dd = (cum_pnl - running_max).min()

    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl <= 0]

    print(f'\n  📊 关键指标:')
    print(f'    总 PnL: {net_pnl.sum():,.2f} CNY')
    print(f'    最大回撤: {max_dd:,.2f} CNY')
    print(f'    胜率: {len(wins)/len(net_pnl)*100:.1f}%')
    print(f'    平均盈利: {wins.mean():.2f} CNY')
    print(f'    平均亏损: {losses.mean():.2f} CNY')
    print(f'    盈亏比: {abs(wins.mean()/losses.mean()):.2f}')
    print(f'    Sharpe (每笔): {net_pnl.mean()/net_pnl.std():.3f}')

    print(f'\n  ✅ Step 8 完成，图表保存至: {CHART_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
