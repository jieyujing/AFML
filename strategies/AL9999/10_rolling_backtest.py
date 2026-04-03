"""
10_rolling_backtest.py - AL9999 真实单仓位滚动回测

修复原回测的严重问题：信号重叠导致收益重复计算

正确逻辑：
1. 无持仓时，新信号直接开仓
2. 持有多头时，新空头信号平多开空
3. 持有空头时，新多头信号平空开多
4. 同向信号忽略（或更新止损）
5. TBM 触发时平仓

输出：真实的回测绩效，而非虚高的收益
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    FEATURES_DIR, BARS_DIR, FIGURES_DIR,
    COMMISSION_RATE, SLIPPAGE_POINTS
)


def load_data():
    """加载所需数据"""
    # TBM 结果
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))

    # Meta Model
    meta_model = joblib.load(os.path.join(FEATURES_DIR.replace('features', 'models'), 'meta_model.pkl'))
    events_features = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))

    # Dollar Bars
    bars = pd.read_parquet(os.path.join(BARS_DIR, 'dollar_bars_target4.parquet'))

    return tbm, meta_model, events_features, bars


def get_meta_predictions(tbm, meta_model, events_features):
    """获取 Meta Model 预测"""
    common_idx = tbm.index.intersection(events_features.index)
    X = events_features.loc[common_idx]
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X = X[feature_cols].fillna(0)

    meta_pred = meta_model.predict(X)
    meta_proba = meta_model.predict_proba(X)[:, 1]

    result = tbm.loc[common_idx].copy()
    result['meta_pred'] = meta_pred
    result['meta_proba'] = meta_proba

    return result


def rolling_backtest(signals, bars, use_meta_filter=True):
    """
    单仓位滚动回测

    :param signals: 信号 DataFrame（含 side, touch_idx, meta_pred）
    :param bars: Dollar Bars DataFrame
    :param use_meta_filter: 是否使用 Meta 过滤
    :returns: 回测结果 DataFrame
    """
    # 过滤信号
    if use_meta_filter:
        signals = signals[signals['meta_pred'] == 1].copy()
        print(f"Meta 过滤后信号数: {len(signals)}")

    # 按时间排序
    signals = signals.sort_index()

    # 初始化状态
    position = 0  # 0=空仓, 1=多头, -1=空头
    entry_idx = None
    entry_price = None

    trades = []  # 交易记录

    for event_time, signal in signals.iterrows():
        event_idx = bars.index.get_loc(event_time) if event_time in bars.index else None
        if event_idx is None:
            continue

        signal_side = signal['side']
        touch_idx = signal['touch_idx']

        # 当前价格
        current_price = bars.iloc[event_idx]['close']

        # 检查是否有持仓需要平仓
        if position != 0:
            # 检查 TBM 触发点是否在当前事件之前
            # 如果 touch_idx < event_idx，说明持仓已平仓
            if touch_idx is not None and touch_idx < event_idx:
                # 持仓已平仓
                exit_price = bars.iloc[touch_idx]['close']
                pnl = (exit_price - entry_price) * position
                ret = np.log(exit_price / entry_price) * position

                trades.append({
                    'entry_time': bars.index[entry_idx],
                    'exit_time': bars.index[touch_idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'side': position,
                    'pnl': pnl,
                    'ret': ret,
                    'exit_reason': 'TBM_trigger'
                })

                position = 0
                entry_idx = None
                entry_price = None

        # 开新仓或更新仓位
        if position == 0:
            # 空仓，直接开仓
            position = signal_side
            entry_idx = event_idx
            entry_price = current_price

        elif position != signal_side:
            # 反向信号，平仓开反向
            exit_price = current_price
            pnl = (exit_price - entry_price) * position
            ret = np.log(exit_price / entry_price) * position

            trades.append({
                'entry_time': bars.index[entry_idx],
                'exit_time': event_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': position,
                'pnl': pnl,
                'ret': ret,
                'exit_reason': 'reverse_signal'
            })

            # 开反向仓
            position = signal_side
            entry_idx = event_idx
            entry_price = current_price

        # else: 同向信号，忽略

    # 平最后持仓
    if position != 0 and entry_idx is not None:
        # 找到最后一个信号的 touch_idx
        last_signal = signals.iloc[-1]
        touch_idx = last_signal['touch_idx']

        if touch_idx is not None and touch_idx < len(bars):
            exit_price = bars.iloc[touch_idx]['close']
        else:
            exit_price = bars.iloc[-1]['close']
            touch_idx = len(bars) - 1

        pnl = (exit_price - entry_price) * position
        ret = np.log(exit_price / entry_price) * position

        trades.append({
            'entry_time': bars.index[entry_idx],
            'exit_time': bars.index[touch_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': position,
            'pnl': pnl,
            'ret': ret,
            'exit_reason': 'final_close'
        })

    return pd.DataFrame(trades)


def calculate_performance(trades_df, bars, annualization_factor=1000):
    """计算绩效指标"""
    if len(trades_df) == 0:
        return {}

    # 计算净收益（扣除成本）
    trades_df['net_pnl'] = trades_df['pnl'] - \
        (trades_df['entry_price'] + trades_df['exit_price']) * COMMISSION_RATE - \
        SLIPPAGE_POINTS * 2

    trades_df['net_ret'] = trades_df['net_pnl'] / trades_df['entry_price']

    # 统计指标
    total_pnl = trades_df['net_pnl'].sum()
    total_ret = trades_df['net_ret'].sum()

    win_trades = trades_df[trades_df['net_pnl'] > 0]
    lose_trades = trades_df[trades_df['net_pnl'] < 0]

    n_trades = len(trades_df)
    n_wins = len(win_trades)
    n_losses = len(lose_trades)

    win_rate = n_wins / n_trades if n_trades > 0 else 0

    avg_win = win_trades['net_pnl'].mean() if len(win_trades) > 0 else 0
    avg_loss = lose_trades['net_pnl'].mean() if len(lose_trades) > 0 else 0

    profit_factor = abs(win_trades['net_pnl'].sum() / lose_trades['net_pnl'].sum()) if len(lose_trades) > 0 and lose_trades['net_pnl'].sum() != 0 else np.inf

    # Sharpe
    returns = trades_df['net_ret']
    sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor) if returns.std() > 0 else 0

    # 累积收益曲线
    cum_pnl = trades_df['net_pnl'].cumsum()

    # 最大回撤
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    mdd = drawdown.min()

    return {
        'n_trades': n_trades,
        'total_pnl': total_pnl,
        'total_ret': total_ret,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'mdd': mdd,
        'trades_df': trades_df
    }


def main():
    """主函数"""
    print("=" * 70)
    print("  AL9999 单仓位滚动回测（修正版）")
    print("=" * 70)

    # 加载数据
    print("\n[Step 1] 加载数据...")
    tbm, meta_model, events_features, bars = load_data()

    # 获取 Meta 预测
    print("\n[Step 2] 获取 Meta Model 预测...")
    signals = get_meta_predictions(tbm, meta_model, events_features)
    print(f"总信号数: {len(signals)}")

    # 运行滚动回测
    print("\n[Step 3] 运行单仓位滚动回测...")

    # Primary Model（无 Meta 过滤）
    print("\n--- Primary Model (无过滤) ---")
    primary_trades = rolling_backtest(signals, bars, use_meta_filter=False)
    primary_perf = calculate_performance(primary_trades, bars)

    print(f"交易次数: {primary_perf['n_trades']}")
    print(f"总收益: {primary_perf['total_pnl']:.2f} 点")
    print(f"胜率: {primary_perf['win_rate']*100:.1f}%")
    print(f"盈亏比: {primary_perf['profit_factor']:.2f}")
    print(f"年化 Sharpe: {primary_perf['sharpe']:.2f}")
    print(f"最大回撤: {primary_perf['mdd']:.2f} 点")

    # Combined Strategy（Meta 过滤）
    print("\n--- Combined Strategy (Meta 过滤) ---")
    combined_trades = rolling_backtest(signals, bars, use_meta_filter=True)
    combined_perf = calculate_performance(combined_trades, bars)

    print(f"交易次数: {combined_perf['n_trades']}")
    print(f"总收益: {combined_perf['total_pnl']:.2f} 点")
    print(f"胜率: {combined_perf['win_rate']*100:.1f}%")
    print(f"盈亏比: {combined_perf['profit_factor']:.2f}")
    print(f"年化 Sharpe: {combined_perf['sharpe']:.2f}")
    print(f"最大回撤: {combined_perf['mdd']:.2f} 点")

    # 对比原回测结果
    print("\n" + "=" * 70)
    print("  对比原回测逻辑")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    原回测 vs 修正后回测                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Primary Sharpe:     0.32 (原) → {primary_perf['sharpe']:.2f} (修正)              │")
    print(f"│  Combined Sharpe:   21.94 (原) → {combined_perf['sharpe']:.2f} (修正)              │")
    print(f"│  Primary 胜率:      48.3% (原) → {primary_perf['win_rate']*100:.1f}% (修正)              │")
    print(f"│  Combined 胜率:     75.3% (原) → {combined_perf['win_rate']*100:.1f}% (修正)              │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # 保存结果
    print("\n[Step 4] 保存结果...")
    primary_trades.to_parquet(os.path.join(FEATURES_DIR, 'rolling_primary_trades.parquet'))
    combined_trades.to_parquet(os.path.join(FEATURES_DIR, 'rolling_combined_trades.parquet'))

    print(f"✅ 已保存滚动回测结果")

    # 可视化
    print("\n[Step 5] 生成对比图...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 累积收益对比
    ax1 = axes[0]
    if len(primary_trades) > 0:
        ax1.plot(primary_trades['exit_time'], primary_trades['net_pnl'].cumsum(),
                 label='Primary Model', alpha=0.7, color='purple', lw=1.5)
    if len(combined_trades) > 0:
        ax1.plot(combined_trades['exit_time'], combined_trades['net_pnl'].cumsum(),
                 label='Combined Strategy', color='orange', lw=2)

    ax1.set_title('Rolling Backtest: Cumulative PnL (Single Position)', fontsize=14)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Net PnL (Points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 信号对比
    ax2 = axes[1]
    categories = ['Sharpe', 'Win Rate (%)', 'Profit Factor']
    primary_vals = [primary_perf['sharpe'], primary_perf['win_rate']*100, min(primary_perf['profit_factor'], 5)]
    combined_vals = [combined_perf['sharpe'], combined_perf['win_rate']*100, min(combined_perf['profit_factor'], 5)]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, primary_vals, width, label='Primary', color='purple', alpha=0.7)
    ax2.bar(x + width/2, combined_vals, width, label='Combined', color='orange')

    ax2.set_ylabel('Value')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '10_rolling_backtest.png'), dpi=150)
    print(f"✅ 图表已保存: {FIGURES_DIR}/10_rolling_backtest.png")
    plt.close()

    print("\n" + "=" * 70)
    print("  回测完成")
    print("=" * 70)


if __name__ == "__main__":
    main()