"""
01b_verify_edge.py - Step 1: 基础趋势 Edge 验证

目标：验证在指定品种、bar 频率下，DMA 趋势跟踪是否存在基础 edge
任务：
1. 加载 Dollar Bars 数据
2. 在全样本上运行 DMA（fast=5, slow=20）生成信号
3. 运行简单回测（固定仓位 1，反向信号出场）
4. 按 70/30 切分 OOS
5. 检查：OOS 收益为正、trade count >= 30、Sharpe > 0
通过标准：
- OOS 净收益 > 0（成本后）
- OOS Sharpe >= 0
- Full 和 OOS 都至少有 30 笔交易
- 参数邻域稳定（fast=4-6, slow=18-22）OOS 不败
若不通过，返回 Step 0 重新审视数据边界（换 bar、换周期、换品种）
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import (
    BARS_DIR, TARGET_DAILY_BARS, DMA_PRIMARY_CONFIG,
    COMMISSION_RATE, SLIPPAGE_POINTS, PASS_CRITERIA, OUTPUT_DIR, RESULTS_DIR
)
from strategies.AL9999.backtest_utils import load_dollar_bars
from afmlkit.feature.core.ma import ewma, sma

def generate_dma_signals(bars: pd.DataFrame, config: dict) -> pd.DataFrame:
    """生成 DMA 信号（简化版，无 TBM）"""
    close = bars['close'].values.astype(np.float64)
    fast_window = config.get('fast_window', 5)
    slow_window = config.get('slow_window', 20)
    ma_type = config.get('ma_type', 'ewma')

    if ma_type == 'ewma':
        ema_fast = ewma(close, fast_window)
        ema_slow = ewma(close, slow_window)
    else:
        ema_fast = sma(close, fast_window)
        ema_slow = sma(close, slow_window)

    # 生成 side 信号：快线上穿慢线→多头，下穿→空头
    # 信号在 bar 的收盘时刻触发，下一 bar 开盘成交
    signal = np.where(ema_fast > ema_slow, 1, np.where(ema_fast < ema_slow, -1, 0))
    signal_series = pd.Series(signal, index=bars.index, name='side')

    # 构建 signals DataFrame，包含 side 和 touch_idx（出场触发点）
    # 简单反向信号出场：当 side 翻转时，在翻转 bar 的下一根出场
    signals = pd.DataFrame(index=bars.index)
    signals['side'] = signal_series
    signals['touch_idx'] = np.nan  # 暂时不用

    # 只保留非零信号点
    signals = signals[signals['side'] != 0].copy()
    return signals

def simple_vector_backtest(signals: pd.DataFrame, bars: pd.DataFrame, commission: float, slippage: float) -> dict:
    """简化向量化回测：固定仓位、反向信号出场、成本扣除"""
    bars = bars[['open', 'high', 'low', 'close']].copy()
    signals = signals.sort_index()

    # 对齐
    common_idx = signals.index.intersection(bars.index)
    signals = signals.loc[common_idx]
    bars = bars.loc[common_idx]

    # 初始化交易记录
    trades = []
    position = 0
    entry_idx = None
    entry_price = None

    for i in range(len(signals)):
        ts = signals.index[i]
        side = signals.iloc[i]['side']
        close_price = bars.iloc[i]['close']

        # 检查是否需要出场（反向信号或无数据时）
        if position != 0:
            # 出场条件：当前 side 与持仓方向相反，或到达最后一根
            reversal = (side * position < 0)
            if reversal or i == len(signals) - 1:
                exit_price = bars.iloc[i]['open'] if i < len(bars) else close_price
                pnl = (exit_price - entry_price) * position
                ret = np.log(exit_price / entry_price) * position
                trades.append({
                    'entry_time': signals.index[entry_idx],
                    'exit_time': ts,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': ret,
                })
                position = 0
                entry_idx = None
                entry_price = None

        # 入场（无持仓且信号明确）
        if position == 0 and side != 0:
            fill_price = bars.iloc[i]['open'] if i+1 < len(bars) else close_price
            position = side
            entry_idx = i
            entry_price = fill_price

    # 计算统计
    if not trades:
        return {'error': 'no_trades'}

    trades_df = pd.DataFrame(trades)
    net_pnl = trades_df['pnl'].sum() - len(trades_df) * commission * bars['close'].iloc[0] - slippage * len(trades_df)
    trades_df['net_pnl'] = trades_df['pnl'] - commission - slippage
    sharpe = (trades_df['net_pnl'].mean() / trades_df['net_pnl'].std()) if len(trades_df) > 1 else 0.0
    trade_count = len(trades_df)

    return {
        'trade_count': trade_count,
        'total_pnl': net_pnl,
        'sharpe': sharpe,
        'mean_trade': trades_df['net_pnl'].mean(),
        'trades': trades_df,
    }

def main():
    print(f"[Step 1] 基础趋势 Edge 验证")
    print(f"  加载 Dollar Bars...")
    bars = load_dollar_bars()
    print(f"  数据范围: {bars.index[0]} ~ {bars.index[-1]}, 共 {len(bars)} bars")

    # OOS 切分
    split_idx = int(len(bars) * 0.7)
    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    results = {}
    for name, sub_bars in [('Full', bars), ('OOS', test_bars)]:
        signals = generate_dma_signals(sub_bars, DMA_PRIMARY_CONFIG)
        res = simple_vector_backtest(signals, sub_bars, COMMISSION_RATE, SLIPPAGE_POINTS)
        results[name] = res
        print(f"\n  {name} 结果:")
        if 'error' in res:
            print(f"    ❌ 无交易")
        else:
            print(f"    交易数: {res['trade_count']}")
            print(f"    净收益: {res['total_pnl']:.2f}")
            print(f"    Sharpe: {res['sharpe']:.3f}")
            print(f"    均交易: {res['mean_trade']:.2f}")

    # 通过判断
    oos_res = results.get('OOS', {})
    full_res = results.get('Full', {})

    passed = True
    if oos_res.get('trade_count', 0) < PASS_CRITERIA['trade_count_min']:
        print(f"\n  ❌ 交易数不足: OOS={oos_res.get('trade_count')} < {PASS_CRITERIA['trade_count_min']}")
        passed = False
    if oos_res.get('sharpe', -1) < PASS_CRITERIA['oos_sharpe_min']:
        print(f"\n  ❌ OOS Sharpe 为负或不足: {oos_res.get('sharpe')} < {PASS_CRITERIA['oos_sharpe_min']}")
        passed = False

    # 参数邻域稳定性（fast ±2, slow ±5）
    print(f"\n  🔍 参数邻域稳定性检查...")
    stability_ok = True
    for df in [-2, -1, 0, 1, 2]:
        for ds in [-5, -2, 0, 2, 5]:
            cfg = {'fast_window': 5 + df, 'slow_window': 20 + ds, 'ma_type': 'ewma'}
            sig = generate_dma_signals(test_bars, cfg)
            res = simple_vector_backtest(sig, test_bars, COMMISSION_RATE, SLIPPAGE_POINTS)
            if res.get('sharpe', -1) < 0:
                stability_ok = False
                break
        if not stability_ok:
            break

    if not stability_ok:
        print(f"  ⚠️  参数邻域存在负 Sharpe，稳定性不足")
        passed = False
    else:
        print(f"  ✅ 参数邻域稳定（所有组合 OOS Sharpe >= 0）")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'edge_verification.parquet')
    summary = {
        'step': 1,
        'timestamp': datetime.now().isoformat(),
        'full_trade_count': full_res.get('trade_count', 0),
        'full_sharpe': full_res.get('sharpe', np.nan),
        'full_total_pnl': full_res.get('total_pnl', np.nan),
        'oos_trade_count': oos_res.get('trade_count', 0),
        'oos_sharpe': oos_res.get('sharpe', np.nan),
        'oos_total_pnl': oos_res.get('total_pnl', np.nan),
        'passed': passed,
        'stability_ok': stability_ok,
    }
    pd.DataFrame([summary]).to_parquet(results_path, index=False)
    print(f"\n  📁 结果保存: {results_path}")

    if passed:
        print(f"\n  ✅ Step 1 通过：基础 Edge 存在，可进入 Step 2")
        return 0
    else:
        print(f"\n  ❌ Step 1 未通过：建议回退到 Step 0 重新审视数据边界（换品种、换 bar、换周期）")
        return 1

if __name__ == '__main__':
    sys.exit(main())
