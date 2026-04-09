"""
03_exit_comparison.py - Step 3: Exit 策略比较

目标：固定 entry（DMA），只比较 exit 策略效果
任务：
1. 加载 Step 2 结果（primary_rule.parquet），获取 fast/slow
2. 加载 Dollar Bars，生成 DMA 信号
3. 测试不同 exit 方案：
   - reverse_signal: 反向信号出场
   - fixed_hold: 固定持有 N 根 (default 20)
   - trailing_stop: ATR 追踪止损 (default 2x ATR)
4. 每个方案在 OOS 上跑回测
5. 比较标准：OOS 平均交易收益、回撤、趋势利润保持能力
输出：exit_comparison.parquet (各方案指标对比)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import (
    BARS_DIR, TARGET_DAILY_BARS, COMMISSION_RATE, SLIPPAGE_POINTS,
    RESULTS_DIR, EXIT_CONFIG
)
from strategies.AL9999.backtest_utils import load_dollar_bars
from afmlkit.feature.core.ma import ewma

def generate_dma_signals(bars: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    close = bars['close'].values.astype(np.float64)
    ema_fast = ewma(close, fast)
    ema_slow = ewma(close, slow)
    signal = np.where(ema_fast > ema_slow, 1, np.where(ema_fast < ema_slow, -1, 0))
    signals = pd.DataFrame({'side': signal}, index=bars.index)
    return signals[signals['side'] != 0].copy()

def calculate_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high = bars['high'].values if 'high' in bars.columns else bars['close'].values
    low = bars['low'].values if 'low' in bars.columns else bars['close'].values
    close = bars['close'].values
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr = pd.Series(tr, index=bars.index).rolling(period).mean()
    return atr

def backtest_with_exit(signals: pd.DataFrame, bars: pd.DataFrame, exit_type: str,
                      exit_params: dict, commission: float, slippage: float) -> dict:
    bars = bars[['open', 'high', 'low', 'close']].copy()
    signals = signals.sort_index()
    common_idx = signals.index.intersection(bars.index)
    signals = signals.loc[common_idx]

    trades = []
    position = 0
    entry_price = None
    entry_idx = None
    atr_series = calculate_atr(bars) if 'trailing_stop' in exit_type else None

    for i in range(len(signals)):
        ts = signals.index[i]
        side = signals.iloc[i]['side']
        close_price = bars.iloc[i]['close']

        # 出场逻辑
        if position != 0:
            exit_triggered = False
            exit_price = None

            if exit_type == 'reverse_signal':
                if side * position < 0:
                    exit_triggered = True
                    exit_price = bars.iloc[i]['open']
            elif exit_type == 'fixed_hold':
                if i - entry_idx >= exit_params.get('hold_bars', 20):
                    exit_triggered = True
                    exit_price = bars.iloc[i]['open']
            elif exit_type == 'trailing_stop':
                if atr_series is not None:
                    current_atr = atr_series.iloc[i]
                    if not np.isnan(current_atr) and current_atr > 0:
                        peak_after_entry = bars.iloc[entry_idx:i+1]['high' if position == 1 else 'low'].max() if position == 1 else bars.iloc[entry_idx:i+1]['low'].min()
                        if position == 1 and close_price < peak_after_entry - exit_params.get('atr_multiplier', 2.0) * current_atr:
                            exit_triggered = True
                            exit_price = bars.iloc[i]['open']
                        if position == -1 and close_price > peak_after_entry + exit_params.get('atr_multiplier', 2.0) * current_atr:
                            exit_triggered = True
                            exit_price = bars.iloc[i]['open']
            elif exit_type == 'time_based':
                if i - entry_idx >= exit_params.get('max_hold_bars', 60):
                    exit_triggered = True
                    exit_price = bars.iloc[i]['open']

            # 最后一根强制出场
            if i == len(signals) - 1 and position != 0:
                exit_triggered = True
                exit_price = bars.iloc[i]['close']

            if exit_triggered and exit_price is not None:
                pnl = (exit_price - entry_price) * position
                trades.append({'pnl': pnl, 'exit_type': exit_type})
                position = 0
                entry_price = None
                entry_idx = None

        # 入场
        if position == 0 and side != 0:
            fill_price = bars.iloc[i]['open'] if i+1 < len(bars) else close_price
            position = side
            entry_price = fill_price
            entry_idx = i

    if not trades:
        return {'trade_count': 0, 'total_pnl': 0, 'sharpe': 0, 'mean_trade': 0}
    pnl_series = pd.Series([t['pnl'] for t in trades])
    net_pnl = pnl_series.sum() - len(trades) * commission - slippage * len(trades)
    sharpe = (pnl_series.mean() / pnl_series.std()) if len(pnl_series) > 1 else 0.0
    return {
        'trade_count': len(trades),
        'total_pnl': net_pnl,
        'sharpe': sharpe,
        'mean_trade': pnl_series.mean(),
    }

def main():
    print(f"[Step 3] Exit 策略比较（固定 DMA entry）")

    # 加载 Step 2 结果
    primary_path = os.path.join(RESULTS_DIR, 'primary_rule.parquet')
    if not os.path.exists(primary_path):
        print(f"  ❌ 未找到 Step 2 结果: {primary_path}，请先运行 Step 2")
        return 1
    primary = pd.read_parquet(primary_path).iloc[0]
    fast, slow = int(primary['selected_fast']), int(primary['selected_slow'])
    print(f"  使用 Primary Rule: DMA({fast}, {slow})")

    # 加载数据
    bars = load_dollar_bars()
    split_idx = int(len(bars) * 0.7)
    test_bars = bars.iloc[split_idx:]  # 只在 OOS 测试
    signals = generate_dma_signals(test_bars, fast, slow)

    # 测试不同 exit 方案
    exit_configs = [
        ('reverse_signal', {'type': 'reverse_signal'}),
        ('fixed_hold', {'type': 'fixed_hold', 'hold_bars': 20}),
        ('trailing_stop', {'type': 'trailing_stop', 'atr_multiplier': 2.0}),
        ('time_based', {'type': 'time_based', 'max_hold_bars': 60}),
    ]

    results = []
    for name, config in exit_configs:
        res = backtest_with_exit(signals, test_bars, config['type'], config, COMMISSION_RATE, SLIPPAGE_POINTS)
        results.append({
            'exit_type': name,
            'trade_count': res['trade_count'],
            'total_pnl': res['total_pnl'],
            'sharpe': res['sharpe'],
            'mean_trade': res['mean_trade'],
        })

    df = pd.DataFrame(results)
    print(f"\n  Exit 策略对比（OOS）:")
    print(df.to_string(index=False))

    # 选择最佳 exit（优先 Sharpe > 0 且 mean_trade 最高）
    valid = df[df['sharpe'] > 0]
    if len(valid) == 0:
        print(f"\n  ❌ 所有 exit 方案 OOS Sharpe <= 0，考虑回退到 Step 2")
        return 1
    best_exit = valid.loc[valid['mean_trade'].idxmax()]
    print(f"\n  首选 Exit: {best_exit['exit_type']} (mean_trade={best_exit['mean_trade']:.2f}, sharpe={best_exit['sharpe']:.3f})")

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'exit_comparison.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\n  📁 结果保存: {out_path}")
    print(f"\n  ✅ Step 3 完成：Exit 策略已选定（{best_exit['exit_type']}）")
    return 0

if __name__ == '__main__':
    sys.exit(main())
