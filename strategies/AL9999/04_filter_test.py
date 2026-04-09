"""
04_filter_test.py - Step 4: Filter 机制测试

目标：测试机制明确的过滤器（只在 entry 前过滤，不影响 exit）
任务：
1. 加载 Step 3 结果（exit_comparison.parquet），获取选定的 exit 策略
2. 加载 Dollar Bars 和 DMA 信号
3. 在 entry 前应用 filter：
   - volatility_regime: 过滤极高/极低波动（避免异常行情）
   - trend_strength: 可选 ADX 过滤（需额外计算）
4. 比较 filtered vs unfiltered 的 OOS 表现
5. 判断：Filter 应改善真实质量（trade quality、Sharpe/Calmar），而非仅减少交易数
通过标准：
- 启用 filter 后 OOS Sharpe 提升或回撤下降
- Trade shrinkage 在接受范围内（不强制）
- Trade quality（平均收益 per trade）不下降
输出：filter_test.parquet (各 filter 方案指标)
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
    RESULTS_DIR, FILTER_CONFIG
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

def calculate_volatility(bars: pd.DataFrame, lookback: int = 20) -> pd.Series:
    returns = bars['close'].pct_change().fillna(0)
    vol = returns.rolling(lookback).std()
    return vol

def apply_volatility_filter(signals: pd.DataFrame, bars: pd.DataFrame, high_thr: float, low_thr: float, lookback: int) -> pd.DataFrame:
    """在 entry 时刻过滤：当前 bar 的标准化波动率超出阈值则不交易"""
    vol = calculate_volatility(bars, lookback)
    vol_z = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()  # 简单标准化

    # 保留的信号：波动率在正常范围
    valid_mask = (vol_z.loc[signals.index] >= low_thr) & (vol_z.loc[signals.index] <= high_thr)
    return signals[valid_mask].copy()

def simple_backtest(signals: pd.DataFrame, bars: pd.DataFrame, exit_type: str, exit_params: dict,
                   commission: float, slippage: float) -> dict:
    """同 Step 3 的简化回测"""
    bars = bars[['open', 'close']].copy()
    signals = signals.sort_index()
    trades = []
    position = 0
    entry_price = None
    entry_idx = None

    for i in range(len(signals)):
        ts = signals.index[i]
        side = signals.iloc[i]['side']
        close_price = bars.iloc[i]['close'] if i < len(bars) else close_price

        if position != 0:
            exit_triggered = False
            if exit_type == 'reverse_signal' and side * position < 0:
                exit_triggered = True
            elif exit_type == 'fixed_hold' and i - entry_idx >= exit_params.get('hold_bars', 20):
                exit_triggered = True
            if exit_triggered:
                exit_price = bars.iloc[i]['open'] if i < len(bars) else close_price
                pnl = (exit_price - entry_price) * position
                trades.append(pnl)
                position = 0
                entry_price = None

        if position == 0 and side != 0:
            fill_price = bars.iloc[i]['open'] if i+1 < len(bars) else close_price
            position = side
            entry_price = fill_price
            entry_idx = i

    if not trades:
        return {'trade_count': 0, 'total_pnl': 0, 'sharpe': 0, 'mean_trade': 0, 'max_dd': 0}
    pnl_series = pd.Series(trades)
    net_pnl = pnl_series.sum() - len(trades) * commission - slippage * len(trades)
    sharpe = (pnl_series.mean() / pnl_series.std()) if len(pnl_series) > 1 else 0.0
    cumsum = pnl_series.cumsum()
    max_dd = (cumsum - cumsum.cummax()).min()
    return {
        'trade_count': len(trades),
        'total_pnl': net_pnl,
        'sharpe': sharpe,
        'mean_trade': pnl_series.mean(),
        'max_dd': max_dd,
    }

def main():
    print(f"[Step 4] Filter 机制测试")

    # 加载 Step 2/3 结果
    primary_path = os.path.join(RESULTS_DIR, 'primary_rule.parquet')
    exit_path = os.path.join(RESULTS_DIR, 'exit_comparison.parquet')
    if not os.path.exists(primary_path) or not os.path.exists(exit_path):
        print(f"  ❌ 缺少前置结果，请先运行 Step 2 和 Step 3")
        return 1
    primary = pd.read_parquet(primary_path).iloc[0]
    fast, slow = int(primary['selected_fast']), int(primary['selected_slow'])
    exit_df = pd.read_parquet(exit_path)
    best_exit_row = exit_df.loc[exit_df['sharpe'].idxmax()]
    exit_type = best_exit_row['exit_type']
    print(f"  使用 Primary: DMA({fast},{slow}), Exit: {exit_type}")

    bars = load_dollar_bars()
    split_idx = int(len(bars) * 0.7)
    test_bars = bars.iloc[split_idx:]
    signals = generate_dma_signals(test_bars, fast, slow)

    # Baseline (no filter)
    baseline_res = simple_backtest(signals, test_bars, exit_type, {}, COMMISSION_RATE, SLIPPAGE_POINTS)

    # Filter: volatility_regime
    filter_cfg = FILTER_CONFIG.get('volatility', {})
    lookback = filter_cfg.get('lookback', 20)
    high_thr = filter_cfg.get('high_threshold', 1.5)
    low_thr = filter_cfg.get('low_threshold', 0.5)
    filtered_signals = apply_volatility_filter(signals, test_bars, high_thr, low_thr, lookback)

    filter_res = simple_backtest(filtered_signals, test_bars, exit_type, {}, COMMISSION_RATE, SLIPPAGE_POINTS)

    # 对比
    print(f"\n  过滤前:")
    print(f"    交易数: {baseline_res['trade_count']}, Sharpe: {baseline_res['sharpe']:.3f}, 均交易: {baseline_res['mean_trade']:.2f}, 回撤: {baseline_res['max_dd']:.2f}")
    print(f"  过滤后 (volatility_regime):")
    print(f"    交易数: {filter_res['trade_count']}, Sharpe: {filter_res['sharpe']:.3f}, 均交易: {filter_res['mean_trade']:.2f}, 回撤: {filter_res['max_dd']:.2f}")

    improve = (filter_res['sharpe'] > baseline_res['sharpe']) or (filter_res['max_dd'] > baseline_res['max_dd'])
    print(f"\n  Filter 效果: {'✅ 改善' if improve else '⚠️  无改善或恶化'}")

    out_df = pd.DataFrame([
        {'label': 'baseline', **baseline_res},
        {'label': 'volatility_filter', **filter_res},
    ])
    out_path = os.path.join(RESULTS_DIR, 'filter_test.parquet')
    out_df.to_parquet(out_path, index=False)
    print(f"\n  📁 结果保存: {out_path}")
    print(f"\n  ✅ Step 4 完成：Filter 测试结束")
    return 0

if __name__ == '__main__':
    sys.exit(main())
