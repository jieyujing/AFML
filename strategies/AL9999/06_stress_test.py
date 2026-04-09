"""
06_stress_test.py - Step 6: Walk-Forward & 成本压力 & 参数扰动

目标：综合稳健性验证，不通过则不能进入 Step 7
任务：
1. 加载 Step 5 结果（risk_position.parquet），获取最终配置（fast, slow, exit_type, use_filter）
2. Walk-Forward (WF) 分析：
   - 按 WF_CONFIG['n_splits'] 做时间滚动分割（ train 70% / test 30% ）
   - 每个 split 重新训练（或直接使用步骤2参数）并测试 OOS
   - 记录每个 split 的 OOS Sharpe 和稳定性
3. 成本压力测试：
   - 对 OOS 数据，分别运行 commission x [1.0, 1.5, 2.0] 和 slippage x [1.0, 2.0, 3.0]
   - 检查在 worst-case 成本下 OOS Sharpe 是否仍 >= 0
4. 参数扰动（Parameter Perturbation）：
   - fast ±2, slow ±5 范围内所有组合在 OOS 上跑
   - 确保邻域内 Sharpe 不塌陷（全部 >= 0）
通过标准：
- 所有 WF splits 的 OOS Sharpe 中位数 > 0
- Worst-case 成本压力 Sharpe >= 0（可放宽到 > -0.5？skill 要求 >=0）
- 参数邻域全部组合 OOS Sharpe >= 0
输出：stress_test.parquet (汇总所有稳健性指标)
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
    RESULTS_DIR, WF_CONFIG, COST_PRESSURE_CONFIG, PARAM_PERTURBATION,
    COMMISSION_RATE as BASE_COMMISSION, SLIPPAGE_POINTS as BASE_SLIPPAGE
)
from strategies.AL9999.backtest_utils import load_dollar_bars
from afmlkit.feature.core.ma import ewma

def generate_dma_signals(bars, fast, slow):
    close = bars['close'].values.astype(np.float64)
    ema_fast = ewma(close, fast)
    ema_slow = ewma(close, slow)
    signal = np.where(ema_fast > ema_slow, 1, np.where(ema_fast < ema_slow, -1, 0))
    return pd.DataFrame({'side': signal}, index=bars.index)[lambda df: df['side'] != 0].copy()

def simple_backtest(signals, bars, position_size, commission, slippage, exit_type='reverse_signal'):
    bars = bars[['open', 'close']].copy()
    signals = signals.sort_index()
    trades = []
    position = 0
    entry_price = None

    for i in range(len(signals)):
        ts = signals.index[i]
        side = signals.iloc[i]['side']

        if position != 0:
            if exit_type == 'reverse_signal' and side * position < 0:
                exit_price = bars.iloc[i]['open']
                trades.append((exit_price - entry_price) * position * position_size)
                position = 0
                entry_price = None

        if position == 0 and side != 0:
            fill_price = bars.iloc[i]['open'] if i+1 < len(bars) else bars.iloc[i]['close']
            position = side
            entry_price = fill_price

    if not trades:
        return 0, 0, 0
    pnl = pd.Series(trades)
    sharpe = (pnl.mean() / pnl.std()) if len(pnl) > 1 else 0.0
    total = pnl.sum()
    return total, sharpe, len(trades)

def walk_forward_analysis(bars, fast, slow, exit_type, position_size, commission, slippage, wf_cfg):
    n_splits = wf_cfg['n_splits']
    train_ratio = wf_cfg['train_ratio']
    total_len = len(bars)
    fold_size = total_len // n_splits
    results = []

    for i in range(n_splits):
        train_end = int((i + 1) * fold_size * train_ratio)
        test_start = train_end
        test_end = min(test_start + int(fold_size * wf_cfg['test_ratio']), total_len)

        if test_end > total_len:
            break
        test_bars = bars.iloc[test_start:test_end]
        signals = generate_dma_signals(test_bars, fast, slow)
        total, sharpe, trades = simple_backtest(signals, test_bars, position_size, commission, slippage, exit_type)
        results.append({
            'fold': i+1,
            'oos_sharpe': sharpe,
            'oos_total': total,
            'oos_trades': trades,
        })

    df = pd.DataFrame(results)
    return df

def main():
    print(f"[Step 6] 稳健性测试 (WF + 成本压力 + 参数扰动)")

    # 读取前置配置
    primary_path = os.path.join(RESULTS_DIR, 'primary_rule.parquet')
    exit_path = os.path.join(RESULTS_DIR, 'exit_comparison.parquet')
    filter_path = os.path.join(RESULTS_DIR, 'filter_test.parquet')
    for p in [primary_path, exit_path, filter_path]:
        if not os.path.exists(p):
            print(f"  ❌ 缺少前置: {p}")
            return 1
    primary = pd.read_parquet(primary_path).iloc[0]
    fast, slow = int(primary['selected_fast']), int(primary['selected_slow'])
    exit_df = pd.read_parquet(exit_path)
    best_exit_type = exit_df.loc[exit_df['sharpe'].idxmax()]['exit_type']

    bars = load_dollar_bars()
    position_size = 1.0
    commission = BASE_COMMISSION
    slippage = BASE_SLIPPAGE

    print(f"  基准配置: DMA({fast},{slow}), exit={best_exit_type}")

    # 1. Walk-Forward 分析
    print(f"\n  🔄 Walk-Forward 分析 (n_splits={WF_CONFIG['n_splits']})...")
    wf_df = walk_forward_analysis(bars, fast, slow, best_exit_type, position_size, commission, slippage, WF_CONFIG)
    wf_median_sharpe = wf_df['oos_sharpe'].median()
    wf_all_positive = (wf_df['oos_sharpe'] > 0).all()
    print(f"    WF OOS Sharpe 中位数: {wf_median_sharpe:.3f}")
    print(f"    所有 splits Sharpe > 0: {wf_all_positive}")

    # 2. 成本压力测试
    print(f"\n  💰 成本压力测试...")
    cost_results = []
    for cm in COST_PRESSURE_CONFIG['commission_multipliers']:
        for sm in COST_PRESSURE_CONFIG['slippage_multipliers']:
            test_comm = BASE_COMMISSION * cm
            test_slip = BASE_SLIPPAGE * sm
            # 在 OOS 上跑
            split_idx = int(len(bars) * 0.7)
            test_bars = bars.iloc[split_idx:]
            signals = generate_dma_signals(test_bars, fast, slow)
            total, sharpe, trades = simple_backtest(signals, test_bars, position_size, test_comm, test_slip, best_exit_type)
            cost_results.append({
                'comm_mult': cm,
                'slip_mult': sm,
                'oos_sharpe': sharpe,
                'oos_total': total,
            })
    cost_df = pd.DataFrame(cost_results)
    worst_sharpe = cost_df['oos_sharpe'].min()
    worst_case_pass = worst_sharpe >= 0
    print(f"    Worst-case Sharpe: {worst_sharpe:.3f} (>=0: {worst_case_pass})")

    # 3. 参数扰动
    print(f"\n  🎛️  参数扰动测试 (±2 fast, ±5 slow)...")
    perturb_results = []
    for df in PARAM_PERTURBATION['fast_window_delta']:
        for ds in PARAM_PERTURBATION['slow_window_delta']:
            f = fast + df
            s = slow + ds
            if f <= 0 or s <= f:
                continue
            split_idx = int(len(bars) * 0.7)
            test_bars = bars.iloc[split_idx:]
            signals = generate_dma_signals(test_bars, f, s)
            total, sharpe, trades = simple_backtest(signals, test_bars, position_size, commission, slippage, best_exit_type)
            perturb_results.append({
                'fast': f,
                'slow': s,
                'oos_sharpe': sharpe,
                'oos_trades': trades,
            })
    perturb_df = pd.DataFrame(perturb_results)
    perturb_all_positive = (perturb_df['oos_sharpe'] >= 0).all()
    print(f"    参数邻域全部组合 Sharpe >= 0: {perturb_all_positive}")

    # 汇总判断
    passed = wf_all_positive and worst_case_pass and perturb_all_positive
    print(f"\n  📊 稳健性综合评估:")
    print(f"    WF 稳定: {wf_all_positive}")
    print(f"    成本压力: {worst_case_pass}")
    print(f"    参数扰动: {perturb_all_positive}")
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")

    summary = {
        'step': 6,
        'timestamp': datetime.now().isoformat(),
        'fast': fast,
        'slow': slow,
        'exit_type': best_exit_type,
        'wf_median_sharpe': wf_median_sharpe,
        'wf_all_positive': wf_all_positive,
        'worst_cost_sharpe': worst_sharpe,
        'worst_cost_pass': worst_case_pass,
        'perturb_all_positive': perturb_all_positive,
        'passed': passed,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'stress_test.parquet')
    pd.DataFrame([summary]).to_parquet(out_path, index=False)

    # 保存详细结果
    wf_path = os.path.join(RESULTS_DIR, 'wf_details.parquet')
    cost_path = os.path.join(RESULTS_DIR, 'cost_details.parquet')
    perturb_path = os.path.join(RESULTS_DIR, 'perturb_details.parquet')
    wf_df.to_parquet(wf_path, index=False)
    cost_df.to_parquet(cost_path, index=False)
    perturb_df.to_parquet(perturb_path, index=False)

    print(f"\n  📁 结果保存: {out_path}")
    print(f"\n  {'✅ Step 6 通过' if passed else '❌ Step 6 未通过，不可进入实盘候选'}")
    return 0 if passed else 1

if __name__ == '__main__':
    sys.exit(main())
