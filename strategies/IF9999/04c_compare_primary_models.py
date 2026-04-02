"""
04c_compare_primary_models.py - MA vs SuperTrend Primary Model 对比

使用最优 TBM 参数 (止损 1.5, 止盈 2.5) 对比两种 Primary Model：
- MA (EWMA span=20)
- SuperTrend (多参数融合)

评估指标：
- Sharpe Ratio
- 总收益
- 胜率
- DSR
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    BARS_DIR, FEATURES_DIR,
    MA_PRIMARY_MODEL, SUPERTREND_CONFIG
)

from afmlkit.feature.core.ma import ewma
from afmlkit.feature.core.trend import supertrend
from afmlkit.label.tbm import triple_barrier


# ============================================================
# 数据加载
# ============================================================

def load_data():
    """加载所有需要的数据。"""
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = pd.read_parquet(bars_path)

    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = pd.read_parquet(events_path)
    if 'timestamp' in events.columns:
        events = events.set_index('timestamp')

    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    events_features = pd.read_parquet(features_path)

    return bars, events, events_features


# ============================================================
# MA 信号生成
# ============================================================

def generate_ma_signals(bars, events):
    """生成 MA Primary Model 信号。"""
    close = bars['close'].values.astype(np.float64)
    span = MA_PRIMARY_MODEL.get('span', 20)
    ma_vals = ewma(close, span)

    signals = events.copy()
    event_indices = [bars.index.get_loc(ts) for ts in events.index]
    signals['idx'] = event_indices
    signals['side'] = np.sign(signals['price'] - ma_vals[event_indices])
    signals['side'] = signals['side'].replace(0, 1).astype(int)

    return signals


# ============================================================
# SuperTrend 信号生成
# ============================================================

def generate_supertrend_signals(bars, events):
    """生成 SuperTrend Primary Model 信号。"""
    high = bars['high'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    close = bars['close'].values.astype(np.float64)

    params_combinations = SUPERTREND_CONFIG.get('params_combinations', [
        {'period': 10, 'multiplier': 3.0, 'name': 'default'}
    ])
    fusion_method = SUPERTREND_CONFIG.get('fusion_method', 'or')

    direction_maps = {}
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

    event_indices = [bars.index.get_loc(ts) for ts in events.index]
    signals = events.copy()
    signals['idx'] = event_indices

    directions_at_events = []
    for params in params_combinations:
        name = params['name']
        direction = direction_maps[name]
        dir_at_events = direction[event_indices]
        signals[f'dir_{name}'] = dir_at_events
        directions_at_events.append(dir_at_events)

    directions_stack = np.vstack(directions_at_events)

    if fusion_method == 'or':
        n_long_signals = np.sum(directions_stack == 1, axis=0)
        side = np.where(n_long_signals >= 1, 1, -1).astype(int)
    elif fusion_method == 'and':
        n_long_signals = np.sum(directions_stack == 1, axis=0)
        side = np.where(n_long_signals == len(params_combinations), 1, -1).astype(int)
    else:
        side = directions_stack[0].astype(int)

    signals['side'] = side
    return signals


# ============================================================
# TBM 计算
# ============================================================

def compute_tbm(bars, signals, events_features, config):
    """使用指定参数计算 TBM。"""
    timestamps = bars.index.values.astype(np.int64)
    close = bars['close'].values.astype(np.float64)

    common_idx = signals.index.intersection(events_features.index)
    signals_aligned = signals.loc[common_idx]

    event_idxs = signals_aligned['idx'].values.astype(np.int64)
    targets = events_features.loc[common_idx, config['target_ret_col']].values.astype(np.float64)

    max_end_idx = len(bars) - 1
    valid_mask = event_idxs + config['vertical_barrier_bars'] < max_end_idx

    event_idxs = event_idxs[valid_mask]
    targets = targets[valid_mask]
    signals_valid = signals_aligned[valid_mask]

    bar_duration_sec = 4 * 3600
    vertical_barrier_sec = config['vertical_barrier_bars'] * bar_duration_sec

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps=timestamps,
        close=close,
        event_idxs=event_idxs,
        targets=targets,
        horizontal_barriers=config['horizontal_barriers'],
        vertical_barrier=vertical_barrier_sec,
        min_close_time_sec=config['min_close_time_sec'],
        side=signals_valid['side'].values.astype(np.int8),
        min_ret=config['min_ret']
    )

    valid_touch_mask = touch_idxs != -1
    exit_prices = np.full(len(touch_idxs), np.nan)
    exit_prices[valid_touch_mask] = close[touch_idxs[valid_touch_mask]]
    pnl = signals_valid['side'].values * (exit_prices - signals_valid['price'].values)

    tbm_df = pd.DataFrame({
        'ret': rets,
        'pnl': pnl,
        'side': signals_valid['side'].values,
    }, index=signals_valid.index)

    return tbm_df


# ============================================================
# 指标计算
# ============================================================

def calculate_metrics(tbm_df, annualization_factor=1500, n_trials=20):
    """计算所有评估指标。"""
    returns = tbm_df['ret']

    # Sharpe
    sr_period = returns.mean() / returns.std() if returns.std() > 0 else 0
    sharpe = sr_period * np.sqrt(annualization_factor)

    # PSR
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3
    sigma_sr = np.sqrt((1 - skew * sr_period + (kurt - 1) / 4 * sr_period**2) / (n - 1))
    psr = norm.cdf(sr_period / sigma_sr) if sigma_sr > 0 else 0.5

    # DSR
    sr_std = 0.5
    gamma = 0.5772
    expected_max_sr = sr_std * (
        (1 - gamma) * norm.ppf(1 - 1/n_trials) +
        gamma * norm.ppf(1 - 1/(n_trials * np.exp(-1)))
    )
    benchmark_sr = expected_max_sr / np.sqrt(annualization_factor)
    dsr = norm.cdf((sr_period - benchmark_sr) / sigma_sr) if sigma_sr > 0 else 0.5

    return {
        'n_samples': len(tbm_df),
        'sharpe': sharpe,
        'total_pnl': tbm_df['pnl'].sum(),
        'win_rate': (tbm_df['pnl'] > 0).mean(),
        'avg_pnl': tbm_df['pnl'].mean(),
        'psr': psr,
        'dsr': dsr,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  MA vs SuperTrend Primary Model 对比 (TBM)")
    print("=" * 70)

    # 加载数据
    bars, events, events_features = load_data()

    # 最优 TBM 参数
    tbm_config = {
        'target_ret_col': 'feat_ewm_vol_20',
        'horizontal_barriers': (1.5, 2.5),  # 最优参数
        'vertical_barrier_bars': 50,
        'min_ret': 0.002,
        'min_close_time_sec': 60,
    }

    print("\nTBM 参数: 止损=1.5, 止盈=2.5, 垂直屏障=50 bars")

    # MA Primary Model
    print("\n" + "-" * 70)
    print("  [1] MA Primary Model (EWMA span=20)")
    print("-" * 70)

    ma_signals = generate_ma_signals(bars, events)
    ma_tbm = compute_tbm(bars, ma_signals, events_features, tbm_config)
    ma_metrics = calculate_metrics(ma_tbm)

    print(f"样本数: {ma_metrics['n_samples']}")
    print(f"年化 Sharpe: {ma_metrics['sharpe']:.4f}")
    print(f"总收益: {ma_metrics['total_pnl']:.2f} 点")
    print(f"胜率: {ma_metrics['win_rate']*100:.1f}%")
    print(f"平均收益: {ma_metrics['avg_pnl']:.2f} 点")
    print(f"PSR: {ma_metrics['psr']:.4f}")
    print(f"DSR: {ma_metrics['dsr']:.4f}")

    # SuperTrend Primary Model
    print("\n" + "-" * 70)
    print("  [2] SuperTrend Primary Model (多参数融合)")
    print("-" * 70)

    st_signals = generate_supertrend_signals(bars, events)
    st_tbm = compute_tbm(bars, st_signals, events_features, tbm_config)
    st_metrics = calculate_metrics(st_tbm)

    print(f"样本数: {st_metrics['n_samples']}")
    print(f"年化 Sharpe: {st_metrics['sharpe']:.4f}")
    print(f"总收益: {st_metrics['total_pnl']:.2f} 点")
    print(f"胜率: {st_metrics['win_rate']*100:.1f}%")
    print(f"平均收益: {st_metrics['avg_pnl']:.2f} 点")
    print(f"PSR: {st_metrics['psr']:.4f}")
    print(f"DSR: {st_metrics['dsr']:.4f}")

    # 对比结果
    print("\n" + "=" * 70)
    print("  对比结果")
    print("=" * 70)

    comparison = pd.DataFrame({
        'MA': ma_metrics,
        'SuperTrend': st_metrics,
    }).T

    comparison['winner'] = comparison.apply(
        lambda row: 'MA' if row['sharpe'] == max(ma_metrics['sharpe'], st_metrics['sharpe']) else 'SuperTrend',
        axis=1
    )

    print(f"\n{'指标':<15} {'MA':<15} {'SuperTrend':<15} {'优势':<10}")
    print("-" * 55)
    for metric in ['sharpe', 'total_pnl', 'win_rate', 'psr', 'dsr']:
        ma_val = ma_metrics[metric]
        st_val = st_metrics[metric]
        winner = 'MA' if ma_val > st_val else 'ST' if st_val > ma_val else '平'
        print(f"{metric:<15} {ma_val:<15.4f} {st_val:<15.4f} {winner:<10}")

    # 最终结论
    print("\n" + "=" * 70)
    if ma_metrics['dsr'] > st_metrics['dsr']:
        print("  结论: MA Primary Model 更优")
    elif st_metrics['dsr'] > ma_metrics['dsr']:
        print("  结论: SuperTrend Primary Model 更优")
    else:
        print("  结论: 两者相当")

    if max(ma_metrics['dsr'], st_metrics['dsr']) > 0.95:
        print("  ✅ 策略通过 DSR 验证")
    else:
        print(f"  ❌ 策略未通过 DSR 验证 (最高 DSR: {max(ma_metrics['dsr'], st_metrics['dsr']):.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()