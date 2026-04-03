"""
04b_tbm_optimization.py - AL9999 TBM 参数网格搜索优化

目标：通过参数网格搜索优化 TBM 止盈止损参数，提升策略表现。

参数搜索空间：
- profit_loss_barriers: 止盈/止损乘数组合 (tp_mult, sl_mult)
- vertical_barrier_bars: 最大持仓 bars 数量
- min_ret: 最小收益门槛

评估指标：
- Sharpe Ratio（年化）
- 总收益
- 胜率
- DSR

输出：
  - 参数搜索结果表
  - Top 5 参数组合
  - 最优参数的 DSR 验证
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    MA_PRIMARY_MODEL, TBM_CONFIG
)

from afmlkit.feature.core.ma import ewma
from afmlkit.label.tbm import triple_barrier


# ============================================================
# 数据加载（复用 04_ma_primary_model.py 逻辑）
# ============================================================

def load_data():
    """加载所有需要的数据。"""
    print("\n[Step 1] 加载数据...")

    # 加载 Dollar Bars
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target4.parquet')
    bars = pd.read_parquet(bars_path)
    print(f"   Dollar Bars: {len(bars)} bars")

    # 加载 CUSUM 事件
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = pd.read_parquet(events_path)
    if 'timestamp' in events.columns:
        events = events.set_index('timestamp')
    print(f"   CUSUM 事件: {len(events)} 个")

    # 加载事件特征
    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    events_features = pd.read_parquet(features_path)
    print(f"   事件特征: {len(events_features)} 个")

    return bars, events, events_features


def generate_ma_signals(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """生成 MA Primary Model 信号。"""
    close = bars['close'].values.astype(np.float64)
    span = MA_PRIMARY_MODEL.get('span', 20)
    ma_vals = ewma(close, span)

    signals = events.copy()
    event_indices = [bars.index.get_loc(ts) for ts in events.index]
    signals['idx'] = event_indices
    signals['ma'] = ma_vals[event_indices]
    signals['side'] = np.sign(signals['price'] - signals['ma'])
    signals['side'] = signals['side'].replace(0, 1).astype(int)

    return signals


# ============================================================
# TBM 计算函数
# ============================================================

def compute_tbm_for_params(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    events_features: pd.DataFrame,
    profit_loss_barriers: tuple,
    vertical_barrier_bars: int,
    min_ret: float,
    target_ret_col: str = 'feat_ewm_vol_20'
) -> pd.DataFrame:
    """
    使用指定参数计算 TBM。

    :param bars: Dollar Bars DataFrame
    :param signals: 信号 DataFrame
    :param events_features: 事件特征
    :param profit_loss_barriers: (tp_mult, sl_mult) 止盈止损乘数
    :param vertical_barrier_bars: 最大持仓 bars 数量
    :param min_ret: 最小收益门槛
    :param target_ret_col: 目标收益列
    :returns: TBM 结果 DataFrame
    """
    timestamps = bars.index.values.astype(np.int64)
    close = bars['close'].values.astype(np.float64)

    # 对齐索引
    common_idx = signals.index.intersection(events_features.index)
    signals_aligned = signals.loc[common_idx]

    event_idxs = signals_aligned['idx'].values.astype(np.int64)
    targets = events_features.loc[common_idx, target_ret_col].values.astype(np.float64)

    # 过滤末尾事件
    max_end_idx = len(bars) - 1
    valid_mask = event_idxs + vertical_barrier_bars < max_end_idx

    event_idxs = event_idxs[valid_mask]
    targets = targets[valid_mask]
    signals_valid = signals_aligned[valid_mask]

    # 计算垂直屏障（秒）
    bar_duration_sec = 4 * 3600  # 4 hours per bar
    vertical_barrier_sec = vertical_barrier_bars * bar_duration_sec

    # 调用 triple_barrier
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps=timestamps,
        close=close,
        event_idxs=event_idxs,
        targets=targets,
        profit_loss_barriers=profit_loss_barriers,
        vertical_barrier=vertical_barrier_sec,
        min_close_time_sec=60,
        side=signals_valid['side'].values.astype(np.int8),
        min_ret=min_ret
    )

    # 计算点数收益
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
# 指标计算函数
# ============================================================

def calculate_sharpe(returns: pd.Series, annualization_factor: float = 1500) -> float:
    """计算年化 Sharpe Ratio。"""
    if returns.std() == 0:
        return 0.0
    sr_period = returns.mean() / returns.std()
    return sr_period * np.sqrt(annualization_factor)


def calculate_psr(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """计算 PSR。"""
    if len(returns) < 50:
        return 0.0

    n = len(returns)
    sr = returns.mean() / returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis() + 3

    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr)

    return psr


def calculate_dsr(returns: pd.Series, n_trials: int = 20, annualization_factor: float = 1500) -> float:
    """计算 DSR。"""
    sr_std = 0.5
    gamma = 0.5772

    expected_max_sr_annual = sr_std * (
        (1 - gamma) * norm.ppf(1 - 1/n_trials) +
        gamma * norm.ppf(1 - 1/(n_trials * np.exp(-1)))
    )

    benchmark_sr_period = expected_max_sr_annual / np.sqrt(annualization_factor)

    return calculate_psr(returns, benchmark_sr=benchmark_sr_period)


# ============================================================
# 参数网格搜索
# ============================================================

def run_optimization():
    """运行 TBM 参数网格搜索。"""
    print("=" * 70)
    print("  AL9999 TBM Parameter Optimization")
    print("=" * 70)

    # 加载数据
    bars, events, events_features = load_data()

    # 生成信号
    print("\n[Step 2] 生成 MA 信号...")
    signals = generate_ma_signals(bars, events)
    print(f"   信号数: {len(signals)}")

    # 定义参数搜索空间
    print("\n[Step 3] 定义参数搜索空间...")

    # profit_loss_barriers = (tp_mult, sl_mult)
    barrier_combinations = [
        (1.0, 1.0),   # 对称
        (1.5, 1.0),   # 止盈更远
        (2.0, 1.0),   # 止盈更远
        (2.5, 1.0),   # 止盈更远
        (3.0, 1.0),   # 止盈更远
        (1.5, 1.5),   # 对称
        (2.0, 1.5),   # 止盈更远
        (2.5, 1.5),   # 止盈更远
        (3.0, 1.5),   # 止盈更远
        (2.0, 2.0),   # 对称
        (2.5, 2.0),   # 止盈更远
        (3.0, 2.0),   # 止盈更远
        (4.0, 1.5),   # 大止盈
        (5.0, 2.0),   # 优化后最优
    ]

    vertical_bars_options = [20, 30, 50, 80]
    min_ret_options = [0.0005, 0.001, 0.002]

    total_combinations = len(barrier_combinations) * len(vertical_bars_options) * len(min_ret_options)
    print(f"   止盈止损组合: {len(barrier_combinations)} 种")
    print(f"   垂直屏障选项: {len(vertical_bars_options)} 种")
    print(f"   最小收益选项: {len(min_ret_options)} 种")
    print(f"   总组合数: {total_combinations}")

    # 网格搜索
    print("\n[Step 4] 运行网格搜索...")
    results = []

    for i, (barriers, vb, min_ret) in enumerate(product(
        barrier_combinations, vertical_bars_options, min_ret_options
    ), 1):
        try:
            tbm_df = compute_tbm_for_params(
                bars, signals, events_features,
                profit_loss_barriers=barriers,
                vertical_barrier_bars=vb,
                min_ret=min_ret
            )

            # 计算指标
            sharpe = calculate_sharpe(tbm_df['ret'])
            total_pnl = tbm_df['pnl'].sum()
            win_rate = (tbm_df['pnl'] > 0).mean()
            dsr = calculate_dsr(tbm_df['ret'])

            results.append({
                'tp_mult': barriers[0],
                'sl_mult': barriers[1],
                'vertical_bars': vb,
                'min_ret': min_ret,
                'sharpe': sharpe,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'dsr': dsr,
                'n_samples': len(tbm_df),
            })

            if i % 20 == 0 or i == total_combinations:
                print(f"   进度: {i}/{total_combinations} ({i/total_combinations*100:.0f}%)")

        except Exception as e:
            print(f"   组合 {i} 失败: {barriers}, {vb}, {min_ret} - {e}")

    # 结果排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('dsr', ascending=False)

    # 输出结果
    print("\n" + "=" * 70)
    print("  参数搜索结果 - Top 10 (按 DSR 排序)")
    print("=" * 70)

    top10 = results_df.head(10)
    print(top10.to_string(index=False))

    # 最优参数
    best = results_df.iloc[0]
    print("\n" + "-" * 70)
    print("  最优参数")
    print("-" * 70)
    print(f"止盈乘数 (tp_mult): {best['tp_mult']}")
    print(f"止损乘数 (sl_mult): {best['sl_mult']}")
    print(f"垂直屏障: {best['vertical_bars']} bars")
    print(f"最小收益: {best['min_ret']}")
    print(f"\n年化 Sharpe: {best['sharpe']:.4f}")
    print(f"总收益: {best['total_pnl']:.2f} 点")
    print(f"胜率: {best['win_rate']*100:.1f}%")
    print(f"DSR: {best['dsr']:.4f}")
    print("-" * 70)

    # 保存结果
    results_path = os.path.join(FEATURES_DIR, 'tbm_optimization_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ 搜索结果已保存: {results_path}")

    # 验收结论
    if best['dsr'] > 0.95:
        verdict = "✅ 通过验证"
    elif best['dsr'] > 0.90:
        verdict = "⚠️ 边缘通过"
    else:
        verdict = "❌ 未通过验证"

    print(f"\n最终结论: {verdict}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = run_optimization()