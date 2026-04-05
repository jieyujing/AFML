"""
04d_tbm_label_optimization.py - AL9999 TBM 标签质量优化

目标：通过优化止盈止损参数提高标签区分度
- 当前：胜率 51%，盈亏比 0.98，标签区分度接近随机
- 目标：提高 Meta Label 的预测力

优化维度：
1. profit_loss_barriers (止盈止损倍数: tp_mult, sl_mult)
2. min_ret (最小收益门槛)
3. vertical_barrier (最大持仓时间)

评估指标：
- 胜率 > 52%
- 盈亏比 > 1.2
- Meta Model F1 > 0.55

注意：TBM 现在使用止盈止损语义
- profit_loss_barriers=(tp_mult, sl_mult)
- 做多时：止盈在上，止损在下
- 做空时：止盈在下，止损在上（自动反转）
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import BARS_DIR, FEATURES_DIR, FIGURES_DIR, TARGET_DAILY_BARS
from afmlkit.label.tbm import triple_barrier


# ============================================================
# 配置
# ============================================================

# 参数搜索空间
# profit_loss_barriers = (tp_mult, sl_mult)
# tp_mult: 止盈倍数（目标收益的多少倍作为止盈）
# sl_mult: 止损倍数（目标收益的多少倍作为止损）
# 扩大搜索范围：更大的止盈倍数，更小的止损倍数
PARAM_GRID = {
    'profit_loss_barriers': [
        # 原有范围
        (1.0, 1.0),   # 对称
        (1.5, 1.0),   # 止盈更远
        (2.0, 1.0),   # 止盈更远
        (2.5, 1.0),   # 止盈更远
        (3.0, 1.0),   # 止盈更远
        (1.0, 1.5),   # 止损更远
        (1.5, 1.5),   # 对称扩大
        (2.0, 1.5),   # 止盈更远
        (2.5, 1.5),   # 止盈更远
        (3.0, 1.5),   # 止盈更远
        (2.0, 2.0),   # 对称
        (2.5, 2.0),   # 止盈更远
        (3.0, 2.0),   # 止盈更远
        # 新增：更大的止盈倍数
        (3.5, 1.0),   # 大止盈，小止损
        (4.0, 1.0),   # 大止盈，小止损
        (4.5, 1.0),   # 大止盈，小止损
        (5.0, 1.0),   # 大止盈，小止损
        (3.5, 1.5),   # 大止盈
        (4.0, 1.5),   # 大止盈
        (4.5, 1.5),   # 大止盈
        (5.0, 1.5),   # 大止盈
        (3.5, 2.0),   # 大止盈
        (4.0, 2.0),   # 大止盈
        (4.5, 2.0),   # 大止盈
        (5.0, 2.0),   # 大止盈
    ],
    'min_ret': [0.0, 0.0005, 0.001, 0.0015, 0.002],
    'vertical_barrier_bars': [30, 50, 70, 100],
}

# 目标收益列
TARGET_RET_COL = 'feat_ewm_vol_20'


# ============================================================
# 函数
# ============================================================

def load_data():
    """加载数据。"""
    # Dollar Bars
    bars = pd.read_parquet(os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet'))

    # 事件特征
    events = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))

    # Primary Signals (MA)
    signals = pd.read_parquet(os.path.join(FEATURES_DIR, 'ma_primary_signals.parquet'))

    return bars, events, signals


def compute_tbm_for_params(bars, events, signals, params):
    """
    计算指定参数的 TBM 结果。

    :param bars: Dollar Bars
    :param events: 事件特征
    :param signals: Primary Model 信号
    :param params: 参数字典
    :returns: TBM DataFrame 和统计指标
    """
    # 对齐索引
    common_idx = signals.index.intersection(events.index)
    signals = signals.loc[common_idx]
    events = events.loc[common_idx]

    # 准备输入
    timestamps = bars.index.values.astype(np.int64)
    close = bars['close'].values.astype(np.float64)
    event_idxs = np.array([bars.index.get_loc(ts) for ts in signals.index])

    # 目标收益
    targets = events[TARGET_RET_COL].values

    # 屏障参数 (tp_mult, sl_mult)
    tp_mult, sl_mult = params['profit_loss_barriers']
    vertical_bars = params['vertical_barrier_bars']
    min_ret = params['min_ret']

    # 垂直屏障时间（秒）
    bar_duration_sec = 4 * 3600  # 6 bars/day ≈ 4 hours/bar
    vertical_sec = vertical_bars * bar_duration_sec

    # 运行 TBM
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps=timestamps,
        close=close,
        event_idxs=event_idxs,
        targets=targets,
        profit_loss_barriers=(tp_mult, sl_mult),
        vertical_barrier=vertical_sec,
        min_close_time_sec=60,
        side=signals['side'].values.astype(np.int8),
        min_ret=min_ret
    )

    # 构建结果 DataFrame
    tbm_df = pd.DataFrame({
        'label': labels,
        'touch_idx': touch_idxs,
        'ret': rets,
        'side': signals['side'].values,
    }, index=signals.index)

    # 判断触碰类型
    touch_types = []
    for i, (event_idx, touch_idx, ret) in enumerate(zip(event_idxs, touch_idxs, rets)):
        expected_vertical_idx = event_idx + vertical_bars
        if touch_idx >= expected_vertical_idx:
            touch_types.append('vertical')
        elif ret > 0:
            touch_types.append('upper')
        else:
            touch_types.append('lower')
    tbm_df['touch_type'] = touch_types

    # 计算统计指标
    n_total = len(tbm_df)
    n_win = (tbm_df['ret'] > 0).sum()
    n_loss = (tbm_df['ret'] < 0).sum()
    win_rate = n_win / n_total if n_total > 0 else 0

    avg_ret = tbm_df['ret'].mean()
    total_ret = tbm_df['ret'].sum()

    avg_win = tbm_df[tbm_df['ret'] > 0]['ret'].mean() if n_win > 0 else 0
    avg_loss = abs(tbm_df[tbm_df['ret'] < 0]['ret'].mean()) if n_loss > 0 else 0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Sharpe
    sharpe = tbm_df['ret'].mean() / tbm_df['ret'].std() * np.sqrt(1500) if tbm_df['ret'].std() > 0 else 0

    # 屏障触碰率
    upper_rate = (tbm_df['touch_type'] == 'upper').mean()
    lower_rate = (tbm_df['touch_type'] == 'lower').mean()
    vertical_rate = (tbm_df['touch_type'] == 'vertical').mean()

    stats = {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'total_ret': total_ret,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pl_ratio': pl_ratio,
        'sharpe': sharpe,
        'upper_rate': upper_rate,
        'lower_rate': lower_rate,
        'vertical_rate': vertical_rate,
        'tp_mult': tp_mult,
        'sl_mult': sl_mult,
        'min_ret': min_ret,
        'vertical_bars': vertical_bars,
    }

    return tbm_df, stats


def evaluate_label_quality(stats):
    """
    评估标签质量评分。

    目标：
    - 胜率 > 52%
    - 盈亏比 > 1.2
    - 样本数 > 200
    """
    # 胜率得分（目标 52%+）
    win_rate_score = min(max(stats['win_rate'] - 0.50, 0) / 0.10, 1.0)

    # 盈亏比得分（目标 1.2+）
    pl_ratio_score = min(max(stats['pl_ratio'] - 0.8, 0) / 0.8, 1.0)

    # 样本数得分
    sample_score = min(stats['n_total'] / 500, 1.0)

    # 综合得分
    # 权重：胜率 40%, 盈亏比 40%, 样本数 20%
    quality_score = 0.4 * win_rate_score + 0.4 * pl_ratio_score + 0.2 * sample_score

    return quality_score


def run_optimization():
    """运行参数优化。"""
    print("=" * 70)
    print("  AL9999 TBM 标签质量优化")
    print("=" * 70)

    # 加载数据
    print("\n[Step 1] 加载数据...")
    bars, events, signals = load_data()
    print(f"  Bars: {len(bars)}, Events: {len(events)}, Signals: {len(signals)}")

    # 生成参数组合
    param_combinations = list(product(
        PARAM_GRID['profit_loss_barriers'],
        PARAM_GRID['min_ret'],
        PARAM_GRID['vertical_barrier_bars']
    ))
    print(f"\n[Step 2] 参数组合数: {len(param_combinations)}")

    # 运行优化
    print("\n[Step 3] 运行参数优化...")
    results = []

    for i, (pl_barriers, min_ret, v_bars) in enumerate(param_combinations):
        params = {
            'profit_loss_barriers': pl_barriers,
            'min_ret': min_ret,
            'vertical_barrier_bars': v_bars,
        }

        try:
            tbm_df, stats = compute_tbm_for_params(bars, events, signals, params)
            quality_score = evaluate_label_quality(stats)
            stats['quality_score'] = quality_score
            results.append(stats)

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(param_combinations)}")
        except Exception as e:
            print(f"  参数 {params} 失败: {e}")

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('quality_score', ascending=False)

    # 输出结果
    print("\n" + "=" * 70)
    print("  Top 10 参数组合")
    print("=" * 70)

    cols = ['tp_mult', 'sl_mult', 'min_ret', 'vertical_bars',
            'win_rate', 'pl_ratio', 'n_total', 'sharpe', 'quality_score']

    for i, row in results_df.head(10).iterrows():
        print(f"\n[#{results_df.head(10).index.get_loc(i)+1}] TP={row['tp_mult']}, SL={row['sl_mult']}, "
              f"min_ret={row['min_ret']}, vertical={row['vertical_bars']}")
        print(f"  胜率: {row['win_rate']*100:.1f}%, 盈亏比: {row['pl_ratio']:.2f}, "
              f"样本数: {row['n_total']}, Sharpe: {row['sharpe']:.2f}")
        print(f"  质量评分: {row['quality_score']:.4f}")
        print(f"  触碰率: 上屏障={row['upper_rate']*100:.1f}%, 下屏障={row['lower_rate']*100:.1f}%")

    # 保存结果
    output_path = os.path.join(FEATURES_DIR, 'tbm_optimization_results.parquet')
    results_df.to_parquet(output_path)
    print(f"\n✅ 优化结果已保存: {output_path}")

    # 最佳参数
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("  推荐参数")
    print("=" * 70)
    print(f"  profit_loss_barriers = ({best['tp_mult']}, {best['sl_mult']})  # (tp_mult, sl_mult)")
    print(f"  min_ret = {best['min_ret']}")
    print(f"  vertical_barrier_bars = {int(best['vertical_bars'])}")
    print(f"\n  预期效果:")
    print(f"  胜率: {best['win_rate']*100:.1f}%")
    print(f"  盈亏比: {best['pl_ratio']:.2f}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results_df = run_optimization()