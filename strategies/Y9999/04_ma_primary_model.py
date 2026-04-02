"""
04_ma_primary_model.py - Y9999 MA Primary Model 回测 (TBM 版本)

根据 AFML 方法论，使用 Triple Barrier Method (TBM) 替代固定持仓周期：
- 止盈：target * top_multiplier
- 止损：target * bottom_multiplier
- 超时：vertical_barrier

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 CUSUM 事件点和特征（含波动率估计）
3. 计算 EWMA
4. 在事件点生成信号（price > MA → long, else → short）
5. 使用 TBM 计算每个信号的结果
6. 统计分析和可视化

输出:
  - 统计报告（文本）
  - 04_tbm_pnl_distribution.png: 收益分布直方图
  - 04_tbm_cumulative_pnl.png: 累积收益曲线
  - 04_tbm_barrier_touch.png: 屏障触碰分布
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    MA_PRIMARY_MODEL, TBM_CONFIG
)

from afmlkit.feature.core.ma import ewma, sma
from afmlkit.label.tbm import triple_barrier

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 数据加载
# ============================================================

def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """
    加载 Dollar Bars 数据。

    :param bars_path: parquet 文件路径
    :returns: DataFrame with timestamp index
    """
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def load_cusum_events(events_path: str) -> pd.DataFrame:
    """
    加载 CUSUM 事件点数据。

    :param events_path: parquet 文件路径
    :returns: DataFrame with timestamp index
    """
    events = pd.read_parquet(events_path)
    # 确保 timestamp 列存在并设置为索引
    if 'timestamp' in events.columns:
        events = events.set_index('timestamp')
    print(f"✅ 加载 CUSUM 事件点: {len(events)} 个")
    print(f"   时间范围: {events.index.min()} ~ {events.index.max()}")
    return events


def load_events_features(features_path: str) -> pd.DataFrame:
    """
    加载事件特征（含波动率估计）。

    :param features_path: parquet 文件路径
    :returns: DataFrame with volatility features
    """
    features = pd.read_parquet(features_path)
    print(f"✅ 加载事件特征: {len(features)} 个")
    return features


# ============================================================
# MA Primary Model 信号生成
# ============================================================

def generate_ma_signals(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    在 CUSUM 事件点生成 MA Primary Model 信号。

    规则: price > MA → long (+1), price < MA → short (-1)

    :param bars: Dollar Bars DataFrame
    :param events: CUSUM 事件点 DataFrame
    :param config: MA_PRIMARY_MODEL 配置
    :returns: DataFrame with columns [timestamp, price, idx, ma, side]
    """
    close = bars['close'].values.astype(np.float64)

    # 计算 MA
    ma_type = config.get('ma_type', 'ewma')
    span = config.get('span', 20)

    if ma_type == 'ewma':
        ma_vals = ewma(close, span)
        print(f"\n[MA] 计算 EWMA: span={span}")
    else:
        ma_vals = sma(close, span)
        print(f"\n[MA] 计算 SMA: window={span}")

    # 在事件点生成信号
    signals = events.copy()

    # 获取事件点在 bars 中的索引位置
    event_indices = [bars.index.get_loc(ts) for ts in events.index]
    signals['idx'] = event_indices
    signals['ma'] = ma_vals[event_indices]

    # side = sign(price - ma)
    signals['side'] = np.sign(signals['price'] - signals['ma'])
    # price == ma 时默认做多
    signals['side'] = signals['side'].replace(0, 1).astype(int)

    # 统计
    n_long = (signals['side'] == 1).sum()
    n_short = (signals['side'] == -1).sum()
    print(f"\n[信号] 生成完成")
    print(f"   MA 类型: {ma_type}, span={span}")
    print(f"   总信号数: {len(signals)}")
    print(f"   多头信号: {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"   空头信号: {n_short} ({n_short/len(signals)*100:.1f}%)")

    return signals


# ============================================================
# TBM (Triple Barrier Method) 计算
# ============================================================

def compute_tbm_labels(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    events_features: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    使用 TBM 计算每个信号的结果。

    :param bars: Dollar Bars DataFrame
    :param signals: 信号 DataFrame (timestamp, price, idx, side)
    :param events_features: 事件特征（含波动率估计）
    :param config: TBM_CONFIG
    :returns: DataFrame with TBM results
    """
    print("\n[TBM] 计算 Triple Barrier Method...")

    # 准备输入
    timestamps = bars.index.values.astype(np.int64)  # 纳秒时间戳
    close = bars['close'].values.astype(np.float64)

    # 事件索引
    event_idxs = signals['idx'].values.astype(np.int64)

    # 目标收益（波动率估计）
    target_ret_col = config.get('target_ret_col', 'feat_ewm_vol_20')

    # 对齐索引
    common_idx = signals.index.intersection(events_features.index)
    signals = signals.loc[common_idx]
    event_idxs = signals['idx'].values.astype(np.int64)

    targets = events_features.loc[common_idx, target_ret_col].values.astype(np.float64)

    # TBM 参数
    horizontal_barriers = config.get('horizontal_barriers', (2.0, 2.0))
    vertical_barrier_bars = config.get('vertical_barrier_bars', 50)
    min_ret = config.get('min_ret', 0.001)
    min_close_time_sec = config.get('min_close_time_sec', 60)

    # 计算垂直屏障（秒）
    # 假设 6 bars/day，1 bar ≈ 4 hours = 14400 秒
    bar_duration_sec = 4 * 3600
    vertical_barrier_sec = vertical_barrier_bars * bar_duration_sec

    print(f"   目标收益列: {target_ret_col}")
    print(f"   止盈止损乘数: {horizontal_barriers}")
    print(f"   垂直屏障: {vertical_barrier_bars} bars = {vertical_barrier_sec/3600:.1f} hours")
    print(f"   最小收益门槛: {min_ret}")

    # 过滤末尾事件（无法评估完整窗口）
    max_end_idx = len(bars) - 1
    valid_mask = event_idxs + vertical_barrier_bars < max_end_idx

    n_before = len(signals)
    signals = signals[valid_mask].copy()
    event_idxs = event_idxs[valid_mask]
    targets = targets[valid_mask]
    print(f"   过滤末尾事件: {n_before} → {len(signals)} 个有效事件")

    # 调用 triple_barrier
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps=timestamps,
        close=close,
        event_idxs=event_idxs,
        targets=targets,
        horizontal_barriers=horizontal_barriers,
        vertical_barrier=vertical_barrier_sec,
        min_close_time_sec=min_close_time_sec,
        side=signals['side'].values.astype(np.int8),
        min_ret=min_ret
    )

    print(f"   TBM 计算完成: {len(labels)} 个标签")

    # 判断触碰类型
    # touch_idx == event_idx + vertical_barrier_bars → 垂直屏障
    # ret > 0 → 上屏障（止盈）
    # ret < 0 → 下屏障（止损）

    touch_types = []
    for i, (event_idx, touch_idx, ret) in enumerate(zip(event_idxs, touch_idxs, rets)):
        expected_vertical_idx = event_idx + vertical_barrier_bars
        if touch_idx >= expected_vertical_idx:
            touch_types.append('vertical')  # 超时
        elif ret > 0:
            touch_types.append('upper')     # 止盈
        else:
            touch_types.append('lower')     # 止损

    # 构建输出 DataFrame
    tbm_df = pd.DataFrame({
        'label': labels,
        'touch_idx': touch_idxs,
        'ret': rets,
        'max_rb_ratio': max_rb_ratios,
        'touch_type': touch_types,
        'side': signals['side'].values,
        'entry_price': signals['price'].values,
        'target': targets,
    }, index=signals.index)

    # 计算退出价格和时间
    valid_touch_mask = touch_idxs != -1
    tbm_df.loc[valid_touch_mask, 'exit_price'] = close[tbm_df.loc[valid_touch_mask, 'touch_idx'].values]
    tbm_df.loc[valid_touch_mask, 'exit_ts'] = pd.to_datetime(timestamps[tbm_df.loc[valid_touch_mask, 'touch_idx'].values])

    # 计算点数收益
    tbm_df['pnl'] = tbm_df['side'] * (tbm_df['exit_price'] - tbm_df['entry_price'])

    # 统计触碰类型分布
    touch_counts = pd.Series(touch_types).value_counts()
    print(f"\n   触碰类型分布:")
    for t, c in touch_counts.items():
        print(f"     {t}: {c} ({c/len(touch_types)*100:.1f}%)")

    return tbm_df


# ============================================================
# 统计分析
# ============================================================

def compute_tbm_stats(tbm_df: pd.DataFrame) -> dict:
    """
    计算 TBM 统计指标。

    :param tbm_df: TBM 结果 DataFrame
    :returns: 统计指标字典
    """
    n_total = len(tbm_df)
    n_win = int((tbm_df['pnl'] > 0).sum())
    n_loss = int((tbm_df['pnl'] < 0).sum())
    n_flat = int((tbm_df['pnl'] == 0).sum())

    win_rate = n_win / n_total if n_total > 0 else 0
    avg_pnl = tbm_df['pnl'].mean()
    total_pnl = tbm_df['pnl'].sum()

    # 盈亏比
    wins = tbm_df[tbm_df['pnl'] > 0]['pnl']
    losses = tbm_df[tbm_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    max_win = tbm_df['pnl'].max()
    max_loss = tbm_df['pnl'].min()

    # 触碰类型统计
    touch_counts = tbm_df['touch_type'].value_counts()
    upper_rate = touch_counts.get('upper', 0) / n_total if n_total > 0 else 0
    lower_rate = touch_counts.get('lower', 0) / n_total if n_total > 0 else 0
    vertical_rate = touch_counts.get('vertical', 0) / n_total if n_total > 0 else 0

    # Meta label 分布
    label_1 = (tbm_df['label'] == 1).sum()
    label_0 = (tbm_df['label'] == 0).sum()

    stats = {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'n_flat': n_flat,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss,
        'upper_rate': upper_rate,
        'lower_rate': lower_rate,
        'vertical_rate': vertical_rate,
        'label_1': label_1,
        'label_0': label_0,
    }

    return stats


def print_tbm_stats(stats: dict):
    """
    打印 TBM 统计报告。

    :param stats: 统计指标字典
    """
    print("\n" + "=" * 70)
    print("  MA Primary Model Backtest (TBM)")
    print("=" * 70)
    print(f"总信号数: {stats['n_total']}")
    print(f"盈利信号: {stats['n_win']} ({stats['win_rate']*100:.1f}%)")
    print(f"亏损信号: {stats['n_loss']} ({stats['n_loss']/stats['n_total']*100:.1f}%)")
    print(f"持平信号: {stats['n_flat']}")
    print("-" * 70)
    print(f"胜率: {stats['win_rate']*100:.1f}%")
    print(f"平均收益: {stats['avg_pnl']:.2f} 点")
    print(f"总收益: {stats['total_pnl']:.2f} 点")
    print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    print(f"平均盈利: {stats['avg_win']:.2f} 点")
    print(f"平均亏损: {stats['avg_loss']:.2f} 点")
    print(f"最大盈利: {stats['max_win']:.2f} 点")
    print(f"最大亏损: {stats['max_loss']:.2f} 点")
    print("-" * 70)
    print("触碰类型分布:")
    print(f"  止盈 (upper): {stats['upper_rate']*100:.1f}%")
    print(f"  止损 (lower): {stats['lower_rate']*100:.1f}%")
    print(f"  超时 (vertical): {stats['vertical_rate']*100:.1f}%")
    print("-" * 70)
    print("Meta Label 分布:")
    print(f"  label=1: {stats['label_1']} ({stats['label_1']/stats['n_total']*100:.1f}%)")
    print(f"  label=0: {stats['label_0']} ({stats['label_0']/stats['n_total']*100:.1f}%)")
    print("=" * 70)


# ============================================================
# 可视化模块
# ============================================================

def plot_pnl_distribution(tbm_df: pd.DataFrame, save_path: str):
    """绘制收益分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    wins = tbm_df[tbm_df['pnl'] > 0]['pnl']
    losses = tbm_df[tbm_df['pnl'] < 0]['pnl']

    bins = np.linspace(tbm_df['pnl'].min(), tbm_df['pnl'].max(), 50)

    if len(losses) > 0:
        ax.hist(losses, bins=bins, color='#ff6b6b', alpha=0.7, label=f'Loss ({len(losses)})')
    if len(wins) > 0:
        ax.hist(wins, bins=bins, color='#4ecdc4', alpha=0.7, label=f'Win ({len(wins)})')

    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(tbm_df['pnl'].mean(), color='blue', linestyle='-', linewidth=2,
                label=f'Mean: {tbm_df["pnl"].mean():.1f}点')

    ax.set_title('MA Primary Model PnL Distribution (TBM)', fontsize=12)
    ax.set_xlabel('PnL (points)')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 收益分布图已保存: {save_path}")


def plot_cumulative_pnl(tbm_df: pd.DataFrame, save_path: str):
    """绘制累积收益曲线。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 按时间排序计算累积收益
    cumulative = tbm_df['pnl'].cumsum()

    ax.plot(tbm_df.index, cumulative.values, color='steelblue', linewidth=1.5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    final_cum = cumulative.iloc[-1]

    ax.scatter([tbm_df.index[-1]], [final_cum], color='red', s=100, zorder=5)
    ax.annotate(f'Final: {final_cum:.1f}点',
                xy=(tbm_df.index[-1], final_cum),
                xytext=(10, 10), textcoords='offset points')

    ax.set_title('MA Primary Model Cumulative PnL (TBM)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative PnL (points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 累积收益图已保存: {save_path}")


def plot_barrier_touch(tbm_df: pd.DataFrame, save_path: str):
    """绘制触碰类型分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 触碰类型分布
    ax1 = axes[0]
    touch_counts = tbm_df['touch_type'].value_counts()
    colors = {'upper': '#4ecdc4', 'lower': '#ff6b6b', 'vertical': '#f39c12'}
    bars = ax1.bar(touch_counts.index, touch_counts.values,
                   color=[colors.get(t, 'gray') for t in touch_counts.index])
    ax1.set_xlabel('Touch Type')
    ax1.set_ylabel('Count')
    ax1.set_title('Barrier Touch Distribution')
    for bar, count in zip(bars, touch_counts.values):
        ax1.annotate(f'{count}\n({count/len(tbm_df)*100:.1f}%)',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom')

    # 2. 触碰类型 vs 收益
    ax2 = axes[1]
    for touch_type in ['upper', 'lower', 'vertical']:
        subset = tbm_df[tbm_df['touch_type'] == touch_type]
        if len(subset) > 0:
            ax2.scatter(subset.index, subset['pnl'],
                       label=f'{touch_type} ({len(subset)})',
                       color=colors.get(touch_type, 'gray'), alpha=0.6, s=30)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('PnL (points)')
    ax2.set_title('PnL by Touch Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 触碰类型分布图已保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    """Y9999 MA Primary Model Backtest (TBM) 主流程."""
    print("=" * 70)
    print("  Y9999 MA Primary Model Backtest (TBM)")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target8.parquet')
    bars = load_dollar_bars(bars_path)

    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = load_cusum_events(events_path)

    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    events_features = load_events_features(features_path)

    # Step 2: 生成信号
    print("\n[Step 2] 生成 MA Primary Model 信号...")
    signals = generate_ma_signals(bars, events, MA_PRIMARY_MODEL)

    # Step 3: TBM 计算
    print("\n[Step 3] TBM 计算...")
    tbm_df = compute_tbm_labels(bars, signals, events_features, TBM_CONFIG)

    # Step 4: 统计分析
    print("\n[Step 4] 统计分析...")
    stats = compute_tbm_stats(tbm_df)
    print_tbm_stats(stats)

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_pnl_distribution(tbm_df, os.path.join(FIGURES_DIR, '04_tbm_pnl_dist.png'))
    plot_cumulative_pnl(tbm_df, os.path.join(FIGURES_DIR, '04_tbm_cum_pnl.png'))
    plot_barrier_touch(tbm_df, os.path.join(FIGURES_DIR, '04_tbm_barrier_touch.png'))

    # Step 6: 保存结果
    print("\n[Step 6] 保存结果...")

    # 保存 TBM 结果
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    tbm_df.to_parquet(tbm_path)
    print(f"✅ TBM 结果已保存: {tbm_path}")

    # 保存信号（带 side）
    signals_to_save = signals.copy()
    signals_path = os.path.join(FEATURES_DIR, 'ma_primary_signals.parquet')
    signals_to_save.to_parquet(signals_path)
    print(f"✅ 信号已保存: {signals_path}")

    print("\n" + "=" * 70)
    print("  MA Primary Model Backtest (TBM) 完成")
    print("=" * 70)
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 04_tbm_pnl_dist.png")
    print(f"  - 04_tbm_cum_pnl.png")
    print(f"  - 04_tbm_barrier_touch.png")


if __name__ == "__main__":
    main()