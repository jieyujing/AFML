"""
AFML Visual Guides
实现 Advances in Financial Machine Learning 核心概念的可视化方法
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union, Optional

# 配置 seaborn 风格，确保科研级图表输出
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def plot_dollar_vs_tick_bars(
    tick_datetimes: Union[pd.DatetimeIndex, pd.Series],
    dollar_datetimes: Union[pd.DatetimeIndex, pd.Series],
    resample_rule: str = 'D',
    title: str = "Tick Bars vs Dollar Bars Generation Frequency",
    save_path: Optional[str] = None
):
    """
    绘制并对比 Tick Bars 和 Dollar Bars 的生成频率走势。
    旨在证明 Dollar Bars 的频率更加稳定（降低异方差性）。
    """
    # 如果传入的是 Series，提取其值或保持 Index
    if isinstance(tick_datetimes, pd.Series):
        tick_datetimes = pd.DatetimeIndex(tick_datetimes)
    if isinstance(dollar_datetimes, pd.Series):
        dollar_datetimes = pd.DatetimeIndex(dollar_datetimes)

    tick_series = pd.Series(1, index=tick_datetimes).resample(resample_rule).sum()
    dollar_series = pd.Series(1, index=dollar_datetimes).resample(resample_rule).sum()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel(f'Time ({resample_rule})')
    ax1.set_ylabel('Tick Bars Count', color=color1)
    ax1.plot(tick_series.index, tick_series.values, color=color1, alpha=0.7, label='Tick Bars')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # 实例化共享相同 x 轴的第二个 y 轴
    color2 = 'tab:orange'
    ax2.set_ylabel('Dollar Bars Count', color=color2)
    ax2.plot(dollar_series.index, dollar_series.values, color=color2, linewidth=2, label='Dollar Bars')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    
    # 组合图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cusum_filter_events(
    price_series: pd.Series,
    event_indices: pd.DatetimeIndex,
    title: str = "CUSUM Filter Sampled Events on Price Series",
    save_path: Optional[str] = None
):
    """
    绘制价格曲线并在发生 CUSUM 事件的地方打点
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制基础价格面
    ax.plot(price_series.index, price_series.values, color='lightgray', linewidth=1.5, label='Price')
    
    # 提取事件对应的价格
    valid_events = event_indices[event_indices.isin(price_series.index)]
    event_prices = price_series.loc[valid_events]
    
    # 将过滤得到的采样点标注在对应发生时间的价格线上
    ax.scatter(event_prices.index, event_prices.values, color='tab:red', marker='v', s=50, label='CUSUM Event(sampled)', zorder=5)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tbm_bounds_single_case(
    price_series: pd.Series,
    t0: pd.Timestamp,
    t3: pd.Timestamp,
    target_up: float,
    target_down: float,
    t_i: Optional[pd.Timestamp] = None,
    side: int = 1,
    num_bars_context: int = 100,
    title: str = "Triple Barrier Method (Single Case)",
    save_path: Optional[str] = None
):
    """
    绘制单个交易样本的微观视角，展示 TBM 三面墙以及首次触碰轨迹。
    基于 K 线根数展示上下文。
    """
    # 找到入场点 t0 在序列中的最接近位置索引
    try:
        # 获取索引位置
        idx_t0 = price_series.index.get_indexer([t0], method='nearest')[0]
        
        # 确定显示范围：入场前展示 10% 的上下文，总共展示 num_bars_context 根
        pre_entry = max(1, int(num_bars_context * 0.1))
        start_idx = max(0, idx_t0 - pre_entry)
        end_idx = min(len(price_series), start_idx + num_bars_context)
        
        local_price = price_series.iloc[start_idx:end_idx]
    except Exception as e:
        print(f"Error indexing price series: {e}")
        local_price = price_series

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制局部真实价格走势
    ax.plot(local_price.index, local_price.values, color='black', linewidth=2, label='Price Path', marker='.', markersize=4, alpha=0.6)
    
    # 画上墙 (止盈) 和下墙 (止损) - 限制在实际持仓的时间内或者全图？
    # 通常画在 [t0, t3] 区间
    ax.hlines(y=target_up, xmin=t0, xmax=t3, color='tab:green', linestyle='-', linewidth=2, label='Upper Barrier (Take Profit)')
    ax.hlines(y=target_down, xmin=t0, xmax=t3, color='tab:red', linestyle='-', linewidth=2, label='Lower Barrier (Stop Loss)')
    
    # 画右墙
    ax.vlines(x=t3, ymin=target_down, ymax=target_up, color='tab:gray', linestyle='--', linewidth=2, label='Max Hold Period ($t_3$)')
    
    try:
        # 标出建仓点
        pt0 = price_series.loc[price_series.index >= t0].iloc[0]
        t0_exact = price_series.loc[price_series.index >= t0].index[0]
        ax.scatter(t0_exact, pt0, color='blue', s=100, marker='o', label='Entry $t_0$', zorder=5)

        # 标出碰撞点 t_i
        if t_i is not None and pd.notnull(t_i):
            pt_i = price_series.loc[price_series.index >= t_i].iloc[0]
            ti_exact = price_series.loc[price_series.index >= t_i].index[0]
            ax.scatter(ti_exact, pt_i, color='purple', s=200, marker='X', label='Hit $t_i$', zorder=6)
    except Exception as e:
        print(f"Warning: unable to exact point on given timestamps {e}")

    ax.set_title(title)
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_tbm_labels(
    price_series: pd.Series,
    labels_df: pd.DataFrame,
    title: str = "Triple Barrier Method Labels (All Events)",
    save_path: Optional[str] = None
):
    """
    在全量价格曲线上标注所有 TBM 标签的结果。
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 绘制背景价格
    ax.plot(price_series.index, price_series.values, color='lightgray', alpha=0.5, label='Price')
    
    # 提取不同结果的采样点
    valid_labels = labels_df.loc[labels_df.index.isin(price_series.index)]
    
    pos_cases = valid_labels[valid_labels['bin'] == 1]
    neg_cases = valid_labels[valid_labels['bin'] == -1]
    null_cases = valid_labels[valid_labels['bin'] == 0]
    
    # 标出价格点
    ax.scatter(pos_cases.index, price_series.loc[pos_cases.index], color='tab:green', marker='^', s=40, label='Hit Upper Barrier (1)', alpha=0.9, zorder=5)
    ax.scatter(neg_cases.index, price_series.loc[neg_cases.index], color='tab:red', marker='v', s=40, label='Hit Lower Barrier (-1)', alpha=0.9, zorder=5)
    ax.scatter(null_cases.index, price_series.loc[null_cases.index], color='tab:gray', marker='o', s=20, label='Hit Vertical Barrier (0)', alpha=0.7, zorder=4)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_uniqueness_and_concurrency(
    t_events: pd.Series,
    t1: pd.Series,
    sample_weights: pd.Series,
    title: str = "Concurrent Labels and Sample Uniqueness",
    save_path: Optional[str] = None
):
    """
    绘制双子图：
    上子图：随时间变化的非重叠（并发）数量 c_t
    下子图：随时间变化的样本权重唯一性 Uniqueness 散点
    """
    # 清理缺失的结束时间，提取有效配对
    valid_mask = t1.notna()
    t_events_valid = t_events[valid_mask]
    t1_valid = t1[valid_mask]
    weights_valid = sample_weights.loc[t_events_valid.index]
    
    # 为了构建 c_t，我们将时间的起止点分离为事件 (+1 和 -1)
    # 起点 +1，即新增一个 bar
    starts = pd.Series(1, index=t_events_valid.values)
    # 终点 -1，即关闭一个 bar (加1微秒确保遇到同一时间的起止点，关闭稍微延后)
    # 若在频次较低的数据，同时间平仓和开仓，通常逻辑上平仓在先或者相互抵消。
    ends = pd.Series(-1, index=t1_valid.values + pd.Timedelta(milliseconds=1))
    
    # 组合为事件流
    event_flow = pd.concat([starts, ends]).sort_index()
    
    # 累计求和即为在每个时间点的状态开启数量 c_t
    c_t = event_flow.cumsum()
    # 移除重复的时间点，保留最终状态
    c_t = c_t[~c_t.index.duplicated(keep='last')]
    
    # 从起始时间的整体区间限制绘图长度
    if not c_t.empty:
        c_t = c_t.loc[t_events.min():t1.max()]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    
    # 上：并发数量
    if not c_t.empty:
        ax1.plot(c_t.index, c_t.values, color='tab:blue', drawstyle='steps-post', label='Concurrent Labels ($c_t$)')
    ax1.set_ylabel('No. of Concurrent Labels')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    
    # 下：Uniqueness
    ax2.scatter(t_events_valid, weights_valid, color='tab:orange', alpha=0.6, marker='o', label='Sample Uniqueness Weight')
    ax2.set_ylabel('Uniqueness Weight')
    ax2.set_xlabel('Event Time $t_0$')
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("This module contains plotting functions for AFML core concepts visualization.")
